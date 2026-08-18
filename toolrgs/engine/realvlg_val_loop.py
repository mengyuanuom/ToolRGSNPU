"""RealVLG-R1 validation loop using the benchmark's public metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from toolrgs.engine.hooks import LoopState
from toolrgs.engine.loops import BaseLoop
from toolrgs.evaluation import (
    REALVLG_GRIPPER_DEPTH,
    apply_affine,
    evaluate_realvlg_grasp,
    inverse_warp,
    realvlg_ciou,
    realvlg_e_measure,
    realvlg_f_measure,
    realvlg_giou,
    realvlg_mask_to_bbox,
    realvlg_s_measure,
    resolve_evaluation_protocol,
)
from toolrgs.models.base import model_requires_depth
from toolrgs.registry import LOOPS
from toolrgs.runtime import current_device, move_to_device
from toolrgs.structures import GraspModelResult


def _resize_prediction(tensor, output_hw, mode="bicubic"):
    if tensor.shape[-2:] == tuple(output_hw):
        return tensor
    return F.interpolate(
        tensor,
        size=tuple(output_hw),
        mode=mode,
        align_corners=False if mode == "bilinear" else True,
    )


def _sample_nearest(array, x, y):
    values = np.asarray(array, dtype=np.float32)
    height, width = values.shape[-2:]
    column = int(np.clip(round(float(x)), 0, width - 1))
    row = int(np.clip(round(float(y)), 0, height - 1))
    return float(values[..., row, column])


def _sample_bilinear(array, x, y):
    values = np.asarray(array, dtype=np.float32).squeeze()
    height, width = values.shape
    x = float(np.clip(x, 0.0, width - 1.0))
    y = float(np.clip(y, 0.0, height - 1.0))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    weight_x, weight_y = x - x0, y - y0
    top = (1.0 - weight_x) * values[y0, x0] + weight_x * values[y0, x1]
    bottom = (1.0 - weight_x) * values[y1, x0] + weight_x * values[y1, x1]
    return float((1.0 - weight_y) * top + weight_y * bottom)


@LOOPS.register_module(name="realvlg_val", aliases=("realvlg_official_val",))
class RealVLGValLoop(BaseLoop):
    """Evaluate masks and one rectangular grasp exactly in source coordinates."""

    def __init__(self, dataloader, model, cfg, hooks=None):
        super().__init__(hooks=hooks)
        self.dataloader = dataloader
        self.model = model
        self.cfg = cfg
        self.device = current_device(int(getattr(cfg, "npu", getattr(cfg, "gpu", 0))))
        self.protocol = resolve_evaluation_protocol(
            getattr(cfg, "evaluation_protocol", "realvlg_official")
        )
        if self.protocol.name != "realvlg_official":
            raise ValueError(
                "RealVLGValLoop requires evaluation_protocol=realvlg_official"
            )
        self.mask_threshold = float(getattr(cfg, "mask_threshold", 0.35))
        self.width_factor = float(getattr(cfg, "realvlg_width_factor", 100.0))
        self.offset_decode_mode = str(
            getattr(cfg, "offset_decode_mode", "radius")
        ).strip().lower()
        if self.offset_decode_mode not in {"radius", "grasp_relative"}:
            raise ValueError(
                "offset_decode_mode must be 'radius' or 'grasp_relative', got "
                f"{self.offset_decode_mode!r}"
            )

    def _offset_radius(self, input_hw):
        configured = getattr(self.cfg, "offset_r", None)
        if configured is not None and float(configured) > 0:
            return float(configured)
        return max(1.0, min(input_hw) / 20.0)

    def _decode_one_grasp(
        self,
        quality,
        sine,
        cosine,
        width,
        inverse,
        scale,
        offset=None,
    ):
        maps = (quality, sine, cosine, width)
        if not all(np.isfinite(value).all() for value in maps):
            return None
        if quality.size == 0:
            return None
        row, column = np.unravel_index(int(np.argmax(quality)), quality.shape)
        center_x, center_y = float(column), float(row)
        sampled_sine = float(sine[row, column])
        sampled_cosine = float(cosine[row, column])
        sampled_width = float(width[row, column])
        if offset is not None:
            if not np.isfinite(offset).all():
                return None
            if self.offset_decode_mode == "grasp_relative":
                input_long_side = sampled_width * self.width_factor
                input_short_side = REALVLG_GRIPPER_DEPTH * float(scale)
                offset_scale = max(
                    1.0,
                    float(
                        np.hypot(
                            input_long_side * 0.25,
                            input_short_side * 0.5,
                        )
                    ),
                )
            else:
                offset_scale = self._offset_radius(quality.shape)
            delta_x = _sample_nearest(offset[0], center_x, center_y)
            delta_y = _sample_nearest(offset[1], center_x, center_y)
            center_x += delta_x * offset_scale
            center_y += delta_y * offset_scale

        if bool(getattr(self.cfg, "offset_resample_geometry", False)):
            sampled_sine = _sample_bilinear(sine, center_x, center_y)
            sampled_cosine = _sample_bilinear(cosine, center_x, center_y)
            sampled_width = _sample_bilinear(width, center_x, center_y)
        angle_degrees = float(
            0.5 * np.arctan2(sampled_sine, sampled_cosine) / np.pi * 180.0
        )
        original_center = apply_affine(
            np.asarray([[center_x, center_y]], dtype=np.float32), inverse
        )[0]
        original_width = sampled_width * self.width_factor / float(scale)
        prediction = np.asarray(
            [
                original_center[0],
                original_center[1],
                original_width,
                REALVLG_GRIPPER_DEPTH,
                angle_degrees,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(prediction).all() or original_width <= 0.0:
            return None
        return prediction

    @staticmethod
    def _reduce_statistics(values, device):
        statistics = torch.tensor(values, dtype=torch.float32, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        return statistics.detach().cpu().tolist()

    @torch.no_grad()
    def run_epoch(self, epoch):
        self.state = LoopState(epoch=epoch)
        self.hooks.call("before_epoch", self, self.state)
        self.model.eval()
        evaluation_model = getattr(self.model, "module", self.model)
        rank = int(getattr(self.cfg, "rank", 0))
        progress = tqdm(self.dataloader, disable=rank != 0)

        total = 0.0
        segmentation_valid = 0.0
        giou_sum = ciou_sum = 0.0
        f_sum = s_sum = e_sum = 0.0
        grasp_valid = 0.0
        grasp_iou_sum = grasp_correct = 0.0

        for iteration, data in enumerate(progress):
            self.state.iteration = iteration
            self.state.batch = data
            self.hooks.call("before_iter", self, self.state)

            image = move_to_device(data["img"], self.device)
            text = move_to_device(data["word_vec"], self.device)
            target_segmentation = move_to_device(data["mask"], self.device).unsqueeze(1)
            target_quality = move_to_device(
                data["grasp_masks"]["qua"], self.device
            ).unsqueeze(1)
            target_sine = move_to_device(
                data["grasp_masks"]["sin"], self.device
            ).unsqueeze(1)
            target_cosine = move_to_device(
                data["grasp_masks"]["cos"], self.device
            ).unsqueeze(1)
            target_width = move_to_device(
                data["grasp_masks"]["wid"], self.device
            ).unsqueeze(1)
            inputs = (
                image,
                text,
                target_segmentation,
                target_quality,
                target_sine,
                target_cosine,
                target_width,
            )
            if model_requires_depth(self.model):
                raise ValueError(
                    "The public RealVLG adapter has no aligned depth input; "
                    "choose an RGB-only ToolRGS architecture."
                )

            result = GraspModelResult.from_legacy(evaluation_model(*inputs))
            predictions = result.predictions
            input_hw = image.shape[-2:]
            segmentation = _resize_prediction(
                torch.sigmoid(predictions.segmentation), input_hw
            )
            quality = _resize_prediction(torch.sigmoid(predictions.quality), input_hw)
            sine = _resize_prediction(predictions.sine, input_hw)
            cosine = _resize_prediction(predictions.cosine, input_hw)
            width = _resize_prediction(torch.sigmoid(predictions.width), input_hw)
            tensors = [segmentation, quality, sine, cosine, width]
            if predictions.offset is not None:
                tensors.append(
                    _resize_prediction(predictions.offset, input_hw, mode="bilinear")
                )
            dense = torch.cat(tensors, dim=1).detach().float().cpu().numpy()
            offset_index = 5 if predictions.offset is not None else None

            for index in range(image.shape[0]):
                total += 1.0
                inverse = np.asarray(data["inverse"][index], dtype=np.float32)
                original_hw = tuple(int(value) for value in data["ori_size"][index])
                predicted_probability = inverse_warp(
                    dense[index, 0],
                    inverse,
                    original_hw,
                    interpolation=cv2.INTER_LINEAR,
                )
                if np.isfinite(predicted_probability).all():
                    predicted_mask = (
                        predicted_probability > self.mask_threshold
                    ).astype(np.uint8)
                    target_mask = (
                        np.asarray(data["mask_original"][index]) > 0
                    ).astype(np.uint8)
                    predicted_bbox = realvlg_mask_to_bbox(predicted_mask)
                    if predicted_bbox is not None:
                        target_bbox = np.asarray(
                            data["bbox_original"][index], dtype=np.float64
                        )
                        segmentation_valid += 1.0
                        giou_sum += realvlg_giou(target_bbox, predicted_bbox)
                        ciou_sum += realvlg_ciou(target_bbox, predicted_bbox)
                        f_sum += realvlg_f_measure(predicted_mask, target_mask)
                        s_sum += realvlg_s_measure(predicted_mask, target_mask)
                        e_sum += realvlg_e_measure(predicted_mask, target_mask)

                offset = (
                    dense[index, offset_index : offset_index + 2]
                    if offset_index is not None
                    else None
                )
                prediction = self._decode_one_grasp(
                    dense[index, 1],
                    dense[index, 2],
                    dense[index, 3],
                    dense[index, 4],
                    inverse,
                    float(data["scale"][index]),
                    offset=offset,
                )
                if prediction is not None:
                    ground_truth = data["grasps_points8"][index]
                    if hasattr(ground_truth, "detach"):
                        ground_truth = ground_truth.detach().cpu().numpy()
                    best_iou, _angle, correct = evaluate_realvlg_grasp(
                        prediction, ground_truth
                    )
                    grasp_valid += 1.0
                    grasp_iou_sum += best_iou
                    grasp_correct += float(correct)

            self.state.result = result
            self.hooks.call("after_iter", self, self.state)

        values = self._reduce_statistics(
            [
                total,
                segmentation_valid,
                giou_sum,
                ciou_sum,
                f_sum,
                s_sum,
                e_sum,
                grasp_valid,
                grasp_iou_sum,
                grasp_correct,
            ],
            self.device,
        )
        (
            total,
            segmentation_valid,
            giou_sum,
            ciou_sum,
            f_sum,
            s_sum,
            e_sum,
            grasp_valid,
            grasp_iou_sum,
            grasp_correct,
        ) = values
        metrics = {
            "split": str(
                getattr(
                    self.dataloader.dataset,
                    "split",
                    getattr(
                        self.cfg,
                        "test_split",
                        getattr(self.cfg, "val_split", ""),
                    ),
                )
            ),
            "Segmentation_Validity_Rate": segmentation_valid / max(1.0, total),
            "mean_gIoU": giou_sum / max(1.0, segmentation_valid),
            "mean_cIoU": ciou_sum / max(1.0, segmentation_valid),
            "F_beta": f_sum / max(1.0, segmentation_valid),
            "S_alpha": s_sum / max(1.0, segmentation_valid),
            "E_measure": e_sum / max(1.0, segmentation_valid),
            "Grasp_Validity_Rate": grasp_valid / max(1.0, total),
            "mIoU": grasp_iou_sum / max(1.0, grasp_valid),
            "gAcc": grasp_correct / max(1.0, grasp_valid),
            "num_samples": int(total),
        }
        precision = {
            "S_alpha": metrics["S_alpha"],
            "E_measure": metrics["E_measure"],
            "mean_gIoU": metrics["mean_gIoU"],
            "mean_cIoU": metrics["mean_cIoU"],
            "Segmentation_Validity_Rate": metrics[
                "Segmentation_Validity_Rate"
            ],
            "Grasp_mIoU": metrics["mIoU"],
            "Grasp_Validity_Rate": metrics["Grasp_Validity_Rate"],
        }
        self.state.logs = {
            "iou": metrics["F_beta"],
            "precision": precision,
            "j_index": [metrics["gAcc"]],
            "realvlg": metrics,
        }
        self.hooks.call("after_epoch", self, self.state)

        if rank == 0:
            logger.info(
                "RealVLG {}: gIoU={:.4f} cIoU={:.4f} F_beta={:.4f} "
                "S_alpha={:.4f} E={:.4f} "
                "SegVR={:.4f} mIoU={:.4f} gAcc={:.4f} GraspVR={:.4f} n={}",
                metrics["split"],
                metrics["mean_gIoU"],
                metrics["mean_cIoU"],
                metrics["F_beta"],
                metrics["S_alpha"],
                metrics["E_measure"],
                metrics["Segmentation_Validity_Rate"],
                metrics["mIoU"],
                metrics["gAcc"],
                metrics["Grasp_Validity_Rate"],
                metrics["num_samples"],
            )
            output_dir = getattr(self.cfg, "output_dir", None)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                result_path = Path(output_dir) / (
                    f"realvlg_{metrics['split']}_metrics.json"
                )
                temporary_path = result_path.with_suffix(".json.tmp")
                with temporary_path.open("w", encoding="utf-8") as stream:
                    json.dump(metrics, stream, indent=2, ensure_ascii=False)
                os.replace(temporary_path, result_path)
                logger.info("Saved RealVLG metrics: {}", result_path)
        return metrics["F_beta"], precision, [metrics["gAcc"]]
