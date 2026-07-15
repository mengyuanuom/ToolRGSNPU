"""Validation loop for segmentation and top-k grasp metrics."""

from loguru import logger
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from toolrgs.engine.hooks import LoopState
from toolrgs.engine.loops import BaseLoop
from toolrgs.evaluation import (
    BinarySegmentationMetric,
    DenseGraspPostProcessor,
    GraspSuccessMetric,
    inverse_warp,
    rectangles_to_five,
    refine_with_offset,
    targets_to_six,
)
from toolrgs.registry import LOOPS, METRICS, POSTPROCESSORS
from toolrgs.runtime import current_device, move_to_device
from toolrgs.structures import GraspModelResult
from utils.grasp_eval import calculate_jacquard_index


def _sample_map(tensor, index):
    array = tensor[index].detach().float().cpu().numpy()
    return np.squeeze(array)


def _resize_prediction(tensor, output_hw, mode="bicubic"):
    if tensor.shape[-2:] == tuple(output_hw):
        return tensor
    return F.interpolate(
        tensor,
        size=tuple(output_hw),
        mode=mode,
        # Preserve the historical evaluator's interpolation contract. Offset
        # vectors deliberately use bilinear/False because they are sampled in
        # input coordinates; the five dense maps use bicubic/True.
        align_corners=False if mode == "bilinear" else True,
    )


@LOOPS.register_module(name="grasp_val", aliases=("validate_with_grasp",))
class GraspValLoop(BaseLoop):
    """Evaluate instance IoU plus top-1/top-5 grasp Jacquard success."""

    def __init__(self, dataloader, model, cfg, hooks=None):
        super().__init__(hooks=hooks)
        self.dataloader = dataloader
        self.model = model
        self.cfg = cfg
        self.device = current_device(int(getattr(cfg, "npu", getattr(cfg, "gpu", 0))))
        self.topk = tuple(getattr(cfg, "grasp_topk", (1, 5)))
        self.segmentation_metric = METRICS.build(
            getattr(cfg, "segmentation_metric", None)
            or {
                "type": "binary_segmentation",
                "mask_threshold": float(getattr(cfg, "mask_threshold", 0.35)),
            }
        )
        self.grasp_metric = METRICS.build(
            getattr(cfg, "grasp_metric", None)
            or {"type": "grasp_success", "topk": self.topk}
        )
        self.postprocessor = POSTPROCESSORS.build(
            getattr(cfg, "grasp_postprocessor", None)
            or {
                "type": "dense_grasp",
                "quality_threshold": float(
                    getattr(cfg, "grasp_quality_threshold", 0.4)
                ),
                "min_distance": int(getattr(cfg, "grasp_min_distance", 2)),
            }
        )

    def _offset_radius(self, input_hw):
        configured = getattr(self.cfg, "offset_r", None)
        if configured is not None and float(configured) > 0:
            return float(configured)
        return max(1.0, min(input_hw) / 20.0)

    def _global_results(self, device):
        ious = np.asarray(self.segmentation_metric.ious, dtype=np.float64)
        values = [float(ious.sum()), float(ious.size)]
        values.extend(float((ious > threshold).sum()) for threshold in self.segmentation_metric.iou_thresholds)
        for topk in self.topk:
            values.extend(
                [self.grasp_metric.correct[topk], self.grasp_metric.total[topk]]
            )
        statistics = torch.tensor(values, dtype=torch.float32, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        statistics = statistics.cpu().tolist()
        count = max(1.0, statistics[1])
        iou = statistics[0] / count
        precision = {
            f"Pr@{int(round(threshold * 100))}": statistics[2 + index] / count
            for index, threshold in enumerate(self.segmentation_metric.iou_thresholds)
        }
        cursor = 2 + len(self.segmentation_metric.iou_thresholds)
        j_index = []
        for _topk in self.topk:
            correct, total = statistics[cursor], statistics[cursor + 1]
            j_index.append(correct / max(1.0, total))
            cursor += 2
        return float(iou), precision, j_index

    @torch.no_grad()
    def run_epoch(self, epoch: int):
        self.state = LoopState(epoch=epoch)
        self.hooks.call("before_epoch", self, self.state)
        self.segmentation_metric.reset()
        self.grasp_metric.reset()
        self.model.eval()
        rank = int(getattr(self.cfg, "rank", 0))
        progress = tqdm(self.dataloader, disable=rank != 0)
        device = self.device

        for iteration, data in enumerate(progress):
            self.state.iteration = iteration
            self.state.batch = data
            self.hooks.call("before_iter", self, self.state)

            image = move_to_device(data["img"], device)
            text = move_to_device(data["word_vec"], device)
            target_segmentation = move_to_device(data["mask"], device).unsqueeze(1)
            target_quality = move_to_device(data["grasp_masks"]["qua"], device).unsqueeze(1)
            target_sine = move_to_device(data["grasp_masks"]["sin"], device).unsqueeze(1)
            target_cosine = move_to_device(data["grasp_masks"]["cos"], device).unsqueeze(1)
            target_width = move_to_device(data["grasp_masks"]["wid"], device).unsqueeze(1)

            result = GraspModelResult.from_legacy(
                self.model(
                    image,
                    text,
                    target_segmentation,
                    target_quality,
                    target_sine,
                    target_cosine,
                    target_width,
                )
            )
            predictions = result.predictions
            input_hw = image.shape[-2:]
            segmentation = _resize_prediction(
                torch.sigmoid(predictions.segmentation), input_hw
            )
            quality = _resize_prediction(torch.sigmoid(predictions.quality), input_hw)
            sine = _resize_prediction(predictions.sine, input_hw)
            cosine = _resize_prediction(predictions.cosine, input_hw)
            width = _resize_prediction(torch.sigmoid(predictions.width), input_hw)
            offset = None
            if predictions.offset is not None:
                offset = _resize_prediction(predictions.offset, input_hw, mode="bilinear")

            for index in range(image.shape[0]):
                inverse_matrix = data["inverse"][index]
                if hasattr(inverse_matrix, "detach"):
                    inverse_matrix = inverse_matrix.detach().cpu().numpy()
                original_hw = (
                    int(data["ori_size"][index][0]),
                    int(data["ori_size"][index][1]),
                )
                predicted_segmentation = inverse_warp(
                    _sample_map(segmentation, index), inverse_matrix, original_hw
                )
                target_segmentation_original = inverse_warp(
                    _sample_map(target_segmentation, index), inverse_matrix, original_hw
                )
                self.segmentation_metric.update(
                    predicted_segmentation,
                    target_segmentation_original > 0.5,
                )

                quality_original = inverse_warp(
                    _sample_map(quality, index), inverse_matrix, original_hw
                )
                sine_original = inverse_warp(
                    _sample_map(sine, index), inverse_matrix, original_hw
                )
                cosine_original = inverse_warp(
                    _sample_map(cosine, index), inverse_matrix, original_hw
                )
                width_original = inverse_warp(
                    _sample_map(width, index), inverse_matrix, original_hw
                )
                grasp_targets = data["grasps"][index]
                if hasattr(grasp_targets, "detach"):
                    grasp_targets = grasp_targets.detach().cpu().numpy()
                target_six = targets_to_six(grasp_targets)

                for topk in self.topk:
                    detections = self.postprocessor(
                        quality_original,
                        sine_original,
                        cosine_original,
                        width_original,
                        num_grasps=topk,
                    )
                    rectangles = [detection.as_rectangle() for detection in detections]
                    if offset is not None and rectangles:
                        rectangles = refine_with_offset(
                            rectangles,
                            offset[index : index + 1],
                            inverse_matrix,
                            self._offset_radius(input_hw),
                        )
                    else:
                        rectangles = rectangles_to_five(rectangles)
                    success = calculate_jacquard_index(rectangles, target_six)
                    self.grasp_metric.update(topk, success)

            self.state.result = result
            self.hooks.call("after_iter", self, self.state)

        iou, precision, j_index = self._global_results(device)
        self.state.logs = {
            "iou": iou,
            "precision": precision,
            "j_index": j_index,
        }
        self.hooks.call("after_epoch", self, self.state)
        if rank == 0:
            precision_text = "  ".join(
                f"{name}: {100.0 * value:.2f}" for name, value in precision.items()
            )
            grasp_text = "  ".join(
                f"J_index@{topk}: {100.0 * value:.2f}"
                for topk, value in zip(self.topk, j_index)
            )
            logger.info(
                "Evaluation: Epoch=[{}/{}]  IoU={:.2f}  {}  {}",
                epoch,
                self.cfg.epochs,
                100.0 * iou,
                grasp_text,
                precision_text,
            )
        return iou, precision, j_index
