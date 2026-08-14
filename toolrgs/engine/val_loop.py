"""Validation loop for segmentation and top-k grasp metrics."""

from loguru import logger
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from toolrgs.engine.hooks import LoopState
from toolrgs.engine.loops import BaseLoop
from toolrgs.models.base import (
    model_predicts_grasp_short_side,
    model_requires_depth,
)
from toolrgs.evaluation import (
    BinarySegmentationMetric,
    DenseGraspPostProcessor,
    GraspSuccessMetric,
    calculate_vcot_grasp_success,
    inverse_warp,
    rectangles_to_five,
    refine_with_offset,
    resolve_evaluation_protocol,
    resample_grasp_geometry,
    targets_to_six,
)
from toolrgs.registry import LOOPS, METRICS, POSTPROCESSORS
from toolrgs.runtime import current_device, move_to_device
from toolrgs.structures import GraspModelResult
from utils.config import resolve_grasp_size_activation
from utils.grasp_eval import calculate_jacquard_index


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


def _resolve_grasp_iou_thresholds(cfg, default):
    """Return the primary threshold and every threshold reported per pass."""
    primary = float(getattr(cfg, "grasp_iou_threshold", default))
    configured = getattr(cfg, "grasp_iou_thresholds", (primary,))
    if isinstance(configured, (int, float)):
        configured = (configured,)
    thresholds = []
    for value in (*configured, primary):
        threshold = float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "grasp IoU thresholds must be between 0 and 1; "
                f"got {threshold}"
            )
        if threshold not in thresholds:
            thresholds.append(threshold)
    return primary, tuple(sorted(thresholds))


@LOOPS.register_module(name="grasp_val", aliases=("validate_with_grasp",))
class GraspValLoop(BaseLoop):
    """Evaluate segmentation plus protocol-selected grasp success."""

    def __init__(self, dataloader, model, cfg, hooks=None):
        super().__init__(hooks=hooks)
        self.dataloader = dataloader
        self.model = model
        self.cfg = cfg
        self.device = current_device(int(getattr(cfg, "npu", getattr(cfg, "gpu", 0))))
        self.evaluation_protocol = resolve_evaluation_protocol(
            getattr(cfg, "evaluation_protocol", "toolrgs")
        )
        self.topk = tuple(
            getattr(
                cfg,
                "grasp_topk",
                self.evaluation_protocol.default_grasp_topk,
            )
        )
        if not self.topk or any(int(value) <= 0 for value in self.topk):
            raise ValueError(f"grasp_topk must contain positive integers, got {self.topk}")
        if self.evaluation_protocol.grasp_evaluator == "vcot_official" and self.topk != (1,):
            raise ValueError("vcot_official evaluates exactly one prediction; set grasp_topk: [1]")
        self.max_topk = max(self.topk)
        self.segmentation_metric = METRICS.build(
            getattr(cfg, "segmentation_metric", None)
            or {
                "type": "binary_segmentation",
                "mask_threshold": float(getattr(cfg, "mask_threshold", 0.35)),
            }
        )
        self.primary_grasp_iou, self.grasp_iou_thresholds = (
            _resolve_grasp_iou_thresholds(
                cfg, self.evaluation_protocol.grasp_iou_threshold
            )
        )
        grasp_metric_cfg = getattr(cfg, "grasp_metric", None) or {
            "type": "grasp_success",
            "topk": self.topk,
        }
        self.grasp_metrics_by_iou = {
            threshold: METRICS.build(grasp_metric_cfg)
            for threshold in self.grasp_iou_thresholds
        }
        # Checkpoint selection continues to use j_index from the configured
        # primary threshold (0.50 for the Grasp-Tools NPU profiles).
        self.grasp_metric = self.grasp_metrics_by_iou[self.primary_grasp_iou]
        self.postprocessor = POSTPROCESSORS.build(
            getattr(cfg, "grasp_postprocessor", None)
            or {
                "type": "dense_grasp",
                "quality_threshold": float(
                    getattr(cfg, "grasp_quality_threshold", 0.4)
                ),
                "min_distance": int(getattr(cfg, "grasp_min_distance", 2)),
                "minimum_width": self.evaluation_protocol.minimum_grasp_width,
                "width_factor": float(
                    getattr(cfg, "grasp_size_factor", 100.0)
                ),
                "grasp_height": float(getattr(cfg, "grasp_height", 20.0)),
                "size_coordinate": str(
                    getattr(cfg, "grasp_size_coordinate", "original")
                ),
            }
        )
        self.grasp_size_activation = resolve_grasp_size_activation(
            getattr(cfg, "grasp_size_activation", "auto"), model=model
        )

    def _decode_size(self, prediction):
        if self.grasp_size_activation == "sigmoid":
            return torch.sigmoid(prediction)
        return prediction.clamp(0.0, 1.0)

    def _offset_radius(self, input_hw):
        configured = getattr(self.cfg, "offset_r", None)
        if configured is not None and float(configured) > 0:
            return float(configured)
        return max(1.0, min(input_hw) / 20.0)

    def _global_results(self, device):
        ious = np.asarray(self.segmentation_metric.ious, dtype=np.float64)
        values = [float(ious.sum()), float(ious.size)]
        values.extend(float((ious > threshold).sum()) for threshold in self.segmentation_metric.iou_thresholds)
        for threshold in self.grasp_iou_thresholds:
            metric = self.grasp_metrics_by_iou[threshold]
            for topk in self.topk:
                values.extend([metric.correct[topk], metric.total[topk]])
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
        j_index_by_iou = {}
        for threshold in self.grasp_iou_thresholds:
            values_for_threshold = []
            for _topk in self.topk:
                correct, total = statistics[cursor], statistics[cursor + 1]
                values_for_threshold.append(correct / max(1.0, total))
                cursor += 2
            j_index_by_iou[f"{threshold:.2f}"] = values_for_threshold
        j_index = j_index_by_iou[f"{self.primary_grasp_iou:.2f}"]
        return float(iou), precision, j_index, j_index_by_iou

    @torch.no_grad()
    def run_epoch(self, epoch: int):
        self.state = LoopState(epoch=epoch)
        self.hooks.call("before_epoch", self, self.state)
        self.segmentation_metric.reset()
        for metric in self.grasp_metrics_by_iou.values():
            metric.reset()
        self.model.eval()
        # Evaluation needs no gradient synchronization. Calling the wrapped
        # module directly lets exact non-padding rank shards have unequal
        # lengths without mismatched DDP forward collectives.
        evaluation_model = getattr(self.model, "module", self.model)
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
            model_kwargs = {}
            target_short_side = None
            if model_predicts_grasp_short_side(self.model):
                short_side_mask = data["grasp_masks"].get("short")
                if short_side_mask is None:
                    raise KeyError(
                        "A short-side model requires batch['grasp_masks']['short']"
                    )
                target_short_side = move_to_device(
                    short_side_mask, device
                ).unsqueeze(1)
                model_kwargs["grasp_short_mask"] = target_short_side

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
                depth = data.get("depth")
                if depth is None:
                    raise KeyError(
                        "The selected model requires batch['depth'], but the "
                        "validation dataset did not provide it."
                    )
                inputs = (image, move_to_device(depth, device), *inputs[1:])
            result = GraspModelResult.from_legacy(
                evaluation_model(*inputs, **model_kwargs)
            )
            predictions = result.predictions
            input_hw = image.shape[-2:]
            segmentation = _resize_prediction(
                torch.sigmoid(predictions.segmentation), input_hw
            )
            quality = _resize_prediction(torch.sigmoid(predictions.quality), input_hw)
            sine = _resize_prediction(predictions.sine, input_hw)
            cosine = _resize_prediction(predictions.cosine, input_hw)
            width = _resize_prediction(
                self._decode_size(predictions.width), input_hw
            )
            offset = None
            short_side = None
            if predictions.short_side is not None:
                short_side = _resize_prediction(
                    self._decode_size(predictions.short_side), input_hw
                )
            if predictions.offset is not None:
                offset = _resize_prediction(predictions.offset, input_hw, mode="bilinear")

            dense_tensors = [
                segmentation,
                target_segmentation,
                quality,
                sine,
                cosine,
                width,
            ]
            if short_side is not None:
                dense_tensors.append(short_side)
            if offset is not None:
                dense_tensors.append(offset)
            dense_maps = (
                torch.cat(dense_tensors, dim=1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            next_channel = 6
            short_side_maps = None
            if short_side is not None:
                short_side_maps = dense_maps[:, next_channel]
                next_channel += 1
            offset_maps = (
                dense_maps[:, next_channel:next_channel + 2] if offset is not None else None
            )

            for index in range(image.shape[0]):
                inverse_matrix = data["inverse"][index]
                if hasattr(inverse_matrix, "detach"):
                    inverse_matrix = inverse_matrix.detach().cpu().numpy()
                original_hw = (
                    int(data["ori_size"][index][0]),
                    int(data["ori_size"][index][1]),
                )
                predicted_segmentation = inverse_warp(
                    dense_maps[index, 0],
                    inverse_matrix,
                    original_hw,
                    interpolation=self.evaluation_protocol.inverse_interpolation,
                )
                target_segmentation_original = inverse_warp(
                    dense_maps[index, 1],
                    inverse_matrix,
                    original_hw,
                    interpolation=self.evaluation_protocol.inverse_interpolation,
                )
                target_mask_threshold = (
                    self.evaluation_protocol.target_mask_threshold
                )
                if target_mask_threshold is not None:
                    target_segmentation_original = (
                        target_segmentation_original > target_mask_threshold
                    )
                self.segmentation_metric.update(
                    predicted_segmentation,
                    target_segmentation_original,
                )

                quality_original = inverse_warp(
                    dense_maps[index, 2],
                    inverse_matrix,
                    original_hw,
                    interpolation=self.evaluation_protocol.inverse_interpolation,
                )
                sine_original = inverse_warp(
                    dense_maps[index, 3],
                    inverse_matrix,
                    original_hw,
                    interpolation=self.evaluation_protocol.inverse_interpolation,
                )
                cosine_original = inverse_warp(
                    dense_maps[index, 4],
                    inverse_matrix,
                    original_hw,
                    interpolation=self.evaluation_protocol.inverse_interpolation,
                )
                width_original = inverse_warp(
                    dense_maps[index, 5],
                    inverse_matrix,
                    original_hw,
                    interpolation=self.evaluation_protocol.inverse_interpolation,
                )
                short_side_original = None
                if short_side_maps is not None:
                    short_side_original = inverse_warp(
                        short_side_maps[index],
                        inverse_matrix,
                        original_hw,
                        interpolation=self.evaluation_protocol.inverse_interpolation,
                    )
                grasp_targets = data["grasps"][index]
                if hasattr(grasp_targets, "detach"):
                    grasp_targets = grasp_targets.detach().cpu().numpy()

                target_six = targets_to_six(grasp_targets)

                size_scale = 1.0
                if self.postprocessor.size_coordinate == "canvas":
                    linear = np.asarray(inverse_matrix, dtype=np.float32)[:, :2]
                    size_scale = float(
                        0.5
                        * (
                            np.linalg.norm(linear[:, 0])
                            + np.linalg.norm(linear[:, 1])
                        )
                    )

                detections = self.postprocessor(
                    quality_original,
                    sine_original,
                    cosine_original,
                    width_original,
                    num_grasps=self.max_topk,
                    short_side=short_side_original,
                    spatial_scale=size_scale,
                )
                rectangles = [detection.as_rectangle() for detection in detections]
                if offset_maps is not None and rectangles:
                    rectangles = refine_with_offset(
                        rectangles,
                        offset_maps[index : index + 1],
                        inverse_matrix,
                        self._offset_radius(input_hw),
                    )
                    if bool(
                        getattr(self.cfg, "offset_resample_geometry", False)
                    ):
                        rectangles = resample_grasp_geometry(
                            rectangles,
                            sine_original,
                            cosine_original,
                            width_original,
                            width_factor=(
                                self.postprocessor.width_factor * size_scale
                            ),
                            short_side=short_side_original,
                        )
                else:
                    rectangles = rectangles_to_five(rectangles)

                for threshold, metric in self.grasp_metrics_by_iou.items():
                    if self.evaluation_protocol.grasp_evaluator == "vcot_official":
                        success = calculate_vcot_grasp_success(
                            rectangles[0] if rectangles else None,
                            target_six,
                            iou_threshold=threshold,
                            angle_threshold=float(
                                getattr(
                                    self.cfg,
                                    "grasp_angle_threshold",
                                    self.evaluation_protocol.grasp_angle_threshold,
                                )
                            ),
                        )
                        metric.update(1, success)
                    else:
                        for topk in self.topk:
                            success = calculate_jacquard_index(
                                rectangles[:topk],
                                target_six,
                                iou_threshold=threshold,
                                shape=self.evaluation_protocol.grasp_canvas,
                                angle_threshold=float(
                                    getattr(
                                        self.cfg,
                                        "grasp_angle_threshold",
                                        self.evaluation_protocol.grasp_angle_threshold,
                                    )
                                ),
                                max_width=self.postprocessor.width_factor,
                                grasp_height=self.postprocessor.grasp_height,
                            )
                            metric.update(topk, success)

            self.state.result = result
            self.hooks.call("after_iter", self, self.state)

        iou, precision, j_index, j_index_by_iou = self._global_results(device)
        self.state.logs = {
            "iou": iou,
            "precision": precision,
            "j_index": j_index,
            "j_index_by_iou": j_index_by_iou,
        }
        self.hooks.call("after_epoch", self, self.state)
        if rank == 0:
            precision_text = "  ".join(
                f"{name}: {100.0 * value:.2f}" for name, value in precision.items()
            )
            if self.evaluation_protocol.grasp_evaluator == "vcot_official":
                grasp_text = "  ".join(
                    f"{self.evaluation_protocol.grasp_metric_label}"
                    f"(IoU={threshold}): {100.0 * values[0]:.2f}"
                    for threshold, values in j_index_by_iou.items()
                )
            else:
                grasp_text = "  ".join(
                    f"J@{topk}(IoU={threshold}): {100.0 * value:.2f}"
                    for threshold, values in j_index_by_iou.items()
                    for topk, value in zip(self.topk, values)
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
