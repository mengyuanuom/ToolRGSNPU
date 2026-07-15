"""Stateful evaluation metrics independent of models and datasets."""

from typing import Iterable

import numpy as np

from toolrgs.registry import METRICS


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@METRICS.register_module(name="binary_segmentation", aliases=("segmentation_iou",))
class BinarySegmentationMetric:
    """Mean per-sample IoU and precision at configurable IoU thresholds."""

    def __init__(
        self,
        mask_threshold: float = 0.35,
        iou_thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9),
        from_logits: bool = False,
    ):
        self.mask_threshold = float(mask_threshold)
        self.iou_thresholds = tuple(float(value) for value in iou_thresholds)
        self.from_logits = bool(from_logits)
        self.reset()

    def reset(self):
        self.ious = []

    def update(self, prediction, target):
        prediction = _numpy(prediction).astype(np.float32)
        target = _numpy(target)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Segmentation prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
            )
        if self.from_logits:
            prediction = 1.0 / (1.0 + np.exp(-prediction))
        if prediction.ndim == 2:
            prediction = prediction[None]
            target = target[None]
        prediction = prediction.reshape(prediction.shape[0], -1) > self.mask_threshold
        target = target.reshape(target.shape[0], -1).astype(bool)
        intersection = np.logical_and(prediction, target).sum(axis=1)
        union = np.logical_or(prediction, target).sum(axis=1)
        self.ious.extend((intersection / (union + 1e-6)).tolist())

    def compute(self):
        values = np.asarray(self.ious, dtype=np.float64)
        mean_iou = float(values.mean()) if values.size else 0.0
        precision = {
            f"Pr@{int(round(threshold * 100))}": (
                float((values > threshold).mean()) if values.size else 0.0
            )
            for threshold in self.iou_thresholds
        }
        return {"iou": mean_iou, "precision": precision, "num_samples": int(values.size)}


@METRICS.register_module(name="grasp_success", aliases=("j_index",))
class GraspSuccessMetric:
    """Aggregate binary Jacquard successes for one or more top-k settings."""

    def __init__(self, topk=(1, 5)):
        self.topk = tuple(int(value) for value in topk)
        self.reset()

    def reset(self):
        self.correct = {value: 0.0 for value in self.topk}
        self.total = {value: 0 for value in self.topk}

    def update(self, topk: int, success):
        topk = int(topk)
        if topk not in self.correct:
            raise KeyError(f"top-k {topk} was not configured; available: {self.topk}")
        self.correct[topk] += float(success)
        self.total[topk] += 1

    def compute(self):
        return {
            f"J@{value}": self.correct[value] / max(1, self.total[value])
            for value in self.topk
        }
