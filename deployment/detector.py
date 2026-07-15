"""Optional MMDetection adapter used by the server demo's detection tab."""

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from .config import resolve_repo_path
from toolrgs.registry import DETECTORS


class MMDetectionAdapter:
    def __init__(self, cfg: Dict[str, Any], repo_root: str):
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as exc:
            raise RuntimeError(
                "Object detection requires a compatible MMDetection/MMCV installation"
            ) from exc
        config_path = resolve_repo_path(cfg.get("config"), repo_root)
        checkpoint_path = resolve_repo_path(cfg.get("checkpoint"), repo_root)
        if config_path is None or not config_path.is_file():
            raise FileNotFoundError(f"Detector config does not exist: {config_path}")
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Detector checkpoint does not exist: {checkpoint_path}")
        self.inference_detector = inference_detector
        self.model = init_detector(
            str(config_path), str(checkpoint_path), device=str(cfg.get("device", "npu:0"))
        )
        self.threshold = float(cfg.get("score_threshold", 0.7))
        self.classes = list(cfg.get("classes") or getattr(self.model, "dataset_meta", {}).get("classes", []))

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        result = self.inference_detector(self.model, frame_bgr)
        instances = result.pred_instances.cpu()
        scores = instances.scores.numpy()
        boxes = instances.bboxes.numpy()
        labels = instances.labels.numpy()
        output = frame_bgr.copy()
        for score, box, label in zip(scores, boxes, labels):
            if float(score) < self.threshold:
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in box)
            cv2.rectangle(output, (x1, y1), (x2, y2), (30, 220, 30), 2)
            name = self.classes[int(label)] if int(label) < len(self.classes) else str(int(label))
            cv2.putText(
                output,
                f"{name} {float(score):.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 220, 30),
                2,
                cv2.LINE_AA,
            )
        return output


DETECTORS.register_module(
    MMDetectionAdapter,
    name="mmdetection",
    aliases=("mmdet", "faster_rcnn"),
)
DETECTOR_REGISTRY = DETECTORS.module_dict


def build_detector(cfg: Dict[str, Any], repo_root: str):
    component_type = cfg.get("type", "mmdetection")
    try:
        detector_class = DETECTORS.require(component_type)
    except KeyError as exc:
        available = ", ".join(sorted(DETECTORS.keys()))
        raise ValueError(
            f"Unknown detector {component_type!r}; available: {available}"
        ) from exc
    return detector_class(cfg, repo_root)
