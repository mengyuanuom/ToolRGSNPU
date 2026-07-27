"""Paper-compatible VCoT-Grasp rotated-rectangle evaluation."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import rect_to_five


def _as_five(rectangle):
    values = np.asarray(rectangle, dtype=np.float32).reshape(-1)
    if values.size in (5, 6):
        return values[:5].copy()
    return rect_to_five(values)


def vcot_angle_within_threshold(angle1, angle2, threshold=30.0):
    """Use VCoT's 180-degree-periodic inclusive angle criterion."""

    difference = abs(float(angle1) - float(angle2)) % 180.0
    difference = min(difference, 180.0 - difference)
    return difference <= float(threshold)


def vcot_rotated_iou(grasp1, grasp2):
    """Compute continuous rotated-rectangle IoU as in VCoT-Grasp."""

    first = _as_five(grasp1)
    second = _as_five(grasp2)
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        return 0.0
    if min(first[2], first[3], second[2], second[3]) <= 0.0:
        return 0.0

    rect1 = (
        (float(first[0]), float(first[1])),
        (float(first[2]), float(first[3])),
        float(first[4]),
    )
    rect2 = (
        (float(second[0]), float(second[1])),
        (float(second[2]), float(second[3])),
        float(second[4]),
    )
    box1 = cv2.boxPoints(rect1)
    box2 = cv2.boxPoints(rect2)
    intersection, _ = cv2.intersectConvexConvex(box1, box2)
    area1 = float(first[2] * first[3])
    area2 = float(second[2] * second[3])
    union = area1 + area2 - float(intersection)
    return float(intersection) / union if union > 0.0 else 0.0


def calculate_vcot_grasp_success(
    prediction,
    targets,
    iou_threshold=0.25,
    angle_threshold=30.0,
):
    """Evaluate one predicted grasp against every VCoT ground-truth grasp."""

    if prediction is None:
        return 0
    for target in targets:
        predicted = _as_five(prediction)
        expected = _as_five(target)
        if (
            vcot_rotated_iou(predicted, expected) >= float(iou_threshold)
            and vcot_angle_within_threshold(
                predicted[4],
                expected[4],
                threshold=angle_threshold,
            )
        ):
            return 1
    return 0
