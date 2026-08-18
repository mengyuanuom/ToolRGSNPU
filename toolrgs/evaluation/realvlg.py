"""Metrics matching the public RealVLG-R1 evaluation implementation.

The benchmark's public evaluator predicts ``(x, y, theta, width)`` and builds
the fifth rectangle dimension with a fixed 40-pixel gripper depth.  Grasp
metrics are conditioned on valid predictions, matching the benchmark's
Validity Rate contract.
"""

from __future__ import annotations

import cv2
import numpy as np


REALVLG_GRIPPER_DEPTH = 40.0
REALVLG_IOU_THRESHOLD = 0.25
REALVLG_ANGLE_THRESHOLD = 30.0


def realvlg_rect_to_points8(
    x,
    y,
    theta,
    width,
    gripper_depth=REALVLG_GRIPPER_DEPTH,
):
    """Copy the official ``(x,y,theta,width)`` rectangle construction."""

    theta_rad = np.deg2rad(float(theta))
    dx = (float(width) / 2.0) * np.cos(theta_rad)
    dy = (float(width) / 2.0) * np.sin(theta_rad)
    dz = float(gripper_depth) / 2.0
    return np.asarray(
        [
            float(x) - dx - dz * np.sin(theta_rad),
            float(y) - dy + dz * np.cos(theta_rad),
            float(x) + dx - dz * np.sin(theta_rad),
            float(y) + dy + dz * np.cos(theta_rad),
            float(x) + dx + dz * np.sin(theta_rad),
            float(y) + dy - dz * np.cos(theta_rad),
            float(x) - dx + dz * np.sin(theta_rad),
            float(y) - dy - dz * np.cos(theta_rad),
        ],
        dtype=np.float64,
    )


def realvlg_points8_to_rect(points8):
    """Copy the official 8-point to ``(x,y,theta,width)`` conversion."""

    values = np.asarray(points8, dtype=np.float64).reshape(-1)
    if values.shape[0] != 8:
        raise ValueError(
            "RealVLG points8 should have 8 elements, got "
            f"{values.shape[0]}"
        )
    corners = values.reshape(4, 2)
    center = corners.mean(axis=0)
    right_midpoint = corners[[1, 2]].mean(axis=0)
    left_midpoint = corners[[0, 3]].mean(axis=0)
    width = np.linalg.norm(right_midpoint - left_midpoint)
    edge = corners[1] - corners[0]
    theta = np.rad2deg(np.arctan2(edge[1], edge[0]))
    return float(center[0]), float(center[1]), float(theta), float(width)


def realvlg_polygon_iou(points1, points2):
    """Continuous polygon IoU for the convex rectangles used by RealVLG."""

    polygon1 = np.asarray(points1, dtype=np.float32).reshape(4, 2)
    polygon2 = np.asarray(points2, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(polygon1).all() or not np.isfinite(polygon2).all():
        return 0.0
    area1 = abs(float(cv2.contourArea(polygon1)))
    area2 = abs(float(cv2.contourArea(polygon2)))
    if area1 <= 0.0 or area2 <= 0.0:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(polygon1, polygon2)
    except cv2.error:
        return 0.0
    union = area1 + area2 - float(intersection)
    return float(intersection) / union if union > 0.0 else 0.0


def realvlg_angular_diff(pred_theta, gt_theta):
    """Match the official evaluator, including its radians auto-detection."""

    pred_theta = float(pred_theta)
    gt_theta = float(gt_theta)
    if abs(pred_theta) <= np.pi and abs(gt_theta) <= np.pi:
        pred_rad, gt_rad = pred_theta, gt_theta
    else:
        pred_rad, gt_rad = np.deg2rad(pred_theta), np.deg2rad(gt_theta)
    pred_vector = np.asarray([np.cos(pred_rad), np.sin(pred_rad)])
    gt_vector = np.asarray([np.cos(gt_rad), np.sin(gt_rad)])
    cosine = np.clip(np.dot(pred_vector, gt_vector), -1.0, 1.0)
    difference = float(np.rad2deg(np.arccos(cosine)))
    return 180.0 - difference if difference > 90.0 else difference


def evaluate_realvlg_grasp(prediction, ground_truth_points8):
    """Evaluate one ToolRGS grasp against every official RealVLG target.

    ``prediction`` uses ToolRGS order ``[x,y,width,height,theta]``.  Its height
    is deliberately ignored because the official metric always uses a fixed
    40-pixel gripper depth.
    """

    if prediction is None:
        return 0.0, 999.0, False
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if values.shape[0] < 5 or not np.isfinite(values[:5]).all():
        return 0.0, 999.0, False
    pred_x, pred_y, pred_width, _pred_height, pred_theta = values[:5]
    if pred_width <= 0.0:
        return 0.0, 999.0, False
    predicted_points = realvlg_rect_to_points8(
        pred_x, pred_y, pred_theta, pred_width
    )
    targets = np.asarray(ground_truth_points8, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[None, :]
    if targets.ndim == 3 and targets.shape[1:] == (4, 2):
        targets = targets.reshape(-1, 8)
    if targets.ndim != 2 or targets.shape[1] != 8:
        raise ValueError(
            "RealVLG ground-truth grasps must have shape [N,8] or [N,4,2], "
            f"got {targets.shape}"
        )

    best_iou, best_angle = 0.0, 999.0
    for target_points in targets:
        if not np.isfinite(target_points).all():
            continue
        _x, _y, target_theta, _width = realvlg_points8_to_rect(target_points)
        iou = realvlg_polygon_iou(predicted_points, target_points)
        difference = realvlg_angular_diff(pred_theta, target_theta)
        if iou > best_iou:
            best_iou, best_angle = iou, difference
    correct = (
        best_iou > REALVLG_IOU_THRESHOLD
        and best_angle < REALVLG_ANGLE_THRESHOLD
    )
    return float(best_iou), float(best_angle), bool(correct)


def realvlg_f_measure(pred_mask, gt_mask):
    """The public evaluator's binary F-measure implementation."""

    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    true_positive = np.logical_and(pred, gt).sum()
    false_positive = np.logical_and(pred, np.logical_not(gt)).sum()
    false_negative = np.logical_and(np.logical_not(pred), gt).sum()
    precision = true_positive / (true_positive + false_positive + 1e-7)
    recall = true_positive / (true_positive + false_negative + 1e-7)
    return float(2 * precision * recall / (precision + recall + 1e-7))


def realvlg_s_measure(pred_mask, gt_mask, alpha=0.5):
    """The public evaluator's S-measure implementation."""

    pred = np.asarray(pred_mask).astype(np.float32)
    gt = np.asarray(gt_mask).astype(np.float32)
    foreground = pred * gt
    background = (1.0 - pred) * (1.0 - gt)
    object_score = alpha * foreground.mean() + (1.0 - alpha) * background.mean()
    height, width = pred.shape
    middle_h, middle_w = height // 2, width // 2
    regions = (
        (0, middle_h, 0, middle_w),
        (0, middle_h, middle_w, width),
        (middle_h, height, 0, middle_w),
        (middle_h, height, middle_w, width),
    )
    region_score = 0.0
    for row0, row1, col0, col1 in regions:
        pred_region = pred[row0:row1, col0:col1]
        gt_region = gt[row0:row1, col0:col1]
        pred_mean = pred_region.mean()
        gt_mean = gt_region.mean()
        region_score += (
            2.0
            * pred_mean
            * gt_mean
            / (pred_mean**2 + gt_mean**2 + 1e-7)
        )
    region_score /= 4.0
    return float(alpha * object_score + (1.0 - alpha) * region_score)


def realvlg_e_measure(pred_mask, gt_mask):
    """The public evaluator's E-measure implementation."""

    pred = np.asarray(pred_mask).astype(np.float32)
    gt = np.asarray(gt_mask).astype(np.float32)
    pred_centered = pred - pred.mean()
    gt_centered = gt - gt.mean()
    alignment = 2.0 * pred_centered * gt_centered / (
        pred_centered**2 + gt_centered**2 + 1e-7
    )
    return float(np.mean((alignment + 1.0) / 2.0))


def realvlg_mask_to_bbox(mask):
    """Return a tight ``[x1,y1,x2,y2]`` box for a nonempty binary mask."""

    rows, columns = np.nonzero(np.asarray(mask).astype(bool))
    if not len(rows):
        return None
    return np.asarray(
        [
            float(columns.min()),
            float(rows.min()),
            float(columns.max() + 1),
            float(rows.max() + 1),
        ],
        dtype=np.float64,
    )


def realvlg_giou(box_a, box_b):
    """The public evaluator's generalized box IoU."""

    first = np.asarray(box_a, dtype=np.float64)
    second = np.asarray(box_b, dtype=np.float64)
    intersection_x1 = max(first[0], second[0])
    intersection_y1 = max(first[1], second[1])
    intersection_x2 = min(first[2], second[2])
    intersection_y2 = min(first[3], second[3])
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    area_first = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    area_second = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = area_first + area_second - intersection
    iou = intersection / union if union > 0.0 else 0.0
    enclosing_x1 = min(first[0], second[0])
    enclosing_y1 = min(first[1], second[1])
    enclosing_x2 = max(first[2], second[2])
    enclosing_y2 = max(first[3], second[3])
    enclosing = (enclosing_x2 - enclosing_x1) * (
        enclosing_y2 - enclosing_y1
    )
    return float(iou - (enclosing - union) / enclosing) if enclosing > 0 else 0.0


def realvlg_ciou(box_a, box_b):
    """The public evaluator's center-distance box IoU variant."""

    first = np.asarray(box_a, dtype=np.float64)
    second = np.asarray(box_b, dtype=np.float64)
    intersection_x1 = max(first[0], second[0])
    intersection_y1 = max(first[1], second[1])
    intersection_x2 = min(first[2], second[2])
    intersection_y2 = min(first[3], second[3])
    intersection = max(0.0, intersection_x2 - intersection_x1) * max(
        0.0, intersection_y2 - intersection_y1
    )
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    iou = intersection / union if union > 0.0 else 0.0
    center_first = np.asarray(
        [(first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0]
    )
    center_second = np.asarray(
        [(second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0]
    )
    center_distance = np.sum((center_first - center_second) ** 2)
    enclosing_x1 = min(first[0], second[0])
    enclosing_y1 = min(first[1], second[1])
    enclosing_x2 = max(first[2], second[2])
    enclosing_y2 = max(first[3], second[3])
    diagonal = (enclosing_x2 - enclosing_x1) ** 2 + (
        enclosing_y2 - enclosing_y1
    ) ** 2
    return float(iou - center_distance / (diagonal + 1e-7))
