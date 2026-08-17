"""Registered metrics and prediction postprocessors."""

from .metrics import BinarySegmentationMetric, GraspSuccessMetric
from .postprocessors import DenseGraspPostProcessor, GraspDetection
from .protocols import EvaluationProtocol, resolve_evaluation_protocol
from .geometry import (
    apply_affine,
    corners_to_five,
    five_to_corners,
    inverse_warp,
    rect_to_five,
    grasp_relative_offset_scale,
    refine_with_grasp_relative_offset,
    rectangles_to_five,
    refine_with_offset,
    resample_grasp_geometry,
    targets_to_six,
)
from .vcot import (
    calculate_vcot_grasp_success,
    vcot_angle_within_threshold,
    vcot_rotated_iou,
)

__all__ = [
    "BinarySegmentationMetric",
    "DenseGraspPostProcessor",
    "EvaluationProtocol",
    "GraspDetection",
    "GraspSuccessMetric",
    "apply_affine",
    "calculate_vcot_grasp_success",
    "corners_to_five",
    "five_to_corners",
    "inverse_warp",
    "grasp_relative_offset_scale",
    "refine_with_grasp_relative_offset",
    "rect_to_five",
    "rectangles_to_five",
    "refine_with_offset",
    "resolve_evaluation_protocol",
    "resample_grasp_geometry",
    "targets_to_six",
    "vcot_angle_within_threshold",
    "vcot_rotated_iou",
]
