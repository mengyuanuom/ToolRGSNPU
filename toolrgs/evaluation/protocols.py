"""Named evaluation protocols for reproducible benchmark comparisons."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass(frozen=True)
class EvaluationProtocol:
    """Implementation details that are not captured by metric names alone."""

    name: str
    inverse_interpolation: int
    grasp_canvas: Tuple[int, int]
    target_mask_threshold: Optional[float]
    minimum_grasp_width: float
    grasp_iou_threshold: float = 0.25
    grasp_angle_threshold: float = 30.0
    grasp_evaluator: str = "rasterized_jacquard"
    default_grasp_topk: Tuple[int, ...] = (1, 5)
    grasp_metric_label: str = "J_index"


_PROTOCOLS = {
    "toolrgs": EvaluationProtocol(
        name="toolrgs",
        inverse_interpolation=cv2.INTER_NEAREST,
        grasp_canvas=(720, 1280),
        target_mask_threshold=0.5,
        minimum_grasp_width=1.0,
    ),
    # Reproduce the public CROG source, including its fixed OCID canvas and
    # historical x/y rasterization behavior in utils.grasp_eval.
    "crog_legacy": EvaluationProtocol(
        name="crog_legacy",
        inverse_interpolation=cv2.INTER_CUBIC,
        grasp_canvas=(480, 640),
        target_mask_threshold=None,
        minimum_grasp_width=0.0,
    ),
    # Match VCoT-Grasp eval_cli.py/inference.py: one grasp prediction,
    # continuous OpenCV rotated IoU, inclusive thresholds, and original GT size.
    "vcot_official": EvaluationProtocol(
        name="vcot_official",
        inverse_interpolation=cv2.INTER_NEAREST,
        grasp_canvas=(416, 416),
        target_mask_threshold=0.5,
        minimum_grasp_width=0.0,
        grasp_evaluator="vcot_official",
        default_grasp_topk=(1,),
        grasp_metric_label="GraspSR",
    ),
}

_ALIASES = {
    "crog": "crog_legacy",
    "crog_source": "crog_legacy",
    "vcot": "vcot_official",
    "vcot_source": "vcot_official",
    "default": "toolrgs",
}


def resolve_evaluation_protocol(name="toolrgs"):
    """Resolve a stable protocol name and reject silent misspellings."""

    normalized = str(name or "toolrgs").strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    try:
        return _PROTOCOLS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_PROTOCOLS))
        raise ValueError(
            f"Unknown evaluation_protocol {name!r}; choose one of: {available}"
        ) from exc
