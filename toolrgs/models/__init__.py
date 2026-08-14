"""Base model contracts and compatibility adapters."""

from .base import BaseGraspModel, model_predicts_grasp_short_side
from .legacy_adapter import LegacyOutputAdapter
from .short_side import ShortSideRegressionAdapter

__all__ = [
    "BaseGraspModel",
    "LegacyOutputAdapter",
    "ShortSideRegressionAdapter",
    "model_predicts_grasp_short_side",
]
