"""Base model contracts and compatibility adapters."""

from .base import BaseGraspModel
from .legacy_adapter import LegacyOutputAdapter

__all__ = ["BaseGraspModel", "LegacyOutputAdapter"]
