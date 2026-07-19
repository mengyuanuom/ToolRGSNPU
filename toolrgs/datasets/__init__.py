"""Dataset registry facade for MMDetection-style imports."""

from .builder import DATASET_REGISTRY, build_dataset

__all__ = ["DATASET_REGISTRY", "build_dataset"]
