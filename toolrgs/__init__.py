"""ToolRGS component framework.

The legacy ``model`` and ``utils`` packages remain public while components are
gradually migrated into this MMDetection-style namespace.
"""

from .registry import (
    AUDIO_INPUTS,
    CAMERAS,
    DATASETS,
    DETECTORS,
    HOOKS,
    LOOPS,
    LOSSES,
    METRICS,
    MODELS,
    POSTPROCESSORS,
    ROBOT_CLIENTS,
    TRANSFORMS,
    Registry,
    build_from_cfg,
)

__all__ = [
    "AUDIO_INPUTS",
    "CAMERAS",
    "DATASETS",
    "DETECTORS",
    "HOOKS",
    "LOOPS",
    "LOSSES",
    "METRICS",
    "MODELS",
    "POSTPROCESSORS",
    "ROBOT_CLIENTS",
    "TRANSFORMS",
    "Registry",
    "build_from_cfg",
]
