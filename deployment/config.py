"""Deployment configuration loading with repository-relative paths."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "config": "config/grasp_tools/drogoff.yaml",
        "checkpoint": "exp/grasp_tools/drogoff_grasp_tools/best_jindex_model.pth",
        "device": "npu:0",
        "prompt": "the tool",
        "mask_threshold": 0.35,
        "quality_threshold": 0.4,
        "num_grasps": 1,
        "postprocessor": {
            "type": "dense_grasp",
            "min_distance": 2,
            "width_factor": 100.0,
            "grasp_height": 20.0,
        },
        "gate_quality_by_mask": True,
        "scale_grasp_to_source": True,
        "overrides": {},
    },
    "camera": {
        "backend": "opencv",
        "device": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "image_path": "",
        "video_path": "",
        "gstreamer_pipeline": "",
    },
    "robot": {
        "type": "legacy_tcp",
        "enabled": False,
        "host": "192.168.38.10",
        "port": 3000,
        "timeout_s": 2.0,
        "auto_send": False,
        "auto_send_interval_s": 2.0,
        "default_depth": 0,
        "coordinate_space": "source",
        "limits": {
            "x": [0, 1280],
            "y": [0, 720],
            "theta": [-90, 90],
            "width": [1, 600],
            "depth": [-1, 1],
        },
    },
    "detector": {
        "type": "mmdetection",
        "enabled": False,
        "config": "config/deployment/faster-rcnn-13.py",
        "checkpoint": "weights/epoch_48_13.pth",
        "device": "npu:0",
        "score_threshold": 0.7,
        "classes": [],
    },
    "audio": {
        "type": "whisper",
        "enabled": False,
        "model": "small",
        "device": "cpu",
        "sample_rate": 16000,
        "duration_s": 4.0,
        "language": "en",
    },
    "gui": {
        "title": "ToolRGSNPU Real-world Grasp Demo",
        "window_width": 1500,
        "window_height": 900,
        "camera_interval_ms": 33,
        "inference_interval_ms": 400,
        "continuous_inference": True,
    },
}


def _deep_merge(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(
    value: Union[str, Path, None], repo_root: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """Resolve a deployment path relative to the ToolRGS repository root."""
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = Path(repo_root).resolve() if repo_root else repository_root()
    return (root / path).resolve()


def load_deployment_config(
    path: Union[str, Path], repo_root: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Deployment config does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Deployment config must contain a YAML mapping: {path}")
    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(
        Path(repo_root).resolve() if repo_root else repository_root()
    )
    return cfg
