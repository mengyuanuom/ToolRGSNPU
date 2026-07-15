"""Preflight a ToolRGS deployment without sending robot commands."""

import argparse
import importlib.util
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deployment.config import load_deployment_config, resolve_repo_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/deployment/lab.example.yaml")
    parser.add_argument(
        "--probe-camera", action="store_true", help="Open the camera and read one frame"
    )
    parser.add_argument(
        "--build-model", action="store_true", help="Load all model weights on the configured device"
    )
    return parser.parse_args()


class Report:
    def __init__(self):
        self.failures = 0

    def ok(self, message):
        print(f"[PASS] {message}")

    def warn(self, message):
        print(f"[WARN] {message}")

    def fail(self, message):
        self.failures += 1
        print(f"[FAIL] {message}")


def require_module(report, module, label=None):
    if importlib.util.find_spec(module) is None:
        report.fail(f"Missing Python package: {label or module}")
    else:
        report.ok(f"Python package available: {label or module}")


def main() -> int:
    args = parse_args()
    report = Report()
    try:
        cfg = load_deployment_config(args.config)
    except Exception as exc:
        report.fail(str(exc))
        return 1

    require_module(report, "torch")
    require_module(report, "cv2", "opencv-python")
    require_module(report, "PyQt5")
    require_module(report, "yaml", "PyYAML")

    model_cfg = cfg["model"]
    experiment = resolve_repo_path(model_cfg["config"], cfg["_repo_root"])
    checkpoint = resolve_repo_path(model_cfg["checkpoint"], cfg["_repo_root"])
    for label, path in (("Experiment config", experiment), ("Checkpoint", checkpoint)):
        if path is not None and path.is_file():
            report.ok(f"{label}: {path}")
        else:
            report.fail(f"{label} not found: {path}")

    if experiment is not None and experiment.is_file():
        with experiment.open("r", encoding="utf-8") as stream:
            experiment_yaml = yaml.safe_load(stream) or {}
        flat = {
            key: value
            for section in experiment_yaml.values()
            if isinstance(section, dict)
            for key, value in section.items()
        }
        for key in ("clip_pretrain", "dino_pretrain", "mamba_pretrain"):
            value = flat.get(key)
            if not value or str(value).startswith(("http://", "https://")):
                continue
            path = resolve_repo_path(value, cfg["_repo_root"])
            if path is not None and path.is_file():
                report.ok(f"{key}: {path}")
            else:
                report.fail(f"{key} not found: {path}")

    backend = str(
        cfg["camera"].get("type", cfg["camera"].get("backend", "opencv"))
    ).lower()
    if backend == "realsense":
        require_module(report, "pyrealsense2")
    if backend == "gstreamer":
        require_module(report, "cv2", "OpenCV with GStreamer support")
        report.warn("Confirm cv2.getBuildInformation() reports GStreamer: YES")
    if cfg.get("detector", {}).get("enabled"):
        require_module(report, "mmdet")
        for key in ("config", "checkpoint"):
            path = resolve_repo_path(cfg["detector"].get(key), cfg["_repo_root"])
            if path is not None and path.is_file():
                report.ok(f"Detector {key}: {path}")
            else:
                report.fail(f"Detector {key} not found: {path}")
    if cfg.get("audio", {}).get("enabled"):
        require_module(report, "sounddevice")
        require_module(report, "whisper", "openai-whisper")

    robot_cfg = cfg["robot"]
    if robot_cfg.get("enabled"):
        report.warn(
            "Robot output is enabled in YAML, but this preflight intentionally does not connect or send"
        )
    else:
        report.ok("Robot output is disabled (safe dry-run state)")
    if str(robot_cfg.get("coordinate_space", "source")).lower() not in {"source", "model"}:
        report.fail("robot.coordinate_space must be source or model")
    for field in ("x", "y", "theta", "width", "depth"):
        bounds = robot_cfg.get("limits", {}).get(field)
        if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] >= bounds[1]:
            report.fail(f"robot.limits.{field} must be an increasing [minimum, maximum] pair")

    if args.probe_camera and report.failures == 0:
        from deployment.sources import build_source

        source = None
        try:
            source = build_source(cfg["camera"], cfg["_repo_root"])
            ok, frame = source.read()
            if not ok or frame is None:
                raise RuntimeError("camera returned no frame")
            report.ok(f"Camera frame: shape={frame.shape}, dtype={frame.dtype}")
        except Exception as exc:
            report.fail(f"Camera probe failed: {exc}")
        finally:
            if source is not None:
                source.close()

    if args.build_model and report.failures == 0:
        try:
            from deployment.inference import ToolRGSInference

            ToolRGSInference(cfg)
            report.ok("Model and checkpoint loaded successfully")
        except Exception as exc:
            report.fail(f"Model build failed: {exc}")

    print(f"\nPreflight completed with {report.failures} failure(s).")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
