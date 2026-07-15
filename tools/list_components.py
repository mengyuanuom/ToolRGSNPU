"""List registered ToolRGS components and their Python implementations."""

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolrgs.registry import (  # noqa: E402
    AUDIO_INPUTS,
    CAMERAS,
    DATASETS,
    DETECTORS,
    HOOKS,
    LOOPS,
    LOSSES,
    METRICS,
    MODELS,
    OPTIM_WRAPPERS,
    PARAM_SCHEDULERS,
    POSTPROCESSORS,
    ROBOT_CLIENTS,
    RUNNERS,
    TRANSFORMS,
)


REGISTRIES = {
    "models": MODELS,
    "datasets": DATASETS,
    "transforms": TRANSFORMS,
    "losses": LOSSES,
    "metrics": METRICS,
    "postprocessors": POSTPROCESSORS,
    "loops": LOOPS,
    "hooks": HOOKS,
    "runners": RUNNERS,
    "optim_wrappers": OPTIM_WRAPPERS,
    "param_schedulers": PARAM_SCHEDULERS,
    "cameras": CAMERAS,
    "robot_clients": ROBOT_CLIENTS,
    "detectors": DETECTORS,
    "audio_inputs": AUDIO_INPUTS,
}

POPULATORS = {
    "models": ("model",),
    "datasets": ("toolrgs.datasets",),
    "metrics": ("toolrgs.evaluation",),
    "postprocessors": ("toolrgs.evaluation",),
    "loops": ("toolrgs.engine.loops", "toolrgs.engine.val_loop"),
    "hooks": ("toolrgs.engine.hooks",),
    "runners": ("toolrgs.engine.runner",),
    "optim_wrappers": ("toolrgs.engine.optim",),
    "param_schedulers": ("toolrgs.engine.optim",),
    "cameras": ("deployment.sources",),
    "robot_clients": ("deployment.robot",),
    "detectors": ("deployment.detector",),
    "audio_inputs": ("deployment.audio",),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=sorted(REGISTRIES), help="Show one group only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = [args.group] if args.group else REGISTRIES
    failures = 0
    for group in names:
        registry = REGISTRIES[group]
        try:
            for module_name in POPULATORS.get(group, ()):
                importlib.import_module(module_name)
        except Exception as exc:
            failures += 1
            print(f"[{group}] unavailable: {exc}")
            continue
        print(f"[{group}] ({len(registry)})")
        if not len(registry):
            print("  <no components migrated yet>")
            continue
        for name in sorted(registry.keys()):
            component = registry.require(name)
            module = getattr(component, "__module__", "<unknown>")
            qualname = getattr(component, "__qualname__", repr(component))
            print(f"  {name:<20} {module}.{qualname}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
