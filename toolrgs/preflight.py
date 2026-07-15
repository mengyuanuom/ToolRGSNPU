"""Fail-fast checks for local model artifacts referenced by flat configs."""

from pathlib import Path
from typing import Dict, Iterable, Tuple


PRETRAINED_KEYS = ("clip_pretrain", "dino_pretrain", "mamba_pretrain")
CHECKPOINT_KEYS = ("resume", "weight")


def _is_remote(value) -> bool:
    return str(value).startswith(("http://", "https://"))


def configured_artifacts(cfg, include_checkpoints: bool = True):
    keys: Iterable[str] = PRETRAINED_KEYS
    if include_checkpoints:
        keys = (*PRETRAINED_KEYS, *CHECKPOINT_KEYS)
    for key in keys:
        value = getattr(cfg, key, None)
        if value is None or str(value).strip() == "" or _is_remote(value):
            continue
        path = Path(str(value)).expanduser()
        resolved = path.resolve()
        yield key, str(value), resolved


def validate_required_artifacts(
    cfg, include_checkpoints: bool = True
) -> Dict[str, Path]:
    """Resolve configured files and raise one actionable error for all misses."""
    resolved: Dict[str, Path] = {}
    missing: list[Tuple[str, str, Path]] = []
    for key, original, path in configured_artifacts(cfg, include_checkpoints):
        if not path.is_file():
            missing.append((key, original, path))
        else:
            resolved[key] = path
    if not missing:
        return resolved

    details = "\n".join(
        f"  - {key}={original!r}\n    resolved to: {path}"
        for key, original, path in missing
    )
    raise FileNotFoundError(
        "Required model artifact(s) were not found:\n"
        f"{details}\n"
        f"Current working directory: {Path.cwd()}\n"
        "Update the corresponding TRAIN path in the experiment YAML, or place "
        "the file at the resolved location. Run tools/check_npu_env.py with the "
        "same config to verify the environment before training."
    )
