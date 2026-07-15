"""Validate CANN/torch_npu and optionally run one ToolRGS model forward pass."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from toolrgs.runtime import (
    autocast,
    device_name,
    get_torch_npu,
    require_npu,
    set_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Local NPU index")
    parser.add_argument("--config", help="Experiment YAML to validate")
    parser.add_argument(
        "--forward", action="store_true", help="Build the configured model and run inference"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_npu()
    torch_npu = get_torch_npu()
    device = set_device(args.device)
    print(f"torch={torch.__version__}")
    print(f"torch_npu={getattr(torch_npu, '__version__', '<unknown>')}")
    print(f"device={device} name={device_name(args.device)}")
    print(f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '<unset>')}")
    print(f"HCCL available={getattr(torch.distributed, 'is_hccl_available', lambda: False)()}")

    left = torch.randn(128, 128, device=device)
    right = torch.randn(128, 128, device=device)
    with autocast():
        result = left @ right
    torch_npu.npu.synchronize()
    print(f"NPU matmul OK: shape={tuple(result.shape)} dtype={result.dtype}")

    if not args.config:
        return 0

    from toolrgs.preflight import configured_artifacts
    from utils.config import load_cfg_from_cfg_file

    cfg = load_cfg_from_cfg_file(args.config)
    cfg.npu = args.device
    cfg.gpu = args.device
    cfg.device = str(device)
    cfg.rank = 0
    missing = []
    for key, _original, path in configured_artifacts(cfg, include_checkpoints=False):
        state = "OK" if path.is_file() else "MISSING"
        print(f"{key}: {path} [{state}]")
        if state == "MISSING":
            missing.append(str(path))
    if missing:
        print("Model check stopped because pretrained files are missing.")
        return 2
    if not args.forward:
        return 0

    from model import build_model
    from toolrgs.models.base import model_requires_depth
    from toolrgs.structures import GraspModelResult
    from utils.dataset import tokenize

    model, _ = build_model(cfg)
    model = model.to(device).eval()
    size = cfg.input_size
    height, width = (int(size[0]), int(size[1])) if isinstance(size, (list, tuple)) else (int(size), int(size))
    image = torch.randn(1, 3, height, width, device=device)
    words = tokenize("grasp the tool", context_length=int(cfg.word_len), truncate=True).to(device)
    inputs = (image, words)
    if model_requires_depth(model):
        depth = torch.rand(1, 1, height, width, device=device)
        inputs = (image, depth, words)
    with torch.no_grad(), autocast():
        output = GraspModelResult.from_legacy(model(*inputs)).predictions
    torch_npu.npu.synchronize()
    print("Model forward OK:", [tuple(tensor.shape) for tensor in output.as_tuple()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
