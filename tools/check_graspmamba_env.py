"""Check the optional GraspMamba runtime before starting a long experiment."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", default="pretrain/RN50.pt")
    parser.add_argument(
        "--mamba", default="pretrain/mambavision_tiny_1k.pth.tar"
    )
    args = parser.parse_args()

    import torch
    from toolrgs.runtime import device_name, get_torch_npu, require_npu, set_device

    print(f"torch: {torch.__version__}")
    require_npu()
    torch_npu = get_torch_npu()
    device = set_device(0)
    print(f"torch_npu: {getattr(torch_npu, '__version__', 'unknown')}")
    print(f"NPU: {device_name(0)}")

    try:
        import mamba_ssm
        from mambavision import create_model
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "MambaVision import failed. Its upstream selective_scan_cuda extension "
            "is not NPU compatible without an Ascend-native implementation.\n"
            f"Original error: {exc}"
        )

    print(f"mamba_ssm: {getattr(mamba_ssm, '__version__', 'unknown')}")
    model = create_model("mamba_vision_T", pretrained=False, num_classes=0)
    try:
        model = model.to(device).eval()
        with torch.no_grad():
            model(torch.randn(1, 3, 224, 224, device=device))
        torch_npu.npu.synchronize()
    except Exception as exc:
        raise SystemExit(
            "MambaVision constructed but failed its NPU forward pass. This usually "
            "means selective_scan_cuda needs an Ascend-native replacement.\n"
            f"Original error: {exc}"
        )
    channels = [80, 160, 320, 640]
    print(f"MambaVision-T NPU forward: OK, expected stage channels={channels}")
    del model

    clip_path = Path(args.clip)
    print(f"CLIP: {'OK' if clip_path.is_file() else 'MISSING'} ({clip_path.resolve()})")
    mamba_path = Path(args.mamba)
    state = "OK" if mamba_path.is_file() else "missing; first model build will download it"
    print(f"MambaVision checkpoint: {state} ({mamba_path.resolve()})")


if __name__ == "__main__":
    main()
