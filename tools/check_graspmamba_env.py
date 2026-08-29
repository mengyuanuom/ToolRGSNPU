"""Check the optional GraspMamba runtime before starting a long experiment."""

import argparse
from importlib.metadata import PackageNotFoundError, version
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
    parser.add_argument(
        "--scan-backend",
        choices=("parallel", "sequential"),
        default="parallel",
    )
    parser.add_argument(
        "--no-scan-checkpoint",
        dest="scan_checkpoint",
        action="store_false",
        help="disable activation checkpointing for the portable scan",
    )
    parser.set_defaults(scan_checkpoint=True)
    args = parser.parse_args()

    import torch
    from toolrgs.runtime import device_name, get_torch_npu, require_npu, set_device

    print(f"torch: {torch.__version__}")
    require_npu()
    torch_npu = get_torch_npu()
    device = set_device(0)
    print(f"torch_npu: {getattr(torch_npu, '__version__', 'unknown')}")
    print(f"NPU: {device_name(0)}")

    from model.mamba_npu import (
        install_mamba_ssm_npu_shim,
        patch_mambavision_for_npu,
        selective_scan_fn,
    )

    install_mamba_ssm_npu_shim(
        backend=args.scan_backend,
        checkpoint_scan=args.scan_checkpoint,
    )
    try:
        from mambavision import create_model
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "MambaVision import failed. Install the Python package without its "
            "CUDA mamba-ssm dependency by running "
            "`bash tools/install_graspmamba_npu.sh`.\n"
            f"Original error: {exc}"
        )
    try:
        mambavision_version = version("mambavision")
    except PackageNotFoundError as exc:
        raise SystemExit(
            "Unable to resolve the installed MambaVision version"
        ) from exc
    if mambavision_version != "1.2.0":
        raise SystemExit(
            "Portable operator validation requires mambavision==1.2.0, "
            f"got {mambavision_version!r}"
        )
    print(f"mambavision: {mambavision_version}")

    scan_u = torch.randn(1, 4, 17, device=device, requires_grad=True)
    scan_delta = torch.randn(1, 4, 17, device=device, requires_grad=True)
    scan_A = (-torch.rand(4, 3, device=device)).requires_grad_()
    scan_B = torch.randn(1, 3, 17, device=device, requires_grad=True)
    scan_C = torch.randn(1, 3, 17, device=device, requires_grad=True)
    scan_D = torch.randn(4, device=device, requires_grad=True)
    scan_output = selective_scan_fn(
        scan_u,
        scan_delta,
        scan_A,
        scan_B,
        scan_C,
        scan_D,
        delta_softplus=True,
        backend=args.scan_backend,
        checkpoint_scan=args.scan_checkpoint,
    )
    scan_output.square().mean().backward()
    scan_inputs = (scan_u, scan_delta, scan_A, scan_B, scan_C, scan_D)
    if any(value.grad is None for value in scan_inputs) or not all(
        bool(torch.isfinite(value.grad).all().item()) for value in scan_inputs
    ):
        raise SystemExit("Portable selective scan produced invalid NPU gradients")
    torch_npu.npu.synchronize()
    print("portable selective scan NPU forward/backward: OK")
    del scan_output, scan_inputs, scan_u, scan_delta, scan_A, scan_B, scan_C, scan_D

    model = create_model("mamba_vision_T", pretrained=False, num_classes=0)
    state_keys = tuple(model.state_dict())
    report = patch_mambavision_for_npu(
        model,
        scan_backend=args.scan_backend,
        checkpoint_scan=args.scan_checkpoint,
    )
    if tuple(model.state_dict()) != state_keys:
        raise SystemExit("Portable patch unexpectedly changed MambaVision parameters")
    if report.mixers != 6 or report.attentions != 6:
        raise SystemExit(
            "Unexpected MambaVision-T block layout: "
            f"mixers={report.mixers}, attentions={report.attentions}"
        )
    print(
        "portable MambaVision patch: "
        f"mixers={report.mixers}, attentions={report.attentions}, "
        f"scan={args.scan_backend}, checkpoint={args.scan_checkpoint}"
    )
    try:
        model = model.to(device).eval()
        with torch.no_grad():
            model(torch.randn(1, 3, 224, 224, device=device))
        torch_npu.npu.synchronize()
    except Exception as exc:
        raise SystemExit(
            "MambaVision constructed but its portable NPU forward pass failed.\n"
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
