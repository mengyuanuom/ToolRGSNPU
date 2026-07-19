"""Explicit PyTorch Ascend device integration.

The project intentionally avoids ``torch_npu.contrib.transfer_to_npu``.  That
module monkey-patches CUDA APIs globally, which makes device mistakes difficult
to diagnose.  All accelerator-specific behavior is kept in this module instead.
"""

from contextlib import nullcontext
from typing import Any, Optional

import torch


try:
    import torch_npu  # type: ignore
except Exception as exc:  # CANN library errors are also useful to preserve.
    torch_npu = None
    _TORCH_NPU_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _TORCH_NPU_IMPORT_ERROR = None


def get_torch_npu():
    """Return the imported adapter or raise an actionable environment error."""
    if torch_npu is None:
        message = (
            "torch_npu is unavailable. Install the torch/torch_npu pair matching "
            "the server's CANN version and source the CANN set_env.sh first."
        )
        raise RuntimeError(message) from _TORCH_NPU_IMPORT_ERROR
    return torch_npu


def is_npu_available() -> bool:
    if torch_npu is None:
        return False
    try:
        return bool(torch_npu.npu.is_available())
    except Exception:
        return False


def require_npu() -> None:
    adapter = get_torch_npu()
    if not adapter.npu.is_available():
        raise RuntimeError(
            "torch_npu imported successfully, but no Ascend NPU is available. "
            "Check npu-smi info, ASCEND_RT_VISIBLE_DEVICES, and the CANN environment."
        )


def set_device(index: int = 0) -> torch.device:
    require_npu()
    index = int(index)
    torch_npu.npu.set_device(f"npu:{index}")
    return torch.device(f"npu:{index}")


def current_device(index: Optional[int] = None) -> torch.device:
    require_npu()
    if index is None:
        index = int(torch_npu.npu.current_device())
    return torch.device(f"npu:{int(index)}")


def device_name(index: int = 0) -> str:
    require_npu()
    return str(torch_npu.npu.get_device_name(int(index)))


def move_to_device(value: Any, device: torch.device, non_blocking: bool = True):
    """Move tensors recursively while leaving metadata and NumPy values intact."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: move_to_device(item, device, non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device, non_blocking) for item in value]
    return value


def autocast(enabled: bool = False):
    if not enabled:
        return nullcontext()
    adapter = get_torch_npu()
    return adapter.npu.amp.autocast(enabled=True)


def build_grad_scaler(enabled: bool = False):
    adapter = get_torch_npu()
    return adapter.npu.amp.GradScaler(enabled=bool(enabled))


def build_optimizer(parameters, cfg):
    """Build standard Adam or the optional Ascend fused Adam implementation."""
    optimizer_name = str(getattr(cfg, "optimizer", "adam")).lower()
    kwargs = {
        "lr": float(cfg.base_lr),
        "weight_decay": float(cfg.weight_decay),
    }
    if optimizer_name in {"npu_fused_adam", "fused_adam"}:
        adapter = get_torch_npu()
        fused_adam = getattr(getattr(adapter, "optim", None), "NpuFusedAdam", None)
        if fused_adam is None:
            raise RuntimeError(
                "This torch_npu build does not expose torch_npu.optim.NpuFusedAdam; "
                "set TRAIN.optimizer to 'adam'."
            )
        return fused_adam(parameters, **kwargs)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    raise ValueError("TRAIN.optimizer must be 'adam' or 'npu_fused_adam'")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch_npu is not None:
        torch_npu.npu.manual_seed_all(seed)
