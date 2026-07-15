"""Ascend NPU runtime helpers used across training and deployment."""

from .device import (
    autocast,
    build_grad_scaler,
    build_optimizer,
    current_device,
    device_name,
    get_torch_npu,
    is_npu_available,
    move_to_device,
    require_npu,
    seed_all,
    set_device,
)

__all__ = [
    "autocast",
    "build_grad_scaler",
    "build_optimizer",
    "current_device",
    "device_name",
    "get_torch_npu",
    "is_npu_available",
    "move_to_device",
    "require_npu",
    "seed_all",
    "set_device",
]
