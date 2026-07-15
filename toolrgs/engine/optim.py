"""MMEngine-style optimization components for Ascend training."""

import torch
from torch.optim.lr_scheduler import MultiStepLR

from toolrgs.registry import OPTIM_WRAPPERS, PARAM_SCHEDULERS


@OPTIM_WRAPPERS.register_module(name="npu_amp", aliases=("amp",))
class NPUAmpOptimWrapper:
    """Own zero-grad, scaled backward, clipping, and optimizer stepping."""

    def __init__(self, optimizer, scaler, max_norm=0.0):
        self.optimizer = optimizer
        self.scaler = scaler
        self.max_norm = float(max_norm or 0.0)

    def update_params(self, loss, model):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        if self.max_norm:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()


PARAM_SCHEDULERS.register_module(
    MultiStepLR,
    name="multi_step",
    aliases=("multisteplr", "multi_step_lr"),
)


def build_optim_wrapper(cfg, optimizer, scaler):
    wrapper_cfg = getattr(cfg, "optim_wrapper", None) or {"type": "npu_amp"}
    if isinstance(wrapper_cfg, str):
        wrapper_cfg = {"type": wrapper_cfg}
    return OPTIM_WRAPPERS.build(
        wrapper_cfg,
        default_args={
            "optimizer": optimizer,
            "scaler": scaler,
            "max_norm": getattr(cfg, "max_norm", 0.0),
        },
    )


def build_param_scheduler(cfg, optimizer):
    scheduler_cfg = getattr(cfg, "param_scheduler", None) or {
        "type": "multi_step",
        "milestones": list(cfg.milestones),
        "gamma": float(cfg.lr_decay),
    }
    if isinstance(scheduler_cfg, str):
        scheduler_cfg = {"type": scheduler_cfg}
    return PARAM_SCHEDULERS.build(
        scheduler_cfg,
        default_args={"optimizer": optimizer},
    )
