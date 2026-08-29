"""MMEngine-style optimization components for Ascend training."""

import torch
from torch.optim.lr_scheduler import MultiStepLR

from toolrgs.registry import OPTIM_WRAPPERS, PARAM_SCHEDULERS


@OPTIM_WRAPPERS.register_module(name="npu_amp", aliases=("amp",))
class NPUAmpOptimWrapper:
    """Own zero-grad, scaled backward, clipping, and optimizer stepping."""

    def __init__(
        self, optimizer, scaler, max_norm=0.0, accumulation_steps=1
    ):
        self.optimizer = optimizer
        self.scaler = scaler
        self.max_norm = float(max_norm or 0.0)
        self.accumulation_steps = int(accumulation_steps)
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least one")
        self._pending_steps = 0

    def _step(self, model, gradient_multiplier=1.0):
        self.scaler.unscale_(self.optimizer)
        if gradient_multiplier != 1.0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_multiplier)
        if self.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self._pending_steps = 0

    def update_params(self, loss, model):
        if self._pending_steps == 0:
            self.optimizer.zero_grad()
        scaled_loss = loss / self.accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        self._pending_steps += 1
        if self._pending_steps == self.accumulation_steps:
            self._step(model)

    def flush(self, model):
        if self._pending_steps:
            multiplier = self.accumulation_steps / self._pending_steps
            self._step(model, gradient_multiplier=multiplier)


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
            "accumulation_steps": getattr(
                cfg, "gradient_accumulation_steps", 1
            ),
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
