"""Adapter exposing historical tuple-returning models as structured models."""

import torch.nn as nn

from toolrgs.structures import GraspModelResult


class LegacyOutputAdapter(nn.Module):
    """Wrap a loaded legacy model without changing its parameters or forward args."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs) -> GraspModelResult:
        return GraspModelResult.from_legacy(self.module(*args, **kwargs))
