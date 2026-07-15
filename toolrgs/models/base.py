"""Base class for new structured-output ToolRGS models."""

from abc import ABC, abstractmethod

import torch.nn as nn

from toolrgs.structures import GraspModelResult


class BaseGraspModel(nn.Module, ABC):
    """New models should return :class:`GraspModelResult` from ``forward``."""

    supports_offset = False
    requires_depth = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> GraspModelResult:
        raise NotImplementedError


def model_requires_depth(model) -> bool:
    """Read the input contract through DataParallel/DDP wrappers."""
    module = getattr(model, "module", model)
    return bool(getattr(module, "requires_depth", False))
