"""Base class for new structured-output ToolRGS models."""

from abc import ABC, abstractmethod

import torch.nn as nn

from toolrgs.structures import GraspModelResult


class BaseGraspModel(nn.Module, ABC):
    """New models should return :class:`GraspModelResult` from ``forward``."""

    supports_offset = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> GraspModelResult:
        raise NotImplementedError
