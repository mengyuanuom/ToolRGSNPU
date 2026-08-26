"""DrogOff projector wrapper with offset-guided grasp feature transport."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import OffsetMultiTaskProjector
from .offset_transport import OffsetGuidedFeatureTransport


class OffsetTransportProjector(nn.Module):
    """Wrap an existing offset projector without reinitializing baseline heads.

    Keeping the original ``base`` and ``offset`` modules preserves their state
    dictionary keys. An Offset V2 DrogOff checkpoint can therefore be loaded as
    initial weight with only the new ``transport`` keys reported missing.
    """

    def __init__(
        self,
        projector: OffsetMultiTaskProjector,
        hidden_dim: int = 64,
        max_displacement: float = 6.0,
        confidence_floor: float = 0.1,
        detach_confidence: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(projector, OffsetMultiTaskProjector):
            raise TypeError(
                "OffsetTransportProjector requires OffsetMultiTaskProjector"
            )
        self.base = projector.base
        self.offset = projector.offset
        self.with_short_side = projector.with_short_side
        self.transport = OffsetGuidedFeatureTransport(
            channels=self.base.in_dim,
            branches=self.base.num_outputs - 1,
            hidden_dim=hidden_dim,
            max_displacement=max_displacement,
            confidence_floor=confidence_floor,
            detach_confidence=detach_confidence,
        )

    def _project_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base.vis(x)
        batch_size, total_channels, height, width = features.shape
        branches = self.base.num_outputs
        if total_channels % branches:
            raise RuntimeError(
                f"Projector channels {total_channels} are not divisible by {branches}"
            )
        channels = total_channels // branches
        if channels != self.base.in_dim:
            raise RuntimeError(
                f"Expected {self.base.in_dim} branch channels, got {channels}"
            )
        return features.reshape(
            batch_size, branches, channels, height, width
        )

    def _dynamic_parameters(self, word: torch.Tensor):
        batch_size = word.shape[0]
        dynamic = self.base.txt(word)
        weight = dynamic[:, :-1].reshape(
            batch_size,
            self.base.in_dim,
            self.base.kernel_size,
            self.base.kernel_size,
        )
        return weight, dynamic[:, -1]

    def _predict_branch(
        self, features: torch.Tensor, word: torch.Tensor
    ) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        weight, bias = self._dynamic_parameters(word)
        return F.conv2d(
            features.reshape(1, batch_size * channels, height, width),
            weight.contiguous(),
            bias=bias.contiguous(),
            padding=self.base.kernel_size // 2,
            groups=batch_size,
        ).reshape(batch_size, 1, height, width)

    def _predict_all(
        self, features: torch.Tensor, word: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        batch_size, branches, channels, height, width = features.shape
        if branches != self.base.num_outputs or channels != self.base.in_dim:
            raise RuntimeError(
                "Projector feature contract mismatch: expected "
                f"K={self.base.num_outputs}, C={self.base.in_dim}, got "
                f"K={branches}, C={channels}"
            )
        weight, bias = self._dynamic_parameters(word)
        grouped_weight = (
            weight[:, None]
            .expand(-1, branches, -1, -1, -1)
            .reshape(
                batch_size * branches,
                channels,
                self.base.kernel_size,
                self.base.kernel_size,
            )
            .contiguous()
        )
        grouped_bias = (
            bias[:, None]
            .expand(-1, branches)
            .reshape(batch_size * branches)
            .contiguous()
        )
        output = F.conv2d(
            features.reshape(
                1, batch_size * branches * channels, height, width
            ),
            grouped_weight,
            bias=grouped_bias,
            padding=self.base.kernel_size // 2,
            groups=batch_size * branches,
        ).reshape(batch_size, branches, height, width)
        return tuple(
            torch.index_select(
                output,
                1,
                torch.arange(index, index + 1, device=output.device),
            )
            for index in range(branches)
        )

    def forward(self, x: torch.Tensor, word: torch.Tensor):
        offset = self.offset(x)
        branch_features = self._project_features(x)

        # Branch order: segmentation, quality, sin, cos, width, [short].
        # Only grasp branches are routed. The segmentation feature and output
        # stay exactly on the historical DrogOff path.
        coarse_quality = self._predict_branch(branch_features[:, 1], word)
        grasp_features = self.transport(
            branch_features[:, 1:], offset, coarse_quality
        )
        routed_features = torch.cat(
            (branch_features[:, :1], grasp_features), dim=1
        )
        return (*self._predict_all(routed_features, word), offset)
