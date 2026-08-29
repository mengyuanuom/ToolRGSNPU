"""Offset-guided feature transport for dense language-conditioned grasping.

The dense Offset V2 target points from each supervised grasp-region pixel to
its owning grasp center. This module turns that auxiliary prediction into a
feature-routing signal: every grasp branch samples context at its predicted
center and injects the transported context through a zero-initialized residual
gate. The zero gate makes a newly constructed transport model functionally
identical to the original projector before optimization starts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 32) -> int:
    groups = min(int(maximum), int(channels))
    while channels % groups:
        groups -= 1
    return groups


class OffsetGuidedFeatureTransport(nn.Module):
    """Propagate predicted grasp-center context into dense grasp features."""

    def __init__(
        self,
        channels: int,
        branches: int,
        hidden_dim: int = 64,
        max_displacement: float = 6.0,
        confidence_floor: float = 0.1,
        detach_confidence: bool = True,
    ) -> None:
        super().__init__()
        channels = int(channels)
        branches = int(branches)
        hidden_dim = max(16, int(hidden_dim))
        if channels <= 0 or branches <= 0:
            raise ValueError("channels and branches must be positive")
        if max_displacement <= 0.0:
            raise ValueError("max_displacement must be positive")
        if not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")

        self.channels = channels
        self.branches = branches
        self.max_displacement = float(max_displacement)
        self.confidence_floor = float(confidence_floor)
        self.detach_confidence = bool(detach_confidence)

        # Separate scalar gates let grasp quantities learn different amounts
        # of center context. Exact zero initialization preserves the baseline.
        self.branch_gate = nn.Parameter(torch.zeros(branches))
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )

    def sample_center_context(
        self, features: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Sample ``features`` at ``pixel + predicted_offset`` locations."""
        if features.ndim != 4:
            raise ValueError("features must have shape [B, C, H, W]")
        if offset.ndim != 4 or offset.shape[1] != 2:
            raise ValueError("offset must have shape [B, 2, H, W]")
        if features.shape[0] != offset.shape[0]:
            raise ValueError("features and offset must have the same batch size")

        batch_size, _, height, width = features.shape
        if offset.shape[-2:] != (height, width):
            # Offset V2 stores normalized vectors, so resizing the vector field
            # does not require multiplying its values by a spatial scale.
            offset = F.interpolate(
                offset,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        dtype = offset.dtype
        device = offset.device
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (
            2.0 / width
        ) - 1.0
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (
            2.0 / height
        ) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        base_grid = base_grid.expand(batch_size, -1, -1, -1)

        displacement = offset.permute(0, 2, 3, 1).clone()
        displacement[..., 0] *= 2.0 * self.max_displacement / width
        displacement[..., 1] *= 2.0 * self.max_displacement / height
        sampling_grid = base_grid + displacement
        return F.grid_sample(
            features,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def forward(
        self,
        branch_features: torch.Tensor,
        offset: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transport grasp branches while retaining their local features.

        ``branch_features`` has shape ``[B, G, C, H, W]``. The segmentation
        branch is intentionally not accepted by this module.
        """
        if branch_features.ndim != 5:
            raise ValueError(
                "branch_features must have shape [B, branches, C, H, W]"
            )
        batch_size, branches, channels, height, width = branch_features.shape
        if branches != self.branches or channels != self.channels:
            raise ValueError(
                "transport feature contract mismatch: expected "
                f"branches={self.branches}, channels={self.channels}, got "
                f"branches={branches}, channels={channels}"
            )

        # All grasp quantities describe the same physical grasp and therefore
        # share one geometric center context. Pooling before routing avoids a
        # branches-times activation expansion at OS4 while retaining separate
        # learned gates for quality, angle and size branches.
        grasp_context = branch_features.mean(dim=1)
        center_context = self.sample_center_context(grasp_context, offset)
        residual = self.refine(center_context - grasp_context)

        if confidence is None:
            spatial_confidence = branch_features.new_ones(
                (batch_size, 1, height, width)
            )
        else:
            if confidence.ndim != 4 or confidence.shape[1] != 1:
                raise ValueError("confidence must have shape [B, 1, H, W]")
            spatial_confidence = confidence
            if spatial_confidence.shape[-2:] != (height, width):
                spatial_confidence = F.interpolate(
                    spatial_confidence,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            spatial_confidence = torch.sigmoid(spatial_confidence)
            if self.detach_confidence:
                spatial_confidence = spatial_confidence.detach()
            spatial_confidence = self.confidence_floor + (
                1.0 - self.confidence_floor
            ) * spatial_confidence

        gate = torch.tanh(self.branch_gate).reshape(1, branches, 1, 1, 1)
        return (
            branch_features
            + gate
            * spatial_confidence[:, None]
            * residual[:, None]
        )
