"""Decoder-free dense visual-language fusion for DROG-OFF Native V4.

The module consumes layer-wise DINOv2 feature maps and CLIP text tokens
directly.  It does not create learned object queries and does not depend on
the DETRIS-style multimodal decoder.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return groups


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class DirectTokenPixelFusion(nn.Module):
    """Fuse valid CLIP tokens into every visual location without learned queries."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        token_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.visual_norm = nn.GroupNorm(_group_count(visual_dim), visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_query = nn.Conv2d(visual_dim, token_dim, 1, bias=False)
        self.text_key = nn.Linear(text_dim, token_dim, bias=False)
        self.text_value = nn.Linear(text_dim, visual_dim, bias=False)
        self.prior_projection = nn.Conv2d(1, visual_dim, 1, bias=False)
        self.state_film = nn.Linear(text_dim, visual_dim * 2)
        self.spatial_gate = nn.Conv2d(visual_dim * 2 + 1, visual_dim, 1)
        self.dropout = nn.Dropout(float(dropout))
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.refine = nn.Sequential(
            nn.Conv2d(
                visual_dim,
                visual_dim,
                3,
                padding=1,
                groups=visual_dim,
                bias=False,
            ),
            nn.GroupNorm(_group_count(visual_dim), visual_dim),
            nn.GELU(),
            nn.Conv2d(visual_dim, visual_dim, 1, bias=False),
        )

        nn.init.zeros_(self.state_film.weight)
        nn.init.zeros_(self.state_film.bias)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        nn.init.zeros_(self.refine[-1].weight)

    @staticmethod
    def _valid_token_mask(text_padding_mask):
        valid = ~text_padding_mask.bool()
        # Defensive fallback for malformed all-padding samples. Real CLIP input
        # always contains at least the start/end tokens.
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        return valid

    def forward(
        self,
        visual,
        text_tokens,
        state,
        text_padding_mask,
        alignment_logits,
    ):
        batch, channels, height, width = visual.shape
        normalized_visual = self.visual_norm(visual)
        queries = self.visual_query(normalized_visual).flatten(2).transpose(1, 2)
        normalized_text = self.text_norm(text_tokens.float())
        keys = self.text_key(normalized_text)
        values = self.text_value(normalized_text)

        scores = torch.einsum("bnd,bld->bnl", queries, keys)
        scores = scores / math.sqrt(float(keys.shape[-1]))
        valid_tokens = self._valid_token_mask(text_padding_mask)
        scores = scores.masked_fill(~valid_tokens.unsqueeze(1), -10000.0)
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum("bnl,blc->bnc", attention, values)
        context = self.dropout(context).transpose(1, 2).reshape(
            batch, channels, height, width
        )

        prior = torch.sigmoid(
            F.interpolate(
                alignment_logits.float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        )
        prior_features = self.prior_projection(prior)
        gate = torch.sigmoid(
            self.spatial_gate(torch.cat((normalized_visual, context, prior), dim=1))
            + self.gate_logit
        )
        fused = visual + gate * (context + prior_features)

        gamma, beta = self.state_film(state.float()).chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        fused = fused * (1.0 + gamma) + beta
        return fused + self.refine(fused)


class NativeDenseFusionPyramid(nn.Module):
    """Language-conditioned FPN used by Native V4 instead of a query decoder."""

    def __init__(
        self,
        in_dim: int = 768,
        pyramid_dim: int = 192,
        out_dim: int = 512,
        text_dim: int = 512,
        token_dim: int = 128,
        dropout: float = 0.0,
        stages: int = 4,
    ) -> None:
        super().__init__()
        if stages != 4:
            raise ValueError("NativeDenseFusionPyramid requires four feature stages")
        if token_dim <= 0:
            raise ValueError("native_dense_token_dim must be positive")
        self.lateral = nn.ModuleList(
            nn.Conv2d(in_dim, pyramid_dim, 1, bias=False) for _ in range(stages)
        )
        self.token_fusion = nn.ModuleList(
            DirectTokenPixelFusion(
                visual_dim=pyramid_dim,
                text_dim=text_dim,
                token_dim=token_dim,
                dropout=dropout,
            )
            for _ in range(stages)
        )
        self.smooth = nn.ModuleList(
            _ConvNormAct(pyramid_dim, pyramid_dim) for _ in range(stages)
        )
        self.aggregate = _ConvNormAct(
            pyramid_dim * stages,
            out_dim,
            kernel_size=1,
        )
        self.output_prior = nn.Conv2d(1, out_dim, 1, bias=False)
        self.output_gate_logit = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _resize(feature, size):
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        text_tokens,
        state,
        text_padding_mask,
        alignment_logits,
    ):
        if len(features) != 4:
            raise ValueError(f"Expected four native feature stages, got {len(features)}")
        if text_padding_mask.shape != text_tokens.shape[:2]:
            raise ValueError(
                "text_padding_mask must match the first two text-token dimensions"
            )

        base_h, base_w = features[0].shape[-2:]
        sizes = (
            (base_h * 2, base_w * 2),
            (base_h, base_w),
            (max(1, base_h // 2), max(1, base_w // 2)),
            (max(1, base_h // 4), max(1, base_w // 4)),
        )
        pyramid = [
            self._resize(lateral(feature), size)
            for lateral, feature, size in zip(self.lateral, features, sizes)
        ]

        # Semantic top-down propagation is convolutional; language enters each
        # scale directly through token-to-pixel attention.
        for index in range(2, -1, -1):
            pyramid[index] = pyramid[index] + self._resize(
                pyramid[index + 1], pyramid[index].shape[-2:]
            )
        pyramid = [
            smooth(
                fuse(
                    feature,
                    text_tokens,
                    state,
                    text_padding_mask,
                    alignment_logits,
                )
            )
            for smooth, fuse, feature in zip(
                self.smooth,
                self.token_fusion,
                pyramid,
            )
        ]

        target_size = sizes[1]
        output = self.aggregate(
            torch.cat(
                [self._resize(feature, target_size) for feature in pyramid],
                dim=1,
            )
        )
        output_prior = self.output_prior(
            torch.sigmoid(
                F.interpolate(
                    alignment_logits.float(),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        )
        return output + torch.sigmoid(self.output_gate_logit) * output_prior
