"""Native DINOv2/CLIP adaptation blocks for DROG-OFF Native V3.

These components intentionally do not depend on DETRIS' DenseAligner.  They
keep the pretrained encoders frozen, add parameter-efficient LoRA updates,
perform padding-aware layer-wise visual-language fusion, and build a real
spatial feature pyramid from ViT features.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Add a trainable low-rank update to an existing frozen linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


def _replace_linear(module: nn.Module, name: str, rank: int, alpha: float, dropout: float):
    linear = getattr(module, name, None)
    if linear is None:
        return
    if isinstance(linear, LoRALinear):
        return
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"Expected {name} to be nn.Linear, got {type(linear).__name__}")
    setattr(module, name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))


def inject_dino_lora(dino: nn.Module, layers: Iterable[int], rank=8, alpha=16.0, dropout=0.0):
    """Inject LoRA into selected DINOv2 attention projections."""

    for index in sorted(set(int(value) for value in layers)):
        block = dino.blocks[index]
        _replace_linear(block.attn, "qkv", rank, alpha, dropout)
        _replace_linear(block.attn, "proj", rank, alpha, dropout)


def inject_clip_lora(clip_model: nn.Module, layers: Iterable[int], rank=8, alpha=16.0, dropout=0.0):
    """Inject LoRA into selected CLIP text-transformer FFN projections."""

    for index in sorted(set(int(value) for value in layers)):
        block = clip_model.transformer.resblocks[index]
        _replace_linear(block.mlp, "c_fc", rank, alpha, dropout)
        _replace_linear(block.mlp, "c_proj", rank, alpha, dropout)


class LowRankTokenAdapter(nn.Module):
    """Transformer-native token adapter using LayerNorm and a gated bottleneck."""

    def __init__(self, dim: int, bottleneck: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens):
        residual = self.up(self.dropout(self.act(self.down(self.norm(tokens)))))
        return tokens + torch.sigmoid(self.gate_logit) * residual


class PaddingAwareCrossModalAdapter(nn.Module):
    """Update visual tokens from text tokens while ignoring padded positions."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("Cross-modal hidden_dim must be divisible by num_heads")
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.query = nn.Linear(visual_dim, hidden_dim, bias=False)
        self.key = nn.Linear(text_dim, hidden_dim, bias=False)
        self.value = nn.Linear(text_dim, hidden_dim, bias=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, visual_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        # A near-zero gate preserves pretrained DINO features while allowing all
        # adapter parameters to receive gradients from the first update.
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, visual_tokens, text_tokens, text_padding_mask=None):
        query = self.query(self.visual_norm(visual_tokens))
        normalized_text = self.text_norm(text_tokens)
        key = self.key(normalized_text)
        value = self.value(normalized_text)
        update = self.attention(
            query,
            key,
            value,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )[0]
        update = self.output(self.dropout(update))
        return visual_tokens + torch.sigmoid(self.gate_logit) * update


class NativeDinoClipFusion(nn.Module):
    """Run CLIP and DINO layer-by-layer and align corresponding stages."""

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, cfg, visual_dim=768, text_dim=512) -> None:
        super().__init__()
        self.visual_layers = tuple(int(v) for v in cfg.native_visual_layers)
        self.text_layers = tuple(int(v) for v in cfg.native_text_layers)
        if len(self.visual_layers) != len(self.text_layers):
            raise ValueError("native_visual_layers and native_text_layers must have equal length")
        if not self.visual_layers:
            raise ValueError("Native V3 requires at least one aligned layer")
        self.input_is_clip_normalized = bool(
            getattr(cfg, "native_input_is_clip_normalized", True)
        )
        bottleneck = int(getattr(cfg, "native_text_adapter_dim", 64))
        hidden_dim = int(getattr(cfg, "native_cross_dim", 256))
        num_heads = int(getattr(cfg, "native_cross_heads", 8))
        dropout = float(getattr(cfg, "native_adapter_dropout", 0.05))
        self.text_adapters = nn.ModuleList(
            LowRankTokenAdapter(text_dim, bottleneck, dropout) for _ in self.text_layers
        )
        self.visual_adapters = nn.ModuleList(
            PaddingAwareCrossModalAdapter(
                visual_dim, text_dim, hidden_dim, num_heads, dropout
            )
            for _ in self.visual_layers
        )
        alignment_dim = int(getattr(cfg, "native_alignment_dim", 256))
        self.patch_projection = nn.Linear(visual_dim, alignment_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, alignment_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.register_buffer(
            "clip_mean", torch.tensor(self.CLIP_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "clip_std", torch.tensor(self.CLIP_STD).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def _dino_input(self, image):
        image = image.float()
        if self.input_is_clip_normalized:
            image = image * self.clip_std + self.clip_mean
        return (image - self.imagenet_mean) / self.imagenet_std

    def _encode_text(self, text, clip_model, padding_mask):
        tokens = clip_model.token_embedding(text).type(clip_model.dtype)
        tokens = tokens + clip_model.positional_embedding[: tokens.shape[1]].type(
            clip_model.dtype
        )
        tokens = tokens.permute(1, 0, 2)
        stage_tokens = []
        stage_lookup = {layer: index for index, layer in enumerate(self.text_layers)}
        for index, block in enumerate(clip_model.transformer.resblocks):
            tokens = block(tokens)
            if index in stage_lookup:
                stage_index = stage_lookup[index]
                adapted = self.text_adapters[stage_index](tokens.permute(1, 0, 2))
                # Padding positions must not leak into later native adapters.
                adapted = adapted.masked_fill(padding_mask.unsqueeze(-1), 0.0)
                tokens = adapted.permute(1, 0, 2)
                stage_tokens.append(adapted)
        tokens = clip_model.ln_final(tokens.permute(1, 0, 2)).type(clip_model.dtype)
        state = tokens[
            torch.arange(tokens.shape[0], device=tokens.device), text.argmax(dim=-1)
        ] @ clip_model.text_projection
        return tokens, state, stage_tokens

    def forward(self, image, text, clip_model, dino):
        padding_mask = text.eq(0)
        text_tokens, state, text_stages = self._encode_text(
            text, clip_model, padding_mask
        )
        if len(text_stages) != len(self.visual_layers):
            raise RuntimeError("CLIP did not emit every configured native text stage")

        dino_tokens = dino.prepare_tokens_with_masks(self._dino_input(image))
        visual_lookup = {layer: index for index, layer in enumerate(self.visual_layers)}
        feature_maps = []
        final_patch_tokens = None
        grid_h = image.shape[-2] // int(dino.patch_size)
        grid_w = image.shape[-1] // int(dino.patch_size)
        special_tokens = 1 + int(dino.num_register_tokens)

        for index, block in enumerate(dino.blocks):
            dino_tokens = block(dino_tokens)
            if index in visual_lookup:
                stage_index = visual_lookup[index]
                dino_tokens = self.visual_adapters[stage_index](
                    dino_tokens,
                    text_stages[stage_index],
                    text_padding_mask=padding_mask,
                )
                normalized = dino.norm(dino_tokens)
                patches = normalized[:, special_tokens:]
                if patches.shape[1] != grid_h * grid_w:
                    raise RuntimeError(
                        "DINO patch-token count does not match the input grid: "
                        f"{patches.shape[1]} vs {grid_h}x{grid_w}"
                    )
                feature_maps.append(
                    patches.reshape(patches.shape[0], grid_h, grid_w, patches.shape[-1])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                final_patch_tokens = patches

        if len(feature_maps) != len(self.visual_layers):
            raise RuntimeError("DINO did not emit every configured native visual stage")
        patch_embedding = F.normalize(self.patch_projection(final_patch_tokens), dim=-1)
        text_embedding = F.normalize(self.text_projection(state.float()), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        alignment = torch.einsum("bnc,bc->bn", patch_embedding, text_embedding) * scale
        alignment = alignment.reshape(image.shape[0], 1, grid_h, grid_w)
        return feature_maps, text_tokens, state, alignment


def _norm_conv(in_channels: int, out_channels: int, kernel_size=3, stride=1):
    padding = kernel_size // 2
    groups = min(32, out_channels)
    while out_channels % groups:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


class NativeFeaturePyramid(nn.Module):
    """Reassemble same-grid ViT stages into a true 64/32/16/8 pyramid."""

    def __init__(self, in_dim=768, pyramid_dim=192, out_dim=512, stages=4) -> None:
        super().__init__()
        if stages != 4:
            raise ValueError("NativeFeaturePyramid currently requires four stages")
        self.lateral = nn.ModuleList(
            nn.Conv2d(in_dim, pyramid_dim, 1, bias=False) for _ in range(stages)
        )
        self.smooth = nn.ModuleList(
            _norm_conv(pyramid_dim, pyramid_dim) for _ in range(stages)
        )
        self.aggregate = _norm_conv(pyramid_dim * stages, out_dim, kernel_size=1)
        self.film = nn.Linear(512, out_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    @staticmethod
    def _resize(feature, size):
        return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)

    def forward(self, features: Sequence[torch.Tensor], state):
        if len(features) != 4:
            raise ValueError(f"Expected four native feature stages, got {len(features)}")
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
        for index in range(2, -1, -1):
            pyramid[index] = pyramid[index] + self._resize(
                pyramid[index + 1], pyramid[index].shape[-2:]
            )
        pyramid = [smooth(feature) for smooth, feature in zip(self.smooth, pyramid)]
        target_size = sizes[1]
        fused = self.aggregate(
            torch.cat([self._resize(feature, target_size) for feature in pyramid], dim=1)
        )
        gamma, beta = self.film(state.float()).chunk(2, dim=-1)
        return fused * (1.0 + torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)


def patch_text_alignment_loss(logits, target_mask):
    """Balanced dense alignment objective for DINO patches and CLIP text."""

    target = F.interpolate(target_mask.float(), logits.shape[-2:], mode="nearest")
    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * (bce + dice)
