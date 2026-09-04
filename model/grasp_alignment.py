"""Hierarchical DINO--CLIP alignment for referring grasp detection.

The module predicts two related dense maps from frozen DINO patch features and
CLIP text features: an object grounding map and a grasp-region grounding map.
Ground-truth grasp maps are used only by the losses below and never enter the
forward feature path.
"""

import math
from typing import Dict, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class LocalTopKAffinity(nn.Module):
    """Propagate logits through a sparse local DINO patch affinity graph."""

    def __init__(self, kernel_size=3, topk=4, steps=1, blend=0.25):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("affinity kernel_size must be a positive odd number")
        neighbours = kernel_size * kernel_size
        if not 0 < int(topk) <= neighbours:
            raise ValueError("affinity topk must be in [1, kernel_size ** 2]")
        if int(steps) < 0:
            raise ValueError("affinity steps must be non-negative")
        if not 0.0 <= float(blend) <= 1.0:
            raise ValueError("affinity blend must be between zero and one")
        self.kernel_size = kernel_size
        self.topk = int(topk)
        self.steps = int(steps)
        self.blend = float(blend)

    def forward(self, features: Tensor, logits: Tensor) -> Tensor:
        if self.steps == 0 or self.blend == 0.0:
            return logits
        if features.ndim != 4 or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("affinity expects BCHW features and B1HW logits")
        if features.shape[0] != logits.shape[0] or features.shape[-2:] != logits.shape[-2:]:
            raise ValueError("affinity feature and logit grids must match")

        batch, channels, height, width = features.shape
        count = height * width
        padding = self.kernel_size // 2
        normalized = F.normalize(features.float(), dim=1, eps=1e-6)
        neighbours = F.unfold(
            normalized,
            kernel_size=self.kernel_size,
            padding=padding,
        ).reshape(batch, channels, self.kernel_size ** 2, count)
        center = normalized.flatten(2).unsqueeze(2)
        similarity = (center * neighbours).sum(dim=1)

        valid = F.unfold(
            torch.ones(
                (batch, 1, height, width),
                dtype=normalized.dtype,
                device=normalized.device,
            ),
            kernel_size=self.kernel_size,
            padding=padding,
        ).reshape(batch, self.kernel_size ** 2, count)
        similarity = similarity.masked_fill(valid < 0.5, torch.finfo(similarity.dtype).min)
        top_values, top_indices = similarity.topk(self.topk, dim=1)
        weights = F.softmax(top_values, dim=1)

        propagated = logits.float()
        for _ in range(self.steps):
            neighbour_logits = F.unfold(
                propagated,
                kernel_size=self.kernel_size,
                padding=padding,
            ).reshape(batch, self.kernel_size ** 2, count)
            selected = neighbour_logits.gather(1, top_indices)
            update = (selected * weights).sum(dim=1).reshape(batch, 1, height, width)
            propagated = propagated + self.blend * (update - propagated)
        return propagated.to(logits.dtype)


class HierarchicalDinoClipAlignment(nn.Module):
    """Align CLIP language with DINO object and grasp-region patch features."""

    def __init__(
        self,
        visual_dim=768,
        text_dim=512,
        hidden_dim=256,
        stages=3,
        text_heads=8,
        dropout=0.05,
        affinity_kernel=3,
        affinity_topk=4,
        affinity_steps=1,
        affinity_blend=0.25,
        temperature=0.07,
    ):
        super().__init__()
        if int(hidden_dim) % int(text_heads):
            raise ValueError("alignment hidden_dim must be divisible by text_heads")
        if int(stages) <= 0:
            raise ValueError("alignment stages must be positive")
        if float(temperature) <= 0.0:
            raise ValueError("alignment temperature must be positive")

        self.stages = int(stages)
        self.visual_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(int(visual_dim), int(hidden_dim), 1, bias=False),
                nn.GroupNorm(1, int(hidden_dim)),
            )
            for _ in range(self.stages)
        )
        self.state_norm = nn.LayerNorm(int(text_dim))
        self.text_norm = nn.LayerNorm(int(text_dim))
        self.object_query = nn.Sequential(
            nn.Linear(int(text_dim), int(hidden_dim), bias=False),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim), bias=False),
        )
        self.text_key = nn.Linear(int(text_dim), int(hidden_dim), bias=False)
        self.text_value = nn.Linear(int(text_dim), int(hidden_dim), bias=False)
        self.grasp_seed = nn.Parameter(torch.empty(1, 1, int(hidden_dim)))
        self.grasp_attention = nn.MultiheadAttention(
            int(hidden_dim),
            int(text_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.grasp_query = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim), bias=False),
        )
        self.grasp_text_scale_logit = nn.Parameter(torch.tensor(-2.0))
        initial_scale = math.log(1.0 / float(temperature))
        self.object_logit_scale = nn.Parameter(torch.tensor(initial_scale))
        self.grasp_logit_scale = nn.Parameter(torch.tensor(initial_scale))
        self.affinity = LocalTopKAffinity(
            kernel_size=affinity_kernel,
            topk=affinity_topk,
            steps=affinity_steps,
            blend=affinity_blend,
        )
        nn.init.normal_(self.grasp_seed, std=0.02)

    @staticmethod
    def _scaled_cosine(features: Tensor, query: Tensor, logit_scale: Tensor) -> Tensor:
        query = F.normalize(query.float(), dim=-1, eps=1e-6)
        scale = logit_scale.float().clamp(max=math.log(100.0)).exp()
        return scale * (features * query[:, :, None, None]).sum(dim=1, keepdim=True)

    def forward(
        self,
        visual_features: Sequence[Tensor],
        text_tokens: Tensor,
        sentence_state: Tensor,
        text_padding_mask: Tensor,
    ) -> Dict[str, Tensor]:
        if len(visual_features) != self.stages:
            raise ValueError(
                f"expected {self.stages} DINO stages, got {len(visual_features)}"
            )
        if text_tokens.ndim != 3 or sentence_state.ndim != 2:
            raise ValueError("alignment expects BLC text tokens and BC sentence state")

        object_query = self.object_query(self.state_norm(sentence_state.float()))
        normalized_text = self.text_norm(text_tokens.float())
        seed = object_query.unsqueeze(1) + self.grasp_seed.expand(
            object_query.shape[0], -1, -1
        )
        grasp_context = self.grasp_attention(
            seed,
            self.text_key(normalized_text),
            self.text_value(normalized_text),
            key_padding_mask=text_padding_mask.bool(),
            need_weights=False,
        )[0].squeeze(1)
        grasp_query = self.grasp_query(
            object_query + torch.sigmoid(self.grasp_text_scale_logit) * grasp_context
        )

        target_size = visual_features[-1].shape[-2:]
        object_logits = []
        grasp_deltas = []
        projected_features = []
        for projection, feature in zip(self.visual_projections, visual_features):
            projected = F.normalize(projection(feature.float()), dim=1, eps=1e-6)
            projected_features.append(projected)
            object_stage = self._scaled_cosine(
                projected, object_query, self.object_logit_scale
            )
            grasp_stage = self._scaled_cosine(
                projected, grasp_query, self.grasp_logit_scale
            )
            object_logits.append(
                F.interpolate(
                    object_stage,
                    target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            grasp_deltas.append(
                F.interpolate(
                    grasp_stage,
                    target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        object_logits = torch.stack(object_logits, dim=0).mean(dim=0)
        # The sparse grasp objective must not distort complete-object grounding.
        grasp_logits = object_logits.detach() + torch.stack(grasp_deltas, dim=0).mean(dim=0)
        alignment_features = F.normalize(
            torch.stack(
                [
                    F.interpolate(
                        feature,
                        target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    for feature in projected_features
                ],
                dim=0,
            ).mean(dim=0),
            dim=1,
            eps=1e-6,
        )
        object_logits = self.affinity(alignment_features, object_logits)
        grasp_logits = self.affinity(alignment_features, grasp_logits)
        return {
            "object_alignment": object_logits,
            "grasp_alignment": grasp_logits,
            "alignment_features": alignment_features,
            "object_query": F.normalize(object_query.float(), dim=-1, eps=1e-6),
            "grasp_query": F.normalize(grasp_query.float(), dim=-1, eps=1e-6),
        }


def _balanced_soft_bce_dice(logits: Tensor, target: Tensor) -> Tensor:
    target = target.float().clamp(0.0, 1.0)
    positive = (target > 0.05).to(target.dtype).sum()
    negative = target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pos_weight)
    probability = torch.sigmoid(logits.float())
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return 0.5 * (bce + dice)


def _gather_for_contrastive(tensor: Tensor):
    if not dist.is_available() or not dist.is_initialized():
        return tensor, 0
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor, 0
    rank = dist.get_rank()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.detach())
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0), rank * tensor.shape[0]


def region_text_contrastive_loss(
    features: Tensor,
    object_query: Tensor,
    target_mask: Tensor,
    text_ids=None,
    temperature=0.07,
) -> Tensor:
    """Symmetric multi-positive InfoNCE with optional cross-rank negatives."""

    if float(temperature) <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    mask = F.interpolate(
        target_mask.float(),
        features.shape[-2:],
        mode="nearest",
    ).detach().clamp(0.0, 1.0)
    region = (features.float() * mask).sum(dim=(2, 3)) / mask.sum(
        dim=(2, 3)
    ).clamp_min(1.0)
    region = F.normalize(region, dim=-1, eps=1e-6)
    query = F.normalize(object_query.float(), dim=-1, eps=1e-6)
    all_queries, label_offset = _gather_for_contrastive(query)
    all_regions, region_offset = _gather_for_contrastive(region)
    if label_offset != region_offset:
        raise RuntimeError("region and text contrastive ranks are inconsistent")

    scale = 1.0 / float(temperature)
    region_to_text = scale * region @ all_queries.transpose(0, 1)
    text_to_region = scale * query @ all_regions.transpose(0, 1)
    if text_ids is None:
        labels = torch.arange(
            region.shape[0], device=region.device, dtype=torch.long
        ) + label_offset
        return 0.5 * (
            F.cross_entropy(region_to_text, labels)
            + F.cross_entropy(text_to_region, labels)
        )

    if text_ids.ndim != 2 or text_ids.shape[0] != region.shape[0]:
        raise ValueError("text_ids must have shape [B, token_count]")
    local_ids = text_ids.detach().to(dtype=torch.int32)
    all_ids, id_offset = _gather_for_contrastive(local_ids)
    if id_offset != label_offset:
        raise RuntimeError("text ids and embeddings have inconsistent ranks")
    positives = (local_ids[:, None, :] == all_ids[None, :, :]).all(dim=-1)
    if not positives.any(dim=1).all():
        raise RuntimeError("every contrastive row must contain a positive pair")

    def multi_positive_loss(logits, positive_mask):
        positive_logits = logits.masked_fill(
            ~positive_mask, torch.finfo(logits.dtype).min
        )
        return (
            torch.logsumexp(logits, dim=1)
            - torch.logsumexp(positive_logits, dim=1)
        ).mean()

    return 0.5 * (
        multi_positive_loss(region_to_text, positives)
        + multi_positive_loss(text_to_region, positives)
    )


def hierarchical_alignment_losses(
    object_logits: Tensor,
    grasp_logits: Tensor,
    target_mask: Tensor,
    grasp_quality: Tensor,
    offset_weight: Tensor,
    text_ids=None,
    alignment_features=None,
    object_query=None,
    quality_mix=0.6,
    ranking_margin=0.2,
    contrastive_temperature=0.07,
) -> Dict[str, Tensor]:
    """Build object/grasp targets and compute M4 auxiliary objectives."""

    size = object_logits.shape[-2:]
    mask = F.interpolate(target_mask.float(), size, mode="nearest").detach().clamp(0.0, 1.0)
    quality = F.interpolate(
        grasp_quality.float(), size, mode="bilinear", align_corners=False
    ).detach().clamp(0.0, 1.0)
    center = F.interpolate(
        offset_weight.float(), size, mode="bilinear", align_corners=False
    ).detach().clamp(0.0, 1.0)
    mix = float(quality_mix)
    if not 0.0 <= mix <= 1.0:
        raise ValueError("quality_mix must be between zero and one")
    grasp_target = mask * (mix * quality + (1.0 - mix) * center)

    object_loss = _balanced_soft_bce_dice(object_logits, mask)
    grasp_loss = _balanced_soft_bce_dice(grasp_logits, grasp_target)
    if (alignment_features is None) != (object_query is None):
        raise ValueError(
            "alignment_features and object_query must be provided together"
        )
    contrastive_loss = (
        region_text_contrastive_loss(
            alignment_features,
            object_query,
            target_mask,
            text_ids=text_ids,
            temperature=contrastive_temperature,
        )
        if alignment_features is not None
        else object_logits.float().sum() * 0.0
    )

    flat_logits = grasp_logits.float().flatten(1)
    flat_target = grasp_target.flatten(1)
    flat_mask = mask.flatten(1)
    positive_weight = flat_target
    hard_weight = flat_mask * (1.0 - flat_target)
    background_weight = 1.0 - flat_mask
    positive_score = (flat_logits * positive_weight).sum(dim=1) / positive_weight.sum(
        dim=1
    ).clamp_min(1.0)
    hard_score = (flat_logits * hard_weight).sum(dim=1) / hard_weight.sum(dim=1).clamp_min(1.0)
    background_score = (flat_logits * background_weight).sum(dim=1) / background_weight.sum(
        dim=1
    ).clamp_min(1.0)
    valid_positive = positive_weight.sum(dim=1) > 0.0
    valid_hard = hard_weight.sum(dim=1) > 0.0
    valid_background = background_weight.sum(dim=1) > 0.0
    ranking_terms = []
    if (valid_positive & valid_hard).any():
        valid = valid_positive & valid_hard
        ranking_terms.append(
            F.softplus(float(ranking_margin) - positive_score[valid] + hard_score[valid]).mean()
        )
    if (valid_positive & valid_background).any():
        valid = valid_positive & valid_background
        ranking_terms.append(
            0.5
            * F.softplus(
                float(ranking_margin) - positive_score[valid] + background_score[valid]
            ).mean()
        )
    ranking_loss = (
        torch.stack(ranking_terms).sum()
        if ranking_terms
        else grasp_logits.float().sum() * 0.0
    )
    object_probability = torch.sigmoid(object_logits.float())
    grasp_probability = torch.sigmoid(grasp_logits.float())
    subset_loss = F.relu(grasp_probability - object_probability).mean()
    return {
        "object": object_loss,
        "contrastive": contrastive_loss,
        "grasp": grasp_loss,
        "ranking": ranking_loss,
        "subset": subset_loss,
        "target": grasp_target,
    }
