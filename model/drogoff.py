"""DROG with dense grasp-center offset refinement."""

import torch
import torch.nn.functional as F

from .drog import DROG
from .native_adapter import (
    NativeDinoClipFusion,
    NativeFeaturePyramid,
    inject_clip_lora,
    inject_dino_lora,
    patch_text_alignment_loss,
)
from .native_dense import NativeDenseFusionPyramid
from .projector_builder import build_projector


class DROGOFF(DROG):
    """DINOv2/CLIP grasp model with optional Native V3/V4 adaptation."""

    supports_offset = True
    grasp_size_loss_activation = "sigmoid"

    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.use_grasp_masks:
            raise ValueError("DROGOFF requires use_grasp_masks=True")
        self.predicts_grasp_short_side = bool(
            getattr(cfg, "predict_grasp_short_side", False)
        )
        self.proj = build_projector(cfg, with_offset=True)
        self.offset_loss_weight = float(getattr(cfg, "offset_loss_weight", 1.0))
        self.short_side_loss_weight = float(
            getattr(cfg, "short_side_loss_weight", 1.0)
        )
        self.native_variant = str(getattr(cfg, "native_variant", "")).strip().lower()
        self.alignment_loss_weight = 0.0
        self.uses_query_decoder = True
        if self.native_variant:
            if self.native_variant not in {"v3", "v4"}:
                raise ValueError(
                    f"Unknown DROG-OFF native_variant: {self.native_variant!r}"
                )
            self._enable_native_variant(cfg)

    def _enable_native_variant(self, cfg):
        if list(getattr(cfg, "visual_adapter_layer", [])):
            raise ValueError(
                f"Native {self.native_variant.upper()} requires visual_adapter_layer: []"
            )
        if list(getattr(cfg, "txtual_adapter_layer", [])):
            raise ValueError(
                f"Native {self.native_variant.upper()} requires txtual_adapter_layer: []"
            )
        # Keep the four native fusion stages independent from the layers that
        # receive LoRA. Older configs omit the dedicated LoRA fields and keep
        # their original behavior through these fallbacks.
        visual_layers = tuple(int(value) for value in cfg.native_visual_layers)
        text_layers = tuple(int(value) for value in cfg.native_text_layers)
        visual_lora_layers = tuple(
            int(value)
            for value in getattr(cfg, "native_visual_lora_layers", visual_layers)
        )
        text_lora_layers = tuple(
            int(value)
            for value in getattr(cfg, "native_text_lora_layers", text_layers)
        )
        if visual_layers != tuple(sorted(set(visual_layers))):
            raise ValueError("native_visual_layers must be sorted and unique")
        if text_layers != tuple(sorted(set(text_layers))):
            raise ValueError("native_text_layers must be sorted and unique")
        if visual_lora_layers != tuple(sorted(set(visual_lora_layers))):
            raise ValueError(
                "native_visual_lora_layers must be sorted and unique"
            )
        if text_lora_layers != tuple(sorted(set(text_lora_layers))):
            raise ValueError("native_text_lora_layers must be sorted and unique")

        rank = int(getattr(cfg, "native_lora_rank", 8))
        alpha = float(getattr(cfg, "native_lora_alpha", rank * 2))
        dropout = float(getattr(cfg, "native_lora_dropout", 0.05))
        inject_dino_lora(
            self.dinov2,
            visual_lora_layers,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        inject_clip_lora(
            self.txt_backbone,
            text_lora_layers,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.fusion = NativeDinoClipFusion(cfg)

        visual_dim = int(getattr(cfg, "native_visual_dim", 768))
        pyramid_dim = int(getattr(cfg, "native_pyramid_dim", 192))
        if self.native_variant == "v3":
            self.neck = NativeFeaturePyramid(
                in_dim=visual_dim,
                pyramid_dim=pyramid_dim,
                out_dim=int(cfg.vis_dim),
                stages=len(visual_layers),
            )
        else:
            self.neck = NativeDenseFusionPyramid(
                in_dim=visual_dim,
                pyramid_dim=pyramid_dim,
                out_dim=int(cfg.vis_dim),
                text_dim=int(getattr(cfg, "word_dim", 512)),
                token_dim=int(getattr(cfg, "native_dense_token_dim", 128)),
                dropout=float(getattr(cfg, "native_dense_dropout", 0.05)),
                stages=len(visual_layers),
            )
            # Remove the inherited DETRIS-style query decoder, including its
            # parameters, from Native V4.
            self.decoder = torch.nn.Identity()
            self.uses_query_decoder = False

        self.alignment_loss_weight = float(
            getattr(cfg, "native_alignment_loss_weight", 0.2)
        )

    def _encode_features(self, img, word, pad_mask):
        if self.native_variant in {"v3", "v4"}:
            vis, text_tokens, state, alignment = self.fusion(
                img, word, self.txt_backbone, self.dinov2
            )
            auxiliary = {"alignment": alignment}
            if self.native_variant == "v4":
                features = self.neck(
                    vis,
                    text_tokens,
                    state,
                    pad_mask,
                    alignment,
                )
                return features, state, auxiliary
        else:
            vis, text_tokens, state = self.fusion(
                img, word, self.txt_backbone, self.dinov2
            )
            auxiliary = {}
        features = self.neck(vis, state)
        b, c, h, w = features.shape
        features = self.decoder(features, text_tokens, pad_mask).reshape(b, c, h, w)
        return features, state, auxiliary

    def _extra_training_losses(self, auxiliary, mask):
        alignment = auxiliary.get("alignment")
        if alignment is None or self.alignment_loss_weight <= 0.0:
            return mask.new_zeros(()), {}
        alignment_loss = patch_text_alignment_loss(alignment, mask)
        return (
            self.alignment_loss_weight * alignment_loss,
            {"m_align": alignment_loss.detach()},
        )

    def forward(self, img, word, mask=None, grasp_qua_mask=None,
                grasp_sin_mask=None, grasp_cos_mask=None,
                grasp_wid_mask=None, grasp_off_mask=None,
                grasp_off_weight=None, grasp_short_mask=None):
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        features, state, auxiliary = self._encode_features(img, word, pad_mask)

        outputs = self.proj(features, state)
        if self.predicts_grasp_short_side:
            seg, qua, sin, cos, width, short_side, offset = outputs
        else:
            seg, qua, sin, cos, width, offset = outputs
            short_side = None

        if mask is None:
            return outputs

        targets = (
            mask,
            grasp_qua_mask,
            grasp_sin_mask,
            grasp_cos_mask,
            grasp_wid_mask,
            grasp_off_mask,
        )
        if self.predicts_grasp_short_side:
            targets = (
                mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
                grasp_short_mask,
                grasp_off_mask,
            )
        if not self.training:
            return tuple(x.detach() for x in outputs), targets

        target_size = seg.shape[-2:]
        mask = F.interpolate(mask, target_size, mode="nearest").detach()
        grasp_qua_mask = F.interpolate(
            grasp_qua_mask, target_size, mode="nearest"
        ).detach()
        grasp_sin_mask = F.interpolate(
            grasp_sin_mask, target_size, mode="nearest"
        ).detach()
        grasp_cos_mask = F.interpolate(
            grasp_cos_mask, target_size, mode="nearest"
        ).detach()
        grasp_wid_mask = F.interpolate(
            grasp_wid_mask, target_size, mode="nearest"
        ).detach()

        if self.predicts_grasp_short_side:
            if grasp_short_mask is None:
                raise ValueError(
                    "Short-side DROGOFF training requires grasp short-side maps"
                )
            grasp_short_mask = F.interpolate(
                grasp_short_mask, target_size, mode="nearest"
            ).detach()

        targets = (
            mask,
            grasp_qua_mask,
            grasp_sin_mask,
            grasp_cos_mask,
            grasp_wid_mask,
            grasp_off_mask,
        )
        if self.predicts_grasp_short_side:
            targets = (
                mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
                grasp_short_mask,
                grasp_off_mask,
            )

        if grasp_off_mask is None or grasp_off_weight is None:
            raise ValueError("DROGOFF training requires offset and offset-weight maps")
        if grasp_off_mask.shape[-2:] != tuple(target_size):
            resized_weight = F.interpolate(
                grasp_off_weight, target_size, mode="bilinear", align_corners=False
            )
            resized_numerator = F.interpolate(
                grasp_off_mask * grasp_off_weight,
                target_size,
                mode="bilinear",
                align_corners=False,
            )
            grasp_off_mask = resized_numerator / resized_weight.clamp_min(1e-6)
            grasp_off_weight = resized_weight
        grasp_off_mask = grasp_off_mask.detach()
        grasp_off_weight = grasp_off_weight.detach()
        targets = (*targets[:-1], grasp_off_mask)

        seg_weight = mask * 0.5 + 1.0
        seg_loss = F.binary_cross_entropy_with_logits(seg, mask, weight=seg_weight)
        qua_loss = F.smooth_l1_loss(qua, grasp_qua_mask)
        sin_loss = F.smooth_l1_loss(sin, grasp_sin_mask)
        cos_loss = F.smooth_l1_loss(cos, grasp_cos_mask)
        width_loss = F.smooth_l1_loss(torch.sigmoid(width), grasp_wid_mask)
        short_side_loss = (
            F.smooth_l1_loss(torch.sigmoid(short_side), grasp_short_mask)
            if self.predicts_grasp_short_side
            else None
        )
        offset_error = F.smooth_l1_loss(
            offset, grasp_off_mask, reduction="none"
        )
        offset_weight = grasp_off_weight.expand_as(offset_error)
        offset_loss = (
            (offset_error * offset_weight).sum()
            / offset_weight.sum().clamp_min(1.0)
        )

        total_loss = (
            seg_loss + qua_loss + sin_loss + cos_loss + width_loss
            + self.offset_loss_weight * offset_loss
        )
        extra_loss, extra_loss_dict = self._extra_training_losses(auxiliary, mask)
        total_loss = total_loss + extra_loss
        if short_side_loss is not None:
            total_loss = total_loss + self.short_side_loss_weight * short_side_loss
        loss_dict = {
            "m_ins": seg_loss.detach(),
            "m_qua": qua_loss.detach(),
            "m_sin": sin_loss.detach(),
            "m_cos": cos_loss.detach(),
            "m_wid": width_loss.detach(),
            "m_off": offset_loss.detach(),
        }
        if short_side_loss is not None:
            loss_dict["m_short"] = short_side_loss.detach()
        loss_dict.update(extra_loss_dict)
        return tuple(x.detach() for x in outputs), targets, total_loss, loss_dict
