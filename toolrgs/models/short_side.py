"""Config-gated short-side regression shared by all dense grasp models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolrgs.structures import GraspModelResult, GraspOutput, GraspTargets


class ShortSideRegressionAdapter(nn.Module):
    """Add a learned short-side map without rewriting every legacy backbone."""

    predicts_grasp_short_side = True

    def __init__(self, base_model: nn.Module, cfg):
        super().__init__()
        self.base_model = base_model
        self.supports_offset = bool(getattr(base_model, "supports_offset", False))
        self.requires_depth = bool(getattr(base_model, "requires_depth", False))
        self.grasp_size_loss_activation = str(
            getattr(base_model, "grasp_size_loss_activation", "clamp")
        ).strip().lower()
        if self.grasp_size_loss_activation not in {"sigmoid", "clamp"}:
            raise ValueError(
                "Short-side regression requires sigmoid or clamp size activation, "
                f"got {self.grasp_size_loss_activation!r}"
            )
        self.short_side_loss_weight = float(
            getattr(cfg, "short_side_loss_weight", 1.0)
        )
        hidden_channels = int(getattr(cfg, "short_side_head_channels", 32))
        if hidden_channels <= 0:
            raise ValueError("short_side_head_channels must be positive")
        self.short_side_head = nn.Sequential(
            nn.Conv2d(8, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

        size_factor = float(getattr(cfg, "grasp_size_factor", 100.0))
        initial_height = float(getattr(cfg, "grasp_height", 20.0))
        normalized = min(max(initial_height / size_factor, 1e-4), 1.0 - 1e-4)
        final = self.short_side_head[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        initial_bias = (
            math.log(normalized / (1.0 - normalized))
            if self.grasp_size_loss_activation == "sigmoid"
            else normalized
        )
        nn.init.constant_(final.bias, initial_bias)

    def _short_for_loss(self, prediction):
        if self.grasp_size_loss_activation == "sigmoid":
            return torch.sigmoid(prediction)
        return prediction

    def forward(self, *args, grasp_short_mask=None, **kwargs):
        result = GraspModelResult.from_legacy(self.base_model(*args, **kwargs))
        prediction = result.predictions
        if not args or not torch.is_tensor(args[0]) or args[0].ndim != 4:
            raise ValueError(
                "Short-side regression requires a BCHW image as the first model input"
            )
        if args[0].shape[1] < 3:
            raise ValueError(
                "Short-side regression requires at least three image channels"
            )
        image_features = F.interpolate(
            args[0][:, :3],
            size=prediction.segmentation.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        short_side = self.short_side_head(
            torch.cat(
                [
                    prediction.segmentation,
                    image_features,
                    prediction.quality,
                    prediction.sine,
                    prediction.cosine,
                    prediction.width,
                ],
                dim=1,
            )
        )
        predictions = GraspOutput(
            segmentation=prediction.segmentation,
            quality=prediction.quality,
            sine=prediction.sine,
            cosine=prediction.cosine,
            width=prediction.width,
            offset=prediction.offset,
            short_side=short_side,
        )

        targets = result.targets
        if grasp_short_mask is not None:
            grasp_short_mask = F.interpolate(
                grasp_short_mask,
                short_side.shape[-2:],
                mode="nearest",
            ).detach()
        if targets is not None:
            targets = GraspTargets(
                segmentation=targets.segmentation,
                quality=targets.quality,
                sine=targets.sine,
                cosine=targets.cosine,
                width=targets.width,
                offset=targets.offset,
                short_side=grasp_short_mask,
            )

        if not self.training:
            return GraspModelResult(predictions=predictions.detach(), targets=targets)
        if result.loss is None:
            raise RuntimeError("Short-side training requires a base model loss")
        if grasp_short_mask is None:
            raise ValueError("VCoT short-side training requires grasp_masks['short']")
        short_loss = F.smooth_l1_loss(
            self._short_for_loss(short_side), grasp_short_mask
        )
        losses = dict(result.losses)
        losses["m_short"] = short_loss.detach()
        return GraspModelResult(
            predictions=predictions.detach(),
            targets=targets,
            loss=result.loss + self.short_side_loss_weight * short_loss,
            losses=losses,
        )
