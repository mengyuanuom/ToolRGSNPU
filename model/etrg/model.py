"""ToolRGS integration of the official ETRG-A RGB-D architecture."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from toolrgs.models import BaseGraspModel
from toolrgs.structures import GraspModelResult, GraspOutput, GraspTargets

from .bridger import Bridger_SA_RN_depth
from .clip import build_model as build_clip_model
from .layers import FPN, MultiTaskProjector, TransformerDecoder


class BCEDiceLoss(nn.Module):
    """Official ETRG instance loss: BCE plus a small soft Dice term."""

    def forward(self, prediction, target):
        bce = F.binary_cross_entropy_with_logits(prediction, target)
        probability = torch.sigmoid(prediction).flatten(1)
        target = target.flatten(1)
        intersection = (probability * target).sum(1)
        dice = 1.0 - (
            (2.0 * intersection) /
            (probability.sum(1) + target.sum(1) + 1e-5)
        ).mean()
        return bce + 0.1 * dice


def _checkpoint_state(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported ResNet-18 checkpoint payload: {path}")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }


def _build_depth_backbone(cfg):
    """Build the official ResNet-18 depth encoder with actionable failures."""
    try:
        import torchvision
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "ETRG requires torchvision matching the installed PyTorch/CUDA build."
        ) from exc

    local_weight = getattr(cfg, "depth_pretrain", None)
    pretrained = bool(getattr(cfg, "depth_backbone_pretrained", True))
    full_model_weight = getattr(cfg, "weight", None) or getattr(cfg, "resume", None)
    if full_model_weight and not local_weight:
        pretrained = False
        logger.info(
            "Skipping separate ETRG ResNet-18 download because full model "
            "weights are configured: {}",
            full_model_weight,
        )
    try:
        if local_weight:
            logger.info("Loading ETRG depth backbone weight: {}", local_weight)
            network = torchvision.models.resnet18(weights=None)
            incompatible = network.load_state_dict(
                _checkpoint_state(Path(local_weight)), strict=False
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "The configured ETRG depth_pretrain is not a compatible "
                    "torchvision ResNet-18 checkpoint: "
                    f"missing={incompatible.missing_keys[:8]}, "
                    f"unexpected={incompatible.unexpected_keys[:8]}"
                )
        elif pretrained:
            logger.info(
                "Loading torchvision ImageNet ResNet-18 for ETRG depth; "
                "torchvision may download it on first use"
            )
            weights_enum = getattr(torchvision.models, "ResNet18_Weights", None)
            if weights_enum is None:
                network = torchvision.models.resnet18(pretrained=True)
            else:
                network = torchvision.models.resnet18(weights=weights_enum.DEFAULT)
        else:
            if full_model_weight:
                logger.info(
                    "Initializing the ETRG depth encoder before full checkpoint restore"
                )
            else:
                logger.warning("ETRG depth backbone is being trained from scratch")
            try:
                network = torchvision.models.resnet18(weights=None)
            except TypeError:  # torchvision < 0.13
                network = torchvision.models.resnet18(pretrained=False)
    except Exception as exc:
        raise RuntimeError(
            "ETRG could not load the ImageNet ResNet-18 depth backbone. "
            "Check network access, set TRAIN.depth_pretrain to a local "
            "torchvision ResNet-18 checkpoint, or set "
            "TRAIN.depth_backbone_pretrained=False to train it from scratch."
        ) from exc

    # This replacement is intentional and follows the official ETRG-A code.
    network.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    nn.init.kaiming_normal_(network.conv1.weight, mode="fan_in", nonlinearity="relu")
    return nn.Sequential(*list(network.children())[:-2])


class ETRG(BaseGraspModel):
    """Parameter-efficient CLIP adapter with RGB or RGB-D auxiliary fusion."""

    requires_depth = True
    supports_offset = False

    def __init__(self, cfg):
        super().__init__()
        self.input_mode = str(getattr(cfg, "etrg_input_mode", "rgbd")).lower()
        if self.input_mode not in {"rgb", "rgbd"}:
            raise ValueError(
                "etrg_input_mode must be 'rgb' or 'rgbd', "
                f"got {self.input_mode!r}"
            )
        self.requires_depth = self.input_mode == "rgbd"
        logger.info("Loading ETRG CLIP backbone: {}", cfg.clip_pretrain)
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_clip_model(
            clip_model.state_dict(), cfg.word_len
        ).float()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.bridger = Bridger_SA_RN_depth(
            d_model=cfg.ladder_dim,
            nhead=cfg.nhead,
            fusion_stage=cfg.multi_stage,
            word_dim=cfg.word_dim,
        )
        # Keep the official attribute names so released ETRG checkpoints load
        # without a state-dict translation layer.
        self.resnet18 = _build_depth_backbone(cfg)
        self.zoom_in = nn.Conv2d(
            512, cfg.ladder_dim, kernel_size=1, stride=1, bias=False
        )
        nn.init.kaiming_normal_(
            self.zoom_in.weight, mode="fan_in", nonlinearity="relu"
        )
        self.depth_normalization = str(
            getattr(cfg, "depth_normalization", "inverse_max")
        ).lower()

        self.visual_sent_fpn = FPN(
            in_channels=cfg.fpn_in,
            out_channels=cfg.fpn_out,
            language_fuser=True,
            decoding=False,
        )
        self.decoder = TransformerDecoder(
            num_layers=cfg.num_layers,
            d_model=cfg.vis_dim,
            nhead=cfg.num_head,
            dim_ffn=cfg.dim_ffn,
            dropout=cfg.dropout,
            return_intermediate=cfg.intermediate,
        )
        self.proj = MultiTaskProjector(cfg.word_dim, cfg.vis_dim // 2, 3)
        self.loss = BCEDiceLoss()

    def _normalize_depth(self, depth):
        depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.depth_normalization in {"none", "meters"}:
            return depth
        if self.depth_normalization != "inverse_max":
            raise ValueError(
                "ETRG depth_normalization must be 'inverse_max', 'meters', or 'none', "
                f"got {self.depth_normalization!r}"
            )
        maximum = depth.flatten(1).amax(1).view(-1, 1, 1, 1).clamp_min(1e-5)
        return 1.0 - depth / maximum

    @staticmethod
    def _resize(outputs, output_size):
        return tuple(
            output
            if output.shape[-2:] == tuple(output_size)
            else F.interpolate(
                output, size=output_size, mode="bilinear", align_corners=False
            )
            for output in outputs
        )

    def forward(
        self,
        image,
        depth,
        word,
        mask=None,
        grasp_qua_mask=None,
        grasp_sin_mask=None,
        grasp_cos_mask=None,
        grasp_wid_mask=None,
        grasp_off_mask=None,
        grasp_off_weight=None,
    ):
        if self.input_mode == "rgb":
            # RGB-only calls omit depth, so the legacy positional signature is
            # shifted one place by the common ToolRGS batch contract.
            (
                word,
                mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
                grasp_off_mask,
                grasp_off_weight,
            ) = (
                depth,
                word,
                mask,
                grasp_qua_mask,
                grasp_sin_mask,
                grasp_cos_mask,
                grasp_wid_mask,
                grasp_off_mask,
            )
            auxiliary = image.float()
        else:
            if depth is None:
                raise ValueError("ETRG RGB-D mode requires aligned depth maps")
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            if depth.ndim != 4 or depth.shape[1] != 1:
                raise ValueError(
                    f"ETRG depth must have shape [B,1,H,W], got {tuple(depth.shape)}"
                )
            if depth.shape[-2:] != image.shape[-2:]:
                depth = F.interpolate(depth, size=image.shape[-2:], mode="nearest")
            auxiliary = self._normalize_depth(depth).repeat(1, 3, 1, 1)

        pad_mask = word.eq(0)
        depth_features = self.zoom_in(self.resnet18(auxiliary))
        depth_features = depth_features.flatten(2).permute(2, 0, 1)

        visual, words, sentence = self.bridger(
            image, word, self.backbone, pad_mask, depth_features
        )
        fused = self.visual_sent_fpn(visual, sentence)
        batch_size, _, height, width = fused.shape
        decoded, _ = self.decoder(fused, words, pad_mask=pad_mask)
        decoded = decoded.reshape(batch_size, -1, height, width)
        outputs = self._resize(
            self.proj(decoded, sentence), image.shape[-2:]
        )
        prediction = GraspOutput(*outputs)

        target_values = (
            mask,
            grasp_qua_mask,
            grasp_sin_mask,
            grasp_cos_mask,
            grasp_wid_mask,
        )
        targets = None
        if all(value is not None for value in target_values):
            targets = GraspTargets(*target_values)

        if not self.training:
            return GraspModelResult(predictions=prediction, targets=targets)
        if targets is None:
            raise ValueError("ETRG training requires all five dense target maps")

        instance = self.loss(outputs[0], mask)
        quality = F.smooth_l1_loss(outputs[1], grasp_qua_mask)
        sine = F.smooth_l1_loss(outputs[2], grasp_sin_mask)
        cosine = F.smooth_l1_loss(outputs[3], grasp_cos_mask)
        width = F.smooth_l1_loss(outputs[4], grasp_wid_mask)
        total = instance + quality + sine + cosine + width
        detached = GraspOutput(*(output.detach() for output in outputs))
        return GraspModelResult(
            predictions=detached,
            targets=targets,
            loss=total,
            losses={
                "m_ins": instance.detach(),
                "m_qua": quality.detach(),
                "m_sin": sine.detach(),
                "m_cos": cosine.detach(),
                "m_wid": width.detach(),
            },
        )
