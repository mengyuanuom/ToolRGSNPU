"""Opt-in sigmoid/masked regression; legacy checkpoints retain legacy losses."""
import torch
import torch.nn.functional as F


def configure(model, cfg):
    profile = getattr(cfg, "grasp_loss_profile", "legacy")
    if profile not in {"legacy", "sigmoid_masked"}:
        raise ValueError(f"Unknown grasp_loss_profile: {profile}")
    model.sigmoid_masked = profile == "sigmoid_masked"
    model.quality_positive_threshold = float(getattr(cfg, "quality_positive_threshold", 0.05))
    model.geometry_mask_threshold = float(getattr(cfg, "geometry_mask_threshold", 1e-6))
    for value in (model.quality_positive_threshold, model.geometry_mask_threshold):
        if not 0 <= value <= 1:
            raise ValueError("Grasp mask thresholds must be in [0, 1]")
    if model.sigmoid_masked:
        if getattr(cfg, "grasp_size_activation", "auto") not in {"auto", "sigmoid"}:
            raise ValueError("sigmoid_masked requires sigmoid size decoding")
        if getattr(cfg, "grasp_quality_activation", "sigmoid") != "sigmoid":
            raise ValueError("sigmoid_masked requires sigmoid quality decoding")
        model.grasp_quality_activation = "sigmoid"
        model.grasp_size_loss_activation = "sigmoid"


def quality_loss(logits, target, threshold=0.05):
    error = F.smooth_l1_loss(logits.sigmoid(), target, reduction="none")
    positive = (target > threshold).to(error.dtype)
    negative = 1 - positive
    pos = (error * positive).sum() / positive.sum().clamp_min(1)
    neg = (error * negative).sum() / negative.sum().clamp_min(1)
    return torch.where(positive.sum() > 0, 0.5 * (pos + neg), neg)


def geometry_loss(prediction, target, mask):
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    weight = mask.to(error.dtype).expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1)
