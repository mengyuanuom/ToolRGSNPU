"""Measure segmentation/grasp gradient cosine similarity in DROG adapters.

This is a read-only diagnostic: it loads a trained checkpoint, evaluates a
small number of batches with gradients enabled, and never steps an optimizer.
"""

import argparse
import json
import math
import re
from collections import defaultdict

import torch
import torch.nn.functional as F

import utils.config as config
from toolrgs.engine.runner import build_runner
from toolrgs.runtime import move_to_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npu", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output")
    return parser.parse_args()


def adapter_group(name):
    if "adapter" not in name.lower():
        return None
    if name.startswith("txt_backbone."):
        match = re.search(r"resblocks\.(\d+)\.", name)
        return f"text_adapter.layer_{match.group(1)}" if match else "text_adapter.other"
    if name.startswith("dinov2."):
        match = re.search(r"blocks\.(\d+)\.", name)
        return f"visual_adapter.layer_{match.group(1)}" if match else "visual_adapter.other"
    return "adapter.other"


def live_losses(model, batch, device):
    image = move_to_device(batch["img"], device)
    word = move_to_device(batch["word_vec"], device)
    mask = move_to_device(batch["mask"], device).unsqueeze(1)
    grasp_masks = batch["grasp_masks"]
    quality = move_to_device(grasp_masks["qua"], device).unsqueeze(1)
    sine = move_to_device(grasp_masks["sin"], device).unsqueeze(1)
    cosine = move_to_device(grasp_masks["cos"], device).unsqueeze(1)
    width = move_to_device(grasp_masks["wid"], device).unsqueeze(1)

    pad_mask = word.eq(0)
    visual, text, state = model.fusion(
        image, word, model.txt_backbone, model.dinov2
    )
    fused = model.neck(visual, state)
    batch_size, channels, height, width_px = fused.shape
    fused = model.decoder(fused, text, pad_mask).reshape(
        batch_size, channels, height, width_px
    )
    outputs = model.proj(fused, state)
    pred, quality_pred, sine_pred, cosine_pred, width_pred = outputs[:5]

    if pred.shape[-2:] != mask.shape[-2:]:
        mask = F.interpolate(mask, pred.shape[-2:], mode="nearest").detach()
    targets = (quality, sine, cosine, width)
    predictions = (quality_pred, sine_pred, cosine_pred, width_pred)
    resized = []
    for target, prediction in zip(targets, predictions):
        if target.shape[-2:] != prediction.shape[-2:]:
            target = F.interpolate(
                target, prediction.shape[-2:], mode="nearest"
            ).detach()
        resized.append(target)
    quality, sine, cosine, width = resized

    seg_weight = mask * 0.5 + 1.0
    seg_loss = F.binary_cross_entropy_with_logits(pred, mask, weight=seg_weight)
    grasp_parts = {
        "quality": F.smooth_l1_loss(quality_pred, quality),
        "sine": F.smooth_l1_loss(sine_pred, sine),
        "cosine": F.smooth_l1_loss(cosine_pred, cosine),
        "width": F.smooth_l1_loss(torch.sigmoid(width_pred), width),
    }
    grasp_loss = sum(grasp_parts.values())
    return seg_loss, grasp_loss, grasp_parts


def grouped_cosines(params, groups, seg_grads, grasp_grads):
    accum = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for group, seg_grad, grasp_grad in zip(groups, seg_grads, grasp_grads):
        if seg_grad is None or grasp_grad is None:
            continue
        seg_grad = seg_grad.detach().float()
        grasp_grad = grasp_grad.detach().float()
        values = accum[group]
        values[0] += torch.sum(seg_grad * grasp_grad).item()
        values[1] += torch.sum(seg_grad.square()).item()
        values[2] += torch.sum(grasp_grad.square()).item()
        values[3] += seg_grad.numel()

    result = {}
    for group, (dot, seg_sq, grasp_sq, count) in accum.items():
        denominator = math.sqrt(seg_sq * grasp_sq)
        result[group] = {
            "cosine": dot / denominator if denominator > 0 else None,
            "dot": dot,
            "seg_norm": math.sqrt(seg_sq),
            "grasp_norm": math.sqrt(grasp_sq),
            "numel": count,
        }
    return result


def natural_group_key(group):
    match = re.search(r"layer_(\d+)$", group)
    return (group.split(".", 1)[0], int(match.group(1)) if match else 10_000)


def main():
    args = parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.npu = args.npu
    cfg.batch_size = args.batch_size
    cfg.batch_size_val = args.batch_size
    cfg.workers = 0
    cfg.workers_val = 0
    cfg.weight = args.checkpoint
    cfg.resume = None
    cfg.exp_name = "gradient_conflict_diagnostic"

    runner = build_runner(cfg).setup()
    model = getattr(runner.model, "module", runner.model)
    # Eval mode avoids changing BatchNorm buffers and removes dropout noise,
    # while autograd remains enabled for local objective-direction analysis.
    model.eval()

    named_params = [
        (name, parameter, adapter_group(name))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and adapter_group(name) is not None
    ]
    if not named_params:
        raise RuntimeError("No trainable adapter parameters were found")
    params = [item[1] for item in named_params]
    groups = [item[2] for item in named_params]
    group_parameters = defaultdict(int)
    for _, parameter, group in named_params:
        group_parameters[group] += parameter.numel()

    dataloader = runner.train_loader if args.split == "train" else runner.val_loader
    observations = defaultdict(list)
    losses = []
    batches = 0
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= args.num_batches:
            break
        model.zero_grad(set_to_none=True)
        seg_loss, grasp_loss, grasp_parts = live_losses(model, batch, runner.device)
        seg_grads = torch.autograd.grad(
            seg_loss, params, retain_graph=True, allow_unused=True
        )
        grasp_grads = torch.autograd.grad(
            grasp_loss, params, allow_unused=True
        )
        per_group = grouped_cosines(params, groups, seg_grads, grasp_grads)
        for group, metrics in per_group.items():
            if metrics["cosine"] is not None:
                observations[group].append(metrics["cosine"])

        all_groups = ["all_adapters"] * len(groups)
        global_metrics = grouped_cosines(
            params, all_groups, seg_grads, grasp_grads
        )["all_adapters"]
        observations["all_adapters"].append(global_metrics["cosine"])
        losses.append(
            {
                "segmentation": seg_loss.item(),
                "grasp": grasp_loss.item(),
                **{name: value.item() for name, value in grasp_parts.items()},
            }
        )
        batches += 1
        print(
            f"batch={batches} seg={seg_loss.item():.6f} "
            f"grasp={grasp_loss.item():.6f} "
            f"adapter_cos={global_metrics['cosine']:+.6f}",
            flush=True,
        )

    summary = {}
    for group, values in observations.items():
        tensor = torch.tensor(values, dtype=torch.float64)
        summary[group] = {
            "mean": tensor.mean().item(),
            "std": tensor.std(unbiased=False).item(),
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "negative_fraction": tensor.lt(0).double().mean().item(),
            "batches": len(values),
            "parameters": sum(group_parameters.values())
            if group == "all_adapters"
            else group_parameters[group],
        }

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "batch_size": args.batch_size,
        "batches": batches,
        "losses": losses,
        "gradient_cosine": {
            group: summary[group]
            for group in sorted(summary, key=natural_group_key)
        },
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
