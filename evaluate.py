"""Evaluate any ToolRGS architecture using one experiment config."""

import argparse
import os

import cv2
from loguru import logger
import torch
from torch.utils.data import DataLoader

import utils.config as config
from model import build_model
from toolrgs.engine import GraspValLoop  # imports and registers the default loop
from toolrgs.preflight import validate_required_artifacts
from toolrgs.registry import LOOPS
from toolrgs.runtime import device_name, require_npu, set_device
from toolrgs.datasets import build_dataset
from utils.misc import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ToolRGS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npu", type=int, default=0)
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    if cli.opts:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)
    cfg.npu = cli.npu
    cfg.resume = cli.checkpoint
    return cfg


def load_state(model, state):
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)


def main():
    args = parse_args()
    require_npu()
    cv2.setNumThreads(0)
    args.gpu = args.npu
    device = set_device(args.npu)
    args.device = str(device)
    args.rank = 0
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    setup_logger(args.output_dir, distributed_rank=0, filename="eval.log", mode="a")

    logger.info("Ascend device: {} ({})", device, device_name(args.npu))
    try:
        artifacts = validate_required_artifacts(args)
    except FileNotFoundError as exc:
        logger.error("Model artifact preflight failed:\n{}", exc)
        raise
    for key, path in artifacts.items():
        logger.info("Artifact {}: {}", key, path)
    try:
        model, _ = build_model(args)
    except Exception:
        logger.exception("Failed to build architecture {!r}", args.architecture)
        raise
    model = model.to(device).eval()
    checkpoint = torch.load(args.resume, map_location="cpu")
    load_state(model, checkpoint.get("state_dict", checkpoint))

    needs_offset = args.architecture.lower() in {"crogoff", "drogoff"}
    dataset = build_dataset(args, args.val_split, with_offset=needs_offset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=bool(getattr(args, "pin_memory", False)),
        collate_fn=dataset.collate_fn,
    )
    val_loop_class = LOOPS.require(getattr(args, "val_loop", "grasp_val"))
    val_loop = val_loop_class(
        dataloader=loader,
        model=model,
        cfg=args,
        hooks=getattr(args, "val_hooks", None),
    )
    iou, precision, j_index = val_loop.run_epoch(getattr(args, "start_epoch", 0))
    logger.info("Final IoU={}, precision={}, J={}", iou, precision, j_index)


if __name__ == "__main__":
    main()
