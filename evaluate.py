"""Evaluate any ToolRGS architecture using one experiment config."""

import argparse
import os

import cv2
from loguru import logger
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import utils.config as config
from model import build_model
from toolrgs.engine.batch import per_process_batch_size
from toolrgs.engine import GraspValLoop, RealVLGValLoop  # register validation loops
from toolrgs.engine.samplers import DistributedEvalSampler
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
    parser.add_argument(
        "--split",
        help="Evaluation split override (for VCoT use 'seen' or 'unseen').",
    )
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    if cli.opts:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)
    cfg.npu = cli.npu
    cfg.resume = cli.checkpoint
    cfg.eval_split = cli.split or getattr(
        cfg, "test_split", getattr(cfg, "val_split", None)
    )
    if cfg.eval_split is None:
        raise ValueError("TEST.test_split or DATA.val_split must be configured")
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
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.npu = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank = 0
        args.world_size = 1
    args.gpu = args.npu
    device = set_device(args.npu)
    args.device = str(device)
    args.distributed = distributed
    if distributed:
        dist.init_process_group(backend="hccl", init_method=args.dist_url)
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    setup_logger(
        args.output_dir,
        distributed_rank=args.rank,
        filename="eval.log",
        mode="a",
    )

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
    logger.info(
        "Evaluation split={} protocol={}",
        args.eval_split,
        getattr(args, "evaluation_protocol", "toolrgs"),
    )
    dataset = build_dataset(args, args.eval_split, with_offset=needs_offset)
    args.global_batch_size_val = int(args.batch_size_val)
    batch_size_per_process = per_process_batch_size(
        args.global_batch_size_val, args.world_size, "batch_size_val"
    )
    sampler = (
        DistributedEvalSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
        )
        if distributed
        else None
    )
    if args.rank == 0:
        logger.info(
            "Evaluation batch: global={} per-process={} world_size={}",
            args.global_batch_size_val,
            batch_size_per_process,
            args.world_size,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_process,
        shuffle=False,
        sampler=sampler,
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
    protocol = str(getattr(args, "evaluation_protocol", "")).lower()
    if protocol in {"realvlg", "realvlg_source", "realvlg_official"}:
        logger.info(
            "Final RealVLG F_beta={}, metrics={}, gAcc={}",
            iou,
            precision,
            j_index[0] if j_index else 0.0,
        )
    elif protocol == "vcot_official":
        logger.info(
            "Final IoU={}, precision={}, GraspSR={}",
            iou,
            precision,
            j_index[0] if j_index else 0.0,
        )
    else:
        logger.info("Final IoU={}, precision={}, J={}", iou, precision, j_index)


if __name__ == "__main__":
    main()
