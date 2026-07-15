"""Single configuration-driven trainer for ToolRGS models and datasets."""

import argparse
import datetime
import os
from functools import partial
from pathlib import Path
import shutil
import time

import cv2
from loguru import logger
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils.config as config
from model import build_model
from toolrgs.engine import GraspTrainLoop, GraspValLoop  # register default loops
from toolrgs.models.base import model_requires_depth
from toolrgs.preflight import validate_required_artifacts
from toolrgs.registry import LOOPS
from toolrgs.runtime import (
    build_grad_scaler,
    build_optimizer,
    device_name,
    require_npu,
    set_device,
)
from utils.data_builder import build_dataset
from utils.misc import init_random_seed, set_random_seed, setup_logger, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train ToolRGS")
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument("--npu", type=int, default=0, help="NPU index for single-process runs")
    parser.add_argument("--opts", nargs=argparse.REMAINDER)
    cli = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(cli.config)
    if cli.opts:
        cfg = config.merge_cfg_from_list(cfg, cli.opts)
    cfg.npu = cli.npu
    return cfg


def setup_distributed(args):
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.npu = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank = 0
        args.world_size = 1
        args.npu = int(args.npu)
    # Keep cfg.gpu for older model/config code, but it now denotes the local NPU.
    args.gpu = args.npu
    args.device = str(set_device(args.npu))
    args.dist_backend = "hccl"
    if distributed:
        dist.init_process_group(backend="hccl", init_method=args.dist_url)
    args.distributed = distributed
    return distributed, torch.device(args.device)


def load_initial_weight(model, filename):
    """Load model initialization without restoring optimizer/epoch state."""
    checkpoint = torch.load(filename, map_location="cpu")
    state = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported initial weight payload: {filename}")
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded initial model weight: {}", filename)
    if incompatible.missing_keys:
        logger.warning(
            "Initial weight did not contain {} model keys: {}",
            len(incompatible.missing_keys),
            incompatible.missing_keys[:10],
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Initial weight contained {} unused keys: {}",
            len(incompatible.unexpected_keys),
            incompatible.unexpected_keys[:10],
        )


def main():
    args = parse_args()
    require_npu()

    cv2.setNumThreads(0)
    distributed, device = setup_distributed(args)
    args.manual_seed = init_random_seed(
        args.manual_seed,
        device=device,
        rank=args.rank,
        world_size=args.world_size,
    )
    set_random_seed(args.manual_seed, deterministic=False)
    is_main = args.rank == 0

    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    setup_logger(args.output_dir, distributed_rank=args.rank,
                 filename="train.log", mode="a")
    logger.info(args)
    logger.info("Ascend device: {} ({})", device, device_name(args.npu))

    try:
        artifacts = validate_required_artifacts(args)
    except FileNotFoundError as exc:
        logger.error("Model artifact preflight failed:\n{}", exc)
        raise
    for key, path in artifacts.items():
        logger.info("Artifact {}: {}", key, path)

    try:
        model, parameter_groups = build_model(args)
    except Exception:
        logger.exception("Failed to build architecture {!r}", args.architecture)
        raise
    if model_requires_depth(model) and not bool(getattr(args, "with_depth", False)):
        raise ValueError(
            f"Model {args.architecture!r} requires aligned depth input, but "
            "DATA.with_depth is false or missing. ETRG-A is currently supported "
            "with the OCID-VLG RGB-D dataset."
        )
    if getattr(args, "weight", None):
        load_initial_weight(model, args.weight)
    if bool(getattr(args, "sync_bn", False)) and distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.npu],
            output_device=args.npu,
            find_unused_parameters=True,
        )

    optimizer = build_optimizer(parameter_groups, args)
    scheduler = MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay
    )
    scaler = build_grad_scaler(enabled=bool(getattr(args, "amp", True)))

    needs_offset = args.architecture.lower() in {"crogoff", "drogoff"}
    train_data = build_dataset(args, args.train_split, with_offset=needs_offset)
    val_data = build_dataset(args, args.val_split, with_offset=needs_offset)

    train_sampler = DistributedSampler(train_data, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_data, shuffle=False) if distributed else None
    init_fn = partial(
        worker_init_fn,
        num_workers=args.workers,
        rank=args.rank,
        seed=args.manual_seed,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=bool(getattr(args, "pin_memory", False)),
        drop_last=True,
        worker_init_fn=init_fn if args.workers else None,
        collate_fn=train_data.collate_fn,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers_val,
        pin_memory=bool(getattr(args, "pin_memory", False)),
        drop_last=False,
        collate_fn=val_data.collate_fn,
    )

    best_iou = 0.0
    best_j = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_epoch = checkpoint["epoch"]
        best_iou = checkpoint.get("best_iou", 0.0)
        best_j = checkpoint.get("best_j_index", 0.0)

    train_loop_class = LOOPS.require(getattr(args, "train_loop", "grasp_train"))
    train_loop = train_loop_class(
        dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        cfg=args,
        hooks=getattr(args, "hooks", None),
    )
    val_loop_class = LOOPS.require(getattr(args, "val_loop", "grasp_val"))
    val_loop = val_loop_class(
        dataloader=val_loader,
        model=model,
        cfg=args,
        hooks=getattr(args, "val_hooks", None),
    )

    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_number = epoch + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_number)

        train_loop.run_epoch(epoch_number)
        iou, precision, j_index = val_loop.run_epoch(epoch_number)

        if is_main:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            last_path = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    "epoch": epoch_number,
                    "best_iou": best_iou,
                    "best_j_index": best_j,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "precision": precision,
                    "j_index": j_index,
                },
                last_path,
            )
            if iou >= best_iou:
                best_iou = iou
                shutil.copyfile(last_path, os.path.join(args.output_dir, "best_iou_model.pth"))
            if j_index[0] >= best_j:
                best_j = j_index[0]
                shutil.copyfile(last_path, os.path.join(args.output_dir, "best_jindex_model.pth"))
        scheduler.step()

    logger.info("Training time: {}", datetime.timedelta(seconds=int(time.time() - start)))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
