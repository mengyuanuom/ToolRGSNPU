"""Configuration-driven Ascend runner owning the complete training lifecycle."""

import datetime
from functools import partial
import os
from pathlib import Path
import shutil
import time

import cv2
from loguru import logger
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import build_model
from toolrgs.datasets import build_dataset
from toolrgs.engine.batch import per_process_batch_size
from toolrgs.engine.hooks import HookList, LoopState
from toolrgs.engine.loops import GraspTrainLoop  # noqa: F401 - registers loop
from toolrgs.engine.optim import build_optim_wrapper, build_param_scheduler
from toolrgs.engine.samplers import DistributedEvalSampler
from toolrgs.engine.val_loop import GraspValLoop  # noqa: F401 - registers loop
from toolrgs.models.base import model_requires_depth
from toolrgs.preflight import validate_required_artifacts
from toolrgs.registry import LOOPS, RUNNERS
from toolrgs.runtime import (
    build_grad_scaler,
    build_optimizer,
    device_name,
    require_npu,
    set_device,
)
from utils.config import resolve_grasp_size_activation
from utils.misc import init_random_seed, set_random_seed, setup_logger, worker_init_fn


def _clean_state_dict(state):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }



@RUNNERS.register_module(name="npu_grasp", aliases=("npu_runner", "runner"))
class NPUGraspRunner:
    """Build and run one ToolRGS experiment from a flat compatibility config."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.distributed = False
        self.device = None
        self.is_main = True
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.optim_wrapper = None
        self.train_sampler = None
        self.train_batch_size_per_process = None
        self.val_batch_size_per_process = None
        self.best_iou = 0.0
        self.best_j1 = 0.0
        self.best_j5 = 0.0
        self._is_setup = False
        hooks = getattr(cfg, "runner_hooks", None) or (
            {"type": "logger"},
            {"type": "checkpoint"},
        )
        self.hooks = HookList(hooks)
        self.state = LoopState()

    def _setup_distributed(self):
        cfg = self.cfg
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        if self.distributed:
            cfg.rank = int(os.environ["RANK"])
            cfg.world_size = int(os.environ["WORLD_SIZE"])
            cfg.npu = int(os.environ.get("LOCAL_RANK", 0))
        else:
            cfg.rank = 0
            cfg.world_size = 1
            cfg.npu = int(getattr(cfg, "npu", 0))
        cfg.gpu = cfg.npu  # historical model/config compatibility
        self.device = set_device(cfg.npu)
        cfg.device = str(self.device)
        cfg.dist_backend = "hccl"
        if self.distributed:
            dist.init_process_group(backend="hccl", init_method=cfg.dist_url)
        cfg.distributed = self.distributed
        self.is_main = cfg.rank == 0

    def _configure_batch_sizes(self):
        cfg = self.cfg
        cfg.global_batch_size = int(cfg.batch_size)
        cfg.global_batch_size_val = int(cfg.batch_size_val)
        self.train_batch_size_per_process = per_process_batch_size(
            cfg.global_batch_size, cfg.world_size, "batch_size"
        )
        self.val_batch_size_per_process = per_process_batch_size(
            cfg.global_batch_size_val, cfg.world_size, "batch_size_val"
        )

    def _load_initial_weight(self, filename):
        checkpoint = torch.load(filename, map_location="cpu")
        state = (
            checkpoint.get("state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported initial weight payload: {filename}")
        target = getattr(self.model, "module", self.model)
        cleaned = _clean_state_dict(state)
        load_target = target
        is_legacy_base_weight = (
            bool(getattr(target, "predicts_grasp_short_side", False))
            and not any(
                key.startswith(("base_model.", "short_side_head."))
                for key in cleaned
            )
        )
        if is_legacy_base_weight:
            load_target = target.base_model
            logger.info("Loading legacy initial weight into the wrapped base model")
        incompatible = load_target.load_state_dict(cleaned, strict=False)
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

    def _load_model_state(self, state):
        try:
            self.model.load_state_dict(state, strict=True)
            return
        except RuntimeError:
            target = getattr(self.model, "module", self.model)
            target.load_state_dict(_clean_state_dict(state), strict=True)

    def _build_dataloader(self, dataset, train):
        cfg = self.cfg
        sampler = None
        if self.distributed:
            if train:
                sampler = DistributedSampler(dataset, shuffle=True)
            else:
                sampler = DistributedEvalSampler(
                    dataset,
                    num_replicas=cfg.world_size,
                    rank=cfg.rank,
                )
        workers = int(cfg.workers if train else cfg.workers_val)
        init_fn = None
        if train and workers:
            init_fn = partial(
                worker_init_fn,
                num_workers=workers,
                rank=cfg.rank,
                seed=cfg.manual_seed,
            )
        loader = DataLoader(
            dataset,
            batch_size=(
                self.train_batch_size_per_process
                if train
                else self.val_batch_size_per_process
            ),
            shuffle=bool(train and sampler is None),
            sampler=sampler,
            num_workers=workers,
            pin_memory=bool(getattr(cfg, "pin_memory", False)),
            drop_last=bool(train),
            worker_init_fn=init_fn,
            collate_fn=dataset.collate_fn,
        )
        return loader, sampler

    def setup(self):
        if self._is_setup:
            return self
        cfg = self.cfg
        require_npu()
        cv2.setNumThreads(0)
        self._setup_distributed()
        self._configure_batch_sizes()
        cfg.manual_seed = init_random_seed(
            cfg.manual_seed,
            device=self.device,
            rank=cfg.rank,
            world_size=cfg.world_size,
        )
        set_random_seed(cfg.manual_seed, deterministic=False)

        cfg.output_dir = os.path.join(cfg.output_folder, cfg.exp_name)
        setup_logger(
            cfg.output_dir,
            distributed_rank=cfg.rank,
            filename="train.log",
            mode="a",
        )
        logger.info(cfg)
        logger.info("Ascend device: {} ({})", self.device, device_name(cfg.npu))
        logger.info(
            "Batch sizes: train global={} per-process={}; "
            "validation global={} per-process={}; world_size={}",
            cfg.global_batch_size,
            self.train_batch_size_per_process,
            cfg.global_batch_size_val,
            self.val_batch_size_per_process,
            cfg.world_size,
        )

        try:
            artifacts = validate_required_artifacts(cfg)
        except FileNotFoundError as exc:
            logger.error("Model artifact preflight failed:\n{}", exc)
            raise
        for key, path in artifacts.items():
            logger.info("Artifact {}: {}", key, path)
        try:
            self.model, parameter_groups = build_model(cfg)
        except Exception:
            logger.exception("Failed to build architecture {!r}", cfg.architecture)
            raise
        self.grasp_size_activation = resolve_grasp_size_activation(
            getattr(cfg, "grasp_size_activation", "auto"), model=self.model
        )
        logger.info(
            "Grasp-size protocol: coordinate={}, factor={}, height={}, activation={}",
            getattr(cfg, "grasp_size_coordinate", "original"),
            float(getattr(cfg, "grasp_size_factor", 100.0)),
            float(getattr(cfg, "grasp_height", 20.0)),
            self.grasp_size_activation,
        )
        if model_requires_depth(self.model) and not bool(
            getattr(cfg, "with_depth", False)
        ):
            raise ValueError(
                f"Model {cfg.architecture!r} requires aligned depth input, but "
                "DATA.with_depth is false or missing."
            )
        if getattr(cfg, "weight", None):
            self._load_initial_weight(cfg.weight)
        if bool(getattr(cfg, "sync_bn", False)) and self.distributed:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.model.to(self.device)
        if self.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[cfg.npu],
                output_device=cfg.npu,
                find_unused_parameters=True,
            )

        self.optimizer = build_optimizer(parameter_groups, cfg)
        self.scaler = build_grad_scaler(enabled=bool(getattr(cfg, "amp", False)))
        self.optim_wrapper = build_optim_wrapper(
            cfg, optimizer=self.optimizer, scaler=self.scaler
        )
        self.scheduler = build_param_scheduler(cfg, optimizer=self.optimizer)
        if self.is_main:
            logger.info(
                "Precision path: amp={}, scaler={}",
                bool(getattr(cfg, "amp", False)),
                type(self.scaler).__name__,
            )
            logger.info(
                "Optimizer learning rates by parameter group: {}",
                [group["lr"] for group in self.optimizer.param_groups],
            )

        needs_offset = str(cfg.architecture).lower() in {"crogoff", "drogoff"}
        train_data = build_dataset(cfg, cfg.train_split, with_offset=needs_offset)
        val_data = build_dataset(cfg, cfg.val_split, with_offset=needs_offset)
        self.train_loader, self.train_sampler = self._build_dataloader(
            train_data, train=True
        )
        self.val_loader, _ = self._build_dataloader(val_data, train=False)
        if self.is_main:
            logger.info(
                "Effective batch: train_per_process={}, world_size={}, "
                "train_global={}, val_per_process={}, val_global={}, steps_per_epoch={}",
                self.train_batch_size_per_process,
                int(cfg.world_size),
                int(cfg.global_batch_size),
                self.val_batch_size_per_process,
                int(cfg.global_batch_size_val),
                len(self.train_loader),
            )

        if getattr(cfg, "resume", None):
            checkpoint = torch.load(cfg.resume, map_location="cpu")
            checkpoint_factor = checkpoint.get("grasp_size_factor")
            checkpoint_coordinate = checkpoint.get("grasp_size_coordinate")
            checkpoint_activation = checkpoint.get("grasp_size_activation")
            checkpoint_short_side = checkpoint.get("predict_grasp_short_side")
            configured_factor = float(getattr(cfg, "grasp_size_factor", 100.0))
            configured_coordinate = str(
                getattr(cfg, "grasp_size_coordinate", "original")
            )
            configured_short_side = bool(
                getattr(cfg, "predict_grasp_short_side", False)
            )
            if checkpoint_factor is not None and not abs(
                float(checkpoint_factor) - configured_factor
            ) < 1e-6:
                raise ValueError(
                    "Resume checkpoint grasp_size_factor does not match the "
                    f"config: {checkpoint_factor} vs {configured_factor}"
                )
            if checkpoint_coordinate is not None and str(
                checkpoint_coordinate
            ) != configured_coordinate:
                raise ValueError(
                    "Resume checkpoint grasp_size_coordinate does not match "
                    f"the config: {checkpoint_coordinate!r} vs "
                    f"{configured_coordinate!r}"
                )
            if checkpoint_activation is not None and str(
                checkpoint_activation
            ) != self.grasp_size_activation:
                raise ValueError(
                    "Resume checkpoint grasp_size_activation does not match "
                    f"the model/config: {checkpoint_activation!r} vs "
                    f"{self.grasp_size_activation!r}"
                )
            if checkpoint_short_side is not None and bool(
                checkpoint_short_side
            ) != configured_short_side:
                raise ValueError(
                    "Resume checkpoint short-side protocol does not match the "
                    f"config: {checkpoint_short_side!r} vs "
                    f"{configured_short_side!r}"
                )
            self._load_model_state(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            cfg.start_epoch = int(checkpoint["epoch"])
            self.best_iou = float(checkpoint.get("best_iou", 0.0))
            self.best_j1 = float(
                checkpoint.get(
                    "best_j1_index", checkpoint.get("best_j_index", 0.0)
                )
            )
            self.best_j5 = float(checkpoint.get("best_j5_index", 0.0))
            logger.info("Resumed experiment from epoch {}", cfg.start_epoch)

        train_loop_class = LOOPS.require(getattr(cfg, "train_loop", "grasp_train"))
        self.train_loop = train_loop_class(
            dataloader=self.train_loader,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            optim_wrapper=self.optim_wrapper,
            cfg=cfg,
            hooks=getattr(cfg, "hooks", None),
        )
        val_loop_class = LOOPS.require(getattr(cfg, "val_loop", "grasp_val"))
        self.val_loop = val_loop_class(
            dataloader=self.val_loader,
            model=self.model,
            cfg=cfg,
            hooks=getattr(cfg, "val_hooks", None),
        )
        self._is_setup = True
        return self

    def save_checkpoint(self, epoch, logs):
        if not self.is_main:
            return
        cfg = self.cfg
        validation = logs.get("validation", {})
        iou = float(validation.get("iou", 0.0))
        precision = validation.get("precision", {})
        j_index = list(validation.get("j_index", []))
        j_at_one = float(j_index[0]) if j_index else 0.0
        j_at_five = float(j_index[1]) if len(j_index) > 1 else 0.0
        improved_iou = iou > self.best_iou
        improved_j1 = j_at_one > self.best_j1
        improved_j5 = j_at_five > self.best_j5
        if improved_iou:
            self.best_iou = iou
        if improved_j1:
            self.best_j1 = j_at_one
        if improved_j5:
            self.best_j5 = j_at_five

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        targets = []
        if improved_iou:
            targets.append(
                (
                    output_dir / f"best_iou_epoch_{int(epoch):03d}.pth",
                    "best_iou_epoch_*.pth",
                )
            )
        if improved_j1:
            targets.append(
                (
                    output_dir / f"best_j1_epoch_{int(epoch):03d}.pth",
                    "best_j1_epoch_*.pth",
                )
            )
        if improved_j5:
            targets.append(
                (
                    output_dir / f"best_j5_epoch_{int(epoch):03d}.pth",
                    "best_j5_epoch_*.pth",
                )
            )

        checkpoint = {
            "epoch": int(epoch),
            "best_iou": self.best_iou,
            # Keep the legacy field for old resume/evaluation consumers. It is
            # explicitly J@1; J@5 has its own field and checkpoint series.
            "best_j_index": self.best_j1,
            "best_j1_index": self.best_j1,
            "best_j5_index": self.best_j5,
            "grasp_size_factor": float(
                getattr(cfg, "grasp_size_factor", 100.0)
            ),
            "grasp_height": float(getattr(cfg, "grasp_height", 20.0)),
            "grasp_size_coordinate": str(
                getattr(cfg, "grasp_size_coordinate", "original")
            ),
            "grasp_size_activation": self.grasp_size_activation,
            "state_dict": self.model.state_dict(),
            "predict_grasp_short_side": bool(
                getattr(cfg, "predict_grasp_short_side", False)
            ),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "precision": precision,
            "j_index": j_index,
            "validation": {
                "iou": iou,
                "j_at_one": j_at_one,
                "j_at_five": j_at_five,
            },
            "meta": {
                "architecture": str(cfg.architecture),
                "config": getattr(cfg, "filename", None),
            },
        }

        # Always keep the most recent fully completed epoch. Write it
        # atomically so interruption during serialization cannot corrupt the
        # previous resumable checkpoint.
        last_path = output_dir / "last.pth"
        temporary_path = output_dir / ".last.pth.tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, last_path)
        logger.info("Saved latest completed epoch checkpoint: {}", last_path)

        for target, pattern in targets:
            shutil.copyfile(last_path, target)
            for previous in output_dir.glob(pattern):
                if previous != target:
                    previous.unlink()
            logger.info("Saved new best checkpoint: {}", target)

    def train(self):
        self.setup()
        cfg = self.cfg
        started = time.time()
        self.hooks.call("before_run", self, self.state)
        try:
            for epoch_index in range(int(cfg.start_epoch), int(cfg.epochs)):
                epoch = epoch_index + 1
                self.state = LoopState(epoch=epoch)
                if self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
                if hasattr(self.train_loader.dataset, "set_epoch"):
                    self.train_loader.dataset.set_epoch(epoch)
                self.hooks.call("before_epoch", self, self.state)
                train_logs = self.train_loop.run_epoch(epoch)
                val_start = int(getattr(cfg, "val_start_epoch", 1))
                val_freq = max(1, int(getattr(cfg, "val_freq", 1)))
                should_validate = (
                    bool(getattr(cfg, "evaluate", True))
                    and epoch >= val_start
                    and (epoch - val_start) % val_freq == 0
                )
                if should_validate:
                    iou, precision, j_index = self.val_loop.run_epoch(epoch)
                    validation_logs = {
                        "iou": iou,
                        "precision": precision,
                        "j_index": j_index,
                    }
                else:
                    validation_logs = {}
                self.state.logs = {
                    "train": train_logs,
                    "validation": validation_logs,
                }
                self.scheduler.step()
                self.hooks.call("after_epoch", self, self.state)
        finally:
            self.hooks.call("after_run", self, self.state)
            if self.distributed and dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        logger.info(
            "Training time: {}",
            datetime.timedelta(seconds=int(time.time() - started)),
        )
        return self.state.logs


def build_runner(cfg):
    runner_cfg = getattr(cfg, "runner", None) or {"type": "npu_grasp"}
    if isinstance(runner_cfg, str):
        runner_cfg = {"type": runner_cfg}
    return RUNNERS.build(runner_cfg, default_args={"cfg": cfg})
