"""Training loops separated from CLI orchestration and model implementations."""

from abc import ABC, abstractmethod
import time
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist

from toolrgs.engine.hooks import HookList, LoopState
from toolrgs.models.base import model_requires_depth
from toolrgs.registry import LOOPS
from toolrgs.runtime import autocast, current_device, move_to_device
from toolrgs.structures import GraspModelResult
from utils.misc import AverageMeter, ProgressMeter, trainMetricGPU


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().mean().item()
    return float(value)


class BaseLoop(ABC):
    def __init__(self, hooks: Optional[Iterable[Any]] = None):
        self.hooks = HookList(hooks)
        self.state = LoopState()

    @abstractmethod
    def run_epoch(self, epoch: int):
        raise NotImplementedError


@LOOPS.register_module(name="grasp_train", aliases=("train_with_grasp",))
class GraspTrainLoop(BaseLoop):
    """One epoch of dense grasp training using the named model-result contract."""

    def __init__(
        self,
        dataloader,
        model,
        optimizer,
        scheduler,
        scaler,
        cfg,
        hooks: Optional[Iterable[Any]] = None,
    ):
        super().__init__(hooks=hooks)
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.cfg = cfg
        self.device = current_device(int(getattr(cfg, "npu", getattr(cfg, "gpu", 0))))

    def _meters(self, epoch):
        meters = {
            "batch": AverageMeter("Batch", ":2.2f"),
            "data": AverageMeter("Data", ":2.2f"),
            "lr": AverageMeter("Lr", ":1.6f"),
            "loss": AverageMeter("Loss", ":2.4f"),
            "quality": AverageMeter("Loss_qua", ":2.4f"),
            "sine": AverageMeter("Loss_sin", ":2.4f"),
            "cosine": AverageMeter("Loss_cos", ":2.4f"),
            "width": AverageMeter("Loss_wid", ":2.4f"),
            "offset": AverageMeter("Loss_off", ":2.4f"),
            "iou": AverageMeter("IoU", ":2.2f"),
            "precision": AverageMeter("Prec@50", ":2.2f"),
        }
        progress = ProgressMeter(
            len(self.dataloader),
            list(meters.values()),
            prefix=f"Training: Epoch=[{epoch}/{self.cfg.epochs}] ",
        )
        return meters, progress

    def _to_device(self, data):
        masks = data["grasp_masks"]
        offset = masks.get("off")
        offset_weight = masks.get("off_w")
        common = (
            move_to_device(data["img"], self.device),
            move_to_device(data["word_vec"], self.device),
            move_to_device(data["mask"], self.device).unsqueeze(1),
            move_to_device(masks["qua"], self.device).unsqueeze(1),
            move_to_device(masks["sin"], self.device).unsqueeze(1),
            move_to_device(masks["cos"], self.device).unsqueeze(1),
            move_to_device(masks["wid"], self.device).unsqueeze(1),
            move_to_device(offset, self.device) if offset is not None else None,
            move_to_device(offset_weight, self.device)
            if offset_weight is not None
            else None,
        )
        if not model_requires_depth(self.model):
            return common
        depth = data.get("depth")
        if depth is None:
            raise KeyError(
                "The selected model requires batch['depth'], but the dataset did "
                "not provide it. Use OCID-VLG with DATA.with_depth=True."
            )
        return (common[0], move_to_device(depth, self.device), *common[1:])

    def run_epoch(self, epoch: int):
        self.state = LoopState(epoch=epoch)
        self.hooks.call("before_epoch", self, self.state)
        meters, progress = self._meters(epoch)
        self.model.train()
        end = time.time()

        for iteration, data in enumerate(self.dataloader):
            self.state.iteration = iteration
            self.state.batch = data
            self.hooks.call("before_iter", self, self.state)
            meters["data"].update(time.time() - end)
            inputs = self._to_device(data)
            image = inputs[0]

            with autocast(enabled=bool(getattr(self.cfg, "amp", True))):
                result = GraspModelResult.from_legacy(self.model(*inputs))
            if result.loss is None:
                raise RuntimeError("GraspTrainLoop requires a model result with a training loss")
            if result.targets is None:
                raise RuntimeError("GraspTrainLoop requires dense supervision targets")
            loss = result.loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.cfg.max_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            iou, precision = trainMetricGPU(
                result.predictions.segmentation,
                result.targets.segmentation,
                0.35,
                0.5,
            )
            reduced_loss = loss.detach().clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(reduced_loss)
                dist.all_reduce(iou)
                dist.all_reduce(precision)
                world_size = dist.get_world_size()
                reduced_loss /= world_size
                iou /= world_size
                precision /= world_size

            batch_size = image.size(0)
            losses = result.losses
            meters["loss"].update(reduced_loss.item(), batch_size)
            meters["quality"].update(_scalar(losses.get("m_qua", 0.0)), batch_size)
            meters["sine"].update(_scalar(losses.get("m_sin", 0.0)), batch_size)
            meters["cosine"].update(_scalar(losses.get("m_cos", 0.0)), batch_size)
            meters["width"].update(_scalar(losses.get("m_wid", 0.0)), batch_size)
            meters["offset"].update(_scalar(losses.get("m_off", 0.0)), batch_size)
            meters["iou"].update(iou.item(), batch_size)
            meters["precision"].update(precision.item(), batch_size)
            meters["lr"].update(self.scheduler.get_last_lr()[-1])
            meters["batch"].update(time.time() - end)
            end = time.time()

            self.state.result = result
            self.state.logs = {name: meter.val for name, meter in meters.items()}
            self.hooks.call("after_iter", self, self.state)
            if (iteration + 1) % self.cfg.print_freq == 0:
                progress.display(iteration + 1)

        summary = {name: meter.avg for name, meter in meters.items()}
        self.state.logs = summary
        self.hooks.call("after_epoch", self, self.state)
        return summary
