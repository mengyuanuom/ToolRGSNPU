"""Runner loops and hooks for ToolRGS, with lazy PyTorch loop imports."""

from .hooks import CheckpointHook, Hook, HookList, LoggerHook, LoopState, NoOpHook

__all__ = [
    "BaseLoop",
    "GraspTrainLoop",
    "GraspValLoop",
    "Hook",
    "HookList",
    "CheckpointHook",
    "LoggerHook",
    "LoopState",
    "NoOpHook",
    "NPUGraspRunner",
    "build_runner",
]


def __getattr__(name):
    if name in {"BaseLoop", "GraspTrainLoop"}:
        from .loops import BaseLoop, GraspTrainLoop

        return {"BaseLoop": BaseLoop, "GraspTrainLoop": GraspTrainLoop}[name]
    if name == "GraspValLoop":
        from .val_loop import GraspValLoop

        return GraspValLoop
    if name in {"NPUGraspRunner", "build_runner"}:
        from .runner import NPUGraspRunner, build_runner

        return {
            "NPUGraspRunner": NPUGraspRunner,
            "build_runner": build_runner,
        }[name]
    raise AttributeError(name)
