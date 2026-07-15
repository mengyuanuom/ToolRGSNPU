"""Minimal MMEngine-style hook lifecycle used by ToolRGS loops."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

from toolrgs.registry import HOOKS


@dataclass
class LoopState:
    epoch: int = 0
    iteration: int = -1
    batch: Optional[Any] = None
    result: Optional[Any] = None
    logs: Dict[str, Any] = field(default_factory=dict)


class Hook:
    priority = 50

    def before_run(self, loop, state: LoopState) -> None:
        pass

    def after_run(self, loop, state: LoopState) -> None:
        pass

    def before_epoch(self, loop, state: LoopState) -> None:
        pass

    def after_epoch(self, loop, state: LoopState) -> None:
        pass

    def before_iter(self, loop, state: LoopState) -> None:
        pass

    def after_iter(self, loop, state: LoopState) -> None:
        pass


@HOOKS.register_module(name="noop")
class NoOpHook(Hook):
    """Configuration placeholder and lifecycle example."""


class HookList:
    def __init__(self, hooks: Optional[Iterable[Any]] = None):
        resolved = []
        for hook in hooks or ():
            resolved.append(HOOKS.build(hook) if isinstance(hook, (dict, str)) else hook)
        if not all(isinstance(hook, Hook) for hook in resolved):
            raise TypeError("Every loop hook must inherit toolrgs.engine.Hook")
        self.hooks = sorted(resolved, key=lambda hook: int(hook.priority))

    def call(self, event: str, loop, state: LoopState) -> None:
        for hook in self.hooks:
            callback = getattr(hook, event, None)
            if callback is None:
                raise AttributeError(f"Hook {type(hook).__name__} has no event {event!r}")
            callback(loop, state)
