"""Small, dependency-free component registry inspired by MMEngine."""

from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Union


def normalise_component_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


class Registry:
    """Map configuration names to classes or factory callables.

    Registration supports decorator and direct-call forms. Names are
    case-insensitive and normalised in the same way as existing ToolRGS config
    values, so ``grasp-tools`` and ``grasp_tools`` address the same component.
    """

    def __init__(self, name: str):
        self.name = str(name)
        self._module_dict: Dict[str, Callable[..., Any]] = {}

    @property
    def module_dict(self):
        return MappingProxyType(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return normalise_component_name(key) in self._module_dict

    def __iter__(self) -> Iterator[str]:
        return iter(self._module_dict)

    def __len__(self) -> int:
        return len(self._module_dict)

    def keys(self):
        return self._module_dict.keys()

    def get(self, key: str, default=None):
        return self._module_dict.get(normalise_component_name(key), default)

    def require(self, key: str):
        component = self.get(key)
        if component is None:
            available = ", ".join(sorted(self._module_dict)) or "<empty>"
            raise KeyError(
                f"{key!r} is not registered in {self.name}; available: {available}"
            )
        return component

    def register_module(
        self,
        module: Optional[Callable[..., Any]] = None,
        name: Optional[Union[str, Iterable[str]]] = None,
        force: bool = False,
        aliases: Iterable[str] = (),
    ):
        """Register a class/factory directly or as a decorator."""

        def register(target):
            if not callable(target):
                raise TypeError(f"Registry entries must be callable, got {type(target)!r}")
            if name is None:
                names = [target.__name__]
            elif isinstance(name, str):
                names = [name]
            else:
                names = list(name)
            names.extend([aliases] if isinstance(aliases, str) else aliases)
            if not names:
                raise ValueError("At least one registry name is required")
            for raw_name in names:
                key = normalise_component_name(raw_name)
                if not key:
                    raise ValueError("Registry names cannot be empty")
                existing = self._module_dict.get(key)
                if existing is not None and existing is not target and not force:
                    raise KeyError(
                        f"{key!r} is already registered in {self.name} as {existing}"
                    )
                self._module_dict[key] = target
            return target

        return register(module) if module is not None else register

    def build(self, cfg, default_args: Optional[Mapping] = None):
        return build_from_cfg(cfg, self, default_args=default_args)

    def __repr__(self) -> str:
        entries = ", ".join(sorted(self._module_dict))
        return f"Registry(name={self.name!r}, items=[{entries}])"


def build_from_cfg(cfg, registry: Registry, default_args: Optional[Mapping] = None):
    """Instantiate one registered component from ``{'type': ..., ...}``."""
    if isinstance(cfg, str):
        args = {"type": cfg}
    elif isinstance(cfg, Mapping):
        args = deepcopy(dict(cfg))
    else:
        raise TypeError(f"cfg must be a string or mapping, got {type(cfg)!r}")
    if default_args is not None:
        if not isinstance(default_args, Mapping):
            raise TypeError("default_args must be a mapping")
        for key, value in default_args.items():
            args.setdefault(key, value)
    if "type" not in args:
        raise KeyError(f"Component config for {registry.name} requires a 'type' field")
    component_type = args.pop("type")
    if isinstance(component_type, str):
        component = registry.require(component_type)
    elif callable(component_type):
        component = component_type
    else:
        raise TypeError("cfg.type must be a registered name or callable")
    try:
        return component(**args)
    except TypeError as exc:
        name = getattr(component, "__name__", repr(component))
        raise TypeError(f"Failed to build {name} from registry {registry.name}: {exc}") from exc


MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
LOSSES = Registry("losses")
METRICS = Registry("metrics")
POSTPROCESSORS = Registry("postprocessors")
LOOPS = Registry("loops")
HOOKS = Registry("hooks")
CAMERAS = Registry("cameras")
ROBOT_CLIENTS = Registry("robot clients")
DETECTORS = Registry("detectors")
AUDIO_INPUTS = Registry("audio inputs")
