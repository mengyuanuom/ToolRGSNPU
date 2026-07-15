"""Compatibility dataset builder backed by the ToolRGS registry."""

import inspect
from collections.abc import Mapping

from toolrgs.registry import DATASETS, normalise_component_name

from utils.dataset import GraspToolDataset
from utils.ocid_vlg_dataset import OCIDVLGDataset
from utils.vcot_dataset import VCoTDataset


DATASETS.register_module(
    GraspToolDataset,
    name="grasptool",
    aliases=("grasp_tool", "grasp_tools"),
)
DATASETS.register_module(
    OCIDVLGDataset,
    name="ocid_vlg",
    aliases=("ocidvlg",),
)
DATASETS.register_module(VCoTDataset, name="vcot", aliases=("vcot_grasp",))

# Historical public name; now a live read-only registry view.
DATASET_REGISTRY = DATASETS.module_dict


def _supported_init_kwargs(dataset_class, candidates):
    parameters = inspect.signature(dataset_class.__init__).parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return candidates
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in candidates.items() if key in accepted}


def build_dataset(cfg, split, with_offset=False):
    """Build the configured dataset without coupling train/eval to its class."""
    name = normalise_component_name(cfg.dataset)
    try:
        dataset_class = DATASETS.require(name)
    except KeyError as error:
        available = ", ".join(sorted(DATASETS.keys()))
        raise ValueError(f"Unknown DATA.dataset {cfg.dataset!r}; available: {available}") from error

    candidates = dict(
        root_dir=cfg.root_path,
        input_size=cfg.input_size,
        split=split,
        word_length=cfg.word_len,
        with_offset=with_offset,
        offset_radius=getattr(cfg, "offset_r", 20.0),
        offset_sigma=getattr(cfg, "offset_sigma", None),
        split_root=getattr(cfg, "split_root", None),
        prompt_template=getattr(cfg, "prompt_template", "Grasp the {object_name}"),
        version=getattr(cfg, "version", "multiple"),
        with_depth=getattr(cfg, "with_depth", True),
    )
    dataset_args = getattr(cfg, "dataset_args", {})
    if dataset_args:
        if not isinstance(dataset_args, Mapping):
            raise TypeError("DATA.dataset_args must be a mapping")
        candidates.update(dataset_args)
    kwargs = _supported_init_kwargs(dataset_class, candidates)
    return DATASETS.build({"type": name, **kwargs})
