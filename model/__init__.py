"""Compatibility model builders backed by the ToolRGS component registry."""

from loguru import logger

from toolrgs.registry import MODELS

from .crog import CROG
from .crogoff import CROGOFF
from .drog import DROG
from .drogoff import DROGOFF
from .etrg import ETRG
from .ggcnnclip import GGCNN_CLIP
from .grconvnetclip import GenerativeResnet_CLIP
from .graspmamba import GraspMamba
from .lgd import LGD
from .segmenter import DETRIS


MODELS.register_module(CROG, name="crog")
MODELS.register_module(CROGOFF, name="crogoff")
MODELS.register_module(DETRIS, name="detris")
MODELS.register_module(DROG, name="drog")
MODELS.register_module(DROGOFF, name="drogoff")
MODELS.register_module(ETRG, name="etrg", aliases=("etrg_a", "etrg_depth"))
MODELS.register_module(GGCNN_CLIP, name="ggcnnclip", aliases=("ggcnn_clip",))
MODELS.register_module(
    GenerativeResnet_CLIP,
    name="grconvnetclip",
    aliases=("grconvnet_clip", "gr_convnet_clip"),
)
MODELS.register_module(GraspMamba, name="graspmamba", aliases=("grasp_mamba",))
MODELS.register_module(LGD, name="lgd")

# Historical public name; it is now a live read-only view of the shared registry.
MODEL_REGISTRY = MODELS.module_dict


def _build_parameter_groups(model, cfg):
    """Keep optimizer grouping separate from component construction."""
    backbone, head, frozen = [], [], []
    for param_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen.append(parameter)
        elif (
            param_name.startswith("backbone")
            or param_name.startswith("bridger")
            or param_name.startswith("txt_backbone")
            or param_name.startswith("dinov2")
        ):
            backbone.append(parameter)
        else:
            head.append(parameter)
    parameter_groups = [
        {"params": backbone, "initial_lr": cfg.lr_multi * cfg.base_lr},
        {"params": head, "initial_lr": cfg.base_lr},
    ]
    return parameter_groups, backbone, head, frozen


def build_model(cfg):
    """Build a registered model while preserving the legacy return signature."""
    name = str(getattr(cfg, "architecture", getattr(cfg, "type", "drog")))
    try:
        model_cls = MODELS.require(name)
    except KeyError as exc:
        available = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model {name!r}; available: {available}") from exc

    model = model_cls(cfg)
    parameter_groups, backbone, head, frozen = _build_parameter_groups(model, cfg)
    logger.info(
        "Build {}: backbone={}, head={}, frozen={}",
        name,
        sum(p.numel() for p in backbone),
        sum(p.numel() for p in head),
        sum(p.numel() for p in frozen),
    )
    return model, parameter_groups


# Compatibility aliases for imported DETRIS scripts.
def build_segmenter(cfg):
    cfg.architecture = "detris"
    return build_model(cfg)


def build_drog(cfg):
    cfg.architecture = "drog"
    return build_model(cfg)


def build_drogoff(cfg):
    cfg.architecture = "drogoff"
    return build_model(cfg)
