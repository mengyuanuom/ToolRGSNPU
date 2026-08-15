"""Named grasp outputs with adapters for the historical tuple contract."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (tuple, list))


def _detach(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    return detach() if callable(detach) else value


@dataclass(frozen=True)
class GraspOutput:
    """Dense prediction maps produced by a language-driven grasp model."""

    segmentation: Any
    quality: Optional[Any]
    sine: Optional[Any]
    cosine: Optional[Any]
    width: Optional[Any]
    offset: Optional[Any] = None
    short_side: Optional[Any] = None

    def __post_init__(self):
        if self.segmentation is None:
            raise ValueError("GraspOutput requires a segmentation prediction")
        optional = (self.quality, self.sine, self.cosine, self.width)
        if any(value is None for value in optional) and not all(
            value is None for value in optional
        ):
            raise ValueError(
                "quality/sine/cosine/width must either all be present or all be absent"
            )

    @property
    def has_offset(self) -> bool:
        return self.offset is not None

    @property
    def has_short_side(self) -> bool:
        return self.short_side is not None

    def as_tuple(self) -> Tuple[Any, ...]:
        values = (self.segmentation, self.quality, self.sine, self.cosine, self.width)
        if self.short_side is not None:
            values += (self.short_side,)
        if self.offset is not None:
            values += (self.offset,)
        return values

    def detach(self):
        return type(self)(
            segmentation=_detach(self.segmentation),
            quality=_detach(self.quality),
            sine=_detach(self.sine),
            cosine=_detach(self.cosine),
            width=_detach(self.width),
            short_side=_detach(self.short_side),
            offset=_detach(self.offset),
        )

    @classmethod
    def from_legacy(cls, value: Any, *, model: Any = None):
        if isinstance(value, cls):
            return value
        if not _is_sequence(value) or len(value) not in (5, 6, 7):
            raise ValueError(
                "Expected five dense maps with optional short-side and offset maps"
            )
        supports_offset = bool(getattr(model, "supports_offset", False))
        predicts_short = bool(
            getattr(model, "predicts_grasp_short_side", False)
        )
        if len(value) == 7:
            return cls(*value[:5], short_side=value[5], offset=value[6])
        if len(value) == 6 and predicts_short and not supports_offset:
            return cls(*value[:5], short_side=value[5])
        if len(value) == 6 and predicts_short and supports_offset:
            raise ValueError(
                "A model advertising short-side and offset heads must return seven maps"
            )
        if len(value) == 6:
            return cls(*value[:5], offset=value[5])
        return cls(*value)


@dataclass(frozen=True)
class GraspTargets:
    """Dense supervision maps paired with :class:`GraspOutput`."""

    segmentation: Any
    quality: Optional[Any]
    sine: Optional[Any]
    cosine: Optional[Any]
    width: Optional[Any]
    offset: Optional[Any] = None
    short_side: Optional[Any] = None

    def as_tuple(self) -> Tuple[Any, ...]:
        values = (self.segmentation, self.quality, self.sine, self.cosine, self.width)
        if self.short_side is not None:
            values += (self.short_side,)
        if self.offset is not None:
            values += (self.offset,)
        return values

    @classmethod
    def from_legacy(cls, value: Any, *, model: Any = None):
        if value is None or isinstance(value, cls):
            return value
        if not _is_sequence(value) or len(value) not in (5, 6, 7):
            raise ValueError(
                "Expected five target maps with optional short-side and offset targets"
            )
        if all(item is None for item in value):
            return None
        supports_offset = bool(getattr(model, "supports_offset", False))
        predicts_short = bool(
            getattr(model, "predicts_grasp_short_side", False)
        )
        if len(value) == 7:
            return cls(*value[:5], short_side=value[5], offset=value[6])
        if len(value) == 6 and predicts_short and not supports_offset:
            return cls(*value[:5], short_side=value[5])
        if len(value) == 6:
            return cls(*value[:5], offset=value[5])
        return cls(*value)


@dataclass(frozen=True)
class GraspModelResult:
    """One normalized model call for training, evaluation, or deployment."""

    predictions: GraspOutput
    targets: Optional[GraspTargets] = None
    loss: Optional[Any] = None
    losses: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_training_result(self) -> bool:
        return self.loss is not None

    @classmethod
    def from_legacy(cls, value: Any, *, model: Any = None):
        if isinstance(value, cls):
            return value
        if (
            _is_sequence(value)
            and len(value) == 4
            and (isinstance(value[0], GraspOutput) or _is_sequence(value[0]))
        ):
            predictions, targets, loss, losses = value
            return cls(
                predictions=GraspOutput.from_legacy(predictions, model=model),
                targets=GraspTargets.from_legacy(targets, model=model),
                loss=loss,
                losses=dict(losses or {}),
            )
        if (
            _is_sequence(value)
            and len(value) == 2
            and (isinstance(value[0], GraspOutput) or _is_sequence(value[0]))
            and (
                value[1] is None
                or isinstance(value[1], GraspTargets)
                or _is_sequence(value[1])
            )
        ):
            predictions, targets = value
            return cls(
                predictions=GraspOutput.from_legacy(predictions, model=model),
                targets=GraspTargets.from_legacy(targets, model=model),
            )
        return cls(predictions=GraspOutput.from_legacy(value, model=model))

    def to_legacy(self):
        predictions = self.predictions.as_tuple()
        targets = self.targets.as_tuple() if self.targets is not None else None
        if self.is_training_result:
            return predictions, targets, self.loss, dict(self.losses)
        return predictions if targets is None else (predictions, targets)
