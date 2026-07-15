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
    quality: Any
    sine: Any
    cosine: Any
    width: Any
    offset: Optional[Any] = None

    def __post_init__(self):
        required = (self.segmentation, self.quality, self.sine, self.cosine, self.width)
        if any(value is None for value in required):
            raise ValueError("A dense GraspOutput requires segmentation/quality/sine/cosine/width")

    @property
    def has_offset(self) -> bool:
        return self.offset is not None

    def as_tuple(self) -> Tuple[Any, ...]:
        values = (self.segmentation, self.quality, self.sine, self.cosine, self.width)
        return values + ((self.offset,) if self.offset is not None else ())

    def detach(self):
        return type(self)(*(_detach(value) for value in self.as_tuple()))

    @classmethod
    def from_legacy(cls, value: Any):
        if isinstance(value, cls):
            return value
        if not _is_sequence(value) or len(value) not in (5, 6):
            raise ValueError(
                "Expected the legacy five-map contract with an optional sixth offset map"
            )
        return cls(*value)


@dataclass(frozen=True)
class GraspTargets:
    """Dense supervision maps paired with :class:`GraspOutput`."""

    segmentation: Any
    quality: Any
    sine: Any
    cosine: Any
    width: Any
    offset: Optional[Any] = None

    def as_tuple(self) -> Tuple[Any, ...]:
        values = (self.segmentation, self.quality, self.sine, self.cosine, self.width)
        return values + ((self.offset,) if self.offset is not None else ())

    @classmethod
    def from_legacy(cls, value: Any):
        if value is None or isinstance(value, cls):
            return value
        if not _is_sequence(value) or len(value) not in (5, 6):
            raise ValueError("Expected five target maps with an optional offset target")
        if all(item is None for item in value):
            return None
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
    def from_legacy(cls, value: Any):
        if isinstance(value, cls):
            return value
        if (
            _is_sequence(value)
            and len(value) == 4
            and (isinstance(value[0], GraspOutput) or _is_sequence(value[0]))
        ):
            predictions, targets, loss, losses = value
            return cls(
                predictions=GraspOutput.from_legacy(predictions),
                targets=GraspTargets.from_legacy(targets),
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
                predictions=GraspOutput.from_legacy(predictions),
                targets=GraspTargets.from_legacy(targets),
            )
        return cls(predictions=GraspOutput.from_legacy(value))

    def to_legacy(self):
        predictions = self.predictions.as_tuple()
        targets = self.targets.as_tuple() if self.targets is not None else None
        if self.is_training_result:
            return predictions, targets, self.loss, dict(self.losses)
        return predictions if targets is None else (predictions, targets)
