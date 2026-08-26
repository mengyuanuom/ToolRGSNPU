"""Backward-compatible DrogOff with optional Offset-Transport routing."""

from .drogoff import DROGOFF as BaseDROGOFF
from .transport_projector import OffsetTransportProjector


class DROGOFFTransport(BaseDROGOFF):
    """Enable grasp-only geometric routing through configuration.

    With ``offset_transport_enabled`` absent or false this class is exactly the
    historical DrogOff implementation. This allows the registry to keep the
    public architecture name ``drogoff`` for old experiment profiles.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.offset_transport_enabled = bool(
            getattr(cfg, "offset_transport_enabled", False)
        )
        if not self.offset_transport_enabled:
            return
        if str(getattr(cfg, "offset_version", "v1")).strip().lower() != "v2":
            raise ValueError("Offset-Transport requires DATA.offset_version: v2")
        self.proj = OffsetTransportProjector(
            self.proj,
            hidden_dim=int(
                getattr(cfg, "offset_transport_hidden_dim", 64)
            ),
            max_displacement=float(
                getattr(cfg, "offset_transport_max_displacement", 6.0)
            ),
            confidence_floor=float(
                getattr(cfg, "offset_transport_confidence_floor", 0.1)
            ),
            detach_confidence=bool(
                getattr(cfg, "offset_transport_detach_confidence", True)
            ),
        )
