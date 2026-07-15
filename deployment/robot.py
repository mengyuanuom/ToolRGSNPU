"""Safe client for the legacy Kinova-side ToolRGS TCP receiver."""

from dataclasses import dataclass
import math
import socket
from typing import Dict, Iterable, Mapping, Optional, Sequence

from toolrgs.registry import ROBOT_CLIENTS


TIER_DEPTH: Dict[str, int] = {"L1": -1, "L2": 0, "L3": 1}
TOOL_TIERS: Dict[str, str] = {
    "box": "L1",
    "clamp": "L3",
    "clip": "L1",
    "crimp": "L2",
    "hex key": "L3",
    "mallet": "L3",
    "marker": "L2",
    "screwdriver": "L2",
    "sponge": "L1",
    "spool": "L2",
    "tape measure": "L1",
    "tape": "L1",
    "wrench": "L3",
}


def semantic_depth(text: str, default: int = 0) -> int:
    """Match the object-tier depth convention used by the server CROG demo."""
    lowered = text.casefold()
    for keyword in sorted(TOOL_TIERS, key=len, reverse=True):
        if keyword in lowered:
            return TIER_DEPTH[TOOL_TIERS[keyword]]
    return int(default)


def _wire_number(value: float) -> str:
    rounded = round(float(value), 3)
    return str(int(rounded)) if rounded.is_integer() else f"{rounded:.3f}".rstrip("0")


@dataclass(frozen=True)
class GraspCommand:
    """Pixel-space grasp command understood by the existing lab receiver."""

    x: float
    y: float
    theta: float
    width: float
    depth: int

    def validate(self) -> None:
        values: Iterable[float] = (self.x, self.y, self.theta, self.width, self.depth)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Grasp command contains a non-finite value: {self}")
        if self.width <= 0:
            raise ValueError(f"Grasp width must be positive: {self.width}")

    def to_wire(self) -> bytes:
        self.validate()
        fields = (self.x, self.y, self.theta, self.width, self.depth)
        return ("{" + ", ".join(_wire_number(value) for value in fields) + "}\n").encode(
            "ascii"
        )

    def validate_limits(self, limits: Mapping[str, Sequence[float]]) -> None:
        self.validate()
        for field in ("x", "y", "theta", "width", "depth"):
            bounds = limits.get(field)
            if bounds is None or len(bounds) != 2:
                raise ValueError(f"Robot limit {field!r} must contain [minimum, maximum]")
            minimum, maximum = float(bounds[0]), float(bounds[1])
            value = float(getattr(self, field))
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"Grasp command {field}={value:g} is outside [{minimum:g}, {maximum:g}]"
                )


class LegacyTCPGraspClient:
    """Explicit-connect TCP sender; it never sends while merely connecting."""

    def __init__(self, host: str, port: int = 3000, timeout_s: float = 2.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._socket: Optional[socket.socket] = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self.connected:
            return
        connection = socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        )
        connection.settimeout(self.timeout_s)
        self._socket = connection

    def send(self, command: GraspCommand) -> None:
        if not self.connected:
            raise RuntimeError("Robot receiver is not connected")
        try:
            self._socket.sendall(command.to_wire())
        except OSError:
            self.close()
            raise

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


ROBOT_CLIENTS.register_module(
    LegacyTCPGraspClient,
    name="legacy_tcp",
    aliases=("kinova_tcp", "tcp"),
)
ROBOT_CLIENT_REGISTRY = ROBOT_CLIENTS.module_dict


def build_robot_client(cfg: Mapping[str, object]):
    """Build a robot transport without connecting or sending anything."""
    component_type = cfg.get("type", "legacy_tcp")
    try:
        client_class = ROBOT_CLIENTS.require(component_type)
    except KeyError as exc:
        available = ", ".join(sorted(ROBOT_CLIENTS.keys()))
        raise ValueError(
            f"Unknown robot client {component_type!r}; available: {available}"
        ) from exc
    return client_class(
        host=cfg["host"],
        port=cfg.get("port", 3000),
        timeout_s=cfg.get("timeout_s", 2.0),
    )
