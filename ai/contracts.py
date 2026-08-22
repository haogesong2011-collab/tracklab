"""AI result contracts. Models speak only this language to the rest of TrackLab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from ai.schema import POSE_KEYPOINTS

# Display threshold: visible samples below this are marked yellow.
LOW_CONFIDENCE = 0.60


class FailureReason(str, Enum):
    NONE = "none"
    LOW_CONFIDENCE = "low_confidence"
    TARGET_LOST = "target_lost"
    OCCLUSION = "occlusion"
    NO_RULER = "no_ruler"
    AMBIGUOUS_AXIS = "ambiguous_axis"
    NO_PERSON = "no_person"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@dataclass
class TrackPoint:
    frame: int
    x: float
    y: float
    visible: bool = True
    confidence: float = 1.0
    manual: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackPoint":
        return cls(
            frame=int(data["frame"]),
            x=float(data["x"]),
            y=float(data["y"]),
            visible=bool(data.get("visible", True)),
            confidence=float(data.get("confidence", 1.0)),
            manual=bool(data.get("manual", False)),
        )


@dataclass
class TrackResult:
    clip_id: str
    points: list[TrackPoint]
    confidence: float = 1.0
    failure_reason: FailureReason = FailureReason.NONE
    model_name: str = ""
    model_version: str = ""
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_reason"] = self.failure_reason.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackResult":
        return cls(
            clip_id=str(data["clip_id"]),
            points=[TrackPoint.from_dict(p) for p in data.get("points", [])],
            confidence=float(data.get("confidence", 1.0)),
            failure_reason=FailureReason(data.get("failure_reason", "none")),
            model_name=str(data.get("model_name", "")),
            model_version=str(data.get("model_version", "")),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        )


class PromptKind(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    BOX = "box"


TRACK_COLORS = (
    "#f0c14b",
    "#6cb6ff",
    "#7dce82",
    "#e07a5f",
    "#c77dff",
    "#80cbc4",
)


@dataclass
class TrackPrompt:
    """User hint consumed by SAM 2 (point or axis-aligned box)."""

    frame: int
    kind: PromptKind
    x: float
    y: float
    x2: float | None = None
    y2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackPrompt":
        kind = data.get("kind", "positive")
        return cls(
            frame=int(data["frame"]),
            kind=PromptKind(kind),
            x=float(data["x"]),
            y=float(data["y"]),
            x2=None if data.get("x2") is None else float(data["x2"]),
            y2=None if data.get("y2") is None else float(data["y2"]),
        )


@dataclass
class TrackLayer:
    """One named trajectory in a project (may wrap a TrackResult)."""

    track_id: str
    name: str
    color: str = TRACK_COLORS[0]
    visible: bool = True
    seed_frame: int = 0
    prompts: list[TrackPrompt] = field(default_factory=list)
    result: TrackResult | None = None
    contours: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    status: str = "idle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "name": self.name,
            "color": self.color,
            "visible": self.visible,
            "seed_frame": self.seed_frame,
            "prompts": [p.to_dict() for p in self.prompts],
            "result": None if self.result is None else self.result.to_dict(),
            "contours": {
                str(frame): [[float(x), float(y)] for x, y in points]
                for frame, points in self.contours.items()
            },
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackLayer":
        contours: dict[int, list[tuple[float, float]]] = {}
        for key, points in (data.get("contours") or {}).items():
            contours[int(key)] = [(float(p[0]), float(p[1])) for p in points]
        result = data.get("result")
        return cls(
            track_id=str(data["track_id"]),
            name=str(data.get("name") or data["track_id"]),
            color=str(data.get("color", TRACK_COLORS[0])),
            visible=bool(data.get("visible", True)),
            seed_frame=int(data.get("seed_frame", 0)),
            prompts=[TrackPrompt.from_dict(p) for p in data.get("prompts", [])],
            result=None if not result else TrackResult.from_dict(result),
            contours=contours,
            status=str(data.get("status", "idle")),
        )


@dataclass
class PoseKeypoint:
    name: str
    x: float
    y: float
    visible: bool = True
    confidence: float = 1.0


@dataclass
class PoseFrame:
    frame: int
    keypoints: list[PoseKeypoint]
    person_id: int = 0
    confidence: float = 1.0


@dataclass
class PoseResult:
    clip_id: str
    frames: list[PoseFrame]
    confidence: float = 1.0
    failure_reason: FailureReason = FailureReason.NONE
    model_name: str = ""
    model_version: str = ""
    elapsed_s: float = 0.0
    keypoint_names: tuple[str, ...] = POSE_KEYPOINTS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_reason"] = self.failure_reason.value
        data["keypoint_names"] = list(self.keypoint_names)
        return data


@dataclass
class CalibrationResult:
    clip_id: str
    pixels_per_meter: float | None
    origin_x: float | None
    origin_y: float | None
    axis_angle_deg: float | None
    confidence: float = 1.0
    failure_reason: FailureReason = FailureReason.NONE
    model_name: str = ""
    model_version: str = ""
    elapsed_s: float = 0.0
    rejected: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_reason"] = self.failure_reason.value
        return data


@dataclass
class PhysicsResult:
    clip_id: str
    period_s: float | None = None
    gravity_ms2: float | None = None
    velocity_ms: float | None = None
    acceleration_ms2: float | None = None
    trajectory_fit_error: float | None = None
    confidence: float = 1.0
    failure_reason: FailureReason = FailureReason.NONE
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_reason"] = self.failure_reason.value
        return data


@dataclass
class CancelToken:
    """Cooperative cancellation shared between UI and AI workers."""

    _cancelled: bool = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


@dataclass
class ProgressEvent:
    clip_id: str
    current: int
    total: int
    stage: str = "infer"


ProgressCb = Callable[[ProgressEvent], None]
