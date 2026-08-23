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


class ExperimentType(str, Enum):
    UNIFORM_LINEAR = "uniform_linear"
    UNIFORM_ACCEL = "uniform_accel"
    FREE_FALL = "free_fall"
    PROJECTILE = "projectile"
    PENDULUM = "pendulum"
    UNKNOWN = "unknown"


class TeachingLevel(str, Enum):
    JUNIOR = "junior"
    HIGH = "high"
    COLLEGE = "college"


EXPERIMENT_LABELS = {
    ExperimentType.UNIFORM_LINEAR: "匀速直线运动",
    ExperimentType.UNIFORM_ACCEL: "匀加速直线运动",
    ExperimentType.FREE_FALL: "自由落体",
    ExperimentType.PROJECTILE: "平抛 / 斜抛",
    ExperimentType.PENDULUM: "单摆",
    ExperimentType.UNKNOWN: "无法可靠识别",
}

TEACHING_LEVEL_LABELS = {
    TeachingLevel.JUNIOR: "初中",
    TeachingLevel.HIGH: "高中",
    TeachingLevel.COLLEGE: "大学基础",
}


@dataclass
class FitResult:
    model: str
    formula_id: str
    frame_start: int
    frame_end: int
    time_start_s: float
    time_end_s: float
    parameters: dict[str, float | None] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    r2: float = 0.0
    nrmse: float = 1.0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FitResult | None":
        if not data:
            return None
        params = {
            str(key): None if val is None else float(val)
            for key, val in (data.get("parameters") or {}).items()
        }
        units = {str(key): str(val) for key, val in (data.get("units") or {}).items()}
        return cls(
            model=str(data.get("model", "")),
            formula_id=str(data.get("formula_id", "")),
            frame_start=int(data.get("frame_start", 0)),
            frame_end=int(data.get("frame_end", 0)),
            time_start_s=float(data.get("time_start_s", 0.0)),
            time_end_s=float(data.get("time_end_s", 0.0)),
            parameters=params,
            units=units,
            r2=float(data.get("r2", 0.0)),
            nrmse=float(data.get("nrmse", 1.0)),
            n_samples=int(data.get("n_samples", 0)),
        )


@dataclass
class ExperimentCandidate:
    experiment_type: ExperimentType
    label: str
    confidence: float
    fit: FitResult
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_type": self.experiment_type.value,
            "label": self.label,
            "confidence": self.confidence,
            "fit": self.fit.to_dict(),
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "missing": list(self.missing),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExperimentCandidate | None":
        if not data:
            return None
        fit = FitResult.from_dict(data.get("fit") or {})
        if fit is None:
            return None
        raw_type = str(data.get("experiment_type", "unknown"))
        try:
            kind = ExperimentType(raw_type)
        except ValueError:
            kind = ExperimentType.UNKNOWN
        return cls(
            experiment_type=kind,
            label=str(data.get("label") or EXPERIMENT_LABELS.get(kind, kind.value)),
            confidence=float(data.get("confidence", 0.0)),
            fit=fit,
            evidence=[str(item) for item in data.get("evidence") or []],
            warnings=[str(item) for item in data.get("warnings") or []],
            missing=[str(item) for item in data.get("missing") or []],
        )


@dataclass
class ExperimentAnalysis:
    clip_id: str
    candidates: list[ExperimentCandidate] = field(default_factory=list)
    selected: ExperimentCandidate | None = None
    auto_confirmable: bool = False
    coverage: float = 0.0
    mean_track_confidence: float = 0.0
    calibration_active: bool = False
    position_unit: str = "px"
    speed_unit: str = "px/s"
    fingerprint: str = ""
    warnings: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    source: str = "local_physics"

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "candidates": [item.to_dict() for item in self.candidates],
            "selected": None if self.selected is None else self.selected.to_dict(),
            "auto_confirmable": self.auto_confirmable,
            "coverage": self.coverage,
            "mean_track_confidence": self.mean_track_confidence,
            "calibration_active": self.calibration_active,
            "position_unit": self.position_unit,
            "speed_unit": self.speed_unit,
            "fingerprint": self.fingerprint,
            "warnings": list(self.warnings),
            "missing": list(self.missing),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExperimentAnalysis | None":
        if not data:
            return None
        candidates = [
            item
            for item in (ExperimentCandidate.from_dict(raw) for raw in data.get("candidates") or [])
            if item is not None
        ]
        return cls(
            clip_id=str(data.get("clip_id", "")),
            candidates=candidates,
            selected=ExperimentCandidate.from_dict(data.get("selected")),
            auto_confirmable=bool(data.get("auto_confirmable", False)),
            coverage=float(data.get("coverage", 0.0)),
            mean_track_confidence=float(data.get("mean_track_confidence", 0.0)),
            calibration_active=bool(data.get("calibration_active", False)),
            position_unit=str(data.get("position_unit", "px")),
            speed_unit=str(data.get("speed_unit", "px/s")),
            fingerprint=str(data.get("fingerprint", "")),
            warnings=[str(item) for item in data.get("warnings") or []],
            missing=[str(item) for item in data.get("missing") or []],
            source=str(data.get("source", "local_physics")),
        )


@dataclass
class ChatMessage:
    role: str
    content: str
    cancelled: bool = False
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = {"role": self.role, "content": self.content, "cancelled": self.cancelled}
        if self.reasoning:
            data["reasoning"] = self.reasoning
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "")),
            cancelled=bool(data.get("cancelled", False)),
            reasoning=str(data.get("reasoning", "")),
        )


@dataclass
class AssistantState:
    analysis: ExperimentAnalysis | None = None
    confirmed_type: ExperimentType | None = None
    fingerprint: str = ""
    stale: bool = False
    messages: list[ChatMessage] = field(default_factory=list)
    report_markdown: str = ""
    report_sections: dict[str, str] = field(default_factory=dict)
    model_id: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""
    teaching_level: TeachingLevel = TeachingLevel.HIGH
    pendulum_length_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        messages = self.messages[-50:]
        return {
            "analysis": None if self.analysis is None else self.analysis.to_dict(),
            "confirmed_type": None
            if self.confirmed_type is None
            else self.confirmed_type.value,
            "fingerprint": self.fingerprint,
            "stale": self.stale,
            "messages": [item.to_dict() for item in messages],
            "report_markdown": self.report_markdown,
            "report_sections": dict(self.report_sections),
            "model_id": self.model_id,
            "token_usage": dict(self.token_usage),
            "generated_at": self.generated_at,
            "teaching_level": self.teaching_level.value,
            "pendulum_length_m": self.pendulum_length_m,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AssistantState":
        if not data:
            return cls()
        raw_type = data.get("confirmed_type")
        confirmed = None
        if raw_type:
            try:
                confirmed = ExperimentType(str(raw_type))
            except ValueError:
                confirmed = ExperimentType.UNKNOWN
        raw_level = str(data.get("teaching_level") or TeachingLevel.HIGH.value)
        try:
            level = TeachingLevel(raw_level)
        except ValueError:
            level = TeachingLevel.HIGH
        usage_raw = data.get("token_usage") or {}
        usage = {str(key): int(val) for key, val in usage_raw.items()}
        length = data.get("pendulum_length_m")
        return cls(
            analysis=ExperimentAnalysis.from_dict(data.get("analysis")),
            confirmed_type=confirmed,
            fingerprint=str(data.get("fingerprint", "")),
            stale=bool(data.get("stale", False)),
            messages=[ChatMessage.from_dict(item) for item in data.get("messages") or []][-50:],
            report_markdown=str(data.get("report_markdown", "")),
            report_sections={
                str(key): str(val) for key, val in (data.get("report_sections") or {}).items()
            },
            model_id=str(data.get("model_id", "")),
            token_usage=usage,
            generated_at=str(data.get("generated_at", "")),
            teaching_level=level,
            pendulum_length_m=None if length is None else float(length),
        )


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
