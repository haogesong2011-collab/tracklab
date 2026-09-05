"""Shared annotation / manifest schemas for TrackLab AI evaluation.

Full-size videos live outside the repo. Only hashes and relative paths are
committed. CI uses the synthetic clips under tests/ai/fixtures/.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "1.0.0"


class TaskKind(str, Enum):
    TRACK = "track"
    POSE = "pose"
    CALIBRATION = "calibration"
    PHYSICS = "physics"


class Difficulty(str, Enum):
    NORMAL = "normal"
    HARD = "hard"


class Scene(str, Enum):
    BALL = "ball"
    PENDULUM = "pendulum"
    SLIDER = "slider"
    PROJECTILE = "projectile"
    TURNTABLE = "turntable"
    JUMP = "jump"
    MIXED = "mixed"
    FREEFALL = "freefall"
    ACCEL = "accel"


# COCO-17 style keypoints used for pose evaluation in physics-education clips.
POSE_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Bone pairs for length-stability checks.
POSE_BONES = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


@dataclass
class Point2D:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class Point3D:
    x: float
    y: float
    z: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> Point2D:
        return Point2D(self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass
class TrackFrameGT:
    frame: int
    center: Point2D
    visible: bool = True
    bbox: BBox | None = None
    occluded: bool = False


@dataclass
class KeypointGT:
    name: str
    x: float
    y: float
    visible: bool = True


@dataclass
class PoseFrameGT:
    frame: int
    keypoints: list[KeypointGT]
    person_id: int = 0


@dataclass
class CalibrationGT:
    ruler_a: Point2D
    ruler_b: Point2D
    length_m: float
    origin: Point2D
    axis_angle_deg: float
    has_reliable_ruler: bool = True


@dataclass
class PhysicsGT:
    """Ground-truth physical quantities derived from the experiment."""

    period_s: float | None = None
    gravity_ms2: float | None = None
    velocity_ms: float | None = None
    acceleration_ms2: float | None = None
    trajectory_fit_error: float | None = None


@dataclass
class ClipAnnotation:
    clip_id: str
    schema_version: str = SCHEMA_VERSION
    task: TaskKind = TaskKind.TRACK
    scene: Scene = Scene.BALL
    difficulty: Difficulty = Difficulty.NORMAL
    width: int = 640
    height: int = 360
    fps: float = 30.0
    frame_count: int = 0
    track: list[TrackFrameGT] = field(default_factory=list)
    pose: list[PoseFrameGT] = field(default_factory=list)
    calibration: CalibrationGT | None = None
    physics: PhysicsGT | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ManifestEntry:
    clip_id: str
    relative_path: str
    annotation_path: str
    sha256: str
    task: TaskKind
    scene: Scene
    difficulty: Difficulty
    width: int
    height: int
    fps: float
    frame_count: int
    split: str  # "ci" | "dev" | "holdout"
    license: str = "synthetic-internal"
    hidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task"] = self.task.value
        data["scene"] = self.scene.value
        data["difficulty"] = self.difficulty.value
        return data


@dataclass
class Manifest:
    schema_version: str
    description: str
    entries: list[ManifestEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "description": self.description,
            "entries": [e.to_dict() for e in self.entries],
        }
