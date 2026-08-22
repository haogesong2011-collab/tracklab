"""TrackLab AI package: contracts, schema, and model adapters."""

from ai.contracts import (
    CalibrationResult,
    CancelToken,
    FailureReason,
    PhysicsResult,
    PoseResult,
    ProgressEvent,
    TrackResult,
)
from ai.schema import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "CalibrationResult",
    "CancelToken",
    "FailureReason",
    "PhysicsResult",
    "PoseResult",
    "ProgressEvent",
    "TrackResult",
]
