"""Validate annotation JSON against the frozen schema."""

from __future__ import annotations

from ai.schema import ClipAnnotation, TaskKind
from tests.ai.dataset import load_annotation, load_manifest


class AnnotationError(ValueError):
    pass


def validate_annotation(ann: ClipAnnotation) -> list[str]:
    errors: list[str] = []
    if ann.frame_count <= 0:
        errors.append("frame_count must be positive")
    if ann.width <= 0 or ann.height <= 0:
        errors.append("width/height must be positive")
    if ann.task == TaskKind.TRACK and not ann.track:
        errors.append("track annotation missing")
    if ann.task == TaskKind.POSE and not ann.pose:
        errors.append("pose annotation missing")
    if ann.task == TaskKind.CALIBRATION and ann.calibration is None:
        errors.append("calibration annotation missing")
    if ann.task == TaskKind.PHYSICS and (not ann.track or ann.physics is None):
        errors.append("physics annotation missing track or quantities")
    for frame in ann.track:
        if not (0 <= frame.frame < ann.frame_count):
            errors.append(f"track frame {frame.frame} out of range")
    for frame in ann.pose:
        if not (0 <= frame.frame < ann.frame_count):
            errors.append(f"pose frame {frame.frame} out of range")
    return errors


def validate_manifest() -> list[str]:
    errors: list[str] = []
    manifest = load_manifest()
    seen: set[str] = set()
    for entry in manifest.entries:
        if entry.clip_id in seen:
            errors.append(f"duplicate clip_id {entry.clip_id}")
        seen.add(entry.clip_id)
        try:
            ann = load_annotation(entry.annotation_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{entry.clip_id}: cannot load annotation ({exc})")
            continue
        errors.extend(f"{entry.clip_id}: {msg}" for msg in validate_annotation(ann))
    return errors
