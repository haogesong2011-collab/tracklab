"""Load manifests and annotations from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai.schema import (
    BBox,
    CalibrationGT,
    ClipAnnotation,
    Difficulty,
    KeypointGT,
    Manifest,
    ManifestEntry,
    PhysicsGT,
    Point2D,
    PoseFrameGT,
    Scene,
    TaskKind,
    TrackFrameGT,
)

ROOT = Path(__file__).resolve().parents[2]


def _point(data: dict[str, Any] | None) -> Point2D | None:
    if not data:
        return None
    return Point2D(float(data["x"]), float(data["y"]))


def _bbox(data: dict[str, Any] | None) -> BBox | None:
    if not data:
        return None
    return BBox(float(data["x"]), float(data["y"]), float(data["w"]), float(data["h"]))


def load_annotation(path: str | Path) -> ClipAnnotation:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    raw = json.loads(path.read_text(encoding="utf-8"))
    track = [
        TrackFrameGT(
            frame=int(f["frame"]),
            center=Point2D(float(f["center"]["x"]), float(f["center"]["y"])),
            visible=bool(f.get("visible", True)),
            bbox=_bbox(f.get("bbox")),
            occluded=bool(f.get("occluded", False)),
        )
        for f in raw.get("track", [])
    ]
    pose = [
        PoseFrameGT(
            frame=int(f["frame"]),
            person_id=int(f.get("person_id", 0)),
            keypoints=[
                KeypointGT(
                    name=k["name"],
                    x=float(k["x"]),
                    y=float(k["y"]),
                    visible=bool(k.get("visible", True)),
                )
                for k in f.get("keypoints", [])
            ],
        )
        for f in raw.get("pose", [])
    ]
    calib = None
    if raw.get("calibration"):
        c = raw["calibration"]
        calib = CalibrationGT(
            ruler_a=Point2D(float(c["ruler_a"]["x"]), float(c["ruler_a"]["y"])),
            ruler_b=Point2D(float(c["ruler_b"]["x"]), float(c["ruler_b"]["y"])),
            length_m=float(c["length_m"]),
            origin=Point2D(float(c["origin"]["x"]), float(c["origin"]["y"])),
            axis_angle_deg=float(c["axis_angle_deg"]),
            has_reliable_ruler=bool(c.get("has_reliable_ruler", True)),
        )
    physics = None
    if raw.get("physics"):
        p = raw["physics"]
        physics = PhysicsGT(
            period_s=p.get("period_s"),
            gravity_ms2=p.get("gravity_ms2"),
            velocity_ms=p.get("velocity_ms"),
            acceleration_ms2=p.get("acceleration_ms2"),
            trajectory_fit_error=p.get("trajectory_fit_error"),
        )
    return ClipAnnotation(
        clip_id=raw["clip_id"],
        schema_version=raw.get("schema_version", "1.0.0"),
        task=TaskKind(raw["task"]),
        scene=Scene(raw["scene"]),
        difficulty=Difficulty(raw["difficulty"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        fps=float(raw["fps"]),
        frame_count=int(raw["frame_count"]),
        track=track,
        pose=pose,
        calibration=calib,
        physics=physics,
        notes=raw.get("notes", ""),
    )


def load_manifest(path: str | Path | None = None) -> Manifest:
    path = Path(path) if path else ROOT / "datasets" / "manifest.json"
    if not path.is_absolute():
        path = ROOT / path
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = [
        ManifestEntry(
            clip_id=e["clip_id"],
            relative_path=e["relative_path"],
            annotation_path=e["annotation_path"],
            sha256=e["sha256"],
            task=TaskKind(e["task"]),
            scene=Scene(e["scene"]),
            difficulty=Difficulty(e["difficulty"]),
            width=int(e["width"]),
            height=int(e["height"]),
            fps=float(e["fps"]),
            frame_count=int(e["frame_count"]),
            split=e.get("split", "dev"),
            license=e.get("license", "unknown"),
            hidden=bool(e.get("hidden", False)),
        )
        for e in raw["entries"]
    ]
    return Manifest(
        schema_version=raw["schema_version"],
        description=raw.get("description", ""),
        entries=entries,
    )


def resolve_video(entry: ManifestEntry) -> Path:
    path = ROOT / entry.relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path
