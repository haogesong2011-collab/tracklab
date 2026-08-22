"""Generate synthetic physics-experiment clips + annotations for CI fixtures."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np

from ai.schema import (
    SCHEMA_VERSION,
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
    POSE_KEYPOINTS,
    Scene,
    TaskKind,
    TrackFrameGT,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "ai" / "fixtures"
ANNOTATIONS = FIXTURES / "annotations"
MANIFEST_PATH = ROOT / "datasets" / "manifest.json"

WIDTH, HEIGHT, FPS = 320, 180, 30.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_video(path: Path, frames: list[np.ndarray], fps: float = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23", "preset": "ultrafast"}
    for array in frames:
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _blank(color: tuple[int, int, int] = (24, 26, 30)) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = color
    return img


def _draw_disk(img: np.ndarray, cx: float, cy: float, r: int, color: tuple[int, int, int]) -> None:
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[mask] = color


def _draw_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(WIDTH, x + w), min(HEIGHT, y + h)
    img[y0:y1, x0:x1] = color


def _draw_line(
    img: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    steps = max(int(math.hypot(x1 - x0, y1 - y0)), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        _draw_disk(img, x, y, thickness, color)


def ball_track(n: int = 45, hard: bool = False) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames: list[np.ndarray] = []
    track: list[TrackFrameGT] = []
    for i in range(n):
        bg = (18, 18, 18) if hard else (30, 34, 40)
        img = _blank(bg)
        if hard and 18 <= i <= 24:
            # Occlusion bar.
            _draw_rect(img, 0, 70, WIDTH, 40, (18, 18, 18))
            occluded = True
            visible = False
            cx = 40 + i * 5.0
            cy = 90.0
        else:
            occluded = False
            visible = True
            cx = 40 + i * 5.0
            cy = 90 + (12 if hard else 0) * math.sin(i / 4.0)
            color = (200, 200, 200) if hard else (220, 80, 60)
            _draw_disk(img, cx, cy, 8 if hard else 10, color)
            if hard:
                # Similar distractor.
                _draw_disk(img, cx + 40, cy - 20, 8, (180, 180, 180))
        frames.append(img)
        track.append(
            TrackFrameGT(
                frame=i,
                center=Point2D(cx, cy),
                visible=visible,
                occluded=occluded,
                bbox=BBox(cx - 10, cy - 10, 20, 20),
            )
        )
    ann = ClipAnnotation(
        clip_id="track_ball_hard" if hard else "track_ball_normal",
        task=TaskKind.TRACK,
        scene=Scene.BALL,
        difficulty=Difficulty.HARD if hard else Difficulty.NORMAL,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=n,
        track=track,
        notes="synthetic translating ball",
    )
    return frames, ann


def pendulum_track(n: int = 60) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames: list[np.ndarray] = []
    track: list[TrackFrameGT] = []
    origin = Point2D(WIDTH / 2, 20)
    length_px = 100.0
    period = 1.2
    omega = 2 * math.pi / period
    amp = math.radians(25)
    for i in range(n):
        t = i / FPS
        theta = amp * math.cos(omega * t)
        cx = origin.x + length_px * math.sin(theta)
        cy = origin.y + length_px * math.cos(theta)
        img = _blank((28, 30, 36))
        _draw_line(img, origin.x, origin.y, cx, cy, (160, 160, 160), 1)
        _draw_disk(img, cx, cy, 9, (70, 160, 220))
        frames.append(img)
        track.append(
            TrackFrameGT(frame=i, center=Point2D(cx, cy), bbox=BBox(cx - 9, cy - 9, 18, 18))
        )
    g = 9.81
    # T = 2π√(L/g) with L derived from period for GT consistency.
    L = (period / (2 * math.pi)) ** 2 * g
    ann = ClipAnnotation(
        clip_id="track_pendulum_normal",
        task=TaskKind.TRACK,
        scene=Scene.PENDULUM,
        difficulty=Difficulty.NORMAL,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=n,
        track=track,
        physics=PhysicsGT(period_s=period, gravity_ms2=g),
        calibration=CalibrationGT(
            ruler_a=Point2D(20, HEIGHT - 20),
            ruler_b=Point2D(20 + length_px, HEIGHT - 20),
            length_m=L,
            origin=origin,
            axis_angle_deg=0.0,
        ),
        notes="synthetic pendulum",
    )
    return frames, ann


def slider_track(n: int = 40) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames: list[np.ndarray] = []
    track: list[TrackFrameGT] = []
    for i in range(n):
        img = _blank((32, 32, 38))
        _draw_rect(img, 20, 110, WIDTH - 40, 8, (90, 90, 90))
        cx = 30 + i * 6.0
        cy = 106.0
        _draw_rect(img, int(cx - 12), int(cy - 8), 24, 16, (240, 180, 60))
        frames.append(img)
        track.append(
            TrackFrameGT(frame=i, center=Point2D(cx, cy), bbox=BBox(cx - 12, cy - 8, 24, 16))
        )
    ann = ClipAnnotation(
        clip_id="track_slider_normal",
        task=TaskKind.TRACK,
        scene=Scene.SLIDER,
        difficulty=Difficulty.NORMAL,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=n,
        track=track,
        physics=PhysicsGT(velocity_ms=1.5, acceleration_ms2=0.0),
    )
    return frames, ann


def projectile_track(n: int = 50) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames: list[np.ndarray] = []
    track: list[TrackFrameGT] = []
    x0, y0 = 20.0, 140.0
    vx, vy = 80.0, -120.0  # px/s
    g_px = 180.0
    for i in range(n):
        t = i / FPS
        cx = x0 + vx * t
        cy = y0 + vy * t + 0.5 * g_px * t * t
        img = _blank((26, 28, 32))
        if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
            _draw_disk(img, cx, cy, 7, (90, 220, 120))
            visible = True
        else:
            visible = False
        frames.append(img)
        track.append(TrackFrameGT(frame=i, center=Point2D(cx, cy), visible=visible))
    ann = ClipAnnotation(
        clip_id="track_projectile_normal",
        task=TaskKind.TRACK,
        scene=Scene.PROJECTILE,
        difficulty=Difficulty.NORMAL,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=n,
        track=track,
        physics=PhysicsGT(gravity_ms2=9.81, trajectory_fit_error=0.0),
        calibration=CalibrationGT(
            ruler_a=Point2D(10, 160),
            ruler_b=Point2D(10 + 50, 160),
            length_m=0.5,
            origin=Point2D(20, 140),
            axis_angle_deg=0.0,
        ),
    )
    return frames, ann


def calibration_clip(reliable: bool = True) -> tuple[list[np.ndarray], ClipAnnotation]:
    img = _blank((40, 42, 48))
    a, b = Point2D(40, 140), Point2D(200, 140)
    if reliable:
        _draw_line(img, a.x, a.y, b.x, b.y, (250, 250, 250), 2)
        _draw_disk(img, a.x, a.y, 3, (250, 80, 80))
        _draw_disk(img, b.x, b.y, 3, (80, 250, 80))
    else:
        # No clear ruler — noisy background only.
        noise = np.random.default_rng(0).integers(30, 60, size=img.shape, dtype=np.uint8)
        img[:] = noise
    frames = [img.copy() for _ in range(8)]
    ann = ClipAnnotation(
        clip_id="calib_reliable" if reliable else "calib_no_ruler",
        task=TaskKind.CALIBRATION,
        scene=Scene.MIXED,
        difficulty=Difficulty.NORMAL if reliable else Difficulty.HARD,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=len(frames),
        calibration=CalibrationGT(
            ruler_a=a,
            ruler_b=b,
            length_m=0.8,
            origin=Point2D(40, 40),
            axis_angle_deg=0.0,
            has_reliable_ruler=reliable,
        ),
    )
    return frames, ann


def pose_clip(hard: bool = False, n: int = 30) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames: list[np.ndarray] = []
    poses: list[PoseFrameGT] = []
    # Stick-figure person walking / jumping.
    for i in range(n):
        img = _blank((22, 22, 26) if hard else (34, 36, 42))
        base_x = 80 + i * 3
        base_y = 50 + (10 * math.sin(i / 3.0) if hard else 0)
        if hard and 10 <= i <= 14:
            _draw_rect(img, 0, 40, WIDTH, 80, (22, 22, 26))
            visible = False
        else:
            visible = True
        kps: list[KeypointGT] = []
        coords = {
            "nose": (base_x, base_y),
            "left_eye": (base_x - 4, base_y - 2),
            "right_eye": (base_x + 4, base_y - 2),
            "left_ear": (base_x - 8, base_y),
            "right_ear": (base_x + 8, base_y),
            "left_shoulder": (base_x - 18, base_y + 20),
            "right_shoulder": (base_x + 18, base_y + 20),
            "left_elbow": (base_x - 28, base_y + 40),
            "right_elbow": (base_x + 28, base_y + 40),
            "left_wrist": (base_x - 34, base_y + 58),
            "right_wrist": (base_x + 34, base_y + 58),
            "left_hip": (base_x - 12, base_y + 60),
            "right_hip": (base_x + 12, base_y + 60),
            "left_knee": (base_x - 14, base_y + 90),
            "right_knee": (base_x + 14, base_y + 90),
            "left_ankle": (base_x - 16, base_y + 120),
            "right_ankle": (base_x + 16, base_y + 120),
        }
        for name in POSE_KEYPOINTS:
            x, y = coords[name]
            if visible:
                _draw_disk(img, x, y, 2, (230, 230, 230))
            kps.append(KeypointGT(name=name, x=float(x), y=float(y), visible=visible))
        if visible:
            _draw_line(
                img,
                coords["left_shoulder"][0],
                coords["left_shoulder"][1],
                coords["right_shoulder"][0],
                coords["right_shoulder"][1],
                (180, 180, 180),
                1,
            )
        frames.append(img)
        # Annotate every frame for CI; production pose GT is every 5 frames.
        poses.append(PoseFrameGT(frame=i, keypoints=kps))
    ann = ClipAnnotation(
        clip_id="pose_jump_hard" if hard else "pose_walk_normal",
        task=TaskKind.POSE,
        scene=Scene.JUMP if hard else Scene.MIXED,
        difficulty=Difficulty.HARD if hard else Difficulty.NORMAL,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=n,
        pose=poses,
    )
    return frames, ann


def _box_blur(img: np.ndarray) -> np.ndarray:
    padded = np.pad(img.astype(np.uint16), ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return (acc // 9).astype(np.uint8)


def turntable_track(n: int = 48) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames: list[np.ndarray] = []
    track: list[TrackFrameGT] = []
    cx0, cy0, radius = WIDTH / 2, HEIGHT / 2, 50.0
    for i in range(n):
        theta = i / n * 2 * math.pi
        cx = cx0 + radius * math.cos(theta)
        cy = cy0 + radius * math.sin(theta)
        img = _blank((26, 28, 32))
        _draw_disk(img, cx0, cy0, 4, (90, 90, 90))
        _draw_disk(img, cx, cy, 8, (220, 90, 40))
        frames.append(img)
        track.append(TrackFrameGT(frame=i, center=Point2D(cx, cy), bbox=BBox(cx - 8, cy - 8, 16, 16)))
    ann = ClipAnnotation(
        clip_id="track_turntable_normal",
        task=TaskKind.TRACK,
        scene=Scene.TURNTABLE,
        difficulty=Difficulty.NORMAL,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=n,
        track=track,
        notes="synthetic rotating marker",
    )
    return frames, ann


def ball_blur_hard(n: int = 40) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames, ann = ball_track(n, hard=False)
    frames = [_box_blur(_box_blur(f)) for f in frames]
    ann.clip_id = "track_ball_blur_hard"
    ann.difficulty = Difficulty.HARD
    ann.notes = "motion-blurred translating ball"
    return frames, ann


def pendulum_shake_hard(n: int = 48) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames, ann = pendulum_track(n)
    rng = np.random.default_rng(1)
    shaken: list[np.ndarray] = []
    for i, img in enumerate(frames):
        dx, dy = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
        shaken.append(np.roll(np.roll(img, dy, axis=0), dx, axis=1))
        pt = ann.track[i]
        pt.center = Point2D(pt.center.x + dx, pt.center.y + dy)
    ann.clip_id = "track_pendulum_shake_hard"
    ann.difficulty = Difficulty.HARD
    ann.notes = "camera shake"
    return shaken, ann


def ball_15fps(n: int = 30) -> tuple[list[np.ndarray], ClipAnnotation]:
    frames, ann = ball_track(n, hard=False)
    ann.clip_id = "track_ball_15fps_hard"
    ann.fps = 15.0
    ann.difficulty = Difficulty.HARD
    ann.notes = "15 fps capture"
    return frames, ann


def calib_oblique() -> tuple[list[np.ndarray], ClipAnnotation]:
    img = _blank((40, 42, 48))
    a, b = Point2D(50, 150), Point2D(210, 95)
    _draw_line(img, a.x, a.y, b.x, b.y, (250, 250, 250), 2)
    _draw_disk(img, a.x, a.y, 3, (250, 80, 80))
    _draw_disk(img, b.x, b.y, 3, (80, 250, 80))
    frames = [img.copy() for _ in range(8)]
    angle = math.degrees(math.atan2(-(b.y - a.y), b.x - a.x))
    ann = ClipAnnotation(
        clip_id="calib_oblique_hard",
        task=TaskKind.CALIBRATION,
        scene=Scene.MIXED,
        difficulty=Difficulty.HARD,
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        frame_count=len(frames),
        calibration=CalibrationGT(
            ruler_a=a,
            ruler_b=b,
            length_m=0.8,
            origin=Point2D(50, 40),
            axis_angle_deg=angle,
            has_reliable_ruler=True,
        ),
        notes="oblique ruler",
    )
    return frames, ann


def pose_sparse() -> tuple[list[np.ndarray], ClipAnnotation]:
    frames, ann = pose_clip(hard=False, n=30)
    dense = {f.frame: f for f in ann.pose}
    sparse = []
    for i in range(ann.frame_count):
        if i % 5 == 0 or (
            i in dense and not all(k.visible for k in dense[i].keypoints)
        ):
            if i in dense:
                sparse.append(dense[i])
    ann.pose = sparse
    ann.clip_id = "pose_walk_sparse"
    ann.notes = "pose labelled every 5 frames"
    return frames, ann


def physics_pendulum_clip() -> tuple[list[np.ndarray], ClipAnnotation]:
    frames, ann = pendulum_track(60)
    ann.clip_id = "physics_pendulum"
    ann.task = TaskKind.PHYSICS
    return frames, ann


CLIP_BUILDERS = [
    ("track_ball_normal.mp4", lambda: ball_track(hard=False)),
    ("track_ball_hard.mp4", lambda: ball_track(hard=True)),
    ("track_pendulum_normal.mp4", lambda: pendulum_track()),
    ("track_slider_normal.mp4", lambda: slider_track()),
    ("track_projectile_normal.mp4", lambda: projectile_track()),
    ("calib_reliable.mp4", lambda: calibration_clip(True)),
    ("calib_no_ruler.mp4", lambda: calibration_clip(False)),
    ("pose_walk_normal.mp4", lambda: pose_clip(hard=False)),
    ("pose_jump_hard.mp4", lambda: pose_clip(hard=True)),
    ("physics_pendulum.mp4", physics_pendulum_clip),
]

# Unique extra clips for the 60-slot set. Not part of the 10-clip CI freeze.
EXTRA_BUILDERS = [
    ("track_turntable_normal.mp4", turntable_track),
    ("track_ball_blur_hard.mp4", ball_blur_hard),
    ("track_pendulum_shake_hard.mp4", pendulum_shake_hard),
    ("track_ball_15fps_hard.mp4", ball_15fps),
    ("calib_oblique_hard.mp4", calib_oblique),
    ("pose_walk_sparse.mp4", pose_sparse),
]


def build_fixtures(holdout_ratio: float = 0.2) -> Manifest:
    """Write the frozen 10-clip CI set and a 60-slot expanded manifest."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    (ROOT / "datasets").mkdir(parents=True, exist_ok=True)

    entries: list[ManifestEntry] = []
    for filename, builder in CLIP_BUILDERS:
        frames, ann = builder()
        video_path = FIXTURES / filename
        _write_video(video_path, frames, ann.fps)
        ann_path = ANNOTATIONS / f"{ann.clip_id}.json"
        ann_path.write_text(json.dumps(ann.to_dict(), indent=2), encoding="utf-8")
        digest = _sha256(video_path)
        entries.append(
            ManifestEntry(
                clip_id=ann.clip_id,
                relative_path=str(video_path.relative_to(ROOT)),
                annotation_path=str(ann_path.relative_to(ROOT)),
                sha256=digest,
                task=ann.task,
                scene=ann.scene,
                difficulty=ann.difficulty,
                width=ann.width,
                height=ann.height,
                fps=ann.fps,
                frame_count=ann.frame_count,
                split="ci",
                hidden=False,
            )
        )

    extra_entries: list[ManifestEntry] = []
    for filename, builder in EXTRA_BUILDERS:
        frames, ann = builder()
        video_path = FIXTURES / filename
        _write_video(video_path, frames, ann.fps)
        ann_path = ANNOTATIONS / f"{ann.clip_id}.json"
        ann_path.write_text(json.dumps(ann.to_dict(), indent=2), encoding="utf-8")
        extra_entries.append(
            ManifestEntry(
                clip_id=ann.clip_id,
                relative_path=str(video_path.relative_to(ROOT)),
                annotation_path=str(ann_path.relative_to(ROOT)),
                sha256=_sha256(video_path),
                task=ann.task,
                scene=ann.scene,
                difficulty=ann.difficulty,
                width=ann.width,
                height=ann.height,
                fps=ann.fps,
                frame_count=ann.frame_count,
                split="dev",
                hidden=False,
            )
        )

    # Expand to 60 slots: 30 track + 15 calibration + 15 pose.
    unique = list(entries) + extra_entries
    track_pool = [e for e in unique if e.task in (TaskKind.TRACK, TaskKind.PHYSICS)]
    calib_pool = [e for e in unique if e.task == TaskKind.CALIBRATION]
    pose_pool = [e for e in unique if e.task == TaskKind.POSE]
    quotas = {
        "track": (30, track_pool),
        "calib": (15, calib_pool),
        "pose": (15, pose_pool),
    }
    expanded = list(unique)
    slot = len(entries) + 1
    for _label, (target, pool) in quotas.items():
        current = sum(
            1
            for e in expanded
            if (e.task in (TaskKind.TRACK, TaskKind.PHYSICS) and _label == "track")
            or (e.task == TaskKind.CALIBRATION and _label == "calib")
            or (e.task == TaskKind.POSE and _label == "pose")
        )
        i = 0
        while current < target:
            src = pool[i % len(pool)]
            expanded.append(
                ManifestEntry(
                    clip_id=f"{src.clip_id}_slot{slot:02d}",
                    relative_path=src.relative_path,
                    annotation_path=src.annotation_path,
                    sha256=src.sha256,
                    task=src.task,
                    scene=src.scene,
                    difficulty=src.difficulty,
                    width=src.width,
                    height=src.height,
                    fps=src.fps,
                    frame_count=src.frame_count,
                    split="dev",
                    hidden=False,
                    license="synthetic-internal",
                )
            )
            slot += 1
            current += 1
            i += 1

    holdout_n = max(1, int(round(len(expanded) * holdout_ratio)))
    extras = [e for e in expanded if e.split != "ci"]
    by_task: dict[str, list[ManifestEntry]] = {}
    for entry in extras:
        by_task.setdefault(entry.task.value, []).append(entry)
    hidden: list[ManifestEntry] = []
    while len(hidden) < holdout_n:
        progressed = False
        for group in by_task.values():
            if not group:
                continue
            hidden.append(group.pop())
            progressed = True
            if len(hidden) >= holdout_n:
                break
        if not progressed:
            break
    for entry in hidden:
        entry.split = "holdout"
        entry.hidden = True

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        description=(
            "TrackLab AI evaluation set. First 10 entries are frozen synthetic CI "
            "clips. Remaining slots reserve the 60-clip full set; replace relative "
            "paths with real footage when available. Holdout clips must not be used "
            "for model selection."
        ),
        entries=expanded,
    )
    MANIFEST_PATH.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build_fixtures()
    print(f"wrote {len(m.entries)} manifest entries; fixtures in {FIXTURES}")
