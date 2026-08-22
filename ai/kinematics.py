"""Per-frame kinematics derived from TrackResult + VideoInfo PTS.

Display-only: never written into project JSON or SAM TrackResult.
Uncalibrated units are px / px/s; after a ruler they switch to m / m/s.
Velocity is differenced in the already-transformed display frame.
Default analysis frame is +x right, +y down so free-fall vy is positive.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai.calibration import CalibrationState
from ai.contracts import LOW_CONFIDENCE, TrackPoint, TrackResult
from engine.video_index import VideoInfo


@dataclass(frozen=True)
class KinematicSample:
    frame: int
    time_s: float
    x: float | None
    y: float | None
    vx: float | None
    vy: float | None
    speed: float | None
    visible: bool
    confidence: float
    manual: bool
    position_unit: str = "px"
    speed_unit: str = "px/s"

    @property
    def x_px(self) -> float | None:
        return self.x

    @property
    def y_px(self) -> float | None:
        return self.y

    @property
    def vx_px_s(self) -> float | None:
        return self.vx

    @property
    def vy_px_s(self) -> float | None:
        return self.vy

    @property
    def speed_px_s(self) -> float | None:
        return self.speed


def is_low_confidence(sample: KinematicSample) -> bool:
    return sample.visible and sample.confidence < LOW_CONFIDENCE


def time_s_for_frame(info: VideoInfo | None, frame: int) -> float:
    if info is not None and 0 <= frame < len(info.pts_ms):
        return info.pts_ms[frame] / 1000.0
    fps = 30.0 if info is None else max(info.fps, 1e-6)
    return frame / fps


def series_for_result(
    result: TrackResult | None,
    info: VideoInfo | None,
    *,
    calibration: CalibrationState | None = None,
) -> list[KinematicSample]:
    if result is None:
        return []
    cal = calibration or CalibrationState()
    position_unit = cal.position_unit
    speed_unit = cal.speed_unit
    points = sorted(result.points, key=lambda p: p.frame)
    display: list[tuple[float | None, float | None]] = [
        _display_xy(point, cal) if point.visible else (None, None) for point in points
    ]
    samples: list[KinematicSample] = []
    for i, point in enumerate(points):
        time_s = time_s_for_frame(info, point.frame)
        x, y = display[i]
        vx = vy = speed = None
        if point.visible and x is not None and y is not None:
            prev_xy = display[i - 1] if i > 0 and points[i - 1].visible else None
            nxt_xy = (
                display[i + 1]
                if i + 1 < len(points) and points[i + 1].visible
                else None
            )
            prev_pt = points[i - 1] if prev_xy is not None else None
            nxt_pt = points[i + 1] if nxt_xy is not None else None
            vx, vy = _velocity(x, y, point, prev_pt, prev_xy, nxt_pt, nxt_xy, info)
            if vx is not None and vy is not None:
                speed = (vx * vx + vy * vy) ** 0.5
        samples.append(
            KinematicSample(
                frame=point.frame,
                time_s=time_s,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                speed=speed,
                visible=point.visible,
                confidence=point.confidence,
                manual=point.manual,
                position_unit=position_unit,
                speed_unit=speed_unit,
            )
        )
    return samples


def sample_at_frame(
    samples: list[KinematicSample], frame: int
) -> KinematicSample | None:
    for sample in samples:
        if sample.frame == frame:
            return sample
    return None


def contiguous_segments(
    samples: list[KinematicSample],
    attr: str,
) -> list[list[KinematicSample]]:
    """Split into runs where `attr` is not None (breaks on occlusion / missing)."""
    segments: list[list[KinematicSample]] = []
    current: list[KinematicSample] = []
    for sample in samples:
        value = getattr(sample, attr)
        if value is None:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(sample)
    if current:
        segments.append(current)
    return segments


def _display_xy(point: TrackPoint, calibration: CalibrationState) -> tuple[float, float]:
    if not calibration.applies_transform():
        return point.x, point.y
    world = calibration.pixel_to_world(point.x, point.y)
    return world.x, world.y


def _velocity(
    x: float,
    y: float,
    point: TrackPoint,
    prev: TrackPoint | None,
    prev_xy: tuple[float | None, float | None] | None,
    nxt: TrackPoint | None,
    nxt_xy: tuple[float | None, float | None] | None,
    info: VideoInfo | None,
) -> tuple[float | None, float | None]:
    if (
        prev is not None
        and nxt is not None
        and prev_xy is not None
        and nxt_xy is not None
        and prev_xy[0] is not None
        and prev_xy[1] is not None
        and nxt_xy[0] is not None
        and nxt_xy[1] is not None
    ):
        dt = time_s_for_frame(info, nxt.frame) - time_s_for_frame(info, prev.frame)
        if dt <= 0:
            return None, None
        return (nxt_xy[0] - prev_xy[0]) / dt, (nxt_xy[1] - prev_xy[1]) / dt
    if (
        nxt is not None
        and nxt_xy is not None
        and nxt_xy[0] is not None
        and nxt_xy[1] is not None
    ):
        dt = time_s_for_frame(info, nxt.frame) - time_s_for_frame(info, point.frame)
        if dt <= 0:
            return None, None
        return (nxt_xy[0] - x) / dt, (nxt_xy[1] - y) / dt
    if (
        prev is not None
        and prev_xy is not None
        and prev_xy[0] is not None
        and prev_xy[1] is not None
    ):
        dt = time_s_for_frame(info, point.frame) - time_s_for_frame(info, prev.frame)
        if dt <= 0:
            return None, None
        return (x - prev_xy[0]) / dt, (y - prev_xy[1]) / dt
    return None, None
