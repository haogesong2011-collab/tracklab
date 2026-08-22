"""Pixel ↔ world coordinate transforms used by calibration and physics.

Existing helpers stay as a uniform-scale wrapper around CalibrationState so
older tests and callers keep working.
"""

from __future__ import annotations

import math

from ai.calibration import (
    CalibrationMode,
    CalibrationState,
    CoordinateFrame,
    RulerRole,
    RulerSegment,
)
from ai.schema import Point2D


def rotate(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return x * c - y * s, x * s + y * c


def _uniform_state(
    origin: Point2D, pixels_per_meter: float, axis_angle_deg: float
) -> CalibrationState:
    if pixels_per_meter <= 0:
        raise ValueError("pixels_per_meter must be positive")
    ruler = RulerSegment(
        a=Point2D(origin.x, origin.y),
        b=Point2D(origin.x + pixels_per_meter, origin.y),
        length_m=1.0,
        role=RulerRole.SINGLE,
    )
    return CalibrationState(
        mode=CalibrationMode.UNIFORM,
        rulers=[ruler],
        frame=CoordinateFrame(
            origin_x=origin.x,
            origin_y=origin.y,
            axis_angle_deg=axis_angle_deg,
            y_up=True,
        ),
    )


def pixel_to_world(
    x: float,
    y: float,
    *,
    origin: Point2D,
    pixels_per_meter: float,
    axis_angle_deg: float = 0.0,
) -> Point2D:
    """Convert image pixels (y down) to a right-handed world frame (y up)."""
    return _uniform_state(origin, pixels_per_meter, axis_angle_deg).pixel_to_world(x, y)


def world_to_pixel(
    x: float,
    y: float,
    *,
    origin: Point2D,
    pixels_per_meter: float,
    axis_angle_deg: float = 0.0,
) -> Point2D:
    return _uniform_state(origin, pixels_per_meter, axis_angle_deg).world_to_pixel(x, y)


def interpolate_xy(
    points: list[tuple[int, float, float]],
    frame: int,
) -> tuple[float, float] | None:
    """Linear interpolation of a sparse (frame, x, y) series."""
    if not points:
        return None
    ordered = sorted(points, key=lambda p: p[0])
    if frame <= ordered[0][0]:
        return ordered[0][1], ordered[0][2]
    if frame >= ordered[-1][0]:
        return ordered[-1][1], ordered[-1][2]
    for (f0, x0, y0), (f1, x1, y1) in zip(ordered, ordered[1:]):
        if f0 <= frame <= f1:
            if f1 == f0:
                return x0, y0
            t = (frame - f0) / (f1 - f0)
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    return None
