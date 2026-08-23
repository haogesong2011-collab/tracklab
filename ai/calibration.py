"""Single-ruler and near/far dual-ruler calibration in the analysis plane.

Rulers and the coordinate origin live in stable (shake-compensated) pixel
space. Perspective is a 1-D scale interpolation along the near→far axis,
not a full homography.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ai.schema import Point2D

DEFAULT_RULER_LENGTH_M = 1.0
MIN_RULER_PIXELS = 8.0
MAX_RULER_ANGLE_DEG = 25.0
MIN_DEPTH_PIXELS = 12.0
PPM_RATIO_MAX = 8.0
EXTRAPOLATE_U = 0.35


class CalibrationMode(str, Enum):
    NONE = "none"
    UNIFORM = "uniform"
    NEAR_FAR = "near_far"


class RulerRole(str, Enum):
    SINGLE = "single"
    NEAR = "near"
    FAR = "far"


@dataclass
class RulerSegment:
    a: Point2D
    b: Point2D
    length_m: float = DEFAULT_RULER_LENGTH_M
    role: RulerRole = RulerRole.SINGLE

    @property
    def midpoint(self) -> Point2D:
        return Point2D((self.a.x + self.b.x) / 2.0, (self.a.y + self.b.y) / 2.0)

    @property
    def pixel_length(self) -> float:
        return math.hypot(self.b.x - self.a.x, self.b.y - self.a.y)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.b.y - self.a.y, self.b.x - self.a.x))

    def pixels_per_meter(self) -> float | None:
        if self.length_m <= 1e-9 or self.pixel_length < MIN_RULER_PIXELS:
            return None
        return self.pixel_length / self.length_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": {"x": self.a.x, "y": self.a.y},
            "b": {"x": self.b.x, "y": self.b.y},
            "length_m": self.length_m,
            "role": self.role.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RulerSegment":
        a = data.get("a") or {}
        b = data.get("b") or {}
        return cls(
            a=Point2D(float(a.get("x", 0.0)), float(a.get("y", 0.0))),
            b=Point2D(float(b.get("x", 0.0)), float(b.get("y", 0.0))),
            length_m=float(data.get("length_m", DEFAULT_RULER_LENGTH_M)),
            role=RulerRole(data.get("role", "single")),
        )


@dataclass
class CoordinateFrame:
    origin_x: float | None = None
    origin_y: float | None = None
    axis_angle_deg: float = 0.0
    # Tracker convention: +x right, +y up. Falling motion therefore has vy < 0.
    y_up: bool = True

    @property
    def origin(self) -> Point2D | None:
        if self.origin_x is None or self.origin_y is None:
            return None
        return Point2D(self.origin_x, self.origin_y)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CoordinateFrame":
        raw = data or {}
        return cls(
            origin_x=None if raw.get("origin_x") is None else float(raw["origin_x"]),
            origin_y=None if raw.get("origin_y") is None else float(raw["origin_y"]),
            axis_angle_deg=float(raw.get("axis_angle_deg", 0.0)),
            y_up=bool(raw["y_up"]) if "y_up" in raw else True,
        )


@dataclass
class CalibrationState:
    mode: CalibrationMode = CalibrationMode.NONE
    rulers: list[RulerSegment] = field(default_factory=list)
    frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "rulers": [item.to_dict() for item in self.rulers],
            "frame": self.frame.to_dict(),
            "warning": self.warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CalibrationState":
        if not data:
            return cls()
        mode_raw = data.get("mode", "none")
        try:
            mode = CalibrationMode(mode_raw)
        except ValueError:
            mode = CalibrationMode.NONE
        rulers = [RulerSegment.from_dict(item) for item in data.get("rulers") or []]
        state = cls(
            mode=mode,
            rulers=rulers,
            frame=CoordinateFrame.from_dict(data.get("frame")),
            warning=str(data.get("warning", "")),
        )
        ok, message = state.validate()
        if not ok:
            state.mode = CalibrationMode.NONE
            state.warning = message
        return state

    @property
    def active(self) -> bool:
        return self.mode is not CalibrationMode.NONE and bool(self.rulers)

    @property
    def position_unit(self) -> str:
        return "m" if self.active else "px"

    @property
    def speed_unit(self) -> str:
        return "m/s" if self.active else "px/s"

    def origin_point(self) -> Point2D:
        origin = self.frame.origin
        if origin is not None:
            return origin
        return Point2D(0.0, 0.0)

    def axis_angle(self) -> float:
        return self.frame.axis_angle_deg

    def applies_transform(self) -> bool:
        return self.active or self.frame.origin is not None

    def uses_placed_origin(self) -> bool:
        return self.frame.origin is not None

    def swap_near_far(self) -> None:
        for ruler in self.rulers:
            if ruler.role is RulerRole.NEAR:
                ruler.role = RulerRole.FAR
            elif ruler.role is RulerRole.FAR:
                ruler.role = RulerRole.NEAR

    def validate(self) -> tuple[bool, str]:
        if self.mode is CalibrationMode.NONE:
            return True, ""
        if self.mode is CalibrationMode.UNIFORM:
            if len(self.rulers) < 1:
                return False, "请先画一把标定尺"
            ppm = self.rulers[0].pixels_per_meter()
            if ppm is None:
                return False, "标定尺太短或长度无效"
            return True, ""
        if len(self.rulers) < 2:
            return False, "透视模式需要近尺和远尺"
        near, far = self._near_far()
        if near is None or far is None:
            return False, "近尺或远尺无效"
        n_ppm, f_ppm = near.pixels_per_meter(), far.pixels_per_meter()
        if n_ppm is None or f_ppm is None:
            return False, "标定尺太短或长度无效"
        angle = _acute_angle_deg(near.angle_deg, far.angle_deg)
        if angle > MAX_RULER_ANGLE_DEG:
            return False, f"两尺夹角过大（{angle:.0f}°），请尽量平行放置"
        depth = math.hypot(
            far.midpoint.x - near.midpoint.x, far.midpoint.y - near.midpoint.y
        )
        if depth < MIN_DEPTH_PIXELS:
            return False, "近尺和远尺距离太近"
        ratio = max(n_ppm, f_ppm) / min(n_ppm, f_ppm)
        if ratio > PPM_RATIO_MAX:
            return False, "近远比例相差过大，请检查尺子长度"
        return True, ""

    def pixels_per_meter_at(self, x: float, y: float) -> float | None:
        if self.mode is CalibrationMode.NONE:
            return None
        if self.mode is CalibrationMode.UNIFORM:
            if not self.rulers:
                return None
            return self.rulers[0].pixels_per_meter()
        near, far = self._near_far()
        if near is None or far is None:
            return None
        n_ppm, f_ppm = near.pixels_per_meter(), far.pixels_per_meter()
        if n_ppm is None or f_ppm is None:
            return None
        u, _extra = self._depth_u(x, y, near, far)
        ppm = n_ppm + u * (f_ppm - n_ppm)
        if ppm <= 1e-9:
            ppm = min(n_ppm, f_ppm)
        return max(ppm, 1e-6)

    def is_extrapolated(self, x: float, y: float) -> bool:
        if self.mode is not CalibrationMode.NEAR_FAR:
            return False
        near, far = self._near_far()
        if near is None or far is None:
            return False
        _u, extra = self._depth_u(x, y, near, far)
        return extra

    def pixel_to_world(self, x: float, y: float) -> Point2D:
        # Tracker default: +x right, +y up. Image y is flipped whenever y_up
        # is set, including a ruler-only scale that has no placed origin.
        if not self.uses_placed_origin():
            dx_m, dy_m = self._image_delta_m(0.0, 0.0, x, y)
            if self.frame.y_up:
                dy_m = -dy_m
            return Point2D(dx_m, dy_m)
        origin = self.origin_point()
        dx_m, dy_m = self._image_delta_m(origin.x, origin.y, x, y)
        if self.frame.y_up:
            dy_m = -dy_m
        wx, wy = _rotate(dx_m, dy_m, -self.axis_angle())
        return Point2D(wx, wy)

    def world_to_pixel(self, x: float, y: float) -> Point2D:
        if not self.uses_placed_origin():
            dy_m = -y if self.frame.y_up else y
            px, py = self._meters_delta_to_pixel(0.0, 0.0, x, dy_m)
            return Point2D(px, py)
        origin = self.origin_point()
        dx_m, dy_m = _rotate(x, y, self.axis_angle())
        if self.frame.y_up:
            dy_m = -dy_m
        px, py = self._meters_delta_to_pixel(origin.x, origin.y, dx_m, dy_m)
        return Point2D(px, py)

    def _near_far(self) -> tuple[RulerSegment | None, RulerSegment | None]:
        near = next((item for item in self.rulers if item.role is RulerRole.NEAR), None)
        far = next((item for item in self.rulers if item.role is RulerRole.FAR), None)
        if near is None and len(self.rulers) >= 1:
            near = self.rulers[0]
        if far is None and len(self.rulers) >= 2:
            far = self.rulers[1]
        return near, far

    def _depth_u(
        self, x: float, y: float, near: RulerSegment, far: RulerSegment
    ) -> tuple[float, bool]:
        dx = far.midpoint.x - near.midpoint.x
        dy = far.midpoint.y - near.midpoint.y
        denom = dx * dx + dy * dy
        if denom < 1e-9:
            return 0.0, False
        u = ((x - near.midpoint.x) * dx + (y - near.midpoint.y) * dy) / denom
        extra = u < -1e-6 or u > 1.0 + 1e-6
        u = max(-EXTRAPOLATE_U, min(1.0 + EXTRAPOLATE_U, u))
        return u, extra

    def _image_delta_m(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> tuple[float, float]:
        if self.mode is CalibrationMode.NONE:
            return x1 - x0, y1 - y0
        if self.mode is CalibrationMode.UNIFORM:
            ppm = self.pixels_per_meter_at(x1, y1) or 1.0
            return (x1 - x0) / ppm, (y1 - y0) / ppm
        integral = self._path_integral(x0, y0, x1, y1)
        return (x1 - x0) * integral, (y1 - y0) * integral

    def _path_integral(self, x0: float, y0: float, x1: float, y1: float) -> float:
        ppm0 = self.pixels_per_meter_at(x0, y0)
        ppm1 = self.pixels_per_meter_at(x1, y1)
        if ppm0 is None or ppm1 is None:
            return 1.0
        delta = ppm1 - ppm0
        if abs(delta) < 1e-9:
            return 1.0 / ppm0
        return math.log(ppm1 / ppm0) / delta

    def _meters_delta_to_pixel(
        self, x0: float, y0: float, dx_m: float, dy_m: float
    ) -> tuple[float, float]:
        if self.mode is CalibrationMode.NONE:
            return x0 + dx_m, y0 + dy_m
        if self.mode is CalibrationMode.UNIFORM:
            ppm = self.pixels_per_meter_at(x0, y0) or 1.0
            return x0 + dx_m * ppm, y0 + dy_m * ppm
        distance = math.hypot(dx_m, dy_m)
        if distance < 1e-12:
            return x0, y0
        ux, uy = dx_m / distance, dy_m / distance
        lo, hi = 0.0, max(distance * (self.pixels_per_meter_at(x0, y0) or 1.0) * 4.0, 8.0)
        target = distance
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            x1, y1 = x0 + ux * mid, y0 + uy * mid
            got = mid * self._path_integral(x0, y0, x1, y1)
            if got < target:
                lo = mid
            else:
                hi = mid
        s = 0.5 * (lo + hi)
        return x0 + ux * s, y0 + uy * s


def _rotate(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return x * c - y * s, x * s + y * c


def _acute_angle_deg(a: float, b: float) -> float:
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return min(diff, 180.0 - diff)


def uniform_state(
    a: Point2D,
    b: Point2D,
    *,
    length_m: float = DEFAULT_RULER_LENGTH_M,
    origin: Point2D | None = None,
    axis_angle_deg: float | None = None,
) -> CalibrationState:
    ruler = RulerSegment(a=a, b=b, length_m=length_m, role=RulerRole.SINGLE)
    angle = ruler.angle_deg if axis_angle_deg is None else axis_angle_deg
    frame = CoordinateFrame(
        origin_x=None if origin is None else origin.x,
        origin_y=None if origin is None else origin.y,
        axis_angle_deg=angle,
    )
    state = CalibrationState(
        mode=CalibrationMode.UNIFORM, rulers=[ruler], frame=frame
    )
    ok, message = state.validate()
    if not ok:
        state.mode = CalibrationMode.NONE
        state.warning = message
    return state


def near_far_state(
    near: RulerSegment,
    far: RulerSegment,
    *,
    origin: Point2D | None = None,
    axis_angle_deg: float | None = None,
) -> CalibrationState:
    near.role = RulerRole.NEAR
    far.role = RulerRole.FAR
    angle = near.angle_deg if axis_angle_deg is None else axis_angle_deg
    frame = CoordinateFrame(
        origin_x=None if origin is None else origin.x,
        origin_y=None if origin is None else origin.y,
        axis_angle_deg=angle,
    )
    state = CalibrationState(
        mode=CalibrationMode.NEAR_FAR, rulers=[near, far], frame=frame
    )
    ok, message = state.validate()
    if not ok:
        state.mode = CalibrationMode.NONE
        state.warning = message
    return state
