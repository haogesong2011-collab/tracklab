"""Calibration in the analysis plane: uniform, near/far, or planar homography.

Rulers, plane corners, and the coordinate origin live in stable
(shake-compensated) pixel space. UNIFORM / NEAR_FAR stay as 1-D scale
models. PLANAR uses a full 2-D homography of a known rectangle.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ai.plane import (
    DEFAULT_PIXEL_SIGMA,
    MAX_REPROJ_RMS_PX,
    WARN_REPROJ_RMS_PX,
    HomographyFit,
    apply_homography,
    distort_point,
    expected_camera_depth,
    homography_pixel_to_world,
    pose_from_world_to_pixel,
    point_in_quad,
    quad_quality,
    undistort_point,
    world_sigma,
)
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
    PLANAR = "planar"


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
class CameraProfile:
    """Optional pinhole + Brown–Conrady distortion. Missing fx skips undistort."""

    width: int = 0
    height: int = 0
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    rms: float | None = None

    @property
    def has_intrinsics(self) -> bool:
        return self.fx is not None and self.fy is not None and self.fx > 1e-6 and self.fy > 1e-6

    def camera_matrix(self) -> np.ndarray | None:
        if not self.has_intrinsics:
            return None
        cx = self.width / 2.0 if self.cx is None else float(self.cx)
        cy = self.height / 2.0 if self.cy is None else float(self.cy)
        return np.array(
            [[float(self.fx), 0.0, cx], [0.0, float(self.fy), cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CameraProfile | None":
        if not data:
            return None
        fx = data.get("fx")
        fy = data.get("fy")
        return cls(
            width=int(data.get("width") or 0),
            height=int(data.get("height") or 0),
            fx=None if fx is None else float(fx),
            fy=None if fy is None else float(fy),
            cx=None if data.get("cx") is None else float(data["cx"]),
            cy=None if data.get("cy") is None else float(data["cy"]),
            k1=float(data.get("k1") or 0.0),
            k2=float(data.get("k2") or 0.0),
            p1=float(data.get("p1") or 0.0),
            p2=float(data.get("p2") or 0.0),
            k3=float(data.get("k3") or 0.0),
            rms=None if data.get("rms") is None else float(data["rms"]),
        )


@dataclass
class PlanePatch:
    """User rectangle on the motion plane: origin, +X, opposite, +Y."""

    corners: list[Point2D] = field(default_factory=list)
    width_m: float = 1.0
    height_m: float = 1.0
    reprojection_rms_px: float = 0.0
    area_px: float = 0.0
    min_angle_deg: float = 0.0
    coverage: float = 0.0

    def world_corners(self) -> list[Point2D]:
        w = max(self.width_m, 1e-9)
        h = max(self.height_m, 1e-9)
        return [
            Point2D(0.0, 0.0),
            Point2D(w, 0.0),
            Point2D(w, h),
            Point2D(0.0, h),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corners": [{"x": p.x, "y": p.y} for p in self.corners],
            "width_m": self.width_m,
            "height_m": self.height_m,
            "reprojection_rms_px": self.reprojection_rms_px,
            "area_px": self.area_px,
            "min_angle_deg": self.min_angle_deg,
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlanePatch | None":
        if not data:
            return None
        corners = [
            Point2D(float(item.get("x", 0.0)), float(item.get("y", 0.0)))
            for item in data.get("corners") or []
        ]
        return cls(
            corners=corners,
            width_m=float(data.get("width_m", 1.0)),
            height_m=float(data.get("height_m", 1.0)),
            reprojection_rms_px=float(data.get("reprojection_rms_px") or 0.0),
            area_px=float(data.get("area_px") or 0.0),
            min_angle_deg=float(data.get("min_angle_deg") or 0.0),
            coverage=float(data.get("coverage") or 0.0),
        )


@dataclass
class CalibrationState:
    mode: CalibrationMode = CalibrationMode.NONE
    rulers: list[RulerSegment] = field(default_factory=list)
    frame: CoordinateFrame = field(default_factory=CoordinateFrame)
    warning: str = ""
    plane: PlanePatch | None = None
    camera: CameraProfile | None = None
    camera_moved: bool = False
    pixel_sigma: float = DEFAULT_PIXEL_SIGMA
    _fit: HomographyFit | None = field(default=None, init=False, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "rulers": [item.to_dict() for item in self.rulers],
            "frame": self.frame.to_dict(),
            "warning": self.warning,
            "plane": None if self.plane is None else self.plane.to_dict(),
            "camera": None if self.camera is None else self.camera.to_dict(),
            "camera_moved": self.camera_moved,
            "pixel_sigma": self.pixel_sigma,
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
            plane=PlanePatch.from_dict(data.get("plane")),
            camera=CameraProfile.from_dict(data.get("camera")),
            camera_moved=bool(data.get("camera_moved", False)),
            pixel_sigma=float(data.get("pixel_sigma") or DEFAULT_PIXEL_SIGMA),
        )
        ok, message = state.validate()
        if not ok:
            state.mode = CalibrationMode.NONE
            state.warning = message
        return state

    @property
    def active(self) -> bool:
        if self.mode is CalibrationMode.NONE:
            return False
        if self.mode is CalibrationMode.PLANAR:
            return (
                self.plane is not None
                and len(self.plane.corners) >= 4
                and self._fit is not None
            )
        return bool(self.rulers)

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
        if self.mode is CalibrationMode.PLANAR:
            return self._validate_plane()
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
        if self.mode is CalibrationMode.PLANAR:
            sigma_x, sigma_y = self.position_sigma(x, y, pixel_sigma=1.0)
            meters = math.hypot(sigma_x, sigma_y)
            if meters <= 1e-12:
                return None
            return 1.0 / meters
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
        if self.mode is CalibrationMode.PLANAR:
            if self.plane is None or len(self.plane.corners) < 4:
                return False
            return not point_in_quad(x, y, self.plane.corners)
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
        if self.mode is CalibrationMode.PLANAR:
            return self._planar_to_world(x, y)
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
        if self.mode is CalibrationMode.PLANAR:
            return self._world_to_planar(x, y)
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

    def position_sigma(
        self,
        x: float,
        y: float,
        *,
        pixel_sigma: float | None = None,
    ) -> tuple[float, float]:
        if self.mode is not CalibrationMode.PLANAR or self._fit is None:
            return 0.0, 0.0
        u, v = self._undistort_pixel(x, y)
        rms = 0.0 if self.plane is None else self.plane.reprojection_rms_px
        return world_sigma(
            self._fit.matrix,
            u,
            v,
            pixel_sigma=self.pixel_sigma if pixel_sigma is None else pixel_sigma,
            control_rms_px=rms,
        )

    def measurement_source(self, x: float, y: float) -> str:
        if not self.active:
            return "pixel"
        if self.is_extrapolated(x, y):
            return "extrapolated"
        if self.mode is CalibrationMode.PLANAR:
            return "geometric"
        return "scaled"

    def quality_label(self) -> str:
        if self.mode is CalibrationMode.PLANAR:
            bits = ["运动平面"]
            if self.camera is None or not self.camera.has_intrinsics:
                bits.append("未去畸变")
            if self.plane and self.plane.reprojection_rms_px > WARN_REPROJ_RMS_PX:
                bits.append(f"重投影 {self.plane.reprojection_rms_px:.2f} px")
            if self.camera_moved:
                bits.append("疑似机位移动")
            return " · ".join(bits)
        if self.mode is CalibrationMode.NEAR_FAR:
            return "近远双尺"
        if self.mode is CalibrationMode.UNIFORM:
            return "单尺"
        return "未标定"

    def plane_to_pixel(self, x: float, y: float) -> Point2D:
        """Map plane-local metres through H⁻¹, ignoring the analysis origin."""
        if self._fit is None:
            self._validate_plane()
        if self._fit is None:
            return Point2D(0.0, 0.0)
        mapped = apply_homography(self._fit.inverse, x, y)
        if mapped is None:
            return Point2D(0.0, 0.0)
        u, v = self._distort_pixel(mapped[0], mapped[1])
        return Point2D(u, v)

    def grid_lines(self, nx: int = 4, ny: int = 4) -> list[tuple[float, float, float, float]]:
        if self.mode is not CalibrationMode.PLANAR or self.plane is None or self._fit is None:
            return []
        width = max(self.plane.width_m, 1e-9)
        height = max(self.plane.height_m, 1e-9)
        lines: list[tuple[float, float, float, float]] = []
        for i in range(max(nx, 1) + 1):
            x = width * i / max(nx, 1)
            a = self.plane_to_pixel(x, 0.0)
            b = self.plane_to_pixel(x, height)
            lines.append((a.x, a.y, b.x, b.y))
        for j in range(max(ny, 1) + 1):
            y = height * j / max(ny, 1)
            a = self.plane_to_pixel(0.0, y)
            b = self.plane_to_pixel(width, y)
            lines.append((a.x, a.y, b.x, b.y))
        return lines

    def expected_depth_m(self, x: float, y: float) -> float | None:
        if self.mode is not CalibrationMode.PLANAR or self._fit is None:
            return None
        world = self._planar_to_world(x, y)
        width = 0.0 if self.camera is None else float(self.camera.width)
        height = 0.0 if self.camera is None else float(self.camera.height)
        pose = pose_from_world_to_pixel(
            self._fit.inverse,
            None if self.camera is None else self.camera.camera_matrix(),
            width=width,
            height=height,
        )
        return expected_camera_depth(pose, world.x, world.y)

    def _validate_plane(self) -> tuple[bool, str]:
        if self.plane is None or len(self.plane.corners) < 4:
            return False, "请依次点选平面的原点、X 端、对角点和 Y 端"
        if self.plane.width_m <= 1e-6 or self.plane.height_m <= 1e-6:
            return False, "平面实际宽高必须大于 0"
        image_area = 0.0
        if self.camera is not None and self.camera.width > 0 and self.camera.height > 0:
            image_area = float(self.camera.width * self.camera.height)
        ok, message, metrics = quad_quality(self.plane.corners, image_area=image_area)
        if not ok:
            self._fit = None
            return False, message
        try:
            fit = homography_pixel_to_world(
                [self._undistort_point(p) for p in self.plane.corners],
                self.plane.world_corners(),
            )
        except Exception:
            self._fit = None
            return False, "无法求解平面单应，请调整四个角点"
        if not np.isfinite(fit.condition) or fit.condition > 1e8:
            self._fit = None
            return False, "平面映射病态，请避免过扁的四边形"
        if fit.reprojection_rms_px > MAX_REPROJ_RMS_PX:
            self._fit = None
            return False, f"重投影误差过大（{fit.reprojection_rms_px:.2f} px）"
        self._fit = fit
        self.plane.reprojection_rms_px = fit.reprojection_rms_px
        self.plane.area_px = metrics["area_px"]
        self.plane.min_angle_deg = metrics["min_angle_deg"]
        self.plane.coverage = metrics["coverage"]
        warnings: list[str] = []
        if fit.reprojection_rms_px > WARN_REPROJ_RMS_PX:
            warnings.append(f"重投影误差 {fit.reprojection_rms_px:.2f} px")
        if self.camera is None or not self.camera.has_intrinsics:
            warnings.append("未配置镜头内参，精度会下降")
        if self.camera_moved:
            warnings.append("检测到画面运动，平面标定可能失效")
        self.warning = "；".join(warnings)
        return True, ""

    def _undistort_point(self, point: Point2D) -> Point2D:
        u, v = self._undistort_pixel(point.x, point.y)
        return Point2D(u, v)

    def _undistort_pixel(self, x: float, y: float) -> tuple[float, float]:
        cam = self.camera
        if cam is None or not cam.has_intrinsics:
            return x, y
        cx = cam.width / 2.0 if cam.cx is None else float(cam.cx)
        cy = cam.height / 2.0 if cam.cy is None else float(cam.cy)
        return undistort_point(
            x,
            y,
            fx=float(cam.fx),
            fy=float(cam.fy),
            cx=cx,
            cy=cy,
            k1=cam.k1,
            k2=cam.k2,
            p1=cam.p1,
            p2=cam.p2,
            k3=cam.k3,
        )

    def _distort_pixel(self, x: float, y: float) -> tuple[float, float]:
        cam = self.camera
        if cam is None or not cam.has_intrinsics:
            return x, y
        cx = cam.width / 2.0 if cam.cx is None else float(cam.cx)
        cy = cam.height / 2.0 if cam.cy is None else float(cam.cy)
        return distort_point(
            x,
            y,
            fx=float(cam.fx),
            fy=float(cam.fy),
            cx=cx,
            cy=cy,
            k1=cam.k1,
            k2=cam.k2,
            p1=cam.p1,
            p2=cam.p2,
            k3=cam.k3,
        )

    def _planar_to_world(self, x: float, y: float) -> Point2D:
        if self._fit is None:
            self._validate_plane()
        if self._fit is None:
            return Point2D(0.0, 0.0)
        u, v = self._undistort_pixel(x, y)
        mapped = apply_homography(self._fit.matrix, u, v)
        if mapped is None:
            return Point2D(0.0, 0.0)
        wx, wy = mapped
        if self.uses_placed_origin() and self.plane is not None:
            origin = self.origin_point()
            ou, ov = self._undistort_pixel(origin.x, origin.y)
            base = apply_homography(self._fit.matrix, ou, ov)
            if base is not None:
                wx -= base[0]
                wy -= base[1]
            wx, wy = _rotate(wx, wy, -self.axis_angle())
        return Point2D(wx, wy)

    def _world_to_planar(self, x: float, y: float) -> Point2D:
        if self._fit is None:
            self._validate_plane()
        if self._fit is None:
            return Point2D(0.0, 0.0)
        wx, wy = x, y
        if self.uses_placed_origin() and self.plane is not None:
            wx, wy = _rotate(x, y, self.axis_angle())
            origin = self.origin_point()
            ou, ov = self._undistort_pixel(origin.x, origin.y)
            base = apply_homography(self._fit.matrix, ou, ov)
            if base is not None:
                wx += base[0]
                wy += base[1]
        mapped = apply_homography(self._fit.inverse, wx, wy)
        if mapped is None:
            return Point2D(0.0, 0.0)
        u, v = self._distort_pixel(mapped[0], mapped[1])
        return Point2D(u, v)

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


def planar_state(
    corners: list[Point2D],
    *,
    width_m: float,
    height_m: float,
    origin: Point2D | None = None,
    axis_angle_deg: float = 0.0,
    camera: CameraProfile | None = None,
    y_up: bool = True,
) -> CalibrationState:
    plane = PlanePatch(corners=list(corners), width_m=width_m, height_m=height_m)
    frame = CoordinateFrame(
        origin_x=None if origin is None else origin.x,
        origin_y=None if origin is None else origin.y,
        axis_angle_deg=axis_angle_deg,
        y_up=y_up,
    )
    state = CalibrationState(
        mode=CalibrationMode.PLANAR,
        plane=plane,
        frame=frame,
        camera=camera,
    )
    ok, message = state.validate()
    if not ok:
        state.mode = CalibrationMode.NONE
        state.warning = message
    return state
