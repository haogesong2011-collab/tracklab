"""Planar homography calibration vs 1-D near/far interpolation."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.calibration import (  # noqa: E402
    CalibrationMode,
    CameraProfile,
    planar_state,
    uniform_state,
)
from ai.plane import (  # noqa: E402
    distort_point,
    point_in_quad,
    pose_from_world_to_pixel,
    project_world_point,
    undistort_point,
)
from ai.schema import Point2D  # noqa: E402


def _pinhole_project(
    x: float,
    y: float,
    z: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[float, float]:
    cam = rvec @ np.array([x, y, z], dtype=np.float64) + tvec
    u = fx * cam[0] / cam[2] + cx
    v = fy * cam[1] / cam[2] + cy
    return float(u), float(v)


def _tilted_camera() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    # Camera looks toward +Z_cam, plane is Z_world=0, camera sits above and
    # in front of the rectangle so the near edge is larger in the image.
    yaw = math.radians(18.0)
    pitch = math.radians(28.0)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rotation = rx @ ry
    tvec = np.array([-0.9, -0.35, 2.4], dtype=np.float64)
    cam = {"fx": 900.0, "fy": 900.0, "cx": 320.0, "cy": 180.0}
    return rotation, tvec, cam


class PlaneCalibrationTests(unittest.TestCase):
    def test_homography_roundtrip_on_tilted_rectangle(self) -> None:
        rotation, tvec, cam = _tilted_camera()
        world = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        corners = []
        for x, y in world:
            u, v = _pinhole_project(x, y, 0.0, rvec=rotation, tvec=tvec, **cam)
            corners.append(Point2D(u, v))
        state = planar_state(corners, width_m=2.0, height_m=1.0)
        self.assertEqual(state.mode, CalibrationMode.PLANAR)
        self.assertTrue(state.active)
        self.assertFalse(state.rulers)
        for x, y in [(0.4, 0.2), (1.5, 0.8), (1.0, 0.5)]:
            u, v = _pinhole_project(x, y, 0.0, rvec=rotation, tvec=tvec, **cam)
            mapped = state.pixel_to_world(u, v)
            self.assertAlmostEqual(mapped.x, x, places=3)
            self.assertAlmostEqual(mapped.y, y, places=3)
            back = state.world_to_pixel(mapped.x, mapped.y)
            self.assertAlmostEqual(back.x, u, places=2)
            self.assertAlmostEqual(back.y, v, places=2)

    def test_planar_beats_near_far_on_perspective(self) -> None:
        rotation, tvec, cam = _tilted_camera()
        world_rect = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        corners = [
            Point2D(*_pinhole_project(x, y, 0.0, rvec=rotation, tvec=tvec, **cam))
            for x, y in world_rect
        ]
        planar = planar_state(corners, width_m=2.0, height_m=1.0)
        near_a = _pinhole_project(0.0, 0.0, 0.0, rvec=rotation, tvec=tvec, **cam)
        near_b = _pinhole_project(2.0, 0.0, 0.0, rvec=rotation, tvec=tvec, **cam)
        # Uniform scale taken from the near edge — the usual 1-D approximation.
        ppm = math.hypot(near_b[0] - near_a[0], near_b[1] - near_a[1]) / 2.0
        uniform = uniform_state(
            Point2D(*near_a),
            Point2D(near_a[0] + ppm, near_a[1]),
            length_m=1.0,
            origin=Point2D(*near_a),
        )
        uniform.frame.y_up = False
        samples = [(0.3, 0.2), (1.1, 0.7), (1.8, 0.9)]
        planar_err = []
        uniform_err = []
        for x, y in samples:
            u, v = _pinhole_project(x, y, 0.0, rvec=rotation, tvec=tvec, **cam)
            p = planar.pixel_to_world(u, v)
            q = uniform.pixel_to_world(u, v)
            planar_err.append(math.hypot(p.x - x, p.y - y))
            uniform_err.append(math.hypot(q.x - x, q.y - y))
        self.assertLess(max(planar_err), 0.01)
        self.assertGreater(max(uniform_err), max(planar_err) * 8)

    def test_reject_degenerate_quad(self) -> None:
        corners = [
            Point2D(10, 10),
            Point2D(20, 10),
            Point2D(30, 10),
            Point2D(40, 10),
        ]
        state = planar_state(corners, width_m=1.0, height_m=1.0)
        self.assertEqual(state.mode, CalibrationMode.NONE)
        self.assertTrue(state.warning)

    def test_extrapolation_flag_and_sigma(self) -> None:
        corners = [
            Point2D(40, 40),
            Point2D(200, 40),
            Point2D(200, 160),
            Point2D(40, 160),
        ]
        state = planar_state(corners, width_m=1.6, height_m=1.2)
        self.assertEqual(state.mode, CalibrationMode.PLANAR)
        self.assertFalse(state.is_extrapolated(80, 80))
        self.assertTrue(state.is_extrapolated(10, 10))
        sx, sy = state.position_sigma(80, 80)
        self.assertGreater(sx, 0.0)
        self.assertGreater(sy, 0.0)

    def test_undistort_roundtrip(self) -> None:
        fx, fy, cx, cy = 800.0, 800.0, 320.0, 180.0
        k1 = -0.12
        u, v = 400.0, 210.0
        du, dv = distort_point(u, v, fx=fx, fy=fy, cx=cx, cy=cy, k1=k1)
        ru, rv = undistort_point(du, dv, fx=fx, fy=fy, cx=cx, cy=cy, k1=k1)
        self.assertAlmostEqual(ru, u, places=3)
        self.assertAlmostEqual(rv, v, places=3)

    def test_distortion_aware_plane(self) -> None:
        rotation, tvec, cam = _tilted_camera()
        profile = CameraProfile(
            width=640,
            height=360,
            fx=cam["fx"],
            fy=cam["fy"],
            cx=cam["cx"],
            cy=cam["cy"],
            k1=-0.08,
        )
        world = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        corners = []
        for x, y in world:
            u, v = _pinhole_project(x, y, 0.0, rvec=rotation, tvec=tvec, **cam)
            du, dv = distort_point(
                u, v, fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"], k1=-0.08
            )
            corners.append(Point2D(du, dv))
        state = planar_state(corners, width_m=2.0, height_m=1.0, camera=profile)
        self.assertEqual(state.mode, CalibrationMode.PLANAR)
        u, v = _pinhole_project(1.2, 0.4, 0.0, rvec=rotation, tvec=tvec, **cam)
        du, dv = distort_point(
            u, v, fx=cam["fx"], fy=cam["fy"], cx=cam["cx"], cy=cam["cy"], k1=-0.08
        )
        mapped = state.pixel_to_world(du, dv)
        self.assertAlmostEqual(mapped.x, 1.2, places=2)
        self.assertAlmostEqual(mapped.y, 0.4, places=2)

    def test_point_in_quad_and_pose_depth(self) -> None:
        corners = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]
        self.assertTrue(point_in_quad(5, 5, corners))
        self.assertFalse(point_in_quad(20, 5, corners))
        rotation, tvec, cam = _tilted_camera()
        world = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        corners = [
            Point2D(*_pinhole_project(x, y, 0.0, rvec=rotation, tvec=tvec, **cam))
            for x, y in world
        ]
        state = planar_state(corners, width_m=2.0, height_m=1.0)
        depth = state.expected_depth_m(*_pinhole_project(1.0, 0.5, 0.0, rvec=rotation, tvec=tvec, **cam))
        self.assertIsNotNone(depth)
        assert depth is not None
        self.assertGreater(depth, 0.5)

    def test_off_plane_projection_changes_pixel(self) -> None:
        rotation, tvec, cam = _tilted_camera()
        pose = pose_from_world_to_pixel(
            np.eye(3),
            np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]]),
            width=640,
            height=360,
        )
        pose.rotation[:] = rotation
        pose.translation[:] = tvec
        on = project_world_point(1.0, 0.4, 0.0, pose)
        off = project_world_point(1.0, 0.4, 0.05, pose)
        self.assertIsNotNone(on)
        self.assertIsNotNone(off)
        assert on is not None and off is not None
        self.assertGreater(math.hypot(on[0] - off[0], on[1] - off[1]), 1.0)


if __name__ == "__main__":
    unittest.main()
