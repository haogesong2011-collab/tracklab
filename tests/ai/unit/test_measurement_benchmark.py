"""Synthetic 3D projection benchmark: planar homography vs 1-D near/far."""

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
    RulerRole,
    RulerSegment,
    near_far_state,
    planar_state,
    uniform_state,
)
from ai.contracts import TrackPoint, TrackResult  # noqa: E402
from ai.depth_audit import FakeDepthEstimator, audit_track  # noqa: E402
from ai.kinematics import series_for_result  # noqa: E402
from ai.schema import Point2D  # noqa: E402
from engine.video_index import VideoInfo  # noqa: E402


def _pinhole(x, y, z, rotation, tvec, cam):  # noqa: ANN001
    cam_pt = rotation @ np.array([x, y, z], dtype=np.float64) + tvec
    u = cam["fx"] * cam_pt[0] / cam_pt[2] + cam["cx"]
    v = cam["fy"] * cam_pt[1] / cam_pt[2] + cam["cy"]
    return float(u), float(v)


def _tilted():
    yaw, pitch = math.radians(18.0), math.radians(28.0)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rotation = rx @ ry
    tvec = np.array([-0.9, -0.35, 2.4], dtype=np.float64)
    cam = {"fx": 900.0, "fy": 900.0, "cx": 320.0, "cy": 180.0}
    return rotation, tvec, cam


class MeasurementBenchmarkTests(unittest.TestCase):
    def test_planar_mae_beats_near_far_on_tilted_plane(self) -> None:
        rotation, tvec, cam = _tilted()
        world_corners = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        corners = [
            Point2D(*_pinhole(x, y, 0.0, rotation, tvec, cam)) for x, y in world_corners
        ]
        planar = planar_state(corners, width_m=2.0, height_m=1.0)
        self.assertEqual(planar.mode, CalibrationMode.PLANAR)
        near_a = Point2D(*_pinhole(0.0, 0.1, 0.0, rotation, tvec, cam))
        near_b = Point2D(*_pinhole(1.0, 0.1, 0.0, rotation, tvec, cam))
        far_a = Point2D(*_pinhole(0.0, 0.9, 0.0, rotation, tvec, cam))
        far_b = Point2D(*_pinhole(1.0, 0.9, 0.0, rotation, tvec, cam))
        nf = near_far_state(
            RulerSegment(near_a, near_b, 1.0, RulerRole.NEAR),
            RulerSegment(far_a, far_b, 1.0, RulerRole.FAR),
        )
        samples = [(0.3, 0.2), (1.1, 0.4), (1.7, 0.8), (0.6, 0.7)]
        planar_err = []
        nf_err = []
        for x, y in samples:
            u, v = _pinhole(x, y, 0.0, rotation, tvec, cam)
            mapped = planar.pixel_to_world(u, v)
            planar_err.append(math.hypot(mapped.x - x, mapped.y - y))
            scaled = nf.pixel_to_world(u, v)
            nf_err.append(math.hypot(scaled.x - x, scaled.y - y))
        planar_mae = sum(planar_err) / len(planar_err)
        nf_mae = sum(nf_err) / len(nf_err)
        self.assertLess(planar_mae, 0.01)
        self.assertLess(planar_mae * 8.0, nf_mae)

    def test_uniform_scale_is_biased_by_perspective(self) -> None:
        rotation, tvec, cam = _tilted()
        corners = [
            Point2D(*_pinhole(x, y, 0.0, rotation, tvec, cam))
            for x, y in [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        ]
        planar = planar_state(corners, width_m=2.0, height_m=1.0)
        a = Point2D(*_pinhole(0.0, 0.5, 0.0, rotation, tvec, cam))
        b = Point2D(*_pinhole(1.0, 0.5, 0.0, rotation, tvec, cam))
        uniform = uniform_state(a, b, length_m=1.0)
        far = Point2D(*_pinhole(0.0, 0.95, 0.0, rotation, tvec, cam))
        far2 = Point2D(*_pinhole(1.0, 0.95, 0.0, rotation, tvec, cam))
        planar_span = math.hypot(
            *(
                np.array([planar.pixel_to_world(far2.x, far2.y).x, planar.pixel_to_world(far2.x, far2.y).y])
                - np.array([planar.pixel_to_world(far.x, far.y).x, planar.pixel_to_world(far.x, far.y).y])
            )
        )
        uniform_span = math.hypot(
            *(
                np.array([uniform.pixel_to_world(far2.x, far2.y).x, uniform.pixel_to_world(far2.x, far2.y).y])
                - np.array([uniform.pixel_to_world(far.x, far.y).x, uniform.pixel_to_world(far.x, far.y).y])
            )
        )
        self.assertAlmostEqual(planar_span, 1.0, delta=0.02)
        self.assertGreater(abs(uniform_span - 1.0), 0.05)

    def test_off_plane_detection_rate(self) -> None:
        rotation, tvec, cam = _tilted()
        profile = CameraProfile(
            width=640,
            height=360,
            fx=cam["fx"],
            fy=cam["fy"],
            cx=cam["cx"],
            cy=cam["cy"],
        )
        corners = [
            Point2D(*_pinhole(x, y, 0.0, rotation, tvec, cam))
            for x, y in [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        ]
        cal = planar_state(corners, width_m=2.0, height_m=1.0, camera=profile)
        points = []
        for i, x in enumerate(np.linspace(0.3, 1.7, 8)):
            u, v = _pinhole(x, 0.4, 0.0, rotation, tvec, cam)
            points.append(TrackPoint(frame=i, x=u, y=v))
        result = TrackResult(clip_id="bench", points=points)

        def load_frame(index: int) -> np.ndarray:
            rgb = np.zeros((360, 640, 3), dtype=np.uint8)
            rgb[0, 0, 0] = index
            return rgb

        extra = {i: 0.2 for i in range(4, 8)}
        state = audit_track(
            result,
            cal,
            load_frame,
            FakeDepthEstimator(cal, extra),
            stride=1,
        )
        flagged = set(state.off_plane_frames())
        self.assertGreaterEqual(len(flagged & set(range(4, 8))), 3)
        self.assertFalse(flagged & set(range(0, 4)))

    def test_kinematics_units_stay_metres(self) -> None:
        rotation, tvec, cam = _tilted()
        corners = [
            Point2D(*_pinhole(x, y, 0.0, rotation, tvec, cam))
            for x, y in [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
        ]
        cal = planar_state(corners, width_m=2.0, height_m=1.0)
        u0, v0 = _pinhole(0.4, 0.3, 0.0, rotation, tvec, cam)
        u1, v1 = _pinhole(1.4, 0.3, 0.0, rotation, tvec, cam)
        result = TrackResult(
            clip_id="c",
            points=[TrackPoint(frame=0, x=u0, y=v0), TrackPoint(frame=1, x=u1, y=v1)],
        )
        info = VideoInfo(
            path=Path("clip.mp4"),
            width=640,
            height=360,
            pts=(0, 1),
            time_base=0.001,
            pts_ms=(0, 1000),
        )
        samples = series_for_result(result, info, calibration=cal)
        self.assertEqual(samples[0].position_unit, "m")
        self.assertAlmostEqual(samples[1].x or 0.0, 1.4, delta=0.02)
        self.assertAlmostEqual((samples[1].x or 0.0) - (samples[0].x or 0.0), 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
