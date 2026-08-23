"""Kinematics derived from TrackResult + VideoInfo PTS (display only)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.calibration import uniform_state, near_far_state, RulerSegment, RulerRole  # noqa: E402
from ai.contracts import TrackPoint, TrackResult  # noqa: E402
from ai.kinematics import (  # noqa: E402
    contiguous_segments,
    fit_quantity,
    sample_at_frame,
    series_for_result,
)
from ai.schema import Point2D  # noqa: E402
from engine.video_index import VideoInfo  # noqa: E402


def _info(pts_ms: tuple[int, ...]) -> VideoInfo:
    pts = tuple(range(len(pts_ms)))
    return VideoInfo(
        path=Path("clip.mp4"),
        width=640,
        height=360,
        pts=pts,
        time_base=0.001,
        pts_ms=pts_ms,
    )


class KinematicsTests(unittest.TestCase):
    def test_center_and_end_differences(self) -> None:
        info = _info((0, 100, 200, 300))
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=0, y=0),
                TrackPoint(frame=1, x=10, y=0),
                TrackPoint(frame=2, x=20, y=0),
                TrackPoint(frame=3, x=30, y=0),
            ],
        )
        samples = series_for_result(result, info)
        self.assertAlmostEqual(samples[0].vx_px_s, 100.0)  # (10-0)/0.1
        self.assertAlmostEqual(samples[1].vx_px_s, 100.0)  # (20-0)/0.2
        self.assertAlmostEqual(samples[2].vx_px_s, 100.0)
        self.assertAlmostEqual(samples[3].vx_px_s, 100.0)
        self.assertAlmostEqual(samples[1].speed_px_s, 100.0)

    def test_irregular_pts_and_speed_magnitude(self) -> None:
        info = _info((0, 50, 250))
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=0, y=0),
                TrackPoint(frame=1, x=3, y=4),
                TrackPoint(frame=2, x=6, y=8),
            ],
        )
        samples = series_for_result(result, info)
        self.assertAlmostEqual(samples[0].time_s, 0.0)
        self.assertAlmostEqual(samples[1].time_s, 0.05)
        self.assertAlmostEqual(samples[2].time_s, 0.25)
        self.assertAlmostEqual(samples[1].speed_px_s, 40.0)  # hypot(6,8)/0.25

    def test_hidden_and_occlusion_break_velocity(self) -> None:
        info = _info((0, 100, 200, 300, 400))
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=0, y=0),
                TrackPoint(frame=1, x=10, y=0),
                TrackPoint(frame=2, x=99, y=99, visible=False),
                TrackPoint(frame=3, x=30, y=0),
                TrackPoint(frame=4, x=40, y=0),
            ],
        )
        samples = series_for_result(result, info)
        hidden = sample_at_frame(samples, 2)
        assert hidden is not None
        self.assertIsNone(hidden.x_px)
        self.assertIsNone(hidden.vx_px_s)
        self.assertAlmostEqual(samples[1].vx_px_s, 100.0)  # end of first run
        self.assertAlmostEqual(samples[3].vx_px_s, 100.0)  # start of second run
        self.assertAlmostEqual(samples[4].vx_px_s, 100.0)
        x_segments = contiguous_segments(samples, "x_px")
        self.assertEqual(len(x_segments), 2)
        self.assertEqual([s.frame for s in x_segments[0]], [0, 1])
        self.assertEqual([s.frame for s in x_segments[1]], [3, 4])

    def test_single_point_and_zero_dt(self) -> None:
        info = _info((10, 10, 20))
        single = series_for_result(
            TrackResult(clip_id="c", points=[TrackPoint(frame=0, x=1, y=2)]),
            info,
        )
        self.assertEqual(len(single), 1)
        self.assertIsNone(single[0].vx_px_s)
        self.assertIsNone(single[0].speed_px_s)

        zero_dt = series_for_result(
            TrackResult(
                clip_id="c",
                points=[
                    TrackPoint(frame=0, x=0, y=0),
                    TrackPoint(frame=1, x=5, y=0),
                ],
            ),
            info,
        )
        self.assertIsNone(zero_dt[0].vx_px_s)
        self.assertIsNone(zero_dt[1].vx_px_s)

    def test_missing_info_falls_back_to_fps(self) -> None:
        result = TrackResult(
            clip_id="c",
            points=[TrackPoint(frame=0, x=0, y=0), TrackPoint(frame=30, x=30, y=0)],
        )
        samples = series_for_result(result, None)
        self.assertAlmostEqual(samples[1].time_s, 1.0)
        self.assertAlmostEqual(samples[1].vx_px_s, 30.0)

    def test_world_units_then_velocity(self) -> None:
        info = _info((0, 100, 200))
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=0, y=100),
                TrackPoint(frame=1, x=100, y=100),
                TrackPoint(frame=2, x=200, y=100),
            ],
        )
        cal = uniform_state(
            Point2D(0, 100),
            Point2D(100, 100),
            length_m=1.0,
            origin=Point2D(0, 100),
            axis_angle_deg=0.0,
        )
        samples = series_for_result(result, info, calibration=cal)
        self.assertEqual(samples[0].position_unit, "m")
        self.assertEqual(samples[0].speed_unit, "m/s")
        self.assertAlmostEqual(samples[0].x or 0.0, 0.0, places=5)
        self.assertAlmostEqual(samples[1].x or 0.0, 1.0, places=5)
        self.assertAlmostEqual(samples[1].vx or 0.0, 10.0, places=5)

    def test_falling_velocity_is_negative_by_default(self) -> None:
        info = _info((0, 100, 200))
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=10, y=0),
                TrackPoint(frame=1, x=10, y=10),
                TrackPoint(frame=2, x=10, y=20),
            ],
        )
        raw = series_for_result(result, info)
        self.assertLess(raw[1].vy or 0.0, 0.0)

        scaled = uniform_state(Point2D(0, 0), Point2D(100, 0), length_m=1.0)
        scaled_samples = series_for_result(result, info, calibration=scaled)
        self.assertLess(scaled_samples[1].vy or 0.0, 0.0)

        scaled.frame.origin_x = 10.0
        scaled.frame.origin_y = 0.0
        world = series_for_result(result, info, calibration=scaled)
        self.assertTrue(scaled.frame.y_up)
        self.assertLess(world[1].vy or 0.0, 0.0)

        scaled.frame.y_up = False
        image = series_for_result(result, info, calibration=scaled)
        self.assertGreater(image[1].vy or 0.0, 0.0)

    def test_occlusion_does_not_cross_segments_in_world(self) -> None:
        info = _info((0, 100, 200, 300, 400))
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=0, y=0),
                TrackPoint(frame=1, x=10, y=0),
                TrackPoint(frame=2, x=99, y=99, visible=False),
                TrackPoint(frame=3, x=30, y=0),
                TrackPoint(frame=4, x=40, y=0),
            ],
        )
        cal = uniform_state(Point2D(0, 0), Point2D(10, 0), length_m=1.0, origin=Point2D(0, 0))
        samples = series_for_result(result, info, calibration=cal)
        hidden = sample_at_frame(samples, 2)
        assert hidden is not None
        self.assertIsNone(hidden.x)
        self.assertIsNone(hidden.vx)
        self.assertEqual(len(contiguous_segments(samples, "x")), 2)

    def test_near_far_speed_uses_world_delta(self) -> None:
        info = _info((0, 1000))
        near = RulerSegment(Point2D(0, 0), Point2D(100, 0), length_m=1.0, role=RulerRole.NEAR)
        far = RulerSegment(Point2D(25, 100), Point2D(75, 100), length_m=1.0, role=RulerRole.FAR)
        cal = near_far_state(near, far, origin=Point2D(0, 0), axis_angle_deg=0.0)
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=0, y=0),
                TrackPoint(frame=1, x=100, y=0),
            ],
        )
        samples = series_for_result(result, info, calibration=cal)
        self.assertEqual(samples[0].position_unit, "m")
        self.assertAlmostEqual(samples[1].x or 0.0, 1.0, places=3)
        self.assertAlmostEqual(samples[1].vx or 0.0, 1.0, places=3)

    def test_projectile_apex_vy_zero_and_linear(self) -> None:
        fps = 50.0
        n = 21
        v0y, g, ppm = 2.0, 10.0, 100.0
        apex = 10
        pts = tuple(int(round(i * 1000.0 / fps)) for i in range(n))
        info = _info(pts)
        points = []
        for i in range(n):
            t = i / fps
            # Image y grows downward; +y up display is v0y t − 0.5 g t².
            y_img = 200.0 - (v0y * t - 0.5 * g * t * t) * ppm
            points.append(TrackPoint(frame=i, x=float(i), y=y_img))
        result = TrackResult(clip_id="proj", points=points)
        cal = uniform_state(Point2D(0, 0), Point2D(ppm, 0), length_m=1.0)
        samples = series_for_result(result, info, calibration=cal, velocity_step=3)
        self.assertEqual(samples[apex].frame, apex)
        self.assertIsNotNone(samples[apex].vy)
        self.assertIsNotNone(samples[apex - 3].vy)
        self.assertIsNotNone(samples[apex + 3].vy)
        self.assertAlmostEqual(samples[apex].vy, 0.0, places=5)
        self.assertGreater(samples[apex - 3].vy, 0.0)
        self.assertLess(samples[apex + 3].vy, 0.0)
        dt = 1.0 / fps
        self.assertAlmostEqual(
            samples[apex + 3].vy - samples[apex].vy,
            -g * 3 * dt,
            places=4,
        )

    def test_velocity_step_averages_jitter(self) -> None:
        info = _info(tuple(range(0, 900, 100)))
        jitter = (0, 3, -2, 4, -1, 2, -3, 1, 0)
        points = [
            TrackPoint(frame=i, x=float(10 * i + jitter[i]), y=0.0)
            for i in range(9)
        ]
        result = TrackResult(clip_id="c", points=points)
        noisy = series_for_result(result, info, velocity_step=1)
        smooth = series_for_result(result, info, velocity_step=3)
        mid = 4
        self.assertGreater(
            abs((noisy[mid].vx or 0.0) - 100.0),
            abs((smooth[mid].vx or 0.0) - 100.0),
        )

    def test_quantity_linear_and_quadratic_fit(self) -> None:
        info = _info(tuple(i * 100 for i in range(8)))
        result = TrackResult(
            clip_id="c",
            points=[TrackPoint(frame=i, x=float(2 + 5 * i), y=float(i * i)) for i in range(8)],
        )
        samples = series_for_result(result, info, velocity_step=1)
        line = fit_quantity(samples, "x", 1)
        assert line is not None
        self.assertGreater(line.r2, 0.999)
        self.assertIn("x =", line.equation("x"))
        quad = fit_quantity(samples, "y", 2)
        assert quad is not None
        self.assertGreater(quad.r2, 0.999)
        self.assertAlmostEqual(quad.evaluate(samples[3].time_s), samples[3].y or 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
