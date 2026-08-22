"""Synthetic translation / rotation / scale / occlusion / noise tests."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import TrackPoint, TrackResult  # noqa: E402
from ai.geometry import interpolate_xy, pixel_to_world, world_to_pixel  # noqa: E402
from ai.schema import Difficulty, Point2D, TrackFrameGT  # noqa: E402
from tests.ai.metrics import summarize_pass, track_metrics  # noqa: E402


def _scale(points: list[tuple[float, float]], s: float, origin: tuple[float, float]):
    ox, oy = origin
    return [(ox + s * (x - ox), oy + s * (y - oy)) for x, y in points]


class SyntheticTransformTests(unittest.TestCase):
    def test_translation_is_recovered_by_metrics(self) -> None:
        gt = [TrackFrameGT(frame=i, center=Point2D(float(i), 10.0)) for i in range(12)]
        pred = TrackResult(
            clip_id="t",
            points=[
                TrackPoint(frame=i, x=float(i) + 1.5, y=10.0, visible=True) for i in range(12)
            ],
        )
        metrics = {m.name: m for m in track_metrics(pred, gt, difficulty=Difficulty.NORMAL)}
        self.assertAlmostEqual(metrics["center_median_px"].value or 0, 1.5, places=5)
        self.assertTrue(metrics["success_at_10px"].passed)

    def test_rotation_roundtrip_world(self) -> None:
        origin = Point2D(100, 80)
        pts = [(110, 80), (100, 60), (80, 90)]
        for deg in (0, 30, 90, -45):
            for x, y in pts:
                w = pixel_to_world(x, y, origin=origin, pixels_per_meter=10, axis_angle_deg=deg)
                back = world_to_pixel(
                    w.x, w.y, origin=origin, pixels_per_meter=10, axis_angle_deg=deg
                )
                self.assertAlmostEqual(back.x, x, places=6)
                self.assertAlmostEqual(back.y, y, places=6)

    def test_scale_preserves_ratios(self) -> None:
        pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        scaled = _scale(pts, 2.0, (0.0, 0.0))
        self.assertEqual(scaled[1], (20.0, 0.0))
        self.assertEqual(scaled[2], (20.0, 20.0))

    def test_occlusion_gap_does_not_fail_completeness_on_hidden_frames(self) -> None:
        gt = []
        points = []
        for i in range(20):
            visible = not (8 <= i <= 12)
            gt.append(
                TrackFrameGT(
                    frame=i,
                    center=Point2D(float(i), 5.0),
                    visible=visible,
                    occluded=not visible,
                )
            )
            points.append(TrackPoint(frame=i, x=float(i), y=5.0, visible=visible))
        metrics = {
            m.name: m
            for m in track_metrics(
                TrackResult(clip_id="t", points=points), gt, difficulty=Difficulty.HARD
            )
        }
        self.assertTrue(metrics["trajectory_completeness"].passed)
        self.assertEqual(metrics["center_median_px"].value, 0.0)

    def test_noise_within_gate(self) -> None:
        gt = [TrackFrameGT(frame=i, center=Point2D(float(i * 2), 40.0)) for i in range(25)]
        pred = TrackResult(
            clip_id="t",
            points=[
                TrackPoint(frame=i, x=float(i * 2) + 0.4, y=40.3, visible=True)
                for i in range(25)
            ],
        )
        self.assertTrue(summarize_pass(track_metrics(pred, gt, difficulty=Difficulty.NORMAL)))

    def test_interpolate_boundaries(self) -> None:
        series = [(2, 2.0, 4.0), (6, 6.0, 8.0)]
        self.assertEqual(interpolate_xy(series, 0), (2.0, 4.0))
        self.assertEqual(interpolate_xy(series, 99), (6.0, 8.0))
        x, y = interpolate_xy(series, 4)  # type: ignore[misc]
        self.assertAlmostEqual(x, 4.0)
        self.assertAlmostEqual(y, 6.0)


if __name__ == "__main__":
    unittest.main()
