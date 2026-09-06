"""Sparse off-plane audit uses geometry as the measurement source."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.calibration import CameraProfile, planar_state  # noqa: E402
from ai.contracts import TrackPoint, TrackResult  # noqa: E402
from ai.depth_audit import (  # noqa: E402
    FakeDepthEstimator,
    audit_track,
    robust_center_depth,
    select_audit_frames,
)
from ai.schema import Point2D  # noqa: E402


def _plane():
    return planar_state(
        [Point2D(40, 40), Point2D(280, 50), Point2D(260, 150), Point2D(50, 140)],
        width_m=1.6,
        height_m=1.0,
        camera=CameraProfile(width=320, height=180, fx=400.0, fy=400.0, cx=160.0, cy=90.0),
    )


class DepthAuditTests(unittest.TestCase):
    def test_selects_ends_and_low_confidence(self) -> None:
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=80, y=80),
                TrackPoint(frame=3, x=90, y=80, confidence=0.4),
                TrackPoint(frame=8, x=100, y=80),
                TrackPoint(frame=16, x=110, y=80),
            ],
        )
        frames = select_audit_frames(result, stride=8)
        self.assertIn(0, frames)
        self.assertIn(16, frames)
        self.assertIn(3, frames)

    def test_robust_center_ignores_edge_outliers(self) -> None:
        depth = np.ones((20, 20), dtype=np.float64) * 2.0
        depth[0, :] = 80.0
        depth[:, 0] = 80.0
        value = robust_center_depth(depth, 10, 10, radius=6)
        self.assertIsNotNone(value)
        assert value is not None
        self.assertAlmostEqual(value, 2.0, places=2)

    def test_audit_flags_known_off_plane_bump(self) -> None:
        cal = _plane()
        self.assertEqual(cal.mode.value, "planar")
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=120, y=90),
                TrackPoint(frame=8, x=140, y=95),
            ],
        )

        def load_frame(index: int) -> np.ndarray:
            rgb = np.zeros((180, 320, 3), dtype=np.uint8)
            rgb[0, 0, 0] = index
            return rgb

        estimator = FakeDepthEstimator(cal, extra={8: 0.25})
        state = audit_track(result, cal, load_frame, estimator, stride=8)
        self.assertTrue(state.available)
        self.assertEqual(state.readings[0].flag, "ok")
        self.assertEqual(state.readings[8].flag, "off_plane")
        self.assertGreater(state.readings[8].residual_m or 0.0, 0.1)

    def test_moge_loader_is_lazy(self) -> None:
        import inspect

        import ai.depth_audit as module

        source = inspect.getsource(module)
        self.assertNotIn("import torch", source.split("def try_load_moge")[0])


if __name__ == "__main__":
    unittest.main()
