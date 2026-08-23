"""Deterministic experiment classification and fitting."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.calibration import uniform_state  # noqa: E402
from ai.contracts import ExperimentType, TrackPoint, TrackResult  # noqa: E402
from ai.physics import (  # noqa: E402
    analyze_experiment,
    source_fingerprint,
    video_info_from_fps,
)
from ai.schema import Point2D  # noqa: E402
from engine.video_index import VideoInfo  # noqa: E402


def _info(pts_ms: tuple[int, ...], fps_hint: float | None = None) -> VideoInfo:
    _ = fps_hint
    pts = tuple(range(len(pts_ms)))
    return VideoInfo(
        path=Path("clip.mp4"),
        width=320,
        height=180,
        pts=pts,
        time_base=0.001,
        pts_ms=pts_ms,
    )


def _track(points: list[TrackPoint], clip_id: str = "c") -> TrackResult:
    return TrackResult(clip_id=clip_id, points=points, confidence=1.0)


def _uniform_pts(n: int, fps: float = 30.0) -> tuple[int, ...]:
    return tuple(int(round(i * 1000.0 / fps)) for i in range(n))


def _cal(ppm: float = 100.0) -> object:
    return uniform_state(Point2D(0.0, 0.0), Point2D(ppm, 0.0), length_m=1.0)


class PhysicsEngineTests(unittest.TestCase):
    def test_uniform_linear_with_calibration(self) -> None:
        fps = 30.0
        n = 40
        v_px = 60.0
        ppm = 100.0
        points = [
            TrackPoint(frame=i, x=20 + v_px * (i / fps), y=90.0, confidence=0.95)
            for i in range(n)
        ]
        analysis = analyze_experiment(
            _track(points, "uniform"),
            _info(_uniform_pts(n, fps)),
            _cal(ppm),
            clip_id="uniform",
        )
        self.assertTrue(analysis.auto_confirmable)
        self.assertIsNotNone(analysis.selected)
        assert analysis.selected is not None
        self.assertEqual(analysis.selected.experiment_type, ExperimentType.UNIFORM_LINEAR)
        v = analysis.selected.fit.parameters["v"]
        assert v is not None
        self.assertLess(abs(abs(v) - v_px / ppm) / (v_px / ppm), 0.05)
        self.assertLessEqual(analysis.selected.fit.nrmse, 0.05)

    def test_uniform_accel(self) -> None:
        fps = 30.0
        n = 45
        v0, a, ppm = 30.0, 120.0, 100.0
        points = [
            TrackPoint(
                frame=i,
                x=20 + v0 * (i / fps) + 0.5 * a * (i / fps) ** 2,
                y=88.0,
                confidence=0.9,
            )
            for i in range(n)
        ]
        analysis = analyze_experiment(
            _track(points, "accel"),
            _info(_uniform_pts(n, fps)),
            _cal(ppm),
            clip_id="accel",
        )
        self.assertIsNotNone(analysis.selected)
        assert analysis.selected is not None
        self.assertEqual(analysis.selected.experiment_type, ExperimentType.UNIFORM_ACCEL)
        got = analysis.selected.fit.parameters["a"]
        assert got is not None
        self.assertLess(abs(got - a / ppm) / (a / ppm), 0.05)
        self.assertLessEqual(analysis.selected.fit.nrmse, 0.05)

    def test_free_fall(self) -> None:
        fps = 30.0
        n = 20
        g_px, ppm = 981.0, 100.0
        points = [
            TrackPoint(frame=i, x=160.0, y=20 + 0.5 * g_px * (i / fps) ** 2, confidence=1.0)
            for i in range(n)
        ]
        analysis = analyze_experiment(
            _track(points, "fall"),
            _info(_uniform_pts(n, fps)),
            _cal(ppm),
            clip_id="fall",
        )
        self.assertIsNotNone(analysis.selected)
        assert analysis.selected is not None
        self.assertEqual(analysis.selected.experiment_type, ExperimentType.FREE_FALL)
        g = analysis.selected.fit.parameters["g"]
        assert g is not None
        self.assertLess(abs(g - 9.81) / 9.81, 0.05)
        self.assertLessEqual(analysis.selected.fit.nrmse, 0.05)

    def test_projectile(self) -> None:
        fps = 30.0
        n = 18
        vx, vy, g_px, ppm = 80.0, 20.0, 981.0, 100.0
        points = [
            TrackPoint(
                frame=i,
                x=20 + vx * (i / fps),
                y=20 + vy * (i / fps) + 0.5 * g_px * (i / fps) ** 2,
                confidence=0.92,
            )
            for i in range(n)
        ]
        analysis = analyze_experiment(
            _track(points, "proj"),
            _info(_uniform_pts(n, fps)),
            _cal(ppm),
            clip_id="proj",
        )
        self.assertIsNotNone(analysis.selected)
        assert analysis.selected is not None
        self.assertEqual(analysis.selected.experiment_type, ExperimentType.PROJECTILE)
        params = analysis.selected.fit.parameters
        self.assertLess(abs((params["v0x"] or 0) - vx / ppm) / (vx / ppm), 0.05)
        self.assertLess(abs((params["g"] or 0) - 9.81) / 9.81, 0.05)
        self.assertLessEqual(analysis.selected.fit.nrmse, 0.05)

    def test_pendulum_period_and_g(self) -> None:
        fps = 30.0
        n = 60
        period = 1.2
        length_px = 100.0
        g = 9.81
        length_m = (period / (2 * math.pi)) ** 2 * g
        ppm = length_px / length_m
        origin_x, origin_y = 160.0, 20.0
        omega = 2 * math.pi / period
        amp = math.radians(25)
        points = []
        for i in range(n):
            t = i / fps
            theta = amp * math.cos(omega * t)
            points.append(
                TrackPoint(
                    frame=i,
                    x=origin_x + length_px * math.sin(theta),
                    y=origin_y + length_px * math.cos(theta),
                    confidence=0.97,
                )
            )
        analysis = analyze_experiment(
            _track(points, "pendulum"),
            _info(_uniform_pts(n, fps)),
            _cal(ppm),
            clip_id="pendulum",
            pendulum_length_m=length_m,
            period_hint=True,
        )
        self.assertIsNotNone(analysis.selected)
        assert analysis.selected is not None
        self.assertEqual(analysis.selected.experiment_type, ExperimentType.PENDULUM)
        t_hat = analysis.selected.fit.parameters["T"]
        g_hat = analysis.selected.fit.parameters["g"]
        assert t_hat is not None and g_hat is not None
        self.assertLess(abs(t_hat - period) / period, 0.02)
        self.assertLess(abs(g_hat - g) / g, 0.05)

    def test_irregular_pts_still_fits_uniform(self) -> None:
        pts_ms = (0, 40, 90, 160, 210, 300, 330, 410, 500, 540, 630, 700)
        v = 50.0
        points = [
            TrackPoint(frame=i, x=v * (ms / 1000.0), y=10.0)
            for i, ms in enumerate(pts_ms)
        ]
        analysis = analyze_experiment(
            _track(points),
            _info(pts_ms),
            _cal(100.0),
        )
        self.assertTrue(analysis.candidates)
        top = analysis.candidates[0]
        self.assertEqual(top.experiment_type, ExperimentType.UNIFORM_LINEAR)
        self.assertLess(abs(abs(top.fit.parameters["v"] or 0) - 0.5) / 0.5, 0.05)

    def test_uncalibrated_omits_si_g(self) -> None:
        fps = 30.0
        n = 20
        points = [
            TrackPoint(frame=i, x=160.0, y=20 + 0.5 * 400 * (i / fps) ** 2)
            for i in range(n)
        ]
        analysis = analyze_experiment(
            _track(points),
            video_info_from_fps(_track(points), fps),
            None,
        )
        self.assertIn("calibration", analysis.missing)
        self.assertTrue(analysis.candidates)
        fall = next(
            item
            for item in analysis.candidates
            if item.experiment_type is ExperimentType.FREE_FALL
        )
        self.assertIsNone(fall.fit.parameters.get("g"))
        self.assertEqual(analysis.position_unit, "px")

    def test_occlusion_uses_longest_segment(self) -> None:
        fps = 30.0
        points: list[TrackPoint] = []
        for i in range(30):
            visible = not (8 <= i <= 12)
            x = 10 + 40 * (i / fps) if visible else 0.0
            points.append(TrackPoint(frame=i, x=x, y=40.0, visible=visible))
        analysis = analyze_experiment(
            _track(points),
            _info(_uniform_pts(30, fps)),
            _cal(100.0),
        )
        self.assertTrue(analysis.candidates)
        fit = analysis.candidates[0].fit
        self.assertGreaterEqual(fit.frame_start, 13)
        self.assertLessEqual(fit.nrmse, 0.05)

    def test_low_confidence_noisy_data_not_auto_confirmed(self) -> None:
        rng = np.random.default_rng(0)
        points = [
            TrackPoint(
                frame=i,
                x=float(rng.normal(40, 25)),
                y=float(rng.normal(40, 25)),
                confidence=0.2,
            )
            for i in range(20)
        ]
        analysis = analyze_experiment(
            _track(points),
            _info(_uniform_pts(20)),
            _cal(100.0),
        )
        self.assertFalse(analysis.auto_confirmable)

    def test_model_ambiguity_rejects_auto_confirm(self) -> None:
        fps = 30.0
        n = 16
        # Almost linear with a tiny quadratic term — both models can look plausible.
        points = [
            TrackPoint(
                frame=i,
                x=10 + 50 * (i / fps) + 0.4 * (i / fps) ** 2,
                y=30.0,
            )
            for i in range(n)
        ]
        analysis = analyze_experiment(
            _track(points),
            _info(_uniform_pts(n, fps)),
            _cal(100.0),
        )
        types = {item.experiment_type for item in analysis.candidates}
        self.assertTrue(
            ExperimentType.UNIFORM_LINEAR in types or ExperimentType.UNIFORM_ACCEL in types
        )
        if analysis.auto_confirmable:
            self.assertIsNotNone(analysis.selected)

    def test_fingerprint_changes_with_track_and_calibration(self) -> None:
        points = [TrackPoint(frame=0, x=1, y=2), TrackPoint(frame=1, x=3, y=4)]
        track = _track(points)
        info = _info((0, 33))
        a = source_fingerprint(track, info, None)
        b = source_fingerprint(track, info, _cal(100.0))
        points2 = [TrackPoint(frame=0, x=1, y=2), TrackPoint(frame=1, x=9, y=4)]
        c = source_fingerprint(_track(points2), info, None)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_pendulum_without_length_omits_g(self) -> None:
        fps = 30.0
        n = 60
        period = 1.0
        omega = 2 * math.pi / period
        points = [
            TrackPoint(
                frame=i,
                x=160 + 40 * math.cos(omega * i / fps),
                y=80 + 4 * math.sin(omega * i / fps) ** 2,
            )
            for i in range(n)
        ]
        analysis = analyze_experiment(
            _track(points),
            _info(_uniform_pts(n, fps)),
            _cal(100.0),
            pendulum_length_m=None,
        )
        pendulum = next(
            item
            for item in analysis.candidates
            if item.experiment_type is ExperimentType.PENDULUM
        )
        self.assertIsNotNone(pendulum.fit.parameters.get("T"))
        self.assertIsNone(pendulum.fit.parameters.get("g"))
        self.assertIn("pendulum_length", analysis.missing)


if __name__ == "__main__":
    unittest.main()
