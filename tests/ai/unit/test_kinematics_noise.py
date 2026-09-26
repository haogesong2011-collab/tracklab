"""Velocity / acceleration on realistic, noisy tracks.

Built after the 45° launch export: a thin blurred dart at ~45 px/m and
29 fps, where the old ±0.1 s local quadratic gave aᵧ anywhere between −60
and +40 m/s², and the widest manual window still swung by 5 m/s².
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai import timebase  # noqa: E402
from ai.calibration import uniform_state  # noqa: E402
from ai.contracts import TrackPoint, TrackPointSource, TrackPointStatus, TrackResult  # noqa: E402
from ai.drag_fit import fit_drag  # noqa: E402
from ai.kinematics import (  # noqa: E402
    FIT_ACCEL,
    FIT_AUTO,
    law_caption,
    law_readout,
    FIT_DRAG,
    law_segments,
    FIT_POINT_WEIGHT,
    FIT_PROJECTILE,
    law_runs,
    measurement_weights,
    series_for_result,
    time_s_for_frame,
    trajectory_curves,
)
from ai.schema import Point2D  # noqa: E402
from engine.video_index import VideoInfo  # noqa: E402

PPM = 45.0  # pixels per metre, as in the launch clip


def _phone_info(fps: float, n: int, *, name: str = "phone", drop: tuple[int, ...] = ()) -> VideoInfo:
    """Timestamps the way an iPhone writes them: 1/600 s ticks."""
    slots = [k for k in range(n + len(drop)) if k not in drop][:n]
    pts = tuple(int(round(k * 600.0 / fps)) for k in slots)
    return VideoInfo(
        path=Path(f"/clips/{name}_{fps}_{n}.mov"),
        width=1280,
        height=720,
        pts=pts,
        time_base=1.0 / 600.0,
        pts_ms=tuple(int(round((p - pts[0]) / 600.0 * 1000.0)) for p in pts),
    )


def _drag_flight(t: np.ndarray, g: float = 9.8, k: float = 0.028, v0=(-9.4, 10.7)):
    """Positions / accelerations of a projectile with quadratic drag (fine RK4)."""
    h = 1e-4
    grid = np.arange(0.0, t[-1] + h, h)
    s = np.array([0.0, 0.0, *v0])
    out = np.empty((len(grid), 4))
    for i in range(len(grid)):
        out[i] = s

        def f(st):
            v = np.hypot(st[2], st[3])
            return np.array([st[2], st[3], -k * v * st[2], -g - k * v * st[3]])

        k1 = f(s)
        k2 = f(s + h / 2 * k1)
        k3 = f(s + h / 2 * k2)
        k4 = f(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    state = np.column_stack([np.interp(t, grid, out[:, j]) for j in range(4)])
    speed = np.hypot(state[:, 2], state[:, 3])
    accel = np.column_stack([-k * speed * state[:, 2], -g - k * speed * state[:, 3]])
    return state[:, :2], state[:, 2:], accel


def _to_points(xy_m: np.ndarray, rng: np.random.Generator, *, noise_px=1.0, slides=False) -> list[TrackPoint]:
    img = np.column_stack([900.0 + xy_m[:, 0] * PPM, 500.0 - xy_m[:, 1] * PPM])
    err = rng.normal(0.0, noise_px, img.shape)
    if slides:
        # The point slides along a blurred object for a few frames at a time.
        i = 0
        while i < len(img):
            run = int(rng.integers(3, 7))
            if rng.random() < 0.4:
                err[i : i + run] += rng.normal(0.0, 5.0, 2)
            i += run
    img = img + err
    return [
        TrackPoint(frame=i, x=float(x), y=float(y), confidence=0.85)
        for i, (x, y) in enumerate(img)
    ]


def _calibration():
    return uniform_state(Point2D(0, 0), Point2D(PPM, 0), length_m=1.0, origin=Point2D(900.0, 500.0))


def _column(samples, name):
    return np.array([np.nan if getattr(s, name) is None else getattr(s, name) for s in samples], dtype=float)


class TimebaseTests(unittest.TestCase):
    def test_rounded_phone_ticks_become_a_uniform_grid(self) -> None:
        info = _phone_info(29.0, 120)
        steps_ms = np.diff(np.asarray(info.pts_ms))
        self.assertGreater(len(set(steps_ms.tolist())), 1)  # 34/35 ms jitter in pts_ms
        times = timebase.frame_times(info)
        assert times is not None
        self.assertTrue(timebase.is_dejittered(info))
        self.assertLess(float(np.ptp(np.diff(times))), 1e-9)
        self.assertAlmostEqual(1.0 / float(np.diff(times)[0]), 29.0, places=3)
        self.assertAlmostEqual(time_s_for_frame(info, 0), 0.0)

    def test_a_dropped_frame_keeps_the_grid(self) -> None:
        info = _phone_info(29.0, 80, name="drop", drop=(30,))
        times = timebase.frame_times(info)
        assert times is not None
        self.assertTrue(timebase.is_dejittered(info))
        self.assertAlmostEqual(float(times[30] - times[29]), 2.0 / 29.0, places=5)

    def test_slow_motion_ticks(self) -> None:
        info = _phone_info(240.0, 300, name="slomo")
        self.assertTrue(timebase.is_dejittered(info))
        self.assertAlmostEqual(1.0 / float(np.diff(timebase.frame_times(info))[5]), 240.0, delta=0.05)

    def test_variable_frame_rate_keeps_its_timestamps(self) -> None:
        rng = np.random.default_rng(0)
        pts = tuple(int(v) for v in np.cumsum(rng.integers(17, 25, 60)))
        info = VideoInfo(
            path=Path("/clips/vfr.mov"),
            width=64,
            height=64,
            pts=pts,
            time_base=1.0 / 600.0,
            pts_ms=tuple(int(round((p - pts[0]) / 0.6)) for p in pts),
        )
        self.assertFalse(timebase.is_dejittered(info))
        times = timebase.frame_times(info)
        assert times is not None
        np.testing.assert_allclose(times, (np.asarray(pts) - pts[0]) / 600.0)


class NoisyProjectileTests(unittest.TestCase):
    def _series(self, points, info, step=0):
        return series_for_result(
            TrackResult(clip_id="launch", points=points), info, calibration=_calibration(), velocity_step=step
        )

    def test_default_acceleration_is_usable_on_a_noisy_dart(self) -> None:
        n = 58
        info = _phone_info(29.0, n)
        t = timebase.frame_times(info)
        pos, _vel, accel = _drag_flight(t)
        errors = []
        for seed in range(6):
            samples = self._series(_to_points(pos, np.random.default_rng(seed), slides=True), info)
            ay = _column(samples, "ay")
            self.assertFalse(np.isnan(ay).any())
            errors.append(float(np.sqrt(np.mean((ay - accel[:, 1]) ** 2))))
        # The old default (±0.1 s) was off by ~27 m/s² RMS on the same tracks.
        self.assertLess(float(np.mean(errors)), 3.0)

    def test_clean_free_fall_gives_g_everywhere_including_the_ends(self) -> None:
        n = 40
        info = _phone_info(60.0, n, name="fall")
        t = timebase.frame_times(info)
        pos = np.column_stack([0.3 * t, -4.9 * t * t])
        samples = self._series(_to_points(pos, np.random.default_rng(1), noise_px=0.3), info)
        ay = _column(samples, "ay")
        self.assertLess(float(np.max(np.abs(ay + 9.8))), 0.8)
        # First and last frame reuse the last full window instead of a one-sided fit.
        self.assertLess(abs(ay[0] + 9.8), 0.8)
        self.assertLess(abs(ay[-1] + 9.8), 0.8)
        self.assertIsNotNone(samples[n // 2].sigma_ay)
        self.assertGreater(samples[n // 2].window_a_s or 0.0, samples[n // 2].window_v_s or 0.0 - 1e-9)

    def test_manual_window_is_still_honoured(self) -> None:
        info = _phone_info(30.0, 30, name="manual")
        t = timebase.frame_times(info)
        pos = np.column_stack([t, -4.9 * t * t])
        samples = self._series(_to_points(pos, np.random.default_rng(2), noise_px=0.0), info, step=3)
        self.assertAlmostEqual(samples[10].window_v_s or 0.0, 0.14, places=3)
        self.assertAlmostEqual(samples[10].window_a_s or 0.0, 0.35, places=3)
        self.assertAlmostEqual(samples[10].ay or 0.0, -9.8, delta=1e-6)

    def test_an_accepted_fit_point_counts_less_than_a_click(self) -> None:
        auto = TrackPoint(frame=0, x=0.0, y=0.0, confidence=0.9)
        fit = TrackPoint(frame=1, x=0.0, y=0.0, manual=True, source=TrackPointSource.MANUAL, diagnostics={"fit_accepted": True})
        click = TrackPoint(frame=2, x=0.0, y=0.0, manual=True, source=TrackPointSource.MANUAL)
        weights = measurement_weights([auto, fit, click], [0, 1, 2])
        self.assertAlmostEqual(weights[0], 0.9)
        self.assertAlmostEqual(weights[1], FIT_POINT_WEIGHT)
        self.assertAlmostEqual(weights[2], 1.0)

    def test_a_partial_mask_counts_less(self) -> None:
        points = [
            TrackPoint(frame=i, x=0.0, y=0.0, confidence=0.9, diagnostics={"area": 100.0})
            for i in range(9)
        ]
        points[4] = TrackPoint(frame=4, x=0.0, y=0.0, confidence=0.9, diagnostics={"area": 25.0})
        weights = measurement_weights(points, list(range(9)))
        self.assertAlmostEqual(weights[0], 0.9)
        self.assertLess(weights[4], 0.2)

    def test_a_short_review_gap_does_not_create_run_edges(self) -> None:
        n = 45
        info = _phone_info(30.0, n, name="gap")
        t = timebase.frame_times(info)
        pos = np.column_stack([t, -4.9 * t * t])
        points = _to_points(pos, np.random.default_rng(4), noise_px=0.0)
        points[22] = TrackPoint(frame=22, x=points[22].x, y=points[22].y, status=TrackPointStatus.REVIEW)
        samples = self._series(points, info, step=3)
        self.assertIsNone(samples[22].ay)
        for i in (20, 21, 23, 24):
            self.assertAlmostEqual(samples[i].ay or 0.0, -9.8, delta=1e-6)


class DragFitTests(unittest.TestCase):
    def test_drag_fit_separates_g_from_air_drag(self) -> None:
        n = 58
        info = _phone_info(29.0, n, name="dragfit")
        t = timebase.frame_times(info)
        pos, vel, accel = _drag_flight(t)
        samples = series_for_result(
            TrackResult(clip_id="d", points=_to_points(pos, np.random.default_rng(5), noise_px=1.0)),
            info,
            calibration=_calibration(),
        )
        runs = law_runs(samples, FIT_DRAG, "ay")
        self.assertEqual(len(runs), 1)
        law = runs[0][2]
        self.assertAlmostEqual(law.fit.gravity, 9.8, delta=0.3)
        self.assertAlmostEqual(law.fit.k, 0.028, delta=0.01)
        self.assertLess(law.fit.gravity_sigma, 0.3)
        mid = n // 2
        self.assertAlmostEqual(law.evaluate(t[mid]), accel[mid, 1], delta=0.4)
        text = law.equation("aᵧ")
        self.assertIn("g = ", text)
        self.assertIn("±", text)
        vx_law = law_runs(samples, FIT_DRAG, "vx")[0][2]
        self.assertAlmostEqual(vx_law.evaluate(t[mid]), vel[mid, 0], delta=0.3)

    def test_no_drag_reduces_to_the_parabola(self) -> None:
        t = np.arange(30) / 30.0
        fit = fit_drag(t, 3.0 * t, 4.0 * t - 4.9 * t * t)
        assert fit is not None
        self.assertAlmostEqual(fit.gravity, 9.8, places=4)
        self.assertLess(fit.k, 1e-6)
        self.assertAlmostEqual(fit.evaluate("vy", 0.5), 4.0 - 9.8 * 0.5, places=4)

    def test_video_curve_in_image_coordinates(self) -> None:
        t = [i / 30.0 for i in range(30)]
        xs = [100.0 + 300.0 * s for s in t]
        ys = [400.0 - (400.0 * s - 490.0 * s * s) for s in t]
        curves = trajectory_curves(t, xs, ys, FIT_DRAG)
        self.assertEqual(len(curves), 1)
        x_end, y_end = curves[0][-1]
        self.assertAlmostEqual(x_end, xs[-1], delta=0.5)
        self.assertAlmostEqual(y_end, ys[-1], delta=0.5)

    def test_too_few_points_gives_no_fit(self) -> None:
        self.assertIsNone(fit_drag([0.0, 0.1, 0.2], [0, 1, 2], [0, 1, 0]))

    def test_projectile_law_reports_an_uncertainty(self) -> None:
        n = 40
        info = _phone_info(30.0, n, name="projsig")
        t = timebase.frame_times(info)
        pos = np.column_stack([2.0 * t, 3.0 * t - 4.9 * t * t])
        samples = series_for_result(
            TrackResult(clip_id="p", points=_to_points(pos, np.random.default_rng(6), noise_px=1.0)),
            info,
            calibration=_calibration(),
        )
        law = law_runs(samples, FIT_PROJECTILE, "ay")[0][2]
        self.assertIn("±", law.equation("aᵧ"))
        self.assertAlmostEqual(law.evaluate(0.5), -9.8, delta=0.5)


class ModelDerivativeTests(unittest.TestCase):
    """v / a from the chosen motion law (图表「v/a：按函数」)."""

    def _flight(self, seed=7, slides=True):
        n = 58
        info = _phone_info(29.0, n, name=f"model{seed}")
        t = timebase.frame_times(info)
        pos, vel, accel = _drag_flight(t)
        points = _to_points(pos, np.random.default_rng(seed), slides=slides)
        return info, t, vel, accel, points

    def test_drag_law_gives_a_straight_vx_and_a_gentle_ax(self) -> None:
        info, t, vel, accel, points = self._flight()
        local = series_for_result(TrackResult(clip_id="m", points=points), info, calibration=_calibration())
        model = series_for_result(
            TrackResult(clip_id="m", points=points), info, calibration=_calibration(), derivative_model=FIT_DRAG
        )
        ax_local = _column(local, "ax")
        ax_model = _column(model, "ax")
        vx_model = _column(model, "vx")
        # Drag only slows the horizontal motion: |vx| falls monotonically, ax keeps one sign.
        self.assertTrue(np.all(np.diff(np.abs(vx_model)) < 1e-9))
        self.assertTrue(np.all(ax_model > 0))
        self.assertLess(float(np.max(np.abs(np.diff(ax_model)))), 0.5)
        self.assertLess(float(np.sqrt(np.mean((ax_model - accel[:, 0]) ** 2))), float(np.sqrt(np.mean((ax_local - accel[:, 0]) ** 2))))
        self.assertEqual(model[10].kinematics_source, FIT_DRAG)
        self.assertAlmostEqual(model[10].local_ax or 0.0, ax_local[10])
        self.assertIsNotNone(model[10].sigma_ax)
        # Positions are still the measurements.
        self.assertEqual(model[10].x, local[10].x)

    def test_polynomial_laws_give_their_derivatives_with_error_bars(self) -> None:
        info, t, _vel, _accel, points = self._flight(seed=8, slides=False)
        samples = series_for_result(
            TrackResult(clip_id="m", points=points), info, calibration=_calibration(), derivative_model=FIT_PROJECTILE
        )
        vx = _column(samples, "vx")
        self.assertLess(float(np.ptp(vx)), 1e-9)  # projectile: vx constant
        self.assertTrue(np.allclose(_column(samples, "ax"), 0.0))
        ay = _column(samples, "ay")
        self.assertLess(float(np.ptp(ay)), 1e-9)
        self.assertIsNotNone(samples[5].sigma_ay)
        accel = series_for_result(
            TrackResult(clip_id="m", points=points), info, calibration=_calibration(), derivative_model=FIT_ACCEL
        )
        self.assertIsNotNone(accel[30].sigma_vy)

    def test_pending_frames_get_chart_values_but_stay_out_of_the_measurement(self) -> None:
        info, t, _vel, _accel, points = self._flight(seed=9, slides=False)
        bad = points[20]
        points[20] = TrackPoint(frame=20, x=bad.x + 40.0, y=bad.y, confidence=0.3, status=TrackPointStatus.REVIEW)
        good = (bad.x, bad.y)
        samples = series_for_result(
            TrackResult(clip_id="m", points=points), info, calibration=_calibration(), pending_positions={20: good}
        )
        pending = samples[20]
        self.assertTrue(pending.pending)
        self.assertIsNone(pending.x)
        self.assertIsNone(pending.ax)
        self.assertAlmostEqual(pending.review_value("x") or 0.0, (good[0] - 900.0) / PPM, places=6)
        self.assertIsNotNone(pending.review_value("vx"))
        self.assertIsNotNone(pending.review_value("ay"))
        self.assertFalse(samples[19].pending)
        # The bad candidate (40 px off) must not leak into its neighbours.
        clean = series_for_result(
            TrackResult(clip_id="m", points=[p for p in points if p.frame != 20]), info, calibration=_calibration()
        )
        neighbour = next(s for s in clean if s.frame == 21)
        self.assertAlmostEqual(samples[21].ax or 0.0, neighbour.ax or 0.0, places=6)

    def test_review_frames_do_not_cut_a_law_but_a_lost_target_does(self) -> None:
        info, _t, _vel, _accel, points = self._flight(seed=10, slides=False)
        points[20] = TrackPoint(frame=20, x=points[20].x, y=points[20].y, status=TrackPointStatus.REVIEW)
        samples = series_for_result(TrackResult(clip_id="m", points=points), info, calibration=_calibration())
        self.assertEqual(len(law_segments(samples, "x")), 1)
        points[30] = TrackPoint(frame=30, x=0.0, y=0.0, visible=False, status=TrackPointStatus.LOST)
        samples = series_for_result(TrackResult(clip_id="m", points=points), info, calibration=_calibration())
        self.assertEqual(len(law_segments(samples, "x")), 2)

    def test_csv_says_where_v_and_a_came_from(self) -> None:
        try:
            from ai.desktop import export_track_csv
        except ImportError:  # the desktop module needs PySide6
            self.skipTest("PySide6 not installed")
        import csv
        import tempfile

        info, _t, _vel, _accel, points = self._flight(seed=11, slides=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            export_track_csv(
                path, TrackResult(clip_id="m", points=points), info, calibration=_calibration(), derivative_model=FIT_DRAG
            )
            rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        self.assertEqual(rows[10]["kinematics_source"], FIT_DRAG)
        self.assertNotEqual(rows[10]["ax_local_m_s2"], "")
        self.assertNotEqual(rows[10]["sigma_ax_m_s2"], "")


class AutoFitTests(unittest.TestCase):
    """函数「自动」: overall velocity / acceleration from a whole-run fit."""

    def _series(self, pos, info, model=FIT_AUTO, seed=0, noise=1.0):
        points = _to_points(pos, np.random.default_rng(seed), noise_px=noise)
        return series_for_result(
            TrackResult(clip_id="a", points=points), info, calibration=_calibration(), derivative_model=model
        )

    def test_straight_x_and_parabolic_y(self) -> None:
        info = _phone_info(30.0, 45, name="auto1")
        t = timebase.frame_times(info)
        pos = np.column_stack([-7.7 * t, 10.0 * t - 4.9 * t * t])
        samples = self._series(pos, info)
        x_law = law_runs(samples, FIT_AUTO, "x")[0][2]
        y_law = law_runs(samples, FIT_AUTO, "y")[0][2]
        self.assertEqual(x_law.degree, 1)
        self.assertEqual(y_law.degree, 2)
        self.assertAlmostEqual(_column(samples, "vx")[10], -7.7, delta=0.1)
        self.assertTrue(np.allclose(_column(samples, "ax"), 0.0))
        self.assertAlmostEqual(_column(samples, "ay")[10], -9.8, delta=0.4)
        caption = law_caption(
            x_law, "x", "x", t0=float(t[0]), position_unit="m", speed_unit="m/s", accel_unit="m/s²"
        )
        self.assertIn("总体速度", caption)
        self.assertIn("±", caption)
        caption_y = law_caption(
            y_law, "y", "y", t0=float(t[0]), position_unit="m", speed_unit="m/s", accel_unit="m/s²"
        )
        self.assertIn("总体加速度", caption_y)
        self.assertIn("τ", caption_y)
        readout = dict(law_readout(y_law, "y", float(t[5])))
        self.assertAlmostEqual(readout["vᵧ"], 10.0 - 9.8 * float(t[5]), delta=0.3)

    def test_a_swing_is_not_forced_onto_a_parabola(self) -> None:
        info = _phone_info(30.0, 120, name="auto2")
        t = timebase.frame_times(info)
        w = 2 * np.pi / 1.4
        pos = np.column_stack([0.3 * np.sin(w * t), 0.03 * np.cos(2 * w * t)])
        samples = self._series(pos, info, noise=0.3)
        self.assertEqual(law_runs(samples, FIT_AUTO, "x"), [])
        # The axis without a law keeps its per-frame values.
        local = series_for_result(
            TrackResult(clip_id="a", points=_to_points(pos, np.random.default_rng(0), noise_px=0.3)),
            info,
            calibration=_calibration(),
        )
        self.assertAlmostEqual(samples[60].ax or 0.0, local[60].ax or 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
