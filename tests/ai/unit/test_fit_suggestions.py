from __future__ import annotations

import unittest

import numpy as np

from ai.contracts import TrackPoint, TrackPointSource, TrackPointStatus, TrackResult
from ai.fit_suggestions import accept_all, accept_suggestion, suggest_points

FPS = 30.0


class _Info:
    fps = FPS
    pts_ms = tuple(int(round(i * 1000 / FPS)) for i in range(400))


def _truth(i: int) -> tuple[float, float]:
    t = i / FPS
    return 900.0 - 420.0 * t, 470.0 - 600.0 * t + 480.0 * t * t


def _track(n: int = 40, bad: dict[int, str] | None = None, noise: float = 0.0) -> list[TrackPoint]:
    rng = np.random.default_rng(1)
    bad = bad or {}
    out = []
    for i in range(n):
        x, y = _truth(i)
        x += rng.normal(0, noise)
        y += rng.normal(0, noise)
        kind = bad.get(i)
        if kind == "lost":
            out.append(TrackPoint(i, 0.0, 0.0, visible=False, confidence=0.0, status=TrackPointStatus.LOST))
        elif kind == "review":
            out.append(TrackPoint(i, x + 30, y - 25, visible=False, confidence=0.6, status=TrackPointStatus.REVIEW))
        else:
            out.append(TrackPoint(i, x, y))
    return out


class FitSuggestionTests(unittest.TestCase):
    def test_gaps_get_accurate_interpolated_suggestions(self) -> None:
        points = _track(bad={10: "lost", 11: "review", 12: "review", 25: "review"}, noise=1.0)
        sug = suggest_points(points, _Info)
        self.assertEqual(sorted(sug), [10, 11, 12, 25])
        for f, s in sug.items():
            tx, ty = _truth(f)
            self.assertLess(float(np.hypot(s.x - tx, s.y - ty)), 3.0, f)
            self.assertEqual(s.method, "interpolate")
            self.assertGreater(s.sigma_px, 0.0)

    def test_trusted_and_manual_frames_get_nothing(self) -> None:
        points = _track()
        points[5] = TrackPoint(5, 1.0, 1.0, visible=True, manual=True, source=TrackPointSource.MANUAL)
        self.assertEqual(suggest_points(points, _Info), {})

    def test_extrapolation_is_short_and_flagged(self) -> None:
        points = _track(n=30, bad={i: "lost" for i in range(24, 30)})
        sug = suggest_points(points, _Info)
        self.assertEqual(sorted(sug), [24, 25, 26])  # at most 3 frames past the end
        self.assertTrue(all(s.method == "extrapolate" for s in sug.values()))
        self.assertLess(sug[24].sigma_px, sug[26].sigma_px)

    def test_a_long_gap_without_neighbours_gets_nothing(self) -> None:
        points = _track(n=60, bad={i: "lost" for i in range(20, 45)})
        sug = suggest_points(points, _Info)
        self.assertFalse(any(26 <= f <= 38 for f in sug))

    def test_accepting_marks_a_user_confirmed_point_and_keeps_the_original(self) -> None:
        points = _track(bad={11: "review"})
        result = TrackResult(clip_id="c", points=points)
        s = suggest_points(points, _Info)[11]
        accepted = accept_suggestion(result, s)
        p = next(q for q in accepted.points if q.frame == 11)
        self.assertTrue(p.usable_for_measurement())
        self.assertTrue(p.manual)
        self.assertIs(p.source, TrackPointSource.MANUAL)
        self.assertTrue(p.diagnostics["fit_accepted"])
        self.assertEqual(p.diagnostics["original_status"], "review")
        self.assertIn("拟合点", p.note)
        # The original result is untouched (undo relies on it).
        self.assertFalse(next(q for q in result.points if q.frame == 11).usable_for_measurement())

    def test_accept_all_uses_only_real_measurements(self) -> None:
        points = _track(bad={10: "lost", 11: "lost", 12: "lost"})
        result = TrackResult(clip_id="c", points=points)
        sug = suggest_points(points, _Info)
        accepted = accept_all(result, sug)
        for f in (10, 11, 12):
            p = next(q for q in accepted.points if q.frame == f)
            self.assertAlmostEqual(p.x, sug[f].x)
            self.assertEqual(p.diagnostics["fit"]["points"], sug[f].points)


if __name__ == "__main__":
    unittest.main()
