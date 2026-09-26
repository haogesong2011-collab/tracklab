"""Round 2 of R3: thin blurred projectile over rails and bystanders.

Built from the 45° launch clip failure: the point stopped on the top of a
yellow railing post for three frames (then jumped 25-35 px) and later sat on
a bystander's head, all while every single check still passed. These tests
use scripted masks and synthetic frames; they check rules, not SAM.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ai.track_guard import (
    AppearanceEvidence,
    LocalTrajectory,
    PredictedBox,
    SamScores,
    TrajectoryEvidence,
    color_signature,
    color_similarity,
    mask_components,
    region_change,
    score_mask,
    trajectory_tolerance,
)
from engine.video_index import VideoInfo

H, W = 240, 400
FPS = 30.0


def _info(n: int) -> VideoInfo:
    pts = tuple(i * 3000 for i in range(n))
    return VideoInfo(
        path=Path("/nonexistent/launch.mp4"),
        width=W,
        height=H,
        pts=pts,
        time_base=1 / 90000,
        pts_ms=tuple(int(round(p / 90)) for p in pts),
    )


def _parabola(i: int) -> tuple[float, float]:
    t = i / FPS
    return 60.0 + 300.0 * t, 200.0 - 330.0 * t + 450.0 * t * t


class _Scene:
    """Grey streak on a parabola; a static yellow rail; a static dark head."""

    rail = (slice(118, 126), slice(0, W))  # horizontal yellow bar
    post = (slice(96, 126), slice(236, 244))  # vertical post, top at y=96

    def __init__(self, n: int = 24) -> None:  # stays inside the frame
        self.n = n
        self.info = _info(n)
        base = np.full((H, W, 3), (92, 96, 100), dtype=np.uint8)
        base[self.rail] = (228, 190, 40)
        base[self.post] = (228, 190, 40)
        base[190:204, 330:344] = (40, 34, 30)  # bystander's hair
        self.base = base
        self.frames: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        rng = np.random.default_rng(4)
        for i in range(n):
            img = base.copy()
            cx, cy = _parabola(i)
            mask = np.zeros((H, W), dtype=bool)
            ys, xs = np.ogrid[:H, :W]
            # A 3 px wide, 14 px long streak along the motion.
            mask |= (np.abs((xs - cx) * 0.6 + (ys - cy) * 0.8) <= 7) & (
                np.abs((xs - cx) * 0.8 - (ys - cy) * 0.6) <= 1.5
            )
            img[mask] = (150, 156, 150)
            img = np.clip(img.astype(int) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
            self.frames.append(img)
            self.masks.append(mask)


def _guard(scene: _Scene, projectile: bool = False):
    from ai.sam2_tracker import _GuardSession

    guard = _GuardSession(scene.info, True, _parabola(0))
    guard._rgb = lambda frame: scene.frames[frame]  # type: ignore[method-assign]
    guard.projectile = projectile
    return guard


GOOD = SamScores(object_score=6.0, iou=None)
WEAK = SamScores(object_score=1.0, iou=None)


class TrajectoryTests(unittest.TestCase):
    def test_parabola_is_predicted_and_one_outlier_is_ignored(self) -> None:
        traj = LocalTrajectory()
        for i in range(10):
            x, y = _parabola(i)
            if i == 6:
                x, y = x + 30.0, y - 25.0  # one bad committed point
            traj.add(i, i / FPS, x, y)
        pred = traj.predict(10, 10 / FPS)
        assert pred is not None
        tx, ty = _parabola(10)
        self.assertLess(float(np.hypot(pred[0] - tx, pred[1] - ty)), 1.5)

    def test_no_prediction_after_a_long_gap(self) -> None:
        traj = LocalTrajectory()
        for i in range(8):
            traj.add(i, i / FPS, *_parabola(i))
        self.assertIsNotNone(traj.predict(9, 9 / FPS))
        self.assertIsNone(traj.predict(12, 12 / FPS))  # 5 frames of extrapolation

    def test_tolerance_widens_with_extrapolation_and_is_capped_by_size(self) -> None:
        self.assertGreater(trajectory_tolerance(2.0, 10.0, gap=3), trajectory_tolerance(2.0, 10.0, gap=1))
        # A large fit residual cannot blow the gate up beyond the object size.
        self.assertLessEqual(trajectory_tolerance(50.0, 10.0), 10.0)


class ColorAndChangeTests(unittest.TestCase):
    def test_grey_streak_and_yellow_rail_differ_but_brightness_does_not_matter(self) -> None:
        img = np.zeros((20, 60, 3), dtype=np.uint8)
        img[:, :20] = (150, 156, 150)  # grey, lit
        img[:, 20:40] = (60, 63, 60)  # the same grey in shade
        img[:, 40:] = (228, 190, 40)  # rail
        m = np.zeros((20, 60), dtype=bool)
        comps = []
        for x0 in (0, 20, 40):
            mm = m.copy()
            mm[5:15, x0 + 5 : x0 + 15] = True
            comps.append(mask_components(mm)[0])
        lit, shade, rail = (color_signature(img, c) for c in comps)
        self.assertGreater(color_similarity(lit, shade) or 0.0, 0.8)
        self.assertLess(color_similarity(lit, rail) or 1.0, 0.05)

    def test_static_candidate_does_not_change(self) -> None:
        scene = _Scene()
        post_mask = np.zeros((H, W), dtype=bool)
        post_mask[96:104, 236:244] = True
        comp = mask_components(post_mask)[0]
        self.assertLess(region_change(scene.frames[12], scene.frames[10], comp) or 99.0, 6.0)
        moving = mask_components(scene.masks[12])[0]
        self.assertGreater(region_change(scene.frames[12], scene.frames[10], moving) or 0.0, 30.0)

    def test_camera_shift_is_compensated(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
        b = np.zeros_like(a)
        b[:, 5:] = a[:, :-5]  # camera panned: content moved 5 px right
        m = np.zeros((80, 80), dtype=bool)
        m[30:40, 30:40] = True
        comp = mask_components(m)[0]
        self.assertGreater(region_change(b, a, comp) or 0.0, 20.0)
        compensated = region_change(b, a, comp, shift=(5.0, 0.0))
        assert compensated is not None
        self.assertLess(compensated, 1e-6)


class ScoreMaskRuleTests(unittest.TestCase):
    def _stats(self):
        m = np.zeros((H, W), dtype=bool)
        m[100:106, 200:206] = True
        return mask_components(m)[0]

    def _decide(self, *, dev=0.0, tol=6.0, weak_score=False, projectile=False, color=0.9, change=None):
        stats = self._stats()
        return score_mask(
            stats,
            PredictedBox(x=stats.x, y=stats.y, w=6.0, h=6.0, step=10.0, sigma=4.0),
            None,
            WEAK if weak_score else GOOD,
            36.0,
            AppearanceEvidence(similarity=0.7, color_similarity=color, change_ratio=change),
            updates=8,
            trajectory=TrajectoryEvidence(dev, tol, projectile=projectile, points=8),
            reference_score=6.0,
        )

    def test_off_trajectory_alone_is_accepted_outside_projectile_mode(self) -> None:
        self.assertTrue(self._decide(dev=40.0).accept)  # a bounce or a collision

    def test_off_trajectory_alone_is_review_in_projectile_mode(self) -> None:
        d = self._decide(dev=40.0, projectile=True)
        self.assertFalse(d.accept)
        self.assertEqual(d.diagnostics["reason_code"], "trajectory_outlier")

    def test_off_trajectory_with_a_weak_model_score_is_review(self) -> None:
        d = self._decide(dev=12.0, weak_score=True)
        self.assertFalse(d.accept)
        self.assertEqual(d.diagnostics["reason_code"], "trajectory_weak")

    def test_on_trajectory_with_one_weak_cue_is_accepted(self) -> None:
        self.assertTrue(self._decide(dev=3.0, weak_score=True).accept)

    def test_two_weak_cues_on_the_trajectory_stay_a_flagged_measurement(self) -> None:
        d = self._decide(dev=3.0, weak_score=True, change=0.1)
        self.assertTrue(d.accept)
        self.assertIn("weak_evidence", d.diagnostics["suspect"])

    def test_two_weak_cues_off_the_trajectory_are_review(self) -> None:
        d = self._decide(dev=7.5, weak_score=True, change=0.1)
        self.assertFalse(d.accept)
        self.assertEqual(d.diagnostics["reason_code"], "weak_evidence")
        self.assertEqual(set(d.diagnostics["weak_evidence"]), {"model_score", "static"})

    def test_colour_alone_never_votes(self) -> None:
        # A blurred streak over a yellow rail looks yellow; the real clip had a
        # correct frame at colour similarity 0.05.
        d = self._decide(dev=0.5, color=0.02)
        self.assertTrue(d.accept)
        self.assertNotIn("color", d.diagnostics.get("weak_evidence", []))

class RailScenarioTests(unittest.TestCase):
    def test_clean_flight_is_trusted_every_frame(self) -> None:
        scene = _Scene()
        guard = _guard(scene)
        for i in range(scene.n):
            d, *_ = guard.observe(i, scene.masks[i], GOOD)
            self.assertTrue(d.accept, (i, d.reason, d.diagnostics.get("weak_evidence")))

    def test_mask_stuck_on_the_rail_post_is_not_trusted_and_recovery_backfills(self) -> None:
        scene = _Scene()
        guard = _guard(scene)
        post = np.zeros((H, W), dtype=bool)
        post[96:103, 236:243] = True  # SAM grabbed the top of the post
        decisions = {}
        for i in range(scene.n):
            if i in (14, 15):
                d, *_ = guard.observe(i, post, WEAK)
            else:
                d, *_ = guard.observe(i, scene.masks[i], GOOD)
            decisions[i] = d
            if i == 17 and guard.backfill is not None:
                decisions["backfilled"] = guard.backfill.frame
        for i in (14, 15):
            self.assertFalse(decisions[i].accept, i)
            self.assertIn(
                decisions[i].diagnostics.get("reason_code"),
                {"trajectory_weak", "weak_evidence", "trajectory_outlier", "color_conflict", "local_conflict"},
            )
        self.assertEqual(decisions[16].diagnostics.get("recovery"), "pending")
        self.assertTrue(decisions[17].accept)
        self.assertEqual(decisions.get("backfilled"), 16)
        self.assertTrue(all(decisions[i].accept for i in range(18, scene.n)))

    def test_bystander_head_near_the_path_is_not_trusted(self) -> None:
        scene = _Scene()
        guard = _guard(scene)
        head = np.zeros((H, W), dtype=bool)
        head[190:204, 330:344] = True
        for i in range(22):
            guard.observe(i, scene.masks[i], GOOD)
        d, *_ = guard.observe(22, head, SamScores(object_score=1.0))
        self.assertFalse(d.accept, d.diagnostics)

    def test_projectile_mode_aims_the_relocalisation_at_the_trajectory(self) -> None:
        scene = _Scene()
        guard = _guard(scene, projectile=True)
        empty = np.zeros((H, W), dtype=bool)
        for i in range(12):
            guard.observe(i, scene.masks[i], GOOD)
        prompt = None
        for i in (12, 13):
            _d, _s, pred, _fg = guard.observe(i, empty, SamScores(object_score=-9.0))
            prompt = guard.maybe_reprompt(i, pred, None) or prompt
        assert prompt is not None
        tx, ty = _parabola(13)
        self.assertLess(float(np.hypot(prompt.x - tx, prompt.y - ty)), 6.0)


if __name__ == "__main__":
    unittest.main()
