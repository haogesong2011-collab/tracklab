"""Motion gate around SAM 2: Kalman, frame difference, memory drop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import PromptKind, TrackPoint, TrackResult  # noqa: E402
from ai.kinematics import is_low_confidence, quality_label, quality_tooltip, series_for_result  # noqa: E402
from ai.track_guard import (  # noqa: E402
    REJECT_BACKGROUND,
    ConstantVelocityKalman,
    MaskStats,
    PredictedBox,
    SamScores,
    drop_memory_frame,
    extract_sam_scores,
    mask_stats,
    motion_foreground,
    reprompt_candidate,
    score_mask,
)
from engine.video_index import VideoInfo  # noqa: E402


def _info(pts_ms: tuple[int, ...]) -> VideoInfo:
    return VideoInfo(
        path=Path("clip.mp4"),
        width=160,
        height=120,
        pts=tuple(range(len(pts_ms))),
        time_base=0.001,
        pts_ms=pts_ms,
    )


class KalmanTests(unittest.TestCase):
    def test_constant_acceleration_stays_near_the_ball(self) -> None:
        filt = ConstantVelocityKalman()
        dt = 1.0 / 30.0
        ax, ay = 400.0, -800.0
        x = y = vx = vy = 0.0
        errors: list[float] = []
        for i in range(24):
            t = i * dt
            if i:
                pred = filt.predict(t)
                assert pred is not None
                errors.append(float(np.hypot(pred.x - x, pred.y - y)))
            filt.update(t, x, y, 12.0, 12.0)
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
        self.assertLess(float(np.median(errors)), 20.0)
        self.assertLess(max(errors), 50.0)

    def test_uneven_pts_does_not_blow_up(self) -> None:
        filt = ConstantVelocityKalman()
        times = [0.0, 0.04, 0.05, 0.12, 0.20, 0.21, 0.33]
        x = 10.0
        for t in times:
            pred = filt.predict(t) if filt.initialized else None
            if pred is not None:
                self.assertLess(abs(pred.x - x), 30.0)
            filt.update(t, x, 40.0, 10.0, 10.0)
            x += 80.0 * (0.04 if t == times[0] else t - times[max(0, times.index(t) - 1)])


class MaskGateTests(unittest.TestCase):
    def test_area_jump_is_rejected(self) -> None:
        stats = MaskStats(x=40, y=40, w=80, h=80, area=6400, contour=[(0, 0), (80, 0), (80, 80), (0, 80)])
        pred = PredictedBox(x=40, y=40, w=12, h=12, step=4, sigma=4)
        decision = score_mask(
            stats, pred, None, SamScores(4.0, 0.9), median_area=80.0, updates=6, view_span=180
        )
        self.assertFalse(decision.accept)
        self.assertEqual(decision.reason, REJECT_BACKGROUND)

    def test_centroid_jump_with_growth_is_rejected(self) -> None:
        stats = MaskStats(
            x=200, y=20, w=40, h=40, area=400, contour=[(180, 0), (220, 0), (220, 40), (180, 40)]
        )
        pred = PredictedBox(x=40, y=40, w=12, h=12, step=4, sigma=4)
        decision = score_mask(
            stats, pred, None, SamScores(4.0, 0.9), median_area=80.0, updates=6, view_span=180
        )
        self.assertFalse(decision.accept)

    def test_fast_ball_is_kept_while_speed_is_unknown(self) -> None:
        stats = MaskStats(x=110, y=40, w=12, h=12, area=90, contour=[(104, 34), (116, 34), (116, 46), (104, 46)])
        pred = PredictedBox(x=40, y=40, w=12, h=12, step=8, sigma=8)
        decision = score_mask(
            stats, pred, None, SamScores(-0.4, 0.7), median_area=80.0, updates=1, view_span=360
        )
        self.assertTrue(decision.accept)

    def test_mild_negative_object_score_stays_visible(self) -> None:
        stats = MaskStats(x=40, y=40, w=12, h=12, area=80, contour=[(34, 34), (46, 34), (46, 46), (34, 46)])
        decision = score_mask(stats, None, None, SamScores(-2.0, 0.4), median_area=None, updates=1)
        self.assertTrue(decision.accept)
        self.assertGreaterEqual(decision.confidence, 0.72)

    def test_stable_mask_is_accepted(self) -> None:
        stats = MaskStats(x=42, y=41, w=12, h=12, area=90, contour=[(36, 35), (48, 35), (48, 47), (36, 47)])
        pred = PredictedBox(x=40, y=40, w=12, h=12, step=5, sigma=4)
        decision = score_mask(stats, pred, None, SamScores(3.0, 0.85), median_area=80.0)
        self.assertTrue(decision.accept)
        self.assertGreater(decision.confidence, 0.5)

    def test_mask_stats_empty(self) -> None:
        self.assertIsNone(mask_stats(np.zeros((8, 8), dtype=bool)))


class MotionForegroundTests(unittest.TestCase):
    def test_stripes_leave_only_the_moving_disk(self) -> None:
        h, w = 80, 120
        prev = np.zeros((h, w, 3), dtype=np.uint8)
        cur = np.zeros((h, w, 3), dtype=np.uint8)
        prev[:, 0::6] = (200, 210, 80)
        cur[:, 0::6] = (200, 210, 80)
        yy, xx = np.ogrid[:h, :w]
        prev[(xx - 30) ** 2 + (yy - 40) ** 2 <= 25] = (240, 40, 40)
        cur[(xx - 42) ** 2 + (yy - 38) ** 2 <= 25] = (240, 40, 40)
        fg = motion_foreground(prev, cur)
        self.assertTrue(fg.reliable)
        self.assertGreaterEqual(len(fg.blobs), 1)
        blob = min(fg.blobs, key=lambda b: abs(b.x * fg.scale - 42) + abs(b.y * fg.scale - 38))
        self.assertLess(abs(blob.x * fg.scale - 42), 8)
        prompt = reprompt_candidate(fg.blobs, PredictedBox(x=40, y=40, w=10, h=10, step=12), scale=fg.scale)
        self.assertIsNotNone(prompt)
        assert prompt is not None
        self.assertEqual(prompt.kind, PromptKind.POSITIVE)
        self.assertLess(abs(prompt.x - 42), 10)

    def test_large_camera_pan_is_unreliable(self) -> None:
        h, w = 64, 64
        prev = np.zeros((h, w), dtype=np.uint8)
        cur = np.zeros((h, w), dtype=np.uint8)
        prev[:, :20] = 200
        cur[:, 30:50] = 200
        fg = motion_foreground(prev, cur)
        self.assertFalse(fg.reliable)


class MemoryAndScoreTests(unittest.TestCase):
    def test_drop_memory_frame_removes_non_cond_output(self) -> None:
        state = {
            "output_dict": {"non_cond_frame_outputs": {3: {"pred_masks": 1}, 4: {"pred_masks": 1}}},
            "output_dict_per_obj": {
                0: {"non_cond_frame_outputs": {3: {"object_score_logits": 2.0}}}
            },
        }
        self.assertTrue(drop_memory_frame(state, 3))
        self.assertNotIn(3, state["output_dict"]["non_cond_frame_outputs"])
        self.assertIn(4, state["output_dict"]["non_cond_frame_outputs"])
        self.assertNotIn(3, state["output_dict_per_obj"][0]["non_cond_frame_outputs"])

    def test_extract_sam_scores_from_state_and_payload(self) -> None:
        payload = (2, [1], [np.zeros((1, 4, 4))], {"object_score_logits": 1.5, "iou_predictions": 0.8})
        scores = extract_sam_scores(payload, {}, 2, 1)
        self.assertAlmostEqual(scores.object_score or 0.0, 1.5)
        state = {
            "output_dict": {
                "non_cond_frame_outputs": {4: {"object_score_logits": -1.0, "iou_predictions": 0.2}}
            }
        }
        scores = extract_sam_scores((4, [1], [0]), state, 4, 1)
        self.assertAlmostEqual(scores.object_score or 0.0, -1.0)


class KinematicsRejectTests(unittest.TestCase):
    def test_rejected_point_is_yellow_in_table_logic(self) -> None:
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=10, y=10, visible=False, confidence=0.2, note=REJECT_BACKGROUND),
                TrackPoint(frame=1, x=12, y=10, visible=True, confidence=0.9),
            ],
        )
        samples = series_for_result(result, _info((0, 33)))
        self.assertTrue(is_low_confidence(samples[0]))
        self.assertEqual(quality_label(samples[0]), "已拒收")
        self.assertIn("已拒收：疑似跳到背景", quality_tooltip(samples[0]))
        self.assertFalse(is_low_confidence(samples[1]))


if __name__ == "__main__":
    unittest.main()
