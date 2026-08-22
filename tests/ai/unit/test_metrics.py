"""Unit tests for metrics and contracts using synthetic numbers (no video I/O)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import (  # noqa: E402
    CalibrationResult,
    CancelToken,
    FailureReason,
    PhysicsResult,
    PoseFrame,
    PoseKeypoint,
    PoseResult,
    TrackPoint,
    TrackResult,
)
from ai.schema import (  # noqa: E402
    POSE_KEYPOINTS,
    CalibrationGT,
    Difficulty,
    KeypointGT,
    PhysicsGT,
    Point2D,
    PoseFrameGT,
    TrackFrameGT,
)
from tests.ai.metrics import (  # noqa: E402
    calibration_metrics,
    physics_metrics,
    pose_metrics,
    summarize_pass,
    track_metrics,
)


class TrackMetricsTests(unittest.TestCase):
    def test_perfect_track_passes_normal(self) -> None:
        gt = [
            TrackFrameGT(frame=i, center=Point2D(10.0 + i, 20.0), visible=True)
            for i in range(20)
        ]
        pred = TrackResult(
            clip_id="t",
            points=[
                TrackPoint(frame=i, x=10.0 + i, y=20.0, visible=True) for i in range(20)
            ],
        )
        metrics = track_metrics(pred, gt, difficulty=Difficulty.NORMAL)
        self.assertTrue(summarize_pass(metrics))
        by_name = {m.name: m for m in metrics}
        self.assertEqual(by_name["center_median_px"].value, 0.0)
        self.assertEqual(by_name["success_at_10px"].value, 1.0)

    def test_occlusion_recovery_counted(self) -> None:
        gt = []
        for i in range(30):
            visible = not (10 <= i <= 14)
            gt.append(
                TrackFrameGT(
                    frame=i,
                    center=Point2D(float(i), 50.0),
                    visible=visible,
                    occluded=not visible,
                )
            )
        points = []
        for i in range(30):
            # Lost during occlusion, recovers at frame 18 (3 frames after GT returns).
            visible = i < 10 or i >= 18
            points.append(TrackPoint(frame=i, x=float(i), y=50.0, visible=visible))
        metrics = track_metrics(
            TrackResult(clip_id="t", points=points),
            gt,
            difficulty=Difficulty.HARD,
        )
        by_name = {m.name: m for m in metrics}
        self.assertGreaterEqual(by_name["occlusion_recovery_frames"].value, 3.0)


class PoseMetricsTests(unittest.TestCase):
    def test_perfect_pose_pck(self) -> None:
        gt_kps = [
            KeypointGT(name=n, x=float(i * 3), y=float(i * 2), visible=True)
            for i, n in enumerate(POSE_KEYPOINTS)
        ]
        # Ensure torso scale is non-trivial.
        gt_kps = [
            KeypointGT(name="nose", x=100, y=40, visible=True),
            KeypointGT(name="left_shoulder", x=80, y=60, visible=True),
            KeypointGT(name="right_shoulder", x=120, y=60, visible=True),
            KeypointGT(name="left_hip", x=85, y=110, visible=True),
            KeypointGT(name="right_hip", x=115, y=110, visible=True),
        ] + [
            KeypointGT(name=n, x=100, y=100, visible=True)
            for n in POSE_KEYPOINTS
            if n
            not in {
                "nose",
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
            }
        ]
        gt = [PoseFrameGT(frame=0, keypoints=gt_kps)]
        pred = PoseResult(
            clip_id="p",
            frames=[
                PoseFrame(
                    frame=0,
                    keypoints=[
                        PoseKeypoint(name=k.name, x=k.x, y=k.y, visible=True)
                        for k in gt_kps
                    ],
                )
            ],
        )
        metrics = pose_metrics(pred, gt, difficulty=Difficulty.NORMAL)
        by_name = {m.name: m for m in metrics}
        self.assertEqual(by_name["pck_at_0_05"].value, 1.0)
        self.assertTrue(by_name["pck_at_0_05"].passed)


class CalibrationMetricsTests(unittest.TestCase):
    def test_reject_when_no_ruler(self) -> None:
        gt = CalibrationGT(
            ruler_a=Point2D(0, 0),
            ruler_b=Point2D(100, 0),
            length_m=1.0,
            origin=Point2D(0, 0),
            axis_angle_deg=0.0,
            has_reliable_ruler=False,
        )
        good = CalibrationResult(
            clip_id="c",
            pixels_per_meter=None,
            origin_x=None,
            origin_y=None,
            axis_angle_deg=None,
            rejected=True,
            failure_reason=FailureReason.NO_RULER,
        )
        bad = CalibrationResult(
            clip_id="c",
            pixels_per_meter=100.0,
            origin_x=0,
            origin_y=0,
            axis_angle_deg=0,
            rejected=False,
        )
        self.assertTrue(
            summarize_pass(
                calibration_metrics(good, gt, difficulty=Difficulty.HARD)
            )
        )
        self.assertFalse(
            summarize_pass(calibration_metrics(bad, gt, difficulty=Difficulty.HARD))
        )

    def test_scale_error(self) -> None:
        gt = CalibrationGT(
            ruler_a=Point2D(0, 0),
            ruler_b=Point2D(100, 0),
            length_m=1.0,
            origin=Point2D(0, 0),
            axis_angle_deg=0.0,
        )
        pred = CalibrationResult(
            clip_id="c",
            pixels_per_meter=100.5,
            origin_x=0.5,
            origin_y=0.5,
            axis_angle_deg=0.2,
        )
        metrics = calibration_metrics(pred, gt, difficulty=Difficulty.NORMAL)
        self.assertTrue(summarize_pass(metrics))


class PhysicsMetricsTests(unittest.TestCase):
    def test_period_gate(self) -> None:
        gt = PhysicsGT(period_s=1.0, gravity_ms2=9.81)
        ok = PhysicsResult(clip_id="p", period_s=1.01, gravity_ms2=9.7)
        bad = PhysicsResult(clip_id="p", period_s=1.1, gravity_ms2=12.0)
        self.assertTrue(summarize_pass(physics_metrics(ok, gt)))
        self.assertFalse(summarize_pass(physics_metrics(bad, gt)))


class CancelTokenTests(unittest.TestCase):
    def test_cancel(self) -> None:
        token = CancelToken()
        self.assertFalse(token.cancelled)
        token.cancel()
        self.assertTrue(token.cancelled)


class TrackResultRoundtripTests(unittest.TestCase):
    def test_from_dict_roundtrip(self) -> None:
        original = TrackResult(
            clip_id="clip",
            points=[TrackPoint(frame=3, x=1.5, y=2.5, visible=True, confidence=0.8)],
            failure_reason=FailureReason.NONE,
            model_name="color_blob_tracker",
            model_version="0.1.0",
            elapsed_s=0.12,
        )
        restored = TrackResult.from_dict(original.to_dict())
        self.assertEqual(restored.clip_id, "clip")
        self.assertEqual(restored.points[0].frame, 3)
        self.assertAlmostEqual(restored.points[0].x, 1.5)
        self.assertEqual(restored.model_name, "color_blob_tracker")


if __name__ == "__main__":
    unittest.main()
