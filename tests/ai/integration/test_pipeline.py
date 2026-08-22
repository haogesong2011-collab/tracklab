"""Integration tests: decoder + AI models on synthetic fixtures (no Qt)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import CancelToken, FailureReason  # noqa: E402
from ai.models import (  # noqa: E402
    ColorBlobTracker,
    EdgeRulerCalibrator,
    OracleCalibrator,
    OracleTracker,
    load_video,
)
from tests.ai.dataset import load_annotation, load_manifest, resolve_video  # noqa: E402
from tests.ai.validate import validate_manifest  # noqa: E402
from tests.ai.generate_fixtures import build_fixtures  # noqa: E402
from tests.ai.metrics import summarize_pass, track_metrics  # noqa: E402


def _ensure_fixtures() -> None:
    manifest = ROOT / "datasets" / "manifest.json"
    fixtures = ROOT / "tests" / "ai" / "fixtures"
    if not manifest.exists() or not any(fixtures.glob("*.mp4")):
        build_fixtures()


class DecoderAIIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_fixtures()

    def test_manifest_annotations_are_valid(self) -> None:
        errors = validate_manifest()
        self.assertEqual(errors, [])
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        self.assertEqual(info.frame_count, ann.frame_count)
        self.assertEqual(info.width, ann.width)
        self.assertEqual(info.height, ann.height)

    def test_oracle_tracker_is_frame_accurate(self) -> None:
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_pendulum_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        result = OracleTracker(ann).track(
            info, (ann.track[0].center.x, ann.track[0].center.y)
        )
        self.assertEqual(len(result.points), len(ann.track))
        for p, g in zip(result.points, ann.track):
            self.assertEqual(p.frame, g.frame)
            self.assertAlmostEqual(p.x, g.center.x, places=5)
            self.assertAlmostEqual(p.y, g.center.y, places=5)
        self.assertTrue(
            summarize_pass(track_metrics(result, ann.track, difficulty=ann.difficulty))
        )

    def test_cancel_stops_baseline_tracker(self) -> None:
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        token = CancelToken()

        def progress(event) -> None:  # noqa: ANN001
            if event.current >= 3:
                token.cancel()

        result = ColorBlobTracker().track(
            info,
            (ann.track[0].center.x, ann.track[0].center.y),
            cancel=token,
            progress=progress,
        )
        self.assertEqual(result.failure_reason, FailureReason.CANCELLED)
        self.assertLess(len(result.points), info.frame_count)

    def test_calibrator_rejects_no_ruler(self) -> None:
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "calib_no_ruler")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        oracle = OracleCalibrator(ann).calibrate(info)
        baseline = EdgeRulerCalibrator().calibrate(
            info, expected_length_m=ann.calibration.length_m if ann.calibration else None
        )
        self.assertTrue(oracle.rejected)
        self.assertTrue(baseline.rejected)

    def test_color_blob_respects_frame_range(self) -> None:
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        result = ColorBlobTracker().track(
            info,
            (ann.track[0].center.x, ann.track[0].center.y),
            start_frame=10,
            end_frame=19,
        )
        self.assertEqual([p.frame for p in result.points], list(range(10, 20)))

    def test_fake_sam_repropagate_merges_from_correction_frame(self) -> None:
        from ai.contracts import PromptKind, TrackPrompt
        from ai.sam2_tracker import FakeVideoPredictor, Sam2Tracker, merge_track_points

        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        seed = (ann.track[0].center.x, ann.track[0].center.y)
        first = Sam2Tracker(predictor=FakeVideoPredictor(drift=(1.0, 0.0))).track(
            info, seed, start_frame=0, end_frame=12
        )
        correction = Sam2Tracker(predictor=FakeVideoPredictor(drift=(0.0, 1.0))).track(
            info,
            seed,
            start_frame=6,
            end_frame=12,
            prompts=[TrackPrompt(frame=6, kind=PromptKind.POSITIVE, x=seed[0], y=seed[1])],
        )
        merged = merge_track_points(first.points, correction.points, 6)
        self.assertEqual(merged[0].frame, 0)
        self.assertEqual(merged[0].x, first.points[0].x)
        self.assertEqual(merged[6].y, correction.points[0].y)
        self.assertEqual(info.pts_ms[merged[-1].frame], info.pts_ms[12])


if __name__ == "__main__":
    unittest.main()
