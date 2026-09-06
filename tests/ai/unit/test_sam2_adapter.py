"""SAM 2 adapter tests using a fake predictor — no weight download."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import (  # noqa: E402
    CancelToken,
    FailureReason,
    PromptKind,
    TrackPoint,
    TrackPrompt,
)
from ai.model_manager import (  # noqa: E402
    DEFAULT_SPEC,
    ModelNotAvailable,
    sha256_file,
    source_urls,
)
from ai.sam2_tracker import (  # noqa: E402
    FakeVideoPredictor,
    Sam2Tracker,
    logits_to_mask,
    mask_centroid,
    merge_track_points,
)
from ai.sam2_frames import shift_prompts, window_prompts  # noqa: E402
from tests.ai.dataset import load_annotation, load_manifest, resolve_video  # noqa: E402
from tests.ai.generate_fixtures import build_fixtures  # noqa: E402
from ai.models import load_video  # noqa: E402


def _ensure_fixtures() -> None:
    if not (ROOT / "datasets" / "manifest.json").exists():
        build_fixtures()


class MaskConversionTests(unittest.TestCase):
    def test_centroid_and_empty_mask(self) -> None:
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:10, 10:16] = True
        x, y, conf, visible, contour = mask_centroid(mask)
        self.assertTrue(visible)
        self.assertAlmostEqual(x, 12.5, places=5)
        self.assertAlmostEqual(y, 7.0, places=5)
        self.assertGreater(conf, 0.0)
        self.assertEqual(len(contour), 4)

        empty = np.zeros((8, 8), dtype=bool)
        x, y, conf, visible, contour = mask_centroid(empty)
        self.assertFalse(visible)
        self.assertEqual(conf, 0.0)
        self.assertEqual(contour, [])

    def test_logits_threshold(self) -> None:
        logits = np.array([[[-1.0, 2.0], [2.0, -1.0]]])
        mask = logits_to_mask(logits)
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[0, 0])

    def test_logits_accept_device_tensor(self) -> None:
        class HostTensor:
            def __init__(self, data: np.ndarray) -> None:
                self._data = data

            def detach(self) -> "HostTensor":
                return self

            def cpu(self) -> "HostTensor":
                return self

            def numpy(self) -> np.ndarray:
                return self._data

        logits = HostTensor(np.array([[[-1.0, 2.0], [2.0, -1.0]]]))
        mask = logits_to_mask(logits)
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[0, 0])

    def test_merge_keeps_prefix(self) -> None:
        existing = [
            TrackPoint(frame=0, x=1, y=1, manual=True),
            TrackPoint(frame=1, x=2, y=2),
            TrackPoint(frame=2, x=3, y=3),
        ]
        incoming = [
            TrackPoint(frame=1, x=9, y=9),
            TrackPoint(frame=2, x=8, y=8),
            TrackPoint(frame=3, x=7, y=7),
        ]
        merged = merge_track_points(existing, incoming, from_frame=1)
        self.assertEqual(merged[0].x, 1)
        self.assertTrue(merged[0].manual)
        self.assertEqual(merged[1].x, 9)
        self.assertEqual(merged[-1].frame, 3)

    def test_prompt_window_shift(self) -> None:
        prompts = [
            TrackPrompt(frame=4, kind=PromptKind.POSITIVE, x=1, y=2),
            TrackPrompt(frame=9, kind=PromptKind.NEGATIVE, x=3, y=4),
            TrackPrompt(frame=20, kind=PromptKind.POSITIVE, x=5, y=6),
        ]
        shifted = shift_prompts(prompts, start=4, last=10)
        self.assertEqual([p.frame for p in shifted], [0, 5])
        filled = window_prompts([], 8, 12, (11.0, 12.0))
        self.assertEqual(len(filled), 1)
        self.assertEqual(filled[0].frame, 0)
        self.assertEqual(filled[0].x, 11.0)


class FakeSam2TrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_fixtures()

    def _ball(self):
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        return info, ann

    def test_positive_prompt_and_range(self) -> None:
        info, ann = self._ball()
        seed = (ann.track[0].center.x, ann.track[0].center.y)
        tracker = Sam2Tracker(predictor=FakeVideoPredictor(radius=8, drift=(1.0, 0.0)))
        result = tracker.track(
            info,
            seed,
            start_frame=5,
            end_frame=14,
            prompts=[
                TrackPrompt(frame=5, kind=PromptKind.POSITIVE, x=seed[0], y=seed[1])
            ],
        )
        self.assertEqual(result.model_name, "sam2.1_hiera_tiny")
        self.assertEqual([p.frame for p in result.points], list(range(5, 15)))
        self.assertTrue(all(p.visible for p in result.points))
        self.assertGreater(result.points[-1].x, result.points[0].x)

    def test_pyav_loader_does_not_need_decord(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from ai.sam2_frames import load_frames_for_sam2

        info, _ann = self._ball()
        images, height, width = load_frames_for_sam2(
            info, 32, 1, 3, offload_video_to_cpu=True
        )
        self.assertIsNotNone(images)
        self.assertEqual(tuple(images.shape), (3, 3, 32, 32))
        self.assertEqual(height, info.height)
        self.assertEqual(width, info.width)

    def test_box_prompt_and_empty_mask(self) -> None:
        info, ann = self._ball()
        seed = (ann.track[0].center.x, ann.track[0].center.y)
        empty = Sam2Tracker(predictor=FakeVideoPredictor(radius=-1))
        lost = empty.track(
            info,
            seed,
            start_frame=0,
            end_frame=2,
            prompts=[
                TrackPrompt(
                    frame=0, kind=PromptKind.BOX, x=10, y=10, x2=40, y2=40
                )
            ],
        )
        self.assertTrue(all(not p.visible for p in lost.points))

    def test_object_isolation(self) -> None:
        info, ann = self._ball()
        a = FakeVideoPredictor(radius=6, drift=(2.0, 0.0))
        b = FakeVideoPredictor(radius=6, drift=(0.0, 2.0))
        seed = (ann.track[0].center.x, ann.track[0].center.y)
        ra = Sam2Tracker(predictor=a).track(info, seed, start_frame=0, end_frame=4)
        rb = Sam2Tracker(predictor=b).track(info, seed, start_frame=0, end_frame=4)
        self.assertNotAlmostEqual(ra.points[-1].x, rb.points[-1].x, places=2)

    def test_cancel_during_propagate(self) -> None:
        info, ann = self._ball()
        seed = (ann.track[0].center.x, ann.track[0].center.y)
        token = CancelToken()

        def progress(event) -> None:  # noqa: ANN001
            if event.current >= 2:
                token.cancel()

        result = Sam2Tracker(predictor=FakeVideoPredictor()).track(
            info, seed, start_frame=0, end_frame=20, cancel=token, progress=progress
        )
        self.assertEqual(result.failure_reason, FailureReason.CANCELLED)
        self.assertLess(len(result.points), 21)

    def test_missing_checkpoint_does_not_fallback(self) -> None:
        info, ann = self._ball()
        seed = (ann.track[0].center.x, ann.track[0].center.y)
        with tempfile.TemporaryDirectory() as tmp:
            import os

            os.environ["TRACKLAB_MODEL_DIR"] = tmp
            try:
                from ai.model_manager import ensure_checkpoint

                with self.assertRaises(ModelNotAvailable):
                    ensure_checkpoint(DEFAULT_SPEC, download=False)
                with self.assertRaises(ModelNotAvailable):
                    Sam2Tracker().track(info, seed, start_frame=0, end_frame=1)
            finally:
                os.environ.pop("TRACKLAB_MODEL_DIR", None)

    def test_source_urls_include_huggingface_mirror(self) -> None:
        urls = source_urls(DEFAULT_SPEC)
        self.assertTrue(urls[0].startswith("https://dl.fbaipublicfiles.com/"))
        self.assertTrue(any("huggingface.co/facebook/sam2.1-hiera-tiny" in u for u in urls))

    def test_bundled_checkpoint_used_when_frozen(self) -> None:
        import os
        from unittest import mock

        from ai import model_manager as mm

        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir()
            ckpt = models / DEFAULT_SPEC.filename
            ckpt.write_bytes(b"x")
            with mock.patch.object(mm.sys, "frozen", True, create=True), mock.patch.object(
                mm.sys, "_MEIPASS", tmp, create=True
            ):
                found = mm.bundled_checkpoint(DEFAULT_SPEC)
                self.assertEqual(found, ckpt)
                os.environ["TRACKLAB_MODEL_DIR"] = str(Path(tmp) / "unused")
                try:
                    self.assertEqual(mm.checkpoint_path(DEFAULT_SPEC), ckpt)
                finally:
                    os.environ.pop("TRACKLAB_MODEL_DIR", None)


if __name__ == "__main__":
    unittest.main()
