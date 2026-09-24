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

from ai.model_manager import (  # noqa: E402
    DEFAULT_SPEC,
    SAM21_SMALL,
    ModelNotAvailable,
    sha256_file,
    source_urls,
    spec_for_mode,
)
from ai.contracts import (  # noqa: E402
    CancelToken,
    FailureReason,
    PromptKind,
    TrackMode,
    TrackPoint,
    TrackPrompt,
)
from ai.sam2_frames import (  # noqa: E402
    OFFLOAD_STATE_FRAMES,
    densify_track_points,
    sampled_frame_indices,
    shift_prompts,
    window_prompts,
)
from ai.sam2_tracker import (  # noqa: E402
    FakeVideoPredictor,
    Sam2Tracker,
    logits_to_mask,
    mask_centroid,
    merge_track_points,
    sample_track_windows,
)
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
        small = source_urls(SAM21_SMALL)
        self.assertTrue(any("huggingface.co/facebook/sam2.1-hiera-small" in u for u in small))
        self.assertEqual(SAM21_SMALL.config, "configs/sam2.1/sam2.1_hiera_s.yaml")
        self.assertEqual(
            SAM21_SMALL.sha256,
            "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        )
        self.assertEqual(spec_for_mode(TrackMode.FAST).model_id, "sam2.1_hiera_tiny")
        self.assertEqual(spec_for_mode(TrackMode.PRECISE).model_id, "sam2.1_hiera_small")

    def test_sha256_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


class StrideDensifyTests(unittest.TestCase):
    def test_sampled_indices_include_prompts_and_last(self) -> None:
        frames = sampled_frame_indices(0, 9, stride=3, extra=[1])
        self.assertEqual(frames, [0, 1, 3, 6, 9])

    def test_densify_interpolates_visible_run_only(self) -> None:
        sampled = [
            TrackPoint(frame=0, x=0.0, y=0.0, visible=True),
            TrackPoint(frame=2, x=2.0, y=0.0, visible=True),
            TrackPoint(frame=4, x=4.0, y=0.0, visible=False),
            TrackPoint(frame=6, x=6.0, y=0.0, visible=False),
        ]
        filled = densify_track_points(sampled, 0, 6)
        by_frame = {point.frame: point for point in filled}
        self.assertTrue(by_frame[1].interpolated)
        self.assertTrue(by_frame[1].visible)
        self.assertAlmostEqual(by_frame[1].x, 1.0)
        self.assertFalse(by_frame[3].visible)
        self.assertFalse(by_frame[3].interpolated)
        self.assertFalse(by_frame[5].visible)

    def test_stride_marks_interpolated_and_keeps_all_frames(self) -> None:
        _ensure_fixtures()
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        seed = (ann.track[0].center.x, ann.track[0].center.y)
        tracker = Sam2Tracker(predictor=FakeVideoPredictor(radius=8, drift=(1.0, 0.0)))
        result = tracker.track(
            info,
            seed,
            start_frame=0,
            end_frame=4,
            stride=2,
            prompts=[TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=seed[0], y=seed[1])],
        )
        self.assertEqual([p.frame for p in result.points], list(range(0, 5)))
        sampled = [p for p in result.points if not p.interpolated]
        interp = [p for p in result.points if p.interpolated]
        self.assertEqual([p.frame for p in sampled], [0, 2, 4])
        self.assertEqual([p.frame for p in interp], [1, 3])
        self.assertTrue(all(p.visible for p in result.points))

    def test_stride_does_not_interpolate_across_occlusion(self) -> None:
        _ensure_fixtures()
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        seed = (ann.track[0].center.x, ann.track[0].center.y)
        tracker = Sam2Tracker(
            predictor=FakeVideoPredictor(radius=8, drift=(1.0, 0.0), blank_from=4)
        )
        result = tracker.track(
            info,
            seed,
            start_frame=0,
            end_frame=6,
            stride=2,
            prompts=[TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=seed[0], y=seed[1])],
        )
        by_frame = {p.frame: p for p in result.points}
        self.assertTrue(by_frame[1].interpolated)
        self.assertTrue(by_frame[1].visible)
        self.assertFalse(by_frame[4].visible)
        self.assertFalse(by_frame[3].visible)
        self.assertFalse(by_frame[3].interpolated)

    def test_prompt_frame_is_forced_onto_sample_grid(self) -> None:
        frames = sampled_frame_indices(0, 9, stride=3, extra=[1, 8])
        self.assertIn(1, frames)
        self.assertIn(8, frames)

    def test_catmull_rom_is_exact_on_a_parabola(self) -> None:
        def y_of(frame: int) -> float:
            return 100.0 + 20.0 * frame - 0.8 * frame * frame

        stride = 3
        sampled = [
            TrackPoint(frame=f, x=5.0 * f, y=y_of(f), visible=True)
            for f in range(0, 25, stride)
        ]
        filled = {p.frame: p for p in densify_track_points(sampled, 0, 24)}
        # Frames 1/2 and 22/23 miss an outer control point and stay linear.
        inner = [f for f in range(4, 21) if filled[f].interpolated]
        self.assertTrue(inner)
        for frame in inner:
            self.assertAlmostEqual(filled[frame].y, y_of(frame), places=6)
        linear_sag = abs(
            (y_of(0) + (y_of(3) - y_of(0)) / 3.0) - y_of(1)
        )
        self.assertGreater(linear_sag, 1.0)

    def test_densify_does_not_bridge_an_occluded_sample(self) -> None:
        sampled = [
            TrackPoint(frame=0, x=0.0, y=0.0, visible=True),
            TrackPoint(frame=2, x=0.0, y=0.0, visible=False),
            TrackPoint(frame=4, x=40.0, y=0.0, visible=True),
        ]
        filled = {p.frame: p for p in densify_track_points(sampled, 0, 4)}
        self.assertFalse(filled[1].visible)
        self.assertFalse(filled[1].interpolated)
        self.assertFalse(filled[3].visible)
        self.assertFalse(filled[3].interpolated)

    def test_windows_overlap_by_exactly_one_frame(self) -> None:
        sampled = list(range(0, 10))
        self.assertEqual(sample_track_windows(sampled, 20), [sampled])
        windows = sample_track_windows(sampled, 4)
        self.assertEqual(windows[0], [0, 1, 2, 3])
        self.assertEqual(windows[1], [3, 4, 5, 6])
        for prev, nxt in zip(windows, windows[1:]):
            self.assertEqual(prev[-1], nxt[0])
        self.assertEqual(sorted({f for w in windows for f in w}), sampled)


class LazyFrameTests(unittest.TestCase):
    def _info(self):
        _ensure_fixtures()
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        return load_video(resolve_video(entry))

    def test_lazy_frames_match_the_eager_stack(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from ai.sam2_frames import LazyFrameTensors, load_frames_for_sam2

        info = self._info()
        indices = [0, 2, 4, 1]
        eager, _h, _w = load_frames_for_sam2(
            info, 32, 0, 4, offload_video_to_cpu=True, frame_indices=indices
        )
        lazy = LazyFrameTensors(info, indices, 32, cache=2)
        try:
            self.assertEqual(len(lazy), len(indices))
            for i in range(len(indices)):
                self.assertTrue(torch.allclose(lazy[i], eager[i], atol=1e-6))
            # Re-reading an evicted index must decode the same pixels again.
            self.assertTrue(torch.allclose(lazy[0], eager[0], atol=1e-6))
        finally:
            lazy.close()

    def test_lazy_frames_keep_only_the_cache_in_memory(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from ai.sam2_frames import LazyFrameTensors

        info = self._info()
        lazy = LazyFrameTensors(info, list(range(8)), 32, cache=3)
        try:
            for i in range(8):
                _ = lazy[i]
            self.assertLessEqual(len(lazy._cache), 3)
        finally:
            lazy.close()

    def test_state_offloads_to_cpu_only_for_long_clips(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from ai.sam2_frames import build_video_state

        class Recorder:
            image_size = 8
            device = torch.device("cpu")

            def _get_image_feature(self, state, frame_idx, batch_size):  # noqa: ANN001
                _ = state["images"][frame_idx]
                return None

        short = build_video_state(
            Recorder(), torch.zeros(4, 3, 8, 8), 10, 10
        )
        self.assertFalse(short["offload_state_to_cpu"])
        long = build_video_state(
            Recorder(), torch.zeros(OFFLOAD_STATE_FRAMES, 3, 8, 8), 10, 10
        )
        self.assertTrue(long["offload_state_to_cpu"])


class _FakeSam2Like:
    """Looks like SAM2VideoPredictor so Sam2Tracker takes the windowed path."""

    image_size = 32

    def __init__(
        self,
        info,  # noqa: ANN001
        *,
        jump_local: set[int] | None = None,
        jump_first_window_last: bool = False,
        truth: list[tuple[float, float]] | None = None,
    ) -> None:
        import torch

        self.device = torch.device("cpu")
        self._info = info
        self.windows: list[int] = []
        self.prompt_kinds: list[list[str]] = []
        self.boxes: list[list[float]] = []
        self._cx = 0.0
        self._cy = 0.0
        self.jump_local = set(jump_local or [])
        self.jump_first_window_last = jump_first_window_last
        self.truth = truth
        self._first_window_jumped = False

    def _get_image_feature(self, state, frame_idx, batch_size):  # noqa: ANN001
        _ = state["images"][frame_idx]
        self.windows.append(state["num_frames"])
        return None

    def add_new_points_or_box(  # noqa: ANN001
        self, inference_state=None, frame_idx=0, obj_id=1, points=None, labels=None, box=None, **kw
    ):
        kinds = []
        if box is not None:
            kinds.append("box")
            box = np.asarray(box, dtype=np.float32).reshape(-1)
            self.boxes.append([float(v) for v in box.tolist()])
            self._cx = float((box[0] + box[2]) / 2.0)
            self._cy = float((box[1] + box[3]) / 2.0)
        if points is not None and len(points):
            kinds.append("points")
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            labs = (
                np.ones(len(pts))
                if labels is None
                else np.asarray(labels, dtype=np.float32).reshape(-1)
            )
            pos = pts[labs > 0] if len(labs) == len(pts) else pts
            if len(pos) == 0:
                pos = pts
            self._cx, self._cy = float(pos[0, 0]), float(pos[0, 1])
        self.prompt_kinds.append(kinds)
        idx = int(frame_idx)
        if idx > 0:
            self.jump_local = {j for j in self.jump_local if j < idx}
        return frame_idx, [obj_id], [self._disk(self._cx, self._cy)]

    def propagate_in_video(  # noqa: ANN001
        self, inference_state, start_frame_idx=0, max_frame_num_to_track=None, **kw
    ):
        n = int(inference_state["num_frames"])
        start = int(start_frame_idx or 0)
        remain = max(0, n - start)
        count = remain if max_frame_num_to_track is None else min(remain, int(max_frame_num_to_track))
        outs = inference_state.setdefault("output_dict", {}).setdefault(
            "non_cond_frame_outputs", {}
        )
        window_no = max(0, len(self.windows) - 1)
        for i in range(start, start + count):
            huge = i in self.jump_local
            if (
                self.jump_first_window_last
                and window_no == 0
                and i == n - 1
                and not self._first_window_jumped
            ):
                huge = True
                self._first_window_jumped = True
            if huge:
                mask = self._huge()
                outs[i] = {"object_score_logits": -3.0, "iou_predictions": 0.1}
            elif self.truth is not None and i < len(self.truth):
                mask = self._disk(*self.truth[i])
                outs[i] = {"object_score_logits": 4.0, "iou_predictions": 0.9}
            else:
                mask = self._disk(self._cx + (i - start), self._cy)
                outs[i] = {"object_score_logits": 4.0, "iou_predictions": 0.9}
            yield i, [1], [mask]

    def _disk(self, cx: float, cy: float):
        h, w = self._info.height, self._info.width
        yy, xx = np.ogrid[:h, :w]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= 36
        return np.where(inside, 8.0, -8.0).astype(np.float32)[None, ...]

    def _huge(self):
        h, w = self._info.height, self._info.width
        logits = np.full((h, w), -8.0, dtype=np.float32)
        logits[2 : h - 2, 2 : w - 2] = 8.0
        return logits[None, ...]


class WindowedTrackTests(unittest.TestCase):
    def _info(self):
        _ensure_fixtures()
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        return load_video(resolve_video(entry))

    def test_windowed_path_covers_every_frame_and_carries_a_box(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        import ai.sam2_tracker as tracker_mod

        info = self._info()
        last = min(11, info.frame_count - 1)
        fake = _FakeSam2Like(info)
        original = tracker_mod.TRACK_WINDOW_SAMPLED
        tracker_mod.TRACK_WINDOW_SAMPLED = 4
        try:
            result = tracker_mod.Sam2Tracker(predictor=fake).track(
                info,
                (20.0, 20.0),
                start_frame=0,
                end_frame=last,
                prompts=[
                    TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=20.0, y=20.0)
                ],
            )
        finally:
            tracker_mod.TRACK_WINDOW_SAMPLED = original

        self.assertEqual([p.frame for p in result.points], list(range(last + 1)))
        # More than one window ran, and none held the whole clip.
        self.assertGreater(len(fake.windows), 1)
        self.assertTrue(all(n <= 4 for n in fake.windows))
        # First window is seeded by the user point, later ones by a carried box.
        self.assertIn("points", fake.prompt_kinds[0])
        self.assertTrue(any("box" in kinds for kinds in fake.prompt_kinds[1:]))

    def test_windowed_path_honours_cancel(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        import ai.sam2_tracker as tracker_mod

        info = self._info()
        fake = _FakeSam2Like(info)
        token = CancelToken()

        def progress(event) -> None:  # noqa: ANN001
            if event.stage == "track" and event.current >= 2:
                token.cancel()

        result = tracker_mod.Sam2Tracker(predictor=fake).track(
            info, (20.0, 20.0), start_frame=0, end_frame=8, cancel=token, progress=progress
        )
        self.assertEqual(result.failure_reason, FailureReason.CANCELLED)


class TrackGuardIntegrationTests(unittest.TestCase):
    def _info(self):
        _ensure_fixtures()
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        return load_video(resolve_video(entry))

    def test_window_carry_uses_last_good_mask_not_jumped_frame(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        import ai.sam2_tracker as tracker_mod

        info = self._info()
        last = min(11, info.frame_count - 1)
        fake = _FakeSam2Like(info, jump_first_window_last=True)
        original = tracker_mod.TRACK_WINDOW_SAMPLED
        tracker_mod.TRACK_WINDOW_SAMPLED = 4
        try:
            tracker = tracker_mod.Sam2Tracker(predictor=fake)
            result = tracker.track(
                info,
                (20.0, 20.0),
                start_frame=0,
                end_frame=last,
                prompts=[
                    TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=20.0, y=20.0)
                ],
            )
        finally:
            tracker_mod.TRACK_WINDOW_SAMPLED = original
        jumped = [p for p in result.points if p.frame == 3]
        self.assertTrue(jumped)
        self.assertFalse(jumped[0].visible)
        self.assertIn("跳到背景", jumped[0].note)
        self.assertIn(3, tracker.dropped_frames)
        self.assertGreaterEqual(len(fake.boxes), 1)
        box = fake.boxes[0]
        self.assertLess(box[2] - box[0], 40.0)
        self.assertLess(box[3] - box[1], 40.0)
        self.assertTrue(any("box" in kinds for kinds in fake.prompt_kinds[1:]))

    def test_jumped_mask_is_rejected_and_reprompt_recovers(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from tests.ai.generate_fixtures import _write_video

        n = 16
        width, height = 160, 120
        truth: list[tuple[float, float]] = []
        frames: list[np.ndarray] = []
        rng = np.random.default_rng(0)
        for i in range(n):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :] = (28, 36, 24)
            img[:, 0::10] = (190, 210, 70)
            img[:, 1::10] = (50, 90, 40)
            img = np.clip(
                img.astype(np.int16) + rng.integers(0, 28, img.shape), 0, 255
            ).astype(np.uint8)
            cx = 28.0 + i * 5.5
            cy = 85.0 - 0.18 * i * i
            yy, xx = np.ogrid[:height, :width]
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 36] = (240, 70, 50)
            frames.append(img)
            truth.append((cx, cy))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stripes.mp4"
            _write_video(path, frames, fps=30)
            info = load_video(path)
            fake = _FakeSam2Like(info, jump_local={6, 7, 8}, truth=truth)
            tracker = Sam2Tracker(predictor=fake)
            result = tracker.track(
                info,
                truth[0],
                start_frame=0,
                end_frame=n - 1,
                prompts=[
                    TrackPrompt(
                        frame=0, kind=PromptKind.POSITIVE, x=truth[0][0], y=truth[0][1]
                    )
                ],
            )
            by_frame = {p.frame: p for p in result.points}
            for frame in (6, 7):
                self.assertFalse(by_frame[frame].visible, msg=str(frame))
                self.assertIn("跳到背景", by_frame[frame].note)
            self.assertTrue({6, 7}.issubset(set(tracker.dropped_frames)))
            recovered = by_frame[n - 1]
            self.assertTrue(recovered.visible)
            self.assertLess(abs(recovered.x - truth[-1][0]), 3.0)
            self.assertLess(abs(recovered.y - truth[-1][1]), 3.0)
            raw = Sam2Tracker(
                predictor=_FakeSam2Like(info, jump_local={6, 7, 8}, truth=truth)
            ).track(
                info,
                truth[0],
                start_frame=0,
                end_frame=n - 1,
                anti_interference=False,
                prompts=[
                    TrackPrompt(
                        frame=0, kind=PromptKind.POSITIVE, x=truth[0][0], y=truth[0][1]
                    )
                ],
            )
            jumped = next(p for p in raw.points if p.frame == 6)
            self.assertTrue(jumped.visible)
            self.assertGreater(jumped.x, 40.0)


class FastModeResolutionTests(unittest.TestCase):
    def test_fast_mode_runs_at_native_resolution(self) -> None:
        from ai.sam_runtime import settings_for_mode

        _spec, stride, image_size = settings_for_mode(TrackMode.FAST)
        self.assertEqual(stride, 3)
        self.assertIsNone(image_size)
        _spec, stride, image_size = settings_for_mode(TrackMode.PRECISE)
        self.assertEqual(stride, 1)
        self.assertIsNone(image_size)

    def test_tracker_never_upsamples_decoded_frames(self) -> None:
        import inspect

        import ai.sam2_tracker as tracker

        source = inspect.getsource(tracker)
        self.assertNotIn("F.interpolate", source)
        self.assertNotIn("FAST_TRACK_IMAGE_SIZE", source)


if __name__ == "__main__":
    unittest.main()
