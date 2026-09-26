"""R2: surface-point query identity, transforms, bounded windows, local route.

A dependency-free NCC matcher stands in for TAPIR. It checks the adapter's
plumbing (which frames, which query, which crop, how results are fused),
not BootsTAPIR accuracy.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ai.contracts import PromptKind, TrackPointStatus, TrackPrompt
from ai.tapir_tracker import (
    LOCAL_SIZE,
    MODEL_SIZE,
    BootsTapirTracker,
    CropTransform,
    RunOutput,
    crop_resize,
    window_starts,
)
from engine.video_index import VideoInfo
from tests.ai.benchmark_tracking import NccPointRunner


def _info(n: int, width: int, height: int) -> VideoInfo:
    pts = tuple(i * 3000 for i in range(n))
    return VideoInfo(
        path=Path("/nonexistent/clip.mp4"),
        width=width,
        height=height,
        pts=pts,
        time_base=1 / 90000,
        pts_ms=tuple(int(round(p / 90)) for p in pts),
    )


def _clip(n=24, width=640, height=360, size=24, speed=(3.0, 1.0), start=(200, 150), hide=None, seed=3):
    rng = np.random.default_rng(seed)
    block = 2 if size <= 8 else 4  # blocky texture, finer for small targets
    tex = rng.integers(0, 256, (size // block, size // block, 3), dtype=np.uint8)
    tex = np.kron(tex, np.ones((block, block, 1), dtype=np.uint8))
    frames, surface, centres = [], [], []
    for i in range(n):
        img = np.full((height, width, 3), 80, dtype=np.uint8)
        x, y = int(round(start[0] + speed[0] * i)), int(round(start[1] + speed[1] * i))
        hidden = hide is not None and hide[0] <= i < hide[1]
        if not hidden:
            img[y : y + size, x : x + size] = tex
        frames.append(img)
        centres.append((x + (size - 1) / 2, y + (size - 1) / 2))
        surface.append(None if hidden else (x + 5.0, y + 7.0))  # off-centre point
    return frames, surface, centres, _info(n, width, height)


class _FakeDecoder:
    frames: list[np.ndarray] = []

    def __init__(self, info) -> None:  # noqa: ANN001
        pass

    def frame(self, index: int) -> np.ndarray:
        return self.frames[index]

    def close(self) -> None:
        pass


class _Recording(NccPointRunner):
    def __init__(self) -> None:
        super().__init__(radius=4, search=48)
        self.feature_calls: list[np.ndarray] = []
        self.track_calls: list[int] = []

    def query_features(self, frame, uv):
        self.feature_calls.append(np.asarray(frame).copy())
        return super().query_features(frame, uv)

    def track(self, frames, features, *, query_t, query_uv):
        self.track_calls.append(len(frames))
        return super().track(frames, features, query_t=query_t, query_uv=query_uv)


def _run(frames, info, runner, *, query, start=0, end=None, local=True, **kw):
    _FakeDecoder.frames = frames
    import ai.tapir_tracker as mod

    with mock.patch.object(mod, "FrameDecoder", _FakeDecoder):
        tracker = BootsTapirTracker(model=runner, device="cpu", local_refine=local, **kw)
        return tracker.track(
            info,
            query[1],
            start_frame=start,
            end_frame=end,
            prompts=[TrackPrompt(frame=query[0], kind=PromptKind.POSITIVE, x=query[1][0], y=query[1][1])],
        ), tracker


class TransformTests(unittest.TestCase):
    def test_round_trip_is_exact(self) -> None:
        cases = [
            CropTransform.full(1280, 720, MODEL_SIZE),
            CropTransform.full(720, 1280, MODEL_SIZE),
            CropTransform.full(1920, 1080, MODEL_SIZE),
            CropTransform.square(10.0, 5.0, 256, LOCAL_SIZE),  # crosses the image edge
            CropTransform.square(900.3, 500.7, 1024, LOCAL_SIZE),
        ]
        rng = np.random.default_rng(0)
        for t in cases:
            for x, y in rng.uniform(0, 700, (20, 2)):
                u, v = t.to_model(x, y)
                bx, by = t.to_original(u, v)
                self.assertLess(abs(bx - x) + abs(by - y), 0.1)

    def test_crop_places_a_pixel_where_the_transform_says(self) -> None:
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[123, 211] = 255
        t = CropTransform.square(200.0, 130.0, 256, LOCAL_SIZE)  # 2x upscale
        crop = crop_resize(img, t)[..., 0].astype(float)
        vy, vx = np.unravel_index(int(np.argmax(crop)), crop.shape)
        u, v = t.to_model(211, 123)  # continuous centre of that pixel
        self.assertLess(abs((vx + 0.5) - u), 1.01)
        self.assertLess(abs((vy + 0.5) - v), 1.01)

    def test_full_frame_resize_matches_torch(self) -> None:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch not installed")
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
        ours = crop_resize(img, CropTransform.full(320, 180, MODEL_SIZE)).astype(int)
        t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
        ref = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
        ref = ref[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy().astype(int)
        self.assertLessEqual(int(np.abs(ours - ref).max()), 1)

    def test_windows_are_bounded_and_overlap(self) -> None:
        wins = window_starts(80, 32, 8)
        self.assertTrue(all(e - s <= 32 for s, e in wins))
        self.assertEqual(wins[0], (0, 32))
        self.assertEqual(wins[1][0], 24)
        self.assertEqual(wins[-1][1], 80)


class QueryIdentityTests(unittest.TestCase):
    def test_query_is_the_seed_prompt_and_other_clicks_are_reported(self) -> None:
        prompts = [
            TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=10.0, y=20.0),
            TrackPrompt(frame=5, kind=PromptKind.POSITIVE, x=40.0, y=50.0),
        ]
        record, others = BootsTapirTracker.resolve_query((40.0, 50.0), prompts, 0)
        self.assertEqual(record.query_frame, 5)
        self.assertEqual(record.query_xy_original, (40.0, 50.0))
        self.assertEqual(others, [{"frame": 0, "xy": [10.0, 20.0]}])

    def test_features_come_from_the_real_query_frame_outside_the_range(self) -> None:
        frames, surface, _c, info = _clip(n=30)
        runner = _Recording()
        result, _ = _run(frames, info, runner, query=(2, surface[2]), start=10, end=20, local=False)
        self.assertEqual([p.frame for p in result.points], list(range(10, 21)))
        self.assertTrue(runner.feature_calls)
        expected = crop_resize(frames[2], CropTransform.full(info.width, info.height, MODEL_SIZE))
        np.testing.assert_array_equal(runner.feature_calls[0], expected)
        self.assertEqual(result.points[0].diagnostics["query_frame"], 2)

    def test_long_clip_runs_in_bounded_windows(self) -> None:
        frames, surface, _c, info = _clip(n=90, speed=(1.5, 0.5))
        runner = _Recording()
        _run(frames, info, runner, query=(0, surface[0]), local=True)
        self.assertTrue(all(n <= 32 for n in runner.track_calls), runner.track_calls)
        self.assertGreater(len(runner.track_calls), 3)


class LocalRouteTests(unittest.TestCase):
    def _errors(self, result, surface):
        out = []
        for p in result.points:
            t = surface[p.frame]
            if t is not None:
                out.append(float(np.hypot(p.x - t[0], p.y - t[1])))
        return out

    def test_off_centre_point_is_not_pulled_to_the_object_centre(self) -> None:
        frames, surface, centres, info = _clip(n=20)
        result, _ = _run(frames, info, _Recording(), query=(0, surface[0]))
        errs = self._errors(result, surface)
        self.assertLess(max(errs), 1.5, errs)
        # The object centre is ~8 px away; staying on the surface point proves
        # the adapter did not substitute a centroid.
        for p in result.points:
            self.assertGreater(float(np.hypot(p.x - centres[p.frame][0], p.y - centres[p.frame][1])), 4.0)

    def test_local_route_keeps_small_texture_the_global_route_loses(self) -> None:
        frames, surface, _c, info = _clip(n=16, width=1600, height=900, size=8, speed=(4.0, 1.0), start=(700, 400))
        glob, _ = _run(frames, info, _Recording(), query=(0, surface[0]), local=False)
        loc, _ = _run(frames, info, _Recording(), query=(0, surface[0]), local=True)
        g = self._errors(glob, surface)
        lo = self._errors(loc, surface)
        self.assertLess(float(np.mean(lo)), float(np.mean(g)))
        self.assertLess(float(np.median(lo)), 1.5, lo)
        self.assertGreaterEqual(sum(p.usable_for_measurement() for p in loc.points), 12)

    def test_fast_motion_rebuilds_the_roi_before_the_edge(self) -> None:
        frames, surface, _c, info = _clip(n=40, width=1600, height=900, speed=(14.0, 3.0), start=(100, 300))
        result, _ = _run(frames, info, _Recording(), query=(0, surface[0]))
        rois = {tuple(p.diagnostics["local_roi"]) for p in result.points if p.diagnostics.get("local_roi")}
        self.assertGreater(len(rois), 1)
        for p in result.points:
            roi = p.diagnostics.get("local_roi")
            if roi and p.diagnostics.get("route") == "local512":
                t = CropTransform(roi[0], roi[1], roi[2], roi[3], int(roi[4]))
                self.assertTrue(t.contains(p.x, p.y, 0.1), (p.frame, roi))

    def test_occlusion_and_uncertainty_are_kept_apart(self) -> None:
        class Heads:
            device = "cpu"
            fallback_reason = ""

            def query_features(self, frame, uv):
                return uv

            def track(self, frames, features, *, query_t, query_uv):
                n = len(frames)
                uv = query_uv or features
                tracks = np.tile(np.array(uv, dtype=float), (n, 1))
                occ = np.full(n, -5.0)
                dist = np.full(n, -5.0)
                if n > 2:
                    occ[1] = 5.0  # occluded
                    dist[2] = 5.0  # visible but imprecise
                return RunOutput(tracks, occ, dist)

        frames, surface, _c, info = _clip(n=4)
        result, _ = _run(frames, info, Heads(), query=(0, surface[0]), local=False, reverse_check=False)
        by = {p.frame: p for p in result.points}
        self.assertEqual(by[0].status, TrackPointStatus.TRUSTED)
        self.assertEqual(by[1].status, TrackPointStatus.LOST)
        self.assertEqual(by[2].status, TrackPointStatus.REVIEW)
        self.assertIn("occlusion_probability", by[2].diagnostics)
        self.assertIn("position_uncertainty", by[2].diagnostics)
        self.assertGreater(by[2].diagnostics["position_uncertainty"], 0.5)
        self.assertLess(by[2].diagnostics["occlusion_probability"], 0.5)

    def test_reappearance_gets_a_reverse_check(self) -> None:
        frames, surface, _c, info = _clip(n=20, hide=(8, 11))
        result, _ = _run(frames, info, _Recording(), query=(0, surface[0]))
        checked = [p for p in result.points if "reverse_check" in p.diagnostics]
        self.assertTrue(checked)
        for p in result.points:
            if surface[p.frame] is None:
                self.assertFalse(p.usable_for_measurement(), p.frame)


if __name__ == "__main__":
    unittest.main()
