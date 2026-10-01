"""R3/R4: component selection, LK evidence, recovery state machine, memory.

These tests use scripted masks and in-memory frames. They check rules, not
model accuracy; real-model numbers come from tests/ai/benchmark_tracking.py.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ai.contracts import PromptKind, TrackPointStatus, TrackPrompt
from ai.local_flow import lk_evidence
from ai.track_guard import (
    AppearanceEvidence,
    ConstantVelocityKalman,
    LKEvidence,
    PredictedBox,
    SamScores,
    drop_memory_frame,
    label_components,
    mask_components,
    score_mask,
    select_component,
)
from engine.video_index import VideoInfo

H, W = 120, 200
FPS = 30.0


def _info(n: int) -> VideoInfo:
    step = 3000
    pts = tuple(i * step for i in range(n))
    return VideoInfo(
        path=Path("/nonexistent/synthetic.mp4"),
        width=W,
        height=H,
        pts=pts,
        time_base=1.0 / 90000.0,
        pts_ms=tuple(int(round(p / 90.0)) for p in pts),
    )


def _texture(size: int, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 255, (size + 4, size + 4)).astype(np.float32)
    # Mild smoothing: a real object's texture, not per-pixel noise.
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    base = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, base)
    base = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, base)
    base = (base - base.mean()) * 2.5 + 128
    return np.clip(base[2:-2, 2:-2], 0, 255)


def _scene(cx: float, cy: float, tex: np.ndarray, *, stripes: bool = True) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((H, W), 90.0, dtype=np.float32)
    if stripes:
        img[:, (np.arange(W) // 8) % 2 == 0] = 150.0
    size = tex.shape[0]
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    img[y0 : y0 + size, x0 : x0 + size] = tex
    mask = np.zeros((H, W), dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    rgb = np.repeat(img[..., None], 3, axis=2).clip(0, 255).astype(np.uint8)
    return rgb, mask


class _Clip:
    """Frames + masks for an object moving at constant speed."""

    def __init__(self, n: int = 14, size: int = 14, speed: tuple[float, float] = (4.0, 1.0)) -> None:
        self.tex = _texture(size)
        self.centres = [(40.0 + speed[0] * i, 50.0 + speed[1] * i) for i in range(n)]
        self.frames = []
        self.masks = []
        for cx, cy in self.centres:
            rgb, mask = _scene(cx, cy, self.tex)
            self.frames.append(rgb)
            self.masks.append(mask)
        self.info = _info(n)


def _guard(clip: _Clip):
    from ai.sam2_tracker import _GuardSession

    guard = _GuardSession(clip.info, True, clip.centres[0])
    guard._rgb = lambda frame: clip.frames[frame]  # type: ignore[method-assign]
    return guard


GOOD = SamScores(object_score=4.0, iou=0.9)
EMPTY = np.zeros((H, W), dtype=bool)


class ComponentTests(unittest.TestCase):
    def test_labels_match_a_flood_fill(self) -> None:
        rng = np.random.default_rng(2)
        mask = rng.random((60, 70)) > 0.62
        labels, count = label_components(mask)
        # Brute-force 8-connected flood fill.
        seen = np.zeros_like(mask)
        expected = 0
        for y, x in zip(*np.nonzero(mask)):
            if seen[y, x]:
                continue
            expected += 1
            stack = [(y, x)]
            seen[y, x] = True
            members = []
            while stack:
                cy, cx = stack.pop()
                members.append(labels[cy, cx])
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < 60 and 0 <= nx < 70 and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            self.assertEqual(len(set(members)), 1)
        self.assertEqual(count, expected)

    def test_target_is_found_beyond_the_first_scan_order_components(self) -> None:
        mask = np.zeros((H, W), dtype=bool)
        for i in range(40):  # distractor specks above the target
            mask[2 + (i // 20) * 4, 3 + (i % 20) * 9] = True
        mask[80:90, 150:160] = True
        pred = PredictedBox(x=155.0, y=85.0, w=10.0, h=10.0)
        chosen = select_component(mask, pred, 100.0)
        assert chosen is not None
        self.assertAlmostEqual(chosen.x, 154.5)
        self.assertAlmostEqual(chosen.y, 84.5)

    def test_outline_is_an_ordered_closed_path(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:30, 5:12] = True
        mask[22:30, 5:30] = True  # L shape (concave)
        comp = select_component(mask, None, None)
        assert comp is not None
        pts = comp.contour
        self.assertGreater(len(pts), 8)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.assertLessEqual(max(abs(x1 - x0), abs(y1 - y0)), 2.0)

    def test_fragments_of_one_object_are_combined(self) -> None:
        mask = np.zeros((H, W), dtype=bool)
        mask[40:50, 40:44] = True
        mask[40:50, 46:50] = True  # a thin bar splits the ball
        pred = PredictedBox(x=45.0, y=45.0, w=10.0, h=10.0)
        chosen = select_component(mask, pred, 100.0)
        assert chosen is not None
        self.assertAlmostEqual(chosen.x, 44.5)
        self.assertEqual(chosen.area, 80.0)

    def test_prediction_never_moves_the_centre_of_a_chosen_region(self) -> None:
        mask = np.zeros((H, W), dtype=bool)
        mask[30:80, 30:80] = True
        a = select_component(mask, PredictedBox(x=31.0, y=55.0, w=8.0, h=8.0), 64.0)
        b = select_component(mask, PredictedBox(x=78.0, y=55.0, w=8.0, h=8.0), 64.0)
        assert a is not None and b is not None
        self.assertEqual((a.x, a.y), (b.x, b.y))


class LocalFlowTests(unittest.TestCase):
    def test_translation_is_confirmed_inside_the_candidate(self) -> None:
        tex = _texture(20)
        f0, m0 = _scene(60, 60, tex)
        f1, m1 = _scene(67, 62, tex)
        prev = mask_components(m0)[0]
        cand = mask_components(m1)[0]
        ev = lk_evidence(f0, f1, prev, cand, init_disp=(7.0, 2.0))
        self.assertTrue(ev.available, ev.missing_reason)
        self.assertGreaterEqual(ev.inside_fraction or 0.0, 0.9)
        self.assertLess(ev.target_distance_px or 99.0, 1.0)

    def test_wrong_candidate_gets_no_point_support(self) -> None:
        tex = _texture(20)
        f0, m0 = _scene(60, 60, tex)
        f1, _ = _scene(67, 62, tex)
        _, wrong_mask = _scene(120, 60, tex)
        prev = mask_components(m0)[0]
        wrong = mask_components(wrong_mask)[0]
        ev = lk_evidence(f0, f1, prev, wrong, init_disp=(7.0, 2.0))
        self.assertTrue(ev.available)
        self.assertEqual(ev.inside_fraction, 0.0)
        self.assertGreater(ev.target_distance_px or 0.0, 40.0)

    def test_flat_object_is_missing_evidence_not_failure(self) -> None:
        flat = np.full((14, 14), 200.0, dtype=np.float32)
        f0, m0 = _scene(60, 60, flat, stripes=False)
        f1, m1 = _scene(64, 60, flat, stripes=False)
        ev = lk_evidence(f0, f1, mask_components(m0)[0], mask_components(m1)[0])
        self.assertFalse(ev.available)
        self.assertIsNone(ev.quality)
        stats = mask_components(m1)[0]
        decision = score_mask(
            stats,
            PredictedBox(x=64.0, y=60.0, w=14.0, h=14.0, step=4.0, sigma=4.0),
            None,
            GOOD,
            196.0,
            AppearanceEvidence(similarity=0.9, lk=ev),
            updates=6,
        )
        self.assertTrue(decision.accept)

    def test_strong_local_conflict_is_review_even_on_the_prediction(self) -> None:
        mask = np.zeros((H, W), dtype=bool)
        mask[55:65, 55:65] = True
        stats = mask_components(mask)[0]
        lk = LKEvidence(
            selected=20,
            valid=18,
            fb_threshold_px=1.5,
            fb_median_px=0.2,
            inside_fraction=0.0,
            target_xy=(100.0, 60.0),
            target_distance_px=40.5,
            spread_px=0.3,
            quality=0.0,
        )
        decision = score_mask(
            stats,
            PredictedBox(x=60.0, y=60.0, w=10.0, h=10.0, step=4.0, sigma=4.0),
            None,
            GOOD,
            100.0,
            AppearanceEvidence(similarity=0.8, lk=lk),
            updates=6,
        )
        self.assertFalse(decision.accept)
        self.assertEqual(decision.diagnostics["reason_code"], "local_conflict")


class KalmanReplayTests(unittest.TestCase):
    def test_same_time_does_not_advance_twice(self) -> None:
        k = ConstantVelocityKalman()
        k.update(0.0, 10, 10, 4, 4)
        k.update(1 / 30, 14, 10, 4, 4)
        k.update(2 / 30, 18, 10, 4, 4)
        before = (k.updates, k._x.copy())  # noqa: SLF001
        k.update(2 / 30, 30, 30, 4, 4)
        self.assertEqual(k.updates, before[0])
        np.testing.assert_allclose(k._x, before[1])  # noqa: SLF001

    def test_small_target_gate_is_not_pinned_to_48px(self) -> None:
        k = ConstantVelocityKalman()
        for i in range(8):
            k.update(i / 30, 10 + 2 * i, 10, 4, 4)
        pred = k.predict(8 / 30)
        assert pred is not None
        self.assertLess(pred.sigma, 20.0)


class RecoveryStateMachineTests(unittest.TestCase):
    def test_normal_tracking_has_no_confirmation_delay(self) -> None:
        clip = _Clip()
        guard = _guard(clip)
        for i in range(len(clip.frames)):
            decision, *_ = guard.observe(i, clip.masks[i], GOOD)
            self.assertTrue(decision.accept, f"frame {i}: {decision.reason} {decision.diagnostics}")

    def test_recovery_takes_two_frames_and_backfills_the_first(self) -> None:
        clip = _Clip()
        guard = _guard(clip)
        for i in range(5):
            self.assertTrue(guard.observe(i, clip.masks[i], GOOD)[0].accept)
        for i in range(5, 8):
            decision, stats, *_ = guard.observe(i, EMPTY, SamScores(object_score=-9.0))
            self.assertFalse(decision.accept)
        updates = guard.kalman.updates
        first, stats8, *_ = guard.observe(8, clip.masks[8], GOOD)
        self.assertFalse(first.accept)
        self.assertEqual(first.diagnostics.get("reason_code"), "recovery_pending")
        self.assertEqual(guard.kalman.updates, updates, "pending must not touch the filter")
        second, *_ = guard.observe(9, clip.masks[9], GOOD)
        self.assertTrue(second.accept)
        assert guard.backfill is not None
        self.assertEqual(guard.backfill.frame, 8)
        self.assertAlmostEqual(guard.backfill.stats.x, stats8.x)
        self.assertEqual(guard.kalman.updates, updates + 2)
        # Back to normal: the next frame is trusted at once.
        self.assertTrue(guard.observe(10, clip.masks[10], GOOD)[0].accept)

    def test_inconsistent_second_frame_discards_the_candidate(self) -> None:
        clip = _Clip()
        guard = _guard(clip)
        for i in range(5):
            guard.observe(i, clip.masks[i], GOOD)
        for i in range(5, 8):
            guard.observe(i, EMPTY, SamScores(object_score=-9.0))
        guard.observe(8, clip.masks[8], GOOD)
        updates = guard.kalman.updates
        guard.observe(9, EMPTY, SamScores(object_score=-9.0))
        assert guard.discarded is not None
        self.assertEqual(guard.discarded.frame, 8)
        self.assertIsNone(guard.pending)
        self.assertEqual(guard.kalman.updates, updates)

    def test_restore_replays_a_frame_without_double_update(self) -> None:
        clip = _Clip()
        guard = _guard(clip)
        for i in range(6):
            guard.observe(i, clip.masks[i], GOOD)
        state = (guard.kalman.updates, guard.kalman._x.copy())  # noqa: SLF001
        self.assertTrue(guard.restore(5))
        guard.observe(5, clip.masks[5], GOOD)
        self.assertEqual(guard.kalman.updates, state[0])
        np.testing.assert_allclose(guard.kalman._x, state[1])  # noqa: SLF001

    def test_foreground_motion_candidate_is_never_committed(self) -> None:
        clip = _Clip()
        guard = _guard(clip)
        for i in range(5):
            guard.observe(i, clip.masks[i], GOOD)
        for i in range(5, 12):
            decision, *_ = guard.observe(i, EMPTY, SamScores(object_score=-9.0))
            self.assertFalse(decision.accept)
        self.assertIsNone(guard.pending)


class _ScriptedPredictor:
    """Single-pass predictor (not SAM-like) returning scripted masks."""

    def __init__(self, masks: list[np.ndarray | None]) -> None:
        self.masks = masks
        self.prompts: list[tuple[int, str]] = []
        self.state: dict = {}

    def init_state(self, video_path: str, **_kw):
        self.state = {
            "obj_id_to_idx": {1: 0},
            "obj_ids": [1],
            "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
            "point_inputs_per_obj": {0: {}},
            "mask_inputs_per_obj": {0: {}},
            "frames_tracked_per_obj": {0: {}},
        }
        return self.state

    def add_new_points_or_box(self, inference_state, frame_idx=0, obj_id=1, points=None, labels=None, box=None, **_kw):
        kind = "box" if box is not None else "points"
        self.prompts.append((int(frame_idx), kind))
        store = inference_state["output_dict_per_obj"][0]["cond_frame_outputs"]
        store[int(frame_idx)] = {"object_score_logits": 4.0, "iou_predictions": 0.9}
        inference_state["point_inputs_per_obj"][0][int(frame_idx)] = {"n": 1}
        mask = self.masks[int(frame_idx)]
        return frame_idx, [obj_id], [self._logits(mask)]

    def propagate_in_video(self, inference_state, start_frame_idx=0, max_frame_num_to_track=None, **_kw):
        end = len(self.masks) if max_frame_num_to_track is None else start_frame_idx + max_frame_num_to_track
        for i in range(start_frame_idx, min(end, len(self.masks))):
            mask = self.masks[i]
            outs = inference_state["output_dict_per_obj"][0]
            if i not in outs["cond_frame_outputs"]:
                good = mask is not None and mask.any()
                outs["non_cond_frame_outputs"][i] = {
                    "object_score_logits": 4.0 if good else -9.0,
                    "iou_predictions": 0.9 if good else 0.05,
                }
            inference_state["frames_tracked_per_obj"][0][i] = {"reverse": False}
            yield i, [1], [self._logits(mask)]

    @staticmethod
    def _logits(mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            mask = EMPTY
        return np.where(mask, 8.0, -8.0).astype(np.float32)[None, ...]


class _FakeDecoder:
    frames: list[np.ndarray] = []

    def __init__(self, info) -> None:  # noqa: ANN001
        pass

    def frame(self, index: int) -> np.ndarray:
        return self.frames[index]

    def close(self) -> None:
        pass


class TrackerRecoveryTests(unittest.TestCase):
    def _run(self, clip: _Clip, masks: list[np.ndarray | None], fill_gaps: bool = False):
        from ai import sam2_tracker as mod

        _FakeDecoder.frames = clip.frames
        predictor = _ScriptedPredictor(masks)
        with mock.patch.object(mod, "FrameDecoder", _FakeDecoder):
            result = mod.Sam2Tracker(predictor=predictor).track(
                clip.info,
                clip.centres[0],
                start_frame=0,
                end_frame=len(masks) - 1,
                prompts=[TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=clip.centres[0][0], y=clip.centres[0][1])],
                fill_gaps=fill_gaps,
            )
        return result, predictor

    def test_lost_frames_are_gaps_and_recovery_backfills_a_real_measurement(self) -> None:
        clip = _Clip(n=14)
        masks: list[np.ndarray | None] = list(clip.masks)
        for i in (5, 6):
            masks[i] = None
        result, _ = self._run(clip, masks)
        by = {p.frame: p for p in result.points}
        for i in (5, 6):
            # SAM gave nothing: a gap, or a motion cue shown for review only.
            self.assertIn(by[i].status, {TrackPointStatus.LOST, TrackPointStatus.REVIEW})
            self.assertFalse(by[i].usable_for_measurement())
        backfilled = by[7]
        self.assertEqual(backfilled.status, TrackPointStatus.TRUSTED, backfilled.diagnostics)
        self.assertTrue(backfilled.usable_for_measurement())
        self.assertEqual(backfilled.diagnostics.get("recovery"), "backfilled")
        self.assertAlmostEqual(backfilled.x, clip.centres[7][0] - 0.5, delta=0.6)
        self.assertTrue(all(by[i].usable_for_measurement() for i in range(8, 14)))
        self.assertTrue(all(not p.interpolated for p in result.points))

    def test_short_gap_is_bridged_by_the_fit_but_not_measured(self) -> None:
        clip = _Clip(n=16)
        masks: list[np.ndarray | None] = list(clip.masks)
        for i in (7, 8):
            masks[i] = None
        result, _ = self._run(clip, masks, fill_gaps=True)
        by = {p.frame: p for p in result.points}
        for i in (7, 8):
            p = by[i]
            if p.usable_for_measurement():
                continue  # SAM re-segmented it after an automatic prompt
            self.assertTrue(p.visible and p.interpolated, p)
            self.assertFalse(p.usable_for_measurement())
            self.assertTrue(p.diagnostics.get("fit_filled"))
            self.assertLess(abs(p.x - (clip.centres[i][0] - 0.5)), 2.0)

    def test_rejected_automatic_prompt_leaves_no_conditioning_memory(self) -> None:
        state = {
            "obj_id_to_idx": {1: 0},
            "obj_ids": [1],
            "output_dict_per_obj": {0: {"cond_frame_outputs": {0: "user", 7: "auto"}, "non_cond_frame_outputs": {6: "x"}}},
            "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
            "point_inputs_per_obj": {0: {0: "user", 7: "auto"}},
            "mask_inputs_per_obj": {0: {}},
            "frames_tracked_per_obj": {0: {6: {}, 7: {}}},
        }
        drop_memory_frame(state, 7, 1, conditioning=True)
        self.assertEqual(state["output_dict_per_obj"][0]["cond_frame_outputs"], {0: "user"})
        self.assertEqual(state["point_inputs_per_obj"][0], {0: "user"})
        self.assertNotIn(7, state["frames_tracked_per_obj"][0])
        # The last conditioning frame is never removed (SAM cannot run without it).
        drop_memory_frame(state, 0, 1, conditioning=True)
        self.assertEqual(state["output_dict_per_obj"][0]["cond_frame_outputs"], {0: "user"})

    def test_rejection_without_conditioning_keeps_user_prompts(self) -> None:
        state = {
            "obj_id_to_idx": {1: 0},
            "output_dict_per_obj": {0: {"cond_frame_outputs": {0: "u", 3: "u2"}, "non_cond_frame_outputs": {3: "x"}}},
            "frames_tracked_per_obj": {0: {3: {}}},
        }
        drop_memory_frame(state, 3, 1)
        self.assertIn(3, state["output_dict_per_obj"][0]["cond_frame_outputs"])


class _LazyStandIn:
    def __init__(self, info, indices, image_size, **_kw) -> None:  # noqa: ANN001
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def close(self) -> None:
        pass


class _WindowPredictor(_ScriptedPredictor):
    """Looks like SAM2VideoPredictor (windowed path); frames are window-local."""

    image_size = 64
    device = "cpu"

    def __init__(self, masks: list[np.ndarray | None]) -> None:
        super().__init__(masks)
        self.windows: list[list[int]] = []
        self.mask_prompts: list[int] = []
        self._frames: list[int] = []

    def _get_image_feature(self, *a, **k):  # noqa: ANN002, ANN003
        return None

    def bind(self, state: dict, frames: list[int]) -> None:
        self._frames = frames
        base = self.init_state("")
        state.update(base)
        self.windows.append(list(frames))

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):  # noqa: ANN001
        self.mask_prompts.append(self._frames[frame_idx])
        inference_state["output_dict_per_obj"][0]["cond_frame_outputs"][frame_idx] = {"object_score_logits": 4.0}
        return frame_idx, [obj_id], [self._logits(mask)]

    def add_new_points_or_box(self, inference_state, frame_idx=0, obj_id=1, points=None, labels=None, box=None, **_kw):
        abs_frame = self._frames[int(frame_idx)]
        self.prompts.append((abs_frame, "box" if box is not None else "points"))
        inference_state["output_dict_per_obj"][0]["cond_frame_outputs"][int(frame_idx)] = {"object_score_logits": 4.0}
        return frame_idx, [obj_id], [self._logits(self.masks[abs_frame])]

    def propagate_in_video(self, inference_state, start_frame_idx=0, max_frame_num_to_track=None, **_kw):
        n = len(self._frames)
        end = n if max_frame_num_to_track is None else min(n, start_frame_idx + max_frame_num_to_track)
        for i in range(start_frame_idx, end):
            mask = self.masks[self._frames[i]]
            good = mask is not None and mask.any()
            inference_state["output_dict_per_obj"][0]["non_cond_frame_outputs"][i] = {
                "object_score_logits": 4.0 if good else -9.0,
                "iou_predictions": 0.9 if good else 0.05,
            }
            yield i, [1], [self._logits(mask)]


class WindowBoundaryTests(unittest.TestCase):
    def test_loss_at_a_window_boundary_does_not_end_the_track(self) -> None:
        from ai import sam2_tracker as mod

        clip = _Clip(n=14)
        masks: list[np.ndarray | None] = list(clip.masks)
        masks[3] = None  # the frame shared by windows 0 and 1
        predictor = _WindowPredictor(masks)

        def fake_state(pred, images, h, w, **_kw):  # noqa: ANN001
            state: dict = {"num_frames": len(images)}
            pred.bind(state, images.indices)
            return state

        _FakeDecoder.frames = clip.frames
        with mock.patch.object(mod, "FrameDecoder", _FakeDecoder), mock.patch.object(
            mod, "LazyFrameTensors", _LazyStandIn
        ), mock.patch.object(mod, "build_video_state", fake_state), mock.patch.object(
            mod, "TRACK_WINDOW_SAMPLED", 4
        ):
            result = mod.Sam2Tracker(predictor=predictor).track(
                clip.info,
                clip.centres[0],
                start_frame=0,
                end_frame=13,
                prompts=[TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=clip.centres[0][0], y=clip.centres[0][1])],
            )
        by = {p.frame: p for p in result.points}
        self.assertEqual(sorted(by), list(range(14)))
        # Window 1 starts from the real frame 2 (last committed), prepended,
        # seeded with its mask, instead of a box relabelled as frame 3.
        self.assertIn([2, 3, 4, 5, 6], predictor.windows)
        self.assertIn(2, predictor.mask_prompts)
        self.assertNotIn((3, "box"), predictor.prompts)
        self.assertTrue(by[13].usable_for_measurement(), by[13].diagnostics)
        self.assertGreaterEqual(sum(p.usable_for_measurement() for p in result.points), 10)


if __name__ == "__main__":
    unittest.main()
