"""SAM 2 adapter: prompts → video masks → TrackResult centroids."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Any, Iterable, Protocol

import numpy as np

from ai.contracts import (
    CancelToken,
    FailureReason,
    ProgressCb,
    PromptKind,
    TrackPoint,
    TrackPointSource,
    TrackPointStatus,
    TRACK_QUALITY_VERSION,
    TrackPrompt,
    TrackResult,
)
from ai.models import Tracker, _emit, load_video
from ai.model_manager import DEFAULT_SPEC, ModelNotAvailable, ModelSpec, load_sam2_predictor
from ai.sam2_frames import (
    LazyFrameTensors,
    build_video_state,
    densify_track_points,
    is_sam2_predictor,
    remap_prompts_to_samples,
    sampled_frame_indices,
)
from ai.fit_suggestions import fill_fit_gaps
from ai.track_guard import (
    RECOVERED_CONFIDENCE,
    REJECT_BACKGROUND,
    REJECT_STREAK,
    ConstantVelocityKalman,
    AppearanceEvidence,
    Foreground,
    GateDecision,
    MaskStats,
    PredictedBox,
    SamScores,
    drop_memory_frame,
    extract_sam_scores,
    mask_logit_quality,
    mask_components,
    mask_stats,
    select_component,
    _appearance_patch,
    appearance_similarity,
    lk_evidence,
    LocalTrajectory,
    TrajectoryEvidence,
    color_signature,
    color_similarity,
    region_change,
    trajectory_tolerance,
    motion_foreground,
    recover_from_foreground,
    reprompt_candidate,
    score_mask,
    time_s as guard_time_s,
)
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo


OFFLOAD_VIDEO_BYTES = 512 * 1024 * 1024
TRACK_WINDOW_SAMPLED = 300


def sample_track_windows(
    sampled: list[int], size: int = TRACK_WINDOW_SAMPLED
) -> list[list[int]]:
    """Split sampled frames into windows of `size` with one overlapping frame."""
    if not sampled:
        return []
    width = max(2, int(size))
    if len(sampled) <= width:
        return [list(sampled)]
    windows: list[list[int]] = []
    start = 0
    n = len(sampled)
    while start < n:
        end = min(start + width, n)
        windows.append(list(sampled[start:end]))
        if end >= n:
            break
        start = end - 1
    return windows


def _release_device_cache(device: Any) -> None:
    kind = str(getattr(device, "type", device)).lower()
    if "mps" not in kind:
        return
    try:
        import torch

        torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        return


def should_offload_video(device: Any, n_frames: int, image_size: int) -> bool:
    kind = str(getattr(device, "type", device)).lower()
    # MPS shares memory with the rest of macOS. Keep the video on CPU.
    if "cpu" in kind or "mps" in kind:
        return True
    return n_frames * 3 * image_size * image_size * 4 > OFFLOAD_VIDEO_BYTES


class VideoPredictor(Protocol):
    def init_state(self, video_path: str, **kwargs: Any) -> dict: ...

    def add_new_points_or_box(self, *args: Any, **kwargs: Any) -> tuple: ...

    def propagate_in_video(self, *args: Any, **kwargs: Any) -> Iterable: ...


def as_numpy(value: Any) -> np.ndarray:
    """Copy MPS/CUDA tensors to host before NumPy sees them."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def logits_to_mask(logits: Any) -> np.ndarray:
    arr = as_numpy(logits)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0]
    return arr > 0.0


def mask_centroid(
    mask: np.ndarray,
) -> tuple[float, float, float, bool, list[tuple[float, float]]]:
    """Return (x, y, confidence, visible, contour) from a boolean mask."""
    binary = as_numpy(mask).astype(bool)
    if binary.ndim != 2:
        binary = binary.reshape(binary.shape[-2], binary.shape[-1])
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return 0.0, 0.0, 0.0, False, []
    x = float(xs.mean())
    y = float(ys.mean())
    area = float(xs.size)
    conf = float(min(1.0, area / 64.0))
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    contour = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return x, y, conf, True, contour


def merge_track_points(
    existing: list[TrackPoint], incoming: list[TrackPoint], from_frame: int
) -> list[TrackPoint]:
    """Keep confirmed points before from_frame; replace from that frame onward."""
    kept = [p for p in existing if p.frame < from_frame]
    by_frame = {p.frame: p for p in kept}
    for point in incoming:
        if point.frame >= from_frame:
            by_frame[point.frame] = point
    return [by_frame[key] for key in sorted(by_frame)]


class FakeVideoPredictor:
    """Deterministic stand-in used by unit tests. Does not load weights."""

    def __init__(
        self,
        radius: int = 8,
        drift: tuple[float, float] = (1.0, 0.0),
        blank_from: int | None = None,
    ) -> None:
        self.radius = radius
        self.drift = drift
        self.blank_from = blank_from
        self._cx = 0.0
        self._cy = 0.0
        self._obj: dict[int, tuple[float, float]] = {}

    def init_state(self, video_path: str, **kwargs: Any) -> dict:
        info = load_video(video_path)
        return {"info": info, "path": video_path}

    def add_new_points_or_box(
        self,
        inference_state: dict,
        frame_idx: int = 0,
        obj_id: int = 1,
        points: Any = None,
        labels: Any = None,
        box: Any = None,
        **kwargs: Any,
    ) -> tuple:
        info: VideoInfo = inference_state["info"]
        if box is not None:
            box = np.asarray(box, dtype=np.float32).reshape(-1)
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
        elif points is not None and len(points) > 0:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            labs = np.ones(len(pts)) if labels is None else np.asarray(labels).reshape(-1)
            pos = pts[labs > 0]
            if len(pos) == 0:
                pos = pts
            cx, cy = float(pos[0, 0]), float(pos[0, 1])
        else:
            cx, cy = info.width / 2.0, info.height / 2.0
        self._obj[int(obj_id)] = (cx, cy)
        self._cx, self._cy = cx, cy
        mask = self._disk(info, cx, cy)
        return frame_idx, [obj_id], [mask]

    def propagate_in_video(
        self,
        inference_state: dict,
        start_frame_idx: int = 0,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        **kwargs: Any,
    ):
        info: VideoInfo = inference_state["info"]
        last = info.frame_count - 1
        if max_frame_num_to_track is None:
            end = last
        else:
            end = min(last, start_frame_idx + max_frame_num_to_track - 1)
        frames = range(start_frame_idx, end + 1)
        if reverse:
            frames = range(start_frame_idx, -1, -1)
        cx, cy = self._cx, self._cy
        for i, frame in enumerate(frames):
            x = cx + self.drift[0] * i
            y = cy + self.drift[1] * i
            if self.blank_from is not None and frame >= self.blank_from:
                yield frame, [1], [self._disk(info, x, y, radius=0)]
            else:
                yield frame, [1], [self._disk(info, x, y)]

    def _disk(
        self, info: VideoInfo, cx: float, cy: float, radius: int | None = None
    ) -> np.ndarray:
        rad = self.radius if radius is None else radius
        if rad <= 0:
            logits = np.full((info.height, info.width), -8.0, dtype=np.float32)
            return logits[None, ...]
        yy, xx = np.ogrid[: info.height, : info.width]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
        logits = np.where(inside, 8.0, -8.0).astype(np.float32)
        return logits[None, ...]


RELOCALIZE_EVERY = 8
CHECKPOINTS_KEPT = 8


@dataclass
class _Pending:
    """First candidate after a loss: measured, validated, not yet committed."""

    frame: int
    t: float
    stats: MaskStats
    decision: GateDecision
    rgb: np.ndarray | None
    patch: np.ndarray | None
    score: float | None = None
    change: float | None = None
    color: np.ndarray | None = None


class _GuardSession:
    """Per-track motion, appearance and recovery state. Closed after track().

    Every candidate goes through propose -> validate -> commit/discard. Only a
    committed measurement updates the Kalman filter, the appearance templates
    and the last trusted region; SAM memory for uncommitted frames is dropped
    by the caller.
    """

    def __init__(
        self, info: VideoInfo, enabled: bool, seed_xy: tuple[float, float] | None = None
    ) -> None:
        self.info = info
        self.enabled = bool(enabled)
        self.kalman = ConstantVelocityKalman()
        self.decoder: FrameDecoder | None = None
        self.prev_rgb: np.ndarray | None = None
        self.streak = 0
        self.lost_run = 0
        self.recovering = False
        self.pending: _Pending | None = None
        self.last_bad: tuple[float, float] | None = None
        self.dropped: list[int] = []
        self.reprompted: set[int] = set()
        self.last_relocalize: int | None = None
        self.seed_xy = seed_xy
        self.current_rgb: np.ndarray | None = None
        self.anchor_patch: np.ndarray | None = None
        self.recent_patch: np.ndarray | None = None
        self.trusted_rgb: np.ndarray | None = None
        self.trusted_stats: MaskStats | None = None
        self.trusted_frame: int | None = None
        # Outcome of the latest observe() for the caller to apply to results.
        self.backfill: _Pending | None = None
        self.discarded: _Pending | None = None
        self._checkpoints: OrderedDict[int, dict[str, Any]] = OrderedDict()
        # Evidence added for thin, blurred projectiles (R3 round 2).
        self.projectile = False
        self.prev2_rgb: np.ndarray | None = None
        self.prev_shift: tuple[float, float] = (0.0, 0.0)
        self.shift_two: tuple[float, float] = (0.0, 0.0)
        self.trajectory = LocalTrajectory()
        self.anchor_color: np.ndarray | None = None
        self.recent_color: np.ndarray | None = None
        self.score_history: deque[float] = deque(maxlen=10)
        self.change_history: deque[float] = deque(maxlen=10)
        self.last_traj: tuple[float, float, float, int] | None = None
        self._obs_score: float | None = None
        self._obs_change: float | None = None
        self._obs_color: np.ndarray | None = None

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self.decoder is not None:
            self.decoder.close()
            self.decoder = None

    _STATE_KEYS = (
        "kalman",
        "prev_rgb",
        "streak",
        "lost_run",
        "recovering",
        "pending",
        "last_bad",
        "anchor_patch",
        "recent_patch",
        "trusted_rgb",
        "trusted_stats",
        "trusted_frame",
        "prev2_rgb",
        "prev_shift",
        "shift_two",
        "trajectory",
        "anchor_color",
        "recent_color",
        "score_history",
        "change_history",
    )
    _DEEP_KEYS = ("kalman", "trajectory", "score_history", "change_history")

    def _checkpoint(self, abs_frame: int) -> None:
        state = {key: getattr(self, key) for key in self._STATE_KEYS}
        for key in self._DEEP_KEYS:
            state[key] = copy.deepcopy(state[key])
        self._checkpoints[abs_frame] = state
        self._checkpoints.move_to_end(abs_frame)
        while len(self._checkpoints) > CHECKPOINTS_KEPT:
            self._checkpoints.popitem(last=False)

    def restore(self, abs_frame: int) -> bool:
        """Return to the state just before `abs_frame` was first observed.

        Re-running a frame (after an automatic prompt) must not advance the
        motion model, templates or recovery counters a second time.
        """
        state = self._checkpoints.get(abs_frame)
        if state is None:
            return False
        for key, value in state.items():
            setattr(self, key, copy.deepcopy(value) if key in self._DEEP_KEYS else value)
        for frame in [f for f in self._checkpoints if f > abs_frame]:
            del self._checkpoints[frame]
        return True

    # -- observation -------------------------------------------------------

    def mark_missing(self, abs_frame: int) -> None:
        """SAM returned nothing for this object on this frame."""
        self._checkpoint(abs_frame)
        self.backfill = None
        self.discarded = None
        self.streak += 1
        self.lost_run += 1
        if self.kalman.updates >= 1 and (self.lost_run >= 2 or self.streak >= 2):
            self.recovering = True
        self._discard_pending()

    def observe(
        self,
        abs_frame: int,
        binary: np.ndarray,
        scores: SamScores,
    ) -> tuple[GateDecision, MaskStats | None, PredictedBox | None, Foreground | None]:
        self._checkpoint(abs_frame)
        self.backfill = None
        self.discarded = None
        t = guard_time_s(self.info, abs_frame)
        expected_area = self.kalman.median_area()
        pred = self.kalman.predict(t)
        if not self.enabled:
            # Raw baseline: the whole mask's pixel mean, no component choice,
            # no gate, no recovery. Used for A/B comparison only.
            stats = mask_stats(binary)
            if stats is None:
                return GateDecision(False, 0.0, "目标不可见", {"missing_mask": True}), None, None, None
            self.kalman.update(t, stats.x, stats.y, stats.w, stats.h, area=stats.area)
            self.trusted_stats = stats
            self.trusted_frame = abs_frame
            return GateDecision(True, 0.5, ""), stats, None, None
        # Component choice may use the prediction only to rank regions. The
        # reported centre is still that region's own centroid.
        # A robust local quadratic through committed centres predicts curved
        # (ballistic) motion better than constant velocity; the Kalman filter
        # still supplies size and uncertainty.
        self.last_traj = self.trajectory.predict(abs_frame, t)
        if pred is not None and self.last_traj is not None:
            pred = replace(pred, x=self.last_traj[0], y=self.last_traj[1])
        self._obs_score = scores.object_score
        self._obs_change = None
        self._obs_color = None
        components = (
            mask_components(binary) if binary.size and binary.mean() <= 0.55 else None
        )
        stats = select_component(binary, pred, expected_area, self.seed_xy, components)
        fg = self._foreground(abs_frame)
        if stats is None:
            self.streak += 1
            self.lost_run += 1
            # One dropped frame is not a loss: the next good frame is trusted
            # at once. Confirmation is only needed after a real gap.
            if self.kalman.updates >= 1 and (self.lost_run >= 2 or self.streak >= 2):
                self.recovering = True
            self._discard_pending()
            if self.kalman.updates >= 2:
                found = recover_from_foreground(fg, pred, self.kalman.median_area())
                if found is not None:
                    # Frame difference is a motion cue, not the object's own
                    # segmentation: it is shown for review, never committed.
                    self.last_bad = (found.x, found.y)
                    return (
                        GateDecision(
                            False,
                            min(RECOVERED_CONFIDENCE, 0.45),
                            "运动候选待模型确认",
                            {"foreground_recovery": True, "reason_code": "motion_candidate"},
                        ),
                        found,
                        pred,
                        fg,
                    )
            return (
                GateDecision(
                    False, 0.0, "目标不可见", {"missing_mask": True, "reason_code": "model_invisible"}
                ),
                None,
                pred,
                fg,
            )
        appearance = self._appearance(stats, binary, pred, components)
        trajectory = None
        if self.last_traj is not None:
            tx, ty, tsigma, tn = self.last_traj
            size = float(np.sqrt(expected_area)) if expected_area else max(stats.w, stats.h)
            gap = abs_frame - self.trajectory.samples[-1][0] if self.trajectory.samples else 1
            trajectory = TrajectoryEvidence(
                deviation_px=float(np.hypot(stats.x - tx, stats.y - ty)),
                tolerance_px=trajectory_tolerance(tsigma, size, gap),
                projectile=self.projectile,
                points=tn,
                recovering=self.recovering,
            )
        reference = float(np.median(self.score_history)) if len(self.score_history) >= 3 else None
        decision = score_mask(
            stats,
            pred,
            fg,
            scores,
            self.kalman.median_area(),
            appearance,
            view_span=float(min(self.info.width, self.info.height)),
            updates=self.kalman.updates,
            trajectory=trajectory,
            reference_score=reference,
        )
        if not decision.accept:
            self.streak += 1
            self.lost_run = 0
            self.last_bad = (stats.x, stats.y)
            if self.streak >= 2:
                self.recovering = True
            self._discard_pending()
            return decision, stats, pred, fg
        if self.recovering and self.kalman.updates >= 1:
            return self._recovery_step(abs_frame, t, stats, decision), stats, pred, fg
        self._commit(abs_frame, t, stats, decision, self.current_rgb)
        return decision, stats, pred, fg

    def _recovery_step(
        self, abs_frame: int, t: float, stats: MaskStats, decision: GateDecision
    ) -> GateDecision:
        """Two-frame confirmation, used only after a loss."""
        patch = None if self.current_rgb is None else _appearance_patch(self.current_rgb, stats)
        pending = self.pending
        consistent, detail = (False, {}) if pending is None else self._consistent(pending, t, stats, patch)
        if pending is None or not consistent:
            if pending is not None:
                self._discard_pending()
            self.pending = _Pending(
                abs_frame,
                t,
                stats,
                decision,
                self.current_rgb,
                patch,
                self._obs_score,
                self._obs_change,
                self._obs_color,
            )
            diagnostics = {
                **decision.diagnostics,
                **detail,
                "reason_code": "recovery_pending",
                "recovery": "pending",
            }
            return GateDecision(False, decision.confidence, "恢复待确认：下一帧通过后入账", diagnostics)
        # Confirmed: commit the stored candidate first (its own time and
        # pixels), then this frame. Nothing predicted is ever committed.
        self.pending = None
        self._commit(
            pending.frame,
            pending.t,
            pending.stats,
            pending.decision,
            pending.rgb,
            evidence=(pending.score, pending.change, pending.color),
        )
        self._commit(abs_frame, t, stats, decision, self.current_rgb)
        self.backfill = pending
        diagnostics = {
            **decision.diagnostics,
            **detail,
            "recovery": "confirmed",
            "recovery_backfilled_frame": pending.frame,
        }
        return GateDecision(True, decision.confidence, "", diagnostics)

    def _consistent(
        self,
        pending: _Pending,
        t: float,
        stats: MaskStats,
        patch: np.ndarray | None,
    ) -> tuple[bool, dict[str, Any]]:
        fps = max(float(getattr(self.info, "fps", 30.0) or 30.0), 1e-6)
        frames = max(1.0, (t - pending.t) * fps)
        size = max(pending.stats.w, pending.stats.h, stats.w, stats.h, 1.0)
        step = self.kalman.median_step()
        span = float(min(self.info.width, self.info.height))
        allowed = max(3.0 * size, 3.0 * step * frames, 0.08 * span)
        moved = float(np.hypot(stats.x - pending.stats.x, stats.y - pending.stats.y))
        ratio = stats.area / max(pending.stats.area, 1.0)
        similarity = None
        if patch is not None and pending.patch is not None:
            similarity = appearance_similarity(pending.patch, patch)
        ok = moved <= allowed and 0.4 <= ratio <= 2.5 and (similarity is None or similarity >= 0.5)
        detail = {
            "recovery_step_px": round(moved, 3),
            "recovery_step_limit_px": round(allowed, 3),
            "recovery_area_ratio": round(ratio, 4),
        }
        if similarity is not None:
            detail["recovery_similarity"] = round(similarity, 4)
        return ok, detail

    def _discard_pending(self) -> None:
        if self.pending is not None:
            self.discarded = self.pending
            self.pending = None

    def _commit(
        self,
        abs_frame: int,
        t: float,
        stats: MaskStats,
        decision: GateDecision,
        rgb: np.ndarray | None,
        *,
        evidence: tuple[float | None, float | None, np.ndarray | None] | None = None,
    ) -> None:
        self.kalman.update(t, stats.x, stats.y, stats.w, stats.h, area=stats.area)
        self._update_appearance(stats, decision, rgb)
        self.trajectory.add(abs_frame, t, stats.x, stats.y)
        score, change, color = (
            evidence
            if evidence is not None
            else (self._obs_score, self._obs_change, self._obs_color)
        )
        if score is not None:
            self.score_history.append(float(score))
        if change is not None:
            self.change_history.append(float(change))
        if color is not None:
            if self.anchor_color is None:
                self.anchor_color = color
                self.recent_color = color
            elif decision.confidence >= 0.70:
                self.recent_color = color
        self.trusted_frame = abs_frame
        self.streak = 0
        self.lost_run = 0
        self.recovering = False
        self.last_bad = None
        self.last_relocalize = None

    def _appearance(
        self,
        stats: MaskStats | None,
        binary: np.ndarray,
        pred: PredictedBox | None = None,
        components: list[MaskStats] | None = None,
    ) -> AppearanceEvidence:
        if stats is None or self.current_rgb is None:
            return AppearanceEvidence()
        patch = _appearance_patch(self.current_rgb, stats)
        if patch is None or self.anchor_patch is None:
            return AppearanceEvidence()
        anchor = appearance_similarity(self.anchor_patch, patch)
        recent = (
            anchor
            if self.recent_patch is None
            else appearance_similarity(self.recent_patch, patch)
        )
        # The immutable anchor stops gradual background drift; the recent
        # template tolerates moderate rotation and blur.
        # A thin, motion-blurred object is partly transparent: its pixels
        # take on whatever is behind it, so the first-frame anchor alone
        # drifts out of reach. A strong match to the recent template (only
        # refreshed on confident commits) is also accepted, slightly
        # discounted; drift onto background is caught by the trajectory and
        # weak-evidence checks instead.
        similarity = max(0.65 * anchor + 0.35 * recent, 0.8 * recent)
        # After a loss the templates are not refreshed until a candidate is
        # confirmed. The pending candidate already passed identity against
        # them, so the next frame may also be compared with it.
        if self.pending is not None and self.pending.patch is not None:
            similarity = max(similarity, 0.9 * appearance_similarity(self.pending.patch, patch))
        ambiguous = False
        if binary.size and binary.mean() <= 0.18:
            alternatives: list[float] = []
            pool = components if components is not None else mask_components(binary)
            for item in pool[:32]:
                if abs(item.x - stats.x) < 1.0 and abs(item.y - stats.y) < 1.0:
                    continue
                if stats.contains(item.x, item.y):
                    continue  # a fragment of the chosen region
                other = _appearance_patch(self.current_rgb, item)
                if other is not None:
                    alternatives.append(appearance_similarity(self.anchor_patch, other))
            ambiguous = bool(alternatives and max(alternatives) >= anchor - 0.08)
        lk = None
        if self.trusted_rgb is not None and self.trusted_stats is not None:
            init = (0.0, 0.0)
            if pred is not None:
                init = (pred.x - self.trusted_stats.x, pred.y - self.trusted_stats.y)
            try:
                lk = lk_evidence(
                    self.trusted_rgb, self.current_rgb, self.trusted_stats, stats, init_disp=init
                )
            except Exception:  # noqa: BLE001
                lk = None
        color = color_signature(self.current_rgb, stats)
        self._obs_color = color
        color_sim = None
        if color is not None and self.anchor_color is not None:
            sims = [color_similarity(self.anchor_color, color)]
            if self.recent_color is not None:
                sims.append(color_similarity(self.recent_color, color))
            color_sim = max(v for v in sims if v is not None)
        change_ratio = None
        if self.prev2_rgb is not None:
            change = region_change(self.current_rgb, self.prev2_rgb, stats, self.shift_two)
            self._obs_change = change
            speed = 0.0
            if pred is not None:
                fps = max(float(getattr(self.info, "fps", 30.0) or 30.0), 1e-6)
                speed = float(np.hypot(pred.vx, pred.vy)) / fps
            size = max(stats.w, stats.h, 1.0)
            if (
                change is not None
                and len(self.change_history) >= 3
                and speed > 0.5 * size  # it moved clear of its old pixels
            ):
                typical = float(np.median(self.change_history))
                if typical > 4.0:
                    change_ratio = float(change / typical)
        return AppearanceEvidence(
            similarity=similarity,
            ambiguous=ambiguous,
            flow_quality=None if lk is None else lk.quality,
            forward_backward_error=None if lk is None else lk.fb_median_px,
            lk=lk,
            color_similarity=color_sim,
            change_ratio=change_ratio,
        )

    def _update_appearance(
        self, stats: MaskStats, decision: GateDecision, rgb: np.ndarray | None = None
    ) -> None:
        rgb = self.current_rgb if rgb is None else rgb
        if rgb is None:
            self.trusted_stats = stats
            return
        patch = _appearance_patch(rgb, stats)
        if patch is not None:
            if self.anchor_patch is None:
                self.anchor_patch = patch
                self.recent_patch = patch
            elif decision.confidence >= 0.70:
                self.recent_patch = patch
        self.trusted_rgb = rgb
        self.trusted_stats = stats

    def _foreground(self, abs_frame: int) -> Foreground | None:
        rgb = self._rgb(abs_frame)
        self.current_rgb = rgb
        fg = None
        if rgb is not None and self.prev_rgb is not None:
            try:
                fg = motion_foreground(self.prev_rgb, rgb)
            except Exception:  # noqa: BLE001
                fg = None
        shift = (0.0, 0.0) if fg is None else (float(fg.shift[0]), float(fg.shift[1]))
        # Camera motion from two frames ago to now, for the "static" check.
        self.shift_two = (shift[0] + self.prev_shift[0], shift[1] + self.prev_shift[1])
        if rgb is not None:
            self.prev2_rgb = self.prev_rgb
            self.prev_rgb = rgb
            self.prev_shift = shift
        return fg

    def _rgb(self, abs_frame: int) -> np.ndarray | None:
        try:
            if self.decoder is None:
                self.decoder = FrameDecoder(self.info)
            return self.decoder.frame(abs_frame)
        except Exception:  # noqa: BLE001
            return None

    def trusted_mask(self) -> tuple[int, np.ndarray] | None:
        """Full-resolution mask of the last committed region, with its frame."""
        stats = self.trusted_stats
        if stats is None or stats.pixels is None or self.trusted_frame is None:
            return None
        mask = np.zeros((self.info.height, self.info.width), dtype=bool)
        ox, oy = stats.origin
        ph, pw = stats.pixels.shape
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(self.info.width, ox + pw), min(self.info.height, oy + ph)
        if x1 <= x0 or y1 <= y0:
            return None
        mask[y0:y1, x0:x1] = stats.pixels[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
        return self.trusted_frame, mask

    def maybe_reprompt(
        self,
        abs_frame: int,
        pred: PredictedBox | None,
        fg: Foreground | None,
    ) -> TrackPrompt | None:
        """Bounded relocalisation while lost: at the 3rd miss, then every 8 frames.

        With a confident local trajectory the first attempt comes one frame
        earlier and aims at the trajectory; in projectile mode, when the frame
        difference shows nothing, the prompt goes to the trajectory itself.
        The resulting mask must still pass every check before it counts.
        """
        traj = self.last_traj
        if self.streak < 1 or abs_frame in self.reprompted:
            return None
        if self.last_relocalize is not None and abs_frame - self.last_relocalize < RELOCALIZE_EVERY:
            return None
        if pred is None:
            return None
        aim = pred if traj is None else replace(pred, x=traj[0], y=traj[1])
        prompt = None
        if fg is not None and fg.blobs:
            prompt = reprompt_candidate(
                fg.blobs,
                aim,
                expected_area=self.kalman.median_area(),
                scale=fg.scale,
            )
        # A moving blob right where the object should be: ask SAM to segment
        # it on this very frame (0.3.9 kept such frames; now SAM measures them).
        needed = 1 if prompt is not None else (2 if traj is not None else REJECT_STREAK)
        if self.streak < needed:
            return None
        if prompt is None and traj is not None and self.projectile:
            h, w = self.info.height, self.info.width
            if 0 <= traj[0] < w and 0 <= traj[1] < h:
                prompt = TrackPrompt(frame=abs_frame, kind=PromptKind.POSITIVE, x=traj[0], y=traj[1])
        if prompt is None:
            return None
        self.reprompted.add(abs_frame)
        self.last_relocalize = abs_frame
        return replace(prompt, frame=abs_frame)


class Sam2Tracker(Tracker):
    """Official desktop tracker. Requires SAM 2 unless a predictor is injected."""

    name = DEFAULT_SPEC.model_id
    version = DEFAULT_SPEC.version

    def __init__(
        self,
        predictor: VideoPredictor | None = None,
        *,
        spec: ModelSpec | None = None,
        anti_interference: bool = True,
    ) -> None:
        self._predictor = predictor
        self._spec = spec or DEFAULT_SPEC
        self.name = self._spec.model_id
        self.version = self._spec.version
        self.anti_interference = bool(anti_interference)
        self.dropped_frames: list[int] = []

    def track(
        self,
        info: VideoInfo,
        seed_xy: tuple[float, float],
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        prompts: list[TrackPrompt] | None = None,
        object_id: int = 1,
        stride: int = 1,
        image_size: int | None = None,
        anti_interference: bool | None = None,
        fill_gaps: bool = True,
        projectile: bool = False,
        **_unused: Any,
    ) -> TrackResult:
        t0 = time.perf_counter()
        start = max(0, int(start_frame))
        last = info.frame_count - 1 if end_frame is None else min(int(end_frame), info.frame_count - 1)
        if last < start:
            return TrackResult(
                clip_id=info.path.stem,
                points=[],
                failure_reason=FailureReason.INTERNAL,
                model_name=self.name,
                model_version=self.version,
                elapsed_s=time.perf_counter() - t0,
            )
        hints = list(prompts or [])
        if not hints:
            hints.append(
                TrackPrompt(
                    frame=start, kind=PromptKind.POSITIVE, x=seed_xy[0], y=seed_xy[1]
                )
            )
        extra = [prompt.frame for prompt in hints]
        sampled = sampled_frame_indices(start, last, stride, extra)
        predictor = self._predictor
        if predictor is None:
            predictor = load_sam2_predictor(self._spec, download=False)
            self._predictor = predictor

        sampled_points: list[TrackPoint] = []
        contours: dict[int, list[tuple[float, float]]] = {}
        total = last - start + 1
        _emit(progress, info.path.stem, 0, total, "track")
        enabled = self.anti_interference if anti_interference is None else bool(anti_interference)
        guard = _GuardSession(info, enabled, seed_xy)
        # Free flight: a candidate far off the local parabola is reviewed even
        # when every other check passes. Off by default (collisions, pendulums).
        guard.projectile = bool(projectile)
        self.dropped_frames = []
        cancelled = False
        try:
            if is_sam2_predictor(predictor):
                cancelled = self._track_windows(
                    predictor,
                    info,
                    sampled,
                    hints,
                    seed_xy,
                    object_id,
                    start,
                    total,
                    sampled_points,
                    contours,
                    guard=guard,
                    cancel=cancel,
                    progress=progress,
                )
            else:
                cancelled = self._track_whole(
                    predictor,
                    info,
                    sampled,
                    hints,
                    seed_xy,
                    object_id,
                    start,
                    last,
                    total,
                    sampled_points,
                    contours,
                    guard=guard,
                    cancel=cancel,
                    progress=progress,
                )
        finally:
            self.dropped_frames = list(guard.dropped)
            guard.close()
        if cancelled:
            return TrackResult(
                clip_id=info.path.stem,
                points=sampled_points,
                failure_reason=FailureReason.CANCELLED,
                model_name=self.name,
                model_version=self.version,
                elapsed_s=time.perf_counter() - t0,
            )

        sampled_points.sort(key=lambda point: point.frame)
        points = densify_track_points(sampled_points, start, last)
        if enabled and fill_gaps:
            # Short untrusted runs are bridged by the local fit (drawn, listed,
            # not measured); long gaps stay gaps.
            points = fill_fit_gaps(points, info)
        result = TrackResult(
            clip_id=info.path.stem,
            points=points,
            confidence=float(np.mean([p.confidence for p in points])) if points else 0.0,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - t0,
            quality_version=TRACK_QUALITY_VERSION,
        )
        result.contours = contours  # type: ignore[attr-defined]
        return result

    def _record(
        self,
        payload: Any,
        abs_frame: int,
        seed_xy: tuple[float, float],
        object_id: int,
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
        *,
        guard: _GuardSession,
        state: dict | None = None,
        local_idx: int | None = None,
    ) -> tuple[GateDecision, MaskStats | None, PredictedBox | None, Foreground | None]:
        _idx, obj_ids, masks = payload[:3]
        mask = self._pick_mask(obj_ids, masks, object_id)
        if mask is None:
            out_points.append(
                TrackPoint(
                    frame=abs_frame,
                    x=seed_xy[0],
                    y=seed_xy[1],
                    visible=False,
                    confidence=0.0,
                    note="这一帧没有该目标的掩膜",
                    status=TrackPointStatus.LOST,
                    source=TrackPointSource.AUTO,
                    diagnostics={"object_id": int(object_id), "missing_object": True},
                )
            )
            guard.mark_missing(abs_frame)
            self._apply_recovery_outcome(guard, abs_frame, out_points, out_contours)
            return GateDecision(False, 0.0, "missing_object"), None, None, None
        binary = logits_to_mask(mask)
        scores = extract_sam_scores(
            payload, state, int(_idx if local_idx is None else local_idx), object_id
        )
        scores = replace(scores, mask_quality=mask_logit_quality(mask, binary))
        decision, stats, pred, fg = guard.observe(abs_frame, binary, scores)
        diagnostics = {
            **decision.diagnostics,
            "object_id": int(object_id),
            "raw_object_score": scores.object_score,
            "raw_iou": scores.iou,
            "mask_quality": scores.mask_quality,
            "memory_admitted": (
                True
                if decision.accept
                else "provisional"
                if decision.diagnostics.get("recovery") == "pending"
                else False
            ),
        }
        if pred is not None:
            diagnostics["predicted_xy"] = [round(pred.x, 3), round(pred.y, 3)]
        self._apply_recovery_outcome(guard, abs_frame, out_points, out_contours)
        if stats is None or not decision.accept:
            x, y = (stats.x, stats.y) if stats is not None else seed_xy
            note = decision.reason or (REJECT_BACKGROUND if guard.enabled else "")
            out_points.append(
                TrackPoint(
                    frame=abs_frame,
                    x=x,
                    y=y,
                    visible=False,
                    confidence=decision.confidence,
                    note=note,
                    status=(
                        TrackPointStatus.REVIEW if stats is not None else TrackPointStatus.LOST
                    ),
                    source=TrackPointSource.AUTO,
                    diagnostics=diagnostics,
                )
            )
            return decision, stats, pred, fg
        out_points.append(
            TrackPoint(
                frame=abs_frame,
                x=stats.x,
                y=stats.y,
                visible=True,
                confidence=decision.confidence,
                status=TrackPointStatus.TRUSTED,
                source=TrackPointSource.AUTO,
                diagnostics=diagnostics,
            )
        )
        out_contours[abs_frame] = stats.contour
        return decision, stats, pred, fg

    @staticmethod
    def _apply_recovery_outcome(
        guard: _GuardSession,
        abs_frame: int,
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
    ) -> None:
        """Promote a confirmed recovery candidate, or explain a failed one."""
        for pending, confirmed in ((guard.backfill, True), (guard.discarded, False)):
            if pending is None:
                continue
            for index in range(len(out_points) - 1, -1, -1):
                point = out_points[index]
                if point.frame != pending.frame:
                    continue
                if point.manual or point.source is not TrackPointSource.AUTO:
                    break
                diagnostics = dict(point.diagnostics)
                if confirmed:
                    diagnostics.update(
                        {
                            "recovery": "backfilled",
                            "recovery_confirmed_by": int(abs_frame),
                            "reason_code": "recovered",
                            "memory_admitted": True,
                        }
                    )
                    out_points[index] = replace(
                        point,
                        x=pending.stats.x,
                        y=pending.stats.y,
                        visible=True,
                        confidence=pending.decision.confidence,
                        note="",
                        status=TrackPointStatus.TRUSTED,
                        diagnostics=diagnostics,
                    )
                    out_contours[pending.frame] = pending.stats.contour
                else:
                    diagnostics.update(
                        {"recovery": "unconfirmed", "reason_code": "recovery_unconfirmed"}
                    )
                    out_points[index] = replace(
                        point,
                        note="恢复未确认：下一帧没有通过同一目标检查",
                        diagnostics=diagnostics,
                    )
                break

    @staticmethod
    def _window_prompts(
        hints: list[TrackPrompt],
        window: list[int],
        seed_xy: tuple[float, float],
        carry: TrackPrompt | None,
    ) -> list[TrackPrompt]:
        if carry is None:
            return remap_prompts_to_samples(hints, window, seed_xy)
        index = {frame: i for i, frame in enumerate(window)}
        local = [carry]
        local += [replace(p, frame=index[p.frame]) for p in hints if p.frame in index]
        return local

    @staticmethod
    def _last_visible_frame(points: list[TrackPoint]) -> int | None:
        for point in reversed(points):
            if point.visible:
                return point.frame
        return None

    @staticmethod
    def _rewind_from(
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
        seen: set[int],
        abs_frame: int,
    ) -> None:
        kept = [p for p in out_points if p.frame < abs_frame]
        out_points.clear()
        out_points.extend(kept)
        for frame in [key for key in out_contours if key >= abs_frame]:
            del out_contours[frame]
        seen.difference_update({frame for frame in seen if frame >= abs_frame})

    @staticmethod
    def _carry_prompt(
        contours: dict[int, list[tuple[float, float]]],
        points: list[TrackPoint],
        frame: int | None,
    ) -> TrackPrompt | None:
        """Box around an accepted outline (fallback when no mask can be passed)."""
        if frame is None:
            return None
        contour = contours.get(frame)
        if contour:
            xs = [pt[0] for pt in contour]
            ys = [pt[1] for pt in contour]
            return TrackPrompt(
                frame=0,
                kind=PromptKind.BOX,
                x=min(xs),
                y=min(ys),
                x2=max(xs),
                y2=max(ys),
            )
        point = next(
            (p for p in reversed(points) if p.frame == frame and p.visible), None
        )
        if point is None:
            return None
        return TrackPrompt(
            frame=0, kind=PromptKind.POSITIVE, x=point.x, y=point.y
        )

    @staticmethod
    def _apply_mask_prompt(
        predictor: Any, state: dict, local_idx: int, object_id: int, mask: np.ndarray
    ) -> str:
        """Seed a window with the real trusted mask; box only if unsupported."""
        add_mask = getattr(predictor, "add_new_mask", None)
        if callable(add_mask):
            add_mask(inference_state=state, frame_idx=local_idx, obj_id=object_id, mask=mask)
            return "mask"
        ys, xs = np.nonzero(mask)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=local_idx,
            obj_id=object_id,
            box=np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32),
        )
        return "box"

    def _track_windows(
        self,
        predictor: Any,
        info: VideoInfo,
        sampled: list[int],
        hints: list[TrackPrompt],
        seed_xy: tuple[float, float],
        object_id: int,
        start: int,
        total: int,
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
        *,
        guard: _GuardSession,
        cancel: CancelToken | None,
        progress: ProgressCb | None,
    ) -> bool:
        """Run SAM window by window so memory does not grow with video length.

        Later windows are seeded with the last *committed* region, on the real
        frame it was measured in. When that frame lies before the window, it is
        decoded and prepended rather than relabelled as the window's first frame.
        """
        native = int(predictor.image_size)
        seen: set[int] = set()
        windows = sample_track_windows(sampled, TRACK_WINDOW_SAMPLED)
        for window_index, window_frames in enumerate(windows):
            if cancel and cancel.cancelled:
                return True
            local_prompts = [prompt for prompt in hints if prompt.frame in set(window_frames)]
            window = list(window_frames)
            seed_mask: np.ndarray | None = None
            if window_index > 0:
                trusted = guard.trusted_mask()
                if trusted is not None:
                    trusted_frame, seed_mask = trusted
                    if trusted_frame != window[0]:
                        window = [trusted_frame] + [f for f in window if f != trusted_frame]
                if seed_mask is None and not local_prompts:
                    for frame in window:
                        if frame in seen:
                            continue
                        out_points.append(
                            TrackPoint(
                                frame=frame,
                                x=seed_xy[0],
                                y=seed_xy[1],
                                visible=False,
                                confidence=0.0,
                                note="窗口边界前目标已失踪",
                                status=TrackPointStatus.LOST,
                                source=TrackPointSource.AUTO,
                                diagnostics={"window_seed_missing": True, "reason_code": "window_recovery"},
                            )
                        )
                        seen.add(frame)
                    continue
            offload = should_offload_video(predictor.device, len(window), native)
            images = LazyFrameTensors(
                info,
                window,
                native,
                compute_device=predictor.device,
                offload_video_to_cpu=offload,
                cancel=cancel,
            )
            auto_frames: set[int] = set()
            try:
                state = build_video_state(
                    predictor,
                    images,
                    info.height,
                    info.width,
                    offload_video_to_cpu=offload,
                )
                index = {frame: i for i, frame in enumerate(window)}
                user_local = [replace(p, frame=index[p.frame]) for p in local_prompts]
                if seed_mask is not None and not any(p.frame == 0 for p in user_local):
                    self._apply_mask_prompt(predictor, state, 0, object_id, seed_mask)
                    auto_frames.add(0)
                    self._apply_prompts(predictor, state, user_local, object_id)
                elif seed_mask is not None:
                    self._apply_prompts(predictor, state, user_local, object_id)
                else:
                    self._apply_prompts(
                        predictor,
                        state,
                        remap_prompts_to_samples(hints, window, seed_xy),
                        object_id,
                    )
                if cancel and cancel.cancelled:
                    return True
                if self._propagate_window(
                    predictor,
                    state,
                    window,
                    seed_xy,
                    object_id,
                    start,
                    total,
                    info.path.stem,
                    out_points,
                    out_contours,
                    seen,
                    guard,
                    auto_frames,
                    cancel=cancel,
                    progress=progress,
                ):
                    return True
            except ModelNotAvailable:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ModelNotAvailable(f"SAM 2 推理失败：{exc}") from exc
            finally:
                images.close()
                _release_device_cache(predictor.device)
        return False

    @staticmethod
    def _forget(
        state: dict | None,
        local_idx: int,
        object_id: int,
        decision: GateDecision,
        auto_frames: set[int],
    ) -> None:
        """Discard an uncommitted frame from SAM memory.

        A pending recovery candidate stays in SAM memory provisionally (its
        output and any automatic prompt): it is what keeps SAM on the object
        while the next frame is checked. Dropping it at once made SAM lose a
        correctly found object on the real clip. If the next frame does not
        confirm it, the caller removes it then.
        """
        if decision.diagnostics.get("recovery") == "pending":
            return
        drop_memory_frame(
            state,
            local_idx,
            object_id,
            conditioning=local_idx in auto_frames,
        )
        auto_frames.discard(local_idx)

    def _propagate_window(
        self,
        predictor: Any,
        state: dict,
        window: list[int],
        seed_xy: tuple[float, float],
        object_id: int,
        start: int,
        total: int,
        clip_id: str,
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
        seen: set[int],
        guard: _GuardSession,
        auto_frames: set[int] | None = None,
        *,
        cancel: CancelToken | None,
        progress: ProgressCb | None,
    ) -> bool:
        auto = auto_frames if auto_frames is not None else set()
        local_of = {frame: i for i, frame in enumerate(window)}
        start_idx = 0
        while start_idx < len(window):
            stream = predictor.propagate_in_video(
                state,
                start_frame_idx=start_idx,
                max_frame_num_to_track=len(window) - start_idx,
            )
            restart: int | None = None
            for payload in stream:
                if cancel and cancel.cancelled:
                    return True
                idx = int(payload[0])
                if idx < 0 or idx >= len(window):
                    continue
                abs_frame = window[idx]
                if abs_frame in seen:
                    continue
                decision, _stats, pred, fg = self._record(
                    payload,
                    abs_frame,
                    seed_xy,
                    object_id,
                    out_points,
                    out_contours,
                    guard=guard,
                    state=state,
                    local_idx=idx,
                )
                seen.add(abs_frame)
                if guard.discarded is not None:
                    old = local_of.get(guard.discarded.frame)
                    if old is not None:
                        drop_memory_frame(state, old, object_id, conditioning=old in auto)
                        auto.discard(old)
                if not decision.accept:
                    self._forget(state, idx, object_id, decision, auto)
                    if abs_frame not in guard.dropped and decision.diagnostics.get("recovery") != "pending":
                        guard.dropped.append(abs_frame)
                    extra = guard.maybe_reprompt(abs_frame, pred, fg)
                    if extra is not None:
                        prompts = [replace(extra, frame=idx)]
                        if guard.last_bad is not None:
                            prompts.append(
                                TrackPrompt(
                                    frame=idx,
                                    kind=PromptKind.NEGATIVE,
                                    x=guard.last_bad[0],
                                    y=guard.last_bad[1],
                                )
                            )
                        self._apply_prompts(predictor, state, prompts, object_id)
                        auto.add(idx)
                        self._rewind_from(out_points, out_contours, seen, abs_frame)
                        guard.restore(abs_frame)
                        restart = idx
                        break
                _emit(progress, clip_id, abs_frame - start + 1, total, "track")
            if restart is None:
                break
            start_idx = restart
        return False

    def _track_whole(
        self,
        predictor: Any,
        info: VideoInfo,
        sampled: list[int],
        hints: list[TrackPrompt],
        seed_xy: tuple[float, float],
        object_id: int,
        start: int,
        last: int,
        total: int,
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
        *,
        guard: _GuardSession,
        cancel: CancelToken | None,
        progress: ProgressCb | None,
    ) -> bool:
        """Predictors that decode the file themselves (tests) stay single-pass."""
        state = predictor.init_state(str(info.path))
        self._apply_prompts(predictor, state, hints, object_id)
        if cancel and cancel.cancelled:
            return True
        allowed = set(sampled)
        auto: set[int] = set()
        memory = state if isinstance(state, dict) else None
        try:
            start_idx = start
            while start_idx <= last:
                stream = predictor.propagate_in_video(
                    state,
                    start_frame_idx=start_idx,
                    max_frame_num_to_track=last - start_idx + 1,
                )
                restart: int | None = None
                for payload in stream:
                    if cancel and cancel.cancelled:
                        return True
                    abs_frame = int(payload[0])
                    if abs_frame not in allowed:
                        continue
                    decision, _stats, pred, fg = self._record(
                        payload,
                        abs_frame,
                        seed_xy,
                        object_id,
                        out_points,
                        out_contours,
                        guard=guard,
                        state=memory,
                        local_idx=abs_frame,
                    )
                    if guard.discarded is not None:
                        frame_d = guard.discarded.frame
                        drop_memory_frame(memory, frame_d, object_id, conditioning=frame_d in auto)
                        auto.discard(frame_d)
                    if not decision.accept:
                        self._forget(memory, abs_frame, object_id, decision, auto)
                        if abs_frame not in guard.dropped and decision.diagnostics.get("recovery") != "pending":
                            guard.dropped.append(abs_frame)
                        extra = guard.maybe_reprompt(abs_frame, pred, fg)
                        if extra is not None:
                            prompts = [replace(extra, frame=abs_frame)]
                            if guard.last_bad is not None:
                                prompts.append(
                                    TrackPrompt(
                                        frame=abs_frame,
                                        kind=PromptKind.NEGATIVE,
                                        x=guard.last_bad[0],
                                        y=guard.last_bad[1],
                                    )
                                )
                            self._apply_prompts(predictor, state, prompts, object_id)
                            auto.add(abs_frame)
                            seen = {p.frame for p in out_points}
                            self._rewind_from(out_points, out_contours, seen, abs_frame)
                            guard.restore(abs_frame)
                            restart = abs_frame
                            break
                    _emit(progress, info.path.stem, abs_frame - start + 1, total, "track")
                if restart is None:
                    break
                start_idx = restart
        except ModelNotAvailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelNotAvailable(f"SAM 2 推理失败：{exc}") from exc
        return False

    @staticmethod
    def _pick_mask(obj_ids: Any, masks: Any, object_id: int) -> Any:
        ids = [int(i) for i in list(obj_ids)]
        if object_id in ids:
            return masks[ids.index(object_id)]
        return None

    @staticmethod
    def _apply_prompts(
        predictor: VideoPredictor,
        state: dict,
        prompts: list[TrackPrompt],
        object_id: int,
    ) -> None:
        by_frame: dict[int, list[TrackPrompt]] = {}
        for prompt in prompts:
            by_frame.setdefault(prompt.frame, []).append(prompt)
        for frame, group in sorted(by_frame.items()):
            points: list[list[float]] = []
            labels: list[int] = []
            box = None
            for prompt in group:
                if prompt.kind == PromptKind.BOX and prompt.x2 is not None and prompt.y2 is not None:
                    box = [prompt.x, prompt.y, prompt.x2, prompt.y2]
                elif prompt.kind == PromptKind.NEGATIVE:
                    points.append([prompt.x, prompt.y])
                    labels.append(0)
                else:
                    points.append([prompt.x, prompt.y])
                    labels.append(1)
            kwargs: dict[str, Any] = {
                "inference_state": state,
                "frame_idx": frame,
                "obj_id": object_id,
            }
            if points:
                kwargs["points"] = np.array(points, dtype=np.float32)
                kwargs["labels"] = np.array(labels, dtype=np.int32)
            if box is not None:
                kwargs["box"] = np.array(box, dtype=np.float32)
            predictor.add_new_points_or_box(**kwargs)


def preview_mask_on_frame(
    info: VideoInfo,
    frame_index: int,
    seed_xy: tuple[float, float],
    radius: int = 12,
) -> np.ndarray:
    """CPU-only preview disk used when SAM is not yet loaded (tests / fallback UI)."""
    decoder = FrameDecoder(info)
    try:
        frame = decoder.frame(frame_index)
    finally:
        decoder.close()
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    return (xx - seed_xy[0]) ** 2 + (yy - seed_xy[1]) ** 2 <= radius**2
