"""Motion-aware gate around SAM 2 masks (SAMURAI-style, no extra weights).

Rejects masks that jump onto static stripes / foliage, drops those frames
from SAM 2 memory, and proposes a positive click near the predicted ball.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ai.contracts import PromptKind, TrackPrompt

REJECT_BACKGROUND = "疑似跳到背景"
REJECT_STREAK = 3
FG_MAX_SIDE = 360
FG_SHIFT_FRAC = 0.08
FG_MAX_FILL = 0.15
FG_MIN_OVERLAP = 0.20
AREA_EXPLODE = 8.0
WARMUP_UPDATES = 4
MIN_DIST_PX = 48.0


@dataclass(frozen=True)
class MaskStats:
    x: float
    y: float
    w: float
    h: float
    area: float
    contour: list[tuple[float, float]]


@dataclass(frozen=True)
class PredictedBox:
    x: float
    y: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    step: float = 0.0
    sigma: float = MIN_DIST_PX


@dataclass(frozen=True)
class SamScores:
    object_score: float | None = None
    iou: float | None = None


@dataclass(frozen=True)
class Blob:
    x: float
    y: float
    area: float
    w: float
    h: float
    roundness: float


@dataclass(frozen=True)
class Foreground:
    mask: np.ndarray
    blobs: tuple[Blob, ...]
    shift: tuple[float, float]
    reliable: bool
    scale: float


@dataclass(frozen=True)
class GateDecision:
    accept: bool
    confidence: float
    reason: str = ""


def mask_stats(mask: np.ndarray) -> MaskStats | None:
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        binary = binary.reshape(binary.shape[-2], binary.shape[-1])
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return MaskStats(
        x=float(xs.mean()),
        y=float(ys.mean()),
        w=max(1.0, x1 - x0 + 1.0),
        h=max(1.0, y1 - y0 + 1.0),
        area=float(xs.size),
        contour=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
    )


class ConstantVelocityKalman:
    """6-state filter: x, y, vx, vy, w, h. Time is seconds (real PTS)."""

    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._t: float | None = None
        self._areas: deque[float] = deque(maxlen=16)
        self._steps: deque[float] = deque(maxlen=12)
        self._residuals: deque[float] = deque(maxlen=16)
        self.updates = 0

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def median_area(self) -> float | None:
        if len(self._areas) < 2:
            return None
        return float(np.median(self._areas))

    def median_step(self) -> float:
        if not self._steps:
            return MIN_DIST_PX
        return float(max(MIN_DIST_PX, np.median(self._steps)))

    def residual_sigma(self) -> float:
        if len(self._residuals) < 3:
            if self._P is None:
                return MIN_DIST_PX
            return float(max(MIN_DIST_PX, np.sqrt(max(self._P[0, 0] + self._P[1, 1], 1e-6))))
        return float(max(MIN_DIST_PX, np.std(self._residuals) * 1.4826))

    def predict(self, t: float) -> PredictedBox | None:
        if self._x is None or self._t is None:
            return None
        dt = max(1e-4, float(t) - self._t)
        x = self._advance(self._x, dt)
        step = self.median_step()
        return PredictedBox(
            x=float(x[0]),
            y=float(x[1]),
            w=float(max(1.0, x[4])),
            h=float(max(1.0, x[5])),
            vx=float(x[2]),
            vy=float(x[3]),
            step=step,
            sigma=self.residual_sigma(),
        )

    def update(
        self, t: float, x: float, y: float, w: float, h: float, area: float | None = None
    ) -> PredictedBox:
        z = np.array([x, y, max(1.0, w), max(1.0, h)], dtype=np.float64)
        pix = float(area if area is not None else z[2] * z[3])
        if self._x is None:
            self._x = np.array([x, y, 0.0, 0.0, z[2], z[3]], dtype=np.float64)
            self._P = np.diag([64.0, 64.0, 2.5e5, 2.5e5, 64.0, 64.0])
            self._t = float(t)
            self._areas.append(pix)
            self.updates = 1
            return self.predict(t) or PredictedBox(x=x, y=y, w=z[2], h=z[3])
        dt = max(1e-4, float(t) - self._t)
        if self.updates == 1:
            vx = (x - float(self._x[0])) / dt
            vy = (y - float(self._x[1])) / dt
            step = float(np.hypot(x - self._x[0], y - self._x[1]))
            self._x = np.array([x, y, vx, vy, z[2], z[3]], dtype=np.float64)
            self._t = float(t)
            self._steps.append(max(step, 1e-3))
            self._areas.append(pix)
            self.updates = 2
            return PredictedBox(
                x=x, y=y, w=float(z[2]), h=float(z[3]), vx=vx, vy=vy, step=step, sigma=MIN_DIST_PX
            )
        self.updates += 1
        pred = self._advance(self._x, dt)
        P = self._F(dt) @ self._P @ self._F(dt).T + self._Q(dt)
        H = self._H()
        R = np.diag([4.0, 4.0, 16.0, 16.0])
        yk = z - H @ pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        self._x = pred + K @ yk
        self._P = (np.eye(6) - K @ H) @ P
        self._t = float(t)
        step = float(np.hypot(yk[0], yk[1]))
        # Use predicted displacement this step, not innovation, for typical speed.
        disp = float(np.hypot(self._x[2] * dt, self._x[3] * dt))
        self._steps.append(max(disp, 1e-3))
        self._residuals.append(step)
        self._areas.append(pix)
        return PredictedBox(
            x=float(self._x[0]),
            y=float(self._x[1]),
            w=float(max(1.0, self._x[4])),
            h=float(max(1.0, self._x[5])),
            vx=float(self._x[2]),
            vy=float(self._x[3]),
            step=self.median_step(),
            sigma=self.residual_sigma(),
        )

    @staticmethod
    def _advance(state: np.ndarray, dt: float) -> np.ndarray:
        out = state.copy()
        out[0] = state[0] + state[2] * dt
        out[1] = state[1] + state[3] * dt
        return out

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    @staticmethod
    def _H() -> np.ndarray:
        H = np.zeros((4, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 4] = 1.0
        H[3, 5] = 1.0
        return H

    @staticmethod
    def _Q(dt: float) -> np.ndarray:
        q_pos = 16.0 * dt
        q_vel = 400.0 * dt
        q_size = 8.0 * dt
        return np.diag([q_pos, q_pos, q_vel, q_vel, q_size, q_size])


def _to_gray(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float32)


def _downsample(gray: np.ndarray, max_side: int = FG_MAX_SIDE) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = max(h, w) / float(max(1, max_side))
    if scale <= 1.01:
        return gray, 1.0
    step = max(1, int(round(scale)))
    return gray[::step, ::step], float(step)


def estimate_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Phase-correlation translation of prev → cur, in downsampled pixels."""
    a = prev - float(prev.mean())
    b = cur - float(cur.mean())
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag = np.maximum(mag, 1e-9)
    peak = np.fft.irfft2(cross / mag, s=a.shape)
    iy, ix = np.unravel_index(int(np.argmax(peak)), peak.shape)
    h, w = peak.shape
    dy = iy if iy <= h // 2 else iy - h
    dx = ix if ix <= w // 2 else ix - w
    return float(dx), float(dy)


def _shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.roll(np.roll(image, int(round(dy)), axis=0), int(round(dx)), axis=1)


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            y0, y1 = max(0, dy), min(h, h + dy)
            x0, x1 = max(0, dx), min(w, w + dx)
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            out[y0:y1, x0:x1] |= mask[sy0:sy1, sx0:sx1]
    return out


def _erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return ~_dilate(~mask, radius)


def _label_blobs(mask: np.ndarray, min_area: int = 4, max_blobs: int = 48) -> list[Blob]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    blobs: list[Blob] = []
    current = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        if labels[y, x] != 0:
            continue
        current += 1
        stack = [(y, x)]
        labels[y, x] = current
        px: list[int] = []
        py: list[int] = []
        while stack:
            cy, cx = stack.pop()
            py.append(cy)
            px.append(cx)
            for ny in (cy - 1, cy, cy + 1):
                if ny < 0 or ny >= h:
                    continue
                for nx in (cx - 1, cx, cx + 1):
                    if nx < 0 or nx >= w:
                        continue
                    if not mask[ny, nx] or labels[ny, nx] != 0:
                        continue
                    labels[ny, nx] = current
                    stack.append((ny, nx))
        area = len(px)
        if area < min_area:
            continue
        arr_x = np.asarray(px, dtype=np.float64)
        arr_y = np.asarray(py, dtype=np.float64)
        bw = float(arr_x.max() - arr_x.min() + 1.0)
        bh = float(arr_y.max() - arr_y.min() + 1.0)
        roundness = float(min(bw, bh) / max(bw, bh, 1.0))
        blobs.append(
            Blob(
                x=float(arr_x.mean()),
                y=float(arr_y.mean()),
                area=float(area),
                w=bw,
                h=bh,
                roundness=roundness,
            )
        )
        if len(blobs) >= max_blobs:
            break
    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs


def motion_foreground(
    prev: np.ndarray,
    cur: np.ndarray,
    *,
    max_side: int = FG_MAX_SIDE,
    shift: tuple[float, float] | None = None,
) -> Foreground:
    """Aligned frame difference. `shift` is optional prev→cur translation in full pixels."""
    prev_g, scale = _downsample(_to_gray(prev), max_side)
    cur_g, _ = _downsample(_to_gray(cur), max_side)
    if shift is None:
        estimated = estimate_shift(prev_g, cur_g)
        candidates = [(0.0, 0.0), estimated]
    else:
        candidates = [(shift[0] / scale, shift[1] / scale)]
    best: tuple[float, float, float, np.ndarray] | None = None
    for dx, dy in candidates:
        aligned = _shift_image(prev_g, dx, dy)
        diff = np.abs(cur_g - aligned)
        med = float(np.median(diff))
        thr = max(18.0, med * 3.5)
        binary = diff > thr
        fill = float(binary.mean()) if binary.size else 1.0
        if best is None or fill < best[0]:
            best = (fill, dx, dy, binary)
    assert best is not None
    fill, dx, dy, binary = best
    binary = _dilate(_erode(binary, 1), 1)
    fill = float(binary.mean()) if binary.size else 1.0
    diag = float(np.hypot(*cur_g.shape))
    shift_px = float(np.hypot(dx, dy))
    blobs = _label_blobs(binary)
    compact = [b for b in blobs if b.area < 0.05 * binary.size]
    reliable = (
        shift_px < FG_SHIFT_FRAC * max(diag, 1.0)
        and fill < FG_MAX_FILL
        and len(compact) >= 1
    )
    return Foreground(
        mask=binary,
        blobs=tuple(compact[:24]),
        shift=(dx * scale, dy * scale),
        reliable=reliable,
        scale=scale,
    )


def _overlap_fraction(stats: MaskStats, fg: Foreground) -> float:
    if stats.area <= 0 or fg.mask.size == 0:
        return 0.0
    scale = fg.scale
    x0 = int(max(0, round(min(p[0] for p in stats.contour) / scale)))
    y0 = int(max(0, round(min(p[1] for p in stats.contour) / scale)))
    x1 = int(min(fg.mask.shape[1], round(max(p[0] for p in stats.contour) / scale) + 1))
    y1 = int(min(fg.mask.shape[0], round(max(p[1] for p in stats.contour) / scale) + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = fg.mask[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


def search_limit(pred: PredictedBox, view_span: float, updates: int) -> float:
    """How far the ball may move before a mask is a jump.

    Until velocity is known, allow a large fraction of the frame. After that,
    follow recent motion. A few pixels of slack was rejecting every fast ball.
    """
    span = max(float(view_span), MIN_DIST_PX)
    if updates < WARMUP_UPDATES:
        return max(0.22 * span, 4.0 * pred.step, MIN_DIST_PX)
    return max(4.0 * pred.step, 3.0 * pred.sigma, 0.04 * span, MIN_DIST_PX)


def score_mask(
    stats: MaskStats | None,
    pred: PredictedBox | None,
    fg: Foreground | None,
    sam: SamScores | None = None,
    median_area: float | None = None,
    *,
    view_span: float = 720.0,
    updates: int = 0,
) -> GateDecision:
    """Reject only a mask that left the ball and grew into the background.

    A mildly negative SAM object score, a smaller mask, or a fast but
    continuous move stays visible. Those used to wipe almost the whole track.
    """
    if stats is None:
        return GateDecision(False, 0.0, REJECT_BACKGROUND)
    sam = sam or SamScores()
    dist = 0.0
    limit = search_limit(
        pred or PredictedBox(x=stats.x, y=stats.y, w=stats.w, h=stats.h),
        view_span,
        updates,
    )
    if pred is not None:
        dist = float(np.hypot(stats.x - pred.x, stats.y - pred.y))
    ratio = 1.0
    if median_area is not None and median_area > 4.0:
        ratio = stats.area / median_area
    exploded = ratio >= AREA_EXPLODE
    far = pred is not None and dist > limit
    # Foliage grab: the mask both leaves the prediction and covers much more.
    jumped = far and ratio >= 3.0
    if updates >= 2 and (exploded or jumped):
        return GateDecision(False, 0.2, REJECT_BACKGROUND)
    # A clearly absent object that is also far from the ball. Logit alone is not enough:
    # small or blurred balls often sit slightly below zero.
    if sam.object_score is not None and sam.object_score < -6.0 and far:
        return GateDecision(False, 0.0, REJECT_BACKGROUND)

    motion = 1.0 if limit <= 0 else float(np.exp(-dist / limit))
    iou = 1.0 if sam.iou is None else float(np.clip(sam.iou, 0.0, 1.0))
    obj = 1.0
    if sam.object_score is not None:
        obj = float(1.0 / (1.0 + np.exp(-float(sam.object_score))))
    blended = 0.45 * max(obj, 0.7) + 0.25 * max(iou, 0.6) + 0.30 * motion
    return GateDecision(True, float(np.clip(max(blended, 0.72), 0.0, 1.0)), "")


def reprompt_candidate(
    blobs: tuple[Blob, ...] | list[Blob],
    pred: PredictedBox | None,
    *,
    expected_area: float | None = None,
    scale: float = 1.0,
) -> TrackPrompt | None:
    if pred is None or not blobs:
        return None
    radius = max(pred.step * 2.0, 2.0 * pred.w, 24.0)
    best: tuple[float, Blob] | None = None
    for blob in blobs:
        x = blob.x * scale
        y = blob.y * scale
        dist = float(np.hypot(x - pred.x, y - pred.y))
        if dist > radius:
            continue
        area = blob.area * scale * scale
        if expected_area and expected_area > 1:
            area_score = min(area, expected_area) / max(area, expected_area)
        else:
            area_score = 1.0
        score = float(np.exp(-dist / radius) * area_score * blob.roundness)
        if best is None or score > best[0]:
            best = (score, Blob(x, y, area, blob.w * scale, blob.h * scale, blob.roundness))
    if best is None:
        return None
    chosen = best[1]
    return TrackPrompt(
        frame=0, kind=PromptKind.POSITIVE, x=chosen.x, y=chosen.y
    )


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        try:
            value = value.numpy()
        except Exception:  # noqa: BLE001
            pass
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:  # noqa: BLE001
        return None
    if arr.size == 0:
        return None
    return float(arr[0])


def extract_sam_scores(
    payload: Any,
    state: dict | None,
    frame_idx: int,
    object_id: int = 1,
) -> SamScores:
    _ = object_id
    if isinstance(payload, (tuple, list)) and len(payload) >= 4:
        extra = payload[3]
        if isinstance(extra, dict):
            score = _as_float(extra.get("object_score_logits", extra.get("object_score")))
            iou = _as_float(extra.get("iou_predictions", extra.get("iou")))
            if score is not None or iou is not None:
                return SamScores(score, iou)
    if not isinstance(state, dict):
        return SamScores()
    for key in ("output_dict_per_obj", "temp_output_dict_per_obj", "output_dict"):
        blob = state.get(key)
        if not isinstance(blob, dict):
            continue
        for store_name in ("non_cond_frame_outputs", "cond_frame_outputs"):
            if store_name in blob and isinstance(blob[store_name], dict):
                item = blob[store_name].get(frame_idx)
                if isinstance(item, dict):
                    score = _as_float(item.get("object_score_logits"))
                    iou = _as_float(item.get("iou_predictions"))
                    if score is not None or iou is not None:
                        return SamScores(score, iou)
        for inner in blob.values():
            if not isinstance(inner, dict):
                continue
            for store_name in ("non_cond_frame_outputs", "cond_frame_outputs"):
                store = inner.get(store_name)
                if isinstance(store, dict) and frame_idx in store:
                    item = store[frame_idx]
                    if isinstance(item, dict):
                        score = _as_float(item.get("object_score_logits"))
                        iou = _as_float(item.get("iou_predictions"))
                        if score is not None or iou is not None:
                            return SamScores(score, iou)
    return SamScores()


def drop_memory_frame(state: dict | None, frame_idx: int) -> bool:
    """Remove a frame from SAM 2 memory so a jumped mask is not reused."""
    if not isinstance(state, dict):
        return False
    removed = False

    def _drop(store: Any) -> None:
        nonlocal removed
        if not isinstance(store, dict):
            return
        for key in ("non_cond_frame_outputs", "cond_frame_outputs"):
            outs = store.get(key)
            if isinstance(outs, dict) and frame_idx in outs:
                outs.pop(frame_idx, None)
                removed = True
        for value in list(store.values()):
            if isinstance(value, dict):
                _drop(value)

    for key in ("output_dict", "output_dict_per_obj", "temp_output_dict_per_obj"):
        _drop(state.get(key))
    tracked = state.get("frames_tracked_per_obj")
    if isinstance(tracked, dict):
        for obj in tracked.values():
            if isinstance(obj, dict) and frame_idx in obj:
                obj.pop(frame_idx, None)
                removed = True
    return removed


def time_s(info: Any, frame: int) -> float:
    pts = getattr(info, "pts_ms", ()) or ()
    if 0 <= frame < len(pts):
        return float(pts[frame]) / 1000.0
    fps = float(getattr(info, "fps", 30.0) or 30.0)
    return frame / max(fps, 1e-6)
