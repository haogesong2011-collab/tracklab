"""Conservative camera-shake compensation by background registration.

Frames are analysed at a bounded scale. Anchors prefer static background
texture and avoid tracked subjects. A translation is estimated first; a
similarity (rotation/scale) is used only when it clearly improves held-out
residuals. Per-frame estimates are gated so compensation cannot inject more
jitter into a track than the raw measurements already have.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ai.contracts import CancelToken, ProgressCb, ProgressEvent, TrackPoint, TrackResult
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo

ANALYZE_MAX = 720
GRID = 8
POINTS_PER_CORNER = 2
MAX_ANCHORS = 20
PATCH = 15
COARSE_SCALE = 3
NEAR_SEARCH = 24
MID_SEARCH = 48
COARSE_SEARCH = 64
FINE_SEARCH = 8
NCC_MIN = 0.50
CORNER_FRAC = 0.18
MIN_SPACING = 28.0
FB_THRESH = 5.0
INLIER_PX = 3.0
TRANS_INLIER_PX = 6.0
RELOCK_EVERY = 6
MAX_GAP = 3
DEADBAND_PX = 1.2
DEADBAND_DEG = 0.35
DEADBAND_SCALE = 0.008
SPIKE_PX = 16.0
NOISE_PX = 1.25
MIN_BASELINE = 8.0
EXCLUDE_RADIUS = 36.0
THETA_UPGRADE = math.radians(0.6)
SCALE_UPGRADE = 0.012
SIM_IMPROVE = 0.70


@dataclass(frozen=True)
class Similarity:
    """x' = a x - b y + tx, y' = b x + a y + ty (scale-rotation + translation)."""

    a: float = 1.0
    b: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return self.a * x - self.b * y + self.tx, self.b * x + self.a * y + self.ty

    def inverse(self) -> Similarity:
        det = self.a * self.a + self.b * self.b
        if det < 1e-12:
            return IDENTITY
        ia = self.a / det
        ib = -self.b / det
        itx = -(ia * self.tx - ib * self.ty)
        ity = -(ib * self.tx + ia * self.ty)
        return Similarity(ia, ib, itx, ity)

    @property
    def scale(self) -> float:
        return math.hypot(self.a, self.b)

    @property
    def theta(self) -> float:
        return math.atan2(self.b, self.a)

    def residual(self, x: float, y: float, xp: float, yp: float) -> float:
        nx, ny = self.apply(x, y)
        return math.hypot(nx - xp, ny - yp)

    def scaled(self, factor: float) -> Similarity:
        if abs(factor - 1.0) < 1e-9:
            return self
        return Similarity(self.a, self.b, self.tx * factor, self.ty * factor)

    def is_similarity(self) -> bool:
        return abs(self.scale - 1.0) > 1e-4 or abs(self.theta) > 1e-5


IDENTITY = Similarity()


@dataclass(frozen=True)
class ShakeCompensation:
    dx: tuple[float, ...]
    dy: tuple[float, ...]
    anchors: tuple[tuple[tuple[float, float] | None, ...], ...]
    model_name: str = "translation_gated"
    model_version: str = "0.4.0"
    transforms: tuple[Similarity, ...] = field(default_factory=tuple)
    usable: tuple[bool, ...] = field(default_factory=tuple)
    inliers: tuple[int, ...] = field(default_factory=tuple)
    residual: tuple[float, ...] = field(default_factory=tuple)
    quality: float = 1.0
    recommended: bool = True
    gains: tuple[float, ...] = field(default_factory=tuple)
    model_kind: str = "translation"
    fallback_reason: str = ""
    raw_jitter: float = 0.0
    stable_jitter: float = 0.0

    def _index(self, frame: int) -> int:
        n = len(self.dx) if self.dx else len(self.transforms)
        if n <= 0:
            return 0
        return max(0, min(int(frame), n - 1))

    def offset(self, frame: int) -> tuple[float, float]:
        if self.transforms:
            index = self._index(frame)
            if 0 <= index < len(self.transforms):
                model = self.transforms[index]
                return model.tx, model.ty
        if not self.dx:
            return 0.0, 0.0
        index = self._index(frame)
        return self.dx[index], self.dy[index]

    def transform_at(self, frame: int) -> Similarity:
        if not self.transforms:
            dx, dy = self.offset(frame)
            return Similarity(1.0, 0.0, dx, dy)
        return self.transforms[self._index(frame)]

    def is_usable(self, frame: int) -> bool:
        if not self.usable:
            return True
        return self.usable[self._index(frame)]

    def gain_at(self, frame: int) -> float:
        if not self.gains:
            return 1.0 if self.is_usable(frame) else 0.0
        return float(self.gains[self._index(frame)])

    def to_stable(self, x: float, y: float, frame: int) -> tuple[float, float]:
        gain = self.gain_at(frame)
        if gain <= 1e-6:
            return x, y
        sx, sy = self.transform_at(frame).inverse().apply(x, y)
        if gain >= 1.0 - 1e-6:
            return sx, sy
        return (1.0 - gain) * x + gain * sx, (1.0 - gain) * y + gain * sy

    def to_raw(self, x: float, y: float, frame: int) -> tuple[float, float]:
        gain = self.gain_at(frame)
        if gain <= 1e-6:
            return x, y
        if gain >= 1.0 - 1e-6:
            return self.transform_at(frame).apply(x, y)
        sx, sy = x, y
        rx, ry = self.transform_at(frame).apply(sx, sy)
        inv = max(gain, 1e-6)
        return (sx - (1.0 - gain) * rx) / inv, (sy - (1.0 - gain) * ry) / inv

    def anchors_at(self, frame: int) -> list[tuple[float, float]]:
        if not self.anchors:
            return []
        index = max(0, min(int(frame), len(self.anchors) - 1))
        return [pt for pt in self.anchors[index] if pt is not None]

    def with_gains(
        self,
        gains: Sequence[float],
        *,
        recommended: bool | None = None,
        fallback_reason: str | None = None,
        raw_jitter: float | None = None,
        stable_jitter: float | None = None,
    ) -> ShakeCompensation:
        values = tuple(float(max(0.0, min(1.0, g))) for g in gains)
        usable = tuple(g > 0.5 for g in values)
        return ShakeCompensation(
            dx=self.dx,
            dy=self.dy,
            anchors=self.anchors,
            model_name=self.model_name,
            model_version=self.model_version,
            transforms=self.transforms,
            usable=self.usable,
            inliers=self.inliers,
            residual=self.residual,
            quality=self.quality,
            recommended=self.recommended if recommended is None else recommended,
            gains=values,
            model_kind=self.model_kind,
            fallback_reason=self.fallback_reason
            if fallback_reason is None
            else fallback_reason,
            raw_jitter=self.raw_jitter if raw_jitter is None else raw_jitter,
            stable_jitter=self.stable_jitter if stable_jitter is None else stable_jitter,
        )

    def summary(self) -> str:
        n = len(self.dx) or len(self.transforms)
        if n <= 0:
            return "无有效估计"
        if self.gains:
            frac = sum(1 for g in self.gains if g > 0.5) / max(len(self.gains), 1)
        elif self.usable:
            frac = sum(self.usable) / max(len(self.usable), 1)
        else:
            frac = 1.0
        if self.residual and self.usable:
            residuals = [res for res, ok in zip(self.residual, self.usable) if ok]
        else:
            residuals = list(self.residual)
        med = float(np.median(residuals)) if residuals else 0.0
        kind = "平移" if self.model_kind == "translation" else "相似"
        extras: list[str] = []
        if self.fallback_reason:
            extras.append(self.fallback_reason)
        if self.raw_jitter or self.stable_jitter:
            extras.append(f"抖动 {self.raw_jitter:.2f}→{self.stable_jitter:.2f}")
        extra = ("，" + "，".join(extras)) if extras else ""
        return f"{kind}，应用 {frac:.0%}，残差 {med:.2f} px，{n} 帧{extra}"


def _gray(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _energy(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return gx * gx + gy * gy


def _down_factor(height: int, width: int) -> int:
    longest = max(height, width)
    if longest <= ANALYZE_MAX:
        return 1
    return max(2, int(round(longest / ANALYZE_MAX)))


def _down(gray: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return gray
    height, width = gray.shape
    nh, nw = height // scale, width // scale
    if nh < 1 or nw < 1:
        return gray
    cropped = gray[: nh * scale, : nw * scale]
    return cropped.reshape(nh, scale, nw, scale).mean(axis=(1, 3))


def _scale_points(
    points: Sequence[tuple[float, float]] | None, factor: float
) -> list[tuple[float, float]]:
    if not points:
        return []
    return [(x * factor, y * factor) for x, y in points]


def _corner_rois(width: int, height: int) -> list[tuple[int, int, int, int]]:
    mx = max(24, int(width * CORNER_FRAC))
    my = max(24, int(height * CORNER_FRAC))
    return [
        (0, 0, mx, my),
        (width - mx, 0, width, my),
        (0, height - my, mx, height),
        (width - mx, height - my, width, height),
    ]


def _edge_rois(width: int, height: int) -> list[tuple[int, int, int, int]]:
    xs = [0, width // 3, 2 * width // 3, width]
    ys = [0, height // 3, 2 * height // 3, height]
    rois: list[tuple[int, int, int, int]] = []
    for row in range(3):
        for col in range(3):
            if row == 1 and col == 1:
                continue
            rois.append((xs[col], ys[row], xs[col + 1], ys[row + 1]))
    return rois


def _far_enough(picked: list[tuple[float, float]], x: float, y: float, spacing: float) -> bool:
    thresh = spacing * spacing
    return all((x - px) ** 2 + (y - py) ** 2 >= thresh for px, py in picked)


def _excluded(
    x: float, y: float, exclude: Sequence[tuple[float, float]], radius: float
) -> bool:
    if not exclude:
        return False
    thresh = radius * radius
    return any((x - ex) ** 2 + (y - ey) ** 2 <= thresh for ex, ey in exclude)


def _peaks_in_roi(
    energy: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    pad: int,
    per_roi: int,
    picked: list[tuple[float, float]],
    exclude: Sequence[tuple[float, float]],
) -> None:
    roi = energy[y0:y1, x0:x1]
    if roi.size == 0:
        return
    peak = float(roi.max())
    if peak < 1e-6:
        return
    height, width = energy.shape
    rh, rw = roi.shape
    cell_h = max(1, rh // GRID)
    cell_w = max(1, rw // GRID)
    cells: list[tuple[float, int, int]] = []
    for row in range(GRID):
        for col in range(GRID):
            yy0, xx0 = row * cell_h, col * cell_w
            patch = roi[yy0 : yy0 + cell_h, xx0 : xx0 + cell_w]
            if patch.size == 0:
                continue
            local = int(patch.argmax())
            ly, lx = np.unravel_index(local, patch.shape)
            score = float(patch.flat[local])
            if score < 0.2 * peak:
                continue
            x = x0 + xx0 + int(lx)
            y = y0 + yy0 + int(ly)
            if pad <= x < width - pad and pad <= y < height - pad:
                cells.append((score, x, y))
    cells.sort(reverse=True)
    added = 0
    for _score, x, y in cells:
        if _excluded(float(x), float(y), exclude, EXCLUDE_RADIUS):
            continue
        if _far_enough(picked, float(x), float(y), MIN_SPACING):
            picked.append((float(x), float(y)))
            added += 1
        if added >= per_roi or len(picked) >= MAX_ANCHORS:
            break


def detect_anchor_points(
    frame: np.ndarray,
    *,
    per_corner: int = POINTS_PER_CORNER,
    exclude: Sequence[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """High-texture peaks: four corners first, then remaining edge tiles."""
    gray = frame if frame.ndim == 2 else _gray(frame)
    energy = _energy(gray)
    height, width = gray.shape
    pad = PATCH // 2 + 1
    picked: list[tuple[float, float]] = []
    blocked = list(exclude or ())
    for x0, y0, x1, y1 in _corner_rois(width, height):
        _peaks_in_roi(
            energy, x0, y0, x1, y1, pad=pad, per_roi=per_corner, picked=picked, exclude=blocked
        )
        if len(picked) >= MAX_ANCHORS:
            return picked
    extra = max(1, per_corner)
    for x0, y0, x1, y1 in _edge_rois(width, height):
        _peaks_in_roi(
            energy, x0, y0, x1, y1, pad=pad, per_roi=extra, picked=picked, exclude=blocked
        )
        if len(picked) >= MAX_ANCHORS:
            break
    return picked


def _extract(gray: np.ndarray, x: float, y: float) -> np.ndarray | None:
    r = PATCH // 2
    xi, yi = int(round(x)), int(round(y))
    if yi - r < 0 or xi - r < 0 or yi + r >= gray.shape[0] or xi + r >= gray.shape[1]:
        return None
    patch = gray[yi - r : yi + r + 1, xi - r : xi + r + 1].copy()
    if float(patch.std()) < 8.0:
        return None
    return patch


def _parabolic_peak(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-9:
        return 0.0
    offset = 0.5 * (left - right) / denom
    return max(-1.0, min(1.0, offset))


def _ncc_at(
    gray: np.ndarray, template: np.ndarray, x: float, y: float, search: int
) -> tuple[float, float, float]:
    th, tw = template.shape
    r_y, r_x = th // 2, tw // 2
    xi, yi = int(round(x)), int(round(y))
    height, width = gray.shape
    y0 = max(0, yi - search - r_y)
    y1 = min(height, yi + search + r_y + 1)
    x0 = max(0, xi - search - r_x)
    x1 = min(width, xi + search + r_x + 1)
    region = gray[y0:y1, x0:x1]
    if region.shape[0] < th or region.shape[1] < tw:
        return x, y, -1.0
    windows = np.lib.stride_tricks.sliding_window_view(region, (th, tw))
    t = template.astype(np.float32, copy=False)
    t = t - t.mean()
    t_norm = float(np.linalg.norm(t))
    if t_norm < 1e-6:
        return x, y, -1.0
    stacked = windows.astype(np.float32, copy=False)
    stacked = stacked - stacked.mean(axis=(2, 3), keepdims=True)
    denom = t_norm * np.linalg.norm(stacked, axis=(2, 3))
    ncc = np.sum(stacked * t, axis=(2, 3)) / np.maximum(denom, 1e-12)
    ncc = np.where(denom < 1e-6, -1.0, ncc)
    loc = np.unravel_index(int(ncc.argmax()), ncc.shape)
    row, col = int(loc[0]), int(loc[1])
    score = float(ncc[row, col])
    dx = dy = 0.0
    if 0 < col < ncc.shape[1] - 1:
        dx = _parabolic_peak(float(ncc[row, col - 1]), score, float(ncc[row, col + 1]))
    if 0 < row < ncc.shape[0] - 1:
        dy = _parabolic_peak(float(ncc[row - 1, col]), score, float(ncc[row + 1, col]))
    bx = x0 + col + r_x + dx
    by = y0 + row + r_y + dy
    return float(bx), float(by), score


def _track_template(
    gray: np.ndarray, template: np.ndarray, x: float, y: float
) -> tuple[float, float, float]:
    bx, by, score = _ncc_at(gray, template, x, y, NEAR_SEARCH)
    if score >= NCC_MIN:
        return bx, by, score
    bx, by, score = _ncc_at(gray, template, x, y, MID_SEARCH)
    if score >= NCC_MIN:
        return bx, by, score
    scale = COARSE_SCALE
    coarse = _down(gray, scale)
    tmpl = _down(template, scale)
    if tmpl.shape[0] >= 3 and tmpl.shape[1] >= 3:
        sx, sy, _score = _ncc_at(coarse, tmpl, x / scale, y / scale, COARSE_SEARCH)
        x, y = sx * scale, sy * scale
    return _ncc_at(gray, template, x, y, FINE_SEARCH)


def _match_bidirectional(
    gray_ref: np.ndarray,
    gray_cur: np.ndarray,
    seed: tuple[float, float],
    template: np.ndarray,
    pred: tuple[float, float],
) -> tuple[float, float, float] | None:
    fx, fy, fscore = _track_template(gray_cur, template, pred[0], pred[1])
    if fscore < NCC_MIN:
        return None
    cur_tmpl = _extract(gray_cur, fx, fy)
    if cur_tmpl is None:
        return None
    bx, by, bscore = _track_template(gray_ref, cur_tmpl, seed[0], seed[1])
    if bscore < NCC_MIN:
        return None
    if (bx - seed[0]) ** 2 + (by - seed[1]) ** 2 > FB_THRESH * FB_THRESH:
        return None
    return fx, fy, fscore


def _similarity_from_two(
    s0: tuple[float, float],
    s1: tuple[float, float],
    d0: tuple[float, float],
    d1: tuple[float, float],
) -> Similarity | None:
    vsx, vsy = s1[0] - s0[0], s1[1] - s0[1]
    vdx, vdy = d1[0] - d0[0], d1[1] - d0[1]
    src_len2 = vsx * vsx + vsy * vsy
    if src_len2 < MIN_BASELINE * MIN_BASELINE:
        return None
    a = (vsx * vdx + vsy * vdy) / src_len2
    b = (vsx * vdy - vsy * vdx) / src_len2
    tx = d0[0] - (a * s0[0] - b * s0[1])
    ty = d0[1] - (b * s0[0] + a * s0[1])
    return Similarity(a, b, tx, ty)


def _similarity_wls(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray
) -> Similarity | None:
    n = src.shape[0]
    if n < 2:
        return None
    design = np.zeros((2 * n, 4), dtype=np.float64)
    rhs = np.zeros(2 * n, dtype=np.float64)
    scale = np.sqrt(np.maximum(weights, 1e-6))
    for i, ((x, y), (xp, yp), w) in enumerate(zip(src, dst, scale)):
        design[2 * i] = (x * w, -y * w, w, 0.0)
        design[2 * i + 1] = (y * w, x * w, 0.0, w)
        rhs[2 * i] = xp * w
        rhs[2 * i + 1] = yp * w
    try:
        params, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return Similarity(float(params[0]), float(params[1]), float(params[2]), float(params[3]))


def _quadrant_count(pts: np.ndarray, cx: float, cy: float) -> int:
    flags = [False, False, False, False]
    for x, y in pts:
        index = (0 if x < cx else 1) + (0 if y < cy else 2)
        flags[index] = True
    return sum(flags)


def _ransac_similarity(
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
    scores: list[float],
    width: int,
    height: int,
) -> tuple[Similarity | None, np.ndarray, float]:
    n = len(src_pts)
    empty = np.zeros(n, dtype=bool)
    if n < 3:
        return None, empty, 0.0
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    weights = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(0)
    best_mask = empty
    best_model: Similarity | None = None
    best_med = float("inf")
    iters = min(64, 10 * n)
    cx, cy = width * 0.5, height * 0.5
    min_inliers = 4 if n >= 6 else 3
    min_quads = 2
    for _ in range(iters):
        i, j = (int(v) for v in rng.choice(n, 2, replace=False))
        model = _similarity_from_two(
            (float(src[i, 0]), float(src[i, 1])),
            (float(src[j, 0]), float(src[j, 1])),
            (float(dst[i, 0]), float(dst[i, 1])),
            (float(dst[j, 0]), float(dst[j, 1])),
        )
        if model is None:
            continue
        if not (0.80 <= model.scale <= 1.20):
            continue
        if abs(model.theta) > math.radians(12):
            continue
        residuals = np.array(
            [model.residual(src[k, 0], src[k, 1], dst[k, 0], dst[k, 1]) for k in range(n)]
        )
        mask = residuals <= INLIER_PX
        count = int(mask.sum())
        if count < min_inliers:
            continue
        if _quadrant_count(src[mask], cx, cy) < min_quads:
            continue
        med = float(np.median(residuals[mask]))
        if count > int(best_mask.sum()) or (count == int(best_mask.sum()) and med < best_med):
            best_mask = mask
            best_model = model
            best_med = med
    if best_model is None or not np.any(best_mask):
        return None, empty, 0.0
    refined = _similarity_wls(src[best_mask], dst[best_mask], weights[best_mask])
    model = refined or best_model
    residuals = np.array(
        [model.residual(src[k, 0], src[k, 1], dst[k, 0], dst[k, 1]) for k in range(n)]
    )
    mask = residuals <= INLIER_PX
    if int(mask.sum()) < min_inliers or _quadrant_count(src[mask], cx, cy) < min_quads:
        return None, empty, 0.0
    if not (0.80 <= model.scale <= 1.20):
        return None, empty, 0.0
    med = float(np.median(residuals[mask])) if np.any(mask) else 0.0
    return model, mask, med


def _robust_translation(
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
    prev: Similarity | None = None,
) -> tuple[Similarity | None, int, float]:
    if not src_pts:
        return None, 0, 0.0
    pairs = [(d[0] - s[0], d[1] - s[1]) for s, d in zip(src_pts, dst_pts)]
    if prev is not None and math.hypot(prev.tx, prev.ty) > 1.0 and len(pairs) >= 2:
        pred = (prev.tx, prev.ty)
        slack = max(24.0, 0.8 * math.hypot(prev.tx, prev.ty) + 24.0)
        close = [
            p
            for p in pairs
            if math.hypot(p[0] - pred[0], p[1] - pred[1]) <= slack
        ]
        if close:
            pairs = close
    if len(pairs) == 1:
        tx, ty = pairs[0]
        return Similarity(1.0, 0.0, tx, ty), 1, 0.0
    thresh = TRANS_INLIER_PX * TRANS_INLIER_PX
    best: list[tuple[float, float]] = []
    for cx, cy in pairs:
        group = [p for p in pairs if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= thresh]
        if len(group) > len(best):
            best = group
        elif prev is not None and len(group) == len(best) and group:
            def _dist(group: list[tuple[float, float]]) -> float:
                mx = float(np.median([p[0] for p in group]))
                my = float(np.median([p[1] for p in group]))
                return math.hypot(mx - prev.tx, my - prev.ty)

            if best and _dist(group) < _dist(best):
                best = group
    if len(pairs) >= 3 and len(best) < max(2, (len(pairs) + 2) // 3):
        return None, 0, 0.0
    if not best:
        return None, 0, 0.0
    tx = float(np.median([p[0] for p in best]))
    ty = float(np.median([p[1] for p in best]))
    residuals = [math.hypot(p[0] - tx, p[1] - ty) for p in pairs]
    inliers = sum(1 for r in residuals if r <= TRANS_INLIER_PX)
    good = [r for r in residuals if r <= TRANS_INLIER_PX]
    med = float(np.median(good)) if good else float(np.median(residuals))
    return Similarity(1.0, 0.0, tx, ty), max(inliers, 1), med


def _choose_model(
    src: list[tuple[float, float]],
    dst: list[tuple[float, float]],
    scores: list[float],
    width: int,
    height: int,
    prev: Similarity,
) -> tuple[Similarity | None, int, float, bool]:
    trans, n_in, med_t = _robust_translation(src, dst, prev)
    sim, mask, med_s = _ransac_similarity(src, dst, scores, width, height)
    if trans is None and sim is None:
        if prev.is_similarity():
            return prev, 0, 0.0, True
        return None, 0, 0.0, False
    if sim is None or trans is None:
        if sim is None and prev.is_similarity():
            return prev, n_in, med_t, True
        model = sim or trans
        n_use = n_in if trans is not None and model is trans else int(mask.sum())
        med = med_s if sim is not None else med_t
        return model, n_use, med, bool(model is not None and model.is_similarity())
    stick = prev.is_similarity() and abs(sim.theta - prev.theta) < math.radians(8)
    upgrade = (
        med_s <= SIM_IMPROVE * max(med_t, 0.05)
        and (abs(sim.theta) >= THETA_UPGRADE or abs(sim.scale - 1.0) >= SCALE_UPGRADE)
        and abs(sim.theta - prev.theta) < math.radians(8)
        and abs(sim.scale - prev.scale) < 0.10
    )
    if upgrade or stick:
        return sim, int(mask.sum()), med_s, True
    return trans, n_in, med_t, False


def _lerp_similarity(a: Similarity, b: Similarity, t: float) -> Similarity:
    t = max(0.0, min(1.0, t))
    return Similarity(
        a.a + (b.a - a.a) * t,
        a.b + (b.b - a.b) * t,
        a.tx + (b.tx - a.tx) * t,
        a.ty + (b.ty - a.ty) * t,
    )


def _median3(values: list[float], index: int) -> float:
    lo = max(0, index - 1)
    hi = min(len(values) - 1, index + 1)
    window = sorted(values[lo : hi + 1])
    return window[len(window) // 2]


def _postprocess_models(
    models: list[Similarity],
    usable: list[bool],
    residuals: list[float],
    inliers: list[int],
) -> tuple[list[Similarity], list[bool]]:
    n = len(models)
    if n == 0:
        return models, list(usable)
    flagged = list(usable)
    for i in range(1, n):
        if not flagged[i] or not flagged[i - 1]:
            continue
        prev, cur = models[i - 1], models[i]
        jump = math.hypot(cur.tx - prev.tx, cur.ty - prev.ty)
        rot = abs(cur.theta - prev.theta)
        scale = abs(cur.scale - prev.scale)
        sigma = max(residuals[i], residuals[i - 1], NOISE_PX)
        prev_jump = 0.0
        if i >= 2 and flagged[i - 2]:
            older = models[i - 2]
            prev_jump = math.hypot(prev.tx - older.tx, prev.ty - older.ty)
        unusual = prev_jump > 0.5 and jump > 2.5 * max(prev_jump, sigma, 1.0)
        if unusual and jump > max(SPIKE_PX, 4.0 * sigma):
            flagged[i] = False
        elif rot > math.radians(8) or scale > 0.08:
            flagged[i] = False
    for i in range(1, n - 1):
        if not flagged[i]:
            continue
        prev, nxt = models[i - 1], models[i + 1]
        cur = models[i]
        jump = math.hypot(cur.tx - prev.tx, cur.ty - prev.ty)
        neighbor = math.hypot(nxt.tx - prev.tx, nxt.ty - prev.ty)
        jump_next = math.hypot(cur.tx - nxt.tx, cur.ty - nxt.ty)
        if jump > SPIKE_PX and jump_next > SPIKE_PX and neighbor < SPIKE_PX * 0.5:
            flagged[i] = False
    filled = list(models)
    filled_ok = list(flagged)
    i = 0
    while i < n:
        if filled_ok[i]:
            i += 1
            continue
        j = i
        while j < n and not filled_ok[j]:
            j += 1
        gap = j - i
        left = i - 1
        if gap <= MAX_GAP and left >= 0 and j < n:
            for k in range(i, j):
                t = (k - left) / (j - left)
                filled[k] = _lerp_similarity(models[left], models[j], t)
                filled_ok[k] = True
        elif gap <= MAX_GAP and left >= 0:
            for k in range(i, j):
                filled[k] = models[left]
                filled_ok[k] = True
        elif gap <= MAX_GAP and j < n:
            for k in range(i, j):
                filled[k] = models[j]
                filled_ok[k] = True
        else:
            for k in range(i, j):
                filled[k] = IDENTITY
                filled_ok[k] = False
                residuals[k] = 0.0
                inliers[k] = 0
        i = j
    txs = [m.tx for m in filled]
    tys = [m.ty for m in filled]
    for i in range(n):
        if n < 3 or not filled_ok[i]:
            continue
        med_x = _median3(txs, i)
        med_y = _median3(tys, i)
        if abs(filled[i].tx - med_x) < NOISE_PX and abs(filled[i].ty - med_y) < NOISE_PX:
            filled[i] = Similarity(
                filled[i].a,
                filled[i].b,
                0.5 * filled[i].tx + 0.5 * med_x,
                0.5 * filled[i].ty + 0.5 * med_y,
            )
    observed_models = [m for m, ok in zip(models, flagged) if ok]
    if observed_models and sum(flagged) / max(n, 1) >= 0.55:
        amp = max((math.hypot(m.tx, m.ty) for m in observed_models), default=0.0)
        rot = max((abs(m.theta) for m in observed_models), default=0.0)
        scale_dev = max((abs(m.scale - 1.0) for m in observed_models), default=0.0)
        if amp < DEADBAND_PX and rot < math.radians(DEADBAND_DEG) and scale_dev < DEADBAND_SCALE:
            filled = [IDENTITY for _ in filled]
            filled_ok = [True for _ in filled]
            for i in range(n):
                residuals[i] = 0.0
            return filled, filled_ok
    return filled, filled_ok


def _quality(
    usable: list[bool], residuals: list[float], inliers: list[int]
) -> tuple[float, bool]:
    if not usable:
        return 1.0, True
    frac = sum(usable) / max(len(usable), 1)
    good_res = [res for value, res in zip(usable, residuals) if value]
    med = float(np.median(good_res)) if good_res else 0.0
    mean_in = float(np.mean(inliers)) if inliers else 0.0
    score = frac * (1.0 / (1.0 + med)) * min(1.0, 0.25 * mean_in + 0.25)
    recommended = frac >= 0.55 and med <= 2.5
    return float(score), recommended


def _empty_shake(n: int, n_anchors: int = 0, reason: str = "") -> ShakeCompensation:
    zeros = tuple(0.0 for _ in range(max(n, 0)))
    empty = tuple(tuple(None for _ in range(n_anchors)) for _ in range(max(n, 0)))
    identity = tuple(IDENTITY for _ in range(max(n, 0)))
    ok = tuple(False for _ in range(max(n, 0)))
    return ShakeCompensation(
        dx=zeros,
        dy=zeros,
        anchors=empty,
        transforms=identity,
        usable=ok,
        inliers=tuple(0 for _ in range(max(n, 0))),
        residual=tuple(0.0 for _ in range(max(n, 0))),
        quality=0.0,
        recommended=False,
        gains=zeros,
        fallback_reason=reason or "锚点不足",
    )


def _detrend(values: np.ndarray, degree: int) -> np.ndarray:
    n = len(values)
    if n < 2:
        return values - np.mean(values)
    degree = max(0, min(int(degree), n - 1))
    t = np.arange(n, dtype=np.float64)
    design = np.column_stack([t ** k for k in range(degree + 1)])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def _jitter_metrics(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0
    degree = 2 if n >= 6 else 1
    rx = _detrend(np.asarray(xs, dtype=np.float64), degree)
    ry = _detrend(np.asarray(ys, dtype=np.float64), degree)
    step = np.hypot(np.diff(rx), np.diff(ry))
    hf = float(np.mean(np.abs(np.diff(step)))) if len(step) > 1 else 0.0
    p95 = float(np.percentile(step, 95))
    jerk = float(np.percentile(np.abs(np.diff(step)), 95)) if len(step) > 1 else 0.0
    return hf, p95, jerk


def _not_worse(
    compensated: tuple[float, float, float], raw: tuple[float, float, float]
) -> bool:
    return (
        compensated[0] <= raw[0] * 1.05 + 0.45
        and compensated[1] <= raw[1] * 1.05 + 0.45
        and compensated[2] <= raw[2] * 1.05 + 0.45
    )


def gate_compensation(
    shake: ShakeCompensation, result: TrackResult | None
) -> ShakeCompensation:
    """Zero per-run gain when compensation would add jitter to a track."""
    n = len(shake.dx) if shake.dx else len(shake.transforms)
    if n <= 0 or result is None or not result.points:
        return shake
    points = sorted(result.points, key=lambda p: p.frame)
    if shake.gains:
        gains = list(shake.gains)
    elif shake.usable:
        gains = [1.0 if ok else 0.0 for ok in shake.usable]
        if len(gains) < n:
            gains.extend([0.0] * (n - len(gains)))
    else:
        gains = [1.0 for _ in range(n)]
    visible = [p for p in points if p.visible]
    runs: list[list[TrackPoint]] = []
    current: list[TrackPoint] = []
    last_frame = None
    for point in visible:
        if last_frame is not None and point.frame > last_frame + 1:
            if current:
                runs.append(current)
            current = []
        current.append(point)
        last_frame = point.frame
    if current:
        runs.append(current)
    raw_hf = stable_hf = 0.0
    evaluated = False
    kept_any = False
    for run in runs:
        if len(run) < 3:
            continue
        evaluated = True
        raw_x = [p.x for p in run]
        raw_y = [p.y for p in run]
        cand_x = []
        cand_y = []
        for point in run:
            model = shake.transform_at(point.frame)
            use = shake.is_usable(point.frame) or not shake.usable
            if use:
                cx, cy = model.inverse().apply(point.x, point.y)
            else:
                cx, cy = point.x, point.y
            cand_x.append(cx)
            cand_y.append(cy)
        raw_m = _jitter_metrics(raw_x, raw_y)
        cand_m = _jitter_metrics(cand_x, cand_y)
        raw_hf = max(raw_hf, raw_m[0])
        stable_hf = max(stable_hf, cand_m[0])
        if _not_worse(cand_m, raw_m):
            kept_any = True
            for point in run:
                index = shake._index(point.frame)
                if 0 <= index < n and (shake.is_usable(point.frame) or not shake.usable):
                    gains[index] = 1.0
        else:
            for point in run:
                index = shake._index(point.frame)
                if 0 <= index < n:
                    gains[index] = 0.0
    if not runs:
        applied = sum(gains) / max(n, 1)
        recommended = bool(shake.recommended and applied >= 0.55)
        return shake.with_gains(
            gains,
            recommended=recommended,
            fallback_reason="" if recommended else "无足够轨迹评估抖动",
        )
    applied = sum(1 for g in gains if g > 0.5) / max(n, 1)
    if evaluated and not kept_any:
        gains = [0.0 for _ in gains]
        applied = 0.0
        reason = "补偿会增加轨迹抖动，已保持原坐标"
        recommended = False
    elif not evaluated:
        reason = ""
        recommended = bool(shake.recommended)
    else:
        reason = ""
        recommended = applied >= 0.55 and shake.recommended
        if not recommended and applied < 0.55:
            reason = "有效补偿帧过少"
    return shake.with_gains(
        gains,
        recommended=recommended,
        fallback_reason=reason,
        raw_jitter=raw_hf,
        stable_jitter=stable_hf,
    )


def estimate_shake_from_frames(
    frames: Iterable[np.ndarray],
    *,
    cancel: CancelToken | None = None,
    progress: ProgressCb | None = None,
    clip_id: str = "shake",
    total: int | None = None,
    exclude: Sequence[tuple[float, float]] | None = None,
    exclude_by_frame: dict[int, Sequence[tuple[float, float]]] | None = None,
) -> ShakeCompensation:
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        return ShakeCompensation(dx=(), dy=(), anchors=())
    height, width = first.shape[0], first.shape[1]
    factor = _down_factor(height, width)
    inv = float(factor)
    gray0_full = _gray(first)
    gray0 = _down(gray0_full, factor)
    ah, aw = gray0.shape
    first_exclude = list(exclude or ())
    if exclude_by_frame and 0 in exclude_by_frame:
        first_exclude.extend(exclude_by_frame[0])
    seeds = detect_anchor_points(
        gray0,
        exclude=_scale_points(first_exclude, 1.0 / inv),
    )
    templates = [_extract(gray0, x, y) for x, y in seeds]
    valid = [
        (seed, tmpl)
        for seed, tmpl in zip(seeds, templates)
        if tmpl is not None
    ]
    rest = list(iterator)
    n = 1 + len(rest)
    if len(valid) < 2:
        return _empty_shake(n, len(seeds), "背景锚点不足")

    seeds = [item[0] for item in valid]
    templates = [item[1] for item in valid]
    orig_seeds = [(x * inv, y * inv) for x, y in seeds]
    tracks: list[list[tuple[float, float] | None]] = [list(orig_seeds)]
    models: list[Similarity] = [IDENTITY]
    usable = [True]
    residuals = [0.0]
    inlier_counts = [len(seeds)]
    kinds = [False]
    expected = total or n
    if progress is not None:
        progress(
            ProgressEvent(
                clip_id=clip_id, current=1, total=max(expected, 1), stage="shake"
            )
        )

    prev = IDENTITY
    for index, frame in enumerate(rest, start=1):
        if cancel is not None and cancel.cancelled:
            break
        gray = _down(_gray(frame), factor)
        raw: list[tuple[float, float] | None] = []
        scores: list[float] = []
        frame_exclude = []
        if exclude_by_frame and index in exclude_by_frame:
            frame_exclude = _scale_points(exclude_by_frame[index], 1.0 / inv)
        relock = index % RELOCK_EVERY == 0
        for k, seed in enumerate(seeds):
            pred = prev.apply(seed[0], seed[1]) if not relock else seed
            matched = _match_bidirectional(gray0, gray, seed, templates[k], pred)
            if matched is None and not relock:
                matched = _match_bidirectional(gray0, gray, seed, templates[k], seed)
            if matched is None or _excluded(matched[0], matched[1], frame_exclude, EXCLUDE_RADIUS):
                raw.append(None)
                scores.append(0.0)
            else:
                raw.append((matched[0], matched[1]))
                scores.append(matched[2])
        src = [seeds[k] for k in range(len(seeds)) if raw[k] is not None]
        dst = [raw[k] for k in range(len(seeds)) if raw[k] is not None]
        sc = [scores[k] for k in range(len(seeds)) if raw[k] is not None]
        model, n_in, med, is_sim = _choose_model(src, dst, sc, aw, ah, prev)
        ok = model is not None
        if model is None:
            model = IDENTITY
            med = 0.0
            n_in = 0
            ok = False
        nxt: list[tuple[float, float] | None] = []
        for k in range(len(seeds)):
            if raw[k] is None:
                nxt.append(None)
            else:
                nxt.append((raw[k][0] * inv, raw[k][1] * inv))
        tracks.append(nxt)
        models.append(model.scaled(inv) if ok else IDENTITY)
        usable.append(ok)
        residuals.append(med * inv)
        inlier_counts.append(n_in)
        kinds.append(is_sim and ok)
        prev = model if ok else prev
        if progress is not None:
            progress(
                ProgressEvent(
                    clip_id=clip_id,
                    current=index + 1,
                    total=max(expected, index + 1),
                    stage="shake",
                )
            )

    models, observed = _postprocess_models(models, usable, residuals, inlier_counts)
    quality, recommended = _quality(observed, residuals, inlier_counts)
    dx_list = tuple(m.tx for m in models)
    dy_list = tuple(m.ty for m in models)
    packed = tuple(tuple(row) for row in tracks)
    sim_frac = sum(kinds) / max(len(kinds), 1)
    kind = "similarity" if sim_frac >= 0.4 else "translation"
    gains = tuple(1.0 if ok else 0.0 for ok in observed)
    return ShakeCompensation(
        dx=dx_list,
        dy=dy_list,
        anchors=packed,
        transforms=tuple(models),
        usable=tuple(observed),
        inliers=tuple(inlier_counts),
        residual=tuple(residuals),
        quality=quality,
        recommended=recommended,
        gains=gains,
        model_kind=kind,
        fallback_reason="" if recommended else "匹配质量不足",
    )


def _pad_shake(shake: ShakeCompensation, before: int, after: int) -> ShakeCompensation:
    before = max(0, int(before))
    after = max(0, int(after))
    if before == 0 and after == 0:
        return shake
    n_anchors = len(shake.anchors[0]) if shake.anchors else 0
    empty_row = tuple(None for _ in range(n_anchors))
    zeros_b = tuple(0.0 for _ in range(before))
    zeros_a = tuple(0.0 for _ in range(after))
    ident_b = tuple(IDENTITY for _ in range(before))
    ident_a = tuple(IDENTITY for _ in range(after))
    false_b = tuple(False for _ in range(before))
    false_a = tuple(False for _ in range(after))
    zero_i_b = tuple(0 for _ in range(before))
    zero_i_a = tuple(0 for _ in range(after))
    if shake.transforms:
        transforms = ident_b + shake.transforms + ident_a
    elif shake.dx:
        transforms = (
            ident_b
            + tuple(Similarity(1.0, 0.0, tx, ty) for tx, ty in zip(shake.dx, shake.dy))
            + ident_a
        )
    else:
        transforms = ident_b + ident_a
    if shake.usable:
        usable = false_b + shake.usable + false_a
    else:
        usable = false_b + tuple(True for _ in shake.dx) + false_a
    if shake.inliers:
        inliers = zero_i_b + shake.inliers + zero_i_a
    else:
        inliers = zero_i_b + tuple(0 for _ in shake.dx) + zero_i_a
    if shake.residual:
        residual = zeros_b + shake.residual + zeros_a
    else:
        residual = zeros_b + tuple(0.0 for _ in shake.dx) + zeros_a
    if shake.anchors:
        anchors = (
            tuple(empty_row for _ in range(before))
            + shake.anchors
            + tuple(empty_row for _ in range(after))
        )
    else:
        anchors = tuple(empty_row for _ in range(before + after))
    gains = shake.gains
    if not gains:
        n = len(shake.dx) or len(shake.transforms)
        gains = tuple(1.0 if shake.is_usable(i) else 0.0 for i in range(n))
    return ShakeCompensation(
        dx=zeros_b + shake.dx + zeros_a,
        dy=zeros_b + shake.dy + zeros_a,
        anchors=anchors,
        model_name=shake.model_name,
        model_version=shake.model_version,
        transforms=transforms,
        usable=usable,
        inliers=inliers,
        residual=residual,
        quality=shake.quality,
        recommended=shake.recommended,
        gains=zeros_b + gains + zeros_a,
        model_kind=shake.model_kind,
        fallback_reason=shake.fallback_reason,
        raw_jitter=shake.raw_jitter,
        stable_jitter=shake.stable_jitter,
    )


def estimate_shake(
    info: VideoInfo,
    *,
    cancel: CancelToken | None = None,
    progress: ProgressCb | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    exclude_by_frame: dict[int, Sequence[tuple[float, float]]] | None = None,
) -> ShakeCompensation:
    decoder = FrameDecoder(info)
    try:
        last = info.frame_count - 1 if end_frame is None else min(end_frame, info.frame_count - 1)
        first = max(0, min(int(start_frame), last))
        shifted: dict[int, Sequence[tuple[float, float]]] | None = None
        if exclude_by_frame:
            shifted = {
                frame - first: pts
                for frame, pts in exclude_by_frame.items()
                if first <= frame <= last
            }

        def _frames() -> Iterable[np.ndarray]:
            for index in range(first, last + 1):
                yield decoder.frame(index)

        shake = estimate_shake_from_frames(
            _frames(),
            cancel=cancel,
            progress=progress,
            clip_id=info.path.name,
            total=last - first + 1,
            exclude_by_frame=shifted,
        )
        return _pad_shake(shake, first, info.frame_count - last - 1)
    finally:
        decoder.close()


def compensate_result(
    result: TrackResult, shake: ShakeCompensation | None
) -> TrackResult:
    if shake is None or (not shake.dx and not shake.transforms):
        return result
    gated = gate_compensation(shake, result)
    points = []
    for point in result.points:
        sx, sy = gated.to_stable(point.x, point.y, point.frame)
        points.append(
            TrackPoint(
                frame=point.frame,
                x=sx,
                y=sy,
                visible=point.visible,
                confidence=point.confidence,
                manual=point.manual,
                interpolated=point.interpolated,
            )
        )
    return TrackResult(
        clip_id=result.clip_id,
        points=points,
        confidence=result.confidence,
        failure_reason=result.failure_reason,
        model_name=result.model_name,
        model_version=result.model_version,
        elapsed_s=result.elapsed_s,
    )
