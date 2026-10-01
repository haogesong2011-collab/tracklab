"""Pyramid forward-backward Lucas-Kanade on a small point group (NumPy only).

Used as independent evidence for a SAM candidate: corners inside the last
trusted object region are tracked into the current frame and back again.
Points that return to where they started say where the object went; the
fraction of them that lands inside the candidate region says whether SAM's
candidate is that object. The point group never supplies the object centre.

Weak texture, heavy blur or too few consistent points yield *missing*
evidence, which callers must not treat as a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ai.track_guard import MaskStats

MAX_POINTS = 32
LEVELS = 3
HALF_WINDOW = 4
MAX_ITERS = 12
MAX_RESIDUAL = 24.0
MAX_ROI_SIDE = 1024


@dataclass(frozen=True)
class LKEvidence:
    selected: int
    valid: int
    fb_threshold_px: float
    fb_median_px: float | None = None
    inside_fraction: float | None = None
    target_xy: tuple[float, float] | None = None
    target_distance_px: float | None = None
    spread_px: float | None = None
    quality: float | None = None
    missing_reason: str = ""

    @property
    def available(self) -> bool:
        return self.quality is not None

    @property
    def strong(self) -> bool:
        """Enough consistent points to overrule a candidate on its own."""
        return (
            self.quality is not None
            and self.valid >= 8
            and self.valid >= 0.6 * max(self.selected, 1)
        )

    def as_diagnostics(self) -> dict:
        out = {
            "lk_selected": self.selected,
            "lk_valid": self.valid,
            "lk_fb_threshold_px": round(self.fb_threshold_px, 3),
        }
        if self.fb_median_px is not None:
            out["lk_fb_median_px"] = round(self.fb_median_px, 3)
        if self.inside_fraction is not None:
            out["lk_inside_fraction"] = round(self.inside_fraction, 4)
        if self.target_distance_px is not None:
            out["lk_target_distance_px"] = round(self.target_distance_px, 3)
        if self.spread_px is not None:
            out["lk_spread_px"] = round(self.spread_px, 3)
        if self.missing_reason:
            out["lk_missing"] = self.missing_reason
        return out


def to_gray(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _blur_half(img: np.ndarray) -> np.ndarray:
    """1-2-1 separable blur then 2x decimation."""
    p = np.pad(img, 1, mode="edge")
    h = 0.25 * p[:, :-2] + 0.5 * p[:, 1:-1] + 0.25 * p[:, 2:]
    v = 0.25 * h[:-2, :] + 0.5 * h[1:-1, :] + 0.25 * h[2:, :]
    return np.ascontiguousarray(v[::2, ::2])


def build_pyramid(img: np.ndarray, levels: int = LEVELS) -> list[np.ndarray]:
    pyr = [np.asarray(img, dtype=np.float32)]
    for _ in range(1, levels):
        if min(pyr[-1].shape) < 2 * (2 * HALF_WINDOW + 1):
            break
        pyr.append(_blur_half(pyr[-1]))
    return pyr


def bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample `img` at float coordinates. Returns (values, in_bounds)."""
    h, w = img.shape
    inside = (xs >= 0) & (ys >= 0) & (xs <= w - 1) & (ys <= h - 1)
    x = np.clip(xs, 0, w - 1.000001)
    y = np.clip(ys, 0, h - 1.000001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy, inside


def _box_sum(img: np.ndarray, r: int) -> np.ndarray:
    p = np.pad(img, ((r + 1, r), (r + 1, r)), mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]).astype(np.float32)


def good_features(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    max_points: int = MAX_POINTS,
    min_distance: float = 2.0,
    block: int = 2,
    abs_floor: float = 150.0,
) -> np.ndarray:
    """Shi-Tomasi corners inside `mask` (same shape as img). Returns (N, 2) xy."""
    gy, gx = np.gradient(img.astype(np.float32))
    a = _box_sum(gx * gx, block)
    b = _box_sum(gx * gy, block)
    c = _box_sum(gy * gy, block)
    score = 0.5 * (a + c) - np.sqrt(0.25 * (a - c) ** 2 + b * b)
    score = np.where(mask, score, 0.0)
    peak = float(score.max()) if score.size else 0.0
    if peak <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    floor = max(abs_floor, 0.05 * peak)
    ys, xs = np.nonzero(score >= floor)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    order = np.argsort(-score[ys, xs], kind="stable")
    chosen: list[tuple[float, float]] = []
    md2 = min_distance * min_distance
    for i in order[: 4096]:
        x, y = float(xs[i]), float(ys[i])
        if all((x - cx) ** 2 + (y - cy) ** 2 >= md2 for cx, cy in chosen):
            chosen.append((x, y))
            if len(chosen) >= max_points:
                break
    return np.asarray(chosen, dtype=np.float32).reshape(-1, 2)


def lk_track(
    prev_pyr: list[np.ndarray],
    cur_pyr: list[np.ndarray],
    points: np.ndarray,
    init_disp: np.ndarray,
    *,
    half_window: int = HALF_WINDOW,
    iters: int = MAX_ITERS,
    weights: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse-to-fine LK. Returns (new_points, ok, residual).

    `weights` (one image per level, 0..1) restricts each window to pixels of
    the tracked region, so a small object moving over a static background is
    not pulled back to zero motion by the background inside its window.
    """
    n = points.shape[0]
    if n == 0:
        return points.copy(), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
    levels = min(len(prev_pyr), len(cur_pyr))
    off = np.arange(-half_window, half_window + 1, dtype=np.float32)
    oy, ox = np.meshgrid(off, off, indexing="ij")
    ox = ox.ravel()[None, :]
    oy = oy.ravel()[None, :]
    guess = np.asarray(init_disp, dtype=np.float32).reshape(n, 2) / (2 ** (levels - 1))
    ok = np.ones(n, dtype=bool)
    residual = np.zeros(n, dtype=np.float32)
    for level in range(levels - 1, -1, -1):
        scale = float(2**level)
        prev = prev_pyr[level]
        cur = cur_pyr[level]
        gy_img, gx_img = np.gradient(prev)
        px = points[:, 0:1] / scale + ox
        py = points[:, 1:2] / scale + oy
        tmpl, in_prev = bilinear(prev, px, py)
        ix, _ = bilinear(gx_img, px, py)
        iy, _ = bilinear(gy_img, px, py)
        if weights is not None and level < len(weights):
            wgt, _ = bilinear(weights[level], px, py)
            wgt = np.clip(wgt, 0.0, 1.0) + 1e-3
        else:
            wgt = np.ones_like(tmpl)
        ix = ix * np.sqrt(wgt)
        iy = iy * np.sqrt(wgt)
        sw = np.sqrt(wgt)
        gxx = (ix * ix).sum(1)
        gxy = (ix * iy).sum(1)
        gyy = (iy * iy).sum(1)
        det = gxx * gyy - gxy * gxy
        trace = gxx + gyy
        min_eig = 0.5 * trace - np.sqrt(np.maximum(0.25 * trace * trace - det, 0.0))
        good = (np.abs(det) > 1e-6) & (min_eig > 1e-3 * ox.size)
        d = np.zeros((n, 2), dtype=np.float32)
        inv_det = np.where(good, 1.0 / np.where(good, det, 1.0), 0.0)
        for _ in range(iters):
            qx = px + guess[:, 0:1] + d[:, 0:1]
            qy = py + guess[:, 1:2] + d[:, 1:2]
            img, in_cur = bilinear(cur, qx, qy)
            err = (tmpl - img) * sw
            bx = (err * ix).sum(1)
            by = (err * iy).sum(1)
            dx = (gyy * bx - gxy * by) * inv_det
            dy = (gxx * by - gxy * bx) * inv_det
            d[:, 0] += dx
            d[:, 1] += dy
            if float(np.max(np.abs(dx) + np.abs(dy))) < 0.01:
                break
        if level == 0:
            ok &= good & in_prev.all(1) & in_cur.all(1)
            residual = (np.abs(err) * sw).sum(1) / np.maximum(wgt.sum(1), 1e-6)
            residual = residual.astype(np.float32)
            guess = guess + d
        else:
            guess = 2.0 * (guess + d)
    new_points = points + guess
    ok &= np.isfinite(new_points).all(1)
    return new_points.astype(np.float32), ok, residual


def _erode(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, constant_values=False)
    out = p[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= p[1 + dy : p.shape[0] - 1 + dy, 1 + dx : p.shape[1] - 1 + dx]
    return out


def lk_evidence(
    prev_frame: np.ndarray,
    cur_frame: np.ndarray,
    prev_stats: "MaskStats",
    candidate: "MaskStats",
    *,
    init_disp: tuple[float, float] = (0.0, 0.0),
) -> LKEvidence:
    """Forward-backward LK from the trusted region into the current frame."""
    diag = float(np.hypot(prev_stats.w, prev_stats.h))
    threshold = max(1.5, 0.03 * diag)
    if prev_stats.pixels is None:
        return LKEvidence(0, 0, threshold, missing_reason="没有上一帧目标区域")
    h, w = np.asarray(prev_frame).shape[:2]
    margin = HALF_WINDOW * (2**LEVELS) + 8
    ox, oy = prev_stats.origin
    ph, pw = prev_stats.pixels.shape
    dx, dy = float(init_disp[0]), float(init_disp[1])
    xs = [ox, ox + pw, ox + dx, ox + pw + dx, candidate.x - candidate.w / 2, candidate.x + candidate.w / 2]
    ys = [oy, oy + ph, oy + dy, oy + ph + dy, candidate.y - candidate.h / 2, candidate.y + candidate.h / 2]
    x0 = int(max(0, np.floor(min(xs)) - margin))
    y0 = int(max(0, np.floor(min(ys)) - margin))
    x1 = int(min(w, np.ceil(max(xs)) + margin))
    y1 = int(min(h, np.ceil(max(ys)) + margin))
    if x1 - x0 > MAX_ROI_SIDE or y1 - y0 > MAX_ROI_SIDE:
        return LKEvidence(0, 0, threshold, missing_reason="搜索区域过大")
    if x1 - x0 < 2 * HALF_WINDOW + 3 or y1 - y0 < 2 * HALF_WINDOW + 3:
        return LKEvidence(0, 0, threshold, missing_reason="区域过小")
    prev_crop = to_gray(np.asarray(prev_frame)[y0:y1, x0:x1])
    cur_crop = to_gray(np.asarray(cur_frame)[y0:y1, x0:x1])
    region = np.zeros(prev_crop.shape, dtype=bool)
    ry0, rx0 = oy - y0, ox - x0
    sy0, sx0 = max(0, -ry0), max(0, -rx0)
    sub = prev_stats.pixels[sy0:, sx0:]
    ty0, tx0 = max(0, ry0), max(0, rx0)
    sub = sub[: region.shape[0] - ty0, : region.shape[1] - tx0]
    region[ty0 : ty0 + sub.shape[0], tx0 : tx0 + sub.shape[1]] = sub
    # Keep corners off the boundary so background texture is not tracked.
    if min(prev_stats.w, prev_stats.h) >= 6:
        eroded = _erode(region)
        if eroded.sum() >= 4:
            region = eroded
    corners = good_features(prev_crop, region)
    selected = int(corners.shape[0])
    if selected < 4:
        return LKEvidence(selected, 0, threshold, missing_reason="纹理不足")
    # Coarse levels are only useful while the object still fills the window;
    # otherwise the static background dominates and drags points to zero motion.
    window = 2 * HALF_WINDOW + 1
    levels = 1
    while levels < LEVELS and min(prev_stats.w, prev_stats.h) / (2**levels) >= window:
        levels += 1
    prev_pyr = build_pyramid(prev_crop, levels)
    cur_pyr = build_pyramid(cur_crop, levels)
    full_region = np.zeros(prev_crop.shape, dtype=np.float32)
    full_region[ty0 : ty0 + sub.shape[0], tx0 : tx0 + sub.shape[1]] = sub
    weights = build_pyramid(full_region, levels)
    init = np.tile(np.array([[dx, dy]], dtype=np.float32), (selected, 1))
    fwd, ok_f, res_f = lk_track(prev_pyr, cur_pyr, corners, init, weights=weights)
    # Backward: the same object pixels, now around the forwarded positions.
    shifted = np.zeros_like(full_region)
    shift = np.median(fwd - corners, axis=0) if ok_f.any() else np.zeros(2)
    sx, sy = int(round(float(shift[0]))), int(round(float(shift[1])))
    hh, ww = full_region.shape
    dst = full_region[max(0, -sy) : hh - max(0, sy), max(0, -sx) : ww - max(0, sx)]
    shifted[max(0, sy) : max(0, sy) + dst.shape[0], max(0, sx) : max(0, sx) + dst.shape[1]] = dst
    back, ok_b, _res_b = lk_track(
        cur_pyr, prev_pyr, fwd, -(fwd - corners), weights=build_pyramid(shifted, levels)
    )
    fb = np.hypot(*(back - corners).T)
    valid = ok_f & ok_b & (fb <= threshold) & (res_f <= MAX_RESIDUAL)
    n_valid = int(valid.sum())
    fb_med = float(np.median(fb[ok_f & ok_b])) if (ok_f & ok_b).any() else None
    if n_valid < 3 or n_valid < 0.3 * selected:
        return LKEvidence(
            selected,
            n_valid,
            threshold,
            fb_median_px=fb_med,
            missing_reason="光流不可靠（模糊、遮挡或形变）",
        )
    moved = fwd[valid] + np.array([x0, y0], dtype=np.float32)
    inside = float(np.mean([candidate.contains(float(x), float(y), pad=1) for x, y in moved]))
    disp = fwd[valid] - corners[valid]
    median_disp = np.median(disp, axis=0)
    spread = float(np.median(np.hypot(*(disp - median_disp).T)))
    target = (prev_stats.x + float(median_disp[0]), prev_stats.y + float(median_disp[1]))
    target_distance = float(np.hypot(candidate.x - target[0], candidate.y - target[1]))
    valid_fraction = n_valid / selected
    quality = float(np.clip(inside * (0.5 + 0.5 * valid_fraction), 0.0, 1.0))
    return LKEvidence(
        selected=selected,
        valid=n_valid,
        fb_threshold_px=threshold,
        fb_median_px=float(np.median(fb[valid])),
        inside_fraction=inside,
        target_xy=target,
        target_distance_px=target_distance,
        spread_px=spread,
        quality=quality,
    )
