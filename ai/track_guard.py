"""Motion-aware gate around SAM 2 masks (SAMURAI-style, no extra weights).

Rejects masks that jump onto static stripes / foliage, drops those frames
from SAM 2 memory, and proposes a positive click near the predicted ball.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ai.contracts import PromptKind, TrackPrompt
from ai.local_flow import LKEvidence, lk_evidence  # noqa: F401  (re-exported)

REJECT_BACKGROUND = "疑似跳到背景"
REJECT_STREAK = 3
FG_MAX_SIDE = 360
FG_SHIFT_FRAC = 0.08
FG_MAX_FILL = 0.15
FG_MIN_OVERLAP = 0.20
AREA_EXPLODE = 8.0
WARMUP_UPDATES = 4
MIN_DIST_PX = 48.0
RECOVERED_CONFIDENCE = 0.55
GAP_MAX_S = 0.12
GAP_MAX_FRAMES = 8
GAP_FIT_POINTS = 4


@dataclass(frozen=True)
class MaskStats:
    x: float
    y: float
    w: float
    h: float
    area: float
    contour: list[tuple[float, float]]
    # Boolean crop of the pixels this region covers, anchored at `origin`
    # (x0, y0). Kept for appearance and optical-flow checks inside the object;
    # never serialised and never used to move the centre.
    pixels: np.ndarray | None = field(default=None, compare=False, repr=False)
    origin: tuple[int, int] = field(default=(0, 0), compare=False, repr=False)

    def contains(self, x: float, y: float, pad: int = 0) -> bool:
        if self.pixels is None:
            half_w, half_h = self.w / 2.0 + pad, self.h / 2.0 + pad
            return abs(x - self.x) <= half_w and abs(y - self.y) <= half_h
        cx = int(round(x)) - self.origin[0]
        cy = int(round(y)) - self.origin[1]
        h, w = self.pixels.shape
        y0, y1 = max(0, cy - pad), min(h, cy + pad + 1)
        x0, x1 = max(0, cx - pad), min(w, cx + pad + 1)
        if y1 <= y0 or x1 <= x0:
            return False
        return bool(self.pixels[y0:y1, x0:x1].any())


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
    mask_quality: float | None = None


@dataclass(frozen=True)
class AppearanceEvidence:
    similarity: float | None = None
    ambiguous: bool = False
    flow_quality: float | None = None
    forward_backward_error: float | None = None
    # Pyramid LK point-group evidence (None when the check had too little
    # texture or could not run; missing evidence is never a failure).
    lk: "LKEvidence | None" = None
    # Chromaticity of the object's own pixels against its reference (0..1).
    color_similarity: float | None = None
    # How much the candidate's pixels changed since two frames ago, relative
    # to what the object usually changes (camera motion removed). Near zero
    # means the candidate sits on something static.
    change_ratio: float | None = None


@dataclass(frozen=True)
class TrajectoryEvidence:
    """Candidate against a robust local quadratic through committed points."""

    deviation_px: float
    tolerance_px: float
    projectile: bool = False  # user said this is free flight: allow motion-only review
    points: int = 0
    recovering: bool = False  # after a loss: only clear outliers count


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
    diagnostics: dict[str, Any] = field(default_factory=dict)


CONTOUR_MAX_POINTS = 96
MAX_COMPONENTS = 256


def _as_binary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask).astype(bool)
    if binary.ndim != 2:
        binary = binary.reshape(binary.shape[-2], binary.shape[-1])
    return binary


def label_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labels via row runs + union-find. Labels start at 1.

    Every component is found regardless of scan order, so a distractor near
    the top of the frame cannot push the target out of a fixed-size list.
    """
    b = _as_binary(binary)
    h, w = b.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if not b.any():
        return labels, 0
    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = b
    edges = np.diff(padded, axis=1)
    run_rows, run_starts = np.nonzero(edges == 1)
    _end_rows, run_ends = np.nonzero(edges == -1)  # exclusive
    n_runs = int(run_rows.size)
    parent = list(range(n_runs))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    row_first = np.searchsorted(run_rows, np.arange(h + 1))
    rows = run_rows.tolist()
    starts = run_starts.tolist()
    ends = run_ends.tolist()
    first = row_first.tolist()
    for r in range(1, h):
        a0, a1 = first[r], first[r + 1]
        b0, b1 = first[r - 1], first[r]
        if a0 == a1 or b0 == b1:
            continue
        j = b0
        for i in range(a0, a1):
            s_i, e_i = starts[i], ends[i]
            while j < b1 and ends[j] < s_i:  # previous run ends before (8-conn: e >= s)
                j += 1
            k = j
            while k < b1 and starts[k] <= e_i:
                ri, rk = find(i), find(k)
                if ri != rk:
                    parent[max(ri, rk)] = min(ri, rk)
                k += 1
    roots = np.array([find(i) for i in range(n_runs)], dtype=np.int64)
    _uniq, comp_of_run = np.unique(roots, return_inverse=True)
    lengths = run_ends - run_starts
    pixel_labels = np.repeat(comp_of_run.astype(np.int32) + 1, lengths)
    labels[b] = pixel_labels  # boolean indexing is row-major, same as runs
    del rows
    return labels, int(_uniq.size)


_MOORE = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def trace_contour(pixels: np.ndarray, origin: tuple[int, int] = (0, 0)) -> list[tuple[float, float]]:
    """Ordered outer boundary (Moore-neighbour tracing) of one component."""
    mask = np.asarray(pixels, dtype=bool)
    if not mask.any():
        return []
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    ys, xs = np.nonzero(padded)
    start = (int(ys[0]), int(xs[0]))  # top-most, then left-most
    path = [start]
    current = start
    direction = 6  # came from the west
    limit = min(4 * int(mask.size) + 8, 20_000)
    for _ in range(limit):
        found = False
        for step in range(8):
            d = (direction + 1 + step) % 8
            ny, nx = current[0] + _MOORE[d][0], current[1] + _MOORE[d][1]
            if padded[ny, nx]:
                # Next search starts just past the backtrack direction.
                direction = (d + 4) % 8
                current = (ny, nx)
                found = True
                break
        if not found:
            break  # isolated pixel
        if current == start and len(path) > 1:
            break
        path.append(current)
    ox, oy = origin
    points = [(float(x - 1 + ox), float(y - 1 + oy)) for y, x in path]
    if len(points) > CONTOUR_MAX_POINTS:
        step = int(np.ceil(len(points) / CONTOUR_MAX_POINTS))
        points = points[::step]
    return points


def _stats_from_pixels(
    pixels: np.ndarray, origin: tuple[int, int], *, contour: bool = True
) -> MaskStats | None:
    ys, xs = np.nonzero(pixels)
    if xs.size == 0:
        return None
    ox, oy = origin
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = pixels[y0 : y1 + 1, x0 : x1 + 1]
    crop_origin = (ox + x0, oy + y0)
    outline = trace_contour(crop, crop_origin) if contour else []
    return MaskStats(
        x=float(xs.mean() + ox),
        y=float(ys.mean() + oy),
        w=float(x1 - x0 + 1),
        h=float(y1 - y0 + 1),
        area=float(xs.size),
        contour=outline
        or [
            (float(ox + x0), float(oy + y0)),
            (float(ox + x1), float(oy + y0)),
            (float(ox + x1), float(oy + y1)),
            (float(ox + x0), float(oy + y1)),
        ],
        pixels=crop,
        origin=crop_origin,
    )


def mask_stats(mask: np.ndarray) -> MaskStats | None:
    """Whole-mask statistics: the centre is the mean of every mask pixel.

    The outline follows the largest component so the overlay is a real shape
    rather than a bounding rectangle; it never changes the centre.
    """
    binary = _as_binary(mask)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = binary[y0 : y1 + 1, x0 : x1 + 1]
    outline: list[tuple[float, float]] = []
    if crop.size <= 4_000_000:
        labels, count = label_components(crop)
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            outline = trace_contour(labels == int(np.argmax(sizes)), (x0, y0))
    return MaskStats(
        x=float(xs.mean()),
        y=float(ys.mean()),
        w=max(1.0, float(x1 - x0 + 1)),
        h=max(1.0, float(y1 - y0 + 1)),
        area=float(xs.size),
        contour=outline
        or [(float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1))],
        pixels=crop,
        origin=(x0, y0),
    )


def with_contour(stats: MaskStats | None) -> MaskStats | None:
    """Trace the ordered outline of a region (done once, for the chosen one)."""
    if stats is None or stats.pixels is None or len(stats.contour) > 4:
        return stats
    outline = trace_contour(stats.pixels, stats.origin)
    return replace(stats, contour=outline) if outline else stats


def mask_components(
    mask: np.ndarray, max_components: int = MAX_COMPONENTS, *, contours: bool = False
) -> list[MaskStats]:
    """Connected SAM components, largest first, each with its own pixels.

    Outlines are traced only on request; callers trace the chosen region with
    `with_contour` so a frame full of fragments stays cheap.
    """
    full = _as_binary(mask)
    rows = np.flatnonzero(full.any(axis=1))
    if rows.size == 0:
        return []
    cols = np.flatnonzero(full.any(axis=0))
    by0, bx0 = int(rows[0]), int(cols[0])
    binary = full[by0 : int(rows[-1]) + 1, bx0 : int(cols[-1]) + 1]
    labels, count = label_components(binary)
    if count == 0:
        return []
    flat = labels.ravel()
    sizes = np.bincount(flat, minlength=count + 1)
    order = [int(i) for i in np.argsort(-sizes[1:], kind="stable")[:max_components] + 1]
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    x_min = np.full(count + 1, np.iinfo(np.int32).max)
    y_min = np.full(count + 1, np.iinfo(np.int32).max)
    x_max = np.full(count + 1, -1)
    y_max = np.full(count + 1, -1)
    np.minimum.at(x_min, lab, xs)
    np.minimum.at(y_min, lab, ys)
    np.maximum.at(x_max, lab, xs)
    np.maximum.at(y_max, lab, ys)
    components: list[MaskStats] = []
    for index in order:
        x0, x1, y0, y1 = int(x_min[index]), int(x_max[index]), int(y_min[index]), int(y_max[index])
        crop = labels[y0 : y1 + 1, x0 : x1 + 1] == index
        stats = _stats_from_pixels(crop, (x0 + bx0, y0 + by0), contour=contours)
        if stats is not None:
            components.append(stats)
    return components


def merge_components(items: list[MaskStats]) -> MaskStats | None:
    """One region made of several fragments; centre is the mean of all pixels."""
    items = [item for item in items if item.pixels is not None]
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    x0 = min(item.origin[0] for item in items)
    y0 = min(item.origin[1] for item in items)
    x1 = max(item.origin[0] + item.pixels.shape[1] for item in items)  # type: ignore[union-attr]
    y1 = max(item.origin[1] + item.pixels.shape[0] for item in items)  # type: ignore[union-attr]
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for item in items:
        assert item.pixels is not None
        oy, ox = item.origin[1] - y0, item.origin[0] - x0
        canvas[oy : oy + item.pixels.shape[0], ox : ox + item.pixels.shape[1]] |= item.pixels
    return _stats_from_pixels(canvas, (x0, y0), contour=False)


def _box_gap(a: MaskStats, b: MaskStats) -> float:
    ax0, ay0 = a.x - a.w / 2.0, a.y - a.h / 2.0
    bx0, by0 = b.x - b.w / 2.0, b.y - b.h / 2.0
    if a.pixels is not None:
        ax0, ay0 = float(a.origin[0]), float(a.origin[1])
    if b.pixels is not None:
        bx0, by0 = float(b.origin[0]), float(b.origin[1])
    dx = max(0.0, max(ax0, bx0) - min(ax0 + a.w, bx0 + b.w))
    dy = max(0.0, max(ay0, by0) - min(ay0 + a.h, by0 + b.h))
    return float(np.hypot(dx, dy))


def select_component(
    mask: np.ndarray,
    pred: PredictedBox | None,
    expected_area: float | None,
    seed_xy: tuple[float, float] | None = None,
    components: list[MaskStats] | None = None,
) -> MaskStats | None:
    """Pick the target region among SAM's connected components.

    The prediction and the seed only rank regions. The returned centre is the
    chosen region's own centroid (fragments of one object may be combined).
    """
    binary = _as_binary(mask)
    # Only skip the component search when SAM has swallowed most of the frame;
    # the area gate rejects that case.
    if binary.size and binary.mean() > 0.55:
        return mask_stats(binary)
    candidates = list(components) if components is not None else mask_components(binary)
    if not candidates:
        return None
    if len(candidates) == 1:
        return with_contour(candidates[0])
    target_x = pred.x if pred is not None else (seed_xy[0] if seed_xy else candidates[0].x)
    target_y = pred.y if pred is not None else (seed_xy[1] if seed_xy else candidates[0].y)
    scale = max(
        12.0,
        pred.sigma if pred is not None else 0.0,
        pred.step * 2.0 if pred is not None else 0.0,
        pred.w * 2.0 if pred is not None else 0.0,
    )
    best: tuple[float, MaskStats] | None = None
    for item in candidates:
        dist = float(np.hypot(item.x - target_x, item.y - target_y))
        distance_score = float(np.exp(-dist / scale))
        if pred is None and seed_xy is not None and item.contains(seed_xy[0], seed_xy[1], pad=1):
            # First frame: the region under the user's click is the target.
            distance_score = 1.0
        if expected_area is not None and expected_area > 1.0:
            area_score = min(item.area, expected_area) / max(item.area, expected_area)
        else:
            area_score = 1.0
        score = 0.72 * distance_score + 0.28 * area_score
        if best is None or score > best[0]:
            best = (score, item)
    assert best is not None
    chosen = best[1]
    # Fragments of one object (a bar across the ball, blur splitting the
    # mask) belong together. Only merge when the chosen piece is clearly
    # smaller than the known object and the union stays object-sized.
    if expected_area is not None and expected_area > 4.0 and chosen.area < 0.8 * expected_area:
        size = float(np.sqrt(expected_area))
        group = [chosen]
        total = chosen.area
        for item in sorted(candidates, key=lambda c: _box_gap(chosen, c)):
            if item is chosen:
                continue
            if _box_gap(chosen, item) > max(2.0, 0.35 * size):
                break
            if total + item.area > 1.6 * expected_area:
                continue
            group.append(item)
            total += item.area
        if len(group) > 1:
            merged = merge_components(group)
            if merged is not None:
                largest = with_contour(max(group, key=lambda c: c.area))
                return replace(merged, contour=list(largest.contour)) if largest else merged
    return with_contour(chosen)


def tighten_mask(
    stats: MaskStats | None,
    binary: np.ndarray,
    pred: PredictedBox | None,
    expected_area: float | None,
) -> MaskStats | None:
    """Keep the point on the object when SAM returns a loose region.

    A mask that covers the railing as well as a small target has its centroid
    in the middle of that region. Crop to the known object size around the
    prediction, but only when that prediction still lies inside the mask.
    """
    if (
        stats is None
        or pred is None
        or expected_area is None
        or expected_area < 8.0
        or stats.area <= expected_area * 2.5
    ):
        return stats
    mask = np.asarray(binary).astype(bool)
    if mask.ndim != 2:
        mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    radius = max(6.0, float(np.sqrt(expected_area)) * 1.5)
    h, w = mask.shape
    x0 = int(max(0, np.floor(pred.x - radius)))
    x1 = int(min(w, np.ceil(pred.x + radius + 1)))
    y0 = int(max(0, np.floor(pred.y - radius)))
    y1 = int(min(h, np.ceil(pred.y + radius + 1)))
    if x1 <= x0 or y1 <= y0 or not mask[y0:y1, x0:x1].any():
        return stats
    window = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(window)
    return MaskStats(
        x=float(xs.mean() + x0),
        y=float(ys.mean() + y0),
        w=float(xs.max() - xs.min() + 1),
        h=float(ys.max() - ys.min() + 1),
        area=float(xs.size),
        contour=[
            (float(x0 + xs.min()), float(y0 + ys.min())),
            (float(x0 + xs.max()), float(y0 + ys.min())),
            (float(x0 + xs.max()), float(y0 + ys.max())),
            (float(x0 + xs.min()), float(y0 + ys.max())),
        ],
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
        """Typical per-update displacement in pixels (no fixed floor)."""
        if not self._steps:
            return 0.0
        return float(np.median(self._steps))

    def _size_floor(self) -> float:
        if self._x is None:
            return 1.0
        return float(max(1.0, 0.5 * max(self._x[4], self._x[5])))

    def residual_sigma(self, dt: float | None = None) -> float:
        """Position uncertainty from the predicted covariance and recent residuals.

        Normalised by the object size instead of a fixed 48 px floor, so small
        targets get a proportionally small gate.
        """
        cov_sigma = 0.0
        if self._P is not None:
            P = self._P
            if dt is not None:
                F = self._F(dt)
                P = F @ P @ F.T + self._Q(dt)
            cov_sigma = float(np.sqrt(max(P[0, 0] + P[1, 1], 1e-6)))
        resid_sigma = 0.0
        if len(self._residuals) >= 3:
            resid = np.asarray(self._residuals, dtype=np.float64)
            resid_sigma = float(np.median(resid) * 1.4826)
        return float(max(self._size_floor(), cov_sigma, resid_sigma))

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
            sigma=self.residual_sigma(dt),
        )

    @property
    def last_time(self) -> float | None:
        return self._t

    def update(
        self, t: float, x: float, y: float, w: float, h: float, area: float | None = None
    ) -> PredictedBox:
        z = np.array([x, y, max(1.0, w), max(1.0, h)], dtype=np.float64)
        pix = float(area if area is not None else z[2] * z[3])
        if self._x is not None and self._t is not None and float(t) <= self._t + 1e-9:
            # Re-running the same (or an earlier) frame must not advance the
            # motion state twice. Callers restore a checkpoint to redo a frame.
            return self.predict(self._t + 1e-4) or PredictedBox(x=x, y=y, w=z[2], h=z[3])
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
                x=x,
                y=y,
                w=float(z[2]),
                h=float(z[3]),
                vx=vx,
                vy=vy,
                step=step,
                sigma=self.residual_sigma(),
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
    # The correlation peak above is the shift that maps current back to
    # previous; callers need the forward previous -> current translation.
    return float(-dx), float(-dy)


def _shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate without wrapping the opposite image edge into the frame."""
    ix, iy = int(round(dx)), int(round(dy))
    out = np.full(image.shape, np.nan, dtype=np.float32)
    h, w = image.shape
    dst_x0, dst_x1 = max(0, ix), min(w, w + ix)
    dst_y0, dst_y1 = max(0, iy), min(h, h + iy)
    src_x0, src_x1 = max(0, -ix), min(w, w - ix)
    src_y0, src_y1 = max(0, -iy), min(h, h - iy)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


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
    # Decimate before the colour conversion: identical values, far less work.
    prev_s, scale = _downsample(np.asarray(prev), max_side)
    cur_s, _ = _downsample(np.asarray(cur), max_side)
    prev_g = _to_gray(prev_s)
    cur_g = _to_gray(cur_s)
    if shift is None:
        estimated = estimate_shift(prev_g, cur_g)
        raw_diff = np.abs(cur_g - prev_g)
        raw_thr = max(18.0, float(np.median(raw_diff)) * 3.5)
        raw_fill = float((raw_diff > raw_thr).mean()) if raw_diff.size else 1.0
        # A sparse moving target can dominate phase correlation on repetitive
        # backgrounds. When the unaligned difference is already sparse, treat
        # the camera as stationary instead of aligning the target away.
        candidates = (
            [(0.0, 0.0)]
            if raw_fill < FG_MAX_FILL * 0.25
            else [(0.0, 0.0), estimated]
        )
    else:
        candidates = [(shift[0] / scale, shift[1] / scale)]
    best: tuple[float, float, float, np.ndarray] | None = None
    for dx, dy in candidates:
        aligned = _shift_image(prev_g, dx, dy)
        diff = np.abs(cur_g - aligned)
        finite = np.isfinite(diff)
        if not finite.any():
            continue
        med = float(np.median(diff[finite]))
        thr = max(18.0, med * 3.5)
        binary = np.zeros_like(diff, dtype=bool)
        binary[finite] = diff[finite] > thr
        fill = float(binary[finite].mean()) if finite.any() else 1.0
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


def _appearance_patch(frame: np.ndarray, stats: MaskStats, size: int = 24) -> np.ndarray | None:
    """Object-only appearance sample.

    Pixels outside the region's own mask are replaced by the object's mean, so
    after centring they carry no texture and a busy background (stripes,
    leaves) cannot dominate the template. Tiny or mask-less regions fall back
    to the surrounding box.
    """
    img = np.asarray(frame)
    if stats.pixels is not None and stats.area >= 9:
        ox, oy = stats.origin
        ph, pw = stats.pixels.shape
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(img.shape[1], ox + pw), min(img.shape[0], oy + ph)
        if x1 - x0 >= 3 and y1 - y0 >= 3:
            patch = _to_gray(img[y0:y1, x0:x1])
            inside = stats.pixels[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
            if inside.any():
                fill = float(patch[inside].mean())
                patch = np.where(inside, patch, fill).astype(np.float32)
                ys = np.linspace(0, patch.shape[0] - 1, size).round().astype(int)
                xs = np.linspace(0, patch.shape[1] - 1, size).round().astype(int)
                return patch[np.ix_(ys, xs)]
    half_w = max(4.0, stats.w * 0.8)
    half_h = max(4.0, stats.h * 0.8)
    x0 = max(0, int(np.floor(stats.x - half_w)))
    x1 = min(img.shape[1], int(np.ceil(stats.x + half_w + 1)))
    y0 = max(0, int(np.floor(stats.y - half_h)))
    y1 = min(img.shape[0], int(np.ceil(stats.y + half_h + 1)))
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None
    patch = _to_gray(img[y0:y1, x0:x1])
    ys = np.linspace(0, patch.shape[0] - 1, size).round().astype(int)
    xs = np.linspace(0, patch.shape[1] - 1, size).round().astype(int)
    return patch[np.ix_(ys, xs)].astype(np.float32)


def background_contrast(frame: np.ndarray, stats: MaskStats, ring: int = 3) -> float | None:
    """How different the object is from its immediate surroundings (0..1)."""
    if stats.pixels is None:
        return None
    gray = _to_gray(frame)
    ox, oy = stats.origin
    ph, pw = stats.pixels.shape
    x0, y0 = max(0, ox - ring), max(0, oy - ring)
    x1, y1 = min(gray.shape[1], ox + pw + ring), min(gray.shape[0], oy + ph + ring)
    region = gray[y0:y1, x0:x1]
    inside = np.zeros(region.shape, dtype=bool)
    iy0, ix0 = oy - y0, ox - x0
    sub = stats.pixels[max(0, -iy0) : ph, max(0, -ix0) : pw]
    inside[max(0, iy0) : max(0, iy0) + sub.shape[0], max(0, ix0) : max(0, ix0) + sub.shape[1]] = sub
    if not inside.any() or inside.all():
        return None
    obj = region[inside]
    bg = region[~inside]
    spread = float(np.sqrt(0.5 * (obj.var() + bg.var())) + 8.0)
    return float(np.clip(abs(float(obj.mean()) - float(bg.mean())) / (3.0 * spread), 0.0, 1.0))


def appearance_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape or reference.size == 0:
        return 0.0
    a = reference.astype(np.float32)
    b = candidate.astype(np.float32)
    mean_score = 1.0 - min(1.0, abs(float(a.mean() - b.mean())) / 96.0)
    ac = a - float(a.mean())
    bc = b - float(b.mean())
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denom <= 1e-6:
        texture_score = mean_score
    else:
        texture_score = float(np.clip((float(np.sum(ac * bc)) / denom + 1.0) * 0.5, 0.0, 1.0))
    return float(np.clip(0.65 * texture_score + 0.35 * mean_score, 0.0, 1.0))


def _point_patch(gray: np.ndarray, x: float, y: float, radius: int) -> np.ndarray | None:
    cx, cy = int(round(x)), int(round(y))
    if cx - radius < 0 or cy - radius < 0:
        return None
    if cx + radius >= gray.shape[1] or cy + radius >= gray.shape[0]:
        return None
    return gray[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]


def _best_patch_match(
    template: np.ndarray,
    image: np.ndarray,
    center: tuple[float, float],
    search: int,
) -> tuple[float, float, float] | None:
    radius = template.shape[0] // 2
    best: tuple[float, float, float] | None = None
    scale = max(16.0, float(template.std()) * 2.0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            x, y = center[0] + dx, center[1] + dy
            patch = _point_patch(image, x, y, radius)
            if patch is None or patch.shape != template.shape:
                continue
            error = float(np.mean(np.abs(template - patch)) / scale)
            if best is None or error < best[0]:
                best = (error, x, y)
    return best


def bidirectional_patch_evidence(
    previous: np.ndarray,
    current: np.ndarray,
    previous_xy: tuple[float, float],
    candidate_xy: tuple[float, float],
    *,
    radius: int = 4,
    search: int = 7,
) -> tuple[float | None, float | None]:
    """Small dependency-free forward/backward optical-flow consistency check."""
    prev_gray = _to_gray(previous)
    cur_gray = _to_gray(current)
    template = _point_patch(prev_gray, previous_xy[0], previous_xy[1], radius)
    if template is None or float(template.std()) < 4.0:
        return None, None
    forward = _best_patch_match(template, cur_gray, candidate_xy, search)
    if forward is None:
        return None, None
    current_patch = _point_patch(cur_gray, forward[1], forward[2], radius)
    if current_patch is None:
        return None, None
    backward = _best_patch_match(current_patch, prev_gray, previous_xy, search)
    if backward is None:
        return None, None
    fb_error = float(np.hypot(backward[1] - previous_xy[0], backward[2] - previous_xy[1]))
    quality = float(np.exp(-0.5 * fb_error * fb_error) * np.exp(-min(forward[0], 4.0)))
    return quality, fb_error


def mask_logit_quality(logits: Any, binary: np.ndarray) -> float | None:
    """Turn SAM mask logits into bounded supporting evidence."""
    try:
        arr = np.asarray(logits.detach().cpu() if hasattr(logits, "detach") else logits)
    except Exception:  # noqa: BLE001
        return None
    while arr.ndim > 2:
        arr = arr[0]
    mask = np.asarray(binary, dtype=bool)
    if arr.shape != mask.shape or not mask.any():
        return None
    values = np.clip(arr[mask].astype(np.float64), -20.0, 20.0)
    return float(np.median(1.0 / (1.0 + np.exp(-values))))


COLOR_SIGMA = 18.0
TRAJ_WINDOW = 10
TRAJ_MIN_POINTS = 5
TRAJ_MAX_GAP_FRAMES = 15
# A quadratic extrapolated past a few frames diverges quickly; beyond this
# many frames since the last committed point it is not used at all.
TRAJ_MAX_EXTRAPOLATION = 3


def color_signature(frame: np.ndarray, stats: MaskStats) -> np.ndarray | None:
    """Median brightness-normalised opponent colour of the region's own pixels.

    Luminance-only templates cannot tell a pale grey streak from a bright
    yellow rail; chromaticity can. Normalising by brightness keeps the value
    stable when the object passes from shade into light.
    """
    if stats.pixels is None or stats.area < 4:
        return None
    img = np.asarray(frame)
    if img.ndim != 3 or img.shape[2] < 3:
        return None
    ox, oy = stats.origin
    ph, pw = stats.pixels.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(img.shape[1], ox + pw), min(img.shape[0], oy + ph)
    if x1 <= x0 or y1 <= y0:
        return None
    inside = stats.pixels[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    px = img[y0:y1, x0:x1][inside].astype(np.float32)
    if px.shape[0] < 4:
        return None
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    bright = (r + g + b) / 3.0
    o1 = (r - g) / np.sqrt(2.0)
    o2 = (r + g - 2.0 * b) / np.sqrt(6.0)
    chroma = np.stack([o1, o2], axis=1) / (bright[:, None] + 30.0) * 100.0
    return np.median(chroma, axis=0)


def color_similarity(reference: np.ndarray | None, candidate: np.ndarray | None) -> float | None:
    if reference is None or candidate is None:
        return None
    d = float(np.hypot(*(np.asarray(candidate) - np.asarray(reference))))
    return float(np.exp(-((d / COLOR_SIGMA) ** 2)))


def region_change(
    current: np.ndarray,
    earlier: np.ndarray,
    stats: MaskStats,
    shift: tuple[float, float] = (0.0, 0.0),
) -> float | None:
    """Mean |I_now - I_earlier| over the region's pixels, earlier frame shifted
    by the camera motion `shift` (earlier -> now, pixels)."""
    if stats.pixels is None or stats.area < 4:
        return None
    cur = np.asarray(current)
    old = np.asarray(earlier)
    ox, oy = stats.origin
    ph, pw = stats.pixels.shape
    sx, sy = int(round(shift[0])), int(round(shift[1]))
    # Pixel (x, y) now was at (x - sx, y - sy) earlier; keep both inside.
    x0, y0 = max(0, ox, sx), max(0, oy, sy)
    x1 = min(cur.shape[1], ox + pw, old.shape[1] + sx)
    y1 = min(cur.shape[0], oy + ph, old.shape[0] + sy)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    inside = stats.pixels[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    if inside.sum() < 4:
        return None
    a = _to_gray(cur[y0:y1, x0:x1])
    b = _to_gray(old[y0 - sy : y1 - sy, x0 - sx : x1 - sx])
    return float(np.abs(a - b)[inside].mean())


class LocalTrajectory:
    """Robust constant-acceleration fit over the latest committed centres.

    Used to rank candidates, to aim a relocalisation prompt and as one piece
    of evidence. It never supplies a measurement.
    """

    def __init__(self, window: int = TRAJ_WINDOW) -> None:
        self.window = int(window)
        self.samples: list[tuple[int, float, float, float]] = []  # frame, t, x, y

    def add(self, frame: int, t: float, x: float, y: float) -> None:
        self.samples = [s for s in self.samples if s[0] != frame]
        self.samples.append((int(frame), float(t), float(x), float(y)))
        self.samples.sort(key=lambda s: s[0])
        self.samples = self.samples[-2 * self.window :]

    def predict(self, frame: int, t: float) -> tuple[float, float, float, int] | None:
        """(x, y, robust residual sigma, points used) or None.

        None when fewer than TRAJ_MIN_POINTS recent points exist, or when the
        last committed point is more than TRAJ_MAX_EXTRAPOLATION frames back.
        """
        use = [s for s in self.samples if 0 < frame - s[0] <= TRAJ_MAX_GAP_FRAMES][-self.window :]
        if len(use) < TRAJ_MIN_POINTS or frame - use[-1][0] > TRAJ_MAX_EXTRAPOLATION:
            return None
        ts = np.array([s[1] - t for s in use])
        xs = np.array([s[2] for s in use])
        ys = np.array([s[3] for s in use])
        if float(ts.max() - ts.min()) <= 0:
            return None
        w = np.ones(len(ts))
        sigma = 1.0
        cx = cy = None
        for _ in range(5):
            cx = np.polyfit(ts, xs, 2, w=w)
            cy = np.polyfit(ts, ys, 2, w=w)
            r = np.hypot(xs - np.polyval(cx, ts), ys - np.polyval(cy, ts))
            sigma = max(1.0, 1.4826 * float(np.median(r)))
            w = np.where(r < 2.5 * sigma, 1.0, 2.5 * sigma / np.maximum(r, 1e-6))
        assert cx is not None and cy is not None
        return float(np.polyval(cx, 0.0)), float(np.polyval(cy, 0.0)), sigma, len(use)


def trajectory_tolerance(sigma: float, size: float, gap: int = 1) -> float:
    """Allowed deviation from the local fit. The fit residual is capped by the
    object size so a few bad points cannot inflate the gate; the gate widens
    with every frame of extrapolation."""
    size = max(size, 1.0)
    base = max(6.0, 0.6 * size, 2.0 * min(sigma, 0.5 * size))
    return float(base * (1.0 + 0.5 * max(0, int(gap) - 1)))


def search_limit(pred: PredictedBox, view_span: float, updates: int) -> float:
    """How far the ball may move before a mask is a jump.

    The floor follows the object size. A fixed 48px gate dominated small targets.
    """
    span = max(float(view_span), 8.0)
    object_floor = max(4.0, max(pred.w, pred.h) * 2.0)
    if updates < WARMUP_UPDATES:
        return max(0.22 * span, 4.0 * pred.step, object_floor)
    return max(4.0 * pred.step, 3.0 * pred.sigma, 0.04 * span, object_floor)


def score_mask(
    stats: MaskStats | None,
    pred: PredictedBox | None,
    fg: Foreground | None,
    sam: SamScores | None = None,
    median_area: float | None = None,
    appearance: AppearanceEvidence | None = None,
    *,
    view_span: float = 720.0,
    updates: int = 0,
    trajectory: TrajectoryEvidence | None = None,
    reference_score: float | None = None,
) -> GateDecision:
    """Score independent evidence; unavailable signals never count as perfect."""
    if stats is None:
        return GateDecision(False, 0.0, REJECT_BACKGROUND, {"missing_mask": True})
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
        return GateDecision(
            False,
            0.05,
            REJECT_BACKGROUND,
            {"distance_px": dist, "area_ratio": ratio, "area_jump": True},
        )
    # A clearly absent object that is also far from the ball. Logit alone is not enough:
    # small or blurred balls often sit slightly below zero.
    if sam.object_score is not None and sam.object_score < -6.0 and far:
        return GateDecision(
            False,
            0.02,
            REJECT_BACKGROUND,
            {"distance_px": dist, "object_probability": 0.0},
        )

    evidence: list[tuple[str, float, float]] = []
    motion = None
    if pred is not None:
        normalized = dist / max(limit, 1e-6)
        motion = float(np.exp(-normalized * normalized))
        evidence.append(("motion", motion, 0.25))
    area_score = float(np.exp(-abs(np.log(max(ratio, 1e-6)))))
    if median_area is not None:
        evidence.append(("shape", area_score, 0.14))
    if sam.object_score is not None:
        obj = float(1.0 / (1.0 + np.exp(-float(sam.object_score))))
        evidence.append(("object", obj, 0.24))
    else:
        obj = None
    if sam.iou is not None:
        evidence.append(("iou", float(np.clip(sam.iou, 0.0, 1.0)), 0.12))
    if sam.mask_quality is not None:
        evidence.append(("mask", float(np.clip(sam.mask_quality, 0.0, 1.0)), 0.10))
    overlap = None
    if fg is not None and fg.reliable:
        overlap = _overlap_fraction(stats, fg)
        # Frame difference is supporting evidence only: a stationary object can
        # have zero overlap and must not be rejected for that alone.
        evidence.append(("foreground", 0.35 + 0.65 * overlap, 0.06))
    similarity = None if appearance is None else appearance.similarity
    if similarity is not None:
        evidence.append(("appearance", float(np.clip(similarity, 0.0, 1.0)), 0.30))
    flow_quality = None if appearance is None else appearance.flow_quality
    if flow_quality is not None:
        evidence.append(("forward_backward", float(np.clip(flow_quality, 0.0, 1.0)), 0.18))

    weight = sum(item[2] for item in evidence)
    quality = sum(value * item_weight for _, value, item_weight in evidence) / max(weight, 1e-9)
    coverage = min(1.0, weight / 0.70)
    quality *= 0.55 + 0.45 * coverage
    diagnostics: dict[str, Any] = {
        "distance_px": round(dist, 3),
        "search_limit_px": round(limit, 3),
        "area": round(stats.area, 3),
        "area_ratio": round(ratio, 4),
        "available_evidence": [name for name, _, _ in evidence],
        "missing_object_score": sam.object_score is None,
        "missing_iou": sam.iou is None,
        "evidence_coverage": round(coverage, 4),
    }
    for name, value, _ in evidence:
        diagnostics[name] = round(value, 4)
    if appearance is not None:
        diagnostics["appearance_ambiguous"] = appearance.ambiguous
        diagnostics["forward_backward_error_px"] = appearance.forward_backward_error
        if appearance.lk is not None:
            diagnostics.update(appearance.lk.as_diagnostics())

    poor_identity = similarity is not None and similarity < 0.30
    poor_flow = flow_quality is not None and flow_quality < 0.12
    ambiguous = bool(appearance and appearance.ambiguous)
    absent = sam.object_score is not None and sam.object_score < -6.0

    # Independent weak cues. On a thin, motion-blurred object each one alone
    # is noisy (45° launch clip): the mask is partial, the model score dips,
    # the pixels change little. The rule throughout is that one noisy cue is
    # never enough; two independent ones are.
    weak: list[str] = []
    if sam.object_score is not None and reference_score is not None and reference_score > 0.5:
        if sam.object_score < 0.45 * reference_score:
            weak.append("model_score")
    if median_area is not None and median_area > 4.0 and ratio < 0.4:
        weak.append("area_shrink")
    change = None if appearance is None else appearance.change_ratio
    if change is not None:
        diagnostics["change_ratio"] = round(change, 4)
        if change < 0.35:
            weak.append("static")
    # Colour is recorded but not voted on: a semi-transparent streak takes the
    # colour of what is behind it (a yellow rail made a correct frame look
    # "yellow" in the real clip).
    color_sim = None if appearance is None else appearance.color_similarity
    if color_sim is not None:
        diagnostics["color"] = round(color_sim, 4)
    dev = None
    if trajectory is not None:
        dev = trajectory.deviation_px / max(trajectory.tolerance_px, 1e-6)
        diagnostics["trajectory_deviation_px"] = round(trajectory.deviation_px, 3)
        diagnostics["trajectory_tolerance_px"] = round(trajectory.tolerance_px, 3)
        diagnostics["trajectory_points"] = trajectory.points
    if weak:
        diagnostics["weak_evidence"] = list(weak)

    # Continuous tracking first (user requirement, 0.3.9 behaviour): a doubt
    # backed by one noisy cue keeps the point as a measurement, flagged as
    # suspect with a lower quality. A point is taken out only when two
    # independent cues agree, or the model says the object is absent.
    off_track = dev is not None and dev > 1.5
    corroborated = bool(weak) or off_track or far
    identity_doubt = poor_identity or (ambiguous and quality < 0.55)
    if updates >= 2 and (absent or (identity_doubt and corroborated)):
        diagnostics["rejected_by_evidence"] = True
        diagnostics["reason_code"] = "appearance_conflict" if not absent else "model_invisible"
        return GateDecision(False, float(np.clip(quality, 0.0, 1.0)), "外观冲突", diagnostics)
    suspect: list[str] = []
    if identity_doubt:
        suspect.append("appearance")
    lk = None if appearance is None else appearance.lk
    # The object's own corners, tracked forward and back, went somewhere else:
    # the candidate is not the object even if it sits on the prediction.
    if (
        updates >= 2
        and lk is not None
        and lk.strong
        and lk.inside_fraction is not None
        and lk.inside_fraction < 0.15
        and lk.target_distance_px is not None
        and lk.target_distance_px > max(3.0, 0.75 * max(stats.w, stats.h))
    ):
        diagnostics["rejected_by_evidence"] = True
        diagnostics["reason_code"] = "local_conflict"
        return GateDecision(False, float(np.clip(quality, 0.0, 1.0)), "局部光流冲突", diagnostics)
    if updates >= 2:
        verdict: tuple[str, str] | None = None
        recovering = trajectory is not None and trajectory.recovering
        if dev is not None and trajectory is not None and trajectory.projectile and dev > 4.0:
            verdict = ("trajectory_outlier", "偏离抛体轨迹")
        elif dev is not None and weak and dev > (3.0 if recovering else 1.5):
            verdict = ("trajectory_weak", "偏离轨迹且证据变弱")
        elif len(weak) >= 2 and (dev is None or dev > 1.0):
            verdict = ("weak_evidence", "多项证据同时变弱")
        elif len(weak) >= 2:
            suspect.append("weak_evidence")
        if verdict is not None:
            diagnostics["rejected_by_evidence"] = True
            diagnostics["reason_code"] = verdict[0]
            return GateDecision(False, float(np.clip(quality, 0.0, 1.0)), verdict[1], diagnostics)
    if updates >= 2 and poor_flow and similarity is not None and similarity < 0.45 and far:
        diagnostics["rejected_by_evidence"] = True
        diagnostics["reason_code"] = "flow_conflict"
        return GateDecision(False, float(np.clip(quality, 0.0, 1.0)), "双向不一致", diagnostics)
    if suspect:
        # Kept as a measurement; the lower quality and the flag show in the
        # table hover so the user can look at it if they want to.
        diagnostics["suspect"] = suspect
        quality *= 0.8
    return GateDecision(True, float(np.clip(quality, 0.0, 1.0)), "", diagnostics)


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


def recover_from_foreground(
    fg: Foreground | None,
    pred: PredictedBox | None,
    expected_area: float | None,
) -> MaskStats | None:
    """Measure the ball from the frame difference when SAM returned no mask."""
    if fg is None or pred is None or not fg.reliable or not fg.blobs:
        return None
    prompt = reprompt_candidate(fg.blobs, pred, expected_area=expected_area, scale=fg.scale)
    if prompt is None:
        return None
    half_w = max(2.0, pred.w / 2.0)
    half_h = max(2.0, pred.h / 2.0)
    x0, x1 = prompt.x - half_w, prompt.x + half_w
    y0, y1 = prompt.y - half_h, prompt.y + half_h
    return MaskStats(
        x=prompt.x,
        y=prompt.y,
        w=2.0 * half_w,
        h=2.0 * half_h,
        area=float(expected_area or pred.w * pred.h),
        contour=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
    )


def _fit_axis(ts: np.ndarray, vs: np.ndarray, t: float) -> float:
    degree = 2 if len(ts) >= 4 else 1
    t0 = float(ts.mean())
    coeffs = np.polyfit(ts - t0, vs, degree)
    return float(np.polyval(coeffs, t - t0))


def fill_short_gaps(
    points: list[Any],
    info: Any,
    *,
    max_gap_s: float = GAP_MAX_S,
    max_gap_frames: int = GAP_MAX_FRAMES,
    fit_points: int = GAP_FIT_POINTS,
) -> list[Any]:
    """Bridge brief SAM dropouts with a local quadratic in real PTS.

    Only gaps with visible, unedited samples on both sides are filled; the
    result is marked interpolated so it is never mistaken for a measurement.
    """
    from dataclasses import replace as _replace
    from ai.contracts import TrackPointSource, TrackPointStatus

    ordered = sorted(points, key=lambda p: p.frame)
    n = len(ordered)
    i = 0
    while i < n:
        if ordered[i].visible or ordered[i].manual:
            i += 1
            continue
        j = i
        while j < n and not ordered[j].visible and not ordered[j].manual:
            j += 1
        if i == 0 or j >= n:
            i = j
            continue
        before = [p for p in ordered[max(0, i - fit_points) : i] if p.visible]
        after = [p for p in ordered[j : j + fit_points] if p.visible]
        gap_frames = ordered[j].frame - ordered[i - 1].frame - 1
        gap_s = time_s(info, ordered[j].frame) - time_s(info, ordered[i - 1].frame)
        if (
            not before
            or not after
            or gap_frames > max_gap_frames
            or gap_s > max_gap_s + 1e-9
        ):
            i = j
            continue
        anchors = before + after
        ts = np.array([time_s(info, p.frame) for p in anchors], dtype=np.float64)
        xs = np.array([p.x for p in anchors], dtype=np.float64)
        ys = np.array([p.y for p in anchors], dtype=np.float64)
        conf = min(before[-1].confidence, after[0].confidence) * 0.8
        for k in range(i, j):
            t = time_s(info, ordered[k].frame)
            ordered[k] = _replace(
                ordered[k],
                x=_fit_axis(ts, xs, t),
                y=_fit_axis(ts, ys, t),
                visible=True,
                confidence=float(conf),
                interpolated=True,
                note="",
                status=TrackPointStatus.REVIEW,
                source=TrackPointSource.INTERPOLATED,
                diagnostics={"interpolation": "local_quadratic"},
            )
        i = j
    return ordered


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
    object_index: int | None = None
    if isinstance(payload, (tuple, list)) and len(payload) >= 2:
        try:
            ids = [int(item) for item in list(payload[1])]
            object_index = ids.index(int(object_id)) if int(object_id) in ids else None
        except Exception:  # noqa: BLE001
            object_index = None

    def _score(value: Any) -> float | None:
        if value is None:
            return None
        if object_index is None:
            return _as_float(value)
        try:
            arr = np.asarray(
                value.detach().cpu() if hasattr(value, "detach") else value,
                dtype=np.float64,
            )
            if arr.ndim > 0 and arr.shape[0] > object_index:
                return _as_float(arr[object_index])
        except Exception:  # noqa: BLE001
            pass
        return _as_float(value) if object_index == 0 else None

    if isinstance(payload, (tuple, list)) and len(payload) >= 4:
        extra = payload[3]
        if isinstance(extra, dict):
            score = _score(extra.get("object_score_logits", extra.get("object_score")))
            iou = _score(extra.get("iou_predictions", extra.get("iou")))
            if score is not None or iou is not None:
                return SamScores(score, iou)
    if not isinstance(state, dict):
        return SamScores()
    obj_idx = None
    mapping = state.get("obj_id_to_idx")
    if isinstance(mapping, dict):
        obj_idx = mapping.get(object_id)
    for key in ("output_dict_per_obj", "temp_output_dict_per_obj"):
        per_object = state.get(key)
        if not isinstance(per_object, dict) or obj_idx is None:
            continue
        blob = per_object.get(obj_idx)
        if not isinstance(blob, dict):
            continue
        for store_name in ("non_cond_frame_outputs", "cond_frame_outputs"):
            store = blob.get(store_name)
            item = store.get(frame_idx) if isinstance(store, dict) else None
            if isinstance(item, dict):
                score = _as_float(item.get("object_score_logits"))
                iou = _as_float(item.get("iou_predictions"))
                if score is not None or iou is not None:
                    return SamScores(score, iou)
    # Older single-object predictors do not expose obj_id_to_idx. Only inspect
    # the shared store when the requested object is the sole known object.
    known_ids = state.get("obj_ids")
    if known_ids is None or list(known_ids) == [object_id]:
        blob = state.get("output_dict")
        if isinstance(blob, dict):
            for store_name in ("non_cond_frame_outputs", "cond_frame_outputs"):
                store = blob.get(store_name)
                item = store.get(frame_idx) if isinstance(store, dict) else None
                if isinstance(item, dict):
                    score = _as_float(item.get("object_score_logits"))
                    iou = _as_float(item.get("iou_predictions"))
                    if score is not None or iou is not None:
                        return SamScores(score, iou)
    return SamScores()


def drop_memory_frame(
    state: dict | None,
    frame_idx: int,
    object_id: int | None = None,
    *,
    conditioning: bool = False,
) -> bool:
    """Remove one object's rejected output from SAM memory.

    Non-conditioning outputs are always removed. With `conditioning=True`
    (an *automatic* prompt on that frame was rejected) the conditioning
    output and its stored point/mask inputs are removed too, except when it
    is the object's last conditioning frame, which SAM needs to run at all.
    User prompts must never be passed with `conditioning=True`.
    """
    if not isinstance(state, dict):
        return False
    removed = False

    def _drop(store: Any) -> None:
        nonlocal removed
        if not isinstance(store, dict):
            return
        for key in ("non_cond_frame_outputs",):
            outs = store.get(key)
            if isinstance(outs, dict) and frame_idx in outs:
                outs.pop(frame_idx, None)
                removed = True
        for value in list(store.values()):
            if isinstance(value, dict):
                _drop(value)

    def _drop_cond(store: Any) -> None:
        nonlocal removed
        if not isinstance(store, dict):
            return
        outs = store.get("cond_frame_outputs")
        if isinstance(outs, dict) and frame_idx in outs and len(outs) > 1:
            outs.pop(frame_idx, None)
            removed = True

    mapping = state.get("obj_id_to_idx")
    obj_idx = mapping.get(object_id) if isinstance(mapping, dict) and object_id is not None else None
    per_object_found = False
    targets: list[Any] = []
    for key in ("output_dict_per_obj", "temp_output_dict_per_obj"):
        store = state.get(key)
        if not isinstance(store, dict):
            continue
        if obj_idx is not None:
            targets.append((key, store.get(obj_idx)))
            per_object_found = True
        elif len(store) == 1:
            targets.append((key, next(iter(store.values()))))
            per_object_found = True
    for _key, blob in targets:
        _drop(blob)
    if conditioning:
        output_blob = next((blob for key, blob in targets if key == "output_dict_per_obj"), None)
        cond_before = (
            len(output_blob.get("cond_frame_outputs", {}))
            if isinstance(output_blob, dict)
            else 0
        )
        if cond_before > 1:
            for _key, blob in targets:
                if isinstance(blob, dict):
                    temp = blob.get("cond_frame_outputs")
                    if _key == "temp_output_dict_per_obj" and isinstance(temp, dict):
                        if temp.pop(frame_idx, None) is not None:
                            removed = True
                    else:
                        _drop_cond(blob)
            for key in ("point_inputs_per_obj", "mask_inputs_per_obj"):
                inputs = state.get(key)
                if isinstance(inputs, dict):
                    slot = inputs.get(obj_idx) if obj_idx is not None else (
                        next(iter(inputs.values())) if len(inputs) == 1 else None
                    )
                    if isinstance(slot, dict) and slot.pop(frame_idx, None) is not None:
                        removed = True
    if not per_object_found or (obj_idx is None and not state.get("obj_ids")):
        _drop(state.get("output_dict"))
        if conditioning:
            _drop_cond(state.get("output_dict"))
    tracked = state.get("frames_tracked_per_obj")
    if isinstance(tracked, dict):
        tracked_targets = [tracked.get(obj_idx)] if obj_idx is not None else list(tracked.values())
        for obj in tracked_targets:
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
