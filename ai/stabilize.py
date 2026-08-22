"""Camera-shake compensation from four-corner reference points.

There is no YOLO weight in this repo. Corner picking uses a simplified
YOLO-style grid (S×S cells, objectness = gradient energy) inside each
corner ROI, then coarse-to-fine NCC template tracking (about 190 px
per-frame jumps). Outlier anchors are dropped so a lost corner cannot
swap identity with another. Offsets are subtracted from analysis
coordinates only; SAM points stay in raw video space.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ai.contracts import CancelToken, ProgressCb, ProgressEvent, TrackPoint, TrackResult
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo

GRID = 8
POINTS_PER_CORNER = 2
PATCH = 15
COARSE_SCALE = 3
NEAR_SEARCH = 20
MID_SEARCH = 48
COARSE_SEARCH = 64
FINE_SEARCH = 8
NCC_MIN = 0.42
CORNER_FRAC = 0.22


@dataclass(frozen=True)
class ShakeCompensation:
    dx: tuple[float, ...]
    dy: tuple[float, ...]
    anchors: tuple[tuple[tuple[float, float] | None, ...], ...]
    model_name: str = "corner_grid_yolo"
    model_version: str = "0.1.0"

    def offset(self, frame: int) -> tuple[float, float]:
        if not self.dx:
            return 0.0, 0.0
        index = max(0, min(int(frame), len(self.dx) - 1))
        return self.dx[index], self.dy[index]

    def anchors_at(self, frame: int) -> list[tuple[float, float]]:
        if not self.anchors:
            return []
        index = max(0, min(int(frame), len(self.anchors) - 1))
        return [pt for pt in self.anchors[index] if pt is not None]


def _gray(frame: np.ndarray) -> np.ndarray:
    rgb = frame.astype(np.float32)
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _energy(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    return gx * gx + gy * gy


def _corner_rois(width: int, height: int) -> list[tuple[int, int, int, int]]:
    mx = max(24, int(width * CORNER_FRAC))
    my = max(24, int(height * CORNER_FRAC))
    return [
        (0, 0, mx, my),
        (width - mx, 0, width, my),
        (0, height - my, mx, height),
        (width - mx, height - my, width, height),
    ]


def detect_anchor_points(
    frame: np.ndarray, *, per_corner: int = POINTS_PER_CORNER
) -> list[tuple[float, float]]:
    """Simplified YOLO-style grid detector in the four image corners."""
    gray = _gray(frame)
    energy = _energy(gray)
    height, width = gray.shape
    pad = PATCH // 2 + 1
    picked: list[tuple[float, float]] = []
    for x0, y0, x1, y1 in _corner_rois(width, height):
        roi = energy[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        peak = float(roi.max())
        if peak < 1e-6:
            continue
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
        local_picked: list[tuple[float, float]] = []
        for _score, x, y in cells:
            if all((x - px) ** 2 + (y - py) ** 2 >= 64 for px, py in local_picked):
                local_picked.append((float(x), float(y)))
            if len(local_picked) >= per_corner:
                break
        picked.extend(local_picked)
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


def _down(gray: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return gray
    height, width = gray.shape
    nh, nw = height // scale, width // scale
    if nh < 1 or nw < 1:
        return gray
    cropped = gray[: nh * scale, : nw * scale]
    return cropped.reshape(nh, scale, nw, scale).mean(axis=(1, 3))


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
    score = float(ncc[loc])
    bx = x0 + int(loc[1]) + r_x
    by = y0 + int(loc[0]) + r_y
    return float(bx), float(by), score


def _track_template(
    gray: np.ndarray, template: np.ndarray, x: float, y: float
) -> tuple[float, float, float]:
    """Expanding NCC: nearby first, then ~48 px, then ~192 px coarse-to-fine."""
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
        sx, sy, _score = _ncc_at(
            coarse, tmpl, x / scale, y / scale, COARSE_SEARCH
        )
        x, y = sx * scale, sy * scale
    return _ncc_at(gray, template, x, y, FINE_SEARCH)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _robust_shift(
    pairs: list[tuple[float, float]], prev_dx: float, prev_dy: float
) -> tuple[float, float]:
    """Largest consistent cluster; ties go to the shift closer to the previous frame."""
    if not pairs:
        return prev_dx, prev_dy
    if len(pairs) == 1:
        return pairs[0]
    thresh_sq = 24.0 * 24.0
    best: list[tuple[float, float]] = []
    best_dist = float("inf")
    for cx, cy in pairs:
        group = [
            p
            for p in pairs
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 <= thresh_sq
        ]
        gx = _median([p[0] for p in group])
        gy = _median([p[1] for p in group])
        dist = (gx - prev_dx) ** 2 + (gy - prev_dy) ** 2
        if len(group) > len(best) or (len(group) == len(best) and dist < best_dist):
            best = group
            best_dist = dist
    return _median([p[0] for p in best]), _median([p[1] for p in best])


def estimate_shake_from_frames(
    frames: Iterable[np.ndarray],
    *,
    cancel: CancelToken | None = None,
    progress: ProgressCb | None = None,
    clip_id: str = "shake",
    total: int | None = None,
) -> ShakeCompensation:
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        return ShakeCompensation(dx=(), dy=(), anchors=())
    seeds = detect_anchor_points(first)
    gray0 = _gray(first)
    templates = [_extract(gray0, x, y) for x, y in seeds]
    valid = [
        (seed, tmpl)
        for seed, tmpl in zip(seeds, templates)
        if tmpl is not None
    ]
    if len(valid) < 2:
        n = 1 + sum(1 for _ in iterator)
        zeros = tuple(0.0 for _ in range(n))
        empty = tuple(tuple(None for _ in seeds) for _ in range(n))
        return ShakeCompensation(dx=zeros, dy=zeros, anchors=empty)

    seeds = [item[0] for item in valid]
    templates = [item[1] for item in valid]
    tracks: list[list[tuple[float, float] | None]] = [list(seeds)]
    last_dx = last_dy = 0.0
    expected = total or 0
    if progress is not None:
        progress(
            ProgressEvent(
                clip_id=clip_id, current=1, total=max(expected, 1), stage="shake"
            )
        )

    for index, frame in enumerate(iterator, start=1):
        if cancel is not None and cancel.cancelled:
            break
        gray = _gray(frame)
        raw: list[tuple[float, float] | None] = []
        for k, seed in enumerate(seeds):
            pred_x = seed[0] + last_dx
            pred_y = seed[1] + last_dy
            nx, ny, score = _track_template(gray, templates[k], pred_x, pred_y)
            raw.append((nx, ny) if score >= NCC_MIN else None)
        pairs = [
            (raw[k][0] - seeds[k][0], raw[k][1] - seeds[k][1])
            for k in range(len(seeds))
            if raw[k] is not None
        ]
        last_dx, last_dy = _robust_shift(pairs, last_dx, last_dy)
        nxt: list[tuple[float, float] | None] = []
        for k, seed in enumerate(seeds):
            point = raw[k]
            if point is None:
                nxt.append(None)
                continue
            err = (point[0] - seed[0] - last_dx) ** 2 + (point[1] - seed[1] - last_dy) ** 2
            nxt.append(None if err > 24.0 * 24.0 else point)
        tracks.append(nxt)
        if progress is not None:
            progress(
                ProgressEvent(
                    clip_id=clip_id,
                    current=index + 1,
                    total=max(expected, index + 1),
                    stage="shake",
                )
            )

    ref = tracks[0]
    dx_list: list[float] = []
    dy_list: list[float] = []
    last_dx = last_dy = 0.0
    for pts in tracks:
        pairs = [
            (pts[k][0] - ref[k][0], pts[k][1] - ref[k][1])
            for k in range(len(ref))
            if pts[k] is not None and ref[k] is not None
        ]
        last_dx, last_dy = _robust_shift(pairs, last_dx, last_dy)
        dx_list.append(last_dx)
        dy_list.append(last_dy)

    packed = tuple(tuple(row) for row in tracks)
    return ShakeCompensation(dx=tuple(dx_list), dy=tuple(dy_list), anchors=packed)


def estimate_shake(
    info: VideoInfo,
    *,
    cancel: CancelToken | None = None,
    progress: ProgressCb | None = None,
) -> ShakeCompensation:
    decoder = FrameDecoder(info)
    try:

        def _frames() -> Iterable[np.ndarray]:
            for index in range(info.frame_count):
                yield decoder.frame(index)

        return estimate_shake_from_frames(
            _frames(),
            cancel=cancel,
            progress=progress,
            clip_id=info.path.name,
            total=info.frame_count,
        )
    finally:
        decoder.close()


def compensate_result(
    result: TrackResult, shake: ShakeCompensation | None
) -> TrackResult:
    if shake is None or not shake.dx:
        return result
    points = []
    for point in result.points:
        dx, dy = shake.offset(point.frame)
        points.append(
            TrackPoint(
                frame=point.frame,
                x=point.x - dx,
                y=point.y - dy,
                visible=point.visible,
                confidence=point.confidence,
                manual=point.manual,
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
