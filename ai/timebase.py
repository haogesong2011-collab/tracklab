"""Frame timestamps for measurement.

Phone videos store presentation times in coarse ticks (iPhone: 1/600 s), so
a steady 29 fps capture comes out as 33.3 / 35.0 ms steps. `pts_ms` then
rounds once more to whole milliseconds. Differencing those numbers adds a
fake ±1 ms jitter to every interval, which the second derivative amplifies.

`frame_times` returns exact seconds from the container timestamps and, when
every timestamp is within half a tick of one straight line (dropped frames
allowed), the ideal uniform grid those ticks were rounded from. Truly
variable-rate footage fails that test and keeps its real timestamps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_CACHE: list[tuple[Any, np.ndarray, bool]] = []
_CACHE_SIZE = 4


def _raw_seconds(info: Any) -> tuple[np.ndarray, float] | None:
    """Seconds from frame 0 plus the timestamp resolution in seconds."""
    pts_ms = tuple(getattr(info, "pts_ms", ()) or ())
    pts = tuple(getattr(info, "pts", ()) or ())
    time_base = float(getattr(info, "time_base", 0.0) or 0.0)
    if len(pts) == len(pts_ms) and pts and time_base > 0:
        exact = (np.asarray(pts, dtype=np.float64) - float(pts[0])) * time_base
        # Only trust pts when it agrees with the millisecond index everyone
        # else uses (hand-built VideoInfo objects in tests may not).
        if np.all(np.abs(exact * 1000.0 - np.asarray(pts_ms, dtype=np.float64) + pts_ms[0]) <= 0.5 + 1e-6):
            return exact + pts_ms[0] / 1000.0, time_base
    if pts_ms:
        return np.asarray(pts_ms, dtype=np.float64) / 1000.0, 0.001
    return None


def _uniform_grid(raw: np.ndarray, tick: float) -> np.ndarray | None:
    n = len(raw)
    if n < 3:
        return None
    steps = np.diff(raw)
    positive = steps[steps > 0]
    if len(positive) < 2 or np.any(steps <= 0):
        return None
    median = float(np.median(positive))
    # Rounded ticks alternate (20/21/21 ...): average the ordinary steps,
    # leaving out dropped-frame gaps, to get the real period. When the
    # ticks are nearly as long as a frame (240 fps in 1/600 s) the steps
    # are 2/3 ticks and that filter fails, so also try the overall mean.
    ordinary = positive[np.abs(positive - median) <= 0.25 * median]
    candidates = [float(np.mean(ordinary)) if len(ordinary) else median]
    candidates.append(float(raw[-1] - raw[0]) / (n - 1))
    for period in candidates:
        if period < 2.0 * tick:
            continue  # ticks are not coarse relative to the frame period
        slots = np.rint((raw - raw[0]) / period)
        if np.any(np.diff(slots) < 1):
            continue
        design = np.column_stack([np.ones(n), slots])
        coef, *_ = np.linalg.lstsq(design, raw, rcond=None)
        fitted = design @ coef
        # A least-squares line is not the minimax line: allow a little over half a tick.
        if float(np.max(np.abs(raw - fitted))) > 0.6 * tick + 1e-9:
            continue
        # Anchor at the first frame so displayed times stay where they were.
        return fitted - fitted[0] + raw[0]
    return None


def frame_times(info: Any) -> np.ndarray | None:
    """Seconds per frame index (de-jittered when the video allows it)."""
    if info is None:
        return None
    for cached_info, times, _uniform in _CACHE:
        if cached_info is info:
            return times
    raw = _raw_seconds(info)
    if raw is None:
        return None
    seconds, tick = raw
    grid = _uniform_grid(seconds, tick)
    times = seconds if grid is None else grid
    times.setflags(write=False)
    _CACHE.insert(0, (info, times, grid is not None))
    del _CACHE[_CACHE_SIZE:]
    return times


def is_dejittered(info: Any) -> bool:
    if frame_times(info) is None:
        return False
    for cached_info, _times, uniform in _CACHE:
        if cached_info is info:
            return uniform
    return False


def time_s(info: Any, frame: int) -> float:
    times = frame_times(info)
    if times is not None and 0 <= frame < len(times):
        return float(times[frame])
    fps = 30.0
    if info is not None:
        try:
            fps = max(float(info.fps), 1e-6)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            fps = 30.0
    return frame / fps
