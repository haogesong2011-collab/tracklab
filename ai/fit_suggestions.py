"""Trajectory-fit suggestions for frames the tracker did not trust.

For each review/lost frame that has trusted measurements close by, fit a
robust local quadratic in real time (x(t), y(t)) through those measurements
and propose where the object should be. A suggestion is only a proposal: it
is drawn on the video and counts as a measurement only after the user
accepts it, and then it is recorded as a user-confirmed fit point, never as
an automatic measurement.

Interpolation (trusted points on both sides) is preferred; extrapolation is
allowed for at most MAX_EXTRAPOLATION frames past the last trusted point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ai import timebase
from ai.contracts import TrackPoint, TrackPointSource, TrackPointStatus, TrackResult

SIDE_POINTS = 6
SEARCH_FRAMES = 12
MIN_POINTS = 5
MAX_EXTRAPOLATION = 3


@dataclass(frozen=True)
class FitSuggestion:
    frame: int
    x: float
    y: float
    sigma_px: float  # robust residual of the fit; the honest error bar
    points: int
    method: str  # "interpolate" | "extrapolate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "sigma_px": round(self.sigma_px, 3),
            "points": self.points,
            "method": self.method,
        }


def _time(info: Any, frame: int) -> float:
    # Same clock as the kinematics (container PTS, de-jittered when uniform).
    return timebase.time_s(info, frame)


def _robust_quadratic(ts: np.ndarray, xs: np.ndarray, ys: np.ndarray, t: float):
    ts = ts - t
    degree = 2 if len(ts) >= 4 else 1
    w = np.ones(len(ts))
    sigma = 1.0
    cx = cy = None
    for _ in range(6):
        cx = np.polyfit(ts, xs, degree, w=w)
        cy = np.polyfit(ts, ys, degree, w=w)
        r = np.hypot(xs - np.polyval(cx, ts), ys - np.polyval(cy, ts))
        sigma = max(0.5, 1.4826 * float(np.median(r)))
        w = np.where(r < 2.5 * sigma, 1.0, 2.5 * sigma / np.maximum(r, 1e-6))
    assert cx is not None and cy is not None
    return float(np.polyval(cx, 0.0)), float(np.polyval(cy, 0.0)), sigma


def suggest_points(points: list[TrackPoint], info: Any = None) -> dict[int, FitSuggestion]:
    """Proposals for every untrusted, non-manual frame that can be fitted."""
    ordered = sorted(points, key=lambda p: p.frame)
    trusted = [p for p in ordered if p.usable_for_measurement()]
    if len(trusted) < MIN_POINTS:
        return {}
    frames = np.array([p.frame for p in trusted])
    out: dict[int, FitSuggestion] = {}
    for p in ordered:
        if p.usable_for_measurement() or p.manual:
            continue
        f = p.frame
        before_idx = np.flatnonzero((frames < f) & (frames >= f - SEARCH_FRAMES))[-SIDE_POINTS:]
        after_idx = np.flatnonzero((frames > f) & (frames <= f + SEARCH_FRAMES))[:SIDE_POINTS]
        if len(before_idx) >= 2 and len(after_idx) >= 2:
            use = np.concatenate([before_idx, after_idx])
            method = "interpolate"
        elif len(before_idx) >= MIN_POINTS and f - frames[before_idx[-1]] <= MAX_EXTRAPOLATION:
            use = before_idx
            method = "extrapolate"
        elif len(after_idx) >= MIN_POINTS and frames[after_idx[0]] - f <= MAX_EXTRAPOLATION:
            use = after_idx
            method = "extrapolate"
        else:
            continue
        if len(use) < MIN_POINTS:
            continue
        chosen = [trusted[i] for i in use]
        ts = np.array([_time(info, q.frame) for q in chosen])
        xs = np.array([q.x for q in chosen])
        ys = np.array([q.y for q in chosen])
        x, y, sigma = _robust_quadratic(ts, xs, ys, _time(info, f))
        if method == "extrapolate":
            gap = min(abs(f - q.frame) for q in chosen)
            sigma *= 1.0 + 0.5 * max(0, gap - 1)
        out[f] = FitSuggestion(f, x, y, sigma, len(chosen), method)
    return out


def accept_suggestion(result: TrackResult, suggestion: FitSuggestion) -> TrackResult:
    """Replace one frame with a user-confirmed fit point (kept honest in data)."""
    points = list(result.points)
    original = next((p for p in points if p.frame == suggestion.frame), None)
    diagnostics = {
        "fit_accepted": True,
        "fit": suggestion.to_dict(),
        "original_status": None if original is None else original.status.value,
        "original_xy": None if original is None else [round(original.x, 3), round(original.y, 3)],
        "original_note": None if original is None else original.note,
    }
    replacement = TrackPoint(
        frame=suggestion.frame,
        x=suggestion.x,
        y=suggestion.y,
        visible=True,
        confidence=1.0,
        manual=True,
        note=f"拟合点（用户确认，±{suggestion.sigma_px:.1f}px）",
        status=TrackPointStatus.TRUSTED,
        source=TrackPointSource.MANUAL,
        diagnostics=diagnostics,
    )
    for i, existing in enumerate(points):
        if existing.frame == suggestion.frame:
            points[i] = replacement
            break
    else:
        points.append(replacement)
        points.sort(key=lambda p: p.frame)
    return TrackResult(
        clip_id=result.clip_id,
        points=points,
        confidence=result.confidence,
        failure_reason=result.failure_reason,
        model_name=result.model_name,
        model_version=result.model_version,
        elapsed_s=result.elapsed_s,
        quality_version=result.quality_version,
    )


def accept_all(result: TrackResult, suggestions: dict[int, FitSuggestion]) -> TrackResult:
    for suggestion in sorted(suggestions.values(), key=lambda s: s.frame):
        result = accept_suggestion(result, suggestion)
    return result


FILL_MAX_FRAMES = 6
FILL_MAX_SECONDS = 0.25
FILL_MAX_SIGMA_PX = 6.0


def fill_fit_gaps(
    points: list[TrackPoint],
    info: Any = None,
    *,
    max_frames: int = FILL_MAX_FRAMES,
    max_seconds: float = FILL_MAX_SECONDS,
    max_sigma: float = FILL_MAX_SIGMA_PX,
) -> list[TrackPoint]:
    """Bridge short untrusted runs with the local fit so the track stays whole.

    Only interpolation between trusted measurements on both sides is used,
    only for short runs, and only when the fit itself is tight. Filled frames
    are drawn and listed as visible but marked as interpolated fit points:
    they are not measurements (velocity and acceleration skip them) until
    the user accepts them. The tracker's own candidate and reason stay in the
    diagnostics.
    """
    suggestions = suggest_points(points, info)
    ordered = sorted(points, key=lambda p: p.frame)
    out = list(ordered)
    n = len(ordered)
    i = 0
    while i < n:
        p = ordered[i]
        if p.usable_for_measurement() or p.manual:
            i += 1
            continue
        j = i
        while j < n and not ordered[j].usable_for_measurement() and not ordered[j].manual:
            j += 1
        run = ordered[i:j]
        i = j
        if not run or len(run) > max_frames:
            continue
        span = _time(info, run[-1].frame) - _time(info, run[0].frame)
        if span > max_seconds + 1e-9:
            continue
        fits = [suggestions.get(q.frame) for q in run]
        if any(f is None or f.method != "interpolate" or f.sigma_px > max_sigma for f in fits):
            continue
        for q, fit in zip(run, fits, strict=True):
            assert fit is not None
            index = out.index(q)
            diagnostics = dict(q.diagnostics)
            diagnostics.update(
                {
                    "fit_filled": True,
                    "fit": fit.to_dict(),
                    "candidate_xy": None if q.status is TrackPointStatus.LOST else [round(q.x, 3), round(q.y, 3)],
                    "original_status": q.status.value,
                    "original_note": q.note,
                }
            )
            out[index] = TrackPoint(
                frame=q.frame,
                x=fit.x,
                y=fit.y,
                visible=True,
                confidence=0.5,
                interpolated=True,
                note=f"拟合补点（±{fit.sigma_px:.1f}px）" + (f"：{q.note}" if q.note else ""),
                status=TrackPointStatus.REVIEW,
                source=TrackPointSource.INTERPOLATED,
                diagnostics=diagnostics,
            )
    return out
