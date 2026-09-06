"""Nice 1-2-5 axis ticks for chart headers. No Qt dependency."""

from __future__ import annotations

import math


def _nice_num(span: float, round_val: bool) -> float:
    span = max(abs(span), 1e-18)
    exp = math.floor(math.log10(span))
    frac = span / (10**exp)
    if round_val:
        if frac < 1.5:
            nice = 1.0
        elif frac < 3.0:
            nice = 2.0
        elif frac < 7.0:
            nice = 5.0
        else:
            nice = 10.0
    elif frac <= 1.0:
        nice = 1.0
    elif frac <= 2.0:
        nice = 2.0
    elif frac <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10**exp)


def _label_format(step: float) -> str:
    if step >= 1.0 - 1e-12:
        return "%.0f"
    decimals = max(0, min(6, -math.floor(math.log10(step) + 1e-12)))
    return f"%.{decimals}f"


def nice_tick_interval(lo: float, hi: float, target: int = 5) -> tuple[float, str]:
    """Return a 1-2-5 tick interval and printf format for [lo, hi]."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-15:
        hi = lo + 1.0
    target = max(2, min(int(target), 12))
    step = _nice_num((hi - lo) / (target - 1), round_val=True)
    if step <= 0:
        step = 1.0
    return step, _label_format(step)


def nice_axis_ticks(
    lo: float, hi: float, target: int = 5
) -> tuple[float, float, float, str]:
    """Snap [lo, hi] to a 1-2-5 range. Prefer integer labels when the step ≥ 1.

    Returns (nice_min, nice_max, interval, label_format).
    """
    step, fmt = nice_tick_interval(lo, hi, target)
    if not math.isfinite(lo) or not math.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    if end <= start:
        end = start + step
    start = round(start / step) * step
    end = round(end / step) * step
    return start, end, step, fmt
