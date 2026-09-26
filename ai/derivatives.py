"""Velocity and acceleration from noisy tracked positions.

Robust local quadratic regression (LOESS-style, tricube kernel) in real time:

* the window is a time span, so 30 / 60 / 240 fps get the same physics;
* near the ends of a run the window stops sliding and the fit of the last
  full window is evaluated at the edge point, so the first and last frames
  do not get a one-sided window with a fraction of the points;
* acceleration uses a wider window than velocity: the second derivative
  amplifies noise far more than the first;
* by default the widths come from the data. Leave-one-out prediction error
  is computed for a ladder of widths; acceleration takes the widest window
  whose error is still within twice the best (a constant-acceleration
  model still fits there), velocity sits between the best width and that;
* frames whose residual is far outside the run's scatter are down-weighted
  (Tukey bisquare), and every value carries a standard error.

Numpy only: this runs inside the desktop app on every edit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTH_LADDER_S = (0.10, 0.13, 0.17, 0.22, 0.30, 0.40, 0.50, 0.60, 0.80)
MIN_WIDTH_FRAMES = 3.0  # half width, in frame periods
MAX_WINDOW_FRAMES = 60.0  # half width cap, in frame periods (240 fps slow motion)
# Acceleration: the widest window whose leave-one-out error is at most twice
# the best one. Tracking errors that persist for a few frames (the point
# sliding along a blurred stick) make the best window look narrower than it
# is; a tighter rule then chases them. Oscillations still stop it early,
# because there the error grows much faster than twofold.
ACCEL_TOLERANCE = 1.0
# Velocity sits three quarters of the way (log scale) from the best
# position window to the acceleration window.
VELOCITY_SHARE = 0.75
ROBUST_ITERATIONS = 2
BISQUARE_C = 4.685


@dataclass(frozen=True)
class RunDerivatives:
    vx: np.ndarray
    vy: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    sigma_vx: np.ndarray
    sigma_vy: np.ndarray
    sigma_ax: np.ndarray
    sigma_ay: np.ndarray
    window_v_s: float
    window_a_s: float
    weights: np.ndarray  # robustness weight per sample (1 = normal)
    position_sigma: tuple[float, float]


def _tricube(u: np.ndarray) -> np.ndarray:
    a = np.clip(1.0 - np.abs(u) ** 3, 0.0, None)
    return a * a * a


def _windows(t: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t0, t1 = float(t[0]), float(t[-1])
    n = len(t)
    if t1 - t0 <= 2.0 * half:
        center = np.full(n, 0.5 * (t0 + t1))
        width = np.full(n, max(0.5 * (t1 - t0), 1e-9) * 1.0001)
    else:
        center = np.clip(t, t0 + half, t1 - half)
        width = np.full(n, half)
    lo = np.searchsorted(t, center - width, side="left")
    hi = np.searchsorted(t, center + width, side="right")
    return center, width, lo, hi


def local_fit(
    t: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    half: float,
    degree: int = 2,
) -> dict[str, np.ndarray]:
    """Weighted local polynomial at every sample of one run.

    `y` is (n, k). `weights` are inverse-variance style (1 = normal frame).
    Returns value, first/second derivative, their variance per unit noise,
    leave-one-out residuals and a validity mask.
    """
    n = len(t)
    center, width, lo, hi = _windows(t, half)
    span = int(np.max(hi - lo)) if n else 0
    offsets = np.arange(max(span, 1))
    idx = lo[:, None] + offsets[None, :]
    inside = idx < hi[:, None]
    idx = np.minimum(idx, n - 1)
    u = (t[idx] - center[:, None]) / width[:, None]
    kernel = _tricube(u) * inside
    w = kernel * weights[idx]
    design = np.stack([u**p for p in range(degree + 1)], axis=-1)
    normal = np.einsum("nl,nlp,nlq->npq", w, design, design)
    count = np.sum(w > 1e-12, axis=1)
    # Distinct times matter, not just points: a zero-length window is singular.
    t_lo = np.where(inside, t[idx], np.inf).min(axis=1)
    t_hi = np.where(inside, t[idx], -np.inf).max(axis=1)
    good = (count >= degree + 1) & (t_hi - t_lo > 1e-9)
    inv = np.zeros_like(normal)
    if np.any(good):
        det = np.linalg.det(normal[good])
        scale = np.max(np.abs(normal[good]), axis=(1, 2)) ** (degree + 1)
        ok = np.abs(det) > 1e-12 * np.maximum(scale, 1e-300)
        sub = np.flatnonzero(good)
        good[sub[~ok]] = False
        if np.any(good):
            inv[good] = np.linalg.inv(normal[good])
    rhs = np.einsum("nl,nlp,nlk->npk", w, design, y[idx])
    coef = np.einsum("npq,nqk->npk", inv, rhs)
    spread = np.einsum("nl,nlp,nlq->npq", w * kernel, design, design)
    cov = np.einsum("npq,nqr,nrs->nps", inv, spread, inv)
    d = (t - center) / width
    zeros = np.zeros(n)
    at = np.stack([d**p for p in range(degree + 1)], axis=-1)
    dv = np.stack([zeros] + [p * d ** (p - 1) for p in range(1, degree + 1)], axis=-1) / width[:, None]
    da = np.stack(
        [zeros, zeros] + [p * (p - 1) * d ** (p - 2) for p in range(2, degree + 1)], axis=-1
    )[:, : degree + 1] / (width**2)[:, None]
    value = np.einsum("np,npk->nk", at, coef)
    vel = np.einsum("np,npk->nk", dv, coef)
    acc = np.einsum("np,npk->nk", da, coef)
    var_v = np.einsum("np,npq,nq->n", dv, cov, dv)
    var_a = np.einsum("np,npq,nq->n", da, cov, da)
    own = np.where(np.abs(d) < 1.0, _tricube(d), 0.0) * weights
    leverage = np.einsum("np,npq,nq->n", at, inv, at) * own
    loo = (y - value) / np.clip(1.0 - leverage, 1e-3, None)[:, None]
    for arr in (value, vel, acc, loo):
        arr[~good] = np.nan
    var_v[~good] = np.nan
    var_a[~good] = np.nan
    return {
        "value": value,
        "v": vel,
        "a": acc,
        "var_v": var_v,
        "var_a": var_a,
        "loo": loo,
        "good": good,
        "count": count,
    }


def _ladder(t: np.ndarray) -> list[float]:
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 30.0
    span = float(t[-1] - t[0])
    lo = max(0.08, MIN_WIDTH_FRAMES * dt)
    hi = min(0.8, max(span / 2.0, 1e-6), MAX_WINDOW_FRAMES * dt)
    ladder = [h for h in WIDTH_LADDER_S if lo * 0.999 <= h <= hi * 1.001]
    if not ladder or ladder[0] > lo * 1.2:
        ladder.insert(0, min(lo, hi))
    if ladder[-1] < hi * 0.95 and hi <= 0.8:
        ladder.append(hi)
    return sorted(set(ladder))


def choose_windows(
    t: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """(velocity half-width, acceleration half-width) in seconds."""
    ladder = _ladder(t)
    errors: list[float] = []
    measured = weights > 0
    for half in ladder:
        fit = local_fit(t, y, weights, half)
        err = np.hypot(*fit["loo"].T) if y.shape[1] == 2 else np.abs(fit["loo"][:, 0])
        err = err[np.isfinite(err) & measured]
        errors.append(float(np.sqrt(np.mean(err**2))) if len(err) else float("inf"))
    cv = np.asarray(errors)
    if not np.any(np.isfinite(cv)):
        half = ladder[-1]
        return half, half
    best_i = int(np.nanargmin(cv))
    best = float(cv[best_i])
    accel = ladder[0]
    for half, err in zip(ladder, cv):
        if err <= (1.0 + ACCEL_TOLERANCE) * best + 1e-12:
            accel = half
        else:
            break
    accel = max(accel, ladder[best_i])
    velocity = float(ladder[best_i] ** (1.0 - VELOCITY_SHARE) * accel**VELOCITY_SHARE)
    return min(velocity, accel), accel


def manual_windows(step: int) -> tuple[float, float]:
    """The chart's 窗口 1–10 as half-widths (3 ≈ ±0.14 s velocity, ±0.35 s acceleration)."""
    velocity = 0.14 * max(1, int(step)) / 3.0
    return velocity, 2.5 * velocity


def _noise_scale(loo: np.ndarray, weights: np.ndarray, robust: np.ndarray) -> tuple[float, float]:
    """Per-axis position noise for the error bars.

    RMS of the leave-one-out residuals (outliers already down-weighted),
    inflated for serial correlation: tracking errors that last several
    frames do not average out the way independent noise does, so the
    effective number of samples is smaller (AR(1) factor (1+ρ)/(1−ρ)).
    """
    out: list[float] = []
    for k in range(loo.shape[1]):
        r = loo[:, k]
        ok = np.isfinite(r) & (robust > 0.5) & (weights > 0)
        if int(np.sum(ok)) < 3:
            out.append(float("nan"))
            continue
        rr = r[ok] * np.sqrt(weights[ok])
        rms = float(np.sqrt(np.mean(rr * rr)))
        centered = rr - float(np.mean(rr))
        denom = float(np.dot(centered, centered))
        rho = float(np.dot(centered[1:], centered[:-1]) / denom) if denom > 0 else 0.0
        rho = min(max(rho, 0.0), 0.9)
        out.append(rms * float(np.sqrt((1.0 + rho) / (1.0 - rho))))
    return out[0], out[1]


def run_derivatives(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    windows: tuple[float, float] | None = None,
) -> RunDerivatives:
    t = np.asarray(t, dtype=np.float64)
    pos = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
    raw = np.asarray(weights, dtype=np.float64)
    # Weight 0 marks a frame that only wants values (a point waiting for
    # review): it is evaluated but never fitted.
    base = np.where(raw > 0, np.clip(raw, 1e-3, None), 0.0)
    measured = base > 0
    n = len(t)
    nan = np.full(n, np.nan)
    if int(np.sum(measured)) < 2 or float(np.ptp(t[measured])) <= 1e-9:
        return RunDerivatives(nan, nan, nan, nan, nan, nan, nan, nan, 0.0, 0.0, np.ones(n), (np.nan, np.nan))
    half_v, half_a = windows if windows is not None else choose_windows(t, pos, base)
    robust = np.ones(n)
    if int(np.sum(measured)) >= 5:
        for _ in range(ROBUST_ITERATIONS):
            fit = local_fit(t, pos, base * robust, half_a)
            err = np.hypot(*fit["loo"].T)
            finite = err[np.isfinite(err) & measured]
            if not len(finite):
                break
            scale = float(np.median(finite)) / 1.1774  # median of a 2-D Gaussian norm
            if scale <= 1e-12:
                break
            u = np.nan_to_num(err / (BISQUARE_C * scale), nan=0.0)
            robust = np.clip(1.0 - u * u, 0.0, None) ** 2
            robust = np.maximum(robust, 1e-3)
    w = base * robust
    quad_v = local_fit(t, pos, w, half_v, degree=2)
    line_v = local_fit(t, pos, w, half_v, degree=1)
    use_line = (quad_v["count"] < 4) | ~quad_v["good"]
    vel = np.where(use_line[:, None], line_v["v"], quad_v["v"])
    var_v = np.where(use_line, line_v["var_v"], quad_v["var_v"])
    acc_fit = local_fit(t, pos, w, half_a, degree=2)
    acc = acc_fit["a"].copy()
    var_a = acc_fit["var_a"].copy()
    too_few = acc_fit["count"] < 4
    acc[too_few] = np.nan
    var_a[too_few] = np.nan
    sx, sy = _noise_scale(acc_fit["loo"], w, robust)
    sd_v = np.sqrt(np.clip(var_v, 0.0, None))
    sd_a = np.sqrt(np.clip(var_a, 0.0, None))
    return RunDerivatives(
        vx=vel[:, 0],
        vy=vel[:, 1],
        ax=acc[:, 0],
        ay=acc[:, 1],
        sigma_vx=sd_v * sx,
        sigma_vy=sd_v * sy,
        sigma_ax=sd_a * sx,
        sigma_ay=sd_a * sy,
        window_v_s=float(half_v),
        window_a_s=float(half_a),
        weights=robust,
        position_sigma=(sx, sy),
    )
