"""Projectile with quadratic air drag, fitted to a whole flight.

    dv/dt = G − k·|v|·v,   G = (gx, gy) constant

Seven parameters: start position, start velocity, the gravity vector (free
direction, so a tilted camera or a rotated axis does not bias it) and the
drag coefficient k ≥ 0 (1 / length unit). k = 0 is the textbook parabola.

A light foam dart or a shuttlecock loses a large part of its speed to air:
its measured vertical acceleration is then genuinely more negative than −g
on the way up and less negative on the way down. A constant-acceleration
fit calls that noise; this model separates g from the drag and gives both
with error bars.

Gauss–Newton / Levenberg–Marquardt with forward-difference Jacobians; the
ODE is integrated by fixed-step RK4 for all parameter variants at once.
Robust (Huber) reweighting keeps a few bad frames from pulling the fit.
Numpy only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

MIN_POINTS = 10
HUBER_K = 2.0
MAX_ITER = 30
MAX_FIT_POINTS = 400  # long, dense runs are thinned for the fit (a flight is short anyway)
MAX_GRID_STEPS = 400
MAX_SPAN_S = 30.0
_CACHE: dict[bytes, Any] = {}
_CACHE_ORDER: list[bytes] = []
_CACHE_SIZE = 16


@dataclass(frozen=True)
class DragFit:
    t_ref: float
    t0: float
    t1: float
    params: tuple[float, ...]  # x0, y0, vx0, vy0, gx, gy, k at t_ref
    stderr: tuple[float, ...]
    rms: float  # residual RMS in position units
    n: int
    grid_t: np.ndarray
    grid_state: np.ndarray  # (m, 4): x, y, vx, vy
    cov: np.ndarray | None = None  # (7, 7) parameter covariance

    @property
    def gravity(self) -> float:
        return float(np.hypot(self.params[4], self.params[5]))

    @property
    def k(self) -> float:
        return float(self.params[6])

    @property
    def k_sigma(self) -> float:
        return float(self.stderr[6])

    @property
    def gravity_sigma(self) -> float:
        gx, gy = self.params[4], self.params[5]
        g = max(self.gravity, 1e-12)
        sx, sy = self.stderr[4], self.stderr[5]
        return float(np.hypot(gx / g * sx, gy / g * sy))

    @property
    def tilt_deg(self) -> float:
        """Angle of G from straight down the vertical axis (either image or world)."""
        gx, gy = self.params[4], self.params[5]
        down = -1.0 if gy <= 0 else 1.0
        return float(np.degrees(np.arctan2(gx, down * gy)))

    def state(self, time_s: float | np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(time_s, dtype=np.float64))
        return _hermite(self.grid_t, self.grid_state, self._rates(), t)

    def _rates(self) -> np.ndarray:
        return _rhs(self.grid_state[None, :, :], np.asarray(self.params)[None, :])[0]

    def series(self, attr: str, times: Any) -> np.ndarray:
        """`attr` at many times at once."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        return _attr_values(self.grid_t, self.grid_state, np.asarray(self.params), attr, t)

    def sigma_series(self, attr: str, times: Any) -> np.ndarray | None:
        """Standard error of `attr` from the parameter covariance (delta method)."""
        if self.cov is None:
            return None
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        p = np.asarray(self.params, dtype=np.float64)
        scale = np.sqrt(np.clip(np.diag(self.cov), 0.0, None))
        steps = np.where(scale > 0, 1e-3 * scale, 1e-9)
        variants = np.repeat(p[None], 8, axis=0)
        for j in range(7):
            variants[j + 1, j] += steps[j]
        states = _integrate(variants, self.grid_t)
        base = _attr_values(self.grid_t, states[0], variants[0], attr, t)
        jac = np.stack(
            [(_attr_values(self.grid_t, states[j + 1], variants[j + 1], attr, t) - base) / steps[j] for j in range(7)],
            axis=1,
        )
        var = np.einsum("ni,ij,nj->n", jac, self.cov, jac)
        return np.sqrt(np.clip(var, 0.0, None))

    def evaluate(self, attr: str, time_s: float) -> float:
        s = self.state(time_s)[0]
        gx, gy, k = self.params[4], self.params[5], self.params[6]
        speed = float(np.hypot(s[2], s[3]))
        if attr == "x":
            return float(s[0])
        if attr == "y":
            return float(s[1])
        if attr == "vx":
            return float(s[2])
        if attr == "vy":
            return float(s[3])
        if attr == "speed":
            return speed
        if attr == "ax":
            return float(gx - k * speed * s[2])
        if attr == "ay":
            return float(gy - k * speed * s[3])
        raise KeyError(attr)


def _attr_values(grid: np.ndarray, states: np.ndarray, params: np.ndarray, attr: str, t: np.ndarray) -> np.ndarray:
    rates = _rhs(states[None], params[None])[0]
    s = _hermite(grid, states, rates, t)
    speed = np.hypot(s[:, 2], s[:, 3])
    gx, gy, k = params[4], params[5], params[6]
    table = {
        "x": lambda: s[:, 0],
        "y": lambda: s[:, 1],
        "vx": lambda: s[:, 2],
        "vy": lambda: s[:, 3],
        "speed": lambda: speed,
        "ax": lambda: gx - k * speed * s[:, 2],
        "ay": lambda: gy - k * speed * s[:, 3],
    }
    if attr not in table:
        raise KeyError(attr)
    return table[attr]()


def _rhs(state: np.ndarray, params: np.ndarray) -> np.ndarray:
    """state (P, m, 4) or (P, 4); params (P, 7)."""
    vx = state[..., 2]
    vy = state[..., 3]
    speed = np.hypot(vx, vy)
    extra = (slice(None),) + (None,) * (state.ndim - 2)
    gx = params[:, 4][extra]
    gy = params[:, 5][extra]
    k = params[:, 6][extra]
    return np.stack([vx, vy, gx - k * speed * vx, gy - k * speed * vy], axis=-1)


def _integrate(params: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """RK4 on a fixed grid for every parameter row at once -> (P, m, 4)."""
    p = params.shape[0]
    out = np.empty((p, len(grid), 4))
    s = params[:, :4].copy()
    out[:, 0] = s
    for i in range(1, len(grid)):
        h = grid[i] - grid[i - 1]
        k1 = _rhs(s, params)
        k2 = _rhs(s + 0.5 * h * k1, params)
        k3 = _rhs(s + 0.5 * h * k2, params)
        k4 = _rhs(s + h * k3, params)
        s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[:, i] = s
    return out


def _hermite(grid: np.ndarray, states: np.ndarray, rates: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cubic Hermite interpolation of (m, 4) states with their time derivatives."""
    t = np.clip(t, grid[0], grid[-1])
    i = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    h = grid[i + 1] - grid[i]
    u = ((t - grid[i]) / np.where(h > 0, h, 1.0))[:, None]
    h = h[:, None]
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00 * states[i] + h10 * h * rates[i] + h01 * states[i + 1] + h11 * h * rates[i + 1]


def _positions(params: np.ndarray, grid: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(P, n, 2) model positions at the sample times."""
    states = _integrate(params, grid)
    out = np.empty((params.shape[0], len(t), 2))
    for j in range(params.shape[0]):
        rates = _rhs(states[j][None], params[j][None])[0]
        out[j] = _hermite(grid, states[j], rates, t)[:, :2]
    return out


def _grid(t: np.ndarray) -> np.ndarray:
    span = float(t[-1] - t[0])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else span
    step = max(min(max(dt, 1e-6), span / 40.0), span / MAX_GRID_STEPS)
    count = int(np.ceil(span / step)) + 1
    return np.linspace(t[0], t[-1], max(count, 3))


def _initial(t: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    tau = t - t[0]
    cx = np.polyfit(tau, x, 2, w=np.sqrt(w))
    cy = np.polyfit(tau, y, 2, w=np.sqrt(w))
    return np.array([cx[2], cy[2], cx[1], cy[1], 2 * cx[0], 2 * cy[0], 0.0])


def _key(t: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for arr in (t, x, y, w):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.digest()


def fit_drag(
    t: Any, x: Any, y: Any, weights: Any = None
) -> DragFit | None:
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(t)) if weights is None else np.clip(np.asarray(weights, dtype=np.float64), 1e-3, None)
    ok = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    t, x, y, w = t[ok], x[ok], y[ok], w[ok]
    order = np.argsort(t, kind="stable")
    t, x, y, w = t[order], x[order], y[order], w[order]
    if len(t) < MIN_POINTS or float(t[-1] - t[0]) <= 1e-6 or float(t[-1] - t[0]) > MAX_SPAN_S:
        return None
    if len(t) > MAX_FIT_POINTS:
        keep = np.unique(np.linspace(0, len(t) - 1, MAX_FIT_POINTS).round().astype(int))
        t, x, y, w = t[keep], x[keep], y[keep], w[keep]
    key = _key(t, x, y, w)
    if key in _CACHE:
        return _CACHE[key]
    fit = _fit(t, x, y, w)
    _CACHE[key] = fit
    _CACHE_ORDER.append(key)
    while len(_CACHE_ORDER) > _CACHE_SIZE:
        _CACHE.pop(_CACHE_ORDER.pop(0), None)
    return fit


def _fit(t: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> DragFit | None:
    n = len(t)
    grid = _grid(t)
    obs = np.column_stack([x, y])
    span = float(t[-1] - t[0])
    length = max(float(np.ptp(x)), float(np.ptp(y)), 1e-9)
    typical = np.array(
        [length, length, length / span, length / span, length / span**2, length / span**2, 1.0 / length]
    )
    params = _initial(t, x, y, w)
    robust = np.ones(n)
    lam = 1e-3

    def residuals(p_rows: np.ndarray) -> np.ndarray:
        return (_positions(p_rows, grid, t) - obs[None]).reshape(p_rows.shape[0], -1)

    def cost(r: np.ndarray, weights_2d: np.ndarray) -> float:
        return float(np.sum(weights_2d * r * r))

    jac = None
    wt2 = np.repeat(w * robust, 2)
    r0 = residuals(params[None])[0]
    current = cost(r0, wt2)
    for iteration in range(MAX_ITER):
        steps = 1e-6 * np.maximum(np.abs(params), typical)
        variants = np.repeat(params[None], 8, axis=0)
        for j in range(7):
            variants[j + 1, j] += steps[j]
        rs = residuals(variants)
        r0 = rs[0]
        jac = ((rs[1:] - r0[None]) / steps[:, None]).T  # (2n, 7)
        if iteration % 4 == 0:
            norm = np.hypot(r0[0::2], r0[1::2])
            scale = float(np.median(norm)) / 1.1774 + 1e-12
            robust = np.minimum(1.0, HUBER_K * scale / np.maximum(norm, 1e-12))
            wt2 = np.repeat(w * robust, 2)
            current = cost(r0, wt2)
        jw = jac * wt2[:, None]
        normal = jac.T @ jw
        grad = jw.T @ r0
        improved = False
        for _ in range(8):
            damped = normal + lam * np.diag(np.diag(normal) + 1e-12)
            try:
                delta = -np.linalg.solve(damped, grad)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            trial = params + delta
            trial[6] = max(trial[6], 0.0)
            rt = residuals(trial[None])[0]
            ct = cost(rt, wt2)
            if np.isfinite(ct) and ct < current:
                rel = (current - ct) / max(current, 1e-300)
                params, current = trial, ct
                lam = max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved or rel < 1e-10:
            break
    if jac is None:
        return None
    r = residuals(params[None])[0]
    norm = np.hypot(r[0::2], r[1::2])
    dof = max(2 * n - 7, 1)
    s2 = float(np.sum(wt2 * r * r) / dof)
    try:
        cov = np.linalg.inv(jac.T @ (jac * wt2[:, None])) * s2
    except np.linalg.LinAlgError:
        cov = np.full((7, 7), np.nan)
    # Serially correlated residuals (the tracked point sliding along the
    # object for a few frames) carry less information than their count.
    rho = []
    for comp in (r[0::2], r[1::2]):
        c = comp - comp.mean()
        d = float(np.dot(c, c))
        rho.append(float(np.dot(c[1:], c[:-1]) / d) if d > 0 else 0.0)
    rho_m = min(max(float(np.mean(rho)), 0.0), 0.9)
    cov = cov * (1.0 + rho_m) / (1.0 - rho_m)
    stderr = tuple(float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan") for v in np.diag(cov))
    states = _integrate(params[None], grid)[0]
    return DragFit(
        t_ref=float(t[0]),
        t0=float(t[0]),
        t1=float(t[-1]),
        params=tuple(float(v) for v in params),
        stderr=stderr,
        rms=float(np.sqrt(np.mean(norm**2))),
        n=n,
        grid_t=grid,
        grid_state=states,
        cov=cov if np.all(np.isfinite(cov)) else None,
    )
