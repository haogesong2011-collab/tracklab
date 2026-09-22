"""Per-frame kinematics derived from TrackResult + VideoInfo PTS.

Display-only: never written into project JSON or SAM TrackResult.
Uncalibrated units are image-projection px / px/s; after a ruler they
switch to m / m/s. Velocity is differenced in the already-transformed
display frame.

Default analysis frame follows Tracker: +x right, +y up, so falling vy < 0.

The default velocity uses a robust local quadratic in real PTS, so 30/60/120
fps share the same physical window. Tracker-style central differences remain
available as a compatibility mode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ai.calibration import CalibrationMode, CalibrationState
from ai.contracts import LOW_CONFIDENCE, TrackPoint, TrackResult
from ai.depth_audit import DepthAuditState
from engine.video_index import VideoInfo

DEFAULT_VELOCITY_STEP = 3
MAX_VELOCITY_STEP = 10
VELOCITY_MODE_LOCAL = "local"
VELOCITY_MODE_TRACKER = "tracker"
DEFAULT_VELOCITY_MODE = VELOCITY_MODE_LOCAL
LOCAL_HALF_WINDOW_S = 0.10
HUBER_C = 1.345


@dataclass(frozen=True)
class KinematicSample:
    frame: int
    time_s: float
    x: float | None
    y: float | None
    vx: float | None
    vy: float | None
    speed: float | None
    visible: bool
    confidence: float
    manual: bool
    position_unit: str = "px"
    speed_unit: str = "px/s"
    sigma_x: float | None = None
    sigma_y: float | None = None
    quality_flags: tuple[str, ...] = ()
    off_plane_m: float | None = None
    source: str = "pixel"
    model_vx: float | None = None
    model_vy: float | None = None

    @property
    def x_px(self) -> float | None:
        return self.x

    @property
    def y_px(self) -> float | None:
        return self.y

    @property
    def vx_px_s(self) -> float | None:
        return self.vx

    @property
    def vy_px_s(self) -> float | None:
        return self.vy

    @property
    def speed_px_s(self) -> float | None:
        return self.speed


@dataclass(frozen=True)
class QuantityFit:
    degree: int
    coeffs: tuple[float, ...]
    r2: float
    n: int

    def evaluate(self, time_s: float) -> float:
        total = 0.0
        power = 1.0
        for coef in self.coeffs:
            total += coef * power
            power *= time_s
        return total

    def equation(self, name: str) -> str:
        if not self.coeffs:
            return f"{name}  拟合失败"
        parts: list[str] = []
        for power, coef in enumerate(self.coeffs):
            if abs(coef) < 1e-12:
                continue
            token = _format_coef(coef)
            if power == 0:
                parts.append(token)
                continue
            sign = " + " if coef >= 0 and parts else (" − " if coef < 0 and parts else "")
            mag = _format_coef(abs(coef))
            var = "t" if power == 1 else f"t^{power}"
            if not parts and coef < 0:
                parts.append(f"−{mag} {var}")
            elif mag == "1" and power >= 1:
                parts.append(f"{sign}{var}".strip() if sign else var)
            else:
                parts.append(f"{sign}{mag} {var}".lstrip() if not parts else f"{sign}{mag} {var}")
        body = "".join(parts) if parts else "0"
        return f"{name} = {body}    R²={self.r2:.3f}"


def is_low_confidence(sample: KinematicSample) -> bool:
    if sample.visible and sample.confidence < LOW_CONFIDENCE:
        return True
    return any(flag in sample.quality_flags for flag in ("extrapolated", "off_plane", "ai_estimate"))


def quality_label(sample: KinematicSample) -> str:
    flags = sample.quality_flags
    if "ai_estimate" in flags:
        return "AI估计"
    if "off_plane" in flags:
        return "疑似离面"
    if "extrapolated" in flags:
        return "外推"
    if sample.source == "geometric":
        return "几何测量"
    if sample.source == "scaled":
        return "比例尺"
    return "像素"


def quality_tooltip(sample: KinematicSample) -> str:
    bits = [quality_label(sample)]
    if sample.source and sample.source not in {"pixel", "scaled", "geometric"}:
        bits.append(sample.source)
    if sample.sigma_x is not None and sample.sigma_y is not None and sample.position_unit == "m":
        bits.append(f"σx={sample.sigma_x:.4f} m  σy={sample.sigma_y:.4f} m")
    if sample.off_plane_m is not None:
        bits.append(f"离面残差 {sample.off_plane_m:.3f} m")
    if sample.visible and sample.confidence < LOW_CONFIDENCE:
        bits.append(f"跟踪置信度 {sample.confidence:.2f}")
    return " · ".join(bits)


def time_s_for_frame(info: VideoInfo | None, frame: int) -> float:
    if info is not None and 0 <= frame < len(info.pts_ms):
        return info.pts_ms[frame] / 1000.0
    fps = 30.0 if info is None else max(info.fps, 1e-6)
    return frame / fps


def series_for_result(
    result: TrackResult | None,
    info: VideoInfo | None,
    *,
    calibration: CalibrationState | None = None,
    velocity_step: int = DEFAULT_VELOCITY_STEP,
    velocity_mode: str = DEFAULT_VELOCITY_MODE,
    depth_audit: DepthAuditState | None = None,
) -> list[KinematicSample]:
    if result is None:
        return []
    cal = calibration or CalibrationState()
    position_unit = cal.position_unit
    speed_unit = cal.speed_unit
    step = max(1, min(int(velocity_step), MAX_VELOCITY_STEP))
    mode = velocity_mode if velocity_mode in {VELOCITY_MODE_LOCAL, VELOCITY_MODE_TRACKER} else DEFAULT_VELOCITY_MODE
    points = sorted(result.points, key=lambda p: p.frame)
    display: list[tuple[float | None, float | None]] = [
        _display_xy(point, cal) if point.visible else (None, None) for point in points
    ]
    bounds = _run_bounds(points, display)
    times = [time_s_for_frame(info, point.frame) for point in points]
    samples: list[KinematicSample] = []
    for i, point in enumerate(points):
        time_s = times[i]
        x, y = display[i]
        vx = vy = speed = None
        if point.visible and x is not None and y is not None and bounds[i] is not None:
            if mode == VELOCITY_MODE_TRACKER:
                vx, vy = _velocity_at(i, points, display, bounds[i], info, step)
            else:
                vx, vy = _local_velocity_at(i, times, display, bounds[i], points, step, info)
            if vx is not None and vy is not None:
                speed = (vx * vx + vy * vy) ** 0.5
        flags: list[str] = []
        source = "pixel"
        sigma_x = sigma_y = off_m = None
        if point.visible:
            source = cal.measurement_source(point.x, point.y)
            if cal.active and cal.mode is CalibrationMode.PLANAR:
                sx, sy = cal.position_sigma(point.x, point.y)
                sigma_x, sigma_y = sx, sy
            if cal.is_extrapolated(point.x, point.y):
                flags.append("extrapolated")
            if depth_audit is not None:
                reading = depth_audit.readings.get(point.frame)
                if reading is not None:
                    off_m = reading.residual_m
                    if reading.flag == "off_plane":
                        flags.append("off_plane")
                        if depth_audit.experimental_correction:
                            flags.append("ai_estimate")
        samples.append(
            KinematicSample(
                frame=point.frame,
                time_s=time_s,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                speed=speed,
                visible=point.visible,
                confidence=point.confidence,
                manual=point.manual,
                position_unit=position_unit,
                speed_unit=speed_unit,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                quality_flags=tuple(flags),
                off_plane_m=off_m,
                source=source,
            )
        )
    return samples


def sample_at_frame(
    samples: list[KinematicSample], frame: int
) -> KinematicSample | None:
    for sample in samples:
        if sample.frame == frame:
            return sample
    return None


def contiguous_segments(
    samples: list[KinematicSample],
    attr: str,
) -> list[list[KinematicSample]]:
    """Split into runs where `attr` is not None (breaks on occlusion / missing)."""
    segments: list[list[KinematicSample]] = []
    current: list[KinematicSample] = []
    for sample in samples:
        value = getattr(sample, attr)
        if value is None:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(sample)
    if current:
        segments.append(current)
    return segments


def fit_quantity(
    samples: list[KinematicSample],
    attr: str,
    degree: int,
) -> QuantityFit | None:
    """Least-squares polynomial of `attr` vs time, Tracker Data Tool style."""
    if degree < 1:
        return None
    pairs = [
        (sample.time_s, float(getattr(sample, attr)))
        for sample in samples
        if getattr(sample, attr) is not None
    ]
    if len(pairs) < degree + 2:
        return None
    times = [item[0] for item in pairs]
    values = [item[1] for item in pairs]
    t0 = times[0]
    rows = []
    for time_s in times:
        tau = time_s - t0
        row = [1.0]
        power = tau
        for _ in range(degree):
            row.append(power)
            power *= tau
        rows.append(row)
    try:
        coeffs_tau = _solve_normal(rows, values)
    except ValueError:
        return None
    coeffs = _shift_poly(coeffs_tau, t0)
    ss_res = 0.0
    mean = sum(values) / len(values)
    ss_tot = 0.0
    for time_s, value in pairs:
        pred = _eval_poly(coeffs, time_s)
        err = value - pred
        ss_res += err * err
        ss_tot += (value - mean) * (value - mean)
    if ss_tot < 1e-18:
        r2 = 1.0 if ss_res < 1e-18 else 0.0
    else:
        r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return QuantityFit(degree=degree, coeffs=tuple(coeffs), r2=r2, n=len(pairs))


def _display_xy(point: TrackPoint, calibration: CalibrationState) -> tuple[float, float]:
    if calibration.applies_transform():
        world = calibration.pixel_to_world(point.x, point.y)
        return world.x, world.y
    if calibration.frame.y_up:
        return point.x, -point.y
    return point.x, point.y


def _run_bounds(
    points: list[TrackPoint],
    display: list[tuple[float | None, float | None]],
) -> list[tuple[int, int] | None]:
    bounds: list[tuple[int, int] | None] = [None] * len(points)
    i = 0
    while i < len(points):
        if not _usable(points, display, i):
            i += 1
            continue
        j = i
        while j + 1 < len(points) and _usable(points, display, j + 1):
            j += 1
        for k in range(i, j + 1):
            bounds[k] = (i, j)
        i = j + 1
    return bounds


def _usable(
    points: list[TrackPoint],
    display: list[tuple[float | None, float | None]],
    index: int,
) -> bool:
    point = points[index]
    x, y = display[index]
    return point.visible and x is not None and y is not None


def attach_model_velocity(
    samples: list[KinematicSample],
    *,
    v0x: float,
    v0y: float,
    ay: float,
    time_start_s: float,
    time_end_s: float,
) -> list[KinematicSample]:
    """Overlay projectile model velocity v_x=v0x, v_y=v0y+ay*t without changing measurements."""
    out: list[KinematicSample] = []
    for sample in samples:
        if sample.time_s < time_start_s - 1e-9 or sample.time_s > time_end_s + 1e-9:
            out.append(sample)
            continue
        tau = sample.time_s - time_start_s
        out.append(
            replace(
                sample,
                model_vx=float(v0x),
                model_vy=float(v0y + ay * tau),
            )
        )
    return out


def _local_velocity_at(
    index: int,
    times: list[float],
    display: list[tuple[float | None, float | None]],
    run: tuple[int, int],
    points: list[TrackPoint],
    step: int,
    info: VideoInfo | None,
) -> tuple[float | None, float | None]:
    left, right = run
    t0 = times[index]
    half = LOCAL_HALF_WINDOW_S * (step / DEFAULT_VELOCITY_STEP)
    lo = index
    while lo > left and t0 - times[lo - 1] <= half:
        lo -= 1
    hi = index
    while hi < right and times[hi + 1] - t0 <= half:
        hi += 1
    xs: list[float] = []
    ys: list[float] = []
    ts: list[float] = []
    weights: list[float] = []
    for j in range(lo, hi + 1):
        x, y = display[j]
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        ts.append(times[j])
        weights.append(max(points[j].confidence, 1e-3))
    if len(ts) < 2:
        return _velocity_at(index, points, display, run, info, step)
    degree = 2 if len(ts) >= 4 else 1
    vx = _poly_derivative(ts, xs, weights, t0, degree)
    vy = _poly_derivative(ts, ys, weights, t0, degree)
    return vx, vy


def _poly_derivative(
    times: list[float],
    values: list[float],
    weights: list[float],
    t0: float,
    degree: int,
) -> float | None:
    n = len(times)
    if n < 2:
        return None
    degree = max(1, min(int(degree), n - 1))
    tau = np.asarray(times, dtype=np.float64) - t0
    y = np.asarray(values, dtype=np.float64)
    w = np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)
    if float(np.max(tau) - np.min(tau)) <= 1e-12:
        return None
    design = np.column_stack([tau ** k for k in range(degree + 1)])
    coef = np.zeros(degree + 1)
    for _ in range(4):
        sw = np.sqrt(w)
        try:
            coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        except np.linalg.LinAlgError:
            return None
        resid = y - design @ coef
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scale = 1.4826 * mad + 1e-9
        cutoff = HUBER_C * scale
        w = np.asarray(weights, dtype=np.float64) / np.maximum(1.0, np.abs(resid) / cutoff)
    if len(coef) < 2:
        return 0.0
    return float(coef[1])


def _velocity_at(
    index: int,
    points: list[TrackPoint],
    display: list[tuple[float | None, float | None]],
    run: tuple[int, int],
    info: VideoInfo | None,
    step: int,
) -> tuple[float | None, float | None]:
    left, right = run
    n_left = index - left
    n_right = right - index
    span = min(step, n_left, n_right)
    if span >= 1:
        return _delta(index - span, index + span, points, display, info)
    if n_right >= 1:
        return _delta(index, index + min(step, n_right), points, display, info)
    if n_left >= 1:
        return _delta(index - min(step, n_left), index, points, display, info)
    return None, None


def _delta(
    i0: int,
    i1: int,
    points: list[TrackPoint],
    display: list[tuple[float | None, float | None]],
    info: VideoInfo | None,
) -> tuple[float | None, float | None]:
    x0, y0 = display[i0]
    x1, y1 = display[i1]
    if x0 is None or y0 is None or x1 is None or y1 is None:
        return None, None
    dt = time_s_for_frame(info, points[i1].frame) - time_s_for_frame(info, points[i0].frame)
    if dt <= 0:
        return None, None
    return (x1 - x0) / dt, (y1 - y0) / dt


def _solve_normal(rows: list[list[float]], values: list[float]) -> list[float]:
    n = len(rows[0])
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, value in zip(rows, values):
        for i in range(n):
            atb[i] += row[i] * value
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    return _gauss(ata, atb)


def _gauss(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(matrix[r][col]))
        if abs(matrix[pivot][col]) < 1e-14:
            raise ValueError("singular")
        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        scale = matrix[col][col]
        for j in range(col, n):
            matrix[col][j] /= scale
        rhs[col] /= scale
        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            for j in range(col, n):
                matrix[row][j] -= factor * matrix[col][j]
            rhs[row] -= factor * rhs[col]
    return rhs


def _shift_poly(coeffs_tau: list[float], t0: float) -> list[float]:
    """Convert coefficients of (t - t0) back to powers of t."""
    degree = len(coeffs_tau) - 1
    out = [0.0] * (degree + 1)
    for k, ck in enumerate(coeffs_tau):
        # ck * (t - t0)^k
        binom = 1
        for i in range(k + 1):
            out[k - i] += ck * binom * ((-t0) ** i)
            binom = binom * (k - i) // (i + 1)
    return out


def _eval_poly(coeffs: list[float] | tuple[float, ...], time_s: float) -> float:
    total = 0.0
    power = 1.0
    for coef in coeffs:
        total += coef * power
        power *= time_s
    return total


def _format_coef(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        text = f"{value:.1f}"
    elif magnitude >= 10:
        text = f"{value:.2f}"
    else:
        text = f"{value:.3g}"
    if text.endswith(".0"):
        text = text[:-2]
    return text
