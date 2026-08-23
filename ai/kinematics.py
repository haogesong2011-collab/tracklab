"""Per-frame kinematics derived from TrackResult + VideoInfo PTS.

Display-only: never written into project JSON or SAM TrackResult.
Uncalibrated units are px / px/s; after a ruler they switch to m / m/s.
Velocity is differenced in the already-transformed display frame.

Default analysis frame follows Tracker: +x right, +y up, so falling vy < 0.
Even without a placed origin, image y is flipped for display so charts match
the overlay axes.

Derivatives use Tracker's step-size scheme: with step N the velocity at a
point is the central difference (p[i+N] - p[i-N]) / (t[i+N] - t[i-N]).
A larger N averages over a wider window and suppresses tracking jitter; for
constant acceleration the result stays exact. Occlusions split the track into
runs and derivatives never span a run boundary; near a boundary the step
shrinks to whatever the run allows, degrading to a one-sided difference.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai.calibration import CalibrationState
from ai.contracts import LOW_CONFIDENCE, TrackPoint, TrackResult
from engine.video_index import VideoInfo

DEFAULT_VELOCITY_STEP = 3
MAX_VELOCITY_STEP = 10


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
    return sample.visible and sample.confidence < LOW_CONFIDENCE


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
) -> list[KinematicSample]:
    if result is None:
        return []
    cal = calibration or CalibrationState()
    position_unit = cal.position_unit
    speed_unit = cal.speed_unit
    step = max(1, min(int(velocity_step), MAX_VELOCITY_STEP))
    points = sorted(result.points, key=lambda p: p.frame)
    display: list[tuple[float | None, float | None]] = [
        _display_xy(point, cal) if point.visible else (None, None) for point in points
    ]
    bounds = _run_bounds(points, display)
    samples: list[KinematicSample] = []
    for i, point in enumerate(points):
        time_s = time_s_for_frame(info, point.frame)
        x, y = display[i]
        vx = vy = speed = None
        if point.visible and x is not None and y is not None and bounds[i] is not None:
            vx, vy = _velocity_at(i, points, display, bounds[i], info, step)
            if vx is not None and vy is not None:
                speed = (vx * vx + vy * vy) ** 0.5
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
