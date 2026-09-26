"""Per-frame kinematics derived from TrackResult + VideoInfo PTS.

Display-only: never written into project JSON or SAM TrackResult.
Uncalibrated units are image-projection px / px/s; after a ruler they
switch to m / m/s. Velocity is differenced in the already-transformed
display frame.

Default analysis frame follows Tracker: +x right, +y up, so falling vy < 0.

The default velocity and acceleration use robust local quadratics in real
time (ai.derivatives): the window is a time span, acceleration gets a wider
window than velocity, run ends reuse the last full window, and by default
both widths are picked from the data. Timestamps are de-jittered when the
container ticks are a rounded uniform grid (ai.timebase). Tracker-style
central differences remain available as a compatibility mode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ai import timebase
from ai.calibration import CalibrationMode, CalibrationState
from ai.contracts import LOW_CONFIDENCE, TrackPoint, TrackPointSource, TrackResult
from ai.depth_audit import DepthAuditState
from ai.derivatives import local_fit, manual_windows, run_derivatives
from ai.drag_fit import DragFit, fit_drag
from engine.video_index import VideoInfo

AUTO_VELOCITY_STEP = 0
DEFAULT_VELOCITY_STEP = AUTO_VELOCITY_STEP
TRACKER_DEFAULT_STEP = 3
MAX_VELOCITY_STEP = 10
VELOCITY_MODE_LOCAL = "local"
VELOCITY_MODE_TRACKER = "tracker"
DEFAULT_VELOCITY_MODE = VELOCITY_MODE_LOCAL
FIT_OFF = "off"
FIT_UNIFORM = "uniform"
FIT_ACCEL = "accel"
FIT_PROJECTILE = "projectile"
FIT_DRAG = "projectile_drag"
FIT_AUTO = "auto"
FIT_MODELS = (FIT_AUTO, FIT_UNIFORM, FIT_ACCEL, FIT_PROJECTILE, FIT_DRAG)
# 自动: per axis, a straight line or a parabola over the whole run, whichever
# the data supports (BIC); none when neither describes the motion (a swing,
# a bounce), so those keep their per-frame values.
AUTO_LOCAL_HALF_S = 0.3
AUTO_RESIDUAL_RATIO = 3.0
AUTO_MIN_R2 = 0.995
LOCAL_HALF_WINDOW_S = 0.10
HUBER_C = 1.345
# A run of measurements is bridged over short gaps (a few review frames);
# only longer gaps start a new run.
RUN_GAP_S = 0.25
RUN_GAP_FRAMES = 4.0
# Weights for the local fits. A fit point the user accepted is an
# interpolation of its neighbours, not new evidence.
FIT_POINT_WEIGHT = 0.15
MIN_POINT_WEIGHT = 0.2
PARTIAL_MASK_RATIO = 0.7


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
    ax: float | None = None
    ay: float | None = None
    accel_unit: str = "px/s²"
    reason: str = ""
    sigma_vx: float | None = None
    sigma_vy: float | None = None
    sigma_ax: float | None = None
    sigma_ay: float | None = None
    window_v_s: float | None = None
    window_a_s: float | None = None
    weight: float | None = None  # trust used by the local fits (1 = normal frame)
    # When v/a come from a fitted motion law, the per-frame estimates stay here.
    local_vx: float | None = None
    local_vy: float | None = None
    local_ax: float | None = None
    local_ay: float | None = None
    kinematics_source: str = "local"
    # A frame waiting for the user (predicted / review): not a measurement, so
    # x…ay stay None, but charts can still show where it would sit.
    pending: bool = False
    review_values: tuple[tuple[str, float], ...] = ()

    def review_value(self, attr: str) -> float | None:
        for name, value in self.review_values:
            if name == attr:
                return value
        return None

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
    reduced: bool = False
    note: str = ""
    stderr: tuple[float, ...] = ()  # of coeffs; only the top one is basis-independent
    # Coefficient covariance in powers of (t − t_ref), for error bands.
    cov_tau: tuple[tuple[float, ...], ...] = ()
    t_ref: float = 0.0

    def derivative(self, time_s: float, order: int) -> float:
        total = 0.0
        for power, coef in enumerate(self.coeffs):
            if power < order:
                continue
            factor = 1.0
            for k in range(order):
                factor *= power - k
            total += factor * coef * time_s ** (power - order)
        return total

    def derivative_sigma(self, time_s: float, order: int) -> float | None:
        if not self.cov_tau:
            return None
        cov = np.asarray(self.cov_tau, dtype=np.float64)
        tau = time_s - self.t_ref
        grad = np.zeros(len(cov))
        for power in range(len(cov)):
            if power < order:
                continue
            factor = 1.0
            for k in range(order):
                factor *= power - k
            grad[power] = factor * tau ** (power - order)
        var = float(grad @ cov @ grad)
        return float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else None

    def evaluate(self, time_s: float) -> float:
        total = 0.0
        power = 1.0
        for coef in self.coeffs:
            total += coef * power
            power *= time_s
        return total

    def as_derivative(self) -> QuantityFit:
        """d/dt of this polynomial. R² stays the position fit's R²."""
        deriv = [index * coef for index, coef in enumerate(self.coeffs) if index > 0]
        errs = [index * err for index, err in enumerate(self.stderr) if index > 0]
        return QuantityFit(
            degree=max(0, self.degree - 1),
            coeffs=tuple(deriv) if deriv else (0.0,),
            r2=self.r2,
            n=self.n,
            reduced=self.reduced,
            note="由位置曲线求导",
            stderr=tuple(errs),
        )

    def coeffs_about(self, origin: float) -> tuple[float, ...]:
        """Coefficients in powers of (t − origin): Taylor expansion at origin."""
        out: list[float] = []
        factorial = 1.0
        for order in range(len(self.coeffs)):
            if order > 0:
                factorial *= order
            out.append(self.derivative(origin, order) / factorial)
        return tuple(out)

    def equation(self, name: str, *, origin: float | None = None) -> str:
        """`name = …`; with `origin`, written in τ = t − origin (start of the run)."""
        if not self.coeffs:
            return f"{name}  拟合失败"
        coeffs = self.coeffs if origin is None else self.coeffs_about(origin)
        symbol = "t" if origin is None else "τ"
        parts: list[str] = []
        for power, coef in enumerate(coeffs):
            if abs(coef) < 1e-12:
                continue
            token = _format_coef(coef).replace("-", "−")
            if power == 0:
                parts.append(token)
                continue
            sign = " + " if coef >= 0 and parts else (" − " if coef < 0 and parts else "")
            mag = _format_coef(abs(coef))
            var = symbol if power == 1 else f"{symbol}²" if power == 2 and origin is not None else f"{symbol}^{power}"
            if not parts and coef < 0:
                parts.append(f"−{mag} {var}")
            elif mag == "1" and power >= 1:
                parts.append(f"{sign}{var}".strip() if sign else var)
            else:
                parts.append(f"{sign}{mag} {var}".lstrip() if not parts else f"{sign}{mag} {var}")
        body = "".join(parts) if parts else "0"
        if self.degree == 0 and len(self.stderr) >= 1 and np.isfinite(self.stderr[0]) and self.stderr[0] > 0:
            body += f" ± {format_uncertainty(self.stderr[0])}"
        suffix = "  （点数不足，已降低次数）" if self.reduced else ""
        extra = f"  （{self.note}）" if self.note else ""
        return f"{name} = {body}    R²={self.r2:.3f}{suffix}{extra}"


_RATE_OF = {"x": "vx", "y": "vy", "vx": "ax", "vy": "ay"}
_SYMBOL = {"x": "x", "y": "y", "vx": "vₓ", "vy": "vᵧ", "ax": "aₓ", "ay": "aᵧ", "speed": "v"}


def law_caption(
    law: Any,
    attr: str,
    name: str,
    *,
    t0: float,
    position_unit: str,
    speed_unit: str,
    accel_unit: str,
) -> str:
    """One readable line for a chart: the fitted law and the overall result.

    Position laws are written in τ = t − t0 (time since the run starts), so
    the constant term is the start value and the τ term the start velocity.
    """
    if isinstance(law, DragLaw):
        return law.equation(name)
    if not isinstance(law, QuantityFit):
        return law.equation(name)
    unit = {"x": position_unit, "y": position_unit, "vx": speed_unit, "vy": speed_unit}.get(attr, accel_unit)
    num = _signed
    head = law.equation(name, origin=t0) if law.degree > 0 else f"{name} = {num(law.coeffs[0])}"
    se_top = law.stderr[-1] if law.stderr and np.isfinite(law.stderr[-1]) and law.stderr[-1] > 0 else None
    rate = _RATE_OF.get(attr)
    rate_symbol = _SYMBOL.get(rate or "", "")
    if law.degree == 0:
        tail = f" ± {format_uncertainty(se_top)}" if se_top is not None else ""
        kind = {"vx": "（匀速）", "vy": "（匀速）"}.get(attr, "")
        return f"{name} = {num(law.coeffs[0])}{tail} {unit}{kind}    R²={law.r2:.3f}"
    t0_text = f"τ 从 {t0:.2f} s 起"
    if law.degree == 1 and rate is not None:
        slope = law.coeffs[1]
        slope_unit = speed_unit if attr in {"x", "y"} else accel_unit
        err = f" ± {format_uncertainty(se_top)}" if se_top is not None else ""
        summary = f"→ 总体{'速度' if attr in {'x', 'y'} else '加速度'} {rate_symbol} = {num(slope)}{err} {slope_unit}"
        return f"{head.split('    R²')[0]}（{t0_text}）{summary}    R²={law.r2:.3f}"
    if law.degree == 2 and attr in {"x", "y"}:
        accel_symbol = _SYMBOL["a" + attr]
        c = law.coeffs_about(t0)
        err = f" ± {format_uncertainty(2 * se_top)}" if se_top is not None else ""
        summary = (
            f"→ 总体加速度 {accel_symbol} = {num(2 * c[2])}{err} {accel_unit}，"
            f"起始速度 {rate_symbol} = {num(c[1])} {speed_unit}"
        )
        return f"{head.split('    R²')[0]}（{t0_text}）{summary}    R²={law.r2:.3f}"
    return head


def _signed(value: float) -> str:
    return _format_coef(value).replace("-", "−")


def law_readout(law: Any, attr: str, time_s: float) -> list[tuple[str, float]]:
    """(symbol, value) pairs a chart shows for the fitted law at one time."""
    out: list[tuple[str, float]] = []
    try:
        out.append((_SYMBOL.get(attr, attr), float(law.evaluate(time_s))))
    except (KeyError, ValueError):
        return out
    rate = _RATE_OF.get(attr)
    if rate is None:
        return out
    if isinstance(law, DragLaw):
        out.append((_SYMBOL[rate], float(law.fit.evaluate(rate, time_s))))
    elif isinstance(law, QuantityFit):
        out.append((_SYMBOL[rate], float(law.derivative(time_s, 1))))
    return out


def is_low_confidence(sample: KinematicSample) -> bool:
    if sample.visible and sample.confidence < LOW_CONFIDENCE:
        return True
    return any(
        flag in sample.quality_flags
        for flag in (
            "extrapolated",
            "off_plane",
            "ai_estimate",
            "background_jump",
            "review",
            "interpolated",
            "predicted",
        )
    )


def quality_label(sample: KinematicSample) -> str:
    flags = sample.quality_flags
    if "background_jump" in flags:
        return "已拒收"
    if "review" in flags:
        return "待复核"
    if "interpolated" in flags:
        return "插值（不测量）"
    if "predicted" in flags:
        return "预测（不测量）"
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
    if sample.reason in {"外观歧义", "双向不一致", "目标不可见"}:
        bits.append(sample.reason)
    elif "background_jump" in sample.quality_flags:
        bits.append("已拒收：疑似跳到背景")
    elif sample.reason:
        bits.append(sample.reason)
    elif sample.visible and sample.confidence < LOW_CONFIDENCE:
        bits.append(f"追踪质量 {sample.confidence:.2f}")
    return " · ".join(bits)


def tracking_reason(point: TrackPoint) -> str:
    """Short hover text for why a frame was not treated as a measurement."""
    diagnostics = point.diagnostics or {}
    if diagnostics.get("appearance_ambiguous"):
        return "外观歧义"
    flow = diagnostics.get("forward_backward")
    if diagnostics.get("rejected_by_evidence") and isinstance(flow, (int, float)) and flow < 0.12:
        return "双向不一致"
    if point.note:
        return point.note
    if diagnostics.get("missing_mask") or not point.visible:
        return "目标不可见"
    return ""


def accel_unit_for(speed_unit: str) -> str:
    if speed_unit.endswith("/s"):
        return f"{speed_unit}²"
    return "px/s²"


def time_s_for_frame(info: VideoInfo | None, frame: int) -> float:
    """Measurement time of a frame (container PTS, de-jittered when uniform)."""
    return timebase.time_s(info, frame)


def series_for_result(
    result: TrackResult | None,
    info: VideoInfo | None,
    *,
    calibration: CalibrationState | None = None,
    velocity_step: int = DEFAULT_VELOCITY_STEP,
    velocity_mode: str = DEFAULT_VELOCITY_MODE,
    depth_audit: DepthAuditState | None = None,
    derivative_model: str = FIT_OFF,
    pending_positions: dict[int, tuple[float, float]] | None = None,
) -> list[KinematicSample]:
    """Per-frame samples.

    `derivative_model`: when a motion law (FIT_*) is given, v and a of every
    measured frame are taken from that law fitted to the whole visible run
    (the per-frame estimates are kept in local_*). `pending_positions` are
    pixel positions of frames waiting for review (fit suggestions); they are
    not measurements but get display values for the charts.
    """
    if result is None:
        return []
    cal = calibration or CalibrationState()
    position_unit = cal.position_unit
    speed_unit = cal.speed_unit
    accel_unit = accel_unit_for(speed_unit)
    requested = int(velocity_step)
    auto = requested <= AUTO_VELOCITY_STEP
    step = TRACKER_DEFAULT_STEP if auto else max(1, min(requested, MAX_VELOCITY_STEP))
    mode = velocity_mode if velocity_mode in {VELOCITY_MODE_LOCAL, VELOCITY_MODE_TRACKER} else DEFAULT_VELOCITY_MODE
    points = sorted(result.points, key=lambda p: p.frame)
    usable = [point.usable_for_measurement() for point in points]
    display: list[tuple[float | None, float | None]] = [
        _display_xy(point, cal) if usable[index] else (None, None)
        for index, point in enumerate(points)
    ]
    bounds = _run_bounds(points, display)
    times = [time_s_for_frame(info, point.frame) for point in points]
    pending_display = _pending_display(points, usable, cal, pending_positions)
    local: dict[int, dict[str, float | None]] = {}
    if mode == VELOCITY_MODE_LOCAL:
        local = _local_derivatives(
            points, display, times, None if auto else manual_windows(step), pending_display
        )
    samples: list[KinematicSample] = []
    for i, point in enumerate(points):
        time_s = times[i]
        x, y = display[i]
        vx = vy = speed = ax = ay = None
        extra: dict[str, float | None] = {}
        if usable[i] and x is not None and y is not None and bounds[i] is not None:
            if mode == VELOCITY_MODE_TRACKER:
                vx, vy = _velocity_at(i, points, display, bounds[i], info, step)
            else:
                extra = local.get(i, {})
                vx, vy = extra.get("vx"), extra.get("vy")
                ax, ay = extra.get("ax"), extra.get("ay")
            if vx is not None and vy is not None:
                speed = (vx * vx + vy * vy) ** 0.5
        flags: list[str] = []
        source = "pixel"
        sigma_x = sigma_y = off_m = None
        if point.note == "疑似跳到背景" or point.diagnostics.get("rejected_by_evidence"):
            flags.append("background_jump")
        elif not point.visible:
            flags.append("lost")
        if point.status.value == "review":
            flags.append("review")
        if point.source.value == "interpolated":
            flags.append("interpolated")
        elif point.source.value == "predicted":
            flags.append("predicted")
        if usable[i]:
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
                visible=usable[i],
                confidence=point.confidence,
                manual=point.manual,
                position_unit=position_unit,
                speed_unit=speed_unit,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                quality_flags=tuple(flags),
                off_plane_m=off_m,
                source=source,
                ax=ax,
                ay=ay,
                accel_unit=accel_unit,
                reason=tracking_reason(point),
                sigma_vx=extra.get("sigma_vx"),
                sigma_vy=extra.get("sigma_vy"),
                sigma_ax=extra.get("sigma_ax"),
                sigma_ay=extra.get("sigma_ay"),
                window_v_s=extra.get("window_v_s"),
                window_a_s=extra.get("window_a_s"),
                weight=extra.get("weight"),
                pending=i in pending_display,
                review_values=_review_values(i, pending_display, local),
            )
        )
    if mode == VELOCITY_MODE_TRACKER:
        samples = _attach_tracker_accel(samples, step)
    if derivative_model in FIT_MODELS:
        samples = apply_model_derivatives(samples, derivative_model)
    return samples


def _pending_display(
    points: list[TrackPoint],
    usable: list[bool],
    cal: CalibrationState,
    pending_positions: dict[int, tuple[float, float]] | None,
) -> dict[int, tuple[float, float]]:
    """Display coordinates of frames waiting for review, by point index."""
    out: dict[int, tuple[float, float]] = {}
    given = pending_positions or {}
    for i, point in enumerate(points):
        if usable[i] or point.manual:
            continue
        if point.frame in given:
            px, py = given[point.frame]
        elif point.visible and pending_positions is None:
            px, py = point.x, point.y
        else:
            continue
        probe = TrackPoint(frame=point.frame, x=float(px), y=float(py))
        out[i] = _display_xy(probe, cal)
    return out


def _review_values(
    index: int,
    pending_display: dict[int, tuple[float, float]],
    local: dict[int, dict[str, float | None]],
) -> tuple[tuple[str, float], ...]:
    if index not in pending_display:
        return ()
    x, y = pending_display[index]
    values: list[tuple[str, float]] = [("x", float(x)), ("y", float(y))]
    extra = local.get(index, {})
    for name in ("vx", "vy", "ax", "ay"):
        value = extra.get(name)
        if value is not None:
            values.append((name, float(value)))
    vx, vy = extra.get("vx"), extra.get("vy")
    if vx is not None and vy is not None:
        values.append(("speed", float((vx * vx + vy * vy) ** 0.5)))
    return tuple(values)


def apply_model_derivatives(samples: list[KinematicSample], model: str) -> list[KinematicSample]:
    """Replace per-frame v / a by the chosen motion law's derivatives.

    Positions stay measured. Frames outside every fitted run keep their
    per-frame estimates. Pending frames get the law's values in review_values.
    """
    if model not in FIT_MODELS or not samples:
        return samples
    values = _model_values(samples, model)
    if not values:
        return samples
    out: list[KinematicSample] = []
    for sample in samples:
        got = values.get(sample.frame)
        if got is None:
            out.append(sample)
            continue
        if sample.pending:
            merged = dict(sample.review_values)
            for name in ("vx", "vy", "ax", "ay", "speed"):
                if got.get(name) is not None:
                    merged[name] = float(got[name])
            out.append(replace(sample, review_values=tuple(merged.items())))
            continue
        if sample.x is None or sample.y is None:
            out.append(sample)
            continue
        merged: dict[str, float | None] = {}
        for name in ("vx", "vy", "ax", "ay"):
            if name in got:
                merged[name] = got[name]
                merged["sigma_" + name] = got.get("sigma_" + name)
            else:
                merged[name] = getattr(sample, name)
                merged["sigma_" + name] = getattr(sample, "sigma_" + name)
        vx, vy = merged["vx"], merged["vy"]
        out.append(
            replace(
                sample,
                local_vx=sample.vx,
                local_vy=sample.vy,
                local_ax=sample.ax,
                local_ay=sample.ay,
                vx=vx,
                vy=vy,
                speed=None if vx is None or vy is None else float((vx * vx + vy * vy) ** 0.5),
                ax=merged["ax"],
                ay=merged["ay"],
                sigma_vx=merged["sigma_vx"],
                sigma_vy=merged["sigma_vy"],
                sigma_ax=merged["sigma_ax"],
                sigma_ay=merged["sigma_ay"],
                kinematics_source=model,
            )
        )
    return out


def _model_values(samples: list[KinematicSample], model: str) -> dict[int, dict[str, float | None]]:
    out: dict[int, dict[str, float | None]] = {}
    targets = [s for s in samples if (s.x is not None and s.y is not None) or s.pending]
    if model == FIT_DRAG:
        for t0, t1, law in _drag_runs(samples, "x"):
            inside = [s for s in targets if t0 - 1e-9 <= s.time_s <= t1 + 1e-9]
            if not inside:
                continue
            times = np.asarray([s.time_s for s in inside])
            fit = law.fit
            cols: dict[str, np.ndarray | None] = {}
            for name in ("vx", "vy", "ax", "ay"):
                cols[name] = fit.series(name, times)
                cols["sigma_" + name] = fit.sigma_series(name, times)
            for k, sample in enumerate(inside):
                out[sample.frame] = {
                    name: (None if arr is None else float(arr[k])) for name, arr in cols.items()
                }
        return out
    fits: dict[str, list[tuple[float, float, QuantityFit]]] = {
        axis: _model_attr_runs(samples, axis, model) for axis in ("x", "y")
    }
    for sample in targets:
        got: dict[str, float | None] = {}
        for axis, (v_name, a_name) in (("x", ("vx", "ax")), ("y", ("vy", "ay"))):
            law = next(
                (fit for t0, t1, fit in fits[axis] if t0 - 1e-9 <= sample.time_s <= t1 + 1e-9),
                None,
            )
            if law is None:
                continue  # this axis keeps its per-frame values
            got[v_name] = law.derivative(sample.time_s, 1)
            got[a_name] = law.derivative(sample.time_s, 2)
            got["sigma_" + v_name] = law.derivative_sigma(sample.time_s, 1)
            got["sigma_" + a_name] = law.derivative_sigma(sample.time_s, 2)
        if got:
            out[sample.frame] = got
    return out


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


def position_degree(model: str, axis: str) -> int:
    """Polynomial degree of x(t) or y(t) for a named motion law."""
    if model == FIT_ACCEL:
        return 2
    if model == FIT_PROJECTILE and axis == "y":
        return 2
    return 1


def auto_position_fit(segment: list[KinematicSample], axis: str) -> QuantityFit | None:
    """Straight line or parabola for one axis of one run, or None if neither fits."""
    pairs = [(s.time_s, float(getattr(s, axis))) for s in segment if getattr(s, axis) is not None]
    n = len(pairs)
    if n < 3:
        return None
    times = np.asarray([p[0] for p in pairs])
    values = np.asarray([p[1] for p in pairs])
    if float(np.ptp(times)) <= 1e-9:
        return None
    best: tuple[float, QuantityFit] | None = None
    for degree in (1, 2):
        if n < degree + 3:
            continue
        fitted = fit_quantity(segment, axis, degree)
        if fitted is None:
            continue
        resid = values - np.asarray([fitted.evaluate(t) for t in times])
        rss = max(float(resid @ resid), 1e-300)
        bic = n * np.log(rss / n) + (degree + 1) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, fitted)
    if best is None:
        return None
    fitted = best[1]
    resid = values - np.asarray([fitted.evaluate(t) for t in times])
    rms = float(np.sqrt(np.mean(resid * resid)))
    local = local_fit(times, values[:, None], np.ones(n), AUTO_LOCAL_HALF_S)["loo"][:, 0]
    local = local[np.isfinite(local)]
    local_rms = float(np.sqrt(np.mean(local * local))) if len(local) else float("inf")
    if not (rms <= AUTO_RESIDUAL_RATIO * local_rms or fitted.r2 >= AUTO_MIN_R2):
        return None
    note = "自动：直线，匀速" if fitted.degree == 1 else "自动：抛物线，匀加速"
    return replace(fitted, note=note)


def position_fit(segment: list[KinematicSample], model: str, axis: str) -> QuantityFit | None:
    if model == FIT_AUTO:
        return auto_position_fit(segment, axis)
    return fit_quantity(segment, axis, position_degree(model, axis))


def _model_attr_runs(
    samples: list[KinematicSample], axis: str, model: str
) -> list[tuple[float, float, QuantityFit]]:
    runs: list[tuple[float, float, QuantityFit]] = []
    for segment in law_segments(samples, axis):
        fitted = position_fit(segment, model, axis)
        if fitted is None:
            continue
        times = [sample.time_s for sample in segment if getattr(sample, axis) is not None]
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            continue
        runs.append((t0, t1, fitted))
    return runs


def law_runs(
    samples: list[KinematicSample],
    model: str,
    attr: str,
) -> list[tuple[float, float, QuantityFit | SpeedLaw]]:
    """Fit one visible segment at a time. Velocity is the derivative of position.

    Samples are weighted by the time gap around them, so a burst of frames
    cannot pull the curve harder than the same motion shot at a lower rate.
    """
    if model == FIT_OFF:
        return []
    if model == FIT_DRAG:
        return _drag_runs(samples, attr)
    if attr == "speed":
        return _speed_runs(samples, model)
    axis = {"x": "x", "vx": "x", "ax": "x", "y": "y", "vy": "y", "ay": "y"}.get(attr)
    if axis is None:
        return []
    runs: list[tuple[float, float, QuantityFit]] = []
    for t0, t1, fitted in _model_attr_runs(samples, axis, model):
        if attr in {"ax", "ay"}:
            law = replace(fitted.as_derivative().as_derivative(), note="由位置曲线求二阶导")
        elif attr in {"vx", "vy"}:
            law = fitted.as_derivative()
        else:
            law = fitted
        runs.append((t0, t1, law))
    return runs


def trajectory_curves(
    times: list[float],
    xs: list[float],
    ys: list[float],
    model: str,
) -> list[list[tuple[float, float]]]:
    """Smooth pixel-space curve for the video. y is image-down, not chart-up."""
    if model == FIT_OFF or len(times) < 2:
        return []
    curves: list[list[tuple[float, float]]] = []
    if model == FIT_DRAG:
        for run_t, run_x, run_y in _split_time_gaps(times, xs, ys, bridge=True):
            fitted = fit_drag(run_t, run_x, run_y)
            if fitted is None:
                continue
            grid = np.linspace(fitted.t0, fitted.t1, 64)
            state = fitted.state(grid)
            curves.append([(float(a), float(b)) for a, b in state[:, :2]])
        return curves
    for run_t, run_x, run_y in _split_time_gaps(times, xs, ys):
        samples = [
            KinematicSample(
                frame=i,
                time_s=time_s,
                x=x,
                y=y,
                vx=None,
                vy=None,
                speed=None,
                visible=True,
                confidence=1.0,
                manual=False,
            )
            for i, (time_s, x, y) in enumerate(zip(run_t, run_x, run_y))
        ]
        x_fit = position_fit(samples, model, "x")
        y_fit = position_fit(samples, model, "y")
        if x_fit is None or y_fit is None:
            continue
        t0, t1 = run_t[0], run_t[-1]
        if t1 <= t0:
            continue
        curve: list[tuple[float, float]] = []
        steps = 48
        for i in range(steps + 1):
            time_s = t0 + (t1 - t0) * i / steps
            curve.append((x_fit.evaluate(time_s), y_fit.evaluate(time_s)))
        curves.append(curve)
    return curves


def fit_quantity(
    samples: list[KinematicSample],
    attr: str,
    degree: int,
) -> QuantityFit | None:
    """Time-weighted least squares of `attr` vs time.

    Each point's weight is the time gap around it. Equal frame weights would
    let a high frame rate dominate the law even when the motion is the same.
    """
    if degree < 0:
        return None
    pairs = [
        (sample.time_s, float(getattr(sample, attr)))
        for sample in samples
        if getattr(sample, attr) is not None
    ]
    if len(pairs) < 2:
        return None
    reduced = False
    degree = int(degree)
    while degree > 0 and len(pairs) < degree + 2:
        degree -= 1
        reduced = True
    times = [item[0] for item in pairs]
    values = [item[1] for item in pairs]
    weights = _time_weights(times)
    if degree == 0:
        return _constant_fit(times, values, weights, reduced)
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
        coeffs_tau = _solve_normal(rows, values, weights)
    except ValueError:
        return None
    coeffs = _shift_poly(coeffs_tau, t0)
    r2 = _weighted_r2(times, values, weights, coeffs)
    cov = _coefficient_cov(rows, values, weights, coeffs_tau)
    return QuantityFit(
        degree=degree,
        coeffs=tuple(coeffs),
        r2=r2,
        n=len(pairs),
        reduced=reduced,
        stderr=_top_coefficient_stderr(cov),
        cov_tau=() if cov is None else tuple(tuple(float(v) for v in row) for row in cov),
        t_ref=t0,
    )


def _top_coefficient_stderr(cov: np.ndarray | None) -> tuple[float, ...]:
    if cov is None:
        return ()
    p = cov.shape[0]
    top = float(cov[-1, -1])
    return tuple(float("nan") for _ in range(p - 1)) + (float(np.sqrt(top)) if top >= 0 else float("nan"),)


def _coefficient_cov(
    rows: list[list[float]],
    values: list[float],
    weights: list[float],
    coeffs_tau: list[float],
) -> np.ndarray | None:
    """Standard error of the highest-power coefficient (NaN for the others).

    Time weights are not inverse variances, so the sandwich form is used,
    and residuals that run in long streaks (tracking bias lasting several
    frames) inflate it with an AR(1) factor instead of pretending every
    frame is independent.
    """
    design = np.asarray(rows, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n, p = design.shape
    if n <= p:
        return None
    resid = y - design @ np.asarray(coeffs_tau, dtype=np.float64)
    s2 = float(np.sum(resid * resid) / (n - p))
    try:
        bread = np.linalg.inv(design.T @ (design * w[:, None]))
    except np.linalg.LinAlgError:
        return None
    meat = design.T @ (design * (w * w)[:, None])
    cov = bread @ meat @ bread * s2
    centered = resid - resid.mean()
    denom = float(np.dot(centered, centered))
    rho = float(np.dot(centered[1:], centered[:-1]) / denom) if denom > 0 else 0.0
    rho = min(max(rho, 0.0), 0.9)
    return cov * (1.0 + rho) / (1.0 - rho)


def law_segments(samples: list[KinematicSample], attr: str) -> list[list[KinematicSample]]:
    """Runs for fitting a motion law.

    Unlike contiguous_segments, a few frames waiting for review do not cut
    the run (they are just skipped); a lost target or a long gap does.
    """
    measured = [s for s in samples if getattr(s, attr) is not None]
    if not measured:
        return []
    steps = [b.time_s - a.time_s for a, b in zip(measured, measured[1:]) if b.time_s > a.time_s]
    limit = max(RUN_GAP_S, RUN_GAP_FRAMES * (float(np.median(steps)) if steps else 0.0))
    segments: list[list[KinematicSample]] = []
    current: list[KinematicSample] = []
    for sample in samples:
        value = getattr(sample, attr)
        if value is None:
            if "lost" in sample.quality_flags and current:
                segments.append(current)
                current = []
            continue
        if current and sample.time_s - current[-1].time_s > limit:
            segments.append(current)
            current = []
        current.append(sample)
    if current:
        segments.append(current)
    return segments


def _fit_attr_runs(
    samples: list[KinematicSample],
    attr: str,
    degree: int,
) -> list[tuple[float, float, QuantityFit]]:
    runs: list[tuple[float, float, QuantityFit]] = []
    for segment in law_segments(samples, attr):
        fitted = fit_quantity(segment, attr, degree)
        if fitted is None:
            continue
        times = [sample.time_s for sample in segment if getattr(sample, attr) is not None]
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            continue
        runs.append((t0, t1, fitted))
    return runs


def _speed_runs(
    samples: list[KinematicSample], model: str
) -> list[tuple[float, float, SpeedLaw]]:
    runs: list[tuple[float, float, QuantityFit]] = []
    segments = [
        [s for s in segment if s.y is not None] for segment in law_segments(samples, "x")
    ]
    for segment in segments:
        x_fit = position_fit(segment, model, "x")
        y_fit = position_fit(segment, model, "y")
        if x_fit is None or y_fit is None:
            continue
        times = [sample.time_s for sample in segment]
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            continue
        runs.append((t0, t1, SpeedLaw(x_fit.as_derivative(), y_fit.as_derivative())))
    return runs


class DragLaw:
    """One quantity of a projectile-with-drag fit, shaped like QuantityFit for the charts."""

    def __init__(self, fit: DragFit, attr: str, position_unit: str, accel_unit: str) -> None:
        self.fit = fit
        self.attr = attr
        self.position_unit = position_unit
        self.accel_unit = accel_unit
        self.reduced = False
        self.r2 = float("nan")
        self.n = fit.n

    def evaluate(self, time_s: float) -> float:
        return self.fit.evaluate(self.attr, time_s)

    def summary(self) -> str:
        fit = self.fit
        pos = self.position_unit
        k_unit = f"1/{pos}"
        g_text = f"g = {_format_coef(fit.gravity)} ± {format_uncertainty(fit.gravity_sigma)} {self.accel_unit}"
        k_text = f"k = {fit.k:.3g} ± {format_uncertainty(fit.k_sigma)} {k_unit}"
        tilt = f"，重力偏离竖直 {fit.tilt_deg:+.1f}°" if abs(fit.tilt_deg) >= 1.0 else ""
        return f"{g_text}，{k_text}{tilt}"

    def equation(self, name: str) -> str:
        fit = self.fit
        return (
            f"{name}：斜抛+空气阻力 a = g − k|v|v，{self.summary()}"
            f"    残差 {fit.rms:.3g} {self.position_unit}，{fit.n} 点"
        )


def _drag_runs(
    samples: list[KinematicSample], attr: str
) -> list[tuple[float, float, DragLaw]]:
    if attr not in {"x", "y", "vx", "vy", "ax", "ay", "speed"}:
        return []
    measured = [s for s in samples if s.x is not None and s.y is not None]
    if not measured:
        return []
    position_unit = measured[0].position_unit
    accel_unit = measured[0].accel_unit
    runs: list[tuple[float, float, DragLaw]] = []
    for segment in law_segments(samples, "x"):
        run_t = [s.time_s for s in segment]
        run_x = [float(s.x) for s in segment]
        run_y = [float(s.y) for s in segment]
        weights = [s.weight if s.weight is not None else max(s.confidence, 0.2) for s in segment]
        fitted = fit_drag(run_t, run_x, run_y, weights)
        if fitted is None or fitted.t1 <= fitted.t0:
            continue
        runs.append((fitted.t0, fitted.t1, DragLaw(fitted, attr, position_unit, accel_unit)))
    return runs


class SpeedLaw:
    """√(vₓ²+vᵧ²) from the two position fits. Not a polynomial."""

    def __init__(self, vx: QuantityFit, vy: QuantityFit) -> None:
        self.r2 = min(vx.r2, vy.r2)
        self.reduced = vx.reduced or vy.reduced
        self._vx = vx
        self._vy = vy

    def evaluate(self, time_s: float) -> float:
        vx = self._vx.evaluate(time_s)
        vy = self._vy.evaluate(time_s)
        return (vx * vx + vy * vy) ** 0.5

    def equation(self, name: str) -> str:
        suffix = "  （点数不足，已降低次数）" if self.reduced else ""
        return f"{name} = √(vₓ²+vᵧ²)    R²≥{self.r2:.3f}{suffix}  （由 x、y 的拟合求导）"


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
    return point.usable_for_measurement() and x is not None and y is not None


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


def measurement_weights(points: list[TrackPoint], indices: list[int]) -> np.ndarray:
    """Relative trust of each measurement in the local fits (1 = a good frame).

    Tracker confidence counts, a mask much smaller or larger than its
    neighbours' (only part of a thin blurred object segmented, so the
    centroid slides along it) counts less, and an accepted fit point counts
    very little because it was computed from the neighbours themselves.
    """
    areas = []
    for i in indices:
        diag = points[i].diagnostics or {}
        area = diag.get("area")
        auto = points[i].source is TrackPointSource.AUTO and not points[i].manual
        areas.append(float(area) if auto and isinstance(area, (int, float)) and area > 0 else np.nan)
    area_arr = np.asarray(areas, dtype=np.float64)
    out = np.ones(len(indices))
    for k, i in enumerate(indices):
        point = points[i]
        diag = point.diagnostics or {}
        if diag.get("fit_accepted"):
            out[k] = FIT_POINT_WEIGHT
            continue
        if point.manual or point.source is TrackPointSource.MANUAL:
            out[k] = 1.0
            continue
        weight = min(1.0, max(float(point.confidence), MIN_POINT_WEIGHT))
        if np.isfinite(area_arr[k]):
            around = area_arr[max(0, k - 6) : k + 7]
            around = around[np.isfinite(around)]
            if len(around) >= 3:
                ratio = area_arr[k] / float(np.median(around))
                closeness = min(ratio, 1.0 / ratio) if ratio > 0 else 0.0
                if closeness < PARTIAL_MASK_RATIO:
                    weight *= max(closeness / PARTIAL_MASK_RATIO, 0.25) ** 2
        out[k] = weight
    return out


def _measurement_runs(times: list[float], indices: list[int]) -> list[list[int]]:
    """Split measured samples only at real gaps (not at a couple of review frames)."""
    if not indices:
        return []
    steps = [times[b] - times[a] for a, b in zip(indices, indices[1:]) if times[b] > times[a]]
    typical = float(np.median(steps)) if steps else 0.0
    limit = max(RUN_GAP_S, RUN_GAP_FRAMES * typical)
    runs: list[list[int]] = [[indices[0]]]
    for prev, cur in zip(indices, indices[1:]):
        if times[cur] - times[prev] > limit:
            runs.append([])
        runs[-1].append(cur)
    return runs


def _local_derivatives(
    points: list[TrackPoint],
    display: list[tuple[float | None, float | None]],
    times: list[float],
    windows: tuple[float, float] | None,
    pending: dict[int, tuple[float, float]] | None = None,
) -> dict[int, dict[str, float | None]]:
    measured = [
        i
        for i, point in enumerate(points)
        if point.usable_for_measurement() and display[i][0] is not None and display[i][1] is not None
    ]
    pending = pending or {}
    out: dict[int, dict[str, float | None]] = {}
    runs = _measurement_runs(times, measured)
    limit = RUN_GAP_S
    extra: list[list[int]] = [[] for _ in runs]
    for i in pending:
        for k, run in enumerate(runs):
            if times[run[0]] - limit <= times[i] <= times[run[-1]] + limit:
                extra[k].append(i)
                break
    for run, waiting in zip(runs, extra):
        measured_run = list(run)
        run = measured_run + waiting
        t = np.asarray([times[i] for i in run], dtype=np.float64)
        order = np.argsort(t, kind="stable")
        run = [run[k] for k in order]
        t = t[order]
        xs = np.asarray([display[i][0] if i not in pending else pending[i][0] for i in run], dtype=np.float64)
        ys = np.asarray([display[i][1] if i not in pending else pending[i][1] for i in run], dtype=np.float64)
        weights = measurement_weights(points, run)
        # Frames waiting for review are evaluated, never fitted.
        weights = np.where([i in pending for i in run], 0.0, weights)
        result = run_derivatives(t, xs, ys, weights, windows=windows)
        for k, i in enumerate(run):
            values: dict[str, float | None] = {}
            for name in ("vx", "vy", "ax", "ay", "sigma_vx", "sigma_vy", "sigma_ax", "sigma_ay"):
                value = float(getattr(result, name)[k])
                values[name] = value if np.isfinite(value) else None
            if values["vx"] is None or values["vy"] is None:
                values["vx"] = values["vy"] = values["sigma_vx"] = values["sigma_vy"] = None
            if values["ax"] is None or values["ay"] is None:
                values["ax"] = values["ay"] = values["sigma_ax"] = values["sigma_ay"] = None
            values["window_v_s"] = result.window_v_s
            values["window_a_s"] = result.window_a_s
            values["weight"] = float(weights[k])
            out[i] = values
    return out


def _local_velocity_at(
    index: int,
    times: list[float],
    display: list[tuple[float | None, float | None]],
    run: tuple[int, int],
    points: list[TrackPoint],
    step: int,
    info: VideoInfo | None,
) -> tuple[float | None, float | None, float | None, float | None]:
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
        vx, vy = _velocity_at(index, points, display, run, info, step)
        return vx, vy, None, None
    degree = 2 if len(ts) >= 4 else 1
    vx, ax = _poly_derivative(ts, xs, weights, t0, degree)
    vy, ay = _poly_derivative(ts, ys, weights, t0, degree)
    return vx, vy, ax, ay


def _attach_tracker_accel(
    samples: list[KinematicSample], step: int
) -> list[KinematicSample]:
    """Second central difference of velocity, staying inside each visible run."""
    accel_x = _difference_series([sample.vx for sample in samples], [sample.time_s for sample in samples], step)
    accel_y = _difference_series([sample.vy for sample in samples], [sample.time_s for sample in samples], step)
    return [
        replace(sample, ax=accel_x[index], ay=accel_y[index])
        for index, sample in enumerate(samples)
    ]


def _difference_series(
    values: list[float | None], times: list[float], step: int
) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    index = 0
    while index < len(values):
        if values[index] is None:
            index += 1
            continue
        end = index
        while end + 1 < len(values) and values[end + 1] is not None:
            end += 1
        for cursor in range(index, end + 1):
            left = cursor - index
            right = end - cursor
            span = min(step, left, right)
            if span >= 1:
                a, b = cursor - span, cursor + span
            elif right >= 1:
                a, b = cursor, cursor + min(step, right)
            elif left >= 1:
                a, b = cursor - min(step, left), cursor
            else:
                continue
            v0, v1 = values[a], values[b]
            dt = times[b] - times[a]
            if v0 is None or v1 is None or dt <= 0:
                continue
            out[cursor] = (v1 - v0) / dt
        index = end + 1
    return out


def _poly_derivative(
    times: list[float],
    values: list[float],
    weights: list[float],
    t0: float,
    degree: int,
) -> tuple[float | None, float | None]:
    n = len(times)
    if n < 2:
        return None, None
    degree = max(1, min(int(degree), n - 1))
    tau = np.asarray(times, dtype=np.float64) - t0
    y = np.asarray(values, dtype=np.float64)
    w = np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)
    if float(np.max(tau) - np.min(tau)) <= 1e-12:
        return None, None
    design = np.column_stack([tau ** k for k in range(degree + 1)])
    coef = np.zeros(degree + 1)
    for _ in range(4):
        sw = np.sqrt(w)
        try:
            coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        except np.linalg.LinAlgError:
            return None, None
        resid = y - design @ coef
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scale = 1.4826 * mad + 1e-9
        cutoff = HUBER_C * scale
        w = np.asarray(weights, dtype=np.float64) / np.maximum(1.0, np.abs(resid) / cutoff)
    slope = float(coef[1]) if len(coef) >= 2 else 0.0
    curvature = float(2.0 * coef[2]) if degree >= 2 and len(coef) >= 3 else None
    return slope, curvature


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


def _time_weights(times: list[float]) -> list[float]:
    if len(times) < 2:
        return [1.0] * len(times)
    weights: list[float] = []
    last = len(times) - 1
    for index, _time_s in enumerate(times):
        if index == 0:
            gap = times[1] - times[0]
        elif index == last:
            gap = times[last] - times[last - 1]
        else:
            gap = 0.5 * (times[index + 1] - times[index - 1])
        weights.append(max(gap, 1e-6))
    return weights


def _constant_fit(
    times: list[float], values: list[float], weights: list[float], reduced: bool
) -> QuantityFit:
    total_w = sum(weights) or 1.0
    mean = sum(weight * value for weight, value in zip(weights, values)) / total_w
    coeffs = [mean]
    return QuantityFit(
        degree=0,
        coeffs=(mean,),
        r2=_weighted_r2(times, values, weights, coeffs),
        n=len(values),
        reduced=reduced,
    )


def _weighted_r2(
    times: list[float],
    values: list[float],
    weights: list[float],
    coeffs: list[float],
) -> float:
    total_w = sum(weights) or 1.0
    mean = sum(weight * value for weight, value in zip(weights, values)) / total_w
    ss_res = 0.0
    ss_tot = 0.0
    for time_s, value, weight in zip(times, values, weights):
        err = value - _eval_poly(coeffs, time_s)
        ss_res += weight * err * err
        ss_tot += weight * (value - mean) * (value - mean)
    if ss_tot < 1e-18:
        return 1.0 if ss_res < 1e-18 else 0.0
    return max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


def _split_time_gaps(
    times: list[float], xs: list[float], ys: list[float], *, bridge: bool = False
) -> list[tuple[list[float], list[float], list[float]]]:
    if not times:
        return []
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1) if times[i + 1] > times[i]]
    median = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    limit = max(median * 2.5, 1e-3)
    if bridge:
        limit = max(RUN_GAP_S, RUN_GAP_FRAMES * median)
    runs: list[tuple[list[float], list[float], list[float]]] = []
    cur_t = [times[0]]
    cur_x = [xs[0]]
    cur_y = [ys[0]]
    for index in range(1, len(times)):
        if times[index] - times[index - 1] > limit:
            runs.append((cur_t, cur_x, cur_y))
            cur_t, cur_x, cur_y = [], [], []
        cur_t.append(times[index])
        cur_x.append(xs[index])
        cur_y.append(ys[index])
    runs.append((cur_t, cur_x, cur_y))
    return runs


def _solve_normal(
    rows: list[list[float]],
    values: list[float],
    weights: list[float] | None = None,
) -> list[float]:
    n = len(rows[0])
    if weights is None:
        weights = [1.0] * len(values)
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, value, weight in zip(rows, values, weights):
        for i in range(n):
            atb[i] += weight * row[i] * value
            for j in range(n):
                ata[i][j] += weight * row[i] * row[j]
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


def format_uncertainty(value: float | None) -> str:
    """Two significant digits, no exponent for the magnitudes seen here."""
    if value is None or not np.isfinite(value):
        return "?"
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.1e}"
    digits = max(0, 1 - int(np.floor(np.log10(abs(value)))))
    return f"{value:.{digits}f}"


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
