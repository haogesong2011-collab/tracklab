"""Deterministic experiment classification and kinematic fitting.

All time comes from VideoInfo PTS via kinematics.series_for_result.
DeepSeek is never asked to invent these numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ai.calibration import CalibrationMode, CalibrationState, uniform_state
from ai.contracts import (
    EXPERIMENT_LABELS,
    ExperimentAnalysis,
    ExperimentCandidate,
    ExperimentType,
    FailureReason,
    FitResult,
    PhysicsResult,
    TrackResult,
)
from ai.depth_audit import DepthAuditState
from ai.kinematics import KinematicSample, contiguous_segments, series_for_result
from ai.schema import Point2D
from engine.video_index import VideoInfo

MIN_LINEAR_POINTS = 6
MIN_QUAD_POINTS = 8
MIN_PENDULUM_POINTS = 12
AUTO_CONFIRM_MIN = 0.62
AUTO_CONFIRM_MARGIN = 0.08
MAX_AUTO_NRMSE = 0.08
LIST_NRMSE = 0.18
SMALL_ACCEL_RATIO = 0.12
FREEFALL_DRIFT = 0.08
PERIOD_HINT_BOOST = 0.04


def video_info_from_fps(
    track: TrackResult,
    fps: float,
    *,
    width: int = 1,
    height: int = 1,
) -> VideoInfo:
    """Build a PTS map when only a constant frame rate is known (eval wrapper)."""
    last = max((point.frame for point in track.points), default=0)
    count = last + 1
    rate = max(float(fps), 1e-6)
    pts_ms = tuple(int(round(index * 1000.0 / rate)) for index in range(count))
    return VideoInfo(
        path=Path("synthetic"),
        width=width,
        height=height,
        pts=tuple(range(count)),
        time_base=0.001,
        pts_ms=pts_ms,
    )


def calibration_from_ppm(pixels_per_meter: float | None) -> CalibrationState:
    if pixels_per_meter is None or pixels_per_meter <= 0:
        return CalibrationState()
    return uniform_state(
        Point2D(0.0, 0.0),
        Point2D(float(pixels_per_meter), 0.0),
        length_m=1.0,
    )


def source_fingerprint(
    result: TrackResult | None,
    info: VideoInfo | None,
    calibration: CalibrationState | None,
    *,
    shake_enabled: bool = False,
    shake_offsets: Iterable[tuple[float, float]] | None = None,
    confirmed_type: ExperimentType | str | None = None,
    pendulum_length_m: float | None = None,
) -> str:
    points = []
    if result is not None:
        points = [
            [
                point.frame,
                round(point.x, 4),
                round(point.y, 4),
                bool(point.visible),
                round(point.confidence, 4),
                bool(point.manual),
            ]
            for point in result.points
        ]
    pts = []
    path = ""
    if info is not None:
        path = str(info.path)
        if info.pts_ms:
            pts = [int(info.pts_ms[0]), int(info.pts_ms[-1]), len(info.pts_ms)]
    confirmed = (
        confirmed_type.value
        if isinstance(confirmed_type, ExperimentType)
        else (None if confirmed_type is None else str(confirmed_type))
    )
    offsets = []
    if shake_offsets:
        offsets = [[round(dx, 3), round(dy, 3)] for dx, dy in shake_offsets]
    payload = {
        "path": path,
        "pts": pts,
        "points": points,
        "calibration": (calibration or CalibrationState()).to_dict(),
        "shake_enabled": bool(shake_enabled),
        "shake_offsets": offsets,
        "confirmed_type": confirmed,
        "pendulum_length_m": None
        if pendulum_length_m is None
        else round(float(pendulum_length_m), 6),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def analyze_experiment(
    result: TrackResult | None,
    info: VideoInfo | None,
    calibration: CalibrationState | None = None,
    *,
    clip_id: str = "",
    pendulum_length_m: float | None = None,
    period_hint: bool = False,
    force_type: ExperimentType | None = None,
    shake_enabled: bool = False,
    shake_offsets: Iterable[tuple[float, float]] | None = None,
    depth_audit: DepthAuditState | None = None,
) -> ExperimentAnalysis:
    cal = calibration or CalibrationState()
    clip = clip_id or (result.clip_id if result is not None else "")
    fingerprint = source_fingerprint(
        result,
        info,
        cal,
        shake_enabled=shake_enabled,
        shake_offsets=shake_offsets,
        pendulum_length_m=pendulum_length_m,
    )
    missing: list[str] = []
    warnings: list[str] = []
    if not cal.active:
        missing.append("calibration")
    if cal.camera_moved:
        warnings.append("检测到机位移动，平面标定可能失效")
    if cal.mode is CalibrationMode.PLANAR and cal.warning:
        warnings.append(cal.warning)
    if depth_audit is not None:
        off = depth_audit.off_plane_frames()
        if off:
            warnings.append(f"{len(off)} 帧可能离开运动平面")
        elif depth_audit.message:
            warnings.append(depth_audit.message)
    if result is None or info is None:
        warnings.append("缺少轨迹或视频索引，无法分析")
        return ExperimentAnalysis(
            clip_id=clip,
            fingerprint=fingerprint,
            calibration_active=cal.active,
            position_unit=cal.position_unit,
            speed_unit=cal.speed_unit,
            warnings=warnings,
            missing=missing,
        )

    samples = series_for_result(result, info, calibration=cal, depth_audit=depth_audit)
    visible = [item for item in samples if item.visible and item.x is not None and item.y is not None]
    coverage = (len(visible) / max(len(samples), 1)) if samples else 0.0
    mean_conf = float(np.mean([item.confidence for item in visible])) if visible else 0.0
    segments = contiguous_segments(samples, "x")
    segment = max(segments, key=len) if segments else []
    if len(segment) < MIN_LINEAR_POINTS:
        warnings.append("有效连续点不足，无法可靠识别实验类型")
        return ExperimentAnalysis(
            clip_id=clip,
            coverage=coverage,
            mean_track_confidence=mean_conf,
            calibration_active=cal.active,
            position_unit=cal.position_unit,
            speed_unit=cal.speed_unit,
            fingerprint=fingerprint,
            warnings=warnings,
            missing=missing,
        )

    cal_quality = _calibration_quality(cal)
    scored = _score_models(
        segment,
        cal,
        pendulum_length_m=pendulum_length_m,
        period_hint=period_hint,
        coverage=coverage,
        mean_conf=mean_conf,
        cal_quality=cal_quality,
    )
    scored.sort(key=lambda item: item.confidence, reverse=True)
    listed = [item for item in scored if item.fit.nrmse <= LIST_NRMSE]
    if not listed:
        listed = scored[:1]

    selected: ExperimentCandidate | None = None
    auto = False
    if force_type is not None and force_type is not ExperimentType.UNKNOWN:
        selected = next((item for item in scored if item.experiment_type is force_type), None)
        if selected is None:
            warnings.append("所选实验类型与当前数据不匹配")
    elif listed:
        top = listed[0]
        second = listed[1].confidence if len(listed) > 1 else 0.0
        margin = top.confidence - second
        enough = top.fit.n_samples >= MIN_QUAD_POINTS or (
            top.experiment_type is ExperimentType.UNIFORM_LINEAR
            and top.fit.n_samples >= MIN_LINEAR_POINTS
        )
        if (
            top.experiment_type is not ExperimentType.UNKNOWN
            and enough
            and top.fit.nrmse <= MAX_AUTO_NRMSE
            and top.confidence >= AUTO_CONFIRM_MIN
            and margin >= AUTO_CONFIRM_MARGIN
        ):
            selected = top
            auto = True
        else:
            selected = None
            if margin < AUTO_CONFIRM_MARGIN and len(listed) > 1:
                warnings.append("多个模型分数接近，需要人工确认实验类型")
            if top.fit.nrmse > MAX_AUTO_NRMSE:
                warnings.append("拟合残差偏大，拒绝自动确认")
            if not enough:
                warnings.append("样本不足，拒绝自动确认")

    if pendulum_length_m is None:
        has_pendulum = any(item.experiment_type is ExperimentType.PENDULUM for item in listed)
        if has_pendulum and "pendulum_length" not in missing:
            missing.append("pendulum_length")

    return ExperimentAnalysis(
        clip_id=clip,
        candidates=listed,
        selected=selected,
        auto_confirmable=auto,
        coverage=coverage,
        mean_track_confidence=mean_conf,
        calibration_active=cal.active,
        position_unit=cal.position_unit,
        speed_unit=cal.speed_unit,
        fingerprint=fingerprint,
        warnings=warnings,
        missing=missing,
    )


def physics_result_from_analysis(analysis: ExperimentAnalysis) -> PhysicsResult:
    candidate = analysis.selected
    if candidate is None and analysis.candidates:
        candidate = analysis.candidates[0]
    if candidate is None:
        return PhysicsResult(
            clip_id=analysis.clip_id,
            confidence=0.0,
            failure_reason=FailureReason.LOW_CONFIDENCE,
            notes="; ".join(analysis.warnings),
        )
    params = candidate.fit.parameters
    units = candidate.fit.units
    period = _si_value(params, units, "T", {"s"})
    gravity = _si_value(params, units, "g", {"m/s^2", "m/s²"})
    velocity = _si_value(params, units, "v", {"m/s"})
    if velocity is None:
        v0x = _si_value(params, units, "v0x", {"m/s"})
        v0y = _si_value(params, units, "v0y", {"m/s"})
        if v0x is not None and v0y is not None:
            velocity = float(math.hypot(v0x, v0y))
        elif v0x is not None:
            velocity = v0x
    acceleration = _si_value(params, units, "a", {"m/s^2", "m/s²"})
    if acceleration is None and gravity is not None and candidate.experiment_type in {
        ExperimentType.FREE_FALL,
        ExperimentType.PROJECTILE,
    }:
        acceleration = gravity
    return PhysicsResult(
        clip_id=analysis.clip_id,
        period_s=period,
        gravity_ms2=gravity,
        velocity_ms=velocity,
        acceleration_ms2=acceleration,
        trajectory_fit_error=float(candidate.fit.nrmse),
        confidence=float(candidate.confidence),
        notes="; ".join(candidate.evidence),
    )


def _si_value(
    params: dict[str, float | None],
    units: dict[str, str],
    name: str,
    accepted: set[str],
) -> float | None:
    value = params.get(name)
    if value is None:
        return None
    unit = units.get(name, "")
    if unit in accepted:
        return float(value)
    return None


def _calibration_quality(cal: CalibrationState) -> float:
    if not cal.active:
        return 0.4
    if cal.camera_moved:
        return 0.45
    if cal.warning:
        return 0.7
    return 1.0


def _score_models(
    segment: list[KinematicSample],
    cal: CalibrationState,
    *,
    pendulum_length_m: float | None,
    period_hint: bool,
    coverage: float,
    mean_conf: float,
    cal_quality: float,
) -> list[ExperimentCandidate]:
    t = np.array([item.time_s for item in segment], dtype=np.float64)
    x = np.array([item.x for item in segment], dtype=np.float64)
    y = np.array([item.y for item in segment], dtype=np.float64)
    w = np.array([max(item.confidence, 1e-3) for item in segment], dtype=np.float64)
    frames = [item.frame for item in segment]
    pos_unit = cal.position_unit
    speed_unit = cal.speed_unit
    accel_unit = "m/s^2" if cal.active else "px/s^2"
    span = _span(segment)
    out: list[ExperimentCandidate] = []

    uniform = _fit_uniform(t, x, y, w, frames, span, pos_unit, speed_unit)
    if uniform is not None:
        out.append(
            _finalize_candidate(
                ExperimentType.UNIFORM_LINEAR,
                uniform,
                coverage,
                mean_conf,
                cal_quality,
                evidence=["沿主运动方向线性拟合 s = s0 + v t"],
                missing=[] if cal.active else ["calibration"],
            )
        )

    accel = _fit_accel(t, x, y, w, frames, span, pos_unit, speed_unit, accel_unit)
    if accel is not None:
        warnings = []
        a = accel.parameters.get("a") or 0.0
        v = accel.parameters.get("v0") or accel.parameters.get("v") or 0.0
        duration = max(span[5] - span[4], 1e-9)
        if abs(v) > 1e-9 and abs(a) * duration / abs(v) < SMALL_ACCEL_RATIO:
            warnings.append("加速度相对速度变化很小，更接近匀速")
        out.append(
            _finalize_candidate(
                ExperimentType.UNIFORM_ACCEL,
                accel,
                coverage,
                mean_conf,
                cal_quality,
                evidence=["沿主运动方向二次拟合 s = s0 + v0 t + 1/2 a t^2"],
                warnings=warnings,
                missing=[] if cal.active else ["calibration"],
            )
        )

    free = _fit_free_fall(t, x, y, w, frames, span, pos_unit, speed_unit, accel_unit, cal.active)
    if free is not None:
        missing = [] if cal.active else ["calibration"]
        out.append(
            _finalize_candidate(
                ExperimentType.FREE_FALL,
                free,
                coverage,
                mean_conf,
                cal_quality,
                evidence=["水平漂移很小，竖直二次模型优于线性"],
                missing=missing,
            )
        )

    proj = _fit_projectile(t, x, y, w, frames, span, pos_unit, speed_unit, accel_unit, cal.active)
    if proj is not None:
        missing = [] if cal.active else ["calibration"]
        warnings = []
        r2x = proj.parameters.get("r2x") or 0.0
        if r2x < 0.98:
            warnings.append("镜头运动、透视或跟踪误差使像素速度不完全满足理想斜抛")
        out.append(
            _finalize_candidate(
                ExperimentType.PROJECTILE,
                proj,
                coverage,
                mean_conf,
                cal_quality,
                evidence=["x(t) 近线性且 y(t) 近二次"],
                missing=missing,
                warnings=warnings,
            )
        )

    pendulum = _fit_pendulum(
        t,
        x,
        y,
        w,
        frames,
        span,
        pos_unit,
        pendulum_length_m=pendulum_length_m,
        calibrated=cal.active,
    )
    if pendulum is not None:
        if period_hint:
            pendulum = FitResult(
                model=pendulum.model,
                formula_id=pendulum.formula_id,
                frame_start=pendulum.frame_start,
                frame_end=pendulum.frame_end,
                time_start_s=pendulum.time_start_s,
                time_end_s=pendulum.time_end_s,
                parameters=pendulum.parameters,
                units=pendulum.units,
                r2=min(1.0, pendulum.r2 + 0.01),
                nrmse=pendulum.nrmse,
                n_samples=pendulum.n_samples,
            )
        missing = []
        if not cal.active:
            missing.append("calibration")
        if pendulum.parameters.get("g") is None:
            missing.append("pendulum_length")
        candidate = _finalize_candidate(
            ExperimentType.PENDULUM,
            pendulum,
            coverage,
            mean_conf,
            cal_quality,
            evidence=["过零/自相关估计周期，并验证谐振或圆弧特征"],
            missing=missing,
        )
        if period_hint:
            candidate.confidence = min(1.0, candidate.confidence + PERIOD_HINT_BOOST)
        out.append(candidate)

    _nudge_uniform_vs_accel(out)
    _prefer_specific_models(out)
    return out


def _prefer_specific_models(candidates: list[ExperimentCandidate]) -> None:
    by_type = {item.experiment_type: item for item in candidates}
    if ExperimentType.FREE_FALL in by_type and ExperimentType.UNIFORM_ACCEL in by_type:
        by_type[ExperimentType.UNIFORM_ACCEL].confidence *= 0.52
        by_type[ExperimentType.UNIFORM_ACCEL].warnings.append(
            "水平漂移很小，一维匀加速不如自由落体模型具体"
        )
    if ExperimentType.PROJECTILE in by_type:
        if ExperimentType.UNIFORM_ACCEL in by_type:
            by_type[ExperimentType.UNIFORM_ACCEL].confidence *= 0.52
            by_type[ExperimentType.UNIFORM_ACCEL].warnings.append(
                "二维抛体模型同时解释了 x 近匀速和 y 近匀加速"
            )
        if ExperimentType.FREE_FALL in by_type:
            by_type[ExperimentType.FREE_FALL].confidence *= 0.6
    if ExperimentType.PENDULUM in by_type:
        for kind in (
            ExperimentType.UNIFORM_LINEAR,
            ExperimentType.UNIFORM_ACCEL,
            ExperimentType.PROJECTILE,
            ExperimentType.FREE_FALL,
        ):
            if kind in by_type:
                by_type[kind].confidence *= 0.55
    for item in candidates:
        item.confidence = float(max(0.0, min(item.confidence, 0.99)))


def _nudge_uniform_vs_accel(candidates: list[ExperimentCandidate]) -> None:
    uniform = next((c for c in candidates if c.experiment_type is ExperimentType.UNIFORM_LINEAR), None)
    accel = next((c for c in candidates if c.experiment_type is ExperimentType.UNIFORM_ACCEL), None)
    if uniform is None or accel is None:
        return
    a = abs(accel.fit.parameters.get("a") or 0.0)
    duration = max(accel.fit.time_end_s - accel.fit.time_start_s, 1e-9)
    v = abs(accel.fit.parameters.get("v0") or accel.fit.parameters.get("v") or 0.0)
    weak_accel = a * duration <= SMALL_ACCEL_RATIO * max(v, 1e-6)
    improve = accel.fit.r2 - uniform.fit.r2
    if weak_accel or improve < 0.015:
        accel.confidence *= 0.72
        accel.warnings.append("二次项增益不足，优先考虑匀速")
    elif improve >= 0.03 and accel.fit.nrmse <= uniform.fit.nrmse + 0.01:
        uniform.confidence *= 0.8


def _finalize_candidate(
    kind: ExperimentType,
    fit: FitResult,
    coverage: float,
    mean_conf: float,
    cal_quality: float,
    *,
    evidence: list[str],
    warnings: list[str] | None = None,
    missing: list[str] | None = None,
) -> ExperimentCandidate:
    fit_term = 0.45 * max(0.0, fit.r2) + 0.25 * max(0.0, 1.0 - min(fit.nrmse / 0.10, 1.0))
    confidence = (
        fit_term
        + 0.12 * coverage
        + 0.10 * mean_conf
        + 0.08 * cal_quality
    )
    confidence = float(max(0.0, min(confidence, 0.99)))
    return ExperimentCandidate(
        experiment_type=kind,
        label=EXPERIMENT_LABELS[kind],
        confidence=confidence,
        fit=fit,
        evidence=evidence,
        warnings=list(warnings or []),
        missing=list(missing or []),
    )


def projectile_velocity_model(
    samples: list[KinematicSample],
    *,
    calibration: CalibrationState | None = None,
) -> FitResult | None:
    """Fit x=x0+v0x t, y=y0+v0y t+0.5 ay t^2 on the longest visible run."""
    cal = calibration or CalibrationState()
    segments = contiguous_segments(samples, "x")
    segment = max(segments, key=len) if segments else []
    if len(segment) < MIN_QUAD_POINTS:
        return None
    t = np.array([item.time_s for item in segment], dtype=np.float64)
    x = np.array([item.x for item in segment], dtype=np.float64)
    y = np.array([item.y for item in segment], dtype=np.float64)
    w = np.array([max(item.confidence, 1e-3) for item in segment], dtype=np.float64)
    frames = [item.frame for item in segment]
    accel_unit = "m/s^2" if cal.active else "px/s^2"
    return _fit_projectile(
        t,
        x,
        y,
        w,
        frames,
        _span(segment),
        cal.position_unit,
        cal.speed_unit,
        accel_unit,
        cal.active,
    )


def _span(segment: list[KinematicSample]) -> tuple[int, int, float, float, float, float]:
    return (
        segment[0].frame,
        segment[-1].frame,
        float(segment[0].x or 0.0),
        float(segment[0].y or 0.0),
        segment[0].time_s,
        segment[-1].time_s,
    )


def _fit_uniform(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    frames: list[int],
    span: tuple[int, int, float, float, float, float],
    pos_unit: str,
    speed_unit: str,
) -> FitResult | None:
    if len(t) < MIN_LINEAR_POINTS:
        return None
    ux, uy = _principal_axis(x, y)
    s = x * ux + y * uy
    coef, pred = _weighted_polyfit(t, s, w, 1)
    r2 = _r2(s, pred, w)
    nrmse = _nrmse(s, pred, w)
    return FitResult(
        model="s=s0+vt",
        formula_id="uniform.s",
        frame_start=span[0],
        frame_end=span[1],
        time_start_s=span[4],
        time_end_s=span[5],
        parameters={"s0": float(coef[0]), "v": float(coef[1]), "ux": ux, "uy": uy},
        units={"s0": pos_unit, "v": speed_unit, "ux": "", "uy": ""},
        r2=r2,
        nrmse=nrmse,
        n_samples=len(t),
    )


def _fit_accel(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    frames: list[int],
    span: tuple[int, int, float, float, float, float],
    pos_unit: str,
    speed_unit: str,
    accel_unit: str,
) -> FitResult | None:
    if len(t) < MIN_QUAD_POINTS:
        return None
    ux, uy = _principal_axis(x, y)
    s = x * ux + y * uy
    coef, pred = _weighted_polyfit(t, s, w, 2)
    r2 = _r2(s, pred, w)
    nrmse = _nrmse(s, pred, w)
    a = float(2.0 * coef[2])
    return FitResult(
        model="s=s0+v0t+0.5at^2",
        formula_id="accel.s",
        frame_start=span[0],
        frame_end=span[1],
        time_start_s=span[4],
        time_end_s=span[5],
        parameters={
            "s0": float(coef[0]),
            "v0": float(coef[1]),
            "a": a,
            "ux": ux,
            "uy": uy,
        },
        units={"s0": pos_unit, "v0": speed_unit, "a": accel_unit, "ux": "", "uy": ""},
        r2=r2,
        nrmse=nrmse,
        n_samples=len(t),
    )


def _fit_free_fall(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    frames: list[int],
    span: tuple[int, int, float, float, float, float],
    pos_unit: str,
    speed_unit: str,
    accel_unit: str,
    calibrated: bool,
) -> FitResult | None:
    if len(t) < MIN_QUAD_POINTS:
        return None
    span_x = float(np.max(x) - np.min(x))
    span_y = float(np.max(y) - np.min(y))
    if span_y < 1e-9:
        return None
    if span_x / span_y > FREEFALL_DRIFT:
        return None
    lin_coef, lin_pred = _weighted_polyfit(t, y, w, 1)
    quad_coef, quad_pred = _weighted_polyfit(t, y, w, 2)
    r2_lin = _r2(y, lin_pred, w)
    r2 = _r2(y, quad_pred, w)
    if r2 < r2_lin + 0.03:
        return None
    nrmse = _nrmse(y, quad_pred, w)
    a_y = float(2.0 * quad_coef[2])
    g = abs(a_y)
    params: dict[str, float | None] = {
        "y0": float(quad_coef[0]),
        "v0y": float(quad_coef[1]),
        "g": g if calibrated else None,
        "a_y": a_y,
    }
    units = {
        "y0": pos_unit,
        "v0y": speed_unit,
        "g": "m/s^2" if calibrated else "",
        "a_y": accel_unit,
    }
    _ = lin_coef
    return FitResult(
        model="y=y0+v0yt+0.5gt^2",
        formula_id="freefall.y",
        frame_start=span[0],
        frame_end=span[1],
        time_start_s=span[4],
        time_end_s=span[5],
        parameters=params,
        units=units,
        r2=r2,
        nrmse=nrmse,
        n_samples=len(t),
    )


def _fit_projectile(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    frames: list[int],
    span: tuple[int, int, float, float, float, float],
    pos_unit: str,
    speed_unit: str,
    accel_unit: str,
    calibrated: bool,
) -> FitResult | None:
    if len(t) < MIN_QUAD_POINTS:
        return None
    span_x = float(np.max(x) - np.min(x))
    span_y = float(np.max(y) - np.min(y))
    if span_x < 0.15 * max(span_y, 1e-9):
        return None
    x_coef, x_pred = _weighted_polyfit(t, x, w, 1)
    y_lin, y_lin_pred = _weighted_polyfit(t, y, w, 1)
    y_coef, y_pred = _weighted_polyfit(t, y, w, 2)
    r2x = _r2(x, x_pred, w)
    r2y = _r2(y, y_pred, w)
    r2y_lin = _r2(y, y_lin_pred, w)
    if r2x < 0.92 or r2y < 0.90 or r2y < r2y_lin + 0.02:
        return None
    nrmse = 0.5 * _nrmse(x, x_pred, w) + 0.5 * _nrmse(y, y_pred, w)
    a_y = float(2.0 * y_coef[2])
    g = abs(a_y)
    params: dict[str, float | None] = {
        "x0": float(x_coef[0]),
        "v0x": float(x_coef[1]),
        "y0": float(y_coef[0]),
        "v0y": float(y_coef[1]),
        "g": g if calibrated else None,
        "a_y": a_y,
        "r2x": float(r2x),
        "r2y": float(r2y),
    }
    units = {
        "x0": pos_unit,
        "v0x": speed_unit,
        "y0": pos_unit,
        "v0y": speed_unit,
        "g": "m/s^2" if calibrated else "",
        "a_y": accel_unit,
    }
    _ = y_lin
    return FitResult(
        model="x=x0+v0xt; y=y0+v0yt+0.5gt^2",
        formula_id="projectile.xy",
        frame_start=span[0],
        frame_end=span[1],
        time_start_s=span[4],
        time_end_s=span[5],
        parameters=params,
        units=units,
        r2=float(min(r2x, r2y)),
        nrmse=nrmse,
        n_samples=len(t),
    )


def _fit_pendulum(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    frames: list[int],
    span: tuple[int, int, float, float, float, float],
    pos_unit: str,
    *,
    pendulum_length_m: float | None,
    calibrated: bool,
) -> FitResult | None:
    if len(t) < MIN_PENDULUM_POINTS:
        return None
    if _looks_circular_orbit(x, y):
        return None
    period = _estimate_period(t, x)
    if period is None or period <= 0:
        period = _estimate_period(t, y)
    if period is None or period <= 0:
        return None
    duration = float(t[-1] - t[0])
    if duration < 0.6 * period:
        return None
    pred_x, r2x = _harmonic_fit(t, x, w, period)
    pred_y, r2y = _harmonic_fit(t, y, w, period)
    span_x = float(np.max(x) - np.min(x))
    span_y = float(np.max(y) - np.min(y))
    primary = "x" if span_x >= span_y else "y"
    r2 = r2x if primary == "x" else r2y
    pred = pred_x if primary == "x" else pred_y
    series = x if primary == "x" else y
    if r2 < 0.75:
        return None
    nrmse = _nrmse(series, pred, w)
    g = None
    length = pendulum_length_m
    if length is not None and length > 0 and period > 0:
        g = float(4.0 * math.pi**2 * length / (period**2))
    params: dict[str, float | None] = {
        "T": float(period),
        "L": None if length is None else float(length),
        "g": g,
    }
    units = {"T": "s", "L": "m", "g": "m/s^2" if g is not None else ""}
    _ = (pred_y, r2y, pos_unit, calibrated)
    return FitResult(
        model="x=A cos(2πt/T + φ) + C",
        formula_id="pendulum.period",
        frame_start=span[0],
        frame_end=span[1],
        time_start_s=span[4],
        time_end_s=span[5],
        parameters=params,
        units=units,
        r2=float(r2),
        nrmse=nrmse,
        n_samples=len(t),
    )


def _looks_circular_orbit(x: np.ndarray, y: np.ndarray) -> bool:
    span_x = float(np.max(x) - np.min(x))
    span_y = float(np.max(y) - np.min(y))
    if span_x < 1e-9 or span_y < 1e-9:
        return False
    ratio = min(span_x, span_y) / max(span_x, span_y)
    if ratio < 0.45:
        return False
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denom = float(np.sqrt(np.mean(xc * xc) * np.mean(yc * yc)))
    if denom < 1e-12:
        return False
    corr = abs(float(np.mean(xc * yc) / denom))
    return corr < 0.35


def _estimate_period(t: np.ndarray, series: np.ndarray) -> float | None:
    centered = series - np.mean(series)
    if float(np.max(np.abs(centered))) < 1e-9:
        return None
    guesses: list[float] = []
    auto = _autocorr_period(t, centered)
    if auto is not None:
        guesses.append(auto)
    zero = _zero_crossing_period(t, centered)
    if zero is not None:
        guesses.append(zero)
    if not guesses:
        return None
    return float(np.median(guesses))


def _autocorr_period(t: np.ndarray, centered: np.ndarray) -> float | None:
    if len(t) < 8:
        return None
    dt = float(np.median(np.diff(t)))
    if dt <= 1e-9:
        return None
    grid = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    if len(grid) < 8:
        return None
    sampled = np.interp(grid, t, centered)
    sampled = sampled - np.mean(sampled)
    corr = np.correlate(sampled, sampled, mode="full")
    corr = corr[len(corr) // 2 :]
    if corr[0] <= 0:
        return None
    min_lag = max(2, int(0.08 / dt))
    peak_i = None
    peak_v = 0.0
    for i in range(min_lag, len(corr) - 1):
        value = float(corr[i])
        if value >= corr[i - 1] and value >= corr[i + 1] and value > 0.25 * float(corr[0]):
            if value > peak_v:
                peak_v = value
                peak_i = i
                break
    if peak_i is None:
        return None
    return float(peak_i * dt)


def _zero_crossing_period(t: np.ndarray, centered: np.ndarray) -> float | None:
    crossings = [
        i
        for i in range(1, len(centered))
        if centered[i - 1] <= 0 < centered[i] or centered[i - 1] >= 0 > centered[i]
    ]
    if len(crossings) < 3:
        return None
    halves = [float(t[crossings[i + 1]] - t[crossings[i]]) for i in range(len(crossings) - 1)]
    halves = [item for item in halves if item > 1e-6]
    if not halves:
        return None
    return float(2.0 * np.median(halves))


def _harmonic_fit(
    t: np.ndarray, series: np.ndarray, w: np.ndarray, period: float
) -> tuple[np.ndarray, float]:
    omega = 2.0 * math.pi / period
    tau = t - t[0]
    A = np.column_stack([np.cos(omega * tau), np.sin(omega * tau), np.ones_like(tau)])
    sw = np.sqrt(np.clip(w, 1e-6, None))
    coef, *_ = np.linalg.lstsq(A * sw[:, None], series * sw, rcond=None)
    pred = A @ coef
    return pred, _r2(series, pred, w)


def _principal_axis(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    norm = math.hypot(dx, dy)
    if norm > 1e-9:
        return dx / norm, dy / norm
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    cov = np.cov(np.vstack((xc, yc)))
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    return float(axis[0]), float(axis[1])


def _weighted_polyfit(
    t: np.ndarray, y: np.ndarray, w: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    tau = t - t[0]
    design = np.column_stack([tau**k for k in range(degree + 1)])
    sw = np.sqrt(np.clip(w, 1e-6, None))
    coef, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    return coef, design @ coef


def _r2(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    weights = np.clip(w, 1e-6, None)
    mean = float(np.average(y, weights=weights))
    ss_res = float(np.sum(weights * (y - pred) ** 2))
    ss_tot = float(np.sum(weights * (y - mean) ** 2))
    if ss_tot < 1e-18:
        return 1.0 if ss_res < 1e-18 else 0.0
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))


def _nrmse(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    weights = np.clip(w, 1e-6, None)
    rmse = float(np.sqrt(np.average((y - pred) ** 2, weights=weights)))
    span = float(np.max(y) - np.min(y))
    if span < 1e-12:
        return 0.0 if rmse < 1e-12 else 1.0
    return float(rmse / span)
