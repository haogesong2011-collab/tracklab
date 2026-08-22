"""Evaluation metrics for tracking, pose, calibration, and physics results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from ai.contracts import CalibrationResult, PhysicsResult, PoseResult, TrackResult
from ai.schema import (
    POSE_BONES,
    CalibrationGT,
    ClipAnnotation,
    Difficulty,
    PhysicsGT,
    PoseFrameGT,
    TrackFrameGT,
)


@dataclass
class MetricSummary:
    name: str
    value: float | None
    threshold: float | None = None
    higher_is_better: bool = True
    passed: bool | None = None
    details: dict = field(default_factory=dict)

    def evaluate(self) -> "MetricSummary":
        if self.value is None or self.threshold is None:
            self.passed = None
            return self
        if self.higher_is_better:
            self.passed = self.value >= self.threshold
        else:
            self.passed = self.value <= self.threshold
        return self


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def track_metrics(
    pred: TrackResult,
    gt_frames: list[TrackFrameGT],
    *,
    difficulty: Difficulty,
) -> list[MetricSummary]:
    gt_by = {f.frame: f for f in gt_frames}
    errors: list[float] = []
    visible_gt = 0
    covered = 0
    identity_swaps = 0
    recovery_gaps: list[int] = []
    last_visible_ok = True
    gap = 0

    pred_by = {p.frame: p for p in pred.points}
    for frame, gt in sorted(gt_by.items()):
        if not gt.visible:
            if not last_visible_ok:
                gap += 1
            continue
        visible_gt += 1
        p = pred_by.get(frame)
        if p is None or not p.visible:
            last_visible_ok = False
            gap += 1
            continue
        err = math.hypot(p.x - gt.center.x, p.y - gt.center.y)
        errors.append(err)
        covered += 1
        if not last_visible_ok and gap > 0:
            recovery_gaps.append(gap)
            gap = 0
        last_visible_ok = True
        # Crude identity-swap detector: sudden jump > 40 px while GT moved < 15.
        if frame > 0 and (frame - 1) in pred_by and (frame - 1) in gt_by:
            prev_p = pred_by[frame - 1]
            prev_g = gt_by[frame - 1]
            if prev_p.visible and prev_g.visible:
                dp = math.hypot(p.x - prev_p.x, p.y - prev_p.y)
                dg = math.hypot(gt.center.x - prev_g.center.x, gt.center.y - prev_g.center.y)
                if dp > 40 and dg < 15:
                    identity_swaps += 1

    success_10 = (
        sum(1 for e in errors if e <= 10.0) / len(errors) if errors else 0.0
    )
    completeness = covered / visible_gt if visible_gt else 0.0
    drift = _mean(errors[-max(1, len(errors) // 5) :]) if errors else None
    recovery = _median(recovery_gaps) if recovery_gaps else 0.0

    if difficulty == Difficulty.NORMAL:
        med_th, suc_th, comp_th = 3.0, 0.95, 0.98
    else:
        med_th, suc_th, comp_th = 8.0, 0.80, 0.90

    return [
        MetricSummary("center_median_px", _median(errors), med_th, False).evaluate(),
        MetricSummary("success_at_10px", success_10, suc_th, True).evaluate(),
        MetricSummary("trajectory_completeness", completeness, comp_th, True).evaluate(),
        MetricSummary("drift_tail_mean_px", drift, None, False).evaluate(),
        MetricSummary(
            "occlusion_recovery_frames",
            float(recovery) if recovery is not None else 0.0,
            10.0 if difficulty == Difficulty.HARD else None,
            False,
        ).evaluate(),
        MetricSummary("identity_swaps", float(identity_swaps), 1.0, False).evaluate(),
    ]


def pose_metrics(
    pred: PoseResult,
    gt_frames: list[PoseFrameGT],
    *,
    difficulty: Difficulty,
    torso_scale_fallback: float = 50.0,
) -> list[MetricSummary]:
    gt_by = {f.frame: f for f in gt_frames}
    hits = 0
    total = 0
    misses = 0
    jitter_samples: list[float] = []
    bone_cvs: list[float] = []

    pred_by = {f.frame: f for f in pred.frames}
    # PCK
    for frame, gt in gt_by.items():
        pf = pred_by.get(frame)
        gt_map = {k.name: k for k in gt.keypoints}
        # torso scale from shoulders/hips when available
        scale = torso_scale_fallback
        if "left_shoulder" in gt_map and "left_hip" in gt_map:
            scale = max(
                math.hypot(
                    gt_map["left_shoulder"].x - gt_map["left_hip"].x,
                    gt_map["left_shoulder"].y - gt_map["left_hip"].y,
                ),
                1.0,
            )
        thresh = 0.05 * scale
        for kp in gt.keypoints:
            if not kp.visible:
                continue
            total += 1
            if pf is None:
                misses += 1
                continue
            pred_map = {k.name: k for k in pf.keypoints}
            pk = pred_map.get(kp.name)
            if pk is None or not pk.visible:
                misses += 1
                continue
            if math.hypot(pk.x - kp.x, pk.y - kp.y) <= thresh:
                hits += 1
            else:
                misses += 1

    # Static jitter: consecutive predicted frames where GT barely moves.
    sorted_gt = sorted(gt_frames, key=lambda f: f.frame)
    for ga, gb in zip(sorted_gt, sorted_gt[1:]):
        am_gt = {k.name: k for k in ga.keypoints}
        bm_gt = {k.name: k for k in gb.keypoints}
        gt_motion = []
        for name in am_gt:
            if name in bm_gt and am_gt[name].visible and bm_gt[name].visible:
                gt_motion.append(
                    math.hypot(
                        am_gt[name].x - bm_gt[name].x,
                        am_gt[name].y - bm_gt[name].y,
                    )
                )
        if not gt_motion or (_mean(gt_motion) or 0) > 1.0:
            continue
        pa, pb = pred_by.get(ga.frame), pred_by.get(gb.frame)
        if pa is None or pb is None:
            continue
        am = {k.name: k for k in pa.keypoints}
        bm = {k.name: k for k in pb.keypoints}
        for name in am:
            if name in bm and am[name].visible and bm[name].visible:
                jitter_samples.append(
                    math.hypot(am[name].x - bm[name].x, am[name].y - bm[name].y)
                )

    # Bone length coefficient of variation.
    for left, right in POSE_BONES:
        lengths: list[float] = []
        for pf in pred.frames:
            m = {k.name: k for k in pf.keypoints}
            if left in m and right in m and m[left].visible and m[right].visible:
                lengths.append(
                    math.hypot(m[left].x - m[right].x, m[left].y - m[right].y)
                )
        if len(lengths) >= 3:
            mu = sum(lengths) / len(lengths)
            if mu > 1e-6:
                var = sum((x - mu) ** 2 for x in lengths) / len(lengths)
                bone_cvs.append(math.sqrt(var) / mu)

    pck = hits / total if total else 0.0
    miss_rate = misses / total if total else 0.0
    pck_th = 0.95 if difficulty == Difficulty.NORMAL else 0.85

    recovery_gaps: list[int] = []
    gap = 0
    waiting = False
    for ga, gb in zip(sorted_gt, sorted_gt[1:]):
        vis_a = any(k.visible for k in ga.keypoints)
        vis_b = any(k.visible for k in gb.keypoints)
        pb = pred_by.get(gb.frame)
        pred_vis = bool(pb and any(k.visible for k in pb.keypoints))
        if vis_a and not vis_b:
            waiting = True
            gap = 0
        elif waiting:
            gap += 1
            if vis_b and pred_vis:
                recovery_gaps.append(gap)
                waiting = False
                gap = 0

    return [
        MetricSummary("pck_at_0_05", pck, pck_th, True).evaluate(),
        MetricSummary("keypoint_miss_rate", miss_rate, None, False).evaluate(),
        MetricSummary(
            "static_jitter_px", _median(jitter_samples), 2.0, False
        ).evaluate(),
        MetricSummary(
            "bone_length_cv", _mean(bone_cvs), 0.03, False
        ).evaluate(),
        MetricSummary(
            "occlusion_recovery_frames",
            float(_median(recovery_gaps) or 0.0),
            10.0 if difficulty == Difficulty.HARD else None,
            False,
        ).evaluate(),
    ]


def calibration_metrics(
    pred: CalibrationResult,
    gt: CalibrationGT,
    *,
    difficulty: Difficulty,
) -> list[MetricSummary]:
    if not gt.has_reliable_ruler:
        # Must reject / low confidence — not invent a scale.
        rejected = pred.rejected or pred.failure_reason.value in {
            "no_ruler",
            "low_confidence",
        }
        return [
            MetricSummary(
                "rejection_when_no_ruler", 1.0 if rejected else 0.0, 1.0, True
            ).evaluate()
        ]

    gt_dist = math.hypot(gt.ruler_b.x - gt.ruler_a.x, gt.ruler_b.y - gt.ruler_a.y)
    gt_ppm = gt_dist / gt.length_m if gt.length_m > 0 else None

    if pred.rejected or pred.pixels_per_meter is None or gt_ppm is None:
        return [
            MetricSummary("scale_relative_error", None, None).evaluate(),
            MetricSummary("false_reject", 1.0, 0.0, False).evaluate(),
        ]

    scale_err = abs(pred.pixels_per_meter - gt_ppm) / gt_ppm
    origin_err = None
    if pred.origin_x is not None and pred.origin_y is not None:
        origin_err = math.hypot(pred.origin_x - gt.origin.x, pred.origin_y - gt.origin.y)
    angle_err = None
    if pred.axis_angle_deg is not None:
        angle_err = abs(((pred.axis_angle_deg - gt.axis_angle_deg + 180) % 360) - 180)

    if difficulty == Difficulty.NORMAL:
        s_th, a_th, o_th = 0.01, 1.0, 3.0
    else:
        s_th, a_th, o_th = 0.03, 3.0, 8.0

    return [
        MetricSummary("scale_relative_error", scale_err, s_th, False).evaluate(),
        MetricSummary("axis_angle_error_deg", angle_err, a_th, False).evaluate(),
        MetricSummary("origin_error_px", origin_err, o_th, False).evaluate(),
    ]


def physics_metrics(pred: PhysicsResult, gt: PhysicsGT) -> list[MetricSummary]:
    out: list[MetricSummary] = []

    def rel(name: str, p: float | None, g: float | None, th: float) -> None:
        if p is None or g is None or abs(g) < 1e-9:
            out.append(MetricSummary(name, None, th, False).evaluate())
            return
        out.append(MetricSummary(name, abs(p - g) / abs(g), th, False).evaluate())

    rel("period_relative_error", pred.period_s, gt.period_s, 0.02)
    rel("gravity_relative_error", pred.gravity_ms2, gt.gravity_ms2, 0.05)
    rel("velocity_relative_error", pred.velocity_ms, gt.velocity_ms, 0.05)
    rel("acceleration_relative_error", pred.acceleration_ms2, gt.acceleration_ms2, 0.05)
    if pred.trajectory_fit_error is not None:
        out.append(
            MetricSummary(
                "trajectory_fit_error", pred.trajectory_fit_error, 0.05, False
            ).evaluate()
        )
    return out


def summarize_pass(metrics: Iterable[MetricSummary]) -> bool:
    judged = [m for m in metrics if m.passed is not None]
    return all(m.passed for m in judged) if judged else False


def thresholds_for(ann: ClipAnnotation) -> dict[str, float]:
    """Documented gates for reporting."""
    if ann.difficulty == Difficulty.NORMAL:
        return {
            "center_median_px": 3.0,
            "success_at_10px": 0.95,
            "trajectory_completeness": 0.98,
            "pck_at_0_05": 0.95,
            "scale_relative_error": 0.01,
        }
    return {
        "center_median_px": 8.0,
        "success_at_10px": 0.80,
        "trajectory_completeness": 0.90,
        "pck_at_0_05": 0.85,
        "scale_relative_error": 0.03,
    }
