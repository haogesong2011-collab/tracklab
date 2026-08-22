"""UI-free evaluator: run any candidate model against the frozen manifest."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import CancelToken  # noqa: E402
from ai.models import (  # noqa: E402
    ColorBlobTracker,
    EdgeRulerCalibrator,
    OracleCalibrator,
    OraclePose,
    OracleTracker,
    TemplatePose,
    derive_physics_from_track,
    load_video,
)
from ai.schema import TaskKind  # noqa: E402
from tests.ai.dataset import load_annotation, load_manifest, resolve_video  # noqa: E402
from tests.ai.metrics import (  # noqa: E402
    MetricSummary,
    calibration_metrics,
    physics_metrics,
    pose_metrics,
    summarize_pass,
    track_metrics,
)

RESULTS_DIR = ROOT / "tests" / "ai" / "results"
BASELINE_DIR = ROOT / "tests" / "ai" / "baselines"


@dataclass
class ClipReport:
    clip_id: str
    task: str
    difficulty: str
    split: str
    model: str
    passed: bool
    metrics: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    created_at: float
    machine: dict[str, str]
    model_bundle: str
    split: str
    seed: int
    clips: list[ClipReport]
    pass_rate: float
    regression: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "machine": self.machine,
            "model_bundle": self.model_bundle,
            "split": self.split,
            "seed": self.seed,
            "pass_rate": self.pass_rate,
            "clips": [c.to_dict() for c in self.clips],
            "regression": self.regression,
        }


def _machine_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "unknown",
    }
    try:
        import importlib.metadata as metadata

        for pkg in ("av", "numpy", "PySide6"):
            try:
                info[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                info[pkg] = "missing"
    except Exception:  # noqa: BLE001
        pass
    return info


def _metrics_to_dicts(metrics: list[MetricSummary]) -> list[dict[str, Any]]:
    return [
        {
            "name": m.name,
            "value": m.value,
            "threshold": m.threshold,
            "higher_is_better": m.higher_is_better,
            "passed": m.passed,
            "details": m.details,
        }
        for m in metrics
    ]


def evaluate_clip(entry, model_bundle: str, seed: int = 0) -> ClipReport:
    ann = load_annotation(entry.annotation_path)
    video = resolve_video(entry)
    info = load_video(video)
    token = CancelToken()
    t0 = time.perf_counter()
    try:
        if entry.task == TaskKind.TRACK:
            if model_bundle == "oracle":
                model = OracleTracker(ann)
            elif model_bundle == "sam2":
                from ai.sam2_tracker import Sam2Tracker  # noqa: PLC0415

                model = Sam2Tracker()
            else:
                model = ColorBlobTracker()
            seed_xy = (ann.track[0].center.x, ann.track[0].center.y)
            result = model.track(info, seed_xy, cancel=token)
            metrics = track_metrics(result, ann.track, difficulty=ann.difficulty)
            model_name = result.model_name
            elapsed = result.elapsed_s
        elif entry.task == TaskKind.POSE:
            model = OraclePose(ann) if model_bundle == "oracle" else TemplatePose()
            result = model.estimate(info, cancel=token)
            metrics = pose_metrics(result, ann.pose, difficulty=ann.difficulty)
            model_name = result.model_name
            elapsed = result.elapsed_s
        elif entry.task == TaskKind.CALIBRATION:
            model = (
                OracleCalibrator(ann)
                if model_bundle == "oracle"
                else EdgeRulerCalibrator()
            )
            length = ann.calibration.length_m if ann.calibration else None
            result = model.calibrate(info, expected_length_m=length, cancel=token)
            assert ann.calibration is not None
            metrics = calibration_metrics(
                result, ann.calibration, difficulty=ann.difficulty
            )
            model_name = result.model_name
            elapsed = result.elapsed_s
        elif entry.task == TaskKind.PHYSICS:
            if model_bundle == "oracle":
                tracker = OracleTracker(ann)
            elif model_bundle == "sam2":
                from ai.sam2_tracker import Sam2Tracker  # noqa: PLC0415

                tracker = Sam2Tracker()
            else:
                tracker = ColorBlobTracker()
            seed_xy = (ann.track[0].center.x, ann.track[0].center.y)
            track = tracker.track(info, seed_xy, cancel=token)
            ppm = None
            if ann.calibration and ann.calibration.has_reliable_ruler:
                d = (
                    (ann.calibration.ruler_b.x - ann.calibration.ruler_a.x) ** 2
                    + (ann.calibration.ruler_b.y - ann.calibration.ruler_a.y) ** 2
                ) ** 0.5
                ppm = d / ann.calibration.length_m
            phys = derive_physics_from_track(
                ann.clip_id,
                track,
                ann.fps,
                pixels_per_meter=ppm,
                length_m=ann.calibration.length_m if ann.calibration else None,
                period_hint=True,
            )
            assert ann.physics is not None
            metrics = physics_metrics(phys, ann.physics)
            # Also require the underlying track to be decent.
            metrics.extend(track_metrics(track, ann.track, difficulty=ann.difficulty))
            model_name = f"{tracker.name}+physics"
            elapsed = track.elapsed_s
        else:
            raise ValueError(f"unknown task {entry.task}")
        return ClipReport(
            clip_id=entry.clip_id,
            task=entry.task.value,
            difficulty=entry.difficulty.value,
            split=entry.split,
            model=model_name,
            passed=summarize_pass(metrics),
            metrics=_metrics_to_dicts(metrics),
            elapsed_s=elapsed or (time.perf_counter() - t0),
        )
    except Exception as exc:  # noqa: BLE001 — report and continue suite
        return ClipReport(
            clip_id=entry.clip_id,
            task=entry.task.value,
            difficulty=entry.difficulty.value,
            split=entry.split,
            model=model_bundle,
            passed=False,
            elapsed_s=time.perf_counter() - t0,
            error=str(exc),
        )


CORE_METRICS = {
    "center_median_px",
    "success_at_10px",
    "trajectory_completeness",
    "pck_at_0_05",
    "scale_relative_error",
    "axis_angle_error_deg",
    "origin_error_px",
    "period_relative_error",
    "gravity_relative_error",
    "rejection_when_no_ruler",
}


def compare_to_baseline(
    report: EvalReport, baseline_path: Path, *, metric_tol: float = 0.02, perf_tol: float = 0.10
) -> dict[str, Any]:
    if not baseline_path.exists():
        return {"status": "no_baseline", "blocking": False}
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_by = {c["clip_id"]: c for c in base.get("clips", [])}
    regressions: list[dict[str, Any]] = []
    for clip in report.clips:
        prev = base_by.get(clip.clip_id)
        if not prev:
            continue
        if prev.get("passed") and not clip.passed:
            regressions.append({"clip_id": clip.clip_id, "reason": "new_failure"})
            continue
        prev_metrics = {m["name"]: m for m in prev.get("metrics", [])}
        for m in clip.metrics:
            if m["name"] not in CORE_METRICS:
                continue
            pm = prev_metrics.get(m["name"])
            if not pm or m["value"] is None or pm.get("value") is None:
                continue
            if m.get("passed") is True and pm.get("passed") is True:
                continue
            old, new = float(pm["value"]), float(m["value"])
            higher = bool(m.get("higher_is_better", True))
            denom = max(abs(old), 1e-3)
            if higher:
                if (old - new) / denom > metric_tol:
                    regressions.append(
                        {
                            "clip_id": clip.clip_id,
                            "metric": m["name"],
                            "old": old,
                            "new": new,
                            "reason": "metric_drop",
                        }
                    )
            elif (new - old) / denom > metric_tol:
                regressions.append(
                    {
                        "clip_id": clip.clip_id,
                        "metric": m["name"],
                        "old": old,
                        "new": new,
                        "reason": "metric_worsen",
                    }
                )
        if prev.get("elapsed_s") and clip.elapsed_s:
            old_t = float(prev["elapsed_s"])
            new_t = float(clip.elapsed_s)
            if old_t >= 0.5 and new_t > old_t * (1.0 + perf_tol):
                regressions.append(
                    {
                        "clip_id": clip.clip_id,
                        "reason": "perf_regression",
                        "old": old_t,
                        "new": new_t,
                    }
                )
    return {
        "status": "compared",
        "blocking": bool(regressions),
        "count": len(regressions),
        "items": regressions,
    }


def run_eval(
    *,
    split: str = "ci",
    model_bundle: str = "baseline",
    seed: int = 0,
    include_holdout: bool = False,
    save_baseline: bool = False,
) -> EvalReport:
    manifest = load_manifest()
    entries = [
        e
        for e in manifest.entries
        if e.split == split or (split == "all" and (include_holdout or not e.hidden))
    ]
    if split == "ci":
        # Unique videos only (expanded slots reuse paths).
        seen: set[str] = set()
        unique = []
        for e in entries:
            if e.clip_id in seen:
                continue
            # Prefer original 10 clip ids without _slot suffix.
            if "_slot" in e.clip_id:
                continue
            seen.add(e.clip_id)
            unique.append(e)
        entries = unique

    clips = [evaluate_clip(e, model_bundle=model_bundle, seed=seed) for e in entries]
    passed = sum(1 for c in clips if c.passed)
    report = EvalReport(
        created_at=time.time(),
        machine=_machine_info(),
        model_bundle=model_bundle,
        split=split,
        seed=seed,
        clips=clips,
        pass_rate=passed / len(clips) if clips else 0.0,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"report_{model_bundle}_{split}.json"
    baseline_path = BASELINE_DIR / f"{model_bundle}_{split}.json"
    report.regression = compare_to_baseline(report, baseline_path)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if save_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TrackLab AI evaluator")
    parser.add_argument("--split", default="ci", choices=["ci", "dev", "holdout", "all"])
    parser.add_argument("--model", default="baseline", choices=["baseline", "oracle", "sam2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args(argv)

    report = run_eval(
        split=args.split,
        model_bundle=args.model,
        seed=args.seed,
        include_holdout=args.include_holdout,
        save_baseline=args.save_baseline,
    )
    print(
        f"evaluated {len(report.clips)} clips  "
        f"pass_rate={report.pass_rate:.1%}  "
        f"regression={report.regression.get('status')} "
        f"blocking={report.regression.get('blocking')}"
    )
    for clip in report.clips:
        flag = "PASS" if clip.passed else "FAIL"
        err = f"  error={clip.error}" if clip.error else ""
        print(f"  [{flag}] {clip.clip_id} ({clip.task}/{clip.difficulty}){err}")
    if report.regression.get("blocking"):
        return 2
    if report.pass_rate < 1.0 and args.model == "oracle":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
