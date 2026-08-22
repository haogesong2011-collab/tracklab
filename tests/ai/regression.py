"""Regression runner for the full 60-slot manifest (dev + optional holdout)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.ai.evaluate import run_eval  # noqa: E402
from tests.ai.generate_fixtures import build_fixtures  # noqa: E402


def ensure_data() -> None:
    manifest = ROOT / "datasets" / "manifest.json"
    if not manifest.exists():
        build_fixtures()


def failure_modes(report) -> dict[str, int]:  # noqa: ANN001
    modes: Counter[str] = Counter()
    for clip in report.clips:
        if clip.passed:
            continue
        if clip.error:
            modes["crash"] += 1
            continue
        failed_metrics = [
            m["name"] for m in clip.metrics if m.get("passed") is False
        ]
        if not failed_metrics:
            modes["unknown_fail"] += 1
        for name in failed_metrics:
            modes[name] += 1
    return dict(modes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=["baseline", "oracle", "sam2"])
    parser.add_argument("--split", default="dev", choices=["ci", "dev", "holdout", "all"])
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--rebuild-fixtures", action="store_true")
    args = parser.parse_args(argv)

    if args.rebuild_fixtures or not (ROOT / "datasets" / "manifest.json").exists():
        build_fixtures()
    else:
        ensure_data()

    report = run_eval(
        split=args.split,
        model_bundle=args.model,
        include_holdout=args.include_holdout,
        save_baseline=args.save_baseline,
    )
    modes = failure_modes(report)
    summary = {
        "pass_rate": report.pass_rate,
        "n": len(report.clips),
        "failure_modes": modes,
        "regression": report.regression,
    }
    out = ROOT / "tests" / "ai" / "results" / f"regression_{args.model}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if report.regression.get("blocking"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
