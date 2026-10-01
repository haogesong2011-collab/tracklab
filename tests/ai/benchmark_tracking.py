"""Real-model tracking benchmark (R0).

One command runs the desktop's own model selection, preprocessing and result
generation on a fixed set of clips and writes a reproducible report:

    .venv/bin/python -m tests.ai.benchmark_tracking \\
        --target surface_point --variant candidate --device cpu \\
        --output-dir bench/tapir-candidate

Variants
    candidate  the code in this working tree (full pipeline)
    raw        raw model output: SAM mask centre with the guard off, or one
               full-frame 256 BootsTAPIR pass. Diagnostic only.
    current    the pre-change working tree. Point --baseline-root at an
               extracted copy of it; its `ai/` and `engine/` are imported
               instead of this tree's. This file does not import `tests.*`
               so the harness itself stays the same for every variant.

Clips
    Built-in frozen synthetic cases (the small-target regression from the
    plan: seed 77, 12 frames, 960x540, 8 px texture, 16 px stripes, 5x5
    Gaussian sigma=0.9, shift (7, 2) per frame; plus size, occlusion, turn
    and cross-window cases). Holdout seeds are fixed here in advance.
    --manifest adds real clips (see `load_manifest_entries`). Clips are
    grouped by SHA-256; the same video in two splits is an error.

Reports separate candidate-coordinate accuracy (every point with a position)
from accepted-measurement accuracy (`usable_for_measurement()` only). A
high-score error rate with no high-score samples is reported as N/A.

`--fake-model` swaps in dependency-free stand-ins (an oracle SAM that returns
the truth mask, an NCC point matcher for TAPIR). Numbers from it measure the
plumbing, never model accuracy, and the report says so.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
TREE_ROOT = HERE.parents[2]


def _install_root(root: Path) -> None:
    """Make `ai` / `engine` resolve under `root` (before any import of them)."""
    for name in list(sys.modules):
        if name == "ai" or name.startswith("ai.") or name == "engine" or name.startswith("engine."):
            del sys.modules[name]
    text = str(root)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


# ---------------------------------------------------------------------------
# Frozen synthetic cases

DEV_SEEDS = (77,)
HOLDOUT_SEEDS = (1077, 2077, 3077)  # frozen before any result was seen


@dataclass
class Case:
    clip_id: str
    width: int
    height: int
    fps: float
    frames: list[np.ndarray]
    centers: list[tuple[float, float] | None]  # object centre truth (None = invisible)
    surface: list[tuple[float, float] | None]  # surface-point truth
    masks: list[np.ndarray | None]  # truth masks (for the fake SAM)
    bbox_diag: float
    prompt_frame: int
    prompt_xy: tuple[float, float]
    surface_query_xy: tuple[float, float]
    split: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)


def _gauss_kernel(size: int = 5, sigma: float = 0.9) -> np.ndarray:
    r = size // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def _blur(img: np.ndarray, size: int = 5, sigma: float = 0.9) -> np.ndarray:
    k = _gauss_kernel(size, sigma)
    r = size // 2
    out = img.astype(np.float64)
    p = np.pad(out, ((r, r), (0, 0), (0, 0)), mode="edge")
    out = sum(k[i] * p[i : i + img.shape[0]] for i in range(size))
    p = np.pad(out, ((0, 0), (r, r), (0, 0)), mode="edge")
    out = sum(k[i] * p[:, i : i + img.shape[1]] for i in range(size))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _stripes(width: int, height: int, period: int = 16) -> np.ndarray:
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = (70, 78, 66)
    on = (np.arange(width) % period) < period // 2
    img[:, on] = (182, 190, 170)
    return img


def small_texture_case(
    seed: int = 77,
    *,
    n: int = 12,
    width: int = 960,
    height: int = 540,
    size: int = 8,
    shift: tuple[float, float] = (7.0, 2.0),
    start: tuple[int, int] = (400, 250),
    occlude: tuple[int, int] | None = None,
    turn_at: int | None = None,
    split: str = "dev",
    family: str = "small_texture",
) -> Case:
    """Textured square over static stripes, blurred, integer-free motion allowed."""
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    background = _stripes(width, height)
    frames: list[np.ndarray] = []
    centers: list[tuple[float, float] | None] = []
    surface: list[tuple[float, float] | None] = []
    masks: list[np.ndarray | None] = []
    x0, y0 = float(start[0]), float(start[1])
    vx, vy = shift
    # Query a pixel off the texture centre so a centroid tracker cannot pass.
    offset = (max(1, size // 4), -max(1, size // 4)) if size >= 4 else (0, 0)
    for i in range(n):
        if turn_at is not None and i >= turn_at:
            px = x0 + vx * turn_at - vx * (i - turn_at)
            py = y0 + vy * turn_at + 3.0 * vy * (i - turn_at)
        else:
            px, py = x0 + vx * i, y0 + vy * i
        ix, iy = int(round(px)), int(round(py))
        img = background.copy()
        hidden = occlude is not None and occlude[0] <= i < occlude[1]
        mask = np.zeros((height, width), dtype=bool)
        if not hidden:
            img[iy : iy + size, ix : ix + size] = texture
            mask[iy : iy + size, ix : ix + size] = True
        else:
            # A solid bar covers the object.
            img[iy - 6 : iy + size + 6, ix - 6 : ix + size + 6] = (30, 30, 34)
        frames.append(_blur(img))
        c = (ix + (size - 1) / 2.0, iy + (size - 1) / 2.0)
        centers.append(None if hidden else c)
        surface.append(None if hidden else (c[0] + offset[0], c[1] + offset[1]))
        masks.append(None if hidden else mask)
    first = next(i for i, c in enumerate(centers) if c is not None)
    return Case(
        clip_id=f"{family}_s{size}_seed{seed}",
        width=width,
        height=height,
        fps=30.0,
        frames=frames,
        centers=centers,
        surface=surface,
        masks=masks,
        bbox_diag=float(math.hypot(size, size)),
        prompt_frame=first,
        prompt_xy=centers[first],  # type: ignore[arg-type]
        surface_query_xy=surface[first],  # type: ignore[arg-type]
        split=split,
        family=family,
        params={
            "seed": seed,
            "frames": n,
            "size": size,
            "shift": list(shift),
            "stripes_period_px": 16,
            "blur": "5x5 gaussian sigma=0.9",
            "occlude": list(occlude) if occlude else None,
            "turn_at": turn_at,
            "resolution": [width, height],
        },
    )


def builtin_cases(split: str = "dev") -> list[Case]:
    seeds = {"dev": DEV_SEEDS, "holdout": HOLDOUT_SEEDS, "all": DEV_SEEDS + HOLDOUT_SEEDS}[split]
    cases: list[Case] = []
    for seed in seeds:
        tag = "dev" if seed in DEV_SEEDS else "holdout"
        cases.append(small_texture_case(seed, split=tag))
        for size in (4, 16, 32):
            cases.append(small_texture_case(seed, size=size, split=tag, family="size"))
        cases.append(
            small_texture_case(seed, n=30, occlude=(12, 18), split=tag, family="occlusion_reappear")
        )
        cases.append(small_texture_case(seed, n=24, turn_at=12, split=tag, family="sharp_turn"))
        cases.append(
            small_texture_case(
                seed, n=48, shift=(5.0, 1.5), start=(240, 200), split=tag, family="cross_window"
            )
        )
        cases.append(
            small_texture_case(
                seed, n=16, width=540, height=960, start=(200, 400), split=tag, family="portrait"
            )
        )
        cases.append(
            small_texture_case(
                seed, n=16, width=1920, height=1080, start=(900, 500), split=tag, family="hd1080"
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Video IO


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    """Lossless when possible (FFV1/MKV); the synthetic truth depends on pixels."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    attempts = [("ffv1", ".mkv", "bgr0", {}), ("libx264rgb", ".mkv", "rgb24", {"crf": "0"}),
                ("libx264", ".mp4", "yuv444p", {"crf": "0", "preset": "ultrafast"})]
    last_error: Exception | None = None
    for codec, suffix, pix_fmt, options in attempts:
        target = path.with_suffix(suffix)
        try:
            container = av.open(str(target), mode="w")
            stream = container.add_stream(codec, rate=int(round(fps)))
            stream.width = frames[0].shape[1]
            stream.height = frames[0].shape[0]
            stream.pix_fmt = pix_fmt
            stream.options = options
            for array in frames:
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            return target
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(f"cannot write {path}: {last_error}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    sidecar = Path(str(path) + ".npz")  # dev-only PyAV shim stores pixels here
    if sidecar.exists():
        with open(sidecar, "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Manifest (real clips)


@dataclass
class ClipSpec:
    clip_id: str
    video: Path
    split: str
    centers: dict[int, tuple[float, float] | None]
    surface: dict[int, tuple[float, float] | None]
    bbox_diag: float | None
    prompt_frame: int
    prompt_xy: tuple[float, float]
    surface_query_xy: tuple[float, float]
    source: str
    case: Case | None = None


def load_manifest_entries(path: Path) -> list[ClipSpec]:
    """Accept this benchmark's format or datasets/manifest.json.

    Benchmark format:
      {"entries": [{"clip_id": "...", "video": "rel/or/abs.mp4", "split": "dev",
                    "prompt": {"frame": 0, "x": 1.0, "y": 2.0},
                    "surface_query": {"x": .., "y": ..},        # optional
                    "track": [{"frame": 0, "x": .., "y": .., "visible": true,
                               "surface_x": .., "surface_y": ..}], "bbox_diag": 12}]}
    Surface truth is used only where given; a centre is never reused as a
    surface point unless the clip is a declared rigid translation.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    out: list[ClipSpec] = []
    for entry in data.get("entries", []):
        if "relative_path" in entry:  # datasets/manifest.json
            if entry.get("task") != "track" or entry.get("hidden"):
                continue
            video = (TREE_ROOT / entry["relative_path"]).resolve()
            ann = json.loads((TREE_ROOT / entry["annotation_path"]).read_text(encoding="utf-8"))
            centers: dict[int, tuple[float, float] | None] = {}
            diag = None
            for row in ann.get("track", []):
                c = row.get("center") or {}
                centers[int(row["frame"])] = (float(c["x"]), float(c["y"])) if row.get("visible", True) else None
                bb = row.get("bbox")
                if bb and diag is None:
                    diag = math.hypot(float(bb["w"]), float(bb["h"]))
            first = min(f for f, c in centers.items() if c is not None)
            out.append(
                ClipSpec(
                    entry["clip_id"], video, entry.get("split", "dev"), centers, {}, diag,
                    first, centers[first], centers[first], "datasets/manifest.json",  # type: ignore[arg-type]
                )
            )
            continue
        video = Path(entry["video"])
        if not video.is_absolute():
            video = (base / video).resolve()
        centers = {}
        surface: dict[int, tuple[float, float] | None] = {}
        for row in entry.get("track", []):
            f = int(row["frame"])
            vis = bool(row.get("visible", True))
            centers[f] = (float(row["x"]), float(row["y"])) if vis and "x" in row else None
            if "surface_x" in row:
                surface[f] = (float(row["surface_x"]), float(row["surface_y"])) if vis else None
        prompt = entry["prompt"]
        sq = entry.get("surface_query") or prompt
        out.append(
            ClipSpec(
                entry["clip_id"], video, entry.get("split", "dev"), centers, surface,
                entry.get("bbox_diag"), int(prompt["frame"]), (float(prompt["x"]), float(prompt["y"])),
                (float(sq["x"]), float(sq["y"])), str(path),
            )
        )
    return out


def group_by_hash(specs: list[ClipSpec]) -> tuple[list[ClipSpec], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate by SHA-256; report duplicates and cross-split leaks."""
    by_hash: dict[str, list[ClipSpec]] = {}
    for spec in specs:
        by_hash.setdefault(sha256_file(spec.video), []).append(spec)
    unique: list[ClipSpec] = []
    duplicates: list[dict[str, Any]] = []
    leaks: list[dict[str, Any]] = []
    for digest, group in by_hash.items():
        unique.append(group[0])
        if len(group) > 1:
            duplicates.append({"sha256": digest, "clips": [g.clip_id for g in group]})
        splits = sorted({g.split for g in group})
        if len(splits) > 1:
            leaks.append({"sha256": digest, "splits": splits, "clips": [g.clip_id for g in group]})
    return unique, duplicates, leaks


# ---------------------------------------------------------------------------
# Metrics (self-contained; same tolerance as tests/ai/metrics.py)


def tolerance(diag: float | None) -> float:
    return max(3.0, 0.1 * diag) if diag else 3.0


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def clip_metrics(points: list[Any], truth: dict[int, tuple[float, float] | None], diag: float | None) -> dict[str, Any]:
    tol = tolerance(diag)
    by = {p.frame: p for p in points}
    visible = [f for f, t in sorted(truth.items()) if t is not None]
    hidden = [f for f, t in sorted(truth.items()) if t is None]
    acc_err: list[float] = []
    cand_err: list[float] = []
    hits = {3: 0, 5: 0, 10: 0}
    cand_hits = {3: 0, 5: 0, 10: 0}
    accepted_wrong = 0
    high = high_wrong = 0
    runs: list[int] = []
    run = 0
    per_frame: list[dict[str, Any]] = []
    for f in sorted(truth):
        t = truth[f]
        p = by.get(f)
        usable = bool(p is not None and p.usable_for_measurement())
        row: dict[str, Any] = {
            "frame": f,
            "truth": None if t is None else [round(t[0], 3), round(t[1], 3)],
            "status": None if p is None else p.status.value,
            "visible": None if p is None else p.visible,
            "usable": usable,
            "confidence": None if p is None else round(float(p.confidence), 4),
            "x": None if p is None else round(float(p.x), 3),
            "y": None if p is None else round(float(p.y), 3),
            "reason": None if p is None else (p.diagnostics or {}).get("reason_code", p.note),
        }
        if t is not None:
            has_coords = p is not None and (
                p.status.value != "lost" or (p.diagnostics or {}).get("raw_xy") is not None
            )
            if has_coords:
                ce = math.hypot(p.x - t[0], p.y - t[1])
                cand_err.append(ce)
                for k in cand_hits:
                    cand_hits[k] += int(ce <= k)
                row["candidate_error_px"] = round(ce, 3)
            if usable:
                e = math.hypot(p.x - t[0], p.y - t[1])
                acc_err.append(e)
                for k in hits:
                    hits[k] += int(e <= k)
                row["accepted_error_px"] = round(e, 3)
                wrong = e > tol
                accepted_wrong += int(wrong)
                if p.confidence >= 0.9 and p.source.value == "auto":
                    high += 1
                    high_wrong += int(wrong)
                # Runs of accepted-but-wrong frames; frames that are not
                # accepted neither extend nor break a run.
                if wrong:
                    run += 1
                elif run:
                    runs.append(run)
                    run = 0
        per_frame.append(row)
    if run:
        runs.append(run)
    n_vis = len(visible)
    accepted = len(acc_err)
    false_accept = sum(1 for f in hidden if by.get(f) is not None and by[f].usable_for_measurement())
    # Recovery after truth occlusion ends.
    recoveries: list[int] = []
    failures = 0
    frames_sorted = sorted(truth)
    for i in range(1, len(frames_sorted)):
        f_prev, f = frames_sorted[i - 1], frames_sorted[i]
        if truth[f_prev] is None and truth[f] is not None:
            got = None
            for j, g in enumerate(frames_sorted[i:]):
                t = truth[g]
                p = by.get(g)
                if t is not None and p is not None and p.usable_for_measurement() and math.hypot(p.x - t[0], p.y - t[1]) <= tol:
                    got = j
                    break
            if got is None:
                failures += 1
            else:
                recoveries.append(got)
    return {
        "tolerance_px": tol,
        "visible_frames": n_vis,
        "hidden_frames": len(hidden),
        "localization_success": {f"{k}px": (hits[k] / n_vis if n_vis else None) for k in hits},
        "candidate_success": {f"{k}px": (cand_hits[k] / n_vis if n_vis else None) for k in cand_hits},
        "accepted_error_px": {
            "median": _pct(acc_err, 50),
            "mean": float(np.mean(acc_err)) if acc_err else None,
            "p95": _pct(acc_err, 95),
        },
        "candidate_error_px": {
            "median": _pct(cand_err, 50),
            "mean": float(np.mean(cand_err)) if cand_err else None,
            "p95": _pct(cand_err, 95),
        },
        "accepted_coverage": accepted / n_vis if n_vis else None,
        "accepted_wrong_rate": accepted_wrong / accepted if accepted else None,
        "occluded_false_accept_rate": false_accept / len(hidden) if hidden else None,
        "high_score": {
            "threshold": 0.9,
            "samples": high,
            "wrong": high_wrong,
            "error_rate": (high_wrong / high) if high else "N/A",
            "coverage": high / n_vis if n_vis else None,
        },
        "identity_switches": len([r for r in runs if r >= 3]),
        "longest_wrong_run": max(runs) if runs else 0,
        "recovery_frames": recoveries,
        "recovery_failures": failures,
        "per_frame": per_frame,
    }


def aggregate(clips: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_of(key: str, sub: str | None = None) -> float | None:
        vals = []
        for c in clips:
            v = c["metrics"].get(key)
            if sub is not None and isinstance(v, dict):
                v = v.get(sub)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    high = sum(c["metrics"]["high_score"]["samples"] for c in clips)
    high_wrong = sum(c["metrics"]["high_score"]["wrong"] for c in clips)
    vis = sum(c["metrics"]["visible_frames"] for c in clips)
    return {
        "clips": len(clips),
        "localization_success_3px": mean_of("localization_success", "3px"),
        "localization_success_5px": mean_of("localization_success", "5px"),
        "localization_success_10px": mean_of("localization_success", "10px"),
        "candidate_success_3px": mean_of("candidate_success", "3px"),
        "accepted_coverage": mean_of("accepted_coverage"),
        "accepted_wrong_rate": mean_of("accepted_wrong_rate"),
        "occluded_false_accept_rate": mean_of("occluded_false_accept_rate"),
        "high_score_samples": high,
        "high_score_wrong": high_wrong,
        "high_score_error_rate": (high_wrong / high) if high else "N/A",
        "high_score_coverage": high / vis if vis else None,
        "recovery_failures": sum(c["metrics"]["recovery_failures"] for c in clips),
        "identity_switches": sum(c["metrics"]["identity_switches"] for c in clips),
        "elapsed_s": float(sum(c["elapsed_s"] for c in clips)),
    }


# ---------------------------------------------------------------------------
# Environment report


def _dist_commit(*names: str) -> dict[str, Any]:
    from importlib import metadata

    for name in names:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        info: dict[str, Any] = {"name": name, "version": dist.version}
        try:
            raw = dist.read_text("direct_url.json")
            if raw:
                data = json.loads(raw)
                info["url"] = data.get("url")
                info["commit"] = (data.get("vcs_info") or {}).get("commit_id")
        except Exception:  # noqa: BLE001
            pass
        return info
    return {"name": names[0], "installed": False}


def environment(device: str, fake: bool) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "requested_device": device,
        "fake_model": fake,
        "code_root": str(sys.path[0]),
        "pinned": {
            "sam2": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
            "tapnet": "730cda1c730877cfedbe01bf87fb1cadb78a565d",
        },
        "installed": {
            "sam2": _dist_commit("SAM-2", "sam-2", "sam2"),
            "tapnet": _dist_commit("tapnet"),
            "torch": _dist_commit("torch"),
            "av": _dist_commit("av"),
        },
    }
    try:
        from ai import __name__ as _ai  # noqa: F401
        import app  # noqa: F401

        env["app_version"] = getattr(sys.modules.get("app"), "__version__", None)
    except Exception:  # noqa: BLE001
        pass
    return env


def _weights(spec: Any) -> dict[str, Any]:
    from ai.model_manager import ensure_checkpoint

    out = {"model_id": spec.model_id, "expected_sha256": spec.sha256}
    try:
        path = ensure_checkpoint(spec, download=False)  # the file actually loaded
        out["path"] = str(path)
        out["actual_sha256"] = sha256_file(Path(path)) if Path(path).exists() else None
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Fakes (plumbing only)


class OracleSamPredictor:
    """Returns the truth mask for each frame. Measures plumbing, not SAM."""

    def __init__(self, masks: list[np.ndarray | None], height: int, width: int) -> None:
        self.masks = masks
        self.h, self.w = height, width

    def init_state(self, video_path: str, **_kw):
        return {}

    def add_new_points_or_box(self, inference_state, frame_idx=0, obj_id=1, **_kw):
        return frame_idx, [obj_id], [self._logits(frame_idx)]

    def propagate_in_video(self, inference_state, start_frame_idx=0, max_frame_num_to_track=None, **_kw):
        end = len(self.masks) if max_frame_num_to_track is None else start_frame_idx + max_frame_num_to_track
        for i in range(start_frame_idx, min(end, len(self.masks))):
            yield i, [1], [self._logits(i)]

    def _logits(self, i: int) -> np.ndarray:
        m = self.masks[i] if 0 <= i < len(self.masks) else None
        if m is None:
            return np.full((1, self.h, self.w), -8.0, dtype=np.float32)
        return np.where(m, 8.0, -8.0).astype(np.float32)[None]


class NccPointRunner:
    """Dependency-free stand-in for TAPIR: normalised cross-correlation.

    Works in the model-input grid, like TAPIR, so down-scaling a small target
    genuinely loses its texture. Occlusion = weak match; uncertainty = a
    second peak almost as strong as the first.
    """

    device = "cpu"
    fallback_reason = ""

    def __init__(self, radius: int = 3, search: int = 40) -> None:
        self.r = radius
        self.search = search

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        f = np.asarray(frame, dtype=np.float32)
        return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]

    def _patch(self, gray: np.ndarray, u: float, v: float) -> np.ndarray:
        from ai.local_flow import bilinear

        off = np.arange(-self.r, self.r + 1, dtype=np.float32)
        oy, ox = np.meshgrid(off, off, indexing="ij")
        vals, _ = bilinear(gray, (u - 0.5 + ox).ravel(), (v - 0.5 + oy).ravel())
        return vals.reshape(ox.shape)

    def query_features(self, frame: np.ndarray, uv: tuple[float, float]):
        return self._patch(self._gray(frame), uv[0], uv[1])

    def track(self, frames, features, *, query_t, query_uv):
        from ai.tapir_tracker import RunOutput

        grays = [self._gray(f) for f in frames]
        tmpl = self._patch(grays[query_t], *query_uv) if query_t is not None else features
        t = (tmpl - tmpl.mean()) / (tmpl.std() + 1e-6)
        n = len(frames)
        tracks = np.zeros((n, 2))
        occ = np.zeros(n)
        dist = np.zeros(n)
        size = grays[0].shape[0]
        guess = np.array(query_uv if query_uv is not None else (size / 2, size / 2), dtype=np.float64)
        order = list(range(n))
        if query_t is not None:
            order = list(range(query_t, n)) + list(range(query_t - 1, -1, -1))
        from numpy.lib.stride_tricks import sliding_window_view

        k = 2 * self.r + 1
        for idx in order:
            g = grays[idx]
            if query_t is not None and idx == query_t - 1:
                guess = np.array(query_uv, dtype=np.float64)
            cx, cy = int(round(guess[0] - 0.5)), int(round(guess[1] - 0.5))
            s = self.search if query_uv is not None or idx != order[0] else size
            x0, x1 = max(self.r, cx - s), min(g.shape[1] - self.r - 1, cx + s)
            y0, y1 = max(self.r, cy - s), min(g.shape[0] - self.r - 1, cy + s)
            if x1 <= x0 or y1 <= y0:
                tracks[idx] = guess
                occ[idx] = 4.0
                continue
            region = g[y0 - self.r : y1 + self.r + 1, x0 - self.r : x1 + self.r + 1]
            win = sliding_window_view(region, (k, k))
            mu = win.mean(axis=(-1, -2), keepdims=True)
            sd = win.std(axis=(-1, -2), keepdims=True) + 1e-6
            score = ((win - mu) / sd * t).mean(axis=(-1, -2))
            j = np.unravel_index(int(np.argmax(score)), score.shape)
            best = float(score[j])
            sy, sx = j
            second = score.copy()
            ex = self.r + 1  # a second peak must be a different place, not a neighbour
            second[max(0, sy - ex) : sy + ex + 1, max(0, sx - ex) : sx + ex + 1] = -np.inf
            runner_up = float(second.max()) if np.isfinite(second).any() else -1.0
            u = x0 + sx + 0.5
            v = y0 + sy + 0.5
            tracks[idx] = (u, v)
            occ[idx] = -4.0 if best > 0.5 else 4.0
            dist[idx] = 2.0 if runner_up > best - 0.05 else -4.0
            guess = np.array((u, v))
        return RunOutput(tracks, occ, dist)


# ---------------------------------------------------------------------------
# Running one clip


def _track_mode(name: str):
    from ai.contracts import TrackMode

    return TrackMode(name)


def run_clip(
    spec: ClipSpec,
    *,
    target: str,
    variant: str,
    device: str,
    mode: str,
    fake: bool,
    projectile: bool = False,
) -> dict[str, Any]:
    from ai.contracts import PromptKind, TrackPrompt
    from ai.models import load_video

    info = load_video(spec.video)
    t0 = time.perf_counter()
    model_info: dict[str, Any] = {}
    prompt_xy = spec.surface_query_xy if target == "surface_point" else spec.prompt_xy
    prompt = TrackPrompt(frame=spec.prompt_frame, kind=PromptKind.POSITIVE, x=prompt_xy[0], y=prompt_xy[1])
    frames = [p for p in range(info.frame_count)]
    if target == "object_center":
        from ai.sam2_tracker import Sam2Tracker

        if fake:
            assert spec.case is not None, "--fake-model needs synthetic cases"
            tracker = Sam2Tracker(predictor=OracleSamPredictor(spec.case.masks, info.height, info.width))
            stride, image_size = 1, None
            model_info = {"model": "oracle-sam (fake)", "input_size": None}
        else:
            from ai.sam_runtime import SamRuntime, settings_for_mode

            spec_m, stride, image_size = settings_for_mode(_track_mode(mode))
            predictor = SamRuntime.instance().predictor_for(spec_m)
            tracker = Sam2Tracker(predictor=predictor, spec=spec_m)
            model_info = {
                "weights": _weights(spec_m),
                "input_size": int(getattr(predictor, "image_size", 0) or 0),
                "actual_device": str(getattr(predictor, "device", device)),
                "stride": stride,
            }
        result = tracker.track(
            info,
            prompt_xy,
            start_frame=0,
            end_frame=info.frame_count - 1,
            prompts=[prompt],
            stride=stride,
            image_size=image_size,
            anti_interference=(variant != "raw"),
            projectile=projectile,
        )
        truth = spec.centers
    else:
        from ai.tapir_tracker import BootsTapirTracker

        kwargs: dict[str, Any] = {"device": device}
        model = NccPointRunner() if fake else None
        if variant == "raw":
            kwargs.update({"local_refine": False, "reverse_check": False, "global_window": 10_000})
        try:
            tracker = BootsTapirTracker(model=model, **kwargs)
        except TypeError:  # the pre-change adapter only accepts model/device
            tracker = BootsTapirTracker(model=model, device=device)
        result = tracker.track(
            info, prompt_xy, start_frame=0, end_frame=info.frame_count - 1, prompts=[prompt]
        )
        if not fake:
            from ai.model_manager import BOOTSTAPIR

            model_info = {"weights": _weights(BOOTSTAPIR)}
        model_info.update(
            {
                "model": "ncc-point (fake)" if fake else "bootstapir",
                "input_size": {"global": 256, "local": 512 if variant == "candidate" else None},
                "run": getattr(tracker, "last_run", {}),
            }
        )
        truth = spec.surface or {}
        if not truth and spec.case is None:
            truth = {}
    elapsed = time.perf_counter() - t0
    metrics = clip_metrics(result.points, truth, spec.bbox_diag) if truth else {"note": "no surface truth"}
    return {
        "clip_id": spec.clip_id,
        "split": spec.split,
        "video": str(spec.video),
        "sha256": sha256_file(spec.video),
        "prompt": {"frame": spec.prompt_frame, "x": prompt_xy[0], "y": prompt_xy[1]},
        "model": model_info,
        "elapsed_s": elapsed,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0),
        "failure": getattr(result.failure_reason, "value", None),
        "metrics": metrics,
        "_points": result.points,
        "_frames": frames,
    }


def write_overlay(path: Path, case: Case, points: list[Any], truth: list[tuple[float, float] | None]) -> None:
    """Truth (green), final measurement (blue = trusted, yellow = review, red = lost)."""
    by = {p.frame: p for p in points}
    out = []
    for i, frame in enumerate(case.frames):
        img = frame.copy()

        def mark(x: float, y: float, colour: tuple[int, int, int], r: int = 6) -> None:
            xi, yi = int(round(x)), int(round(y))
            for d in range(-r, r + 1):
                for yy, xx in ((yi, xi + d), (yi + d, xi)):
                    if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                        img[yy, xx] = colour

        if truth[i] is not None:
            mark(*truth[i], (40, 220, 60), 8)
        p = by.get(i)
        if p is not None:
            colour = {"trusted": (40, 120, 255), "review": (250, 210, 40), "lost": (230, 40, 40)}[p.status.value]
            mark(p.x, p.y, colour, 5)
        out.append(img)
    write_video(path, out, case.fps)


# ---------------------------------------------------------------------------
# Driver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["object_center", "surface_point"], required=True)
    parser.add_argument("--variant", choices=["current", "raw", "candidate"], required=True)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--no-builtin", action="store_true", help="only clips from --manifest")
    parser.add_argument("--family", action="append", default=[], help="limit built-in families")
    parser.add_argument("--mode", choices=["precise", "fast"], default="precise")
    parser.add_argument("--baseline-root", type=Path, help="pre-change tree for --variant current")
    parser.add_argument("--fake-model", action="store_true", help="plumbing check only")
    parser.add_argument("--projectile", action="store_true", help="object_center: projectile mode")
    parser.add_argument("--overlay", action="store_true", help="write overlay videos")
    parser.add_argument("--allow-split-leak", action="store_true")
    args = parser.parse_args(argv)

    if args.variant == "current":
        if args.baseline_root is None:
            parser.error("--variant current needs --baseline-root (an extracted pre-change tree)")
        _install_root(args.baseline_root.resolve())
    else:
        _install_root(TREE_ROOT)

    # Device choice goes through the desktop's own selector.
    import ai.model_manager as mm

    mm.select_device = lambda: args.device  # type: ignore[assignment]

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    specs: list[ClipSpec] = []
    if not args.no_builtin:
        for case in builtin_cases(args.split):
            if args.family and case.family not in args.family:
                continue
            path = write_video(video_dir / case.clip_id, case.frames, case.fps)
            specs.append(
                ClipSpec(
                    case.clip_id,
                    path,
                    case.split,
                    {i: c for i, c in enumerate(case.centers)},
                    {i: c for i, c in enumerate(case.surface)},
                    case.bbox_diag,
                    case.prompt_frame,
                    case.prompt_xy,
                    case.surface_query_xy,
                    "builtin",
                    case,
                )
            )
    for manifest in args.manifest:
        specs.extend(load_manifest_entries(manifest))
    unique, duplicates, leaks = group_by_hash(specs)
    if leaks and not args.allow_split_leak:
        print(json.dumps({"split_leaks": leaks}, ensure_ascii=False, indent=2))
        print("同一视频出现在多个划分里；修正清单或加 --allow-split-leak。", file=sys.stderr)
        return 3

    report: dict[str, Any] = {
        "created_at": time.time(),
        "target": args.target,
        "variant": args.variant,
        "mode": args.mode,
        "projectile": args.projectile,
        "environment": environment(args.device, args.fake_model),
        "duplicates": duplicates,
        "split_leaks": leaks,
        "clips": [],
    }
    if args.fake_model:
        report["warning"] = "fake model: plumbing check only, not model accuracy"
    for spec in unique:
        try:
            row = run_clip(
                spec,
                target=args.target,
                variant=args.variant,
                device=args.device,
                mode=args.mode,
                fake=args.fake_model,
                projectile=args.projectile,
            )
        except Exception as exc:  # noqa: BLE001
            report["clips"].append({"clip_id": spec.clip_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{spec.clip_id}] 失败：{exc}", file=sys.stderr)
            continue
        points = row.pop("_points")
        row.pop("_frames")
        per_frame = row["metrics"].pop("per_frame", [])
        with open(out_dir / f"{spec.clip_id}.frames.csv", "w", newline="", encoding="utf-8") as fh:
            if per_frame:
                writer = csv.DictWriter(fh, fieldnames=sorted({k for r in per_frame for k in r}))
                writer.writeheader()
                writer.writerows(per_frame)
        with open(out_dir / f"{spec.clip_id}.points.json", "w", encoding="utf-8") as fh:
            json.dump([p.to_dict() for p in points], fh, ensure_ascii=False, default=str)
        if args.overlay and spec.case is not None:
            truth = spec.case.surface if args.target == "surface_point" else spec.case.centers
            write_overlay(out_dir / f"{spec.clip_id}.overlay", spec.case, points, truth)
        report["clips"].append(row)
        m = row["metrics"]
        if "localization_success" in m:
            print(
                f"[{spec.clip_id}] 3px {m['localization_success']['3px']:.2f} "
                f"候选3px {m['candidate_success']['3px'] or 0:.2f} "
                f"覆盖 {m['accepted_coverage'] or 0:.2f} "
                f"高分错 {m['high_score']['error_rate']} ({m['high_score']['samples']})"
            )
    ok = [c for c in report["clips"] if "metrics" in c and "localization_success" in c["metrics"]]
    report["summary"] = {
        "all": aggregate(ok),
        **{f"split:{s}": aggregate([c for c in ok if c["split"] == s]) for s in sorted({c["split"] for c in ok})},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, default=str))
    return 0


def compare(paths: list[Path]) -> str:
    """Markdown table for several report.json files (ablation view)."""
    rows = []
    for path in paths:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        s = rep["summary"]["all"]
        rows.append(
            f"| {rep['target']} | {rep['variant']} | {s['clips']} | {s['localization_success_3px']} | "
            f"{s['candidate_success_3px']} | {s['accepted_coverage']} | {s['accepted_wrong_rate']} | "
            f"{s['high_score_error_rate']} ({s['high_score_samples']}) | {s['recovery_failures']} |"
        )
    head = (
        "| 目标 | 变体 | 片段 | 3px 成功率 | 候选 3px | 接受覆盖 | 接受错误率 | 高分错误率 (样本) | 未恢复 |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    return "\n".join([head, *rows])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        print(compare([Path(p) for p in sys.argv[2:]]))
        raise SystemExit(0)
    raise SystemExit(main())
