"""Model adapters. Real weights plug in behind these interfaces later."""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
import numpy as np

from ai.contracts import (
    CalibrationResult,
    CancelToken,
    FailureReason,
    PhysicsResult,
    PoseFrame,
    PoseKeypoint,
    PoseResult,
    ProgressCb,
    ProgressEvent,
    TrackPoint,
    TrackResult,
)
from ai.schema import POSE_KEYPOINTS, ClipAnnotation
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo, index_video


class Tracker(ABC):
    name: str = "tracker"
    version: str = "0"

    @abstractmethod
    def track(
        self,
        info: VideoInfo,
        seed_xy: tuple[float, float],
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
    ) -> TrackResult:
        raise NotImplementedError


class PoseEstimator(ABC):
    name: str = "pose"
    version: str = "0"

    @abstractmethod
    def estimate(
        self,
        info: VideoInfo,
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
    ) -> PoseResult:
        raise NotImplementedError


class Calibrator(ABC):
    name: str = "calibrator"
    version: str = "0"

    @abstractmethod
    def calibrate(
        self,
        info: VideoInfo,
        *,
        expected_length_m: float | None = None,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
    ) -> CalibrationResult:
        raise NotImplementedError


def _emit(
    progress: ProgressCb | None, clip_id: str, current: int, total: int, stage: str
) -> None:
    if progress is not None:
        progress(ProgressEvent(clip_id=clip_id, current=current, total=total, stage=stage))


class OracleTracker(Tracker):
    """Cheats with ground-truth — used to validate the evaluator itself."""

    name = "oracle_tracker"
    version = "1.0.0"

    def __init__(self, annotation: ClipAnnotation) -> None:
        self._ann = annotation

    def track(
        self,
        info: VideoInfo,
        seed_xy: tuple[float, float],
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
    ) -> TrackResult:
        t0 = time.perf_counter()
        points = [
            TrackPoint(
                frame=f.frame,
                x=f.center.x,
                y=f.center.y,
                visible=f.visible,
                confidence=1.0 if f.visible else 0.0,
            )
            for f in self._ann.track
        ]
        _emit(progress, self._ann.clip_id, len(points), len(points), "done")
        return TrackResult(
            clip_id=self._ann.clip_id,
            points=points,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - t0,
        )


class ColorBlobTracker(Tracker):
    """Simple baseline: follow the brightest saturated blob near the previous point."""

    name = "color_blob_tracker"
    version = "0.1.0"

    def track(
        self,
        info: VideoInfo,
        seed_xy: tuple[float, float],
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        prompts=None,  # noqa: ANN001
        **_unused,
    ) -> TrackResult:
        t0 = time.perf_counter()
        decoder = FrameDecoder(info)
        points: list[TrackPoint] = []
        cx, cy = seed_xy
        start = max(0, int(start_frame))
        last = info.frame_count - 1 if end_frame is None else min(int(end_frame), info.frame_count - 1)
        try:
            for i in range(start, last + 1):
                if cancel and cancel.cancelled:
                    return TrackResult(
                        clip_id=info.path.stem,
                        points=points,
                        failure_reason=FailureReason.CANCELLED,
                        model_name=self.name,
                        model_version=self.version,
                        elapsed_s=time.perf_counter() - t0,
                    )
                frame = decoder.frame(i)
                nx, ny, conf, visible = _find_blob(frame, cx, cy)
                if visible:
                    cx, cy = nx, ny
                points.append(
                    TrackPoint(frame=i, x=cx, y=cy, visible=visible, confidence=conf)
                )
                if i < start + 3 or i % 5 == 0:
                    _emit(progress, info.path.stem, i - start + 1, last - start + 1, "track")
                    if cancel and cancel.cancelled:
                        return TrackResult(
                            clip_id=info.path.stem,
                            points=points,
                            failure_reason=FailureReason.CANCELLED,
                            model_name=self.name,
                            model_version=self.version,
                            elapsed_s=time.perf_counter() - t0,
                        )
        finally:
            decoder.close()
        return TrackResult(
            clip_id=info.path.stem,
            points=points,
            confidence=float(np.mean([p.confidence for p in points])) if points else 0.0,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - t0,
        )


def _find_blob(
    frame: np.ndarray, cx: float, cy: float, search: int = 48
) -> tuple[float, float, float, bool]:
    h, w, _ = frame.shape
    x0 = max(0, int(cx - search))
    x1 = min(w, int(cx + search))
    y0 = max(0, int(cy - search))
    y1 = min(h, int(cy + search))
    if x1 <= x0 or y1 <= y0:
        return cx, cy, 0.0, False
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    chroma = patch.max(axis=2) - patch.min(axis=2)
    score = patch.max(axis=2) + chroma * 0.5
    peak = float(score.max())
    if peak < 90:
        return cx, cy, peak / 255.0, False
    mask = score >= max(peak * 0.65, peak - 50)
    blobs = _connected_centroids(mask, score)
    if not blobs:
        return cx, cy, peak / 255.0, False
    # Prefer the compact blob closest to the previous estimate (avoids rods / distractors).
    best = min(
        blobs,
        key=lambda b: (b[0] - (cx - x0)) ** 2
        + (b[1] - (cy - y0)) ** 2
        + 0.05 * b[3],
    )
    lx, ly, mass, _spread, blob_score = best
    # Restrict the centroid to pixels near the previous point so a rod does not pull it.
    local = _centroid_near(mask, score, lx, ly, radius=12)
    if local is not None:
        lx, ly = local
    nx = float(x0 + lx)
    ny = float(y0 + ly)
    return nx, ny, min(blob_score / 255.0, 1.0), mass >= 6


def _connected_centroids(
    mask: np.ndarray, score: np.ndarray
) -> list[tuple[float, float, int, float, float]]:
    """Return (cx, cy, mass, spread, peak) for each 4-connected component."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[tuple[float, float, int, float, float]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            mass = len(cells)
            if mass < 4:
                continue
            ys = np.array([c[0] for c in cells], dtype=np.float64)
            xs = np.array([c[1] for c in cells], dtype=np.float64)
            weights = np.array([score[c[0], c[1]] for c in cells], dtype=np.float64)
            cx = float(np.average(xs, weights=weights))
            cy = float(np.average(ys, weights=weights))
            spread = float(np.hypot(xs.std(), ys.std()))
            out.append((cx, cy, mass, spread, float(weights.max())))
    return out


def _centroid_near(
    mask: np.ndarray, score: np.ndarray, cx: float, cy: float, radius: int
) -> tuple[float, float] | None:
    h, w = mask.shape
    y0 = max(0, int(cy - radius))
    y1 = min(h, int(cy + radius) + 1)
    x0 = max(0, int(cx - radius))
    x1 = min(w, int(cx + radius) + 1)
    local = mask[y0:y1, x0:x1]
    if not local.any():
        return None
    ys, xs = np.nonzero(local)
    weights = score[y0:y1, x0:x1][ys, xs]
    return (
        float(x0 + np.average(xs, weights=weights)),
        float(y0 + np.average(ys, weights=weights)),
    )


class OraclePose(PoseEstimator):
    name = "oracle_pose"
    version = "1.0.0"

    def __init__(self, annotation: ClipAnnotation) -> None:
        self._ann = annotation

    def estimate(
        self,
        info: VideoInfo,
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
    ) -> PoseResult:
        t0 = time.perf_counter()
        frames = [
            PoseFrame(
                frame=f.frame,
                keypoints=[
                    PoseKeypoint(name=k.name, x=k.x, y=k.y, visible=k.visible)
                    for k in f.keypoints
                ],
            )
            for f in self._ann.pose
        ]
        return PoseResult(
            clip_id=self._ann.clip_id,
            frames=frames,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - t0,
        )


class TemplatePose(PoseEstimator):
    """Baseline: assign bright local maxima to a stick-figure template."""

    name = "template_pose"
    version = "0.2.0"

    OFFSETS = {
        "nose": (0, 0),
        "left_eye": (-4, -2),
        "right_eye": (4, -2),
        "left_ear": (-8, 0),
        "right_ear": (8, 0),
        "left_shoulder": (-18, 20),
        "right_shoulder": (18, 20),
        "left_elbow": (-28, 40),
        "right_elbow": (28, 40),
        "left_wrist": (-34, 58),
        "right_wrist": (34, 58),
        "left_hip": (-12, 60),
        "right_hip": (12, 60),
        "left_knee": (-14, 90),
        "right_knee": (14, 90),
        "left_ankle": (-16, 120),
        "right_ankle": (16, 120),
    }

    def estimate(
        self,
        info: VideoInfo,
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
    ) -> PoseResult:
        t0 = time.perf_counter()
        decoder = FrameDecoder(info)
        out: list[PoseFrame] = []
        try:
            for i in range(info.frame_count):
                if cancel and cancel.cancelled:
                    return PoseResult(
                        clip_id=info.path.stem,
                        frames=out,
                        failure_reason=FailureReason.CANCELLED,
                        model_name=self.name,
                        model_version=self.version,
                        elapsed_s=time.perf_counter() - t0,
                    )
                frame = decoder.frame(i)
                peaks = _bright_peaks(frame.mean(axis=2))
                pose_map = _fit_pose_template(peaks, self.OFFSETS)
                if pose_map is None:
                    out.append(PoseFrame(frame=i, keypoints=[], confidence=0.0))
                    continue
                kps = [
                    PoseKeypoint(name=name, x=xy[0], y=xy[1], visible=True, confidence=conf)
                    for name, (xy, conf) in pose_map.items()
                ]
                out.append(PoseFrame(frame=i, keypoints=kps, confidence=0.8))
                if i % 5 == 0:
                    _emit(progress, info.path.stem, i + 1, info.frame_count, "pose")
        finally:
            decoder.close()
        return PoseResult(
            clip_id=info.path.stem,
            frames=out,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - t0,
        )


def _bright_peaks(gray: np.ndarray, threshold: float = 160.0) -> list[tuple[float, float]]:
    center = gray[1:-1, 1:-1]
    mask = (
        (center >= threshold)
        & (center >= gray[:-2, 1:-1])
        & (center >= gray[2:, 1:-1])
        & (center >= gray[1:-1, :-2])
        & (center >= gray[1:-1, 2:])
    )
    ys, xs = np.nonzero(mask)
    peaks = [(float(x + 1), float(y + 1)) for y, x in zip(ys.tolist(), xs.tolist())]
    merged: list[tuple[float, float]] = []
    used = [False] * len(peaks)
    for i, (x, y) in enumerate(peaks):
        if used[i]:
            continue
        cluster = [(x, y)]
        used[i] = True
        for j in range(i + 1, len(peaks)):
            if used[j]:
                continue
            if abs(peaks[j][0] - x) <= 3 and abs(peaks[j][1] - y) <= 3:
                used[j] = True
                cluster.append(peaks[j])
        merged.append(
            (sum(p[0] for p in cluster) / len(cluster), sum(p[1] for p in cluster) / len(cluster))
        )
    return merged


def _fit_pose_template(
    peaks: list[tuple[float, float]],
    offsets: dict[str, tuple[float, float]],
) -> dict[str, tuple[tuple[float, float], float]] | None:
    """Try each peak as the nose origin and keep the assignment with most inliers."""
    if len(peaks) < 4:
        return None
    best_hits = -1
    best: dict[str, tuple[tuple[float, float], float]] | None = None
    names = list(offsets)
    for nx, ny in peaks:
        unused = list(peaks)
        assigned: dict[str, tuple[tuple[float, float], float]] = {}
        hits = 0
        for name in names:
            dx, dy = offsets[name]
            tx, ty = nx + dx, ny + dy
            if not unused:
                assigned[name] = ((tx, ty), 0.4)
                continue
            j = min(
                range(len(unused)),
                key=lambda k: (unused[k][0] - tx) ** 2 + (unused[k][1] - ty) ** 2,
            )
            px, py = unused[j]
            dist = math.hypot(px - tx, py - ty)
            if dist <= 6:
                unused.pop(j)
                assigned[name] = ((px, py), 0.95)
                hits += 1
            else:
                assigned[name] = ((tx, ty), 0.55)
        if hits > best_hits:
            best_hits = hits
            best = assigned
    if best is None or best_hits < 6:
        return None
    return best


class OracleCalibrator(Calibrator):
    name = "oracle_calibrator"
    version = "1.0.0"

    def __init__(self, annotation: ClipAnnotation) -> None:
        self._ann = annotation

    def calibrate(
        self,
        info: VideoInfo,
        *,
        expected_length_m: float | None = None,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
    ) -> CalibrationResult:
        t0 = time.perf_counter()
        gt = self._ann.calibration
        assert gt is not None
        if not gt.has_reliable_ruler:
            return CalibrationResult(
                clip_id=self._ann.clip_id,
                pixels_per_meter=None,
                origin_x=None,
                origin_y=None,
                axis_angle_deg=None,
                confidence=0.0,
                failure_reason=FailureReason.NO_RULER,
                rejected=True,
                model_name=self.name,
                model_version=self.version,
                elapsed_s=time.perf_counter() - t0,
            )
        dist = math.hypot(gt.ruler_b.x - gt.ruler_a.x, gt.ruler_b.y - gt.ruler_a.y)
        return CalibrationResult(
            clip_id=self._ann.clip_id,
            pixels_per_meter=dist / gt.length_m,
            origin_x=gt.origin.x,
            origin_y=gt.origin.y,
            axis_angle_deg=gt.axis_angle_deg,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - t0,
        )


class EdgeRulerCalibrator(Calibrator):
    """Baseline: find the longest nearly-horizontal bright edge as a ruler."""

    name = "edge_ruler_calibrator"
    version = "0.1.0"

    def calibrate(
        self,
        info: VideoInfo,
        *,
        expected_length_m: float | None = None,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
    ) -> CalibrationResult:
        t0 = time.perf_counter()
        decoder = FrameDecoder(info)
        try:
            frame = decoder.frame(0)
        finally:
            decoder.close()
        _emit(progress, info.path.stem, 1, 1, "calib")
        red = frame[:, :, 0].astype(np.float32) - frame[:, :, 1] - frame[:, :, 2]
        green = frame[:, :, 1].astype(np.float32) - frame[:, :, 0] - frame[:, :, 2]
        a = _color_peak(red, 40.0)
        b = _color_peak(green, 40.0)
        length_m = expected_length_m if expected_length_m and expected_length_m > 0 else 0.8

        if a is not None and b is not None:
            span = math.hypot(b[0] - a[0], b[1] - a[1])
            if span >= 40:
                ppm = span / length_m
                angle = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))
                return CalibrationResult(
                    clip_id=info.path.stem,
                    pixels_per_meter=ppm,
                    origin_x=float(a[0]),
                    origin_y=40.0,
                    axis_angle_deg=angle,
                    confidence=0.85,
                    model_name=self.name,
                    model_version=self.version,
                    elapsed_s=time.perf_counter() - t0,
                )

        gray = frame.mean(axis=2)
        # Fallback: longest nearly-horizontal bright run.
        best = None
        for y in range(0, info.height, 2):
            row = gray[y]
            bright = row > 200
            run_start = None
            for x, on in enumerate(bright):
                if on and run_start is None:
                    run_start = x
                elif not on and run_start is not None:
                    length = x - run_start
                    if best is None or length > best[0]:
                        best = (length, run_start, x - 1, y)
                    run_start = None
            if run_start is not None:
                length = info.width - run_start
                if best is None or length > best[0]:
                    best = (length, run_start, info.width - 1, y)

        if best is None or best[0] < 40:
            return CalibrationResult(
                clip_id=info.path.stem,
                pixels_per_meter=None,
                origin_x=None,
                origin_y=None,
                axis_angle_deg=None,
                confidence=0.0,
                failure_reason=FailureReason.NO_RULER,
                rejected=True,
                model_name=self.name,
                model_version=self.version,
                elapsed_s=time.perf_counter() - t0,
            )
        _, x0, x1, y = best
        span = max(x1 - x0, 1)
        ppm = span / length_m
        return CalibrationResult(
            clip_id=info.path.stem,
            pixels_per_meter=ppm,
            origin_x=float(x0),
            origin_y=40.0,
            axis_angle_deg=0.0,
            confidence=0.55,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - t0,
        )


def _color_peak(score: np.ndarray, min_score: float) -> tuple[float, float] | None:
    peak = float(score.max())
    if peak < min_score:
        return None
    mask = score >= max(peak * 0.6, min_score)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    weights = score[ys, xs]
    return float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))


def derive_physics_from_track(
    clip_id: str,
    track: TrackResult,
    fps: float,
    *,
    pixels_per_meter: float | None = None,
    length_m: float | None = None,
    period_hint: bool = False,
) -> PhysicsResult:
    """Lightweight physics estimates used for end-to-end gates on synthetic data."""
    pts = [(p.frame, p.x, p.y) for p in track.points if p.visible]
    if len(pts) < 5:
        return PhysicsResult(
            clip_id=clip_id,
            failure_reason=FailureReason.LOW_CONFIDENCE,
            confidence=0.0,
        )
    ys = np.array([p[2] for p in pts], dtype=np.float64)
    xs = np.array([p[1] for p in pts], dtype=np.float64)
    ts = np.array([p[0] / fps for p in pts], dtype=np.float64)

    period = None
    if period_hint or (ys.max() - ys.min() > 5):
        # Zero-crossing of x around mean for pendulum-like motion.
        xc = xs - xs.mean()
        crossings = [
            i
            for i in range(1, len(xc))
            if xc[i - 1] <= 0 < xc[i] or xc[i - 1] >= 0 > xc[i]
        ]
        if len(crossings) >= 3:
            half_periods = [
                ts[crossings[i + 1]] - ts[crossings[i]]
                for i in range(len(crossings) - 1)
            ]
            period = float(2 * np.median(half_periods))

    gravity = None
    fit_err = None
    if period and length_m and length_m > 0:
        # Small-angle pendulum: T = 2π √(L/g)  →  g = 4π² L / T²
        gravity = float(4 * math.pi**2 * length_m / (period**2))
    elif len(ts) >= 6:
        # Fit y = a + b t + c t^2  (screen y grows downward) for projectiles.
        A = np.column_stack([np.ones_like(ts), ts, ts**2])
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        residual = ys - A @ coef
        fit_err = float(np.sqrt(np.mean(residual**2)) / max(abs(ys).max(), 1.0))
        if pixels_per_meter and pixels_per_meter > 0:
            gravity = float(abs(2 * coef[2] / pixels_per_meter))

    velocity = None
    if len(ts) >= 2 and pixels_per_meter and pixels_per_meter > 0:
        dx = (xs[-1] - xs[0]) / pixels_per_meter
        dt = ts[-1] - ts[0]
        if dt > 0:
            velocity = float(dx / dt)

    return PhysicsResult(
        clip_id=clip_id,
        period_s=period,
        gravity_ms2=gravity,
        velocity_ms=velocity,
        trajectory_fit_error=fit_err,
        confidence=0.8,
    )


def load_video(path) -> VideoInfo:
    return index_video(path)
