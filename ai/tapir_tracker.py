"""BootsTAPIR surface-point tracking adapter.

The dependency and checkpoint are optional. Object-centre tracking remains on
SAM 2; this adapter tracks the exact physical point the user clicked.

Pipeline (all coordinates are original-image pixels unless stated):

1. The query identity is fixed once: frame, original xy, optional object ROI.
   Later prompts never replace it silently.
2. Global route: the full frame at 256x256, in bounded windows (32 frames,
   8 overlap). Query features come from the real query frame and are reused
   for every window through TAPIR's feature interface; a window never uses a
   previous window's prediction as a new query.
3. Local route (precise mode): a square ROI resized to 512, in windows of 16
   frames (8 overlap), grown outward from the query frame. The ROI covers the
   last trusted local positions, the motion expected within the window and a
   margin; it never looks at ground truth. Query features for each ROI scale
   are taken from the *original query frame* at the same scale. A window ends
   early when the point nears the crop edge, and the ROI is rebuilt there.
4. Fusion: one measurement per frame, never an average. When global and local
   disagree, independent image matching against the query patch and temporal
   consistency choose; ambiguous cases stay under review.
5. Hard frames (conflict, reappearance) get a reverse check: TAPIR is queried
   at the candidate and tracked back; it must return to the trusted position.
   The reverse query is only used for verification.

TAPIR's two heads are kept separately: occlusion (is the point visible) and
expected distance (is the position accurate). Occluded -> lost; visible but
uncertain -> review.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ai.contracts import (
    CancelToken,
    FailureReason,
    ProgressCb,
    TrackPoint,
    TrackPointSource,
    TrackPointStatus,
    TrackResult,
    TRACK_QUALITY_VERSION,
)
from ai.model_manager import BOOTSTAPIR, ModelNotAvailable, ensure_checkpoint, select_device
from ai.models import Tracker, _emit
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo

MODEL_SIZE = 256
LOCAL_SIZE = 512
GLOBAL_WINDOW = 32
GLOBAL_OVERLAP = 8
LOCAL_WINDOW = 16
LOCAL_OVERLAP = 8
ROI_MIN = 256
ROI_MAX = 1024
ROI_STEP = 64
EDGE_MARGIN = 0.12
REVERSE_CONTEXT = 32
MAX_REVERSE_CHECKS = 12
FRAME_CACHE = 48


# ---------------------------------------------------------------------------
# Geometry


@dataclass(frozen=True)
class QueryRecord:
    """The identity of the tracked point. Never replaced implicitly."""

    query_id: str
    query_frame: int
    query_xy_original: tuple[float, float]
    optional_object_roi: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_frame": self.query_frame,
            "query_xy_original": [self.query_xy_original[0], self.query_xy_original[1]],
            "optional_object_roi": None
            if self.optional_object_roi is None
            else list(self.optional_object_roi),
        }


def make_query(
    frame: int, xy: tuple[float, float], roi: tuple[float, float, float, float] | None = None
) -> QueryRecord:
    digest = hashlib.sha1(f"{frame}:{xy[0]:.4f}:{xy[1]:.4f}".encode()).hexdigest()[:12]
    return QueryRecord(f"q-{digest}", int(frame), (float(xy[0]), float(xy[1])), roi)


@dataclass(frozen=True)
class CropTransform:
    """Original pixel coordinates <-> model-input coordinates.

    Pixel-index convention: original pixel (i, j) has its centre at continuous
    (i + 0.5, j + 0.5). The model sees `out x out` pixels covering the source
    rectangle [x0, x0 + side_w) x [y0, y0 + side_h), which may extend past the
    image (edge pixels are repeated there).
    """

    x0: float
    y0: float
    side_w: float
    side_h: float
    out: int

    @classmethod
    def full(cls, width: int, height: int, out: int = MODEL_SIZE) -> "CropTransform":
        return cls(0.0, 0.0, float(width), float(height), int(out))

    @classmethod
    def square(cls, cx: float, cy: float, side: float, out: int = LOCAL_SIZE) -> "CropTransform":
        # Centre in continuous coordinates is (cx + 0.5, cy + 0.5).
        return cls(cx + 0.5 - side / 2.0, cy + 0.5 - side / 2.0, float(side), float(side), int(out))

    @property
    def scale_x(self) -> float:
        return self.out / self.side_w

    @property
    def scale_y(self) -> float:
        return self.out / self.side_h

    def to_model(self, x: float, y: float) -> tuple[float, float]:
        """Original pixel index -> model continuous coordinate (TAPIR raster)."""
        return (x + 0.5 - self.x0) * self.scale_x, (y + 0.5 - self.y0) * self.scale_y

    def to_original(self, u: float, v: float) -> tuple[float, float]:
        return u / self.scale_x + self.x0 - 0.5, v / self.scale_y + self.y0 - 0.5

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        u, v = self.to_model(x, y)
        m = margin * self.out
        return m <= u <= self.out - m and m <= v <= self.out - m

    def as_list(self) -> list[float]:
        return [round(self.x0, 3), round(self.y0, 3), round(self.side_w, 3), round(self.side_h, 3), self.out]


def crop_resize(frame: np.ndarray, transform: CropTransform) -> np.ndarray:
    """Bilinear sample of the transform's source rectangle (no antialiasing).

    For the full-frame transform this equals torch `F.interpolate(...,
    mode="bilinear", align_corners=False)`, which the adapter used before.
    """
    img = np.asarray(frame)
    h, w = img.shape[:2]
    out = transform.out
    u = np.arange(out, dtype=np.float64) + 0.5
    xs = transform.x0 + u / transform.scale_x - 0.5
    ys = transform.y0 + u / transform.scale_y - 0.5
    xs = np.clip(xs, 0.0, w - 1.0)
    ys = np.clip(ys, 0.0, h - 1.0)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[None, :, None]
    fy = (ys - y0)[:, None, None]
    src = img.astype(np.float32)
    if src.ndim == 2:
        src = src[..., None]
    top = src[y0][:, x0] * (1 - fx) + src[y0][:, x1] * fx
    bottom = src[y1][:, x0] * (1 - fx) + src[y1][:, x1] * fx
    result = top * (1 - fy) + bottom * fy
    return np.clip(result, 0, 255).astype(np.uint8)


def _resize_frame(frame: np.ndarray, size: int = MODEL_SIZE) -> np.ndarray:
    """Backward-compatible full-frame resize (square output)."""
    h, w = np.asarray(frame).shape[:2]
    return crop_resize(frame, CropTransform.full(w, h, size))


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=np.float64), -40.0, 40.0)))


def tapir_quality(occlusion_logit: Any, expected_distance_logit: Any):
    """Probability-like usefulness used by the official TAPIR demo."""
    if hasattr(occlusion_logit, "detach"):
        import torch

        return (1.0 - torch.sigmoid(occlusion_logit)) * (
            1.0 - torch.sigmoid(expected_distance_logit)
        )
    return (1.0 - sigmoid(occlusion_logit)) * (1.0 - sigmoid(expected_distance_logit))


# ---------------------------------------------------------------------------
# Model runners


@dataclass
class RunOutput:
    tracks: np.ndarray  # (N, 2) xy in model coordinates
    occlusion: np.ndarray  # (N,) logits
    expected_dist: np.ndarray  # (N,) logits


class PointRunner(Protocol):
    device: str

    def query_features(self, frame: np.ndarray, uv: tuple[float, float]) -> Any: ...

    def track(
        self,
        frames: np.ndarray,
        features: Any,
        *,
        query_t: int | None,
        query_uv: tuple[float, float] | None,
    ) -> RunOutput: ...


def _is_device_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    keys = ("mps", "not implemented", "notimplemented", "out of memory", "unsupported", "metal")
    return isinstance(exc, NotImplementedError) or any(key in text for key in keys)


class TorchTapirRunner:
    """Official TAPIR through its feature interface (bounded windows)."""

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = str(device)
        self.fallback_reason = ""

    def _video(self, frames: np.ndarray):
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device).float()
        return (tensor / 255.0 * 2.0 - 1.0)[None]

    def _with_fallback(self, fn):
        try:
            return fn()
        except (RuntimeError, NotImplementedError) as exc:
            if self.device != "mps" or not _is_device_error(exc):
                raise ModelNotAvailable(f"BootsTAPIR 推理失败：{exc}") from exc
            # Some TAPIR kernels lack MPS implementations. Fall back only for
            # recognised device errors and say why.
            self.fallback_reason = f"MPS 不支持，已改用 CPU：{str(exc)[:160]}"
            self.device = "cpu"
            self.model = self.model.to("cpu")
            return None

    def query_features(self, frame: np.ndarray, uv: tuple[float, float]) -> Any:
        import torch

        def run():
            video = self._video(frame[None])
            query = torch.tensor([[[0.0, uv[1], uv[0]]]], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                return self.model.get_query_features(video, False, query)

        out = self._with_fallback(run)
        if out is None:
            out = run()
        return out

    def track(
        self,
        frames: np.ndarray,
        features: Any,
        *,
        query_t: int | None,
        query_uv: tuple[float, float] | None,
    ) -> RunOutput:
        import torch

        def run():
            video = self._video(frames)
            with torch.inference_mode():
                grids = self.model.get_feature_grids(video, False)
                points = None
                feats = features
                if query_t is not None and query_uv is not None:
                    points = torch.tensor(
                        [[[float(query_t), query_uv[1], query_uv[0]]]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    feats = self.model.get_query_features(video, False, points, grids)
                elif hasattr(feats, "lowres") and feats.lowres[0].device != video.device:
                    feats = type(feats)(
                        tuple(t.to(video.device) for t in feats.lowres),
                        tuple(t.to(video.device) for t in feats.hires),
                        feats.resolutions,
                    )
                traj = self.model.estimate_trajectories(
                    video.shape[-3:-1], False, grids, feats, points, query_chunk_size=64
                )
                p = self.model.num_pips_iter
                # Same aggregation as TAPIR.forward(): mean over refinement outputs.
                tracks = torch.mean(torch.stack(traj["tracks"][p::p]), dim=0)
                occ = torch.mean(torch.stack(traj["occlusion"][p::p]), dim=0)
                dist = torch.mean(torch.stack(traj["expected_dist"][p::p]), dim=0)
            return RunOutput(
                tracks[0, 0].detach().cpu().numpy().astype(np.float64),
                occ[0, 0].detach().cpu().numpy().astype(np.float64),
                dist[0, 0].detach().cpu().numpy().astype(np.float64),
            )

        out = self._with_fallback(run)
        if out is None:
            if hasattr(features, "lowres"):
                features = type(features)(
                    tuple(t.cpu() for t in features.lowres),
                    tuple(t.cpu() for t in features.hires),
                    features.resolutions,
                )
            out = run()
        return out


class CallableRunner:
    """Black-box `model(video, query)` (tests, other checkpoints).

    Without a feature interface the real query frame is prepended to each
    window (as its own frame, t=0), so identity still comes from that frame.
    """

    def __init__(self, model: Any, device: str = "cpu") -> None:
        self.model = model
        self.device = str(device)
        self.fallback_reason = ""

    def query_features(self, frame: np.ndarray, uv: tuple[float, float]) -> Any:
        return (np.asarray(frame), (float(uv[0]), float(uv[1])))

    def track(
        self,
        frames: np.ndarray,
        features: Any,
        *,
        query_t: int | None,
        query_uv: tuple[float, float] | None,
    ) -> RunOutput:
        import torch

        prepend = query_t is None
        if prepend:
            qframe, quv = features
            stack = np.concatenate([qframe[None], frames], axis=0)
            t = 0
        else:
            stack = frames
            quv = query_uv  # type: ignore[assignment]
            t = int(query_t)
        video = torch.from_numpy(np.ascontiguousarray(stack)).to(self.device).float() / 255.0 * 2.0 - 1.0
        query = torch.tensor([[float(t), quv[1], quv[0]]], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            out = self.model(video[None], query[None])

        def last(value: Any) -> Any:
            return value[-1] if isinstance(value, (list, tuple)) else value

        tracks = last(out["tracks"])[0, 0].detach().cpu().numpy().astype(np.float64)
        occ = last(out["occlusion"])[0, 0].detach().cpu().numpy().astype(np.float64)
        dist = last(out["expected_dist"])[0, 0].detach().cpu().numpy().astype(np.float64)
        if prepend:
            tracks, occ, dist = tracks[1:], occ[1:], dist[1:]
        return RunOutput(tracks, occ, dist)


def load_bootstapir_model(*, device: str | None = None):
    try:
        # Import torch/torchvision first; tapnet's package initialiser may import
        # optional TensorFlow code on some revisions.
        import torch
        import torchvision  # noqa: F401
        from tapnet.torch import tapir_model
    except ImportError as exc:
        raise ModelNotAvailable(
            "未安装 BootsTAPIR。请执行：pip install -r requirements-ai.txt"
        ) from exc
    checkpoint = ensure_checkpoint(BOOTSTAPIR, download=False)
    target = device or select_device()
    model = tapir_model.TAPIR(pyramid_level=1)
    try:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        weights = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(weights)
    return model.eval().to(target), target


def _make_runner(model: Any, device: str) -> PointRunner:
    if hasattr(model, "track") and hasattr(model, "query_features"):
        return model  # already a runner
    if hasattr(model, "get_feature_grids") and hasattr(model, "estimate_trajectories"):
        return TorchTapirRunner(model, device)
    return CallableRunner(model, device)


# ---------------------------------------------------------------------------
# Per-frame estimates


@dataclass
class Estimate:
    frame: int
    x: float
    y: float
    occlusion: float  # probability
    uncertainty: float  # probability of being far from the true position
    route: str
    roi: list[float] | None = None
    edge_distance: float = 1.0  # distance to crop edge, fraction of crop
    centrality: float = 0.0  # frames to the nearer temporal window boundary

    @property
    def visible(self) -> bool:
        return self.occlusion < 0.5

    @property
    def precise(self) -> bool:
        return self.uncertainty < 0.5

    @property
    def quality(self) -> float:
        return float((1.0 - self.occlusion) * (1.0 - self.uncertainty))


def window_starts(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) windows over n items, consecutive windows share `overlap`."""
    if n <= 0:
        return []
    size = max(2, int(size))
    step = max(1, size - max(0, int(overlap)))
    out = []
    start = 0
    while True:
        end = min(n, start + size)
        out.append((start, end))
        if end >= n:
            break
        start += step
    return out


def _edge_distance(u: float, v: float, out: int) -> float:
    return float(min(u, v, out - u, out - v) / max(out, 1))


def _patch(gray: np.ndarray, x: float, y: float, r: int) -> np.ndarray | None:
    cx, cy = int(round(x)), int(round(y))
    if cx - r < 0 or cy - r < 0 or cx + r >= gray.shape[1] or cy + r >= gray.shape[0]:
        return None
    return gray[cy - r : cy + r + 1, cx - r : cx + r + 1].astype(np.float32)


def _ncc(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-6:
        return None
    return float(np.sum(a * b) / denom)


def _gray(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame, dtype=np.float32)
    if rgb.ndim == 2:
        return rgb
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


# ---------------------------------------------------------------------------
# Tracker


class _Frames:
    """Small LRU over decoded original frames."""

    def __init__(self, info: VideoInfo, cancel: CancelToken | None) -> None:
        self.decoder = FrameDecoder(info)
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cancel = cancel

    def get(self, index: int) -> np.ndarray:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        frame = self.decoder.frame(index)
        self.cache[index] = frame
        while len(self.cache) > FRAME_CACHE:
            self.cache.popitem(last=False)
        return frame

    def close(self) -> None:
        self.decoder.close()


class _Cancelled(Exception):
    pass


class BootsTapirTracker(Tracker):
    name = BOOTSTAPIR.model_id
    version = BOOTSTAPIR.version

    def __init__(
        self,
        model: Any | None = None,
        *,
        device: str | None = None,
        local_refine: bool = True,
        reverse_check: bool = True,
        global_window: int = GLOBAL_WINDOW,
        local_window: int = LOCAL_WINDOW,
    ) -> None:
        self._model = model
        self._device = device
        self.local_refine = bool(local_refine)
        self.reverse_check = bool(reverse_check)
        self.global_window = int(global_window)
        self.local_window = int(local_window)
        self.last_run: dict[str, Any] = {}

    # -- query ---------------------------------------------------------------

    @staticmethod
    def resolve_query(
        seed_xy: tuple[float, float],
        prompts: list | None,
        first: int,
        query: QueryRecord | None = None,
    ) -> tuple[QueryRecord, list[dict[str, Any]]]:
        """Fix the query identity; report other positive prompts, never swap.

        The desktop picks the seed from the latest positive prompt and passes
        its centre as `seed_xy`. The query is that same prompt (matched by
        position), so the adapter and the UI agree on which click is the point.
        """
        if query is not None:
            return query, []
        positives = []
        for prompt in prompts or []:
            kind = getattr(prompt, "kind", None)
            if getattr(kind, "value", kind) == "negative":
                continue
            positives.append(prompt)

        def center(p: Any) -> tuple[float, float]:
            fn = getattr(p, "center", None)
            return fn() if callable(fn) else (float(p.x), float(p.y))

        chosen = None
        for prompt in reversed(positives):
            cx, cy = center(prompt)
            if abs(cx - seed_xy[0]) < 1e-6 and abs(cy - seed_xy[1]) < 1e-6:
                chosen = prompt
                break
        if chosen is None and positives:
            chosen = positives[-1]
        if chosen is None:
            record = make_query(first, seed_xy)
        else:
            roi = None
            kind = getattr(getattr(chosen, "kind", None), "value", None)
            if kind == "box" and getattr(chosen, "x2", None) is not None:
                roi = (float(chosen.x), float(chosen.y), float(chosen.x2), float(chosen.y2))
            record = make_query(int(chosen.frame), center(chosen), roi)
        others = [
            {"frame": int(p.frame), "xy": list(center(p))}
            for p in positives
            if p is not chosen
        ]
        return record, others

    # -- main ----------------------------------------------------------------

    def track(
        self,
        info: VideoInfo,
        seed_xy: tuple[float, float],
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        prompts: list | None = None,
        query: QueryRecord | None = None,
        local_refine: bool | None = None,
        **_unused: Any,
    ) -> TrackResult:
        started = time.perf_counter()
        first = max(0, int(start_frame))
        last = info.frame_count - 1 if end_frame is None else min(int(end_frame), info.frame_count - 1)
        if last < first:
            return TrackResult(
                clip_id=info.path.stem,
                points=[],
                failure_reason=FailureReason.INTERNAL,
                model_name=self.name,
                model_version=self.version,
            )
        record, other_prompts = self.resolve_query(seed_xy, prompts, first, query)
        local = self.local_refine if local_refine is None else bool(local_refine)
        model, device = (
            (self._model, self._device or "cpu")
            if self._model is not None
            else load_bootstapir_model(device=self._device)
        )
        runner = _make_runner(model, device)
        frames = _Frames(info, cancel)
        outputs = list(range(first, last + 1))
        total_steps = max(1, len(outputs) * (2 if local else 1))
        done = [0]

        def tick(n: int, stage: str) -> None:
            done[0] += n
            _emit(progress, info.path.stem, min(done[0], total_steps), total_steps, stage)
            if cancel is not None and cancel.cancelled:
                raise _Cancelled

        try:
            global_est = self._global_route(info, frames, runner, record, outputs, tick)
            local_est: dict[int, Estimate] = {}
            if local:
                local_est = self._local_route(info, frames, runner, record, outputs, global_est, tick)
            points = self._fuse(info, frames, runner, record, outputs, global_est, local_est, local)
        except _Cancelled:
            return TrackResult(
                clip_id=info.path.stem,
                points=[],
                failure_reason=FailureReason.CANCELLED,
                model_name=self.name,
                model_version=self.version,
                elapsed_s=time.perf_counter() - started,
            )
        finally:
            frames.close()
        for point in points:
            point.diagnostics["device"] = str(runner.device)
            if other_prompts:
                point.diagnostics["unused_positive_prompts"] = other_prompts
        self.last_run = {
            "query": record.to_dict(),
            "device": str(runner.device),
            "fallback_reason": getattr(runner, "fallback_reason", ""),
            "global_input": MODEL_SIZE,
            "local_input": LOCAL_SIZE if local else None,
        }
        confidences = [p.confidence for p in points]
        result = TrackResult(
            clip_id=info.path.stem,
            points=points,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - started,
            quality_version=TRACK_QUALITY_VERSION,
        )
        result.query = record.to_dict()  # type: ignore[attr-defined]
        return result

    # -- global route --------------------------------------------------------

    def _global_route(
        self,
        info: VideoInfo,
        frames: _Frames,
        runner: PointRunner,
        record: QueryRecord,
        outputs: list[int],
        tick,
    ) -> dict[int, Estimate]:
        transform = CropTransform.full(info.width, info.height, MODEL_SIZE)
        quv = transform.to_model(*record.query_xy_original)
        features = runner.query_features(crop_resize(frames.get(record.query_frame), transform), quv)
        best: dict[int, tuple[float, Estimate]] = {}
        for s, e in window_starts(len(outputs), self.global_window, GLOBAL_OVERLAP):
            idx = outputs[s:e]
            stack = np.stack([crop_resize(frames.get(f), transform) for f in idx])
            qt = idx.index(record.query_frame) if record.query_frame in idx else None
            out = runner.track(stack, features, query_t=qt, query_uv=quv if qt is not None else None)
            for k, frame in enumerate(idx):
                u, v = float(out.tracks[k, 0]), float(out.tracks[k, 1])
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                x, y = transform.to_original(u, v)
                est = Estimate(
                    frame,
                    x,
                    y,
                    float(sigmoid(out.occlusion[k])),
                    float(sigmoid(out.expected_dist[k])),
                    "global256",
                )
                # Overlap: keep the estimate farthest from its window's edges.
                centrality = float(min(k, len(idx) - 1 - k))
                if frame not in best or centrality > best[frame][0]:
                    best[frame] = (centrality, est)
            tick(len(idx) - (GLOBAL_OVERLAP if s > 0 else 0), "track")
        return {frame: est for frame, (_c, est) in best.items()}

    # -- local route ---------------------------------------------------------

    def _roi_side(self, record: QueryRecord, span: float) -> float:
        side = max(float(ROI_MIN), span)
        if record.optional_object_roi is not None:
            x0, y0, x1, y1 = record.optional_object_roi
            side = max(side, 2.0 * max(abs(x1 - x0), abs(y1 - y0)))
        side = min(float(ROI_MAX), side)
        return float(int(np.ceil(side / ROI_STEP)) * ROI_STEP)

    def _local_route(
        self,
        info: VideoInfo,
        frames: _Frames,
        runner: PointRunner,
        record: QueryRecord,
        outputs: list[int],
        global_est: dict[int, Estimate],
        tick,
    ) -> dict[int, Estimate]:
        """Grow outward from the query frame in bounded local windows."""
        feature_cache: dict[float, tuple[Any, CropTransform]] = {}

        def query_features_for(side: float):
            # Features for this scale always come from the original query frame,
            # at the original query point, never from a predicted position.
            if side not in feature_cache:
                qx, qy = record.query_xy_original
                t = CropTransform.square(qx, qy, side, LOCAL_SIZE)
                feats = runner.query_features(crop_resize(frames.get(record.query_frame), t), t.to_model(qx, qy))
                feature_cache[side] = (feats, t)
            return feature_cache[side][0]

        result: dict[int, Estimate] = {}
        qf = record.query_frame
        forward = [f for f in outputs if f >= qf]
        backward = [f for f in reversed(outputs) if f <= qf]
        if qf not in outputs:
            # Query frame outside the requested range: grow from the nearest end.
            if qf < outputs[0]:
                forward, backward = list(outputs), []
            else:
                forward, backward = [], list(reversed(outputs))
        for order in (forward, backward):
            # The only anchor to start from is the user's own query point.
            anchors: list[tuple[int, float, float]] = [(qf, *record.query_xy_original)]
            pos = 0
            while pos < len(order):
                window = order[pos : pos + self.local_window]
                side, cx, cy = self._plan_roi(info, record, anchors, window, global_est)
                transform = CropTransform.square(cx, cy, side, LOCAL_SIZE)
                feats = query_features_for(side)
                stack = np.stack([crop_resize(frames.get(f), transform) for f in window])
                qt = window.index(qf) if qf in window else None
                quv = transform.to_model(*record.query_xy_original) if qt is not None else None
                if qt is not None and not transform.contains(*record.query_xy_original, 0.0):
                    qt, quv = None, None
                out = runner.track(stack, feats, query_t=qt, query_uv=quv)
                accepted = 0
                for k, frame in enumerate(window):
                    u, v = float(out.tracks[k, 0]), float(out.tracks[k, 1])
                    edge = _edge_distance(u, v, LOCAL_SIZE) if np.isfinite(u) and np.isfinite(v) else -1.0
                    if edge < EDGE_MARGIN and k > 0:
                        break  # rebuild the ROI from here
                    if edge < 0:
                        break
                    x, y = transform.to_original(u, v)
                    est = Estimate(
                        frame,
                        x,
                        y,
                        float(sigmoid(out.occlusion[k])),
                        float(sigmoid(out.expected_dist[k])),
                        "local512",
                        roi=transform.as_list(),
                        edge_distance=edge,
                        centrality=float(min(k, len(window) - 1 - k)),
                    )
                    # Overlap: one measurement per frame, from the window where
                    # it sits farthest from the temporal boundary (then the crop edge).
                    old = result.get(frame)
                    if old is None or (est.centrality, est.edge_distance) > (old.centrality, old.edge_distance):
                        result[frame] = est
                    accepted = k + 1
                    if est.visible and est.precise:
                        anchors.append((frame, x, y))
                        anchors = anchors[-4:]
                if accepted == 0:
                    accepted = 1  # never stall; the frame keeps its global estimate
                advance = accepted if accepted < len(window) else max(1, len(window) - LOCAL_OVERLAP)
                if pos + len(window) >= len(order) and accepted >= len(window):
                    advance = len(window)
                tick(advance, "review")
                pos += advance
        return result

    def _plan_roi(
        self,
        info: VideoInfo,
        record: QueryRecord,
        anchors: list[tuple[int, float, float]],
        window: list[int],
        global_est: dict[int, Estimate],
    ) -> tuple[float, float, float]:
        """ROI from trusted local anchors + expected motion + margin (no truth)."""
        f_last, x_last, y_last = anchors[-1]
        vx = vy = 0.0
        if len(anchors) >= 2:
            f_prev, x_prev, y_prev = anchors[-2]
            dt = max(1, abs(f_last - f_prev))
            vx, vy = (x_last - x_prev) / dt, (y_last - y_prev) / dt
        xs = [x_last]
        ys = [y_last]
        for frame in window:
            steps = abs(frame - f_last)
            xs.append(x_last + vx * steps * np.sign(frame - f_last or 1))
            ys.append(y_last + vy * steps * np.sign(frame - f_last or 1))
        speed = float(np.hypot(vx, vy))
        margin = max(64.0, 2.0 * speed * len(window) * 0.5)
        span = max(max(xs) - min(xs), max(ys) - min(ys)) + 2.0 * margin
        side = self._roi_side(record, span)
        cx = 0.5 * (max(xs) + min(xs))
        cy = 0.5 * (max(ys) + min(ys))
        # Confident global positions extend the ROI when they still fit.
        gx = [global_est[f].x for f in window if f in global_est and global_est[f].visible and global_est[f].precise]
        gy = [global_est[f].y for f in window if f in global_est and global_est[f].visible and global_est[f].precise]
        if gx:
            all_x, all_y = xs + gx, ys + gy
            wide = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) + 2.0 * margin
            if wide <= ROI_MAX:
                side = self._roi_side(record, wide)
                cx = 0.5 * (max(all_x) + min(all_x))
                cy = 0.5 * (max(all_y) + min(all_y))
        return side, cx, cy

    # -- fusion and review ---------------------------------------------------

    def _fuse(
        self,
        info: VideoInfo,
        frames: _Frames,
        runner: PointRunner,
        record: QueryRecord,
        outputs: list[int],
        global_est: dict[int, Estimate],
        local_est: dict[int, Estimate],
        local: bool,
    ) -> list[TrackPoint]:
        qx, qy = record.query_xy_original
        span = float(min(info.width, info.height))
        tol = max(3.0, 0.01 * span)
        radius = 6
        query_gray = _gray(frames.get(record.query_frame))
        query_patch = _patch(query_gray, qx, qy, radius)
        points: list[TrackPoint] = []
        reverse_budget = MAX_REVERSE_CHECKS if self.reverse_check else 0
        prev_visible = True
        last_trusted: tuple[int, float, float] | None = (record.query_frame, qx, qy)
        for frame in outputs:
            g = global_est.get(frame)
            loc = local_est.get(frame)
            chosen = loc if (local and loc is not None) else g
            diagnostics: dict[str, Any] = {
                "query_frame": record.query_frame,
                "query_xy": [qx, qy],
                "query_id": record.query_id,
            }
            if g is not None:
                diagnostics["global_xy"] = [round(g.x, 3), round(g.y, 3)]
                diagnostics["global_occlusion_probability"] = round(g.occlusion, 4)
                diagnostics["global_position_uncertainty"] = round(g.uncertainty, 4)
            if loc is not None:
                diagnostics["local_xy"] = [round(loc.x, 3), round(loc.y, 3)]
                diagnostics["local_roi"] = loc.roi
            if chosen is None:
                points.append(
                    TrackPoint(
                        frame=frame,
                        x=qx,
                        y=qy,
                        visible=False,
                        confidence=0.0,
                        status=TrackPointStatus.LOST,
                        source=TrackPointSource.AUTO,
                        note="没有可用的定位结果",
                        diagnostics=diagnostics,
                    )
                )
                prev_visible = False
                continue
            conflict = False
            if local and loc is not None and g is not None and g.visible and g.precise and not loc.visible:
                # The local crop says occluded, the global view sees it: review.
                chosen = g
                conflict = True
                diagnostics["local_says_occluded"] = True
            elif local and loc is not None and g is not None and g.visible and loc.visible:
                gap = float(np.hypot(loc.x - g.x, loc.y - g.y))
                diagnostics["global_local_gap_px"] = round(gap, 3)
                if gap > tol:
                    gray = _gray(frames.get(frame))
                    s_loc = _ncc(query_patch, _patch(gray, loc.x, loc.y, radius))
                    s_glb = _ncc(query_patch, _patch(gray, g.x, g.y, radius))
                    diagnostics["match_local"] = None if s_loc is None else round(s_loc, 4)
                    diagnostics["match_global"] = None if s_glb is None else round(s_glb, 4)
                    t_loc = t_glb = 0.0
                    if last_trusted is not None:
                        _f, lx, ly = last_trusted
                        t_loc = float(np.hypot(loc.x - lx, loc.y - ly))
                        t_glb = float(np.hypot(g.x - lx, g.y - ly))
                    if s_loc is not None and s_glb is not None and s_glb > s_loc + 0.15 and t_glb <= t_loc:
                        chosen = g
                    elif s_loc is not None and s_glb is not None and abs(s_loc - s_glb) <= 0.15:
                        conflict = True
                    elif s_loc is None or s_glb is None:
                        conflict = t_glb < t_loc
            occ_p = chosen.occlusion
            unc_p = chosen.uncertainty
            diagnostics.update(
                {
                    "route": chosen.route,
                    "occlusion_probability": round(occ_p, 4),
                    "position_uncertainty": round(unc_p, 4),
                    "raw_xy": [round(chosen.x, 3), round(chosen.y, 3)],
                }
            )
            status = TrackPointStatus.TRUSTED
            note = ""
            if not chosen.visible:
                status, note = TrackPointStatus.LOST, "目标被遮挡或不可见"
                diagnostics["reason_code"] = "model_invisible"
            elif not chosen.precise:
                status, note = TrackPointStatus.REVIEW, "位置不确定"
                diagnostics["reason_code"] = "position_uncertain"
            elif conflict:
                status, note = TrackPointStatus.REVIEW, "全图与局部结果冲突"
                diagnostics["reason_code"] = "local_conflict"
            reappeared = chosen.visible and not prev_visible
            if status is not TrackPointStatus.LOST and (conflict or reappeared) and reverse_budget > 0:
                reverse_budget -= 1
                verdict = self._reverse_check(info, frames, runner, record, frame, chosen, last_trusted, local)
                diagnostics["reverse_check"] = verdict
                if verdict == "pass" and chosen.precise:
                    status, note = TrackPointStatus.TRUSTED, ""
                    diagnostics.pop("reason_code", None)
                elif verdict == "fail":
                    status, note = TrackPointStatus.REVIEW, "反向复核没有回到原点"
                    diagnostics["reason_code"] = "reverse_mismatch"
            visible = status is TrackPointStatus.TRUSTED
            points.append(
                TrackPoint(
                    frame=frame,
                    x=float(chosen.x),
                    y=float(chosen.y),
                    visible=visible,
                    confidence=float(np.clip(chosen.quality, 0.0, 1.0)),
                    status=status,
                    source=TrackPointSource.AUTO,
                    note=note,
                    diagnostics=diagnostics,
                )
            )
            prev_visible = chosen.visible
            if visible:
                last_trusted = (frame, float(chosen.x), float(chosen.y))
        return points

    def _reverse_check(
        self,
        info: VideoInfo,
        frames: _Frames,
        runner: PointRunner,
        record: QueryRecord,
        frame: int,
        candidate: Estimate,
        last_trusted: tuple[int, float, float] | None,
        local: bool,
    ) -> str:
        """Track back from the candidate; it must return to the trusted point."""
        if last_trusted is None:
            return "insufficient"
        ref_frame, rx, ry = last_trusted
        if abs(frame - ref_frame) > REVERSE_CONTEXT:
            # Fall back to the query itself when it is close enough.
            if abs(frame - record.query_frame) <= REVERSE_CONTEXT:
                ref_frame, (rx, ry) = record.query_frame, record.query_xy_original
            else:
                return "insufficient"
        if ref_frame == frame:
            return "insufficient"
        step = 1 if ref_frame > frame else -1
        span_frames = list(range(frame, ref_frame + step, step))
        if local:
            side = self._roi_side(record, float(np.hypot(candidate.x - rx, candidate.y - ry)) + 128.0)
            cx, cy = 0.5 * (candidate.x + rx), 0.5 * (candidate.y + ry)
            transform = CropTransform.square(cx, cy, side, LOCAL_SIZE)
        else:
            transform = CropTransform.full(info.width, info.height, MODEL_SIZE)
        stack = np.stack([crop_resize(frames.get(f), transform) for f in span_frames])
        cuv = transform.to_model(candidate.x, candidate.y)
        # The reverse query exists only for this check; it is never stored.
        out = runner.track(stack, None, query_t=0, query_uv=cuv)
        u, v = float(out.tracks[-1, 0]), float(out.tracks[-1, 1])
        if not (np.isfinite(u) and np.isfinite(v)):
            return "insufficient"
        bx, by = transform.to_original(u, v)
        err = float(np.hypot(bx - rx, by - ry))
        tol = max(3.0, 0.01 * float(min(info.width, info.height)))
        if float(sigmoid(out.occlusion[-1])) >= 0.5:
            return "insufficient"
        return "pass" if err <= tol else "fail"


__all__ = [
    "BootsTapirTracker",
    "CallableRunner",
    "CropTransform",
    "Estimate",
    "LOCAL_SIZE",
    "MODEL_SIZE",
    "QueryRecord",
    "RunOutput",
    "TorchTapirRunner",
    "crop_resize",
    "load_bootstapir_model",
    "make_query",
    "tapir_quality",
    "window_starts",
]
