"""SAM 2 adapter: prompts → video masks → TrackResult centroids."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Iterable, Protocol

import numpy as np

from ai.contracts import (
    CancelToken,
    FailureReason,
    ProgressCb,
    PromptKind,
    TrackPoint,
    TrackPrompt,
    TrackResult,
)
from ai.models import Tracker, _emit, load_video
from ai.model_manager import DEFAULT_SPEC, ModelNotAvailable, ModelSpec, load_sam2_predictor
from ai.sam2_frames import (
    LazyFrameTensors,
    build_video_state,
    densify_track_points,
    is_sam2_predictor,
    remap_prompts_to_samples,
    sampled_frame_indices,
)
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo


OFFLOAD_VIDEO_BYTES = 512 * 1024 * 1024
TRACK_WINDOW_SAMPLED = 300


def sample_track_windows(
    sampled: list[int], size: int = TRACK_WINDOW_SAMPLED
) -> list[list[int]]:
    """Split sampled frames into windows of `size` with one overlapping frame."""
    if not sampled:
        return []
    width = max(2, int(size))
    if len(sampled) <= width:
        return [list(sampled)]
    windows: list[list[int]] = []
    start = 0
    n = len(sampled)
    while start < n:
        end = min(start + width, n)
        windows.append(list(sampled[start:end]))
        if end >= n:
            break
        start = end - 1
    return windows


def should_offload_video(device: Any, n_frames: int, image_size: int) -> bool:
    kind = str(getattr(device, "type", device)).lower()
    if "cpu" in kind:
        return True
    return n_frames * 3 * image_size * image_size * 4 > OFFLOAD_VIDEO_BYTES


class VideoPredictor(Protocol):
    def init_state(self, video_path: str, **kwargs: Any) -> dict: ...

    def add_new_points_or_box(self, *args: Any, **kwargs: Any) -> tuple: ...

    def propagate_in_video(self, *args: Any, **kwargs: Any) -> Iterable: ...


def as_numpy(value: Any) -> np.ndarray:
    """Copy MPS/CUDA tensors to host before NumPy sees them."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value)


def logits_to_mask(logits: Any) -> np.ndarray:
    arr = as_numpy(logits)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0]
    return arr > 0.0


def mask_centroid(
    mask: np.ndarray,
) -> tuple[float, float, float, bool, list[tuple[float, float]]]:
    """Return (x, y, confidence, visible, contour) from a boolean mask."""
    binary = as_numpy(mask).astype(bool)
    if binary.ndim != 2:
        binary = binary.reshape(binary.shape[-2], binary.shape[-1])
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return 0.0, 0.0, 0.0, False, []
    x = float(xs.mean())
    y = float(ys.mean())
    area = float(xs.size)
    conf = float(min(1.0, area / 64.0))
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    contour = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return x, y, conf, True, contour


def merge_track_points(
    existing: list[TrackPoint], incoming: list[TrackPoint], from_frame: int
) -> list[TrackPoint]:
    """Keep confirmed points before from_frame; replace from that frame onward."""
    kept = [p for p in existing if p.frame < from_frame]
    by_frame = {p.frame: p for p in kept}
    for point in incoming:
        if point.frame >= from_frame:
            by_frame[point.frame] = point
    return [by_frame[key] for key in sorted(by_frame)]


class FakeVideoPredictor:
    """Deterministic stand-in used by unit tests. Does not load weights."""

    def __init__(
        self,
        radius: int = 8,
        drift: tuple[float, float] = (1.0, 0.0),
        blank_from: int | None = None,
    ) -> None:
        self.radius = radius
        self.drift = drift
        self.blank_from = blank_from
        self._cx = 0.0
        self._cy = 0.0
        self._obj: dict[int, tuple[float, float]] = {}

    def init_state(self, video_path: str, **kwargs: Any) -> dict:
        info = load_video(video_path)
        return {"info": info, "path": video_path}

    def add_new_points_or_box(
        self,
        inference_state: dict,
        frame_idx: int = 0,
        obj_id: int = 1,
        points: Any = None,
        labels: Any = None,
        box: Any = None,
        **kwargs: Any,
    ) -> tuple:
        info: VideoInfo = inference_state["info"]
        if box is not None:
            box = np.asarray(box, dtype=np.float32).reshape(-1)
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
        elif points is not None and len(points) > 0:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            labs = np.ones(len(pts)) if labels is None else np.asarray(labels).reshape(-1)
            pos = pts[labs > 0]
            if len(pos) == 0:
                pos = pts
            cx, cy = float(pos[0, 0]), float(pos[0, 1])
        else:
            cx, cy = info.width / 2.0, info.height / 2.0
        self._obj[int(obj_id)] = (cx, cy)
        self._cx, self._cy = cx, cy
        mask = self._disk(info, cx, cy)
        return frame_idx, [obj_id], [mask]

    def propagate_in_video(
        self,
        inference_state: dict,
        start_frame_idx: int = 0,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        **kwargs: Any,
    ):
        info: VideoInfo = inference_state["info"]
        last = info.frame_count - 1
        if max_frame_num_to_track is None:
            end = last
        else:
            end = min(last, start_frame_idx + max_frame_num_to_track - 1)
        frames = range(start_frame_idx, end + 1)
        if reverse:
            frames = range(start_frame_idx, -1, -1)
        cx, cy = self._cx, self._cy
        for i, frame in enumerate(frames):
            x = cx + self.drift[0] * i
            y = cy + self.drift[1] * i
            if self.blank_from is not None and frame >= self.blank_from:
                yield frame, [1], [self._disk(info, x, y, radius=0)]
            else:
                yield frame, [1], [self._disk(info, x, y)]

    def _disk(
        self, info: VideoInfo, cx: float, cy: float, radius: int | None = None
    ) -> np.ndarray:
        rad = self.radius if radius is None else radius
        if rad <= 0:
            logits = np.full((info.height, info.width), -8.0, dtype=np.float32)
            return logits[None, ...]
        yy, xx = np.ogrid[: info.height, : info.width]
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
        logits = np.where(inside, 8.0, -8.0).astype(np.float32)
        return logits[None, ...]


class Sam2Tracker(Tracker):
    """Official desktop tracker. Requires SAM 2 unless a predictor is injected."""

    name = DEFAULT_SPEC.model_id
    version = DEFAULT_SPEC.version

    def __init__(
        self,
        predictor: VideoPredictor | None = None,
        *,
        spec: ModelSpec | None = None,
    ) -> None:
        self._predictor = predictor
        self._spec = spec or DEFAULT_SPEC
        self.name = self._spec.model_id
        self.version = self._spec.version

    def track(
        self,
        info: VideoInfo,
        seed_xy: tuple[float, float],
        *,
        cancel: CancelToken | None = None,
        progress: ProgressCb | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        prompts: list[TrackPrompt] | None = None,
        object_id: int = 1,
        stride: int = 1,
        image_size: int | None = None,
        **_unused: Any,
    ) -> TrackResult:
        t0 = time.perf_counter()
        start = max(0, int(start_frame))
        last = info.frame_count - 1 if end_frame is None else min(int(end_frame), info.frame_count - 1)
        if last < start:
            return TrackResult(
                clip_id=info.path.stem,
                points=[],
                failure_reason=FailureReason.INTERNAL,
                model_name=self.name,
                model_version=self.version,
                elapsed_s=time.perf_counter() - t0,
            )
        hints = list(prompts or [])
        if not hints:
            hints.append(
                TrackPrompt(
                    frame=start, kind=PromptKind.POSITIVE, x=seed_xy[0], y=seed_xy[1]
                )
            )
        extra = [prompt.frame for prompt in hints]
        sampled = sampled_frame_indices(start, last, stride, extra)
        predictor = self._predictor
        if predictor is None:
            predictor = load_sam2_predictor(self._spec, download=False)
            self._predictor = predictor

        sampled_points: list[TrackPoint] = []
        contours: dict[int, list[tuple[float, float]]] = {}
        total = last - start + 1
        _emit(progress, info.path.stem, 0, total, "track")
        cancelled = False
        if is_sam2_predictor(predictor):
            cancelled = self._track_windows(
                predictor,
                info,
                sampled,
                hints,
                seed_xy,
                object_id,
                start,
                total,
                sampled_points,
                contours,
                cancel=cancel,
                progress=progress,
            )
        else:
            cancelled = self._track_whole(
                predictor,
                info,
                sampled,
                hints,
                seed_xy,
                object_id,
                start,
                last,
                total,
                sampled_points,
                contours,
                cancel=cancel,
                progress=progress,
            )
        if cancelled:
            return TrackResult(
                clip_id=info.path.stem,
                points=sampled_points,
                failure_reason=FailureReason.CANCELLED,
                model_name=self.name,
                model_version=self.version,
                elapsed_s=time.perf_counter() - t0,
            )

        sampled_points.sort(key=lambda point: point.frame)
        points = densify_track_points(sampled_points, start, last)
        result = TrackResult(
            clip_id=info.path.stem,
            points=points,
            confidence=float(np.mean([p.confidence for p in points])) if points else 0.0,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - t0,
        )
        result.contours = contours  # type: ignore[attr-defined]
        return result

    def _record(
        self,
        payload: Any,
        abs_frame: int,
        seed_xy: tuple[float, float],
        object_id: int,
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
    ) -> None:
        _idx, obj_ids, masks = payload[:3]
        mask = self._pick_mask(obj_ids, masks, object_id)
        x, y, conf, visible, contour = mask_centroid(logits_to_mask(mask))
        if not visible:
            x, y = seed_xy
        out_points.append(
            TrackPoint(frame=abs_frame, x=x, y=y, visible=visible, confidence=conf)
        )
        if contour:
            out_contours[abs_frame] = contour

    @staticmethod
    def _window_prompts(
        hints: list[TrackPrompt],
        window: list[int],
        seed_xy: tuple[float, float],
        carry: TrackPrompt | None,
    ) -> list[TrackPrompt]:
        if carry is None:
            return remap_prompts_to_samples(hints, window, seed_xy)
        index = {frame: i for i, frame in enumerate(window)}
        local = [carry]
        local += [replace(p, frame=index[p.frame]) for p in hints if p.frame in index]
        return local

    @staticmethod
    def _carry_prompt(
        contours: dict[int, list[tuple[float, float]]],
        points: list[TrackPoint],
        frame: int,
    ) -> TrackPrompt | None:
        """Seed the next window from the last mask of the previous one."""
        contour = contours.get(frame)
        if contour:
            xs = [pt[0] for pt in contour]
            ys = [pt[1] for pt in contour]
            return TrackPrompt(
                frame=0,
                kind=PromptKind.BOX,
                x=min(xs),
                y=min(ys),
                x2=max(xs),
                y2=max(ys),
            )
        point = next(
            (p for p in reversed(points) if p.frame == frame and p.visible), None
        )
        if point is None:
            return None
        return TrackPrompt(
            frame=0, kind=PromptKind.POSITIVE, x=point.x, y=point.y
        )

    def _track_windows(
        self,
        predictor: Any,
        info: VideoInfo,
        sampled: list[int],
        hints: list[TrackPrompt],
        seed_xy: tuple[float, float],
        object_id: int,
        start: int,
        total: int,
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
        *,
        cancel: CancelToken | None,
        progress: ProgressCb | None,
    ) -> bool:
        """Run SAM window by window so memory does not grow with video length."""
        native = int(predictor.image_size)
        seen: set[int] = set()
        carry: TrackPrompt | None = None
        for window in sample_track_windows(sampled, TRACK_WINDOW_SAMPLED):
            if cancel and cancel.cancelled:
                return True
            offload = should_offload_video(predictor.device, len(window), native)
            images = LazyFrameTensors(
                info,
                window,
                native,
                compute_device=predictor.device,
                offload_video_to_cpu=offload,
                progress=progress,
                cancel=cancel,
            )
            try:
                state = build_video_state(
                    predictor,
                    images,
                    info.height,
                    info.width,
                    offload_video_to_cpu=offload,
                )
                self._apply_prompts(
                    predictor,
                    state,
                    self._window_prompts(hints, window, seed_xy, carry),
                    object_id,
                )
                if cancel and cancel.cancelled:
                    return True
                stream = predictor.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=len(window)
                )
                for payload in stream:
                    if cancel and cancel.cancelled:
                        return True
                    idx = int(payload[0])
                    if idx < 0 or idx >= len(window):
                        continue
                    abs_frame = window[idx]
                    if abs_frame in seen:
                        continue
                    seen.add(abs_frame)
                    self._record(
                        payload, abs_frame, seed_xy, object_id, out_points, out_contours
                    )
                    _emit(
                        progress, info.path.stem, abs_frame - start + 1, total, "track"
                    )
            except ModelNotAvailable:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ModelNotAvailable(f"SAM 2 推理失败：{exc}") from exc
            finally:
                images.close()
            carry = self._carry_prompt(out_contours, out_points, window[-1])
        return False

    def _track_whole(
        self,
        predictor: Any,
        info: VideoInfo,
        sampled: list[int],
        hints: list[TrackPrompt],
        seed_xy: tuple[float, float],
        object_id: int,
        start: int,
        last: int,
        total: int,
        out_points: list[TrackPoint],
        out_contours: dict[int, list[tuple[float, float]]],
        *,
        cancel: CancelToken | None,
        progress: ProgressCb | None,
    ) -> bool:
        """Predictors that decode the file themselves (tests) stay single-pass."""
        state = predictor.init_state(str(info.path))
        self._apply_prompts(predictor, state, hints, object_id)
        if cancel and cancel.cancelled:
            return True
        allowed = set(sampled)
        try:
            stream = predictor.propagate_in_video(
                state,
                start_frame_idx=start,
                max_frame_num_to_track=last - start + 1,
            )
            for payload in stream:
                if cancel and cancel.cancelled:
                    return True
                abs_frame = int(payload[0])
                if abs_frame not in allowed:
                    continue
                self._record(
                    payload, abs_frame, seed_xy, object_id, out_points, out_contours
                )
                _emit(progress, info.path.stem, abs_frame - start + 1, total, "track")
        except ModelNotAvailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelNotAvailable(f"SAM 2 推理失败：{exc}") from exc
        return False

    @staticmethod
    def _pick_mask(obj_ids: Any, masks: Any, object_id: int) -> Any:
        ids = [int(i) for i in list(obj_ids)]
        if object_id in ids:
            return masks[ids.index(object_id)]
        return masks[0]

    @staticmethod
    def _apply_prompts(
        predictor: VideoPredictor,
        state: dict,
        prompts: list[TrackPrompt],
        object_id: int,
    ) -> None:
        by_frame: dict[int, list[TrackPrompt]] = {}
        for prompt in prompts:
            by_frame.setdefault(prompt.frame, []).append(prompt)
        for frame, group in sorted(by_frame.items()):
            points: list[list[float]] = []
            labels: list[int] = []
            box = None
            for prompt in group:
                if prompt.kind == PromptKind.BOX and prompt.x2 is not None and prompt.y2 is not None:
                    box = [prompt.x, prompt.y, prompt.x2, prompt.y2]
                elif prompt.kind == PromptKind.NEGATIVE:
                    points.append([prompt.x, prompt.y])
                    labels.append(0)
                else:
                    points.append([prompt.x, prompt.y])
                    labels.append(1)
            kwargs: dict[str, Any] = {
                "inference_state": state,
                "frame_idx": frame,
                "obj_id": object_id,
            }
            if points:
                kwargs["points"] = np.array(points, dtype=np.float32)
                kwargs["labels"] = np.array(labels, dtype=np.int32)
            if box is not None:
                kwargs["box"] = np.array(box, dtype=np.float32)
            predictor.add_new_points_or_box(**kwargs)


def preview_mask_on_frame(
    info: VideoInfo,
    frame_index: int,
    seed_xy: tuple[float, float],
    radius: int = 12,
) -> np.ndarray:
    """CPU-only preview disk used when SAM is not yet loaded (tests / fallback UI)."""
    decoder = FrameDecoder(info)
    try:
        frame = decoder.frame(frame_index)
    finally:
        decoder.close()
    h, w = frame.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    return (xx - seed_xy[0]) ** 2 + (yy - seed_xy[1]) ** 2 <= radius**2
