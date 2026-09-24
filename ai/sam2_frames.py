"""Load SAM 2 video tensors with TrackLab's PyAV decoder."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any

import numpy as np

from ai.contracts import CancelToken, ProgressCb, PromptKind, TrackPrompt, TrackPoint
from ai.geometry import interpolate_xy
from ai.models import _emit
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def sampled_frame_indices(
    start: int,
    last: int,
    stride: int = 1,
    extra: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """Frames to run SAM on: start, start+stride, …, last, plus prompt frames."""
    if last < start:
        return []
    step = max(1, int(stride))
    frames = set(range(start, last + 1, step))
    frames.add(start)
    frames.add(last)
    if extra:
        for frame in extra:
            if start <= int(frame) <= last:
                frames.add(int(frame))
    return sorted(frames)


def _catmull_rom_scalar(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _control_points(
    ordered: list[TrackPoint], prev_i: int, next_i: int, span: int
) -> tuple[TrackPoint, TrackPoint] | None:
    """Outer Catmull-Rom controls, only when visible and evenly spaced by `span`."""
    if prev_i < 1 or next_i + 1 >= len(ordered):
        return None
    p0, p3 = ordered[prev_i - 1], ordered[next_i + 1]
    if not p0.visible or not p3.visible:
        return None
    if ordered[prev_i].frame - p0.frame != span or p3.frame - ordered[next_i].frame != span:
        return None
    return p0, p3


def densify_track_points(
    sampled: list[TrackPoint],
    start: int,
    last: int,
) -> list[TrackPoint]:
    """Fill [start, last]. Interpolate only between two visible sampled points.

    Uses Catmull-Rom when the two evenly spaced neighbours outside the gap are
    also visible; otherwise falls back to linear interpolation.
    """
    if last < start:
        return []
    by_frame = {point.frame: point for point in sampled}
    ordered = sorted(by_frame.values(), key=lambda point: point.frame)
    filled: list[TrackPoint] = []
    for frame in range(start, last + 1):
        existing = by_frame.get(frame)
        if existing is not None:
            filled.append(
                TrackPoint(
                    frame=existing.frame,
                    x=existing.x,
                    y=existing.y,
                    visible=existing.visible,
                    confidence=existing.confidence,
                    manual=existing.manual,
                    interpolated=False,
                    note=existing.note,
                )
            )
            continue
        prev_i: int | None = None
        next_i: int | None = None
        for index, point in enumerate(ordered):
            if point.frame < frame:
                prev_i = index
            elif point.frame > frame:
                next_i = index
                break
        prev = None if prev_i is None else ordered[prev_i]
        nxt = None if next_i is None else ordered[next_i]
        xy: tuple[float, float] | None = None
        if prev is not None and nxt is not None and prev.visible and nxt.visible:
            span = nxt.frame - prev.frame
            controls = _control_points(ordered, prev_i, next_i, span)
            if span > 0 and controls is not None:
                p0, p3 = controls
                t = (frame - prev.frame) / span
                xy = (
                    _catmull_rom_scalar(p0.x, prev.x, nxt.x, p3.x, t),
                    _catmull_rom_scalar(p0.y, prev.y, nxt.y, p3.y, t),
                )
            else:
                xy = interpolate_xy(
                    [(prev.frame, prev.x, prev.y), (nxt.frame, nxt.x, nxt.y)],
                    frame,
                )
        if xy is not None:
            filled.append(
                TrackPoint(
                    frame=frame,
                    x=xy[0],
                    y=xy[1],
                    visible=True,
                    confidence=min(prev.confidence, nxt.confidence),
                    interpolated=True,
                )
            )
            continue
        source = prev if prev is not None else nxt
        filled.append(
            TrackPoint(
                frame=frame,
                x=0.0 if source is None else source.x,
                y=0.0 if source is None else source.y,
                visible=False,
                confidence=0.0,
                interpolated=False,
            )
        )
    return filled


def remap_prompts_to_samples(
    prompts: list[TrackPrompt],
    sampled: list[int],
    seed_xy: tuple[float, float],
) -> list[TrackPrompt]:
    index = {frame: i for i, frame in enumerate(sampled)}
    remapped: list[TrackPrompt] = []
    for prompt in prompts:
        if prompt.frame in index:
            remapped.append(replace(prompt, frame=index[prompt.frame]))
    if remapped:
        return remapped
    return [
        TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=seed_xy[0], y=seed_xy[1])
    ]


def shift_prompts(
    prompts: list[TrackPrompt], start: int, last: int
) -> list[TrackPrompt]:
    """Map project frame numbers onto a sliced SAM window starting at 0."""
    shifted: list[TrackPrompt] = []
    for prompt in prompts:
        if prompt.frame < start or prompt.frame > last:
            continue
        shifted.append(replace(prompt, frame=prompt.frame - start))
    return shifted


OFFLOAD_STATE_FRAMES = 400
LAZY_FRAME_CACHE = 8


def rgb_to_sam_tensor(rgb: np.ndarray, image_size: int):
    """Resize an RGB uint8 frame to SAM 2's ImageNet-normalized CHW tensor."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).float()
    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
    return (tensor / 255.0 - mean) / std


class LazyFrameTensors:
    """SAM 2 only needs __getitem__ / __len__. Decode on demand with a small LRU."""

    def __init__(
        self,
        info: VideoInfo,
        indices: list[int],
        image_size: int,
        *,
        compute_device: Any = "cpu",
        offload_video_to_cpu: bool = True,
        cache: int = LAZY_FRAME_CACHE,
        progress: ProgressCb | None = None,
        cancel: CancelToken | None = None,
    ) -> None:
        if not indices:
            raise ValueError("empty frame range")
        self._info = info
        self._indices = list(indices)
        self._image_size = int(image_size)
        self._compute_device = compute_device
        self._offload = bool(offload_video_to_cpu)
        self._cache_n = max(1, int(cache))
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._decoder: FrameDecoder | None = None
        self._progress = progress
        self._cancel = cancel
        self._decoded = 0
        self.video_height = info.height
        self.video_width = info.width

    def __len__(self) -> int:
        return len(self._indices)

    def close(self) -> None:
        self._cache.clear()
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None

    def __getitem__(self, index: int):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        pos = int(index)
        if pos < 0:
            pos += len(self._indices)
        cached = self._cache.get(pos)
        if cached is not None:
            self._cache.move_to_end(pos)
            return cached
        if self._cancel is not None and self._cancel.cancelled:
            raise RuntimeError("frame decode cancelled")
        if self._decoder is None:
            self._decoder = FrameDecoder(self._info)
        tensor = rgb_to_sam_tensor(
            self._decoder.frame(self._indices[pos]), self._image_size
        )
        if not self._offload:
            tensor = tensor.to(self._compute_device)
        self._cache[pos] = tensor
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        self._decoded += 1
        _emit(
            self._progress,
            self._info.path.stem,
            self._decoded,
            len(self._indices),
            "decode",
        )
        return tensor


def load_frames_for_sam2(
    info: VideoInfo,
    image_size: int,
    start: int,
    last: int,
    *,
    compute_device: Any = "cpu",
    offload_video_to_cpu: bool = True,
    progress: ProgressCb | None = None,
    cancel: CancelToken | None = None,
    frame_indices: list[int] | None = None,
):
    """Eager stack of selected frames. Tests compare this to LazyFrameTensors."""
    import torch

    if frame_indices is None:
        indices = list(range(start, last + 1))
    else:
        indices = list(frame_indices)
    total = len(indices)
    if total <= 0:
        raise ValueError("empty frame range")
    decoder = FrameDecoder(info)
    chunks: list = []
    try:
        for n, i in enumerate(indices):
            if cancel and cancel.cancelled:
                break
            chunks.append(rgb_to_sam_tensor(decoder.frame(i), image_size))
            _emit(progress, info.path.stem, n + 1, total, "decode")
    finally:
        decoder.close()
    if not chunks:
        return None, info.height, info.width
    images = torch.stack(chunks, dim=0)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
    return images, info.height, info.width


def build_video_state(
    predictor: Any,
    images,
    video_height: int,
    video_width: int,
    *,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool | None = None,
) -> dict:
    """Same inference-state dict as SAM2VideoPredictor.init_state, using our tensors."""
    import torch

    n_frames = len(images)
    if offload_state_to_cpu is None:
        offload_state_to_cpu = n_frames >= OFFLOAD_STATE_FRAMES
    compute_device = predictor.device
    state = {
        "images": images,
        "num_frames": n_frames,
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
        "video_height": video_height,
        "video_width": video_width,
        "device": compute_device,
        "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "frames_tracked_per_obj": {},
    }
    with torch.inference_mode():
        predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    return state


def is_sam2_predictor(predictor: Any) -> bool:
    return hasattr(predictor, "image_size") and hasattr(predictor, "_get_image_feature")


def window_prompts(
    prompts: list[TrackPrompt],
    start: int,
    last: int,
    seed_xy: tuple[float, float],
) -> list[TrackPrompt]:
    shifted = shift_prompts(prompts, start, last)
    if shifted:
        return shifted
    return [
        TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=seed_xy[0], y=seed_xy[1])
    ]
