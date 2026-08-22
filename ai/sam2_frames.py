"""Load SAM 2 video tensors with TrackLab's PyAV decoder."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any

import numpy as np

from ai.contracts import CancelToken, ProgressCb, PromptKind, TrackPrompt
from ai.models import _emit
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


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
):
    """Decode [start, last] with FrameDecoder and match SAM 2's ImageNet tensor layout."""
    import torch
    import torch.nn.functional as F

    total = last - start + 1
    if total <= 0:
        raise ValueError("empty frame range")
    decoder = FrameDecoder(info)
    chunks: list = []
    try:
        for i in range(start, last + 1):
            if cancel and cancel.cancelled:
                break
            rgb = decoder.frame(i)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            tensor = tensor.unsqueeze(0).float()
            tensor = F.interpolate(
                tensor,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )
            chunks.append(tensor.squeeze(0))
            _emit(progress, info.path.stem, i - start + 1, total, "decode")
    finally:
        decoder.close()
    if not chunks:
        return None, info.height, info.width
    images = torch.stack(chunks, dim=0) / 255.0
    mean = torch.tensor(IMG_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMG_STD, dtype=torch.float32)[:, None, None]
    images = (images - mean) / std
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        mean = mean.to(compute_device)
        std = std.to(compute_device)
    return images, info.height, info.width


def build_video_state(
    predictor: Any,
    images,
    video_height: int,
    video_width: int,
    *,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = False,
) -> dict:
    """Same inference-state dict as SAM2VideoPredictor.init_state, using our tensors."""
    import torch

    compute_device = predictor.device
    state = {
        "images": images,
        "num_frames": len(images),
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
