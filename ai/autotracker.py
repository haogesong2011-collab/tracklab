"""Tracker-style single-object autotracker.

This is an independent Python implementation of the algorithm documented by
Open Source Physics Tracker's AutoTracker/TemplateMatcher: RGB squared
difference matching, peak-height rejection, sub-pixel parabolic refinement,
look-ahead prediction, and evolved templates tethered to the key frame.
"""

from __future__ import annotations

import math
import time
from typing import Any

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
from ai.models import Tracker, _emit
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo

DEFAULT_TEMPLATE = 31
MIN_TEMPLATE = 15
MAX_TEMPLATE = 72
SEARCH_RADIUS = 48
EXPANDED_SEARCH_RADIUS = 96
GOOD_MATCH = 4.0
EVOLVE_RATE = 0.20
TETHER_RATE = 0.05
PREDICTION_LOOKBACK = 4


def _template_spec(
    prompts: list[TrackPrompt], seed_xy: tuple[float, float]
) -> tuple[float, float, int, int, bool]:
    cx, cy = seed_xy
    width = height = DEFAULT_TEMPLATE
    rectangular = False
    for prompt in prompts:
        if prompt.kind is PromptKind.NEGATIVE:
            continue
        if prompt.kind is PromptKind.BOX and prompt.x2 is not None and prompt.y2 is not None:
            x0, x1 = sorted((prompt.x, prompt.x2))
            y0, y1 = sorted((prompt.y, prompt.y2))
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            width = int(round(max(MIN_TEMPLATE, min(MAX_TEMPLATE, x1 - x0))))
            height = int(round(max(MIN_TEMPLATE, min(MAX_TEMPLATE, y1 - y0))))
            rectangular = True
        else:
            cx, cy = prompt.x, prompt.y
            width = height = DEFAULT_TEMPLATE
            rectangular = False
    return cx, cy, width, height, rectangular


def _template_mask(width: int, height: int, rectangular: bool) -> np.ndarray:
    if rectangular:
        return np.ones((height, width), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    rx = max((width - 1) / 2.0, 1.0)
    ry = max((height - 1) / 2.0, 1.0)
    return (
        ((xx - rx) / rx) ** 2 + ((yy - ry) / ry) ** 2 <= 1.0
    ).astype(np.float32)


def _extract(
    frame: np.ndarray, cx: float, cy: float, width: int, height: int
) -> np.ndarray | None:
    image_h, image_w = frame.shape[:2]
    x0 = int(round(cx - width / 2.0))
    y0 = int(round(cy - height / 2.0))
    x1, y1 = x0 + width, y0 + height
    if x0 < 0 or y0 < 0 or x1 > image_w or y1 > image_h:
        return None
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    if float(patch.std()) < 3.0:
        return None
    return patch


def _conv_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    ih, iw = image.shape
    kh, kw = kernel.shape
    out_h, out_w = ih - kh + 1, iw - kw + 1
    if out_h < 1 or out_w < 1:
        return np.empty((0, 0), dtype=np.float64)
    shape = (ih + kh - 1, iw + kw - 1)
    full = np.fft.irfft2(
        np.fft.rfft2(image, s=shape) * np.fft.rfft2(kernel[::-1, ::-1], s=shape),
        s=shape,
    )
    return full[kh - 1 : kh - 1 + out_h, kw - 1 : kw - 1 + out_w]


def rgb_squared_difference_map(
    image: np.ndarray, template: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Return Tracker TemplateMatcher's RGBSqD at every valid template position."""
    image = np.asarray(image, dtype=np.float32)
    template = np.asarray(template, dtype=np.float32)
    th, tw = template.shape[:2]
    if image.shape[0] < th or image.shape[1] < tw:
        return np.empty((0, 0), dtype=np.float64)
    weights = (
        np.ones((th, tw), dtype=np.float32)
        if mask is None
        else np.asarray(mask, dtype=np.float32)
    )
    result: np.ndarray | None = None
    for channel in range(3):
        target = image[:, :, channel]
        source = template[:, :, channel]
        target_sq = _conv_valid(target * target, weights)
        cross = _conv_valid(target, weights * source)
        source_sq = float(np.sum(weights * source * source))
        channel_ssd = target_sq + source_sq - 2.0 * cross
        result = channel_ssd if result is None else result + channel_ssd
    assert result is not None
    return np.maximum(result, 0.0)


def _parabolic_offset(left: float, center: float, right: float) -> tuple[float, float]:
    curvature = 0.5 * (left + right) - center
    if curvature <= 1e-9:
        return 0.0, math.inf
    offset = 0.25 * (left - right) / curvature
    offset = max(-1.0, min(1.0, offset))
    width = math.sqrt(max(2.0 * center / curvature, 0.0))
    return offset, width


def _match(
    frame: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    predicted_x: float,
    predicted_y: float,
    radius: int,
) -> tuple[float, float, float, float]:
    th, tw = template.shape[:2]
    image_h, image_w = frame.shape[:2]
    x0 = max(0, int(round(predicted_x - tw / 2.0 - radius)))
    y0 = max(0, int(round(predicted_y - th / 2.0 - radius)))
    x1 = min(image_w, int(round(predicted_x + tw / 2.0 + radius)) + 1)
    y1 = min(image_h, int(round(predicted_y + th / 2.0 + radius)) + 1)
    search = frame[y0:y1, x0:x1]
    diffs = rgb_squared_difference_map(search, template, mask)
    if diffs.size == 0:
        return predicted_x, predicted_y, 0.0, math.inf
    row, col = np.unravel_index(int(np.argmin(diffs)), diffs.shape)
    minimum = float(diffs[row, col])
    average = float(np.mean(diffs))
    peak = math.inf if minimum <= 1e-9 else max(average / minimum - 1.0, 0.0)

    dx = dy = 0.0
    widths: list[float] = []
    if 0 < col < diffs.shape[1] - 1:
        dx, width = _parabolic_offset(
            float(diffs[row, col - 1]), minimum, float(diffs[row, col + 1])
        )
        if math.isfinite(width):
            widths.append(width)
    if 0 < row < diffs.shape[0] - 1:
        dy, width = _parabolic_offset(
            float(diffs[row - 1, col]), minimum, float(diffs[row + 1, col])
        )
        if math.isfinite(width):
            widths.append(width)
    peak_width = float(np.mean(widths)) if widths else math.inf
    if math.isfinite(peak) and math.isfinite(peak_width) and peak_width > 1.0:
        peak /= peak_width
    x = x0 + col + tw / 2.0 + dx
    y = y0 + row + th / 2.0 + dy
    return x, y, peak, peak_width


def _predict(points: list[TrackPoint], fallback: tuple[float, float]) -> tuple[float, float]:
    recent = [point for point in points[-PREDICTION_LOOKBACK:] if point.visible]
    if not recent:
        return fallback
    if len(recent) == 1:
        return recent[-1].x, recent[-1].y
    newest = recent[::-1]

    def predict_axis(attribute: str) -> float:
        values = [float(getattr(point, attribute)) for point in newest]
        velocity = [values[i] - values[i + 1] for i in range(len(values) - 1)]
        velocity_mean = abs(float(np.mean(velocity)))
        velocity_valid = len(values) < 3 or abs(velocity[0] - velocity[1]) < velocity_mean
        acceleration = [
            velocity[i] - velocity[i + 1] for i in range(len(velocity) - 1)
        ]
        acceleration_valid = False
        if len(values) >= 3:
            acceleration_valid = len(values) < 4
            if len(values) >= 4:
                acceleration_mean = abs(float(np.mean(acceleration)))
                jerk = acceleration[0] - acceleration[1]
                acceleration_valid = abs(jerk) < acceleration_mean
        if acceleration_valid:
            return values[0] + velocity[0] + acceleration[0]
        if velocity_valid:
            return values[0] + velocity[0]
        return values[0]

    return predict_axis("x"), predict_axis("y")


class TrackerAutoTracker(Tracker):
    """Fast desktop tracker modeled after official Tracker Autotracker."""

    name = "tracker_autotracker"
    version = "1.0.0"

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
        **_unused: Any,
    ) -> TrackResult:
        started = time.perf_counter()
        start = max(0, int(start_frame))
        last = info.frame_count - 1 if end_frame is None else min(
            int(end_frame), info.frame_count - 1
        )
        if last < start:
            return self._result(info, [], FailureReason.INTERNAL, started)

        cx, cy, width, height, rectangular = _template_spec(
            list(prompts or []), seed_xy
        )
        mask = _template_mask(width, height, rectangular)
        decoder = FrameDecoder(info)
        points: list[TrackPoint] = []
        total = last - start + 1
        lost_streak = 0
        try:
            seed_frame = decoder.frame(start)
            key_template = _extract(seed_frame, cx, cy, width, height)
            if key_template is None:
                return self._result(info, [], FailureReason.LOW_CONFIDENCE, started)
            template = key_template.copy()
            _emit(progress, info.path.stem, 0, total, "track")
            for frame_number in range(start, last + 1):
                if cancel and cancel.cancelled:
                    return self._result(
                        info, points, FailureReason.CANCELLED, started
                    )
                frame = seed_frame if frame_number == start else decoder.frame(frame_number)
                if frame_number == start:
                    x, y, peak = cx, cy, math.inf
                    visible = True
                else:
                    predicted = _predict(points, (cx, cy))
                    x, y, peak, _width = _match(
                        frame,
                        template,
                        mask,
                        predicted[0],
                        predicted[1],
                        SEARCH_RADIUS,
                    )
                    if peak < GOOD_MATCH:
                        x, y, peak, _width = _match(
                            frame,
                            template,
                            mask,
                            predicted[0],
                            predicted[1],
                            EXPANDED_SEARCH_RADIUS,
                        )
                    visible = peak >= GOOD_MATCH
                confidence = 1.0 if math.isinf(peak) else peak / (peak + GOOD_MATCH)
                if visible:
                    cx, cy = x, y
                    lost_streak = 0
                    match_image = _extract(frame, cx, cy, width, height)
                    if match_image is not None and match_image.shape == template.shape:
                        evolved = (1.0 - EVOLVE_RATE) * template + EVOLVE_RATE * match_image
                        template = (
                            (1.0 - TETHER_RATE) * evolved
                            + TETHER_RATE * key_template
                        )
                else:
                    lost_streak += 1
                    x, y = _predict(points, (cx, cy))
                points.append(
                    TrackPoint(
                        frame=frame_number,
                        x=float(x),
                        y=float(y),
                        visible=visible,
                        confidence=float(max(0.0, min(1.0, confidence))),
                    )
                )
                if frame_number == start or frame_number == last or (frame_number - start) % 4 == 0:
                    _emit(
                        progress,
                        info.path.stem,
                        frame_number - start + 1,
                        total,
                        "track",
                    )
        finally:
            decoder.close()

        reason = (
            FailureReason.TARGET_LOST
            if lost_streak >= 8
            else FailureReason.NONE
        )
        return self._result(info, points, reason, started)

    def _result(
        self,
        info: VideoInfo,
        points: list[TrackPoint],
        reason: FailureReason,
        started: float,
    ) -> TrackResult:
        return TrackResult(
            clip_id=info.path.stem,
            points=points,
            confidence=float(np.mean([p.confidence for p in points])) if points else 0.0,
            failure_reason=reason,
            model_name=self.name,
            model_version=self.version,
            elapsed_s=time.perf_counter() - started,
        )
