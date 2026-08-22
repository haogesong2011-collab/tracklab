from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

import av


@dataclass(frozen=True)
class VideoInfo:
    """Frame-accurate map of a video, built by demuxing packets without decoding."""

    path: Path
    width: int
    height: int
    pts: tuple[int, ...]
    """Presentation timestamps in stream units, ascending, one per frame."""
    time_base: float
    """Seconds per pts unit."""
    pts_ms: tuple[int, ...] = field(default_factory=tuple)

    @property
    def frame_count(self) -> int:
        return len(self.pts)

    @property
    def fps(self) -> float:
        if self.frame_count < 2:
            return 30.0
        span = self.pts_ms[-1] - self.pts_ms[0]
        if span <= 0:
            return 30.0
        return (self.frame_count - 1) * 1000.0 / span

    @property
    def duration_ms(self) -> int:
        if not self.pts_ms:
            return 0
        if self.frame_count == 1:
            return self.pts_ms[0]
        return self.pts_ms[-1] + max(self.pts_ms[-1] - self.pts_ms[-2], 0)

    def frame_delay_ms(self, index: int) -> int:
        """How long frame `index` stays on screen before the next one."""
        if self.frame_count < 2:
            return 33
        if index >= self.frame_count - 1:
            return max(self.pts_ms[-1] - self.pts_ms[-2], 1)
        return max(self.pts_ms[index + 1] - self.pts_ms[index], 1)

    def index_of_pts(self, value: int) -> int:
        i = bisect_left(self.pts, value)
        if i >= len(self.pts):
            return len(self.pts) - 1
        return i


def index_video(path: str | Path) -> VideoInfo:
    """Build the frame index. Demux-only, so this stays fast on long clips."""
    path = Path(path).expanduser().resolve()
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError("这个文件里没有视频轨道")
        stream = container.streams.video[0]
        time_base = float(stream.time_base) if stream.time_base else 1.0 / 1000.0
        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)

        raw = [
            packet.pts
            for packet in container.demux(stream)
            if packet.size and packet.pts is not None
        ]

    # Packets arrive in decode order; B-frames make that differ from display order.
    pts = tuple(sorted(set(raw)))
    if len(pts) != len(raw) or not pts:
        pts, width, height, time_base = _index_by_decoding(path)
    elif width <= 0 or height <= 0:
        width, height = _probe_size(path)

    if not pts:
        raise RuntimeError("视频里没有可用的视频帧")

    first = pts[0]
    pts_ms = tuple(int(round((value - first) * time_base * 1000)) for value in pts)
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        pts=pts,
        time_base=time_base,
        pts_ms=pts_ms,
    )


def _index_by_decoding(path: Path) -> tuple[tuple[int, ...], int, int, float]:
    """Fallback for containers with missing or duplicated packet timestamps."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate) if stream.average_rate else 30.0
        if rate <= 0:
            rate = 30.0
        time_base = float(stream.time_base) if stream.time_base else 1.0 / rate
        step = max(int(round(1.0 / rate / time_base)), 1)
        values: list[int] = []
        width = height = 0
        for position, frame in enumerate(container.decode(stream)):
            width, height = frame.width, frame.height
            value = frame.pts if frame.pts is not None else position * step
            if values and value <= values[-1]:
                value = values[-1] + step
            values.append(int(value))
    return tuple(values), width, height, time_base


def _probe_size(path: Path) -> tuple[int, int]:
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            return frame.width, frame.height
    return 0, 0
