from __future__ import annotations

from collections.abc import Iterator

import av
import numpy as np

from engine.video_index import VideoInfo


class FrameDecoder:
    """Random-access frame reader over an open container.

    Reading the frame right after the previous one costs about a millisecond;
    jumping anywhere else costs a keyframe seek, typically under 25 ms. Not
    thread safe: PyAV state belongs to whichever thread created it.
    """

    def __init__(self, info: VideoInfo) -> None:
        self._info = info
        self._container = av.open(str(info.path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._iter: Iterator[av.VideoFrame] | None = None
        self._next_index = -1

    def close(self) -> None:
        self._iter = None
        self._container.close()

    def frame(self, index: int) -> np.ndarray:
        """RGB24 pixels for `index`, as an (H, W, 3) array."""
        index = max(0, min(index, self._info.frame_count - 1))
        if self._iter is None or index != self._next_index:
            self._seek(index)
        return self._advance_to(index, allow_rewind=True)

    def _seek(self, index: int) -> None:
        target = self._info.pts[index]
        self._container.seek(target, stream=self._stream, backward=True)
        self._iter = self._container.decode(self._stream)
        self._next_index = -1

    def _rewind(self) -> None:
        self._container.seek(self._info.pts[0], stream=self._stream, backward=True)
        self._iter = self._container.decode(self._stream)
        self._next_index = -1

    def _advance_to(self, index: int, *, allow_rewind: bool) -> np.ndarray:
        assert self._iter is not None
        last: av.VideoFrame | None = None
        while True:
            try:
                frame = next(self._iter)
            except StopIteration:
                if last is not None:
                    return last.to_ndarray(format="rgb24")
                raise RuntimeError("解码到文件末尾仍未取到该帧") from None
            position = self._position_of(frame)
            self._next_index = position + 1
            if position == index:
                return frame.to_ndarray(format="rgb24")
            if position > index:
                if not allow_rewind:
                    return frame.to_ndarray(format="rgb24")
                # The seek landed past the target; restart from the top once.
                self._rewind()
                return self._advance_to(index, allow_rewind=False)
            last = frame

    def _position_of(self, frame: av.VideoFrame) -> int:
        if frame.pts is None:
            return max(self._next_index, 0)
        return self._info.index_of_pts(frame.pts)
