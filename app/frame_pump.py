from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QMutex, QMutexLocker, QThread, QWaitCondition, Signal
from PySide6.QtGui import QImage

from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo, index_video

CACHE_BUDGET_BYTES = 128 * 1024 * 1024


class FramePump(QThread):
    """Serves frames off the UI thread, newest request first.

    Only the most recent pending request survives, so dragging the timeline
    never queues up work the user has already scrolled past.
    """

    opened = Signal(object)
    ready = Signal(int, QImage)
    failed = Signal(str)

    def __init__(self, path: Path, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self._path = path
        self._mutex = QMutex()
        self._wake = QWaitCondition()
        self._target: int | None = None
        self._stopping = False
        self._cache: OrderedDict[int, QImage] = OrderedDict()
        self._cache_bytes = 0

    def request(self, index: int) -> None:
        with QMutexLocker(self._mutex):
            self._target = index
            self._wake.wakeAll()

    def clear_cache(self) -> int:
        with QMutexLocker(self._mutex):
            freed = self._cache_bytes
            self._cache.clear()
            self._cache_bytes = 0
            return freed

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._stopping = True
            self._wake.wakeAll()

    def run(self) -> None:
        try:
            info = index_video(self._path)
            decoder = FrameDecoder(info)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            with QMutexLocker(self._mutex):
                stopping = self._stopping
            if not stopping:
                self.failed.emit(str(exc))
            return

        self.opened.emit(info)
        try:
            self._serve(decoder)
        finally:
            decoder.close()
            self._cache.clear()
            self._cache_bytes = 0

    def _serve(self, decoder: FrameDecoder) -> None:
        while True:
            with QMutexLocker(self._mutex):
                while self._target is None and not self._stopping:
                    self._wake.wait(self._mutex)
                if self._stopping:
                    return
                index = self._target
                self._target = None

            image = None
            with QMutexLocker(self._mutex):
                cached = self._cache.get(index)
                if cached is not None:
                    self._cache.move_to_end(index)
                    image = cached
            if image is None:
                try:
                    array = decoder.frame(index)
                except Exception as exc:  # noqa: BLE001
                    self.failed.emit(str(exc))
                    return
                image = _to_qimage(array)
                self._remember(index, image)

            # Requests are served newest-first and in order, so whatever we just
            # produced is the freshest frame available. Always show it: a frame
            # one drag-step behind beats a frozen picture.
            with QMutexLocker(self._mutex):
                stopping = self._stopping
            if not stopping:
                self.ready.emit(index, image)

    def _remember(self, index: int, image: QImage) -> None:
        size = image.sizeInBytes()
        if size >= CACHE_BUDGET_BYTES:
            return
        with QMutexLocker(self._mutex):
            self._cache[index] = image
            self._cache_bytes += size
            while self._cache_bytes > CACHE_BUDGET_BYTES and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._cache_bytes -= evicted.sizeInBytes()


def _to_qimage(array) -> QImage:  # noqa: ANN001
    height, width, _ = array.shape
    view = QImage(
        array.data,
        width,
        height,
        array.strides[0],
        QImage.Format.Format_RGB888,
    )
    # RGB32 both detaches from the numpy buffer and blits without conversion.
    return view.convertToFormat(QImage.Format.Format_RGB32)
