"""Photo Booth frames are padded per row; the picture must still show."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.frame_pump import _to_qimage  # noqa: E402
from app.widgets import clip_index, object_follow_box  # noqa: E402


def _padded_rgb(height: int, width: int, stride: int, fill: int) -> np.ndarray:
    """RGB view whose row stride is wider than width * 3, like a Photo Booth frame."""
    raw = np.full((height, stride), 0xAB, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            raw[y, x * 3 : x * 3 + 3] = (fill, y, x % 256)
    return np.lib.stride_tricks.as_strided(
        raw,
        shape=(height, width, 3),
        strides=(stride, 3, 1),
    )


class FrameImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_padded_row_becomes_a_real_picture(self) -> None:
        # 1620 * 3 = 4860, Photo Booth's decoder hands back stride 4896.
        frame = _padded_rgb(4, 1620, 4896, fill=30)
        self.assertFalse(frame.flags.c_contiguous)

        image = _to_qimage(frame)

        self.assertFalse(image.isNull())
        self.assertEqual(image.width(), 1620)
        self.assertEqual(image.height(), 4)
        self.assertEqual(image.format(), QImage.Format.Format_RGB32)
        corner = image.pixelColor(0, 0)
        self.assertEqual((corner.red(), corner.green(), corner.blue()), (30, 0, 0))
        far = image.pixelColor(1619, 3)
        self.assertEqual((far.red(), far.green(), far.blue()), (30, 3, 1619 % 256))

    def test_analysis_range_holds_the_playhead(self) -> None:
        self.assertEqual(clip_index(0, 10, 40), 10)
        self.assertEqual(clip_index(80, 10, 40), 40)
        self.assertEqual(clip_index(20, 10, 40), 20)

    def test_follow_box_moves_with_the_point(self) -> None:
        box = object_follow_box([], (100.0, 80.0), (0.0, 0.0, 40.0, 20.0))
        self.assertEqual(box, (80.0, 70.0, 120.0, 90.0))
        bounds = object_follow_box([(5.0, 6.0), (15.0, 26.0)], (100.0, 80.0), None)
        self.assertEqual(bounds, (5.0, 6.0, 15.0, 26.0))
