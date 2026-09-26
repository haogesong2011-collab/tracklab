"""Video pan: trackpad drags, mouse wheel zooms, the picture stays on screen."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt  # noqa: E402

from app.gestures import clamp_pan, classify_scroll, pointer_pans  # noqa: E402


class GestureTests(unittest.TestCase):
    def test_trackpad_pans_and_mouse_wheel_zooms(self) -> None:
        pan = classify_scroll(
            pixel_dx=12,
            pixel_dy=-8,
            angle_dx=0,
            angle_dy=0,
            touchpad=True,
            scrolling=True,
            inverted=True,
        )
        self.assertEqual(pan.kind, "pan")
        self.assertEqual(pan.dx, 12)
        self.assertEqual(pan.dy, -8)
        zoom = classify_scroll(
            pixel_dx=0,
            pixel_dy=0,
            angle_dx=0,
            angle_dy=120,
            touchpad=False,
            scrolling=False,
            inverted=False,
        )
        self.assertEqual(zoom.kind, "zoom")
        self.assertEqual(zoom.steps, 1)

    def test_plain_left_drag_pans_and_control_does_not(self) -> None:
        self.assertTrue(
            pointer_pans(Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, surface="video")
        )
        self.assertFalse(
            pointer_pans(
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ControlModifier,
                surface="video",
            )
        )

    def test_shift_left_places_a_point_instead_of_panning(self) -> None:
        self.assertFalse(
            pointer_pans(
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.ShiftModifier,
                surface="video",
            )
        )

    def test_pan_cannot_push_the_picture_fully_off_screen(self) -> None:
        x, y = clamp_pan(5000, -5000, 800, 600, 400, 300)
        self.assertLess(abs(x), 5000)
        self.assertLess(abs(y), 5000)
        self.assertGreater(400 + ((800 - 400) / 2 + x), 0)
