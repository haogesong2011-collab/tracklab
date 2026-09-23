"""Pan and zoom gestures for the video view and the function charts."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ai.kinematics import KinematicSample  # noqa: E402
from app.data_views import TrackChartView  # noqa: E402
from app.gestures import classify_scroll, pointer_pans, shift_view_center  # noqa: E402
from app.widgets import MODE_RULER, MODE_TRACK, VideoView  # noqa: E402


def _sample(frame: int, time_s: float, x: float) -> KinematicSample:
    return KinematicSample(
        frame=frame,
        time_s=time_s,
        x=x,
        y=0.0,
        vx=0.0,
        vy=0.0,
        speed=0.0,
        visible=True,
        confidence=1.0,
        manual=False,
    )


class GestureMathTests(unittest.TestCase):
    def test_trackpad_drag_pans_with_the_fingers(self) -> None:
        # Positive pixel y is fingers up. Natural scrolling moves the picture up.
        gesture = classify_scroll(
            pixel_dx=12,
            pixel_dy=8,
            angle_dx=0,
            angle_dy=80,
            touchpad=True,
            scrolling=True,
            inverted=True,
        )
        self.assertEqual(gesture.kind, "pan")
        self.assertAlmostEqual(gesture.dx, 12.0)
        self.assertAlmostEqual(gesture.dy, -8.0)

    def test_classic_trackpad_scroll_moves_the_picture_opposite_the_fingers(self) -> None:
        gesture = classify_scroll(
            pixel_dx=12,
            pixel_dy=8,
            angle_dx=0,
            angle_dy=80,
            touchpad=True,
            scrolling=True,
            inverted=False,
        )
        self.assertEqual(gesture.kind, "pan")
        self.assertAlmostEqual(gesture.dx, -12.0)
        self.assertAlmostEqual(gesture.dy, 8.0)

    def test_mouse_wheel_zooms_instead_of_panning(self) -> None:
        gesture = classify_scroll(
            pixel_dx=0,
            pixel_dy=0,
            angle_dx=0,
            angle_dy=120,
            touchpad=False,
            scrolling=False,
            inverted=False,
        )
        self.assertEqual(gesture.kind, "zoom")
        self.assertAlmostEqual(gesture.steps, 1.0)

    def test_chart_drag_right_reveals_earlier_time(self) -> None:
        center_t, center_v = shift_view_center(5.0, 10.0, 50.0, 0.0, 100.0, 80.0, 0.0, 10.0, 0.0, 20.0)
        self.assertAlmostEqual(center_t, 0.0)
        self.assertAlmostEqual(center_v, 10.0)

    def test_chart_drag_down_reveals_higher_values(self) -> None:
        _center_t, center_v = shift_view_center(5.0, 10.0, 0.0, 40.0, 100.0, 80.0, 0.0, 10.0, 0.0, 20.0)
        self.assertAlmostEqual(center_v, 20.0)

    def test_pointer_pan_does_not_steal_track_box_or_chart_scrub(self) -> None:
        control = Qt.KeyboardModifier.ControlModifier
        self.assertTrue(
            pointer_pans(Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, surface="video")
        )
        self.assertFalse(pointer_pans(Qt.MouseButton.LeftButton, control, surface="video"))
        self.assertFalse(
            pointer_pans(Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, surface="chart")
        )
        self.assertTrue(pointer_pans(Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, surface="chart"))


class GestureWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_plain_left_drag_pans_the_video(self) -> None:
        view = VideoView()
        image = QImage(320, 180, QImage.Format.Format_RGB32)
        image.fill(0)
        view.set_frame(image)
        view.set_interaction_mode(MODE_TRACK)
        view.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(40, 50),
                QPointF(40, 50),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        view.mouseMoveEvent(
            QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(70, 80),
                QPointF(70, 80),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertAlmostEqual(view._pan.x(), 30.0)
        self.assertAlmostEqual(view._pan.y(), 30.0)

    def test_left_drag_does_not_pan_while_drawing_a_ruler(self) -> None:
        view = VideoView()
        image = QImage(320, 180, QImage.Format.Format_RGB32)
        image.fill(0)
        view.set_frame(image)
        view.set_interaction_mode(MODE_RULER)
        view.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(40, 50),
                QPointF(40, 50),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertIsNone(view._panning)

    def test_mouse_wheel_event_zooms_the_video(self) -> None:
        view = VideoView()
        image = QImage(320, 180, QImage.Format.Format_RGB32)
        image.fill(0)
        view.set_frame(image)
        view.wheelEvent(
            QWheelEvent(
                QPointF(20, 20),
                QPointF(20, 20),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        )
        self.assertGreater(view.zoom(), 1.0)

    def test_chart_right_drag_pans_and_left_drag_still_scrubs(self) -> None:
        chart = TrackChartView([("x", "x", "#8ec8ff")], "x (m)")
        chart.resize(480, 280)
        chart.show()
        chart.set_samples([_sample(0, 0.0, 0.0), _sample(1, 2.0, 4.0)])
        QApplication.processEvents()
        before = chart._axis_time.min()
        viewport = chart.viewport()
        QApplication.sendEvent(
            viewport,
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(200, 100),
                QPointF(200, 100),
                Qt.MouseButton.RightButton,
                Qt.MouseButton.RightButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        QApplication.sendEvent(
            viewport,
            QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(260, 100),
                QPointF(260, 100),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.RightButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertLess(chart._axis_time.min(), before)
        QApplication.sendEvent(
            viewport,
            QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                QPointF(260, 100),
                QPointF(260, 100),
                Qt.MouseButton.RightButton,
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertIsNone(chart._panning)
        frames: list[int] = []
        chart.frame_activated.connect(frames.append)
        QApplication.sendEvent(
            viewport,
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(180, 120),
                QPointF(180, 120),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertTrue(frames)
        self.assertIsNone(chart._panning)
