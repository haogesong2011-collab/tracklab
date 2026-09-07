"""First-run spotlight tutorial (offscreen Qt, no real QSettings)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TRACKLAB_SKIP_UPDATE_CHECK", "1")

from app.tutorial import (  # noqa: E402
    DemoKind,
    TourStep,
    TutorialOverlay,
    maybe_start_tutorial,
    tutorial_auto_start_allowed,
    tutorial_seen,
)


def _settings(tmp: str) -> QSettings:
    return QSettings(str(Path(tmp) / "prefs.ini"), QSettings.Format.IniFormat)


class TutorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self) -> None:
        self._skip_flag = os.environ.pop("TRACKLAB_SKIP_TUTORIAL", None)

    def tearDown(self) -> None:
        if self._skip_flag is None:
            os.environ.pop("TRACKLAB_SKIP_TUTORIAL", None)
        else:
            os.environ["TRACKLAB_SKIP_TUTORIAL"] = self._skip_flag

    def _window(self):
        from app.main_window import MainWindow

        window = MainWindow()
        window.resize(1280, 800)
        window.show()
        self._app.processEvents()
        return window

    def test_auto_start_blocked_on_offscreen_and_flags(self) -> None:
        self.assertFalse(tutorial_auto_start_allowed())
        old = os.environ.get("TRACKLAB_SKIP_TUTORIAL")
        argv = list(sys.argv)
        try:
            os.environ["TRACKLAB_SKIP_TUTORIAL"] = "1"
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            self.assertFalse(tutorial_auto_start_allowed())
            os.environ.pop("TRACKLAB_SKIP_TUTORIAL", None)
            sys.argv = [argv[0], "--smoke"]
            self.assertFalse(tutorial_auto_start_allowed())
        finally:
            sys.argv = argv
            if old is None:
                os.environ.pop("TRACKLAB_SKIP_TUTORIAL", None)
            else:
                os.environ["TRACKLAB_SKIP_TUTORIAL"] = old

    def test_maybe_start_shows_first_step_on_drop_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            self.assertTrue(maybe_start_tutorial(window, settings=settings))
            overlay = window._tutorial_overlay
            self.assertIsNotNone(overlay)
            self.assertTrue(overlay.isVisible())
            self.assertEqual(overlay.current_index, 0)
            self.assertEqual(overlay._title.text(), "导入视频")
            target = overlay.current_target()
            self.assertIs(target, window._hint)
            hole = overlay.hole_rect
            self.assertFalse(hole.isEmpty())
            mapped = overlay.mapFromGlobal(window._hint.mapToGlobal(window._hint.rect().center()))
            self.assertTrue(hole.contains(mapped))
            self.assertEqual(overlay.current_demo(), DemoKind.DROP)
            overlay._demo_tick(1.0)
            self.assertGreater(overlay.demo_phase, 0.3)
            self.assertLess(overlay.demo_phase, 0.5)
            cursor = overlay.cursor_pos
            expanded = hole.adjusted(-120, -140, 120, 80)
            self.assertTrue(expanded.contains(cursor), (cursor, hole))
            window.close()

    def test_next_advances_then_finish_marks_seen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            self.assertTrue(maybe_start_tutorial(window, settings=settings))
            overlay = window._tutorial_overlay
            nxt = overlay.findChild(QPushButton, "tutorialNext")
            self.assertIsNotNone(nxt)
            nxt.click()
            self._app.processEvents()
            self.assertEqual(overlay.current_index, 1)
            self.assertEqual(overlay._title.text(), "打开视频")
            self.assertIs(overlay.current_target(), window._toolbar_buttons["open"])
            self.assertEqual(overlay.current_demo(), DemoKind.CLICK)
            self.assertLess(overlay.demo_phase, 0.05)
            overlay._index = len(overlay._steps) - 1
            overlay._show_step()
            self.assertEqual(nxt.text(), "完成")
            nxt.click()
            self._app.processEvents()
            self.assertTrue(tutorial_seen(settings))
            self.assertFalse(overlay.isVisible())
            self.assertFalse(overlay._demo_timer.isActive())
            self.assertFalse(maybe_start_tutorial(window, settings=settings))
            window.close()

    def test_skip_marks_seen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            self.assertTrue(maybe_start_tutorial(window, settings=settings))
            overlay = window._tutorial_overlay
            skip = overlay.findChild(QPushButton, "tutorialSkip")
            skip.click()
            self._app.processEvents()
            self.assertTrue(tutorial_seen(settings))
            self.assertFalse(overlay.isVisible())
            self.assertFalse(overlay._demo_timer.isActive())
            self.assertFalse(maybe_start_tutorial(window, settings=settings))
            window.close()

    def test_force_replays_after_seen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            self.assertTrue(maybe_start_tutorial(window, settings=settings))
            window._tutorial_overlay.findChild(QPushButton, "tutorialSkip").click()
            self._app.processEvents()
            self.assertTrue(tutorial_seen(settings))
            self.assertTrue(maybe_start_tutorial(window, force=True, settings=settings))
            overlay = window._tutorial_overlay
            self.assertIsNotNone(overlay)
            self.assertTrue(overlay.isVisible())
            self.assertEqual(overlay.current_index, 0)
            window.close()

    def test_skip_env_and_smoke_block_auto_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            old = os.environ.get("TRACKLAB_SKIP_TUTORIAL")
            argv = list(sys.argv)
            try:
                os.environ["TRACKLAB_SKIP_TUTORIAL"] = "1"
                self.assertFalse(maybe_start_tutorial(window, settings=settings))
                os.environ.pop("TRACKLAB_SKIP_TUTORIAL", None)
                sys.argv = [argv[0], "--smoke"]
                self.assertFalse(maybe_start_tutorial(window, settings=settings))
            finally:
                sys.argv = argv
                if old is None:
                    os.environ.pop("TRACKLAB_SKIP_TUTORIAL", None)
                else:
                    os.environ["TRACKLAB_SKIP_TUTORIAL"] = old
            window.close()

    def test_chart_step_reveals_hidden_dock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            window._workspace.set_chart_visible(False)
            self.assertFalse(window._workspace.chart_visible)
            self.assertTrue(maybe_start_tutorial(window, settings=settings))
            overlay = window._tutorial_overlay
            overlay._index = 4
            overlay._show_step()
            self.assertTrue(window._workspace.chart_visible)
            self.assertEqual(overlay._title.text(), "分图")
            self.assertEqual(overlay.current_demo(), DemoKind.SCRUB)
            self.assertLess(overlay.demo_phase, 0.05)
            window.close()

    def test_quick_start_uses_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            window._tutorial_overlay.findChild(QPushButton, "tutorialSkip").click()
            self._app.processEvents()
            window._show_quick_start()
            self._app.processEvents()
            overlay = window._tutorial_overlay
            self.assertIsNotNone(overlay)
            self.assertTrue(overlay.isVisible())
            window.close()

    def test_toolbar_object_names(self) -> None:
        window = self._window()
        self.assertEqual(window._toolbar_buttons["open"].objectName(), "toolOpen")
        self.assertEqual(window._toolbar_buttons["track"].objectName(), "toolTrack")
        self.assertEqual(window._toolbar_buttons["ruler"].objectName(), "toolRuler")
        self.assertEqual(window._toolbar_buttons["ai"].objectName(), "toolAi")
        self.assertEqual(window._transport.objectName(), "transportBar")
        window.close()

    def test_escape_skips(self) -> None:
        from PySide6.QtGui import QKeyEvent

        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            event = QKeyEvent(
                QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
            )
            overlay.keyPressEvent(event)
            self._app.processEvents()
            self.assertTrue(tutorial_seen(settings))
            self.assertFalse(overlay.isVisible())
            self.assertFalse(overlay._demo_timer.isActive())
            window.close()

    def test_demo_tick_click_moves_toward_hole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay.advance()
            self.assertEqual(overlay.current_demo(), DemoKind.CLICK)
            overlay._demo_tick(1.0)
            hole = overlay.hole_rect
            self.assertTrue(
                hole.adjusted(-40, -40, 40, 40).contains(overlay.cursor_pos),
                overlay.cursor_pos,
            )
            window.close()

    def test_notify_video_skips_open_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay.notify_host_action("click")
            self.assertEqual(overlay.current_index, 0)
            overlay.notify_host_action("video")
            self.assertEqual(overlay.current_index, 2)
            self.assertEqual(overlay._title.text(), "跟踪目标")
            overlay.notify_host_action("video")
            self.assertEqual(overlay.current_index, 2)
            window.close()

    def test_click_through_hole_reaches_target(self) -> None:
        host = QWidget()
        host.resize(640, 400)
        btn = QPushButton("hit", host)
        btn.setGeometry(80, 80, 120, 36)
        clicked: list[int] = []
        btn.clicked.connect(lambda: clicked.append(1))
        host.show()
        self._app.processEvents()
        overlay = TutorialOverlay(
            host,
            [TourStep("点这里", "请点高亮按钮。", lambda: btn)],
        )
        overlay.begin()
        self._app.processEvents()
        hole = overlay.hole_rect
        self.assertFalse(hole.isEmpty())
        center = hole.center()
        global_pos = overlay.mapToGlobal(center)

        def send(typ, button, buttons) -> None:
            event = QMouseEvent(
                typ,
                QPointF(center),
                global_pos,
                button,
                buttons,
                Qt.KeyboardModifier.NoModifier,
            )
            QApplication.sendEvent(overlay, event)
            self._app.processEvents()

        send(
            QEvent.Type.MouseButtonPress,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
        send(
            QEvent.Type.MouseButtonRelease,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
        )
        self.assertEqual(clicked, [1])
        self.assertTrue(overlay._advance_timer.isActive())
        overlay._advance_from_action()
        self._app.processEvents()
        self.assertFalse(overlay.isVisible())
        self.assertFalse(overlay._advance_timer.isActive())
        overlay.discard()
        host.close()

    def test_skip_stops_advance_timer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay._schedule_advance()
            self.assertTrue(overlay._advance_timer.isActive())
            skip = overlay.findChild(QPushButton, "tutorialSkip")
            skip.click()
            self._app.processEvents()
            self.assertFalse(overlay._advance_timer.isActive())
            window.close()

    def test_box_demo_draws_on_real_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay._index = 2
            overlay._show_step()
            self._app.processEvents()
            self.assertIs(window._stack.currentWidget(), window._video)
            self.assertIsNotNone(window._video._image)
            overlay._phase = 0.42
            overlay._cursor = overlay._cursor_at(0.42)
            overlay._drive_demo()
            overlay._phase = 0.72
            overlay._cursor = overlay._cursor_at(0.72)
            overlay._drive_demo()
            self._app.processEvents()
            self.assertIsNotNone(window._video._box)
            overlay._complete()
            self._app.processEvents()
            self.assertIs(window._stack.currentWidget(), window._hint)
            self.assertIsNone(window._video._image)
            window.close()

    def test_line_demo_draws_ruler_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay._index = 3
            overlay._show_step()
            self._app.processEvents()
            overlay._phase = 0.42
            overlay._cursor = overlay._cursor_at(0.42)
            overlay._drive_demo()
            overlay._phase = 0.72
            overlay._cursor = overlay._cursor_at(0.72)
            overlay._drive_demo()
            self._app.processEvents()
            self.assertIsNotNone(window._video._draft)
            overlay.discard()
            window.close()

    def test_click_demo_presses_real_button(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay.advance()
            overlay._phase = 0.55
            overlay._cursor = overlay._cursor_at(0.55)
            overlay._drive_demo()
            self.assertTrue(window._toolbar_buttons["open"].isDown())
            overlay.discard()
            window.close()


if __name__ == "__main__":
    unittest.main()
