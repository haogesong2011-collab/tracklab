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
from app.widgets import MODE_TRACK  # noqa: E402

STEP_TITLES = [
    "导入视频",
    "打开视频",
    "新建轨迹",
    "框选目标",
    "加提示点",
    "按 T 跟踪",
    "快速 / 精准",
    "跟丢了怎么修",
    "跟踪范围与背景补偿",
    "标定尺",
    "坐标系",
    "分图",
    "数据表",
    "当前读数",
    "AI 助手",
    "播放与保存",
]


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
            self.assertEqual(overlay.stage.demo_kind, DemoKind.DROP)
            overlay._demo_tick(1.0)
            self.assertGreater(overlay.demo_phase, 0.3)
            self.assertLess(overlay.demo_phase, 0.5)
            self.assertGreater(overlay.stage.demo_phase, 0.3)
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
            self.assertEqual(overlay.current_demo(), DemoKind.OPEN)
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
            overlay._index = overlay.step_index("分图")
            overlay._show_step()
            self.assertTrue(window._workspace.chart_visible)
            self.assertEqual(overlay._title.text(), "分图")
            self.assertEqual(overlay.current_demo(), DemoKind.CHART)
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
        self.assertEqual(window._toolbar_buttons["axis"].objectName(), "toolAxis")
        self.assertEqual(window._toolbar_buttons["save"].objectName(), "toolSave")
        self.assertEqual(window._toolbar_buttons["ai"].objectName(), "toolAi")
        self.assertEqual(window._transport.objectName(), "transportBar")
        self.assertEqual(window._view_bar.objectName(), "viewToolbar")
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

    def test_demo_tick_advances_stage_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay.advance()
            self.assertEqual(overlay.current_demo(), DemoKind.OPEN)
            overlay._demo_tick(1.0)
            self.assertGreater(overlay.stage.demo_phase, 0.3)
            self.assertEqual(overlay.stage.demo_kind, DemoKind.OPEN)
            window.close()

    def test_opening_video_does_not_skip_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            self.assertEqual(overlay.current_index, 0)
            self.assertFalse(hasattr(overlay, "notify_host_action"))
            overlay._demo_tick(0.5)
            self.assertEqual(overlay.current_index, 0)
            self.assertEqual(overlay._title.text(), "导入视频")
            window.close()

    def test_click_through_hole_reaches_target_without_advancing(self) -> None:
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
            [TourStep("点这里", "请点高亮按钮。", lambda: btn), TourStep("下一步", "x", lambda: btn)],
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
        self.assertTrue(overlay.isVisible())
        self.assertEqual(overlay.current_index, 0)
        self.assertFalse(hasattr(overlay, "_advance_timer"))
        overlay.discard()
        host.close()

    def test_box_demo_stays_on_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay._index = overlay.step_index("框选目标")
            overlay._show_step()
            self._app.processEvents()
            self.assertEqual(overlay.current_demo(), DemoKind.BOX)
            self.assertIsNone(window._video._image)
            self.assertEqual(window._video.interaction_mode(), MODE_TRACK)
            overlay._demo_tick(1.2)
            self.assertGreater(overlay.stage.demo_phase, 0.3)
            self.assertIsNone(window._video._image)
            self.assertIsNone(window._video._box)
            self.assertIs(window._stack.currentWidget(), window._hint)
            overlay._complete()
            window.close()

    def test_line_demo_does_not_draw_ruler_on_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay._index = overlay.step_index("标定尺")
            overlay._show_step()
            self._app.processEvents()
            overlay._demo_tick(1.2)
            self.assertEqual(overlay.current_demo(), DemoKind.RULER)
            self.assertIsNone(window._video._draft)
            self.assertEqual(window._video.interaction_mode(), MODE_TRACK)
            overlay.discard()
            window.close()

    def test_open_demo_does_not_press_real_button(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            overlay.advance()
            overlay._demo_tick(1.6)
            self.assertFalse(window._toolbar_buttons["open"].isDown())
            overlay.discard()
            window.close()

    def test_detailed_steps_cover_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            window = self._window()
            maybe_start_tutorial(window, settings=settings)
            overlay = window._tutorial_overlay
            titles = [step.title for step in overlay._steps]
            self.assertEqual(titles, STEP_TITLES)
            box_body = overlay._steps[overlay.step_index("框选目标")].body
            self.assertIn("Control", box_body)
            self.assertIn("不是搜索范围", box_body)
            self.assertNotIn("负点", box_body)
            seed_body = overlay._steps[overlay.step_index("加提示点")].body
            self.assertIn("Shift+Control", seed_body)
            self.assertIn("正点", seed_body)
            track_body = overlay._steps[overlay.step_index("按 T 跟踪")].body
            self.assertIn("按 T", track_body)
            mode_body = overlay._steps[overlay.step_index("快速 / 精准")].body
            self.assertIn("Tiny", mode_body)
            self.assertIn("Small", mode_body)
            ruler_body = overlay._steps[overlay.step_index("标定尺")].body
            self.assertIn("像素", ruler_body)
            self.assertIn("透视", ruler_body)
            self.assertGreater(len(track_body), 40)
            back = overlay.findChild(QPushButton, "tutorialBack")
            self.assertIsNotNone(back)
            self.assertFalse(back.isVisible())
            overlay.advance()
            self._app.processEvents()
            self.assertTrue(back.isVisible())
            self.assertEqual(overlay._title.text(), "打开视频")
            back.click()
            self._app.processEvents()
            self.assertEqual(overlay.current_index, 0)
            self.assertEqual(overlay._title.text(), "导入视频")
            self.assertFalse(back.isVisible())
            overlay._index = overlay.step_index("坐标系")
            overlay._show_step()
            self.assertIs(overlay.current_target(), window._toolbar_buttons["axis"])
            overlay._index = overlay.step_index("当前读数")
            overlay._show_step()
            self.assertIs(overlay.current_target(), window._view_bar)
            overlay._index = overlay.step_index("播放与保存")
            overlay._show_step()
            self.assertIs(overlay.current_target(), window._transport)
            nxt = overlay.findChild(QPushButton, "tutorialNext")
            self.assertEqual(nxt.text(), "完成")
            overlay.discard()
            window.close()

    def test_stage_repaints_every_demo_kind(self) -> None:
        stage_host = QWidget()
        overlay = TutorialOverlay(stage_host, [TourStep("x", "y", lambda: None, demo=DemoKind.DROP)])
        overlay.begin()
        kinds = [
            DemoKind.DROP,
            DemoKind.OPEN,
            DemoKind.NEW_TRACK,
            DemoKind.BOX,
            DemoKind.SEED_POINT,
            DemoKind.TRACK_RUN,
            DemoKind.FAST_PRECISE,
            DemoKind.FIX_POINT,
            DemoKind.RANGE,
            DemoKind.RULER,
            DemoKind.AXIS,
            DemoKind.CHART,
            DemoKind.TABLE,
            DemoKind.READOUT,
            DemoKind.ASSISTANT,
            DemoKind.PLAY,
            DemoKind.SAVE,
        ]
        for kind in kinds:
            overlay.stage.set_demo(kind, 0.6)
            overlay.stage.repaint()
            self.assertEqual(overlay.stage.demo_kind, kind)
            image = overlay.stage.grab()
            self.assertFalse(image.isNull())
            self.assertGreater(image.width(), 0)
        overlay.discard()
        stage_host.close()


if __name__ == "__main__":
    unittest.main()
