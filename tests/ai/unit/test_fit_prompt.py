"""Guided review of predicted (trajectory-fit) points on the video."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.fit_prompt import (  # noqa: E402
    STATE_DONE,
    STATE_HIDDEN,
    STATE_MANUAL,
    STATE_PREDICT,
    FitPromptCard,
    done_text,
    manual_text,
    predict_text,
)
from app.gestures import pointer_pans  # noqa: E402
from app.widgets import OverlayTrack, VideoView  # noqa: E402


class CardTextTests(unittest.TestCase):
    def test_prediction_is_short_and_offers_two_choices(self) -> None:
        text = predict_text(109, 1.8, 12, "interpolate", 3)
        self.assertIn("第 110 帧", text.title)
        self.assertIn("预测", text.title)
        self.assertIn("±1.8 px", text.body)
        self.assertLessEqual(len(text.body), 20)
        self.assertIn("保留", text.primary)
        self.assertIn("F", text.primary)
        self.assertIn("手动打点", text.secondary)
        self.assertIn("剩 2", text.footer)

    def test_extrapolation_warns(self) -> None:
        self.assertIn("外推", predict_text(5, 4.0, 5, "extrapolate", 1).body)

    def test_manual_explains_shift_click(self) -> None:
        text = manual_text(41)
        self.assertIn("Shift", text.body)
        self.assertIn("左键", text.body)
        self.assertEqual(text.primary, "")
        self.assertIn("Esc", text.secondary)

    def test_done_offers_the_next_point_only_when_there_is_one(self) -> None:
        self.assertIn("下一个", done_text("已保留", 2).primary)
        self.assertEqual(done_text("已保留", 0).primary, "")


class CardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_buttons_follow_the_state(self) -> None:
        card = FitPromptCard()
        events: list[str] = []
        card.keep_requested.connect(lambda: events.append("keep"))
        card.manual_requested.connect(lambda: events.append("manual"))
        card.cancel_requested.connect(lambda: events.append("cancel"))
        card.next_requested.connect(lambda: events.append("next"))
        self.assertEqual(card.state, STATE_HIDDEN)
        card.show_prediction(10, 2.0, 8, "interpolate", 2)
        self.assertEqual(card.state, STATE_PREDICT)
        card._primary.click()
        card._secondary.click()
        card.show_manual(10)
        self.assertEqual(card.state, STATE_MANUAL)
        card._secondary.click()
        card.show_done(10, "已手动打点", 1)
        self.assertEqual(card.state, STATE_DONE)
        card._primary.click()
        self.assertEqual(events, ["keep", "manual", "cancel", "next"])
        card.dismiss()
        self.assertEqual(card.state, STATE_HIDDEN)
        self.assertEqual(card.focusPolicy(), Qt.FocusPolicy.NoFocus)


class VideoMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_shift_left_is_a_point_not_a_pan(self) -> None:
        self.assertFalse(
            pointer_pans(Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, surface="video")
        )
        self.assertTrue(
            pointer_pans(Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, surface="video")
        )

    def test_pending_point_breathes_and_manual_pick_toggles(self) -> None:
        view = VideoView()
        overlay = OverlayTrack(points=[(3, 10.0, 10.0, True)], color="#ff0000", active=True, suggestions={3: (12.0, 11.0, 2.0)})
        view.set_overlays([overlay], index=3)
        self.assertEqual(view.current_suggestion(), (12.0, 11.0, 2.0))
        levels = set()
        for _ in range(15):
            view._advance_pulse()
            levels.add(round(view.pulse_level(), 2))
        self.assertGreater(max(levels) - min(levels), 0.5)
        view.set_manual_pick(True)
        self.assertTrue(view.manual_pick_active())
        cancelled: list[bool] = []
        view.manual_pick_cancelled.connect(lambda: cancelled.append(True))

        class _Key:
            def key(self):
                return Qt.Key.Key_Escape

            def accept(self):
                pass

        view.keyPressEvent(_Key())
        self.assertEqual(cancelled, [True])
        view.set_manual_pick(False)
        view.set_track_index(4)
        self.assertIsNone(view.current_suggestion())
        _ = QPointF  # imported for parity with the other widget tests


if __name__ == "__main__":
    unittest.main()
