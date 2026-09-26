"""Shared chart scale, table copy, and the closed assistant."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from ai.contracts import TrackPoint, TrackResult  # noqa: E402
from ai.desktop import export_track_csv  # noqa: E402
from ai.kinematics import KinematicSample  # noqa: E402
from app.data_views import TrackChartPanel  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from app.track_panels import TrackDataPanel  # noqa: E402
from engine.video_index import VideoInfo  # noqa: E402


def _sample(frame: int, x: float, y: float) -> KinematicSample:
    return KinematicSample(
        frame=frame,
        time_s=frame / 30.0,
        x=x,
        y=y,
        vx=0.0,
        vy=1.0,
        speed=1.0,
        visible=True,
        confidence=1.0,
        manual=False,
        ax=0.0,
        ay=9.81,
        accel_unit="px/s²",
    )


class ChartScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_same_unit_charts_share_the_larger_span(self) -> None:
        panel = TrackChartPanel()
        samples = [_sample(i, 0.02 * ((-1) ** i), i * 2.0) for i in range(8)]
        panel.set_samples(samples)
        left, right = panel._charts
        self.assertEqual(left.quantity_key(), "x")
        self.assertEqual(right.quantity_key(), "y")
        left_span = left._fitted_v[1] - left._fitted_v[0]
        right_span = right._fitted_v[1] - right._fitted_v[0]
        self.assertAlmostEqual(left_span, right_span, places=6)
        self.assertAlmostEqual(
            left._axis_value.max() - left._axis_value.min(),
            right._axis_value.max() - right._axis_value.min(),
            places=4,
        )
        self.assertGreater(left_span, left.data_value_span() + 1.0)
        left.set_manual_range((1.0, 4.0), None)
        self.assertEqual(left._fitted_v, (1.0, 4.0))
        self.assertAlmostEqual(left._axis_value.max() - left._axis_value.min(), 3.0, places=4)
        panel._same_scale.setChecked(False)
        self.assertGreater(
            right._fitted_v[1] - right._fitted_v[0],
            left._fitted_v[1] - left._fitted_v[0],
        )
        panel.close()


class TableCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_copy_is_tsv_with_header(self) -> None:
        panel = TrackDataPanel()
        panel.set_samples([_sample(0, 1.0, 2.0), _sample(1, 3.0, 4.0), _sample(2, 5.0, 6.0)])
        panel._table.selectRow(0)
        panel._table.selectionModel().select(
            panel._table.model().index(2, 0),
            panel._table.selectionModel().SelectionFlag.Select
            | panel._table.selectionModel().SelectionFlag.Rows,
        )
        panel.copy_selection()
        text = QGuiApplication.clipboard().text()
        html = QGuiApplication.clipboard().mimeData().html()
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        header = lines[0].split("\t")
        self.assertEqual(header[0], "帧")
        self.assertEqual(len(header), panel._table.columnCount())
        self.assertEqual(len(lines[1].split("\t")), len(header))
        self.assertTrue(lines[1].startswith("1\t"))
        self.assertTrue(lines[2].startswith("3\t"))
        self.assertIn("<table>", html)
        self.assertIn("<th>", html)
        panel.close()

    def test_csv_has_bom_and_accel_columns(self) -> None:
        info = VideoInfo(
            path=Path("clip.mp4"),
            width=64,
            height=64,
            pts=(0, 1, 2, 3),
            time_base=0.001,
            pts_ms=(0, 33, 66, 99),
        )
        result = TrackResult(
            clip_id="c",
            points=[TrackPoint(frame=i, x=float(i), y=float(i * i)) for i in range(4)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "track.csv"
            export_track_csv(path, result, info)
            raw = path.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        header = raw.decode("utf-8-sig").splitlines()[0].split(",")
        self.assertIn("ax_px_s2", header)
        self.assertIn("ay_px_s2", header)


class AssistantClosedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_opening_assistant_only_shows_a_notice(self) -> None:
        window = MainWindow()
        with patch.object(QMessageBox, "information", return_value=QMessageBox.StandardButton.Ok) as info:
            window._set_assistant_visible(True)
            window._ai_panel_action.setChecked(True)
        self.assertFalse(window._assistant_window.isVisible())
        self.assertFalse(window._ai_panel_action.isChecked())
        self.assertIn("暂未开放", info.call_args[0][2])
        window.close()
