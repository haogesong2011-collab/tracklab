"""Desktop acceptance smoke test (requires display / offscreen Qt)."""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ai.contracts import PromptKind, TrackLayer, TrackPoint, TrackPrompt, TrackResult  # noqa: E402
from ai.desktop import (  # noqa: E402
    apply_manual_override,
    export_track_csv,
    layer_from_result,
    read_track_project,
    write_track_project,
)
from ai.schema import Point2D  # noqa: E402
from tests.ai.dataset import load_annotation, load_manifest, resolve_video  # noqa: E402
from tests.ai.generate_fixtures import build_fixtures  # noqa: E402


class DesktopAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        if not (ROOT / "datasets" / "manifest.json").exists():
            build_fixtures()
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def test_manual_override_updates_point(self) -> None:
        result = TrackResult(
            clip_id="demo",
            points=[
                TrackPoint(frame=0, x=1, y=1),
                TrackPoint(frame=1, x=2, y=2),
            ],
        )
        updated = apply_manual_override(result, 1, Point2D(9, 9))
        self.assertEqual(updated.points[1].x, 9)
        self.assertEqual(updated.points[1].y, 9)

    def test_manual_override_persists_in_project_file(self) -> None:
        result = TrackResult(
            clip_id="demo",
            points=[TrackPoint(frame=0, x=1, y=1), TrackPoint(frame=1, x=2, y=2)],
            model_name="color_blob_tracker",
        )
        updated = apply_manual_override(result, 1, Point2D(9.5, 11.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.json"
            video = Path("/tmp/synthetic.mp4")
            write_track_project(path, video, updated)
            doc = read_track_project(path)
        self.assertEqual(doc.video_path, video)
        loaded = doc.tracks[0].result
        assert loaded is not None
        self.assertEqual(loaded.points[1].x, 9.5)
        self.assertEqual(loaded.points[1].y, 11.0)
        self.assertTrue(loaded.points[1].manual)
        self.assertEqual(loaded.model_name, "color_blob_tracker")

    def test_ai_module_does_not_use_frame_pump(self) -> None:
        import ai.desktop as desktop
        import ai.models as models

        desktop_src = inspect.getsource(desktop)
        models_src = inspect.getsource(models)
        self.assertNotIn("from app.frame_pump", desktop_src)
        self.assertNotIn("from app import frame_pump", desktop_src)
        self.assertNotIn("app.frame_pump", models_src)
        import ai.sam2_tracker as sam2
        import ai.sam2_frames as frames

        self.assertNotIn("app.frame_pump", inspect.getsource(sam2))
        self.assertNotIn("import decord", inspect.getsource(frames))
        self.assertNotIn("app.frame_pump", inspect.getsource(frames))
        import ai.stabilize as stabilize
        from ai.desktop import ShakeWorker

        self.assertNotIn("app.frame_pump", inspect.getsource(stabilize))
        self.assertNotIn("frame_pump", inspect.getsource(ShakeWorker))
        self.assertNotIn("FramePump", inspect.getsource(ShakeWorker.run))
        import ai.assistant_worker as assistant_worker
        from ai.assistant_worker import AssistantWorker

        self.assertNotIn("app.frame_pump", inspect.getsource(assistant_worker))
        self.assertNotIn("FramePump", inspect.getsource(AssistantWorker.run))

    def test_track_frames_cover_decoder_indices(self) -> None:
        from ai.models import ColorBlobTracker, load_video

        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        result = ColorBlobTracker().track(
            info, (ann.track[0].center.x, ann.track[0].center.y)
        )
        self.assertEqual([p.frame for p in result.points], list(range(info.frame_count)))

    def test_cancel_path_on_fixture(self) -> None:
        from ai.desktop import check_cancel_latency

        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        video = resolve_video(entry)
        outcome = check_cancel_latency(
            video, (ann.track[0].center.x, ann.track[0].center.y)
        )
        self.assertTrue(outcome.passed, msg=outcome.details)

    def test_multitrack_project_and_csv_roundtrip(self) -> None:
        from tests.ai.dataset import load_annotation, load_manifest, resolve_video
        from ai.models import load_video

        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        info = load_video(resolve_video(entry))
        a = TrackResult(
            clip_id="a",
            points=[TrackPoint(frame=0, x=1, y=2, visible=True, confidence=0.9)],
            model_name="sam2.1_hiera_tiny",
        )
        b = TrackResult(
            clip_id="b",
            points=[TrackPoint(frame=1, x=3, y=4, visible=True, confidence=0.8)],
        )
        tracks = [
            layer_from_result(a, name="摆球", color="#f0c14b"),
            layer_from_result(b, name="滑块", color="#6cb6ff"),
        ]
        tracks[0].prompts = [
            TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=1, y=2)
        ]
        tracks[0].visible = False
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "lab.json"
            csv_path = Path(tmp) / "a.csv"
            write_track_project(
                project,
                info.path,
                tracks=tracks,
                active_track_id=tracks[1].track_id,
                model_name="sam2.1_hiera_tiny",
                model_license="Apache-2.0",
            )
            export_track_csv(csv_path, a, info)
            doc = read_track_project(project)
            text = csv_path.read_text(encoding="utf-8")
        self.assertEqual(len(doc.tracks), 2)
        self.assertEqual(doc.active_track_id, tracks[1].track_id)
        self.assertEqual(doc.tracks[0].name, "摆球")
        self.assertFalse(doc.tracks[0].visible)
        self.assertEqual(doc.tracks[0].prompts[0].kind, PromptKind.POSITIVE)
        self.assertEqual(doc.model_license, "Apache-2.0")
        self.assertIn("frame,time_s,x_px,y_px,vx_px_s,vy_px_s,v_px_s,visible,confidence", text)
        self.assertIn("0,", text)

    def test_workbench_list_and_table_linkage(self) -> None:
        from PySide6.QtWidgets import QApplication

        from app.track_panels import TrackDataPanel, TrackListPanel

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        layer = TrackLayer(
            track_id="t1",
            name="轨迹 1",
            result=TrackResult(
                clip_id="c",
                points=[
                    TrackPoint(frame=0, x=1, y=1),
                    TrackPoint(frame=4, x=2, y=2),
                ],
            ),
        )
        lst = TrackListPanel()
        table = TrackDataPanel()
        received: list[int] = []
        table.frame_activated.connect(received.append)
        lst.set_tracks([layer], "t1")
        table.set_layer(layer, None)
        self.assertEqual(lst._list.count(), 1)
        self.assertEqual(table._table.rowCount(), 2)
        table._on_cell(1, 0)
        self.assertEqual(received, [4])

    def test_legacy_v1_project_loads(self) -> None:
        result = TrackResult(
            clip_id="legacy",
            points=[TrackPoint(frame=0, x=4, y=5)],
            model_name="color_blob_tracker",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "tracklab.project.v1",
                        "video_path": "/tmp/legacy.mp4",
                        "track": result.to_dict(),
                    }
                ),
                encoding="utf-8",
            )
            doc = read_track_project(path)
        self.assertEqual(doc.schema, "tracklab.project.v1")
        self.assertEqual(len(doc.tracks), 1)
        assert doc.tracks[0].result is not None
        self.assertEqual(doc.tracks[0].result.points[0].x, 4)

    def test_workbench_main_window_smoke(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window._new_track()
        window._new_track()
        self.assertEqual(len(window._tracks), 2)
        window._select_track(window._tracks[1].track_id)
        self.assertEqual(window._active_id, window._tracks[1].track_id)
        window.show()
        self.assertIs(window._workspace.centralWidget(), window._stage)
        self.assertIs(window._workspace.chart_dock.widget(), window._chart_panel)
        self.assertIs(window._workspace.data_dock.widget(), window._data_panel)
        self.assertEqual(
            window._workspace.chart_dock.allowedAreas(),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.assertEqual(
            window._workspace.data_dock.allowedAreas(),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.assertFalse(window._track_window.isVisible())
        window._toggle_track_window()
        self.assertTrue(window._track_window.isVisible())
        window._toggle_track_window()
        self.assertTrue(window._track_window.isVisible())
        window._track_window.close()
        self.assertFalse(window._track_window.isVisible())
        self.assertEqual(len(window._tracks), 2)
        window._data_column.hide()
        window._table_action.setChecked(False)
        window._restore_layout()
        self.assertTrue(window._track_window.isVisible())
        self.assertFalse(window._data_column.isHidden())
        self.assertTrue(window._table_action.isChecked())
        window._video.set_zoom(2.0)
        self.assertAlmostEqual(window._video.zoom(), 2.0)
        window._reset_zoom()
        self.assertAlmostEqual(window._video.zoom(), 1.0)
        self.assertEqual(window._zoom_readout.text(), "100%")
        self.assertEqual(window._track_mode.value, "fast")
        self.assertFalse(hasattr(window, "_toolbar_mode_combo"))
        self.assertEqual(window._list_panel._mode_combo.currentData(), "fast")
        window._apply_track_mode("precise", persist=False)
        self.assertEqual(window._track_mode.value, "precise")
        self.assertTrue(window._precise_mode_action.isChecked())
        src = inspect.getsource(MainWindow._start_or_cancel_ai_track)
        self.assertIn("TrackMode.PRECISE", src)
        self.assertIn("create_tracker", src)
        self.assertNotIn("ColorBlobTracker", src)
        window._clear_cache()
        self.assertFalse(window._shake_action.isEnabled())
        self.assertTrue(window._shake_apply_action.isChecked())
        self.assertTrue(window._list_panel._shake_box.isChecked())
        window._list_panel._shake_box.setChecked(False)
        self.assertFalse(window._shake_apply_action.isChecked())
        self.assertFalse(window._shake_enabled)
        window._shake_apply_action.setChecked(True)
        self.assertTrue(window._list_panel._shake_box.isChecked())
        self.assertTrue(window._shake_enabled)
        flags = window._track_window.windowFlags()
        self.assertTrue(bool(flags & Qt.WindowType.Tool))
        self.assertTrue(bool(flags & Qt.WindowType.WindowStaysOnTopHint))
        self.assertEqual(set(window._view_bar._fields), {"t", "x", "y"})
        self.assertEqual(window._video_info.objectName(), "videoInfoLabel")
        self.assertEqual(window._video_info.text(), "")
        help_menu = None
        for action in window.menuBar().actions():
            if action.text().replace("&", "") == "帮助":
                help_menu = action.menu()
                break
        self.assertIsNotNone(help_menu)
        help_labels = [
            action.text().replace("&", "")
            for action in help_menu.actions()
            if action.text()
        ]
        self.assertIn("检查更新…", help_labels)
        self.assertIn("启动时自动检查更新", help_labels)
        self.assertIn("关于 TrackLab", help_labels)
        self.assertTrue(window._auto_update_action.isEnabled())
        window.close()

    def test_view_bar_chart_and_manual_edit(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window._new_track()
        layer = window._tracks[0]
        layer.result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=1, y=2),
                TrackPoint(frame=1, x=2, y=2),
                TrackPoint(frame=2, x=9, y=9, visible=False),
                TrackPoint(frame=3, x=5, y=6),
                TrackPoint(frame=4, x=6, y=6),
            ],
        )
        window._refresh_track_ui()
        self.assertEqual(window._data_panel._table.columnCount(), 12)
        self.assertEqual(window._data_panel._table.rowCount(), 5)
        self.assertEqual(window._data_panel._table.item(2, 2).text(), "")
        self.assertEqual(set(window._view_bar._fields), {"t", "x", "y"})
        self.assertEqual(window._view_bar._fields["x"].text(), "1.00")
        names = {
            series.name()
            for chart in window._chart_panel._charts
            for series in chart.chart().series()
            if series.name()
        }
        self.assertEqual(names, {"x", "y"})
        self.assertEqual(len(window._chart_panel._charts), 2)
        self.assertEqual(window._chart_panel._charts[0]._axis_value.titleText(), "x (px)")
        self.assertEqual(window._chart_panel._charts[0]._axis_time.titleText(), "t (s)")
        self.assertEqual(window._data_panel._table.horizontalHeaderItem(2).text(), "x (px)")
        window._chart_panel._selects[1].setCurrentIndex(2)
        self.assertEqual(window._chart_panel._charts[1]._axis_value.titleText(), "vₓ (px/s)")
        v_names = {
            series.name()
            for series in window._chart_panel._charts[1].chart().series()
            if series.name()
        }
        self.assertEqual(v_names, {"vₓ"})
        self.assertEqual(len(window._chart_panel._charts[0].chart().axes()), 2)
        x_segments = sum(
            1
            for series in window._chart_panel._charts[0]._series
            if series.pen().color() == QColor("#6cb6ff")
        )
        self.assertEqual(x_segments, 2)
        chart = window._chart_panel._charts[0]
        span0 = chart._axis_time.max() - chart._axis_time.min()
        chart.zoom_at(2.0, (chart._axis_time.min() + chart._axis_time.max()) / 2, 0.0)
        self.assertAlmostEqual(chart._zoom, 2.0, places=3)
        self.assertLess(chart._axis_time.max() - chart._axis_time.min(), span0)
        window._chart_panel.reset_zoom()
        self.assertAlmostEqual(chart._zoom, 1.0, places=3)
        self.assertAlmostEqual(chart._axis_time.max() - chart._axis_time.min(), span0, places=4)
        window._apply_track_overlay(3)
        self.assertEqual(window._video._track_index, 3)
        self.assertEqual(window._video._overlays[0].points[3][0], 3)
        jumped: list[int] = []
        window._chart_panel.frame_activated.connect(jumped.append)
        window._data_panel.frame_activated.connect(jumped.append)
        window._chart_panel.frame_activated.emit(2)
        window._data_panel._on_cell(0, 0)
        self.assertEqual(jumped, [2, 0])
        window._index = 0
        window._on_position_edited(9.5, 8.25)
        self.assertEqual(layer.result.points[0].x, 9.5)
        self.assertEqual(layer.result.points[0].y, 8.25)
        self.assertTrue(layer.result.points[0].manual)
        self.assertEqual(window._view_bar._fields["x"].text(), "9.50")
        window._index = 1
        window._step_track_point(1)
        self.assertEqual(window._index, 3)
        window._view_bar._visible.click()
        self.assertFalse(layer.visible)
        item = window._list_panel._list.item(0)
        self.assertEqual(item.checkState(), Qt.CheckState.Unchecked)
        window._new_track()
        window._select_track(window._tracks[0].track_id)
        self.assertEqual(window._view_bar._track.currentData(), window._tracks[0].track_id)
        received: list[str] = []
        window._view_bar.track_selected.connect(received.append)
        window._view_bar._updating = False
        window._view_bar._on_track_changed(1)
        self.assertEqual(received, [window._tracks[1].track_id])
        window.close()

    def test_undo_and_redo_track_edits(self) -> None:
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window._new_track()
        track_id = window._tracks[0].track_id
        window._tracks[0].result = TrackResult(
            clip_id="c",
            points=[TrackPoint(frame=0, x=1, y=2), TrackPoint(frame=1, x=2, y=3)],
        )
        window._refresh_track_ui()
        window._on_position_edited(9.5, 8.25)
        self.assertEqual(window._tracks[0].result.points[0].x, 9.5)
        self.assertTrue(window._undo_action.isEnabled())
        window._undo()
        self.assertEqual(window._tracks[0].track_id, track_id)
        self.assertEqual(window._tracks[0].result.points[0].x, 1)
        self.assertEqual(window._tracks[0].result.points[0].y, 2)
        self.assertTrue(window._redo_action.isEnabled())
        window._redo()
        self.assertEqual(window._tracks[0].result.points[0].x, 9.5)
        window._undo()
        window._delete_track()
        self.assertEqual(window._tracks, [])
        window._undo()
        self.assertEqual(len(window._tracks), 1)
        self.assertEqual(window._tracks[0].result.points[0].x, 1)
        window._undo()
        self.assertEqual(window._tracks, [])
        window.close()

    def test_shake_compensation_analysis_not_overlay(self) -> None:
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow
        from ai.stabilize import ShakeCompensation

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window._new_track()
        layer = window._tracks[0]
        layer.result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=10, y=20),
                TrackPoint(frame=1, x=16, y=27),
            ],
        )
        window._shake = ShakeCompensation(
            dx=(0.0, 6.0),
            dy=(0.0, 7.0),
            anchors=(((5.0, 5.0),), ((11.0, 12.0),)),
        )
        window._shake_enabled = True
        window._refresh_track_ui()
        self.assertEqual(window._data_panel._table.item(1, 2).text(), "10.00")
        self.assertEqual(window._data_panel._table.item(1, 3).text(), "-20.00")
        window._apply_track_overlay(1)
        self.assertEqual(window._video._overlays[0].points[1][1], 16)
        self.assertEqual(window._video._overlays[0].points[1][2], 27)
        self.assertEqual(window._video._anchors, [(11.0, 12.0)])
        window._index = 1
        window._sync_view_bar()
        self.assertEqual(window._view_bar._fields["x"].text(), "10.00")
        window._on_position_edited(11.0, 21.0)
        self.assertAlmostEqual(layer.result.points[1].x, 17.0)
        self.assertAlmostEqual(layer.result.points[1].y, 28.0)
        self.assertTrue(layer.result.points[1].manual)
        self.assertEqual(window._view_bar._fields["x"].text(), "11.00")
        window._on_shake_apply_toggled(False)
        self.assertEqual(window._view_bar._fields["x"].text(), "17.00")
        self.assertEqual(window._data_panel._table.item(1, 2).text(), "17.00")
        self.assertFalse(window._list_panel._shake_box.isChecked())
        window._on_anchor_toggled(False)
        self.assertEqual(window._video._anchors, [])
        window.close()

    def test_default_worker_uses_tracker_autotracker_not_color_blob(self) -> None:
        from ai.contracts import TrackMode
        from ai.desktop import TrackWorker, create_tracker
        from ai.autotracker import TrackerAutoTracker
        from ai.sam2_tracker import Sam2Tracker

        source = inspect.getsource(TrackWorker.run)
        self.assertIn("create_tracker", source)
        self.assertIn("TrackMode.FAST", source)
        self.assertNotIn("ColorBlobTracker()", source)
        self.assertIsInstance(create_tracker(TrackMode.FAST), TrackerAutoTracker)
        self.assertIsInstance(create_tracker(TrackMode.PRECISE), Sam2Tracker)

    def test_loop_marker_drag_emits_preview_frame(self) -> None:
        from PySide6.QtCore import QPointF

        from app.main_window import MainWindow
        from app.widgets import TimelineSlider

        class MoveEvent:
            def __init__(self, x: float) -> None:
                self._position = QPointF(x, 0)

            def position(self) -> QPointF:
                return self._position

        slider = TimelineSlider()
        slider.resize(500, 36)
        slider.setRange(0, 100)
        slider.set_loop_range(20, 80)
        moved: list[int] = []
        slider.loop_marker_moved.connect(moved.append)
        slider._dragging = "start"
        slider.mouseMoveEvent(MoveEvent(slider._x_for(40)))
        self.assertEqual(slider.loop_range(), (40, 80))
        self.assertEqual(moved[-1], 40)
        source = inspect.getsource(MainWindow._on_loop_marker_moved)
        self.assertIn("_show_frame(frame)", source)

    def test_project_v3_calibration_roundtrip_and_legacy_v2(self) -> None:
        from ai.calibration import CalibrationMode, uniform_state
        from ai.schema import Point2D

        result = TrackResult(
            clip_id="demo",
            points=[TrackPoint(frame=0, x=10, y=20)],
        )
        cal = uniform_state(
            Point2D(0, 0), Point2D(100, 0), length_m=1.0, origin=Point2D(5, 6)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lab.json"
            write_track_project(
                path,
                Path("/tmp/clip.mp4"),
                result,
                calibration=cal,
                show_calibration=False,
                track_mode="precise",
            )
            doc = read_track_project(path)
            v2 = Path(tmp) / "v2.json"
            v2.write_text(
                json.dumps(
                    {
                        "schema": "tracklab.project.v2",
                        "video_path": "/tmp/old.mp4",
                        "tracks": [layer_from_result(result).to_dict()],
                    }
                ),
                encoding="utf-8",
            )
            old = read_track_project(v2)
        self.assertEqual(doc.schema, "tracklab.project.v5")
        self.assertFalse(doc.show_calibration)
        self.assertEqual(doc.track_mode.value, "precise")
        self.assertEqual(doc.calibration.mode, CalibrationMode.UNIFORM)
        self.assertAlmostEqual(doc.calibration.frame.origin_x or 0.0, 5.0)
        self.assertEqual(old.schema, "tracklab.project.v2")
        self.assertEqual(old.calibration.mode, CalibrationMode.NONE)

    def test_low_confidence_yellow_and_calibration_units(self) -> None:
        from PySide6.QtCharts import QScatterSeries
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QApplication

        from ai.calibration import uniform_state
        from ai.schema import Point2D
        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window._new_track()
        layer = window._tracks[0]
        layer.result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=100, y=0, confidence=0.59),
                TrackPoint(frame=1, x=200, y=0, confidence=0.60),
            ],
        )
        window._refresh_track_ui()
        warn = QColor("#f0c14b")
        self.assertEqual(window._data_panel._table.item(0, 2).foreground().color().name(), warn.name())
        self.assertIn("低可信度", window._data_panel._table.item(0, 2).toolTip())
        self.assertNotEqual(window._data_panel._table.item(1, 2).foreground().color().name(), warn.name())
        window._index = 0
        window._sync_view_bar()
        self.assertIn("#f0c14b", window._view_bar._fields["x"].styleSheet())
        window._index = 1
        window._sync_view_bar()
        self.assertNotIn("#f0c14b", window._view_bar._fields["x"].styleSheet())
        scatter_count = sum(
            1
            for series in window._chart_panel._charts[0].chart().series()
            if isinstance(series, QScatterSeries)
        )
        self.assertEqual(scatter_count, 1)

        window._push_undo()
        window._calibration = uniform_state(
            Point2D(0, 0),
            Point2D(100, 0),
            length_m=1.0,
            origin=Point2D(0, 0),
        )
        window._refresh_track_ui()
        self.assertEqual(window._view_bar._labels["x"].text(), "x (m)")
        self.assertEqual(window._data_panel._table.horizontalHeaderItem(2).text(), "x (m)")
        self.assertEqual(window._chart_panel._charts[0]._axis_value.titleText(), "x (m)")
        self.assertTrue(window._video._rulers)
        self.assertEqual(window._video._overlays[0].points[0][1], 100)
        window._index = 0
        window._on_position_edited(0.5, 0.0)
        self.assertAlmostEqual(window._tracks[0].result.points[0].x, 50.0, places=3)
        self.assertTrue(window._tracks[0].result.points[0].manual)
        self.assertEqual(window._tracks[0].result.points[0].confidence, 1.0)
        window._undo()
        self.assertAlmostEqual(window._tracks[0].result.points[0].x, 100.0, places=3)
        window._undo()
        self.assertEqual(window._calibration.mode.value, "none")
        window.close()

    def test_axis_tool_shows_large_draggable_overlay(self) -> None:
        import math

        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow
        from app.widgets import (
            AXIS_ROTATE_SPAN_DEG,
            MODE_AXIS,
            MODE_TRACK,
            VideoView,
            axis_display_length,
            axis_rotate_radius,
        )
        from ai.calibration import uniform_state
        from ai.schema import Point2D
        from engine.video_index import VideoInfo

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window._info = VideoInfo(
            path=Path("clip.mp4"),
            width=1920,
            height=1080,
            pts=(0, 1, 2),
            time_base=0.001,
            pts_ms=(0, 33, 66),
        )
        window._start_axis_tool()
        self.assertEqual(window._video.interaction_mode(), MODE_AXIS)
        origin = window._calibration.frame.origin
        self.assertIsNotNone(origin)
        assert origin is not None
        self.assertAlmostEqual(origin.x, 960.0)
        self.assertAlmostEqual(origin.y, 540.0)
        overlay = window._video._axis_overlay
        self.assertIsNotNone(overlay)
        assert overlay is not None
        ox, oy, xx, xy, yx, yy, *_rest = overlay
        length = math.hypot(xx - ox, xy - oy)
        expected = axis_display_length(1920, 1080)
        self.assertAlmostEqual(length, expected, places=1)
        self.assertGreater(length, 600.0)
        self.assertGreater(xx, ox)
        self.assertLess(yy, oy)
        window._on_axis_dragged("origin", 100.0, 200.0)
        self.assertAlmostEqual(window._calibration.frame.origin_x or 0.0, 100.0)
        self.assertAlmostEqual(window._calibration.frame.origin_y or 0.0, 200.0)
        window._on_axis_drag_finished()
        window._on_axis_dragged("rotate", 200.0, 200.0, False)
        window._on_axis_dragged("rotate", 200.0, 100.0, False)
        self.assertAlmostEqual(window._calibration.frame.axis_angle_deg, 45.0, places=1)
        window._on_axis_drag_finished()
        window._on_axis_dragged("rotate", 200.0, 200.0, False)
        window._on_axis_dragged("rotate", 200.0, 100.0, True)
        self.assertAlmostEqual(window._calibration.frame.axis_angle_deg % 360.0, 90.0, places=1)
        window._exit_interaction()
        self.assertEqual(window._video.interaction_mode(), MODE_TRACK)
        window._video.axis_edit_requested.emit()
        self.assertEqual(window._video.interaction_mode(), MODE_AXIS)

        view = VideoView()
        view.resize(960, 540)
        image = QImage(960, 540, QImage.Format.Format_RGB32)
        image.fill(0)
        view.set_frame(image)
        view.set_interaction_mode(MODE_AXIS)
        span = axis_display_length(960, 540)
        view.set_axis_overlay((480, 270), (480 + span, 270), (480, 270 - span), "x", "y")
        self.assertEqual(view._axis_hit((480, 270)), "origin")
        self.assertIsNone(view._axis_hit((480, 270 - span)))
        handle = axis_rotate_radius(span)
        mid = math.radians(AXIS_ROTATE_SPAN_DEG / 2.0)
        hx = 480 + (span - handle) + handle * math.cos(mid)
        hy = 270 - handle * math.sin(mid)
        self.assertEqual(view._axis_hit((hx, hy)), "rotate")
        self.assertTrue(view._axis_frame_hit((480 + span / 2, 270)))

        window._calibration = uniform_state(
            Point2D(0, 0), Point2D(100, 0), length_m=1.0
        )
        window._refresh_track_ui()
        self.assertEqual(window._data_panel._table.horizontalHeaderItem(2).text(), "x (m)")
        self.assertEqual(window._chart_panel._charts[0]._axis_value.titleText(), "x (m)")
        self.assertIn("vₓ", window._chart_panel._selects[0].itemText(2))
        self.assertIn("m/s", window._chart_panel._selects[0].itemText(2))
        self.assertEqual(window._chart_panel.velocity_step, 3)
        self.assertEqual(window._chart_panel._fit.itemText(1), "线性")
        window.close()

    def test_docks_float_and_redock_right_only(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window.show()
        chart = window._workspace.chart_dock
        data = window._workspace.data_dock
        self.assertEqual(chart.allowedAreas(), Qt.DockWidgetArea.RightDockWidgetArea)
        chart.setFloating(True)
        self.assertTrue(chart.isFloating())
        self.assertTrue(window._chart_action.isChecked())
        window._workspace.dock_chart()
        self.assertFalse(chart.isFloating())
        window._workspace.set_data_visible(False)
        self.assertFalse(window._workspace.data_visible)
        self.assertFalse(window._table_action.isChecked())
        self.assertTrue(window._workspace.chart_visible)
        window._workspace.dock_all()
        self.assertTrue(window._workspace.chart_visible)
        self.assertTrue(window._workspace.data_visible)
        self.assertFalse(chart.isFloating())
        self.assertFalse(data.isFloating())
        self.assertTrue(window._chart_action.isChecked())
        self.assertTrue(window._table_action.isChecked())
        from PySide6.QtCore import QByteArray, QSettings

        from app.dock_workspace import STATE_KEY

        with tempfile.TemporaryDirectory() as tmp:
            settings = QSettings(
                str(Path(tmp) / "prefs.ini"),
                QSettings.Format.IniFormat,
            )
            settings.setValue("workspace/state", QByteArray(b"corrupt-nested-dock-blob"))
            window._workspace.restore_prefs(settings)
        self.assertEqual(STATE_KEY, "workspace/state_v2")
        self.assertEqual(
            window._workspace.dockWidgetArea(chart),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.assertEqual(window._workspace.tabifiedDockWidgets(chart), [])
        window.close()

    def test_assistant_entry_and_default_hidden(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window.show()
        self.assertFalse(window._assistant_window.isVisible())
        self.assertIsNone(getattr(window._workspace, "assistant_dock", None))
        self.assertFalse(window._workspace.isAncestorOf(window._assistant_panel))
        self.assertEqual(window._assistant_window.windowTitle(), "TrackLab 助手")
        self.assertTrue(window._assistant_panel._length_box.isHidden())
        self.assertEqual(
            window._workspace.chart_dock.allowedAreas(),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        window._show_assistant_panel()
        self.assertTrue(window._assistant_window.isVisible())
        self.assertIs(window._assistant_window.centralWidget(), window._assistant_panel)
        window._set_assistant_visible(False)
        self.assertFalse(window._assistant_window.isVisible())
        window._assistant_window_action.setChecked(True)
        self.assertTrue(window._assistant_window.isVisible())
        window._assistant_panel.add_user_message("会话应保留")
        window._assistant_window.close()
        self.assertTrue(window.isVisible())
        self.assertFalse(window._assistant_window.isVisible())
        self.assertIn("会话应保留", window._assistant_panel.chat_text())
        window._show_assistant_panel()
        self.assertTrue(window._assistant_window.isVisible())
        self.assertIn("会话应保留", window._assistant_panel.chat_text())
        window.close()

    def test_assistant_open_then_data_table_click_stays_alive(self) -> None:
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication, QTableWidgetItem

        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        window = MainWindow()
        window.resize(1280, 800)
        window.show()
        app.processEvents()
        workspace = window._workspace
        self.assertIsNone(getattr(workspace, "assistant_dock", None))
        self.assertEqual(
            workspace.dockWidgetArea(workspace.chart_dock),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.assertEqual(
            workspace.dockWidgetArea(workspace.data_dock),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        window._show_assistant_panel()
        app.processEvents()
        self.assertTrue(window._assistant_window.isVisible())
        self.assertFalse(workspace.isAncestorOf(window._assistant_panel))
        self.assertEqual(
            workspace.dockWidgetArea(workspace.chart_dock),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.assertEqual(
            workspace.dockWidgetArea(workspace.data_dock),
            Qt.DockWidgetArea.RightDockWidgetArea,
        )
        self.assertEqual(workspace.tabifiedDockWidgets(workspace.chart_dock), [])
        self.assertEqual(workspace.tabifiedDockWidgets(workspace.data_dock), [])
        table = window._data_panel._table
        table.setRowCount(2)
        for row in range(2):
            item = QTableWidgetItem(str(row + 1))
            item.setData(Qt.ItemDataRole.UserRole, row)
            table.setItem(row, 0, item)
        app.processEvents()
        viewport = table.viewport()
        QTest.mouseClick(
            viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(16, 12)
        )
        app.processEvents()
        QTest.mouseClick(
            table.horizontalHeader(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(24, 6),
        )
        app.processEvents()
        workspace.data_dock.raise_()
        app.processEvents()
        workspace._normalize()
        app.processEvents()
        self.assertTrue(window.isVisible())
        self.assertTrue(workspace.data_visible)
        self.assertTrue(window._assistant_window.isVisible())
        self.assertFalse(workspace.data_dock.isHidden())
        self.assertEqual(workspace.tabifiedDockWidgets(workspace.chart_dock), [])
        self.assertEqual(workspace.tabifiedDockWidgets(workspace.data_dock), [])
        window.close()

    def test_assistant_gating_without_key_or_track(self) -> None:
        from PySide6.QtWidgets import QApplication

        from app.main_window import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        self.assertFalse(window._ai_analyze_action.isEnabled())
        self.assertFalse(window._ai_report_action.isEnabled())
        window._generate_assistant_report()
        self.assertFalse(window._assistant_busy)
        window.close()

    def test_assistant_stream_stop_and_stale(self) -> None:
        from PySide6.QtWidgets import QApplication, QFrame

        from ai.api_credentials import set_api_key
        from ai.contracts import TrackLayer, TrackPoint, TrackResult
        from app.main_window import MainWindow
        from engine.video_index import VideoInfo

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        set_api_key("sk-test", persist=False)
        window = MainWindow()
        window._info = VideoInfo(
            path=Path("clip.mp4"),
            width=64,
            height=64,
            pts=tuple(range(8)),
            time_base=0.001,
            pts_ms=tuple(i * 33 for i in range(8)),
        )
        points = [TrackPoint(frame=i, x=float(i * 4), y=10.0) for i in range(8)]
        window._tracks = [
            TrackLayer(track_id="t1", name="轨迹 1", result=TrackResult(clip_id="c", points=points))
        ]
        window._active_id = "t1"
        chunks = [
            'data: {"choices":[{"delta":{"content":"甲"}}]}',
            'data: {"choices":[{"delta":{"content":"乙"}}]}',
            "data: [DONE]",
        ]

        def transport(url, headers, payload, timeout, stream):
            self.assertNotIn("sk-test", url)
            return list(chunks)

        window._assistant_transport = transport
        window._refresh_track_ui()
        window._analyze_experiment()
        deadline = time.time() + 3
        while window._assistant_busy and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        self.assertIsNotNone(window._assistant_state.analysis)
        self.assertIn("本地轨迹拟合", window._assistant_panel.chat_text())
        analysis = window._assistant_state.analysis
        chosen = analysis.selected or (analysis.candidates[0] if analysis.candidates else None)
        self.assertIsNotNone(chosen)
        window._assistant_state.confirmed_type = chosen.experiment_type
        window._send_assistant_chat("解释一下")
        deadline = time.time() + 3
        while window._assistant_busy and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        self.assertTrue(any(m.role == "assistant" for m in window._assistant_state.messages))
        thoughts = window._assistant_panel._chat._host.findChildren(QFrame)
        thinking = [w for w in thoughts if w.objectName() == "assistantThinking"]
        self.assertTrue(thinking)
        self.assertIn("已思考", thinking[-1]._toggle.text())
        window._tracks[0].result.points[0].x = 99.0
        window._refresh_track_ui()
        self.assertTrue(window._assistant_state.stale)
        window.close()
        from ai import api_credentials as creds

        creds._session_key = None

    def test_project_v4_assistant_roundtrip_and_v3_compat(self) -> None:
        from ai.contracts import AssistantState, ExperimentType, TeachingLevel

        result = TrackResult(
            clip_id="demo",
            points=[TrackPoint(frame=0, x=10, y=20)],
        )
        state = AssistantState(
            confirmed_type=ExperimentType.UNIFORM_LINEAR,
            fingerprint="abc123",
            report_markdown="# 报告\nv = 1.5 m/s",
            teaching_level=TeachingLevel.HIGH,
            model_id="deepseek-v4-flash",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lab.json"
            write_track_project(
                path,
                Path("/tmp/clip.mp4"),
                result,
                assistant=state,
            )
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("sk-", raw)
            self.assertNotIn("DEEPSEEK", raw)
            doc = read_track_project(path)
            v3 = Path(tmp) / "v3.json"
            v3.write_text(
                json.dumps(
                    {
                        "schema": "tracklab.project.v3",
                        "video_path": "/tmp/old.mp4",
                        "tracks": [layer_from_result(result).to_dict()],
                    }
                ),
                encoding="utf-8",
            )
            old = read_track_project(v3)
            md_path = Path(tmp) / "report.md"
            from ai.desktop import export_assistant_report

            export_assistant_report(md_path, doc.assistant.report_markdown)
            self.assertEqual(doc.schema, "tracklab.project.v5")
            self.assertEqual(doc.assistant.confirmed_type, ExperimentType.UNIFORM_LINEAR)
            self.assertIn("1.5", doc.assistant.report_markdown)
            self.assertEqual(old.schema, "tracklab.project.v3")
            self.assertIsNone(old.assistant.analysis)
            self.assertTrue(md_path.read_text(encoding="utf-8").startswith("# 报告"))

    def test_report_numbers_come_from_local_analysis(self) -> None:
        from ai.assistant_report import render_report_markdown
        from ai.contracts import ExperimentAnalysis, ExperimentCandidate, ExperimentType, FitResult

        fit = FitResult(
            model="s=s0+vt",
            formula_id="uniform.s",
            frame_start=0,
            frame_end=9,
            time_start_s=0.0,
            time_end_s=0.3,
            parameters={"v": 1.23456, "s0": 0.0},
            units={"v": "m/s", "s0": "m"},
            r2=0.99,
            nrmse=0.01,
            n_samples=10,
        )
        candidate = ExperimentCandidate(
            experiment_type=ExperimentType.UNIFORM_LINEAR,
            label="匀速直线运动",
            confidence=0.9,
            fit=fit,
        )
        analysis = ExperimentAnalysis(
            clip_id="demo",
            candidates=[candidate],
            selected=candidate,
            calibration_active=True,
            position_unit="m",
        )
        markdown = render_report_markdown(
            analysis,
            confirmed_type=ExperimentType.UNIFORM_LINEAR,
            sections={"purpose": "测速度", "conclusion": "模型不得改写 9.9"},
        )
        self.assertIn("1.23456", markdown)
        self.assertIn("测速度", markdown)
        self.assertIn("几何标定", markdown)

    def test_planar_apply_undo_project_and_csv(self) -> None:
        from PySide6.QtWidgets import QApplication

        from ai.calibration import CalibrationMode
        from ai.schema import Point2D
        from app.main_window import MainWindow
        from tests.ai.dataset import load_manifest, resolve_video
        from ai.models import load_video

        app = QApplication.instance() or QApplication(sys.argv)
        _ = app
        window = MainWindow()
        window._cal_dialog.set_mode(CalibrationMode.PLANAR)
        window._cal_dialog.set_plane_size(1.6, 1.2)
        window._pending_plane = [
            Point2D(40, 40),
            Point2D(200, 40),
            Point2D(200, 160),
            Point2D(40, 160),
        ]
        window._apply_pending_calibration()
        self.assertEqual(window._calibration.mode, CalibrationMode.PLANAR)
        self.assertTrue(window._calibration.active)
        self.assertTrue(window._calibration.rulers == [])
        window._undo()
        self.assertEqual(window._calibration.mode, CalibrationMode.NONE)
        window._redo()
        self.assertEqual(window._calibration.mode, CalibrationMode.PLANAR)
        window._new_track()
        layer = window._tracks[0]
        layer.result = TrackResult(
            clip_id="c",
            points=[TrackPoint(frame=0, x=80, y=80), TrackPoint(frame=1, x=120, y=90)],
        )
        window._refresh_track_ui()
        self.assertEqual(window._data_panel._table.horizontalHeaderItem(2).text(), "x (m)")
        self.assertEqual(window._data_panel._table.horizontalHeaderItem(10).text(), "质量")
        self.assertIn("几何", window._data_panel._table.item(0, 10).text())
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        info = load_video(resolve_video(entry))
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "plane.json"
            csv_path = Path(tmp) / "plane.csv"
            write_track_project(
                project,
                info.path,
                layer.result,
                calibration=window._calibration,
                depth_audit=window._depth_audit,
            )
            export_track_csv(
                csv_path,
                layer.result,
                info,
                calibration=window._calibration,
            )
            doc = read_track_project(project)
            text = csv_path.read_text(encoding="utf-8")
        self.assertEqual(doc.schema, "tracklab.project.v5")
        self.assertEqual(doc.calibration.mode, CalibrationMode.PLANAR)
        self.assertEqual(len(doc.calibration.plane.corners), 4)
        self.assertIn("sigma_x", text)
        self.assertIn("quality", text)
        self.assertIn("几何测量", text)


if __name__ == "__main__":
    unittest.main()
