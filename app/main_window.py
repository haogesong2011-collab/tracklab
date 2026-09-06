from __future__ import annotations

import json
import os
import time
from pathlib import Path

from datetime import datetime

from PySide6.QtCore import QByteArray, QEvent, QSettings, QThread, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QDragEnterEvent, QDropEvent, QImage, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import GITHUB_REPO, __version__
from app.icons import (
    icon_size,
    loop_icon,
    pause_icon,
    play_icon,
    prev_icon,
    toolbar_icon,
)
from app.assistant_panel import (
    AssistantPanel,
    AssistantWindow,
    DeepSeekSettingsDialog,
    show_payload_preview,
    warn_missing_key,
)
from app.calibration_dialog import CalibrationDialog, CameraCalibDialog
from app.dock_workspace import DockWorkspace, _on_screen
from app.frame_pump import FramePump
from app.track_panels import TrackDataPanel, TrackListPanel
from app.track_window import TrackManagerWindow
from app.data_views import TrackChartPanel
from app.download_toast import DownloadToast
from app.paths import frozen_app_bundle, style_path
from app.update_checker import (
    SETTINGS_AUTO_CHECK,
    SETTINGS_LAST_CHECK,
    SETTINGS_SKIPPED,
    UpdateStatus,
    can_self_update,
    should_auto_check,
    status_bar_message,
    update_checks_allowed,
)
from app.update_dialog import (
    open_release_page,
    run_self_update_in_thread,
    run_update_check_in_thread,
    show_update_result,
)
from app.view_toolbar import ViewToolbar
from app.widgets import (
    AXIS_MIN_LENGTH,
    MODE_AXIS,
    MODE_PLANE,
    MODE_RULER,
    MODE_TRACK,
    DropHint,
    OverlayTrack,
    StepStepper,
    TimelineSlider,
    VideoInfoLabel,
    VideoStage,
    VideoView,
    axis_arm_ends,
    axis_display_length,
    axis_pointer_angle,
    snap_axis_angle,
)
from ai.api_credentials import get_api_key, has_api_key
from ai.assistant_context import (
    assert_private_context,
    build_teaching_context,
    chat_messages,
    preview_payload,
    report_messages,
)
from ai.assistant_report import merge_report_sections, render_report_markdown
from ai.assistant_worker import AssistantJob, AssistantOutcome, run_assistant_in_thread
from ai.calibration import (
    CalibrationMode,
    CalibrationState,
    CameraProfile,
    CoordinateFrame,
    RulerRole,
    RulerSegment,
    near_far_state,
    planar_state,
    uniform_state,
)
from ai.contracts import (
    TRACK_COLORS,
    AssistantState,
    ChatMessage,
    ExperimentType,
    ProgressEvent,
    PromptKind,
    TeachingLevel,
    TrackLayer,
    TrackMode,
    TrackPrompt,
    TrackResult,
)
from ai.deepseek_client import DEFAULT_CHAT_MODEL, DEFAULT_REPORT_MODEL
from ai.desktop import (
    apply_manual_override,
    build_assistant_report,
    export_assistant_report,
    export_track_csv,
    new_track_id,
    read_track_project,
    run_audit_in_thread,
    run_shake_in_thread,
    run_track_in_thread,
    run_checkpoint_download,
    write_track_project,
)
from ai.depth_audit import DepthAuditState
from ai.kinematics import sample_at_frame, series_for_result
from ai.physics import source_fingerprint
from ai.model_manager import spec_for_mode
from ai.sam2_tracker import merge_track_points
from ai.schema import Point2D
from ai.stabilize import ShakeCompensation, compensate_result
from engine.decoder import FrameDecoder
from engine.video_index import VideoInfo

VIDEO_FILTER = (
    "视频文件 (*.mp4 *.mov *.m4v *.avi *.mkv *.webm *.mpg *.mpeg);;"
    "所有文件 (*)"
)
PROJECT_FILTER = "TrackLab 项目 (*.json);;所有文件 (*)"
VIDEO_SUFFIXES = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
}
STYLE_PATH = style_path()
SHAKE_INVALIDATE_PX = 8.0


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("TrackLab")
        self.resize(1280, 800)
        self.setMinimumSize(900, 560)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE_PATH.read_text(encoding="utf-8"))

        self._info: VideoInfo | None = None
        self._index = 0
        self._playing = False
        self._scrubbing = False
        self._speed_factor = 1.0
        self._pump: FramePump | None = None
        self._pending_play_index: int | None = None
        self._deadline = 0.0
        self._ai_thread = None
        self._ai_worker = None
        self._sam_thread: QThread | None = None
        self._track_mode = TrackMode.PRECISE
        self._shake_thread = None
        self._shake_worker = None
        self._shake: ShakeCompensation | None = None
        self._shake_enabled = True
        self._show_anchors = True
        self._tracks: list[TrackLayer] = []
        self._active_id: str | None = None
        self._undo_stack: list[tuple[list[TrackLayer], str | None, CalibrationState]] = []
        self._redo_stack: list[tuple[list[TrackLayer], str | None, CalibrationState]] = []
        self._undoing = False
        self._pending_project = None
        self._show_contours = True
        self._show_prompts = True
        self._calibration = CalibrationState()
        self._show_calibration = True
        self._pending_rulers: list[RulerSegment] | None = None
        self._pending_plane: list[Point2D] | None = None
        self._axis_undo_pushed = False
        self._plane_undo_pushed = False
        self._axis_rotate_base: tuple[float, float] | None = None
        self._cal_drawn_before_shake = False
        self._depth_audit = DepthAuditState()
        self._depth_thread = None
        self._depth_worker = None
        self._charuco_views: list = []
        self._assistant_thread = None
        self._assistant_worker = None
        self._assistant_state = AssistantState()
        self._assistant_busy = False
        self._chat_model = DEFAULT_CHAT_MODEL
        self._report_model = DEFAULT_REPORT_MODEL
        self._assistant_transport = None
        self._download_thread: QThread | None = None
        self._download_worker = None
        self._download_toast: DownloadToast | None = None
        self._download_resume_track = False
        self._update_thread = None
        self._update_worker = None
        self._update_manual = False
        self._self_update_thread = None
        self._self_update_worker = None
        self._applying_update = False

        self._video = VideoView()
        self._hint = DropHint()
        self._hint.clicked.connect(self._open_dialog)
        self._stage = VideoStage(self._hint, self._video)
        self._stack = self._stage.stack
        self._video.clicked_at.connect(self._on_video_click)
        self._video.prompted.connect(self._on_prompted)
        self._video.boxed.connect(self._on_boxed)
        self._video.ruler_drawn.connect(self._on_ruler_drawn)
        self._video.plane_point_picked.connect(self._on_plane_point_picked)
        self._video.plane_corner_dragged.connect(self._on_plane_corner_dragged)
        self._video.plane_drag_finished.connect(self._on_plane_drag_finished)
        self._video.axis_dragged.connect(self._on_axis_dragged)
        self._video.axis_drag_finished.connect(self._on_axis_drag_finished)
        self._video.axis_edit_requested.connect(self._start_axis_tool)
        self._video.interaction_cancelled.connect(self._on_interaction_cancelled)

        self._list_panel = TrackListPanel()
        self._list_panel.new_requested.connect(self._new_track)
        self._list_panel.delete_requested.connect(self._delete_track)
        self._list_panel.selection_changed.connect(self._select_track)
        self._list_panel.visibility_toggled.connect(self._toggle_track_visible)
        self._list_panel.rename_requested.connect(self._rename_track)
        self._list_panel.track_requested.connect(self._start_or_cancel_ai_track)
        self._list_panel.cancel_requested.connect(self._cancel_ai_track)
        self._list_panel.shake_toggled.connect(self._on_shake_apply_toggled)
        self._track_window = TrackManagerWindow(self._list_panel, self)
        self._track_window.set_visibility_hook(self._on_track_window_visible)

        self._view_bar = ViewToolbar()
        self._view_bar.track_selected.connect(self._select_track)
        self._view_bar.visibility_toggled.connect(self._toggle_active_visible)
        self._view_bar.position_edited.connect(self._on_position_edited)
        self._view_bar.prev_point_requested.connect(lambda: self._step_track_point(-1))
        self._view_bar.next_point_requested.connect(lambda: self._step_track_point(1))

        self._data_panel = TrackDataPanel()
        self._data_panel.frame_activated.connect(self._show_frame)
        self._chart_panel = TrackChartPanel()
        self._chart_panel.frame_activated.connect(self._on_chart_frame_activated)
        self._chart_panel.velocity_step_changed.connect(self._refresh_track_ui)
        self._assistant_panel = AssistantPanel()
        self._assistant_panel.analyze_requested.connect(self._analyze_experiment)
        self._assistant_panel.confirm_requested.connect(self._confirm_experiment)
        self._assistant_panel.send_requested.connect(self._send_assistant_chat)
        self._assistant_panel.stop_requested.connect(self._cancel_assistant)
        self._assistant_panel.report_requested.connect(self._generate_assistant_report)
        self._assistant_panel.export_requested.connect(self._export_assistant_report)
        self._assistant_panel.copy_report_requested.connect(self._copy_assistant_report)
        self._assistant_panel.preview_requested.connect(self._preview_assistant_payload)
        self._assistant_panel.clear_chat_requested.connect(self._clear_assistant_chat)
        self._assistant_panel.level_changed.connect(self._on_teaching_level)
        self._assistant_panel.settings_requested.connect(self._open_deepseek_settings)
        self._assistant_panel.length_changed.connect(self._on_pendulum_length)
        self._assistant_window = AssistantWindow(self._assistant_panel, self)
        self._assistant_window.setStyleSheet(self.styleSheet())
        self._assistant_window.visibility_changed.connect(self._on_assistant_window_visible)
        self._workspace = DockWorkspace(self._stage, self._chart_panel, self._data_panel)
        self._data_column = self._workspace.data_dock
        self._workspace.chart_visibility_changed.connect(self._on_chart_dock_visible)
        self._workspace.data_visibility_changed.connect(self._on_data_dock_visible)

        self._cal_dialog = CalibrationDialog(self)
        self._cal_dialog.mode_changed.connect(self._on_cal_mode_changed)
        self._cal_dialog.apply_requested.connect(self._apply_pending_calibration)
        self._cal_dialog.redraw_requested.connect(self._redraw_rulers)
        self._cal_dialog.swap_requested.connect(self._swap_pending_rulers)
        self._cal_dialog.length_changed.connect(self._on_cal_lengths)
        self._cal_dialog.plane_size_changed.connect(self._on_plane_size_changed)
        self._cal_dialog.charuco_requested.connect(self._on_charuco_detect)
        self._cal_dialog.camera_calib_requested.connect(self._on_camera_calib)
        self._cal_dialog.audit_requested.connect(self._on_depth_audit)
        self._cal_dialog.experimental_changed.connect(self._on_experimental_correction)
        self._camera_dialog = CameraCalibDialog(self)
        self._camera_dialog.capture_requested.connect(self._capture_calib_view)
        self._camera_dialog.compute_requested.connect(self._compute_camera_profile)

        self._frame_readout = QLabel("帧 0 / 0")
        self._frame_readout.setObjectName("frameReadout")
        self._frame_readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self._play_btn = QPushButton()
        self._play_btn.setObjectName("playButton")
        self._play_btn.setEnabled(False)
        self._play_btn.setIconSize(icon_size())
        self._play_btn.setToolTip("播放 / 暂停（空格）")
        self._play_btn.clicked.connect(self._toggle_play)
        self._set_play_icon(False)

        self._prev_btn = self._icon_button(prev_icon(), "回到开头", self._jump_start)

        self._slider = TimelineSlider()
        self._slider.setEnabled(False)
        self._slider.setRange(0, 0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(1)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.valueChanged.connect(self._on_slider_value)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.loop_range_changed.connect(self._on_loop_range_changed)

        toolbar = self._build_toolbar()
        self._video.zoom_changed.connect(self._on_zoom_changed)

        self._speed = QComboBox()
        self._speed.setObjectName("speedControl")
        self._speed.addItems(["25%", "50%", "75%", "100%", "125%", "150%", "200%"])
        self._speed.setCurrentText("100%")
        self._speed.setFixedWidth(72)
        self._speed.setToolTip("播放速度")
        self._speed.currentTextChanged.connect(self._on_speed_changed)

        self._stepper = StepStepper()
        self._stepper.step_requested.connect(self._step)

        self._loop_btn = QPushButton()
        self._loop_btn.setObjectName("loopButton")
        self._loop_btn.setIcon(loop_icon())
        self._loop_btn.setIconSize(icon_size())
        self._loop_btn.setCheckable(True)
        self._loop_btn.setEnabled(False)
        self._loop_btn.setToolTip("在标记范围内循环播放（拖动进度条上方的三角调整范围）")
        self._loop_btn.toggled.connect(self._on_loop_toggled)

        toolbar_shell = QWidget()
        toolbar_shell.setObjectName("toolbarShell")
        toolbar_shell.setFixedHeight(80)
        toolbar_shell_layout = QVBoxLayout(toolbar_shell)
        toolbar_shell_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_shell_layout.setSpacing(0)
        toolbar_shell_layout.addWidget(toolbar)
        toolbar_shell_layout.addWidget(self._view_bar)

        transport = QWidget()
        transport.setObjectName("transportBar")
        transport.setFixedHeight(56)
        controls = QHBoxLayout(transport)
        controls.setContentsMargins(16, 6, 14, 6)
        controls.setSpacing(6)
        controls.addWidget(self._speed)
        controls.addWidget(self._prev_btn)
        controls.addWidget(self._play_btn)
        controls.addSpacing(8)
        controls.addWidget(self._slider, stretch=1)
        controls.addWidget(self._frame_readout)
        controls.addSpacing(6)
        controls.addWidget(self._stepper)
        controls.addSpacing(4)
        controls.addWidget(self._loop_btn)

        transport_shell = QWidget()
        transport_shell.setObjectName("transportShell")
        transport_shell.setFixedHeight(56)
        transport_shell_layout = QVBoxLayout(transport_shell)
        transport_shell_layout.setContentsMargins(0, 0, 0, 0)
        transport_shell_layout.addWidget(transport)

        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar_shell)
        layout.addWidget(self._workspace, stretch=1)
        layout.addWidget(transport_shell)
        self.setCentralWidget(root)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().setFixedHeight(24)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_play_tick)

        self._build_menu()
        self._restore_window_prefs()
        self._refresh_track_ui()
        if update_checks_allowed():
            self._update_startup_timer = QTimer(self)
            self._update_startup_timer.setSingleShot(True)
            self._update_startup_timer.timeout.connect(self._maybe_auto_check_updates)
            self._update_startup_timer.start(2500)

    def _icon_button(self, icon, tooltip: str, slot) -> QPushButton:
        button = QPushButton()
        button.setObjectName("iconButton")
        button.setEnabled(False)
        button.setIcon(icon)
        button.setIconSize(icon_size())
        button.setToolTip(tooltip)
        button.clicked.connect(slot)
        return button

    def _build_toolbar(self) -> QWidget:
        toolbar = QWidget()
        toolbar.setObjectName("toolBar")
        toolbar.setFixedHeight(44)
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(4)

        tools = [
            ("open", "打开视频", self._open_dialog),
            ("save", "保存项目", self._save_project),
            ("video", "视频设置", None),
            ("ruler", "标定尺", self._start_ruler_tool),
            ("axis", "坐标系：点击工具或双击坐标轴进入编辑", self._start_axis_tool),
            ("track", "轨迹", self._toggle_track_window),
            ("ai", "AI 助手", self._show_assistant_panel),
            ("view", "显示选项", self._toggle_overlays),
            ("zoom", "滚轮缩放，点击还原", self._reset_zoom),
        ]
        self._zoom_readout = QLabel("100%")
        self._zoom_readout.setObjectName("zoomReadout")
        self._ruler_btn: QToolButton | None = None
        self._axis_btn: QToolButton | None = None
        for index, (name, tooltip, slot) in enumerate(tools):
            if index in (2, 5, 7):
                divider = QWidget()
                divider.setObjectName("toolDivider")
                divider.setFixedSize(1, 22)
                layout.addWidget(divider)
                layout.addSpacing(4)
            button = QToolButton()
            button.setObjectName("toolButton")
            button.setIcon(toolbar_icon(name))
            button.setIconSize(icon_size())
            button.setToolTip(tooltip)
            if slot is not None:
                button.clicked.connect(slot)
            if name in {"ruler", "axis"}:
                button.setCheckable(True)
            if name == "ruler":
                self._ruler_btn = button
            elif name == "axis":
                self._axis_btn = button
            layout.addWidget(button)
            if name == "zoom":
                layout.addWidget(self._zoom_readout)

        layout.addStretch()
        self._video_info = VideoInfoLabel()
        layout.addWidget(self._video_info, stretch=1)
        cache_btn = QToolButton()
        cache_btn.setObjectName("toolButton")
        cache_btn.setIcon(toolbar_icon("cache"))
        cache_btn.setIconSize(icon_size())
        cache_btn.setToolTip("清理缓存")
        cache_btn.clicked.connect(self._clear_cache)
        layout.addWidget(cache_btn)
        return toolbar

    def _build_menu(self) -> None:
        self.menuBar().setNativeMenuBar(False)
        file_menu = self.menuBar().addMenu("文件")
        open_action = QAction("打开视频…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_dialog)
        file_menu.addAction(open_action)

        open_project = QAction("打开项目…", self)
        open_project.triggered.connect(self._open_project)
        file_menu.addAction(open_project)

        self._close_action = QAction("关闭视频", self)
        self._close_action.setShortcut(QKeySequence.StandardKey.Close)
        self._close_action.setEnabled(False)
        self._close_action.triggered.connect(self.close_video)
        file_menu.addAction(self._close_action)

        self._save_action = QAction("保存项目…", self)
        self._save_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_action.setEnabled(False)
        self._save_action.triggered.connect(self._save_project)
        file_menu.addAction(self._save_action)

        self._export_track_action = QAction("导出轨迹 JSON…", self)
        self._export_track_action.setEnabled(False)
        self._export_track_action.triggered.connect(self._export_track)
        file_menu.addAction(self._export_track_action)

        self._export_csv_action = QAction("导出轨迹 CSV…", self)
        self._export_csv_action.setEnabled(False)
        self._export_csv_action.triggered.connect(self._export_csv)
        file_menu.addAction(self._export_csv_action)

        edit_menu = self.menuBar().addMenu("编辑")
        self._undo_action = QAction("撤销", self)
        self._undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_action.setEnabled(False)
        self._undo_action.triggered.connect(self._undo)
        edit_menu.addAction(self._undo_action)
        self._redo_action = QAction("重做", self)
        self._redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_action.setEnabled(False)
        self._redo_action.triggered.connect(self._redo)
        edit_menu.addAction(self._redo_action)
        settings_action = QAction("DeepSeek 设置…", self)
        settings_action.triggered.connect(self._open_deepseek_settings)
        edit_menu.addAction(settings_action)

        play_menu = self.menuBar().addMenu("视频")
        play_action = QAction("播放/暂停", self)
        play_action.setShortcut(Qt.Key.Key_Space)
        play_action.triggered.connect(self._toggle_play)
        prev_action = QAction("上一帧", self)
        prev_action.setShortcut(Qt.Key.Key_Left)
        prev_action.triggered.connect(lambda: self._step(-self._stepper.value()))
        next_action = QAction("下一帧", self)
        next_action.setShortcut(Qt.Key.Key_Right)
        next_action.triggered.connect(lambda: self._step(self._stepper.value()))
        play_menu.addAction(play_action)
        play_menu.addAction(prev_action)
        play_menu.addAction(next_action)
        play_menu.addSeparator()

        mark_in = QAction("循环起点设为当前帧", self)
        mark_in.setShortcut(Qt.Key.Key_I)
        mark_in.triggered.connect(lambda: self._mark_loop("start"))
        mark_out = QAction("循环终点设为当前帧", self)
        mark_out.setShortcut(Qt.Key.Key_O)
        mark_out.triggered.connect(lambda: self._mark_loop("end"))
        reset_loop = QAction("循环范围恢复整段", self)
        reset_loop.triggered.connect(self._reset_loop_range)
        play_menu.addAction(mark_in)
        play_menu.addAction(mark_out)
        play_menu.addAction(reset_loop)

        track_menu = self.menuBar().addMenu("轨迹")
        self._ai_track_action = QAction("SAM 自动跟踪", self)
        self._ai_track_action.setShortcut(Qt.Key.Key_T)
        self._ai_track_action.setEnabled(False)
        self._ai_track_action.triggered.connect(self._start_or_cancel_ai_track)
        track_menu.addAction(self._ai_track_action)
        self._cancel_ai_action = QAction("取消 AI 任务", self)
        self._cancel_ai_action.setEnabled(False)
        self._cancel_ai_action.triggered.connect(self._cancel_ai_track)
        track_menu.addAction(self._cancel_ai_action)
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        self._fast_track_action = QAction("快速预览（Tiny）", self)
        self._fast_track_action.setCheckable(True)
        self._precise_track_action = QAction("精准分析（Small）", self)
        self._precise_track_action.setCheckable(True)
        self._precise_track_action.setChecked(True)
        mode_group.addAction(self._fast_track_action)
        mode_group.addAction(self._precise_track_action)
        self._fast_track_action.triggered.connect(lambda: self._set_track_mode(TrackMode.FAST))
        self._precise_track_action.triggered.connect(
            lambda: self._set_track_mode(TrackMode.PRECISE)
        )
        track_menu.addSeparator()
        track_menu.addAction(self._fast_track_action)
        track_menu.addAction(self._precise_track_action)
        new_track = track_menu.addAction("新建轨迹")
        new_track.triggered.connect(self._new_track)
        mgr = track_menu.addAction("轨迹管理器")
        mgr.triggered.connect(self._toggle_track_window)
        self._shake_action = QAction("背景补偿", self)
        self._shake_action.setEnabled(False)
        self._shake_action.triggered.connect(self._run_shake_compensation)
        track_menu.addSeparator()
        track_menu.addAction(self._shake_action)

        view_menu = self.menuBar().addMenu("显示")
        self._contour_action = QAction("显示分割轮廓", self)
        self._contour_action.setCheckable(True)
        self._contour_action.setChecked(True)
        self._contour_action.toggled.connect(self._on_display_toggled)
        self._prompt_action = QAction("显示提示点", self)
        self._prompt_action.setCheckable(True)
        self._prompt_action.setChecked(True)
        self._prompt_action.toggled.connect(self._on_display_toggled)
        self._shake_apply_action = QAction("应用背景补偿", self)
        self._shake_apply_action.setCheckable(True)
        self._shake_apply_action.setChecked(True)
        self._shake_apply_action.toggled.connect(self._on_shake_apply_toggled)
        self._anchor_action = QAction("显示四角参照点", self)
        self._anchor_action.setCheckable(True)
        self._anchor_action.setChecked(True)
        self._anchor_action.toggled.connect(self._on_anchor_toggled)
        view_menu.addAction(self._contour_action)
        view_menu.addAction(self._prompt_action)
        view_menu.addSeparator()
        view_menu.addAction(self._shake_apply_action)
        view_menu.addAction(self._anchor_action)

        for title, items in (
            ("坐标系", []),
            ("AI助手", []),
            ("窗口", []),
            ("帮助", ["快速开始", "快捷键", "关于 TrackLab"]),
        ):
            menu = self.menuBar().addMenu(title)
            if title == "坐标系":
                set_ruler = menu.addAction("设置/重设标定尺")
                set_ruler.triggered.connect(self._start_ruler_tool)
                set_plane = menu.addAction("设置运动平面")
                set_plane.triggered.connect(self._start_plane_menu)
                set_origin = menu.addAction("设置原点")
                set_origin.triggered.connect(self._start_origin_only)
                set_axis = menu.addAction("设置坐标轴")
                set_axis.triggered.connect(self._start_axis_direction)
                menu.addSeparator()
                self._cal_overlay_action = QAction("显示标定叠加", self)
                self._cal_overlay_action.setCheckable(True)
                self._cal_overlay_action.setChecked(True)
                self._cal_overlay_action.toggled.connect(self._on_cal_overlay_toggled)
                menu.addAction(self._cal_overlay_action)
                clear_cal = menu.addAction("清除标定")
                clear_cal.triggered.connect(self._clear_calibration)
            elif title == "AI助手":
                self._ai_panel_action = QAction("显示面板", self)
                self._ai_panel_action.setCheckable(True)
                self._ai_panel_action.toggled.connect(self._set_assistant_visible)
                menu.addAction(self._ai_panel_action)
                self._ai_analyze_action = QAction("识别实验", self)
                self._ai_analyze_action.triggered.connect(self._analyze_experiment)
                menu.addAction(self._ai_analyze_action)
                motion = QAction("运动分析", self)
                motion.triggered.connect(self._analyze_experiment)
                menu.addAction(motion)
                self._ai_report_action = QAction("生成报告", self)
                self._ai_report_action.triggered.connect(self._generate_assistant_report)
                menu.addAction(self._ai_report_action)
                menu.addSeparator()
                menu.addAction("DeepSeek 设置…").triggered.connect(self._open_deepseek_settings)
                self._ai_assist_action = self._ai_panel_action
                self._ai_analyze_action.setEnabled(False)
                self._ai_report_action.setEnabled(False)
            elif title == "窗口":
                self._track_window_action = QAction("轨迹管理器", self)
                self._track_window_action.setCheckable(True)
                self._track_window_action.toggled.connect(self._set_track_window_visible)
                menu.addAction(self._track_window_action)
                self._chart_action = QAction("分图", self)
                self._chart_action.setCheckable(True)
                self._chart_action.setChecked(True)
                self._chart_action.toggled.connect(self._workspace.set_chart_visible)
                menu.addAction(self._chart_action)
                self._table_action = QAction("数据表", self)
                self._table_action.setCheckable(True)
                self._table_action.setChecked(True)
                self._table_action.toggled.connect(self._workspace.set_data_visible)
                menu.addAction(self._table_action)
                self._assistant_window_action = QAction("AI 助手", self)
                self._assistant_window_action.setCheckable(True)
                self._assistant_window_action.toggled.connect(self._set_assistant_visible)
                menu.addAction(self._assistant_window_action)
                menu.addSeparator()
                menu.addAction("分图归位").triggered.connect(self._workspace.dock_chart)
                menu.addAction("数据表归位").triggered.connect(self._workspace.dock_data)
                menu.addAction("全部归位").triggered.connect(self._workspace.dock_all)
                restore = menu.addAction("恢复默认布局")
                restore.triggered.connect(self._restore_layout)
            elif title == "帮助":
                start = menu.addAction("快速开始")
                start.triggered.connect(self._show_quick_start)
                keys = menu.addAction("快捷键")
                keys.triggered.connect(self._show_shortcuts)
                menu.addSeparator()
                check = menu.addAction("检查更新…")
                check.triggered.connect(lambda: self._check_for_updates(manual=True))
                self._auto_update_action = QAction("启动时自动检查更新", self)
                self._auto_update_action.setCheckable(True)
                self._auto_update_action.setChecked(self._auto_check_enabled())
                self._auto_update_action.toggled.connect(self._on_auto_check_toggled)
                menu.addAction(self._auto_update_action)
                menu.addSeparator()
                about = menu.addAction("关于 TrackLab")
                about.triggered.connect(self._show_about)

    @staticmethod
    def _add_placeholders(menu, labels: list[str]) -> None:
        for label in labels:
            action = menu.addAction(label)
            action.setEnabled(False)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._save_window_prefs()
        self._pause()
        self._stop_ai()
        self._stop_checkpoint_download()
        self._stop_sam_thread()
        self._stop_assistant()
        self._stop_shake()
        self._stop_depth_audit()
        self._stop_update_check()
        if not self._applying_update:
            self._stop_self_update()
        self._stop_pump()
        if self._track_window is not None:
            self._track_window.hide()
        if getattr(self, "_assistant_window", None) is not None:
            self._assistant_window.hide()
        if self._cal_dialog is not None:
            self._cal_dialog.hide()
        if getattr(self, "_camera_dialog", None) is not None:
            self._camera_dialog.hide()
        super().closeEvent(event)

    def _auto_check_enabled(self) -> bool:
        value = QSettings().value(SETTINGS_AUTO_CHECK, True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no"}

    def _on_auto_check_toggled(self, checked: bool) -> None:
        QSettings().setValue(SETTINGS_AUTO_CHECK, checked)

    def _maybe_auto_check_updates(self) -> None:
        if not update_checks_allowed() or not self._auto_check_enabled():
            return
        settings = QSettings()
        try:
            last_check = float(settings.value(SETTINGS_LAST_CHECK, 0.0) or 0.0)
        except (TypeError, ValueError):
            last_check = 0.0
        if not should_auto_check(last_check, time.time()):
            return
        self._check_for_updates(manual=False)

    def _check_for_updates(self, *, manual: bool) -> None:
        if self._update_thread is not None:
            if manual:
                self.statusBar().showMessage("正在检查更新…", 4000)
            return
        self._update_manual = manual
        skipped = str(QSettings().value(SETTINGS_SKIPPED, "") or "")
        if manual:
            self.statusBar().showMessage("正在检查更新…")
        thread, worker = run_update_check_in_thread(
            __version__,
            skipped=skipped,
            honor_skip=not manual,
            on_finished=self._on_update_check_finished,
        )
        self._update_thread = thread
        self._update_worker = worker
        thread.start()

    def _on_update_check_finished(self, info) -> None:  # noqa: ANN001
        manual = self._update_manual
        self._stop_update_check()
        QSettings().setValue(SETTINGS_LAST_CHECK, time.time())
        choice = show_update_result(self, info, manual=manual)
        if choice == "skip" and info.latest:
            QSettings().setValue(SETTINGS_SKIPPED, info.latest)
        if choice == "download" and can_self_update(info):
            self._start_self_update(info)
        if manual:
            if info.status is UpdateStatus.AVAILABLE:
                self.statusBar().clearMessage()
            return
        if info.status in {
            UpdateStatus.AVAILABLE,
            UpdateStatus.SKIPPED,
            UpdateStatus.LATEST,
        }:
            return
        self.statusBar().showMessage(status_bar_message(info), 6000)

    def _stop_update_check(self) -> None:
        if self._update_thread is not None:
            self._update_thread.quit()
            self._update_thread.wait(2000)
        self._update_thread = None
        self._update_worker = None

    def _start_self_update(self, info) -> None:  # noqa: ANN001
        if self._self_update_worker is not None:
            return
        if self._download_worker is not None:
            QMessageBox.information(self, "正在下载", "请等待当前下载完成后再更新。")
            return
        bundle = frozen_app_bundle()
        if bundle is None:
            open_release_page(info)
            return
        toast = self._ensure_download_toast()
        name = info.installer.name if info.installer is not None else "TrackLab.dmg"
        toast.begin(f"正在更新到 {info.latest or '新版本'}", name)
        toast.show()
        toast.reposition()
        thread, worker = run_self_update_in_thread(
            info,
            bundle,
            __version__,
            on_progress=toast.set_progress,
            on_stage=toast.set_stage,
            on_finished=self._on_self_update_ready,
            on_failed=self._on_self_update_failed,
            on_cancelled=self._on_self_update_cancelled,
        )
        self._self_update_thread = thread
        self._self_update_worker = worker
        thread.start()
        self.statusBar().showMessage("正在下载新版本…")

    def _cancel_self_update(self) -> None:
        if self._self_update_worker is not None:
            self._self_update_worker.cancel()
            self.statusBar().showMessage("正在取消更新…", 3000)

    def _stop_self_update(self) -> None:
        worker = self._self_update_worker
        if worker is not None:
            worker.cancel()
        thread = self._self_update_thread
        self._self_update_thread = None
        self._self_update_worker = None
        if thread is not None:
            thread.quit()
            thread.wait(2000)

    def _on_self_update_ready(self, new_app) -> None:  # noqa: ANN001
        from app.self_update import launch_replacer

        toast = self._download_toast
        if toast is not None:
            toast.set_finished("即将重启并完成安装")
        self._self_update_thread = None
        self._self_update_worker = None
        bundle = frozen_app_bundle()
        if bundle is None:
            self._hide_download_toast()
            QMessageBox.warning(self, "更新", "找不到当前安装包，请到网页下载。")
            return
        QMessageBox.information(
            self,
            "更新已就绪",
            "新版本已下载并校验。TrackLab 将退出，随后自动替换并重新打开。",
        )
        self._applying_update = True
        launch_replacer(pid=os.getpid(), bundle=bundle, new_app=Path(new_app))
        QTimer.singleShot(200, self.close)

    def _on_self_update_failed(self, message: str) -> None:
        self._self_update_thread = None
        self._self_update_worker = None
        self._hide_download_toast()
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "更新失败", message + "\n\n可改从网页下载安装包。")

    def _on_self_update_cancelled(self) -> None:
        self._self_update_thread = None
        self._self_update_worker = None
        self._hide_download_toast()
        self.statusBar().showMessage("已取消更新", 4000)

    def _hide_download_toast(self) -> None:
        if self._download_toast is not None:
            self._download_toast.hide()

    def _show_quick_start(self) -> None:
        QMessageBox.information(
            self,
            "快速开始",
            "1. 打开或拖入视频（mp4 / mov 等）。\n"
            "2. 新建轨迹，框选或点击目标。快速用 Tiny 隔帧，精准用 Small 逐帧。\n"
            "3. 用坐标系菜单设置标定尺或运动平面。\n"
            "4. 在数据表和分图中查看结果，需要时导出 CSV / JSON。\n\n"
            "标准安装包捆绑 Tiny 权重；精准 Small 首次使用时下载。\n"
            "安装包可在帮助菜单检查更新；点「立即更新」会下载并替换当前应用。\n"
            "AI 离面抽检需从源码安装完整依赖。",
        )

    def _show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "快捷键",
            "空格：播放 / 暂停\n"
            "T：开始或取消自动跟踪\n"
            "左右方向键：按底栏步长逐帧移动\n"
            "I / O：设置循环起点 / 终点\n"
            "点击画面：正点选；Shift+点击：负点；拖动：框选",
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 TrackLab",
            f"TrackLab {__version__}\n\n"
            "物理视频分析工具。快速 Tiny 随安装包提供，精准 Small 首次使用时下载。"
            "含手工平面测量。安装包可在应用内检查并安装更新。\n"
            "AI 离面抽检仍为可选能力。\n\n"
            f"项目主页：https://github.com/{GITHUB_REPO}",
        )

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._reposition_download_toast()

    def moveEvent(self, event) -> None:  # noqa: ANN001
        super().moveEvent(event)
        self._reposition_download_toast()

    def _reposition_download_toast(self) -> None:
        if self._download_toast is not None and self._download_toast.isVisible():
            self._download_toast.reposition()

    def changeEvent(self, event) -> None:  # noqa: ANN001
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self._track_window.isVisible():
                self._track_window.raise_()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._first_video_path(event.mimeData().urls()):
            self._hint.set_hover(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: ANN001
        self._hint.set_hover(False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        self._hint.set_hover(False)
        path = self._first_video_path(event.mimeData().urls())
        if path:
            self.open_video(path)
            event.acceptProposedAction()

    def _first_video_path(self, urls) -> Path | None:
        for url in urls:
            path = Path(url.toLocalFile())
            if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
                return path
        return None

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", "", VIDEO_FILTER)
        if path:
            self.open_video(Path(path))

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", PROJECT_FILTER)
        if not path:
            return
        doc = read_track_project(Path(path))
        self._pending_project = doc
        self.open_video(doc.video_path)

    def close_video(self) -> None:
        had_video = self._info is not None
        self._pause()
        self._stop_ai()
        self._stop_assistant()
        self._stop_shake()
        self._stop_depth_audit()
        self._stop_pump()
        self._info = None
        self._index = 0
        self._tracks = []
        self._active_id = None
        self._pending_project = None
        self._shake = None
        self._calibration = CalibrationState()
        self._reset_assistant_state()
        self._pending_rulers = None
        self._pending_plane = None
        self._axis_undo_pushed = False
        self._plane_undo_pushed = False
        self._cal_drawn_before_shake = False
        self._depth_audit = DepthAuditState()
        self._video.clear_track()
        self._video_info.set_info(None)
        self._cal_dialog.hide()
        if getattr(self, "_camera_dialog", None) is not None:
            self._camera_dialog.hide()
        self._clear_history()
        self._video.set_frame(None)
        self._export_track_action.setEnabled(False)
        self._export_csv_action.setEnabled(False)
        self._set_has_video(False)
        self._slider.blockSignals(True)
        self._slider.setRange(0, 0)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._reset_empty_chrome()
        self._refresh_track_ui()
        self.statusBar().showMessage(
            "已关闭视频并释放解码缓存" if had_video else "当前没有打开的视频", 4000
        )

    def open_video(self, path: Path) -> None:
        path = path.expanduser().resolve()
        self._pause()
        self._stop_ai()
        self._stop_assistant()
        self._stop_shake()
        self._stop_depth_audit()
        self._stop_pump()
        self._info = None
        self._index = 0
        self._shake = None
        self._calibration = CalibrationState()
        self._pending_rulers = None
        self._pending_plane = None
        self._axis_undo_pushed = False
        self._plane_undo_pushed = False
        self._cal_drawn_before_shake = False
        if self._pending_project is None:
            self._tracks = []
            self._active_id = None
            self._depth_audit = DepthAuditState()
        self._clear_history()
        self._video.clear_track()
        self._video_info.set_info(None)
        self._cal_dialog.hide()
        if getattr(self, "_camera_dialog", None) is not None:
            self._camera_dialog.hide()
        self._video.set_frame(None)
        self._export_track_action.setEnabled(False)
        self._export_csv_action.setEnabled(False)
        self._set_has_video(False)
        self._slider.setRange(0, 0)
        self._stack.setCurrentWidget(self._hint)
        self._hint.set_status("正在读取视频索引…")
        self.setWindowTitle(f"TrackLab — {path.name}")
        self._set_cache_actions_enabled(True)
        self.statusBar().clearMessage()

        pump = FramePump(path, self)
        pump.opened.connect(self._on_opened)
        pump.ready.connect(self._on_frame_ready)
        pump.failed.connect(self._on_open_failed)
        self._pump = pump
        pump.start()

    def _reset_empty_chrome(self) -> None:
        self._frame_readout.setText("帧 0 / 0")
        self._stack.setCurrentWidget(self._hint)
        self._hint.set_status("")
        self.setWindowTitle("TrackLab")

    def _stop_pump(self) -> None:
        if self._pump is None:
            return
        pump = self._pump
        self._pump = None
        pump.blockSignals(True)
        pump.stop()
        pump.wait(5000)

    def _on_opened(self, info: object) -> None:
        if self.sender() is not self._pump:
            return
        assert isinstance(info, VideoInfo)
        self._info = info
        last = info.frame_count - 1
        self._slider.setRange(0, last)
        self._slider.set_loop_range(0, last)
        self._set_has_video(True)
        self._hint.set_status("")
        if self._pending_project is not None:
            self._tracks = list(self._pending_project.tracks)
            self._active_id = self._pending_project.active_track_id
            self._show_contours = self._pending_project.show_contours
            self._show_prompts = self._pending_project.show_prompts
            self._show_calibration = self._pending_project.show_calibration
            self._calibration = self._pending_project.calibration
            self._depth_audit = getattr(
                self._pending_project, "depth_audit", DepthAuditState()
            )
            self._cal_dialog.set_experimental_correction(
                self._depth_audit.experimental_correction
            )
            self._contour_action.setChecked(self._show_contours)
            self._prompt_action.setChecked(self._show_prompts)
            self._cal_overlay_action.setChecked(self._show_calibration)
            if getattr(self._pending_project, "track_mode", None) is not None:
                self._set_track_mode(self._pending_project.track_mode, announce=False)
            self._restore_assistant_from_project(self._pending_project)
            self._pending_project = None
        elif not self._tracks:
            self._new_track()
            self._reset_assistant_state()
        self._video_info.set_info(info)
        self._refresh_track_ui()
        self.statusBar().showMessage(
            f"{info.path.name}  ·  {info.width}×{info.height}"
            f"  ·  {info.fps:.4g} fps  ·  {info.frame_count} 帧",
            6000,
        )
        self._show_frame(0)

    def _on_open_failed(self, message: str) -> None:
        if self.sender() is not self._pump:
            return
        self._pause()
        self._info = None
        self._pending_project = None
        self._set_has_video(False)
        self._reset_empty_chrome()
        self.statusBar().showMessage(message or "无法打开该视频", 8000)

    def _on_frame_ready(self, index: int, image: QImage) -> None:
        if self.sender() is not self._pump or self._info is None:
            return
        self._stack.setCurrentWidget(self._video)
        self._video.set_frame(image, repaint=False)
        self._apply_track_overlay(index)
        if self._playing and index == self._pending_play_index:
            self._schedule_next_frame()

    def _set_has_video(self, enabled: bool) -> None:
        self._play_btn.setEnabled(enabled)
        self._prev_btn.setEnabled(enabled)
        self._slider.setEnabled(enabled)
        self._stepper.setEnabled(enabled)
        self._loop_btn.setEnabled(enabled)
        self._speed.setEnabled(enabled)
        self._ai_track_action.setEnabled(enabled)
        self._shake_action.setEnabled(enabled)
        self._save_action.setEnabled(enabled)
        self._set_cache_actions_enabled(enabled)
        self._sync_assistant_actions()

    def _set_cache_actions_enabled(self, enabled: bool) -> None:
        self._close_action.setEnabled(enabled)

    def _toggle_play(self) -> None:
        if self._info is None:
            self._open_dialog()
            return
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if self._info is None:
            return
        start, end = self._play_bounds()
        self._playing = True
        self._set_play_icon(True)
        self._deadline = time.perf_counter()
        if self._index >= end:
            self._request_play_frame(start)
        else:
            self._schedule_next_frame()

    def _pause(self) -> None:
        self._playing = False
        self._pending_play_index = None
        self._timer.stop()
        self._set_play_icon(False)

    def _jump_start(self) -> None:
        if self._info is None:
            return
        self._pause()
        self._show_frame(self._play_bounds()[0])

    def _step(self, delta: int) -> None:
        if self._info is None:
            return
        self._pause()
        self._show_frame(self._index + delta)

    def _play_bounds(self) -> tuple[int, int]:
        assert self._info is not None
        if self._loop_btn.isChecked():
            start, end = self._slider.loop_range()
            if end > start:
                return start, end
        return 0, self._info.frame_count - 1

    def _schedule_next_frame(self) -> None:
        assert self._info is not None
        self._deadline += self._info.frame_delay_ms(self._index) / 1000.0 / self._speed_factor
        remaining = self._deadline - time.perf_counter()
        if remaining < -0.25:
            self._deadline = time.perf_counter()
            remaining = 0.0
        self._timer.start(max(0, round(remaining * 1000)))

    def _on_play_tick(self) -> None:
        if not self._playing or self._info is None:
            return
        start, end = self._play_bounds()
        nxt = self._index + 1
        if nxt > end:
            if not self._loop_btn.isChecked():
                self._pause()
                self._show_frame(end)
                return
            nxt = start
        self._request_play_frame(nxt)

    def _request_play_frame(self, index: int) -> None:
        self._pending_play_index = index
        self._show_frame(index)

    def _on_speed_changed(self, text: str) -> None:
        try:
            self._speed_factor = max(0.05, int(text.rstrip("%")) / 100.0)
        except ValueError:
            self._speed_factor = 1.0
        if self._playing:
            self._deadline = time.perf_counter()
            self._schedule_next_frame()

    def _on_loop_toggled(self, enabled: bool) -> None:
        if self._info is None:
            return
        if enabled:
            start, end = self._slider.loop_range()
            self.statusBar().showMessage(
                f"循环播放：帧 {start + 1} – {end + 1}（拖动进度条上方三角调整，I / O 设为当前帧）",
                6000,
            )
            if not start <= self._index <= end:
                self._show_frame(start)
        else:
            self.statusBar().showMessage("循环播放已关闭", 3000)

    def _on_loop_range_changed(self, start: int, end: int) -> None:
        if self._info is None:
            return
        self.statusBar().showMessage(f"循环范围：帧 {start + 1} – {end + 1}", 3000)

    def _mark_loop(self, which: str) -> None:
        if self._info is None:
            return
        start, end = self._slider.loop_range()
        if which == "start":
            self._slider.set_loop_range(self._index, max(end, self._index))
        else:
            self._slider.set_loop_range(min(start, self._index), self._index)

    def _reset_loop_range(self) -> None:
        if self._info is None:
            return
        self._slider.set_loop_range(0, self._info.frame_count - 1)

    def _on_chart_frame_activated(self, index: int) -> None:
        self._pause()
        self._show_frame(index)

    def _on_slider_pressed(self) -> None:
        self._pause()
        self._scrubbing = True
        self._show_frame(self._slider.value())

    def _on_slider_value(self, value: int) -> None:
        if self._info is None:
            return
        if self._scrubbing or not self._playing:
            self._show_frame(value)

    def _on_slider_released(self) -> None:
        self._scrubbing = False
        if self._info is None:
            return
        self._show_frame(self._slider.value())

    def _show_frame(self, index: int) -> None:
        if self._info is None or self._pump is None:
            return
        index = max(0, min(index, self._info.frame_count - 1))
        self._index = index
        if self._slider.value() != index:
            self._slider.blockSignals(True)
            self._slider.setValue(index)
            self._slider.blockSignals(False)
        self._frame_readout.setText(
            f"帧 {index + 1} / {self._info.frame_count}"
            f"   {_fmt_ms(self._info.pts_ms[index])}"
        )
        self._data_panel.highlight_frame(index)
        self._chart_panel.highlight_frame(index)
        self._sync_view_bar()
        self._pump.request(index)

    def _active_layer(self) -> TrackLayer | None:
        return next((t for t in self._tracks if t.track_id == self._active_id), None)

    def _copy_history(self) -> tuple[list[TrackLayer], str | None, CalibrationState]:
        return (
            [TrackLayer.from_dict(layer.to_dict()) for layer in self._tracks],
            self._active_id,
            CalibrationState.from_dict(self._calibration.to_dict()),
        )

    def _clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._sync_undo_actions()

    def _push_undo(self) -> None:
        if self._undoing:
            return
        self._undo_stack.append(self._copy_history())
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._sync_undo_actions()

    def _restore_history(
        self, snapshot: tuple[list[TrackLayer], str | None, CalibrationState]
    ) -> None:
        tracks, active_id, calibration = snapshot
        self._tracks = [TrackLayer.from_dict(layer.to_dict()) for layer in tracks]
        self._active_id = active_id
        self._calibration = CalibrationState.from_dict(calibration.to_dict())
        for layer in self._tracks:
            if layer.status == "running":
                layer.status = "done" if layer.result is not None else "idle"
        self._refresh_track_ui()

    def _sync_undo_actions(self) -> None:
        busy = self._ai_worker is not None
        if getattr(self, "_undo_action", None) is None:
            return
        self._undo_action.setEnabled(bool(self._undo_stack) and not busy)
        self._redo_action.setEnabled(bool(self._redo_stack) and not busy)

    def _undo(self) -> None:
        if not self._undo_stack or self._ai_worker is not None:
            return
        self._undoing = True
        self._redo_stack.append(self._copy_history())
        self._restore_history(self._undo_stack.pop())
        self._undoing = False
        self._sync_undo_actions()
        self.statusBar().showMessage("已撤销", 3000)

    def _redo(self) -> None:
        if not self._redo_stack or self._ai_worker is not None:
            return
        self._undoing = True
        self._undo_stack.append(self._copy_history())
        self._restore_history(self._redo_stack.pop())
        self._undoing = False
        self._sync_undo_actions()
        self.statusBar().showMessage("已重做", 3000)

    def _new_track(self, *_args, record: bool = True) -> None:
        if record:
            self._push_undo()
        index = len(self._tracks)
        layer = TrackLayer(
            track_id=new_track_id(),
            name=f"轨迹 {index + 1}",
            color=TRACK_COLORS[index % len(TRACK_COLORS)],
            seed_frame=self._index,
        )
        self._tracks.append(layer)
        self._active_id = layer.track_id
        self._refresh_track_ui()
        self._list_panel.set_hint("Control 拖动框选目标，Shift+Control 点击加点")

    def _delete_track(self) -> None:
        layer = self._active_layer()
        if layer is None:
            return
        self._push_undo()
        self._tracks = [t for t in self._tracks if t.track_id != layer.track_id]
        self._active_id = self._tracks[0].track_id if self._tracks else None
        self._refresh_track_ui()

    def _select_track(self, track_id: str) -> None:
        if track_id == self._active_id:
            return
        self._active_id = track_id
        self._refresh_track_ui()

    def _toggle_track_visible(self, track_id: str, visible: bool) -> None:
        for layer in self._tracks:
            if layer.track_id == track_id:
                break
        else:
            return
        if layer.visible == visible:
            return
        self._push_undo()
        layer.visible = visible
        self._apply_track_overlay()
        self._view_bar.set_tracks(self._tracks, self._active_id)

    def _toggle_active_visible(self, visible: bool) -> None:
        if self._active_id is None:
            return
        self._toggle_track_visible(self._active_id, visible)
        self._list_panel.set_tracks(self._tracks, self._active_id)

    def _toggle_track_window(self, *_args) -> None:
        self._set_track_window_visible(True)
        self._track_window.raise_()
        self._track_window.activateWindow()

    def _set_track_window_visible(self, visible: bool) -> None:
        if visible:
            self._track_window.show()
            self._track_window.raise_()
        else:
            self._track_window.hide()

    def _on_track_window_visible(self, visible: bool) -> None:
        if getattr(self, "_track_window_action", None) is None:
            return
        self._track_window_action.blockSignals(True)
        self._track_window_action.setChecked(visible)
        self._track_window_action.blockSignals(False)

    def _on_chart_dock_visible(self, visible: bool) -> None:
        if getattr(self, "_chart_action", None) is None:
            return
        self._chart_action.blockSignals(True)
        self._chart_action.setChecked(visible)
        self._chart_action.blockSignals(False)

    def _on_data_dock_visible(self, visible: bool) -> None:
        if getattr(self, "_table_action", None) is None:
            return
        self._table_action.blockSignals(True)
        self._table_action.setChecked(visible)
        self._table_action.blockSignals(False)

    def _on_assistant_window_visible(self, visible: bool) -> None:
        for action in (
            getattr(self, "_ai_panel_action", None),
            getattr(self, "_assistant_window_action", None),
        ):
            if action is None:
                continue
            action.blockSignals(True)
            action.setChecked(visible)
            action.blockSignals(False)

    def _on_position_edited(self, x: float, y: float) -> None:
        layer = self._active_layer()
        if layer is None or layer.result is None:
            return
        px, py = x, y
        if self._calibration.applies_transform():
            pixel = self._calibration.world_to_pixel(x, y)
            px, py = pixel.x, pixel.y
        if self._shake_enabled and self._shake is not None:
            dx, dy = self._shake.offset(self._index)
            px += dx
            py += dy
        self._push_undo()
        layer.result = apply_manual_override(layer.result, self._index, Point2D(px, py))
        self._refresh_track_ui()
        self.statusBar().showMessage(
            f"已修正第 {self._index + 1} 帧 → ({x:.1f}, {y:.1f})", 4000
        )

    def _step_track_point(self, direction: int) -> None:
        layer = self._active_layer()
        if layer is None or layer.result is None:
            return
        frames = [p.frame for p in layer.result.points if p.visible]
        if not frames:
            return
        if direction > 0:
            nxt = next((f for f in frames if f > self._index), None)
        else:
            nxt = next((f for f in reversed(frames) if f < self._index), None)
        if nxt is None:
            return
        if self._info is not None:
            self._show_frame(nxt)
            return
        self._index = nxt
        self._data_panel.highlight_frame(nxt)
        self._chart_panel.highlight_frame(nxt)
        self._sync_view_bar()

    def _rename_track(self, track_id: str, name: str) -> None:
        cleaned = name.strip()
        if not cleaned:
            return
        for layer in self._tracks:
            if layer.track_id == track_id:
                if layer.name == cleaned:
                    return
                self._push_undo()
                layer.name = cleaned
                self._view_bar.set_tracks(self._tracks, self._active_id)
                return

    def _on_video_click(self, x: float, y: float) -> None:
        # prompted signal handles seed / correction; keep slot for tests.
        return

    def _on_prompted(self, x: float, y: float, kind: str) -> None:
        if self._info is None:
            return
        self._push_undo()
        if self._active_layer() is None:
            self._new_track(record=False)
        layer = self._active_layer()
        assert layer is not None
        if layer.result is not None and kind == "positive":
            layer.result = apply_manual_override(layer.result, self._index, Point2D(x, y))
            layer.prompts.append(
                TrackPrompt(frame=self._index, kind=PromptKind.POSITIVE, x=x, y=y)
            )
            self.statusBar().showMessage(
                f"已修正第 {self._index + 1} 帧 → ({x:.1f}, {y:.1f})，可再按 T 从该帧重跟踪",
                5000,
            )
        else:
            prompt_kind = PromptKind.NEGATIVE if kind == "negative" else PromptKind.POSITIVE
            layer.prompts.append(
                TrackPrompt(frame=self._index, kind=prompt_kind, x=x, y=y)
            )
            layer.seed_frame = self._index
            self.statusBar().showMessage(
                f"{'负' if prompt_kind is PromptKind.NEGATIVE else '正'}点 "
                f"({x:.1f}, {y:.1f}) @ 帧 {self._index + 1}",
                4000,
            )
        self._refresh_track_ui()

    def _on_boxed(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self._info is None:
            return
        self._push_undo()
        if self._active_layer() is None:
            self._new_track(record=False)
        layer = self._active_layer()
        assert layer is not None
        layer.prompts.append(
            TrackPrompt(
                frame=self._index,
                kind=PromptKind.BOX,
                x=x0,
                y=y0,
                x2=x1,
                y2=y1,
            )
        )
        layer.seed_frame = self._index
        self._refresh_track_ui()
        self.statusBar().showMessage(
            f"已添加框选 ({x0:.0f},{y0:.0f})–({x1:.0f},{y1:.0f})", 4000
        )

    def _start_or_cancel_ai_track(self, *_args) -> None:
        if self._info is None:
            return
        if self._ai_worker is not None:
            self._cancel_ai_track()
            return
        layer = self._active_layer()
        if layer is None:
            self._new_track()
            layer = self._active_layer()
        assert layer is not None
        if not layer.prompts:
            self._list_panel.set_hint("请先 Control 拖动框选目标，或 Shift+Control 点击加点")
            self.statusBar().showMessage("请先框选或加点，再开始跟踪", 5000)
            return
        if not self._ensure_sam_runtime():
            return
        self._push_undo()
        seed_prompt = next(
            (p for p in reversed(layer.prompts) if p.kind != PromptKind.NEGATIVE),
            layer.prompts[-1],
        )
        seed = (seed_prompt.x, seed_prompt.y)
        start = layer.seed_frame
        if layer.prompts:
            start = layer.prompts[-1].frame
        end = self._track_end_frame()
        layer.status = "running"
        self._refresh_track_ui()
        self._stop_shake()
        thread = self._ensure_sam_thread()
        _, worker = run_track_in_thread(
            self._info.path,
            seed,
            on_progress=self._on_ai_progress,
            on_finished=self._on_ai_finished,
            on_failed=self._on_ai_failed,
            track_id=layer.track_id,
            start_frame=start,
            end_frame=end,
            prompts=layer.prompts,
            track_mode=self._track_mode,
            thread=thread,
        )
        self._ai_worker = worker
        self._cancel_ai_action.setEnabled(True)
        self._ai_track_action.setText("取消 SAM 跟踪")
        self._list_panel.set_running(True)
        self._sync_undo_actions()
        kind = "快速Tiny" if self._track_mode is TrackMode.FAST else "精准Small"
        device = self._sam_device_label()
        self.statusBar().showMessage(
            f"跟踪 第 {start + 1}–{end + 1} 帧 · {kind} · {device}"
        )

    def _ensure_sam_runtime(self) -> bool:
        """Prompt once for Apache-2.0 weights. Never silently fall back to color blobs."""
        try:
            import sam2  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            QMessageBox.warning(
                self,
                "未安装 SAM 2",
                "桌面跟踪需要 PyTorch 与 SAM 2。\n请执行：pip install -r requirements-ai.txt",
            )
            return False
        from ai.model_manager import ModelNotAvailable, checkpoint_path, ensure_checkpoint

        if self._download_worker is not None:
            toast = self._ensure_download_toast()
            toast.show()
            toast.reposition()
            self.statusBar().showMessage("正在下载权重，请稍候…", 4000)
            return False
        spec = spec_for_mode(self._track_mode)
        try:
            ensure_checkpoint(spec, download=False)
            return True
        except ModelNotAvailable:
            path = checkpoint_path(spec)
            label = "SAM 2.1 Tiny" if spec.model_id.endswith("tiny") else "SAM 2.1 Small"
            reply = QMessageBox.question(
                self,
                f"下载 {label}",
                (
                    f"首次使用{label}需要下载官方权重（Apache-2.0）。\n\n"
                    f"保存到：{path}\n"
                    f"来源：{spec.url}\n"
                    f"Hugging Face：{spec.hf_id}\n"
                    f"SHA-256：{spec.sha256}\n\n"
                    "确认下载？"
                ),
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
            self._start_checkpoint_download(spec)
            return False

    def _ensure_download_toast(self) -> DownloadToast:
        if self._download_toast is None:
            self._download_toast = DownloadToast(self)
            self._download_toast.cancelled.connect(self._on_download_toast_cancelled)
        return self._download_toast

    def _on_download_toast_cancelled(self) -> None:
        if self._self_update_worker is not None:
            self._cancel_self_update()
            return
        self._cancel_checkpoint_download()

    def _start_checkpoint_download(self, spec) -> None:  # noqa: ANN001
        if self._download_worker is not None or self._self_update_worker is not None:
            return
        toast = self._ensure_download_toast()
        toast.set_download(spec)
        toast.show()
        toast.reposition()
        self._download_resume_track = True
        thread, worker = run_checkpoint_download(
            spec,
            on_progress=toast.set_progress,
            on_stage=toast.set_stage,
            on_finished=self._on_checkpoint_downloaded,
            on_failed=self._on_checkpoint_download_failed,
            on_cancelled=self._on_checkpoint_download_cancelled,
        )
        self._download_thread = thread
        self._download_worker = worker
        thread.start()
        label = "SAM 2.1 Tiny" if spec.model_id.endswith("tiny") else "SAM 2.1 Small"
        self.statusBar().showMessage(f"正在下载 {label} 权重…")

    def _cancel_checkpoint_download(self) -> None:
        if self._download_worker is not None:
            self._download_worker.cancel()
            self.statusBar().showMessage("正在取消下载…", 3000)

    def _release_download_thread(self) -> None:
        thread = self._download_thread
        self._download_thread = None
        self._download_worker = None
        if thread is not None:
            thread.quit()
            thread.wait(8000)

    def _stop_checkpoint_download(self) -> None:
        self._download_resume_track = False
        if self._download_worker is not None:
            self._download_worker.cancel()
        self._release_download_thread()
        if self._download_toast is not None:
            self._download_toast.hide()

    def _on_checkpoint_downloaded(self, _path: str) -> None:
        resume = self._download_resume_track
        self._download_resume_track = False
        self._release_download_thread()
        toast = self._download_toast
        if toast is not None:
            toast.set_finished()
            QTimer.singleShot(1200, toast.hide)
        self.statusBar().showMessage("权重已就绪", 4000)
        if resume:
            QTimer.singleShot(0, self._start_or_cancel_ai_track)

    def _on_checkpoint_download_failed(self, message: str) -> None:
        self._download_resume_track = False
        self._release_download_thread()
        if self._download_toast is not None:
            self._download_toast.hide()
        QMessageBox.warning(self, "权重下载失败", message)

    def _on_checkpoint_download_cancelled(self) -> None:
        self._download_resume_track = False
        self._release_download_thread()
        if self._download_toast is not None:
            self._download_toast.hide()
        self.statusBar().showMessage("已取消下载", 4000)

    def _cancel_ai_track(self, *_args) -> None:
        if self._ai_worker is not None:
            self._ai_worker.cancel()
            self.statusBar().showMessage("正在取消跟踪…", 3000)

    def _stop_ai(self) -> None:
        if self._ai_worker is not None:
            self._ai_worker.cancel()
        self._ai_worker = None
        self._cancel_ai_action.setEnabled(False)
        self._ai_track_action.setText("SAM 自动跟踪")
        self._list_panel.set_running(False)
        self._sync_undo_actions()
        if getattr(self, "_assistant_panel", None) is not None:
            self._sync_assistant()

    def _ensure_sam_thread(self) -> QThread:
        if self._sam_thread is None:
            self._sam_thread = QThread(self)
            self._sam_thread.setObjectName("samRuntime")
        if not self._sam_thread.isRunning():
            self._sam_thread.start()
        return self._sam_thread

    def _stop_sam_thread(self) -> None:
        if self._sam_thread is None:
            return
        self._sam_thread.quit()
        self._sam_thread.wait(5000)
        self._sam_thread = None

    def _track_end_frame(self) -> int:
        assert self._info is not None
        _start, end = self._slider.loop_range()
        last = self._info.frame_count - 1
        if end > 0:
            return min(end, last)
        return last

    def _sam_device_label(self) -> str:
        try:
            from ai.model_manager import select_device

            return select_device()
        except Exception:  # noqa: BLE001
            return "cpu"

    def _set_track_mode(self, mode: TrackMode, *, announce: bool = True) -> None:
        self._track_mode = mode
        if getattr(self, "_fast_track_action", None) is not None:
            self._fast_track_action.setChecked(mode is TrackMode.FAST)
            self._precise_track_action.setChecked(mode is TrackMode.PRECISE)
        if not announce:
            return
        layer = self._active_layer()
        interpolated = bool(
            layer is not None
            and layer.result is not None
            and any(point.interpolated for point in layer.result.points)
        )
        if mode is TrackMode.PRECISE and interpolated:
            self.statusBar().showMessage("已切换精准（Small）。请再按 T 重跟踪。", 5000)
        elif mode is TrackMode.FAST:
            self.statusBar().showMessage("已切换快速预览（Tiny，隔帧插值）。", 4000)
        else:
            self.statusBar().showMessage("已切换精准分析（Small，逐帧）。", 4000)

    def _stop_shake(self) -> None:
        if self._shake_worker is not None:
            self._shake_worker.cancel()
        if self._shake_thread is not None:
            self._shake_thread.quit()
            self._shake_thread.wait(5000)
        self._shake_thread = None
        self._shake_worker = None

    def _run_shake_compensation(self, *_args) -> None:
        if self._info is None:
            return
        if self._ai_worker is not None:
            self.statusBar().showMessage("请等待 SAM 跟踪完成后再做背景补偿", 4000)
            return
        self._start_shake()

    def _maybe_start_shake(self) -> None:
        if self._info is None or self._shake is not None:
            return
        if self._ai_worker is not None or self._shake_worker is not None:
            return
        self._start_shake()

    def _start_shake(self) -> None:
        if self._info is None:
            return
        self._stop_shake()
        thread, worker = run_shake_in_thread(
            self._info.path,
            on_progress=self._on_shake_progress,
            on_finished=self._on_shake_finished,
            on_failed=self._on_shake_failed,
        )
        self._shake_thread = thread
        self._shake_worker = worker
        self.statusBar().showMessage("正在根据四角参照点估计镜头抖动…")
        thread.start()

    def _on_shake_progress(self, event: ProgressEvent) -> None:
        if self._ai_worker is not None:
            return
        self.statusBar().showMessage(
            f"背景补偿 {event.current} / {event.total}", 800
        )

    def _migrate_calibration_after_shake(self) -> None:
        """Rulers drawn before shake existed were stored in raw video pixels."""
        if self._shake is None or not self._cal_drawn_before_shake:
            return
        self._cal_drawn_before_shake = False
        dx, dy = self._shake.offset(0)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return
        for ruler in self._calibration.rulers:
            ruler.a = Point2D(ruler.a.x - dx, ruler.a.y - dy)
            ruler.b = Point2D(ruler.b.x - dx, ruler.b.y - dy)
        if self._calibration.plane is not None:
            self._calibration.plane.corners = [
                Point2D(point.x - dx, point.y - dy) for point in self._calibration.plane.corners
            ]
        if self._calibration.frame.origin_x is not None:
            self._calibration.frame.origin_x -= dx
        if self._calibration.frame.origin_y is not None:
            self._calibration.frame.origin_y -= dy
        if self._calibration.mode is CalibrationMode.PLANAR:
            shift = (dx * dx + dy * dy) ** 0.5
            if shift >= SHAKE_INVALIDATE_PX:
                self._calibration.camera_moved = True
            self._calibration.validate()

    def _on_shake_finished(self, shake: ShakeCompensation) -> None:
        self._stop_shake()
        if self._info is None:
            return
        self._shake = shake
        self._migrate_calibration_after_shake()
        self._refresh_track_ui()
        n_frames = len(shake.dx)
        n_anchors = len(shake.anchors_at(0)) if shake.anchors else 0
        self.statusBar().showMessage(
            f"背景补偿完成：已处理 {n_frames} 帧，锁定 {n_anchors} 个四角参照点。"
            "数据表、分图和 CSV 使用补偿后坐标，画面上的轨迹仍与原视频对齐。",
            8000,
        )

    def _on_shake_failed(self, message: str) -> None:
        self._stop_shake()
        self.statusBar().showMessage(message or "背景补偿失败", 8000)

    def _analysis_result(self, layer: TrackLayer | None) -> TrackResult | None:
        if layer is None or layer.result is None:
            return None
        if self._shake_enabled and self._shake is not None:
            return compensate_result(layer.result, self._shake)
        return layer.result

    def _on_ai_progress(self, event: ProgressEvent) -> None:
        self._list_panel.set_progress(event.current, event.total)
        self.statusBar().showMessage(
            f"SAM 跟踪 {event.current} / {event.total}", 800
        )

    def _on_ai_finished(self, result: TrackResult) -> None:
        track_id = getattr(result, "track_id", None) or (
            self._ai_worker.track_id if self._ai_worker is not None else self._active_id
        )
        layer = next((t for t in self._tracks if t.track_id == track_id), self._active_layer())
        if layer is not None:
            incoming_start = min((p.frame for p in result.points), default=self._index)
            if layer.result is not None:
                merged = merge_track_points(
                    layer.result.points, result.points, incoming_start
                )
                result = TrackResult(
                    clip_id=result.clip_id,
                    points=merged,
                    confidence=result.confidence,
                    failure_reason=result.failure_reason,
                    model_name=result.model_name,
                    model_version=result.model_version,
                    elapsed_s=result.elapsed_s,
                )
                layer.contours = {
                    frame: pts
                    for frame, pts in layer.contours.items()
                    if frame < incoming_start
                }
            layer.result = result
            layer.status = "done"
            contours = getattr(result, "contours", None)
            if isinstance(contours, dict):
                layer.contours.update(contours)
        self._export_track_action.setEnabled(True)
        self._export_csv_action.setEnabled(True)
        self._stop_ai()
        self._refresh_track_ui()
        n = sum(1 for p in result.points if p.visible)
        self.statusBar().showMessage(
            f"跟踪完成：{n} 个可见点（{result.model_name} {result.elapsed_s:.2f}s）。"
            "Shift+Control 点击可修正当前帧，Control 拖动可再框选后重跟踪。",
            8000,
        )
        self._maybe_start_shake()

    def _on_ai_failed(self, message: str) -> None:
        track_id = self._ai_worker.track_id if self._ai_worker is not None else self._active_id
        layer = next((t for t in self._tracks if t.track_id == track_id), self._active_layer())
        if layer is not None:
            layer.status = "error"
        self._stop_ai()
        self._refresh_track_ui()
        self.statusBar().showMessage(message or "SAM 跟踪失败", 8000)
        if message:
            QMessageBox.warning(self, "SAM 2 跟踪失败", message)

    def _apply_track_overlay(self, index: int | None = None) -> None:
        overlays: list[OverlayTrack] = []
        seed = None
        frame = self._index if index is None else index
        for layer in self._tracks:
            if not layer.visible:
                continue
            points = []
            if layer.result is not None:
                points = [
                    (p.frame, p.x, p.y, p.visible, p.confidence) for p in layer.result.points
                ]
            contour = layer.contours.get(frame, [])
            overlays.append(
                OverlayTrack(
                    points=points,
                    color=layer.color,
                    active=layer.track_id == self._active_id,
                    contour=contour,
                    prompts=layer.prompts,
                )
            )
            if (
                layer.track_id == self._active_id
                and layer.prompts
                and layer.result is None
            ):
                last = layer.prompts[-1]
                seed = (last.x, last.y)
        self._video.set_overlays(overlays, index=frame, seed=seed)
        self._video.set_display_options(
            contours=self._show_contours, prompts=self._show_prompts
        )
        anchors: list[tuple[float, float]] = []
        if self._show_anchors and self._shake is not None:
            anchors = self._shake.anchors_at(frame)
        self._video.set_anchors(anchors)
        self._sync_cal_overlay()

    def _refresh_track_ui(self) -> None:
        self._list_panel.set_tracks(self._tracks, self._active_id)
        self._view_bar.set_tracks(self._tracks, self._active_id)
        layer = self._active_layer()
        analyzed = self._analysis_result(layer)
        samples = series_for_result(
            analyzed,
            self._info,
            calibration=self._calibration,
            velocity_step=self._chart_panel.velocity_step,
            depth_audit=self._depth_audit,
        )
        self._data_panel.set_units(
            self._calibration.position_unit, self._calibration.speed_unit
        )
        self._chart_panel.set_units(
            self._calibration.position_unit, self._calibration.speed_unit
        )
        if layer is None:
            self._data_panel.set_layer(None, self._info)
        else:
            self._data_panel.set_samples(samples)
        self._chart_panel.set_samples(samples)
        self._view_bar.set_units(self._calibration.position_unit)
        self._sync_view_bar()
        has_result = layer is not None and layer.result is not None
        self._export_track_action.setEnabled(bool(has_result))
        self._export_csv_action.setEnabled(bool(has_result))
        self._apply_track_overlay()
        if self._info is not None:
            self._chart_panel.highlight_frame(self._index)
            self._data_panel.highlight_frame(self._index)
        self._sync_assistant()

    def _sync_view_bar(self) -> None:
        layer = self._active_layer()
        samples = series_for_result(
            self._analysis_result(layer),
            self._info,
            calibration=self._calibration,
            velocity_step=self._chart_panel.velocity_step,
            depth_audit=self._depth_audit,
        )
        sample = sample_at_frame(samples, self._index)
        self._view_bar.set_sample(sample)
        if (
            sample is not None
            and sample.visible
            and self._calibration.active
        ):
            analyzed = self._analysis_result(layer)
            point = None if analyzed is None else next(
                (p for p in analyzed.points if p.frame == self._index), None
            )
            if point is not None and self._calibration.is_extrapolated(point.x, point.y):
                self.statusBar().showMessage("超出双尺覆盖区域，结果为外推值", 800)

    def _toggle_overlays(self, *_args) -> None:
        self._show_contours = not self._show_contours
        self._contour_action.setChecked(self._show_contours)
        self._apply_track_overlay()

    def _on_display_toggled(self) -> None:
        self._show_contours = self._contour_action.isChecked()
        self._show_prompts = self._prompt_action.isChecked()
        self._apply_track_overlay()

    def _on_shake_apply_toggled(self, checked: bool) -> None:
        self._shake_enabled = checked
        if getattr(self, "_shake_apply_action", None) is not None:
            self._shake_apply_action.blockSignals(True)
            self._shake_apply_action.setChecked(checked)
            self._shake_apply_action.blockSignals(False)
        self._list_panel.set_shake_enabled(checked)
        self._refresh_track_ui()

    def _on_anchor_toggled(self, checked: bool) -> None:
        self._show_anchors = checked
        self._apply_track_overlay()

    def _restore_layout(self) -> None:
        self._workspace.restore_default()
        if getattr(self, "_chart_action", None) is not None:
            self._chart_action.setChecked(True)
        if getattr(self, "_table_action", None) is not None:
            self._table_action.setChecked(True)
        self._set_assistant_visible(False)
        self._toggle_track_window()

    def _reset_zoom(self, *_args) -> None:
        self._video.reset_zoom()

    def _on_zoom_changed(self, zoom: float) -> None:
        self._zoom_readout.setText(f"{zoom * 100:.0f}%")

    def _clear_cache(self, *_args) -> None:
        freed = self._pump.clear_cache() if self._pump is not None else 0
        self._video.clear_scaled_cache()
        if freed:
            self.statusBar().showMessage(
                f"已释放解码缓存 {freed / (1024 * 1024):.1f} MB", 4000
            )
        else:
            self.statusBar().showMessage("没有可清理的解码缓存", 4000)

    def _export_track(self) -> None:
        layer = self._active_layer()
        if layer is None or layer.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出轨迹 JSON", "", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(
            json.dumps(layer.result.to_dict(), indent=2), encoding="utf-8"
        )
        self.statusBar().showMessage(f"已保存轨迹 {path}", 4000)

    def _export_csv(self) -> None:
        layer = self._active_layer()
        if layer is None or layer.result is None or self._info is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出轨迹 CSV", "", "CSV (*.csv)")
        if not path:
            return
        dest = Path(path)
        if dest.suffix.lower() != ".csv":
            dest = dest.with_suffix(".csv")
        analyzed = self._analysis_result(layer)
        export_track_csv(
            dest,
            analyzed if analyzed is not None else layer.result,
            self._info,
            calibration=self._calibration,
            velocity_step=self._chart_panel.velocity_step,
            depth_audit=self._depth_audit,
        )
        self.statusBar().showMessage(f"已导出 CSV {dest}", 4000)

    def _save_project(self) -> None:
        if self._info is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存项目", "", "TrackLab 项目 (*.json)"
        )
        if not path:
            return
        dest = Path(path)
        if dest.suffix.lower() != ".json":
            dest = dest.with_suffix(".json")
        active = self._active_layer()
        spec = spec_for_mode(self._track_mode)
        write_track_project(
            dest,
            self._info.path,
            None if active is None else active.result,
            tracks=self._tracks,
            active_track_id=self._active_id,
            model_name=spec.model_id,
            model_version=spec.version,
            model_license=spec.license,
            show_contours=self._show_contours,
            show_prompts=self._show_prompts,
            show_calibration=self._show_calibration,
            track_mode=self._track_mode,
            calibration=self._calibration,
            assistant=self._assistant_state,
            depth_audit=self._depth_audit,
        )
        self.statusBar().showMessage(f"已保存项目 {dest}", 4000)

    def _set_play_icon(self, playing: bool) -> None:
        self._play_btn.setIcon(pause_icon() if playing else play_icon())

    def _shake_offset(self, frame: int | None = None) -> tuple[float, float]:
        if self._shake is None:
            return 0.0, 0.0
        return self._shake.offset(self._index if frame is None else frame)

    def _video_to_stable(self, x: float, y: float, frame: int | None = None) -> tuple[float, float]:
        dx, dy = self._shake_offset(frame)
        return x - dx, y - dy

    def _stable_to_video(self, x: float, y: float, frame: int | None = None) -> tuple[float, float]:
        dx, dy = self._shake_offset(frame)
        return x + dx, y + dy

    def _exit_interaction(self) -> None:
        self._video.set_interaction_mode(MODE_TRACK)
        self._pending_rulers = None
        self._pending_plane = None
        self._axis_undo_pushed = False
        self._plane_undo_pushed = False
        self._axis_rotate_base = None
        if self._ruler_btn is not None:
            self._ruler_btn.setChecked(False)
        if self._axis_btn is not None:
            self._axis_btn.setChecked(False)
        self._sync_cal_overlay()

    def _start_ruler_tool(self, *_args) -> None:
        if self._cal_dialog.mode() is CalibrationMode.PLANAR:
            self._start_plane_tool()
            return
        if self.sender() is self._ruler_btn and self._ruler_btn is not None:
            if not self._ruler_btn.isChecked():
                self._exit_interaction()
                self._cal_dialog.hide()
                return
        if self._info is None:
            if self._ruler_btn is not None:
                self._ruler_btn.setChecked(False)
            return
        if self._axis_btn is not None:
            self._axis_btn.setChecked(False)
        if self._ruler_btn is not None:
            self._ruler_btn.setChecked(True)
        self._pending_plane = None
        self._pending_rulers = []
        self._video.set_interaction_mode(MODE_RULER)
        self._cal_dialog.show()
        self._cal_dialog.raise_()
        hint = (
            "拖出标定尺，默认 1.000 m。"
            if self._cal_dialog.mode() is CalibrationMode.UNIFORM
            else "先拖出近尺，再拖出远尺。两尺应尽量平行。"
        )
        self.statusBar().showMessage(hint + " Esc 取消当前步骤。")
        self._cal_dialog.set_status(hint)
        self._sync_cal_overlay()

    def _start_plane_menu(self, *_args) -> None:
        self._cal_dialog.set_mode(CalibrationMode.PLANAR)
        self._start_plane_tool()

    def _start_plane_tool(self, *_args) -> None:
        if self.sender() is self._ruler_btn and self._ruler_btn is not None:
            if not self._ruler_btn.isChecked():
                self._exit_interaction()
                self._cal_dialog.hide()
                return
        if self._info is None:
            if self._ruler_btn is not None:
                self._ruler_btn.setChecked(False)
            return
        if self._axis_btn is not None:
            self._axis_btn.setChecked(False)
        if self._ruler_btn is not None:
            self._ruler_btn.setChecked(True)
        self._cal_dialog.set_mode(CalibrationMode.PLANAR)
        self._pending_rulers = None
        existing = self._calibration.plane
        if existing is not None and len(existing.corners) >= 4:
            self._pending_plane = None
            self._cal_dialog.set_plane_size(existing.width_m, existing.height_m)
            hint = "可拖动角点微调，或点重画重新点选。确认后点击应用。"
        else:
            self._pending_plane = []
            hint = self._cal_dialog.plane_step_hint(0)
        self._video.set_interaction_mode(MODE_PLANE)
        self._cal_dialog.show()
        self._cal_dialog.raise_()
        self._cal_dialog.set_status(hint)
        self.statusBar().showMessage(hint + " Esc 取消。")
        self._sync_cal_overlay()

    def _start_axis_tool(self, *_args) -> None:
        if self.sender() is self._axis_btn and self._axis_btn is not None:
            if not self._axis_btn.isChecked():
                self._exit_interaction()
                return
        if self._info is None:
            if self._axis_btn is not None:
                self._axis_btn.setChecked(False)
            return
        if self._ruler_btn is not None:
            self._ruler_btn.setChecked(False)
        if self._axis_btn is not None:
            self._axis_btn.setChecked(True)
        self._pending_rulers = None
        self._pending_plane = None
        self._cal_dialog.hide()
        created = self._ensure_default_axes()
        self._video.set_interaction_mode(MODE_AXIS)
        self._sync_cal_overlay()
        if created:
            self._refresh_track_ui()
        self.statusBar().showMessage(
            "拖动原点移动，拖动轴尖弯箭头旋转。按住 Shift 以 90° 对齐。Esc 结束编辑。"
        )

    def _start_origin_only(self, *_args) -> None:
        self._start_axis_tool()

    def _start_axis_direction(self, *_args) -> None:
        self._start_axis_tool()

    def _ensure_default_axes(self) -> bool:
        if self._info is None:
            return False
        self._show_calibration = True
        self._cal_overlay_action.setChecked(True)
        self._video.set_show_calibration(True)
        if self._calibration.frame.origin is not None:
            return False
        self._push_undo()
        cx = self._info.width / 2.0
        cy = self._info.height / 2.0
        sx, sy = self._video_to_stable(cx, cy)
        self._calibration.frame.origin_x = sx
        self._calibration.frame.origin_y = sy
        if self._shake is None:
            self._cal_drawn_before_shake = True
        return True

    def _on_interaction_cancelled(self) -> None:
        self._exit_interaction()
        self._cal_dialog.hide()
        self.statusBar().showMessage("已取消当前标定步骤", 3000)

    def _on_cal_mode_changed(self, _mode: str) -> None:
        if self._pending_rulers is not None:
            self._pending_rulers = []
        if self._pending_plane is not None:
            self._pending_plane = []
        if self._cal_dialog.mode() is CalibrationMode.PLANAR:
            self._start_plane_tool()
            return
        if self._video.interaction_mode() in {MODE_RULER, MODE_PLANE}:
            self._start_ruler_tool()

    def _on_cal_lengths(self, near_m: float, far_m: float) -> None:
        if not self._pending_rulers:
            return
        self._pending_rulers[0].length_m = near_m
        if len(self._pending_rulers) > 1:
            self._pending_rulers[1].length_m = far_m
        self._sync_cal_overlay()

    def _on_cal_overlay_toggled(self, checked: bool) -> None:
        self._show_calibration = checked
        self._video.set_show_calibration(checked)

    def _on_plane_size_changed(self, _width_m: float, _height_m: float) -> None:
        self._sync_cal_overlay()

    def _redraw_rulers(self) -> None:
        if self._cal_dialog.mode() is CalibrationMode.PLANAR:
            self._pending_plane = []
            self._video.set_interaction_mode(MODE_PLANE)
            if self._ruler_btn is not None:
                self._ruler_btn.setChecked(True)
            self._cal_dialog.set_status(self._cal_dialog.plane_step_hint(0))
            self._sync_cal_overlay()
            return
        self._pending_rulers = []
        self._video.set_interaction_mode(MODE_RULER)
        if self._ruler_btn is not None:
            self._ruler_btn.setChecked(True)
        self._cal_dialog.set_status("重新拖出标定尺。")
        self._sync_cal_overlay()

    def _swap_pending_rulers(self) -> None:
        target = self._pending_rulers if self._pending_rulers else self._calibration.rulers
        if len(target) < 2:
            self._cal_dialog.set_status("需要两把尺才能交换近/远。", error=True)
            return
        if self._pending_rulers:
            self._pending_rulers[0].role, self._pending_rulers[1].role = (
                RulerRole.FAR,
                RulerRole.NEAR,
            )
            self._pending_rulers.reverse()
            near_m, far_m = self._cal_dialog.lengths()
            self._cal_dialog.set_lengths(far_m, near_m)
        else:
            self._push_undo()
            self._calibration.swap_near_far()
            self._refresh_track_ui()
        self._sync_cal_overlay()

    def _on_ruler_drawn(self, x0: float, y0: float, x1: float, y1: float) -> None:
        ax, ay = self._video_to_stable(x0, y0)
        bx, by = self._video_to_stable(x1, y1)
        near_m, far_m = self._cal_dialog.lengths()
        if self._pending_rulers is None:
            self._pending_rulers = []
        if self._cal_dialog.mode() is CalibrationMode.UNIFORM:
            self._pending_rulers = [
                RulerSegment(Point2D(ax, ay), Point2D(bx, by), near_m, RulerRole.SINGLE)
            ]
            self._cal_dialog.set_status("已画单尺。确认长度后点击应用。")
            self.statusBar().showMessage("标定尺已画出，在面板中应用或重画。", 4000)
        elif not self._pending_rulers:
            self._pending_rulers = [
                RulerSegment(Point2D(ax, ay), Point2D(bx, by), near_m, RulerRole.NEAR)
            ]
            self._cal_dialog.set_status("已画近尺。请再拖出远尺。")
            self.statusBar().showMessage("近尺已画出，请拖出远尺。", 4000)
        else:
            self._pending_rulers = [
                self._pending_rulers[0],
                RulerSegment(Point2D(ax, ay), Point2D(bx, by), far_m, RulerRole.FAR),
            ]
            self._cal_dialog.set_status("近尺和远尺已就绪。确认后点击应用。")
            self.statusBar().showMessage("双尺已画出，在面板中应用或重画。", 4000)
        self._sync_cal_overlay()

    def _apply_pending_calibration(self) -> None:
        if self._cal_dialog.mode() is CalibrationMode.PLANAR:
            self._apply_pending_plane()
            return
        if not self._pending_rulers:
            self._cal_dialog.set_status("请先画标定尺。", error=True)
            return
        near_m, far_m = self._cal_dialog.lengths()
        rulers = list(self._pending_rulers)
        rulers[0].length_m = near_m
        frame = self._calibration.frame
        if self._cal_dialog.mode() is CalibrationMode.NEAR_FAR:
            if len(rulers) < 2:
                self._cal_dialog.set_status("透视模式需要近尺和远尺。", error=True)
                return
            rulers[1].length_m = far_m
            state = near_far_state(
                rulers[0],
                rulers[1],
                origin=frame.origin,
                axis_angle_deg=frame.axis_angle_deg,
            )
        else:
            state = uniform_state(
                rulers[0].a,
                rulers[0].b,
                length_m=near_m,
                origin=frame.origin,
                axis_angle_deg=frame.axis_angle_deg,
            )
        ok, message = state.validate()
        if not ok or state.mode is CalibrationMode.NONE:
            self._cal_dialog.set_status(state.warning or message, error=True)
            return
        state.frame = CoordinateFrame(
            origin_x=frame.origin_x,
            origin_y=frame.origin_y,
            axis_angle_deg=frame.axis_angle_deg,
            y_up=frame.y_up,
        )
        self._push_undo()
        self._calibration = state
        self._pending_rulers = None
        if self._shake is None:
            self._cal_drawn_before_shake = True
        self._cal_dialog.set_status("标定已应用。")
        self._exit_interaction()
        self._cal_dialog.hide()
        self._refresh_track_ui()
        extra = " 超出覆盖区域时将提示外推。" if state.mode is CalibrationMode.NEAR_FAR else ""
        self.statusBar().showMessage(f"标定已应用，单位切换为 m / m/s。{extra}", 5000)

    def _apply_pending_plane(self) -> None:
        corners = None
        if self._pending_plane is not None and len(self._pending_plane) >= 4:
            corners = list(self._pending_plane[:4])
        elif self._calibration.plane is not None and len(self._calibration.plane.corners) >= 4:
            corners = list(self._calibration.plane.corners[:4])
        if corners is None:
            self._cal_dialog.set_status("请依次点选原点、X 端、对角点和 Y 端。", error=True)
            return
        width_m, height_m = self._cal_dialog.plane_size()
        frame = self._calibration.frame
        camera = self._calibration.camera
        if self._info is not None:
            if camera is None:
                camera = CameraProfile(width=self._info.width, height=self._info.height)
            else:
                camera.width = self._info.width
                camera.height = self._info.height
        state = planar_state(
            corners,
            width_m=width_m,
            height_m=height_m,
            origin=frame.origin,
            axis_angle_deg=frame.axis_angle_deg,
            camera=camera,
            y_up=frame.y_up,
        )
        ok, message = state.validate()
        if not ok or state.mode is CalibrationMode.NONE:
            self._cal_dialog.set_status(state.warning or message, error=True)
            return
        self._push_undo()
        self._calibration = state
        self._pending_plane = None
        if self._shake is None:
            self._cal_drawn_before_shake = True
        note = state.warning or "标定已应用。"
        self._cal_dialog.set_status(note)
        self._exit_interaction()
        self._cal_dialog.hide()
        self._refresh_track_ui()
        extra = " 未配置镜头内参时精度会下降。" if not state.camera or not state.camera.has_intrinsics else ""
        self.statusBar().showMessage(f"运动平面已应用，单位切换为 m / m/s。{extra}", 5000)

    def _on_plane_point_picked(self, x: float, y: float) -> None:
        if self._pending_plane is None:
            self._pending_plane = []
        if len(self._pending_plane) >= 4:
            return
        sx, sy = self._video_to_stable(x, y)
        self._pending_plane.append(Point2D(sx, sy))
        hint = self._cal_dialog.plane_step_hint(len(self._pending_plane))
        self._cal_dialog.set_status(hint)
        self.statusBar().showMessage(hint, 4000)
        self._sync_cal_overlay()

    def _on_plane_corner_dragged(self, index: int, x: float, y: float) -> None:
        sx, sy = self._video_to_stable(x, y)
        if self._pending_plane is not None:
            if 0 <= index < len(self._pending_plane):
                self._pending_plane[index] = Point2D(sx, sy)
            self._sync_cal_overlay()
            return
        if self._calibration.plane is None or not (0 <= index < len(self._calibration.plane.corners)):
            return
        if not self._plane_undo_pushed:
            self._push_undo()
            self._plane_undo_pushed = True
            if self._shake is None:
                self._cal_drawn_before_shake = True
        self._calibration.plane.corners[index] = Point2D(sx, sy)
        self._calibration.validate()
        self._sync_cal_overlay()

    def _on_plane_drag_finished(self) -> None:
        self._plane_undo_pushed = False
        if self._pending_plane is None and self._calibration.mode is CalibrationMode.PLANAR:
            self._refresh_track_ui()

    def _clear_calibration(self) -> None:
        self._push_undo()
        frame = self._calibration.frame
        self._calibration = CalibrationState(frame=frame, camera=self._calibration.camera)
        self._pending_rulers = None
        self._pending_plane = None
        self._depth_audit = DepthAuditState(
            experimental_correction=self._cal_dialog.experimental_correction()
        )
        self._exit_interaction()
        self._refresh_track_ui()
        self.statusBar().showMessage("已清除标定尺，单位恢复为 px。", 4000)

    def _on_axis_dragged(self, kind: str, x: float, y: float, shift: bool = False) -> None:
        if not self._axis_undo_pushed:
            self._push_undo()
            self._axis_undo_pushed = True
            if self._shake is None:
                self._cal_drawn_before_shake = True
        sx, sy = self._video_to_stable(x, y)
        frame = self._calibration.frame
        if kind == "origin":
            self._axis_rotate_base = None
            frame.origin_x = sx
            frame.origin_y = sy
            self._video.set_axis_origin((x, y))
            self._sync_cal_overlay()
            return
        origin = frame.origin
        if origin is None:
            frame.origin_x = sx
            frame.origin_y = sy
            self._sync_cal_overlay()
            return
        mouse_ang = axis_pointer_angle(
            origin.x, origin.y, sx, sy, y_up=frame.y_up
        )
        if self._axis_rotate_base is None:
            self._axis_rotate_base = (frame.axis_angle_deg, mouse_ang)
        base_axis, base_mouse = self._axis_rotate_base
        angle = base_axis + (mouse_ang - base_mouse)
        if shift:
            angle = snap_axis_angle(angle)
        frame.axis_angle_deg = angle
        self._sync_cal_overlay()

    def _on_axis_drag_finished(self) -> None:
        self._axis_undo_pushed = False
        self._axis_rotate_base = None
        origin = self._calibration.frame.origin
        angle = self._calibration.axis_angle()
        self._refresh_track_ui()
        if origin is not None:
            self.statusBar().showMessage(
                f"坐标系：原点 ({origin.x:.1f}, {origin.y:.1f})，x 正向 {angle:.1f}°",
                4000,
            )

    def _sync_cal_overlay(self) -> None:
        source = (
            self._pending_rulers
            if self._pending_rulers is not None
            else self._calibration.rulers
        )
        rulers: list[tuple[float, float, float, float, str, str]] = []
        for ruler in source:
            ax, ay = self._stable_to_video(ruler.a.x, ruler.a.y)
            bx, by = self._stable_to_video(ruler.b.x, ruler.b.y)
            role = ""
            if ruler.role is RulerRole.NEAR:
                role = "近"
            elif ruler.role is RulerRole.FAR:
                role = "远"
            rulers.append((ax, ay, bx, by, f"{ruler.length_m:.3f} m", role))
        self._video.set_ruler_overlay(rulers)
        origin = self._calibration.frame.origin
        if origin is not None:
            ov = self._stable_to_video(origin.x, origin.y)
            unit = self._calibration.position_unit
            if self._info is not None:
                length = axis_display_length(self._info.width, self._info.height)
            else:
                length = AXIS_MIN_LENGTH
            ang = self._calibration.axis_angle()
            x_end, y_end = axis_arm_ends(
                ov[0],
                ov[1],
                length,
                ang,
                y_up=self._calibration.frame.y_up,
            )
            self._video.set_axis_overlay(ov, x_end, y_end, f"x ({unit})", f"y ({unit})")
        else:
            self._video.set_axis_overlay(None, None, None, "", "")
        self._sync_plane_overlay()
        self._video.set_show_calibration(self._show_calibration)

    def _sync_plane_overlay(self) -> None:
        corners: list[Point2D] = []
        if self._pending_plane is not None:
            corners = list(self._pending_plane)
        elif self._calibration.plane is not None:
            corners = list(self._calibration.plane.corners)
        mapped = [self._stable_to_video(point.x, point.y) for point in corners]
        grid_video: list[tuple[float, float, float, float]] = []
        label = ""
        if len(corners) >= 4:
            width_m, height_m = self._cal_dialog.plane_size()
            if self._pending_plane is None and self._calibration.plane is not None:
                width_m = self._calibration.plane.width_m
                height_m = self._calibration.plane.height_m
            preview = planar_state(
                corners,
                width_m=width_m,
                height_m=height_m,
                camera=self._calibration.camera,
            )
            if preview.mode is CalibrationMode.PLANAR and preview.plane is not None:
                for x0, y0, x1, y1 in preview.grid_lines():
                    a = self._stable_to_video(x0, y0)
                    b = self._stable_to_video(x1, y1)
                    grid_video.append((a[0], a[1], b[0], b[1]))
                label = f"RMS {preview.plane.reprojection_rms_px:.2f} px"
                if preview.warning:
                    label += " · " + preview.warning.split("；")[0]
        self._video.set_plane_overlay(mapped, grid_video, label)

    def _stop_depth_audit(self) -> None:
        if self._depth_worker is not None:
            self._depth_worker.cancel()
        if self._depth_thread is not None:
            self._depth_thread.quit()
            self._depth_thread.wait(5000)
        self._depth_thread = None
        self._depth_worker = None

    def _current_rgb(self):
        if self._info is None:
            return None
        decoder = FrameDecoder(self._info)
        try:
            return decoder.frame(self._index)
        finally:
            decoder.close()

    def _on_charuco_detect(self) -> None:
        from ai.charuco import detect_plane_from_frame

        rgb = self._current_rgb()
        if rgb is None:
            self._cal_dialog.set_status("请先打开视频。", error=True)
            return
        detection, message = detect_plane_from_frame(rgb, camera=self._calibration.camera)
        if detection is None:
            self._cal_dialog.set_status(message, error=True)
            return
        self._pending_plane = [
            Point2D(*self._video_to_stable(point.x, point.y)) for point in detection.corners
        ]
        self._cal_dialog.set_plane_size(detection.width_m, detection.height_m)
        self._cal_dialog.set_status("已检测棋盘格四角，确认宽高后点击应用。")
        self.statusBar().showMessage("ChArUco 平面已填入四个角点。", 4000)
        self._sync_cal_overlay()

    def _on_camera_calib(self) -> None:
        from ai.charuco import opencv_available

        if not opencv_available():
            self._cal_dialog.set_status(
                "未安装 OpenCV。可选：python -m pip install opencv-contrib-python",
                error=True,
            )
            return
        self._charuco_views = []
        self._camera_dialog.set_status("在不同角度显示棋盘格，采集至少 3 张后计算内参。")
        self._camera_dialog.show()
        self._camera_dialog.raise_()

    def _capture_calib_view(self) -> None:
        from ai.charuco import collect_calibration_view

        rgb = self._current_rgb()
        if rgb is None:
            self._camera_dialog.set_status("请先打开视频。", error=True)
            return
        corners, ids, message = collect_calibration_view(rgb)
        if corners is None or ids is None:
            self._camera_dialog.set_status(message, error=True)
            return
        self._charuco_views.append((corners, ids))
        self._camera_dialog.set_status(f"已采集 {len(self._charuco_views)} 张。至少 3 张后可计算内参。")

    def _compute_camera_profile(self) -> None:
        from ai.charuco import calibrate_camera

        if self._info is None:
            self._camera_dialog.set_status("请先打开视频。", error=True)
            return
        profile, message = calibrate_camera(
            self._charuco_views, (self._info.width, self._info.height)
        )
        if profile is None:
            self._camera_dialog.set_status(message, error=True)
            return
        self._push_undo()
        self._calibration.camera = profile
        self._calibration.validate()
        rms = "" if profile.rms is None else f" RMS {profile.rms:.3f} px"
        self._camera_dialog.set_status(f"镜头内参已保存。{rms}")
        self._cal_dialog.set_status(f"已写入镜头内参。{rms}")
        self._refresh_track_ui()

    def _on_experimental_correction(self, enabled: bool) -> None:
        self._depth_audit.experimental_correction = enabled
        self._refresh_track_ui()

    def _on_depth_audit(self) -> None:
        layer = self._active_layer()
        analyzed = self._analysis_result(layer)
        if self._info is None or analyzed is None:
            self._cal_dialog.set_status("需要已跟踪的轨迹才能抽检。", error=True)
            return
        if self._calibration.mode is not CalibrationMode.PLANAR or not self._calibration.active:
            self._cal_dialog.set_status("离面抽检需要已应用的运动平面。", error=True)
            return
        if self._depth_worker is not None:
            self._cal_dialog.set_status("正在抽检…")
            return
        self._cal_dialog.set_status("正在进行 AI 离面抽检…")
        thread, worker = run_audit_in_thread(
            self._info.path,
            analyzed,
            self._calibration,
            experimental_correction=self._cal_dialog.experimental_correction(),
            on_finished=self._on_depth_audit_finished,
            on_failed=self._on_depth_audit_failed,
        )
        self._depth_thread = thread
        self._depth_worker = worker
        thread.start()

    def _on_depth_audit_finished(self, state) -> None:  # noqa: ANN001
        self._stop_depth_audit()
        self._depth_audit = state
        self._cal_dialog.set_experimental_correction(state.experimental_correction)
        self._cal_dialog.set_status(state.message or "离面抽检完成。")
        off = len(state.off_plane_frames())
        self.statusBar().showMessage(
            state.message or f"离面抽检完成，{off} 帧告警。",
            6000,
        )
        self._refresh_track_ui()

    def _on_depth_audit_failed(self, message: str) -> None:
        self._stop_depth_audit()
        text = message or "离面抽检失败"
        self._cal_dialog.set_status(text, error=True)
        self.statusBar().showMessage(text, 6000)

    def _restore_window_prefs(self) -> None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        settings = QSettings()
        geo = settings.value("main/geometry")
        if isinstance(geo, QByteArray) and not geo.isEmpty():
            self.restoreGeometry(geo)
            if not _on_screen(self.frameGeometry()):
                self.resize(1280, 800)
                self.move(80, 60)
        self._workspace.restore_prefs(settings)
        if getattr(self, "_chart_action", None) is not None:
            self._chart_action.setChecked(self._workspace.chart_visible)
        if getattr(self, "_table_action", None) is not None:
            self._table_action.setChecked(self._workspace.data_visible)
        geo_asst = settings.value("assistant/geometry")
        if isinstance(geo_asst, QByteArray) and not geo_asst.isEmpty():
            self._assistant_window.restoreGeometry(geo_asst)
            if not _on_screen(self._assistant_window.frameGeometry()):
                self._assistant_window.resize(960, 720)
                self._assistant_window.move(120, 80)
        self._load_assistant_prefs(settings)
        mode = settings.value("track/mode", TrackMode.PRECISE.value)
        try:
            self._set_track_mode(TrackMode(str(mode)), announce=False)
        except ValueError:
            self._set_track_mode(TrackMode.PRECISE, announce=False)

    def _save_window_prefs(self) -> None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        settings = QSettings()
        settings.setValue("main/geometry", self.saveGeometry())
        self._workspace.save_prefs(settings)
        settings.setValue("assistant/geometry", self._assistant_window.saveGeometry())
        settings.setValue("assistant/chat_model", self._chat_model)
        settings.setValue("assistant/report_model", self._report_model)
        settings.setValue("assistant/teaching_level", self._assistant_state.teaching_level.value)
        settings.setValue("track/mode", self._track_mode.value)

    def _load_assistant_prefs(self, settings: QSettings) -> None:
        chat = settings.value("assistant/chat_model", DEFAULT_CHAT_MODEL)
        report = settings.value("assistant/report_model", DEFAULT_REPORT_MODEL)
        level = settings.value("assistant/teaching_level", TeachingLevel.HIGH.value)
        if isinstance(chat, str) and chat.strip():
            self._chat_model = chat.strip()
        if isinstance(report, str) and report.strip():
            self._report_model = report.strip()
        try:
            self._assistant_state.teaching_level = TeachingLevel(str(level))
        except ValueError:
            self._assistant_state.teaching_level = TeachingLevel.HIGH
        self._assistant_panel.set_teaching_level(self._assistant_state.teaching_level)
        self._assistant_panel.set_models(self._chat_model, self._report_model)

    def _set_assistant_visible(self, visible: bool) -> None:
        if visible:
            self._assistant_window.show()
            self._assistant_window.raise_()
            self._assistant_window.activateWindow()
        else:
            self._assistant_window.hide()

    def _show_assistant_panel(self, *_args) -> None:
        self._set_assistant_visible(True)

    def _assistant_dialog_parent(self) -> QWidget:
        if self._assistant_window.isVisible():
            return self._assistant_window
        return self

    def _reset_assistant_state(self) -> None:
        level = self._assistant_state.teaching_level
        self._assistant_state = AssistantState(teaching_level=level)
        self._assistant_panel.clear_chat()
        self._assistant_panel.set_report("")
        self._assistant_panel.set_analysis(None, None)
        self._assistant_panel.set_stale(False)
        self._assistant_panel.set_pendulum_length(None)

    def _restore_assistant_from_project(self, doc) -> None:  # noqa: ANN001
        self._assistant_state = doc.assistant or AssistantState()
        current = self._current_fingerprint()
        if self._assistant_state.fingerprint and self._assistant_state.fingerprint != current:
            self._assistant_state.stale = True
        self._assistant_panel.set_teaching_level(self._assistant_state.teaching_level)
        self._assistant_panel.set_pendulum_length(self._assistant_state.pendulum_length_m)
        self._assistant_panel.clear_chat()
        for message in self._assistant_state.messages:
            if message.role == "user":
                self._assistant_panel.add_user_message(message.content)
            else:
                self._assistant_panel.begin_assistant_message(
                    live=False, reasoning=message.reasoning
                )
                self._assistant_panel.finish_assistant_message(
                    message.content, cancelled=message.cancelled
                )
        self._assistant_panel.set_report(self._assistant_state.report_markdown)
        self._assistant_panel.set_analysis(
            self._assistant_state.analysis, self._assistant_state.confirmed_type
        )

    def _shake_offsets(self) -> tuple[tuple[float, float], ...]:
        if self._shake is None or self._info is None:
            return ()
        last = max(self._info.frame_count - 1, 0)
        return (self._shake.offset(0), self._shake.offset(last))

    def _current_fingerprint(self) -> str:
        layer = self._active_layer()
        return source_fingerprint(
            self._analysis_result(layer),
            self._info,
            self._calibration,
            shake_enabled=self._shake_enabled,
            shake_offsets=self._shake_offsets(),
            pendulum_length_m=self._assistant_state.pendulum_length_m,
        )

    def _sync_assistant(self) -> None:
        layer = self._active_layer()
        analyzed = self._analysis_result(layer)
        visible = 0 if analyzed is None else sum(1 for point in analyzed.points if point.visible)
        current = self._current_fingerprint()
        if (
            self._assistant_state.fingerprint
            and self._assistant_state.fingerprint != current
        ):
            self._assistant_state.stale = True
        self._assistant_panel.set_stale(self._assistant_state.stale)
        self._assistant_panel.set_models(self._chat_model, self._report_model)
        self._assistant_panel.set_readiness(
            has_video=self._info is not None,
            visible_points=visible,
            calibrated=self._calibration.active,
            shake_on=self._shake_enabled and self._shake is not None,
            has_key=has_api_key(),
            sam_running=self._ai_worker is not None,
            confirmed=self._assistant_state.confirmed_type is not None
            and not self._assistant_state.stale,
        )
        self._assistant_panel.set_analysis(
            self._assistant_state.analysis, self._assistant_state.confirmed_type
        )
        self._sync_assistant_actions()

    def _sync_assistant_actions(self) -> None:
        layer = self._active_layer()
        analyzed = self._analysis_result(layer)
        visible = 0 if analyzed is None else sum(1 for point in analyzed.points if point.visible)
        sam = self._ai_worker is not None
        can_analyze = self._info is not None and visible >= 6 and not sam and not self._assistant_busy
        confirmed = (
            self._assistant_state.confirmed_type is not None and not self._assistant_state.stale
        )
        if getattr(self, "_ai_analyze_action", None) is not None:
            self._ai_analyze_action.setEnabled(can_analyze)
        if getattr(self, "_ai_report_action", None) is not None:
            self._ai_report_action.setEnabled(
                confirmed and has_api_key() and not sam and not self._assistant_busy
            )

    def _assistant_samples(self):
        layer = self._active_layer()
        return series_for_result(
            self._analysis_result(layer),
            self._info,
            calibration=self._calibration,
            velocity_step=self._chart_panel.velocity_step,
            depth_audit=self._depth_audit,
        )

    def _teaching_context(self) -> dict:
        plane = self._calibration.plane
        return build_teaching_context(
            self._assistant_state.analysis,
            confirmed_type=self._assistant_state.confirmed_type,
            samples=self._assistant_samples(),
            teaching_level=self._assistant_state.teaching_level,
            pendulum_length_m=self._assistant_state.pendulum_length_m,
            stale=self._assistant_state.stale,
            interpolated=self._active_track_interpolated(),
            calibration_mode=self._calibration.mode.value,
            reprojection_rms_px=None if plane is None else plane.reprojection_rms_px,
            off_plane_frames=len(self._depth_audit.off_plane_frames()),
            camera_moved=self._calibration.camera_moved,
            quality_label=self._calibration.quality_label(),
        )

    def _active_track_interpolated(self) -> bool:
        layer = self._active_layer()
        if layer is None or layer.result is None:
            return False
        return any(point.interpolated for point in layer.result.points)

    def _analyze_experiment(self, *_args) -> None:
        self._set_assistant_visible(True)
        if self._ai_worker is not None:
            message = "请等待 SAM 跟踪完成后再分析"
            self._assistant_panel.set_notice(message)
            self.statusBar().showMessage(message, 4000)
            return
        if self._assistant_busy:
            message = "请先停止当前助手任务"
            self._assistant_panel.set_notice(message)
            self.statusBar().showMessage(message, 3000)
            return
        layer = self._active_layer()
        analyzed = self._analysis_result(layer)
        if self._info is None or analyzed is None:
            message = "需要先打开视频并完成轨迹，才能识别实验。"
            self._assistant_panel.set_notice(message)
            self.statusBar().showMessage(message, 4000)
            return
        visible = sum(1 for point in analyzed.points if point.visible)
        if visible < 6:
            message = f"可见轨迹点只有 {visible} 个，至少需要 6 个才能识别。"
            self._assistant_panel.set_notice(message)
            self.statusBar().showMessage(message, 4000)
            return
        self._assistant_panel.set_notice("正在用本地轨迹识别实验…")
        self._start_assistant_job(
            AssistantJob(
                kind="analyze",
                track=analyzed,
                info=self._info,
                calibration=self._calibration,
                shake_enabled=self._shake_enabled,
                shake_offsets=self._shake_offsets(),
                clip_id=analyzed.clip_id,
                pendulum_length_m=self._assistant_state.pendulum_length_m,
                period_hint=True,
            )
        )

    def _confirm_experiment(self, type_value: str) -> None:
        try:
            kind = ExperimentType(type_value)
        except ValueError:
            return
        analysis = self._assistant_state.analysis
        if analysis is None:
            self._analyze_experiment()
            return
        match = next((item for item in analysis.candidates if item.experiment_type is kind), None)
        if match is None:
            self._start_assistant_job(
                AssistantJob(
                    kind="analyze",
                    track=self._analysis_result(self._active_layer()),
                    info=self._info,
                    calibration=self._calibration,
                    shake_enabled=self._shake_enabled,
                    shake_offsets=self._shake_offsets(),
                    clip_id="" if self._info is None else self._info.path.name,
                    pendulum_length_m=self._assistant_state.pendulum_length_m,
                    force_type=kind,
                )
            )
            self._assistant_state.confirmed_type = kind
            return
        analysis.selected = match
        self._assistant_state.confirmed_type = kind
        self._assistant_state.stale = False
        self._assistant_state.fingerprint = analysis.fingerprint or self._current_fingerprint()
        self._sync_assistant()
        self.statusBar().showMessage(f"已确认实验：{match.label}", 4000)

    def _send_assistant_chat(self, text: str) -> None:
        if not has_api_key():
            warn_missing_key(self._assistant_dialog_parent())
            return
        if self._assistant_busy:
            return
        self._set_assistant_visible(True)
        self._assistant_state.messages.append(ChatMessage(role="user", content=text))
        self._assistant_panel.add_user_message(text)
        self._assistant_panel.begin_assistant_message()
        context = self._teaching_context()
        try:
            assert_private_context(context)
        except ValueError:
            context = {key: value for key, value in context.items() if key not in {"path", "video_path"}}
        self._start_assistant_job(
            AssistantJob(
                kind="chat",
                api_key=get_api_key() or "",
                chat_model=self._chat_model,
                report_model=self._report_model,
                messages=chat_messages(
                    context,
                    self._assistant_state.messages[:-1],
                    text,
                    teaching_level=self._assistant_state.teaching_level,
                ),
                stream=True,
                transport=self._assistant_transport,
            )
        )

    def _generate_assistant_report(self, *_args) -> None:
        if self._ai_worker is not None:
            self.statusBar().showMessage("请等待 SAM 跟踪完成后再生成报告", 4000)
            return
        if self._assistant_state.confirmed_type is None:
            self.statusBar().showMessage("请先确认实验类型", 4000)
            return
        if not has_api_key():
            markdown = render_report_markdown(
                self._assistant_state.analysis,
                confirmed_type=self._assistant_state.confirmed_type,
                teaching_level=self._assistant_state.teaching_level,
                stale=self._assistant_state.stale,
            )
            self._assistant_state.report_markdown = markdown
            self._assistant_panel.set_report(markdown)
            warn_missing_key(self._assistant_dialog_parent())
            return
        if self._assistant_busy:
            return
        context = self._teaching_context()
        self._start_assistant_job(
            AssistantJob(
                kind="report",
                api_key=get_api_key() or "",
                chat_model=self._chat_model,
                report_model=self._report_model,
                messages=report_messages(
                    context, teaching_level=self._assistant_state.teaching_level
                ),
                json_mode=True,
                transport=self._assistant_transport,
            )
        )

    def _export_assistant_report(self) -> None:
        markdown = self._assistant_state.report_markdown or build_assistant_report(
            self._assistant_state
        )
        if not markdown.strip():
            self.statusBar().showMessage("还没有可导出的报告", 3000)
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出实验报告", "", "Markdown (*.md)")
        if not path:
            return
        export_assistant_report(Path(path), markdown)
        self.statusBar().showMessage("已导出 Markdown 报告", 4000)

    def _copy_assistant_report(self) -> None:
        text = self._assistant_state.report_markdown or self._assistant_panel.report_text()
        QApplication.clipboard().setText(text or "")
        self.statusBar().showMessage("已复制报告", 2500)

    def _preview_assistant_payload(self) -> None:
        payload = preview_payload(self._teaching_context())
        show_payload_preview(self._assistant_dialog_parent(), payload)

    def _clear_assistant_chat(self) -> None:
        self._assistant_state.messages = []
        self._assistant_panel.clear_chat()

    def _on_teaching_level(self, value: str) -> None:
        try:
            self._assistant_state.teaching_level = TeachingLevel(value)
        except ValueError:
            self._assistant_state.teaching_level = TeachingLevel.HIGH

    def _on_pendulum_length(self, value: float) -> None:
        self._assistant_state.pendulum_length_m = None if value <= 0 else float(value)
        if self._assistant_state.fingerprint:
            self._assistant_state.stale = True
        self._sync_assistant()

    def _open_deepseek_settings(self, *_args) -> None:
        dialog = DeepSeekSettingsDialog(
            self._assistant_dialog_parent(),
            chat_model=self._chat_model,
            report_model=self._report_model,
            teaching_level=self._assistant_state.teaching_level,
        )
        if dialog.exec():
            self._chat_model = dialog.chat_model()
            self._report_model = dialog.report_model()
            self._assistant_state.teaching_level = dialog.teaching_level()
            self._assistant_panel.set_teaching_level(self._assistant_state.teaching_level)
            self._assistant_panel.set_models(self._chat_model, self._report_model)
            self._sync_assistant()
            if dialog.test_requested:
                self._test_deepseek_key()

    def _test_deepseek_key(self) -> None:
        if not has_api_key():
            warn_missing_key(self._assistant_dialog_parent())
            return
        self._start_assistant_job(
            AssistantJob(
                kind="test_key",
                api_key=get_api_key() or "",
                chat_model=self._chat_model,
                messages=[
                    {"role": "system", "content": "只回复 ok。"},
                    {"role": "user", "content": "ping json"},
                ],
                transport=self._assistant_transport,
            )
        )

    def _start_assistant_job(self, job: AssistantJob) -> None:
        if self._assistant_busy:
            self.statusBar().showMessage("请先停止当前助手任务", 3000)
            return
        self._stop_assistant()
        thread, worker = run_assistant_in_thread(
            job,
            on_chunk=self._on_assistant_chunk,
            on_reasoning=self._on_assistant_reasoning,
            on_finished=self._on_assistant_finished,
            on_failed=self._on_assistant_failed,
            on_cancelled=self._on_assistant_cancelled,
            on_usage=self._on_assistant_usage,
        )
        self._assistant_thread = thread
        self._assistant_worker = worker
        self._assistant_busy = True
        self._assistant_panel.set_busy(True)
        self._sync_assistant_actions()
        thread.start()

    def _cancel_assistant(self) -> None:
        if self._assistant_worker is not None:
            self._assistant_worker.cancel()
            self.statusBar().showMessage("正在停止助手…", 2500)

    def _release_assistant_thread(self) -> None:
        thread = self._assistant_thread
        self._assistant_thread = None
        self._assistant_worker = None
        self._assistant_busy = False
        if getattr(self, "_assistant_panel", None) is not None:
            self._assistant_panel.set_busy(False)
        if thread is not None:
            thread.quit()
            thread.wait(5000)

    def _stop_assistant(self) -> None:
        if self._assistant_worker is not None:
            self._assistant_worker.cancel()
        self._release_assistant_thread()

    def _on_assistant_chunk(self, text: str) -> None:
        self._assistant_panel.append_chunk(text)

    def _on_assistant_reasoning(self, text: str) -> None:
        self._assistant_panel.append_reasoning(text)

    def _on_assistant_usage(self, usage: dict) -> None:
        self._assistant_state.token_usage = {
            str(key): int(value) for key, value in usage.items()
        }

    def _on_assistant_cancelled(self) -> None:
        self._release_assistant_thread()
        self._sync_assistant_actions()

    def _on_assistant_failed(self, message: str) -> None:
        self._release_assistant_thread()
        self._sync_assistant_actions()
        self._assistant_panel.set_notice(message or "助手请求失败")
        self._assistant_panel.finish_assistant_message(message or "助手请求失败")
        self.statusBar().showMessage(message or "助手请求失败", 8000)
        if message:
            QMessageBox.warning(self._assistant_dialog_parent(), "AI 助手", message)

    def _on_assistant_finished(self, outcome: object) -> None:
        if not isinstance(outcome, AssistantOutcome):
            return
        self._release_assistant_thread()
        self._sync_assistant_actions()
        if outcome.usage:
            self._assistant_state.token_usage = outcome.usage
        if outcome.kind == "analyze":
            self._assistant_state.analysis = outcome.analysis
            self._assistant_state.fingerprint = (
                outcome.analysis.fingerprint if outcome.analysis is not None else self._current_fingerprint()
            )
            self._assistant_state.stale = False
            if outcome.analysis is not None and outcome.analysis.auto_confirmable and outcome.analysis.selected:
                self._assistant_state.confirmed_type = outcome.analysis.selected.experiment_type
            elif self._assistant_state.confirmed_type is not None and outcome.analysis is not None:
                match = next(
                    (
                        item
                        for item in outcome.analysis.candidates
                        if item.experiment_type is self._assistant_state.confirmed_type
                    ),
                    None,
                )
                if match is not None:
                    outcome.analysis.selected = match
            self._sync_assistant()
            self._assistant_panel.add_analysis_result(outcome.analysis)
            self._assistant_panel.set_notice("本地识别完成，结果已写在对话里，也可在左侧查看。")
            self.statusBar().showMessage("本地实验识别完成", 4000)
            return
        if outcome.kind == "chat":
            self._assistant_panel.finish_assistant_message(outcome.text, cancelled=outcome.cancelled)
            self._assistant_state.messages.append(
                ChatMessage(
                    role="assistant",
                    content=outcome.text,
                    cancelled=outcome.cancelled,
                    reasoning=outcome.reasoning,
                )
            )
            if outcome.cancelled:
                self.statusBar().showMessage("生成已取消", 3000)
            self._sync_assistant()
            return
        if outcome.kind == "report":
            sections = merge_report_sections(outcome.json_data)
            self._assistant_state.report_sections = sections
            self._assistant_state.model_id = outcome.model
            self._assistant_state.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            markdown = render_report_markdown(
                self._assistant_state.analysis,
                confirmed_type=self._assistant_state.confirmed_type,
                sections=sections,
                teaching_level=self._assistant_state.teaching_level,
                generated_at=self._assistant_state.generated_at,
                model_id=outcome.model,
                stale=self._assistant_state.stale,
            )
            self._assistant_state.report_markdown = markdown
            self._assistant_panel.set_report(markdown)
            self.statusBar().showMessage("报告已生成", 4000)
            self._sync_assistant()
            return
        if outcome.kind == "test_key":
            self.statusBar().showMessage("DeepSeek 连接正常", 4000)
            self._sync_assistant()


def _fmt_ms(ms: int) -> str:
    total = max(ms, 0)
    minutes, rest = divmod(total // 1000, 60)
    hours, minutes = divmod(minutes, 60)
    frac = (total % 1000) // 10
    if hours:
        return f"{hours:d}:{minutes:02d}:{rest:02d}.{frac:02d}"
    return f"{minutes:02d}:{rest:02d}.{frac:02d}"
