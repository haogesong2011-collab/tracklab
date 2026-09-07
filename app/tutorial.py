"""First-run spotlight tour over the main window."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygon,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

SETTINGS_TUTORIAL_SEEN = "tutorial/seen"
HOLE_PAD = 8
HOLE_RADIUS = 8
CARD_WIDTH = 360
CARD_MARGIN = 16
MASK_ALPHA = 150
HIGHLIGHT = QColor("#4da3ff")
DEMO_CYCLE_MS = 2500
DEMO_TICK_MS = 16
ADVANCE_DELAY_MS = 450


class DemoKind(str, Enum):
    DROP = "drop"
    CLICK = "click"
    BOX = "box"
    LINE = "line"
    SCRUB = "scrub"
    PLAY = "play"


@dataclass(frozen=True)
class TourStep:
    title: str
    body: str
    target: Callable[[], QWidget | None]
    prepare: Callable[[], None] | None = None
    demo: DemoKind = DemoKind.CLICK
    advance_on: str = "click"


def tutorial_seen(settings: QSettings | None = None) -> bool:
    store = settings or QSettings()
    value = store.value(SETTINGS_TUTORIAL_SEEN, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def mark_tutorial_seen(settings: QSettings | None = None) -> None:
    store = settings or QSettings()
    store.setValue(SETTINGS_TUTORIAL_SEEN, True)
    store.sync()


def tutorial_auto_start_allowed() -> bool:
    flag = os.environ.get("TRACKLAB_SKIP_TUTORIAL", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return False
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return False
    if "--smoke" in sys.argv:
        return False
    return True


def default_steps(window) -> list[TourStep]:  # noqa: ANN001
    def drop_target() -> QWidget | None:
        hint = getattr(window, "_hint", None)
        if hint is not None and hint.isVisible():
            return hint
        return getattr(window, "_stage", None)

    def tool(name: str) -> Callable[[], QWidget | None]:
        def lookup() -> QWidget | None:
            buttons = getattr(window, "_toolbar_buttons", {})
            return buttons.get(name)
        return lookup

    def ensure_chart() -> None:
        workspace = getattr(window, "_workspace", None)
        if workspace is not None:
            workspace.set_chart_visible(True)
        action = getattr(window, "_chart_action", None)
        if action is not None:
            action.setChecked(True)

    def ensure_data() -> None:
        workspace = getattr(window, "_workspace", None)
        if workspace is not None:
            workspace.set_data_visible(True)
        action = getattr(window, "_table_action", None)
        if action is not None:
            action.setChecked(True)

    def chart_dock() -> QWidget | None:
        workspace = getattr(window, "_workspace", None)
        return None if workspace is None else workspace.chart_dock

    def data_dock() -> QWidget | None:
        workspace = getattr(window, "_workspace", None)
        return None if workspace is None else workspace.data_dock

    def transport() -> QWidget | None:
        return getattr(window, "_transport", None)

    return [
        TourStep(
            "导入视频",
            "把 mp4 / mov 拖进高亮区域，或点它打开文件。指针只是示范，请你自己操作。也可以点下一步。",
            drop_target,
            demo=DemoKind.DROP,
            advance_on="video",
        ),
        TourStep(
            "打开视频",
            "点高亮的打开按钮选视频。打开成功后会继续。也可以点下一步。",
            tool("open"),
            demo=DemoKind.CLICK,
            advance_on="video",
        ),
        TourStep(
            "跟踪目标",
            "点高亮的轨迹按钮打开管理器。指针会在真实画面上框选目标；你也可以自己框。也可以点下一步。",
            tool("track"),
            demo=DemoKind.BOX,
        ),
        TourStep(
            "标定",
            "点高亮的尺子。指针会在真实画面上画标定尺；你也可以自己拖。也可以点下一步。",
            tool("ruler"),
            demo=DemoKind.LINE,
        ),
        TourStep(
            "分图",
            "指针会在真实分图上左右拖动对齐当前帧；你也可以自己拖。也可以点下一步。",
            chart_dock,
            prepare=ensure_chart,
            demo=DemoKind.SCRUB,
        ),
        TourStep(
            "数据表",
            "点高亮的数据表查看每帧数值。也可以点下一步。",
            data_dock,
            prepare=ensure_data,
            demo=DemoKind.CLICK,
        ),
        TourStep(
            "AI 助手",
            "点高亮按钮打开助手。首次使用请在菜单填写密钥。也可以点下一步。",
            tool("ai"),
            demo=DemoKind.CLICK,
        ),
        TourStep(
            "播放与逐帧",
            "点高亮的播放按钮，或按空格。左右方向键按底栏步长逐帧移动。也可以点完成。",
            transport,
            demo=DemoKind.PLAY,
        ),
    ]


def _clamp01(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t


def _smooth(t: float) -> float:
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def _mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _along(start: QPoint, end: QPoint, t: float) -> QPoint:
    t = _smooth(t)
    return QPoint(
        int(round(_mix(start.x(), end.x(), t))),
        int(round(_mix(start.y(), end.y(), t))),
    )


def _event_pos(event) -> QPoint:  # noqa: ANN001
    position = getattr(event, "position", None)
    if callable(position):
        return position().toPoint()
    return event.pos()


def _is_separate_window(widget: QWidget, host: QWidget) -> bool:
    flags = widget.windowFlags()
    if not (flags & Qt.WindowType.Window):
        return False
    return widget.window() is not host.window()


def _child_at_excluding(
    widget: QWidget,
    pos: QPoint,
    skip: QWidget,
) -> QWidget | None:
    children = [child for child in widget.children() if isinstance(child, QWidget)]
    for child in reversed(children):
        if child is skip or skip.isAncestorOf(child):
            continue
        if not child.isVisible():
            continue
        if _is_separate_window(child, widget):
            continue
        geo = child.geometry()
        if not geo.contains(pos):
            continue
        inner = _child_at_excluding(child, pos - geo.topLeft(), skip)
        return inner if inner is not None else child
    return None


def _hit_widget(host: QWidget | None, global_pos: QPoint, skip: QWidget) -> QWidget | None:
    if host is None:
        return None
    local = host.mapFromGlobal(global_pos)
    found = _child_at_excluding(host, local, skip)
    return found if found is not None else host


def tutorial_preview_image() -> QImage:
    image = QImage(640, 360, QImage.Format.Format_RGB32)
    image.fill(QColor("#1a1c20"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.fillRect(0, 252, 640, 108, QColor("#24262c"))
    painter.setPen(QPen(QColor("#3a3d44"), 2))
    painter.drawLine(0, 252, 640, 252)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#3d2a1c"))
    painter.drawEllipse(QPoint(292, 248), 38, 10)
    painter.setBrush(QColor("#e07a3a"))
    painter.drawEllipse(QPoint(288, 218), 32, 32)
    painter.setBrush(QColor("#f0b27a"))
    painter.drawEllipse(QPoint(276, 206), 10, 10)
    painter.end()
    return image


def _demo_chart_samples():
    from ai.kinematics import KinematicSample

    samples = []
    for index in range(48):
        time_s = index / 30.0
        x = 120.0 + 48.0 * math.sin(time_s * 2.4)
        y = 90.0 + 36.0 * math.cos(time_s * 2.4)
        samples.append(
            KinematicSample(
                frame=index,
                time_s=time_s,
                x=x,
                y=y,
                vx=48.0 * 2.4 * math.cos(time_s * 2.4),
                vy=-36.0 * 2.4 * math.sin(time_s * 2.4),
                speed=70.0,
                visible=True,
                confidence=1.0,
                manual=False,
            )
        )
    return samples


class TutorialOverlay(QWidget):
    finished = Signal()

    def __init__(
        self,
        host: QWidget,
        steps: list[TourStep],
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(
            host,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setObjectName("tutorialOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        host_window = host.window()
        if host_window is not None and host_window.styleSheet():
            self.setStyleSheet(host_window.styleSheet())
        self._steps = steps
        self._settings = settings
        self._index = 0
        self._done = False
        self._hole = QRect()
        self._phase = 0.0
        self._cursor = QPoint()
        self._press = 0.0
        self._live_pointer = False
        self._forward_grab: QWidget | None = None
        self._demo_down: QWidget | None = None
        self._demo_grabbing = False
        self._installed_preview = False
        self._installed_chart = False
        self._saved_video_mode: str | None = None
        self._demo_timer = QTimer(self)
        self._demo_timer.setInterval(DEMO_TICK_MS)
        self._demo_timer.timeout.connect(lambda: self._demo_tick(float(DEMO_TICK_MS)))
        self._advance_timer = QTimer(self)
        self._advance_timer.setSingleShot(True)
        self._advance_timer.setInterval(ADVANCE_DELAY_MS)
        self._advance_timer.timeout.connect(self._advance_from_action)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)

        self._card = QFrame(self)
        self._card.setObjectName("tutorialCard")
        self._card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._title = QLabel()
        self._title.setObjectName("tutorialTitle")
        self._title.setWordWrap(True)
        self._body = QLabel()
        self._body.setObjectName("tutorialBody")
        self._body.setWordWrap(True)
        self._counter = QLabel()
        self._counter.setObjectName("tutorialCounter")

        self._skip = QPushButton("跳过")
        self._skip.setObjectName("tutorialSkip")
        self._skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip.clicked.connect(self._complete)
        self._next = QPushButton("下一步")
        self._next.setObjectName("tutorialNext")
        self._next.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next.clicked.connect(self.advance)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self._title, stretch=1)
        header.addWidget(self._counter, stretch=0)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(8)
        buttons.addWidget(self._skip)
        buttons.addStretch(1)
        buttons.addWidget(self._next)

        layout = QVBoxLayout(self._card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self._body)
        layout.addLayout(buttons)

        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._complete)
        self.hide()

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def hole_rect(self) -> QRect:
        return QRect(self._hole)

    @property
    def cursor_pos(self) -> QPoint:
        return QPoint(self._cursor)

    @property
    def demo_phase(self) -> float:
        return self._phase

    def current_demo(self) -> DemoKind:
        if not self._steps:
            return DemoKind.CLICK
        return self._steps[self._index].demo

    def current_target(self) -> QWidget | None:
        if not self._steps:
            return None
        return self._steps[self._index].target()

    def discard(self) -> None:
        self._done = True
        self._stop_demo()
        self._release_demo_stage()
        self._advance_timer.stop()
        self._forward_grab = None
        self.hide()

    def begin(self) -> None:
        self._done = False
        self._index = 0
        self._show_step()
        self.show()
        self.reposition()
        self._start_demo()
        app = QApplication.instance()
        if app is not None and app.platformName() != "offscreen":
            self.raise_()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def advance(self) -> None:
        self._advance_timer.stop()
        if self._index >= len(self._steps) - 1:
            self._complete()
            return
        self._index += 1
        self._show_step()

    def notify_host_action(self, kind: str) -> None:
        if self._done or not self._steps:
            return
        if self._steps[self._index].advance_on != kind:
            return
        moved = False
        while self._index < len(self._steps) and self._steps[self._index].advance_on == kind:
            moved = True
            if self._index >= len(self._steps) - 1:
                self._complete()
                return
            self._index += 1
        if moved:
            self._show_step()

    def reposition(self) -> None:
        host = self.parentWidget()
        if host is None:
            return
        origin = host.mapToGlobal(QPoint(0, 0))
        self.setGeometry(QRect(origin, host.size()))
        self._refresh_hole()
        self._place_card()
        self._cursor = self._cursor_at(self._phase)
        app = QApplication.instance()
        if app is not None and app.platformName() != "offscreen":
            self.raise_()
        self.update()

    def _demo_tick(self, dt_ms: float = 16.0) -> None:
        if self._done:
            return
        step = float(dt_ms)
        if 0.0 < step <= 5.0:
            step *= 1000.0
        self._phase = (self._phase + step / DEMO_CYCLE_MS) % 1.0
        self._cursor = self._cursor_at(self._phase)
        self._press = self._press_at(self._phase)
        if not self._live_pointer:
            self._drive_demo()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        mask = QPainterPath()
        mask.addRect(self.rect())
        for rect in self._cutout_rects():
            hole = QPainterPath()
            hole.addRoundedRect(rect, HOLE_RADIUS, HOLE_RADIUS)
            mask -= hole
        painter.fillPath(mask, QColor(0, 0, 0, MASK_ALPHA))
        self._paint_pulse(painter)
        if not self._live_pointer:
            self._paint_demo(painter)
            self._paint_cursor(painter)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if self._forward_to_host(event, grab=True):
            return
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        self._update_live_pointer(_event_pos(event))
        if self._forward_to_host(event, grab=False):
            return
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._forward_to_host(event, grab=False):
            self._forward_grab = None
            return
        self._forward_grab = None
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        if self._forward_to_host(event, grab=False):
            return
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: ANN001
        del event
        self._live_pointer = False
        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if not self._is_interactive_pos(_event_pos(event)):
            event.ignore()
            return
        host = self.parentWidget()
        if host is None:
            event.ignore()
            return
        host.dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if self._is_interactive_pos(_event_pos(event)):
            event.acceptProposedAction()
            return
        host = self.parentWidget()
        hint = getattr(host, "_hint", None) if host is not None else None
        if hint is not None:
            hint.set_hover(False)
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: ANN001
        host = self.parentWidget()
        if host is not None:
            host.dragLeaveEvent(event)
            return
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        if not self._is_interactive_pos(_event_pos(event)):
            event.ignore()
            return
        host = self.parentWidget()
        if host is None:
            event.ignore()
            return
        host.dropEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Escape:
            self._complete()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._complete()
        super().closeEvent(event)

    def _show_step(self) -> None:
        if not self._steps:
            self._complete()
            return
        step = self._steps[self._index]
        if step.prepare is not None:
            step.prepare()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        self._title.setText(step.title)
        self._body.setText(step.body)
        self._counter.setText(f"{self._index + 1} / {len(self._steps)}")
        last = self._index >= len(self._steps) - 1
        self._next.setText("完成" if last else "下一步")
        self._phase = 0.0
        self._press = 0.0
        self._live_pointer = False
        self._forward_grab = None
        self._advance_timer.stop()
        self._halt_demo_input()
        self.reposition()
        self._sync_demo_stage()
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        self.reposition()
        if not self._done:
            self._start_demo()

    def _start_demo(self) -> None:
        if self._done:
            return
        if not self._demo_timer.isActive():
            self._demo_timer.start()
        self._cursor = self._cursor_at(self._phase)
        self._press = self._press_at(self._phase)
        if not self._live_pointer:
            self._drive_demo()

    def _stop_demo(self) -> None:
        self._demo_timer.stop()
        self._advance_timer.stop()
        self._halt_demo_input()

    def _is_interactive_pos(self, pos: QPoint) -> bool:
        if self._card.geometry().contains(pos):
            return False
        return any(rect.contains(pos) for rect in self._cutout_rects())

    def _cutout_rects(self) -> list[QRect]:
        rects: list[QRect] = []
        if self._hole.isValid() and not self._hole.isEmpty():
            rects.append(QRect(self._hole))
        if self.current_demo() in {DemoKind.BOX, DemoKind.LINE}:
            stage = self._picture_rect()
            if stage.isValid() and not stage.isEmpty():
                if not any(existing.contains(stage.center()) for existing in rects):
                    rects.append(stage)
        return rects

    def _update_live_pointer(self, pos: QPoint) -> None:
        live = self._is_interactive_pos(pos)
        if live and not self._live_pointer:
            self._halt_demo_input()
        if live == self._live_pointer:
            return
        self._live_pointer = live
        self.update()

    def _target_for_event(self, event) -> QWidget | None:  # noqa: ANN001
        if self._forward_grab is not None:
            try:
                self._forward_grab.isVisible()
            except RuntimeError:
                self._forward_grab = None
            else:
                return self._forward_grab
        global_pos = event.globalPosition().toPoint()
        return _hit_widget(self.parentWidget(), global_pos, self)

    def _send_mouse(self, target: QWidget, event: QMouseEvent) -> None:
        local = target.mapFromGlobal(event.globalPosition().toPoint())
        forwarded = QMouseEvent(
            event.type(),
            QPointF(local),
            event.globalPosition(),
            event.button(),
            event.buttons(),
            event.modifiers(),
        )
        QApplication.sendEvent(target, forwarded)

    def _forward_to_host(self, event: QMouseEvent, *, grab: bool) -> bool:
        pos = _event_pos(event)
        interactive = self._is_interactive_pos(pos) or self._forward_grab is not None
        if not interactive:
            return False
        target = self._target_for_event(event)
        if target is None or target is self or self.isAncestorOf(target):
            return False
        if grab:
            self._forward_grab = target
        self._send_mouse(target, event)
        if event.type() == QEvent.Type.MouseButtonPress:
            self._maybe_advance_on_click(target, pos)
        event.accept()
        return True

    def _maybe_advance_on_click(self, target: QWidget, overlay_pos: QPoint) -> None:
        if self._done or not self._steps:
            return
        step = self._steps[self._index]
        if step.advance_on != "click":
            return
        expected = self.current_target()
        if expected is not None:
            if target is expected or expected.isAncestorOf(target):
                self._schedule_advance()
                return
        if self.current_demo() in {DemoKind.BOX, DemoKind.LINE}:
            if self._stage_rect().contains(overlay_pos):
                self._schedule_advance()

    def _schedule_advance(self) -> None:
        if self._done:
            return
        if not self._advance_timer.isActive():
            self._advance_timer.start()

    def _advance_from_action(self) -> None:
        if self._done:
            return
        self.advance()

    def _video_widget(self):
        host = self.parentWidget()
        return None if host is None else getattr(host, "_video", None)

    def _overlay_to_widget(self, widget: QWidget, overlay_pos: QPoint) -> QPoint:
        return widget.mapFromGlobal(self.mapToGlobal(overlay_pos))

    def _inject_mouse(
        self,
        widget: QWidget,
        etype,
        pos: QPoint,
        button,
        buttons,
        modifiers=Qt.KeyboardModifier.NoModifier,
    ) -> None:
        event = QMouseEvent(
            etype,
            QPointF(pos),
            widget.mapToGlobal(pos),
            button,
            buttons,
            modifiers,
        )
        QApplication.sendEvent(widget, event)

    def _set_demo_down(self, button: QWidget | None) -> None:
        current = self._demo_down
        if current is button:
            return
        if current is not None:
            try:
                if hasattr(current, "setDown"):
                    current.setDown(False)
            except RuntimeError:
                pass
        self._demo_down = None
        if button is not None and hasattr(button, "setDown"):
            try:
                button.setDown(True)
                self._demo_down = button
            except RuntimeError:
                self._demo_down = None

    def _halt_demo_input(self) -> None:
        self._set_demo_down(None)
        host = self.parentWidget()
        hint = getattr(host, "_hint", None) if host is not None else None
        if hint is not None:
            hint.set_hover(False)
        video = self._video_widget()
        if self._demo_grabbing and video is not None:
            try:
                video.cancel_stroke()
            except RuntimeError:
                pass
        chart = self._chart_view()
        if self._demo_grabbing and chart is not None:
            try:
                chart._stop_scrub()
            except RuntimeError:
                pass
        self._demo_grabbing = False

    def _chart_view(self):
        host = self.parentWidget()
        panel = getattr(host, "_chart_panel", None) if host is not None else None
        charts = getattr(panel, "_charts", None) if panel is not None else None
        if not charts:
            return None
        return charts[0]

    def _sync_demo_stage(self) -> None:
        from app.widgets import MODE_RULER, MODE_TRACK

        demo = self.current_demo()
        if demo in {DemoKind.BOX, DemoKind.LINE}:
            video = self._ensure_demo_picture()
            if video is not None:
                want = MODE_RULER if demo is DemoKind.LINE else MODE_TRACK
                if video.interaction_mode() != want:
                    if self._saved_video_mode is None:
                        self._saved_video_mode = video.interaction_mode()
                    video.set_interaction_mode(want)
        else:
            self._release_demo_picture()
        if demo is DemoKind.SCRUB:
            self._ensure_demo_chart()
        else:
            self._release_demo_chart()

    def _ensure_demo_picture(self):
        host = self.parentWidget()
        video = self._video_widget()
        stack = getattr(host, "_stack", None) if host is not None else None
        if video is None:
            return None
        if getattr(host, "_info", None) is not None:
            if stack is not None:
                stack.setCurrentWidget(video)
            return video
        if not self._installed_preview:
            video.set_frame(tutorial_preview_image())
            if stack is not None:
                stack.setCurrentWidget(video)
            self._installed_preview = True
        return video

    def _release_demo_picture(self) -> None:
        from app.widgets import MODE_TRACK

        host = self.parentWidget()
        video = self._video_widget()
        if video is not None:
            try:
                video.cancel_stroke()
                if self._saved_video_mode is not None:
                    video.set_interaction_mode(self._saved_video_mode)
                elif video.interaction_mode() != MODE_TRACK:
                    video.set_interaction_mode(MODE_TRACK)
            except RuntimeError:
                pass
        self._saved_video_mode = None
        if self._installed_preview and getattr(host, "_info", None) is None:
            if video is not None:
                video.set_frame(None)
            stack = getattr(host, "_stack", None) if host is not None else None
            hint = getattr(host, "_hint", None) if host is not None else None
            if stack is not None and hint is not None:
                stack.setCurrentWidget(hint)
        self._installed_preview = False

    def _ensure_demo_chart(self) -> None:
        host = self.parentWidget()
        panel = getattr(host, "_chart_panel", None) if host is not None else None
        chart = self._chart_view()
        if panel is None or chart is None:
            return
        if chart._samples:
            return
        panel.set_samples(_demo_chart_samples())
        self._installed_chart = True

    def _release_demo_chart(self) -> None:
        if not self._installed_chart:
            return
        host = self.parentWidget()
        panel = getattr(host, "_chart_panel", None) if host is not None else None
        if panel is not None:
            panel.set_samples([])
        self._installed_chart = False

    def _release_demo_stage(self) -> None:
        self._halt_demo_input()
        self._release_demo_picture()
        self._release_demo_chart()

    def _drive_demo(self) -> None:
        if self._done:
            return
        demo = self.current_demo()
        t = self._phase
        cursor = self._cursor
        if demo is DemoKind.DROP:
            host = self.parentWidget()
            hint = getattr(host, "_hint", None) if host is not None else None
            if hint is not None and hint.isVisible():
                hovering = self._hole.contains(cursor) and t < 0.82
                hint.set_hover(hovering)
            self._set_demo_down(None)
            return
        if demo is DemoKind.CLICK:
            pressing = 0.46 <= t <= 0.74
            self._set_demo_down(self.current_target() if pressing else None)
            return
        if demo is DemoKind.PLAY:
            host = self.parentWidget()
            button = getattr(host, "_play_btn", None) if host is not None else None
            pressing = 0.46 <= t <= 0.74
            self._set_demo_down(button if pressing else None)
            return
        if demo is DemoKind.BOX:
            self._drive_box(t, cursor)
            return
        if demo is DemoKind.LINE:
            self._drive_line(t, cursor)
            return
        if demo is DemoKind.SCRUB:
            self._drive_scrub(t, cursor)

    def _drive_box(self, t: float, cursor: QPoint) -> None:
        pressing_tool = 0.22 <= t < 0.34
        self._set_demo_down(self.current_target() if pressing_tool else None)
        video = self._ensure_demo_picture()
        if video is None:
            return
        if t < 0.42 or t > 0.88:
            if self._demo_grabbing:
                video.cancel_stroke()
                self._demo_grabbing = False
            return
        pos = self._overlay_to_widget(video, cursor)
        mods = Qt.KeyboardModifier.ControlModifier
        if not self._demo_grabbing:
            self._inject_mouse(
                video,
                QEvent.Type.MouseButtonPress,
                pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                mods,
            )
            self._demo_grabbing = True
            return
        self._inject_mouse(
            video,
            QEvent.Type.MouseMove,
            pos,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            mods,
        )

    def _drive_line(self, t: float, cursor: QPoint) -> None:
        pressing_tool = 0.22 <= t < 0.34
        self._set_demo_down(self.current_target() if pressing_tool else None)
        video = self._ensure_demo_picture()
        if video is None:
            return
        if t < 0.42 or t > 0.88:
            if self._demo_grabbing:
                video.cancel_stroke()
                self._demo_grabbing = False
            return
        pos = self._overlay_to_widget(video, cursor)
        if not self._demo_grabbing:
            self._inject_mouse(
                video,
                QEvent.Type.MouseButtonPress,
                pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
            )
            self._demo_grabbing = True
            return
        self._inject_mouse(
            video,
            QEvent.Type.MouseMove,
            pos,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
        )

    def _drive_scrub(self, t: float, cursor: QPoint) -> None:
        self._set_demo_down(None)
        chart = self._chart_view()
        if chart is None:
            return
        viewport = chart.viewport()
        pos = self._overlay_to_widget(viewport, cursor)
        if t < 0.04 or t > 0.96:
            if self._demo_grabbing:
                self._inject_mouse(
                    viewport,
                    QEvent.Type.MouseButtonRelease,
                    pos,
                    Qt.MouseButton.LeftButton,
                    Qt.MouseButton.NoButton,
                )
                self._demo_grabbing = False
            return
        if not self._demo_grabbing:
            self._inject_mouse(
                viewport,
                QEvent.Type.MouseButtonPress,
                pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
            )
            self._demo_grabbing = True
            return
        self._inject_mouse(
            viewport,
            QEvent.Type.MouseMove,
            pos,
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
        )

    def _map_widget(self, widget: QWidget | None) -> QRect:
        if widget is None or not widget.isVisible():
            return QRect()
        top_left = self.mapFromGlobal(widget.mapToGlobal(QPoint(0, 0)))
        return QRect(top_left, widget.size())

    def _aim_rect(self) -> QRect:
        if self.current_demo() is DemoKind.PLAY:
            host = self.parentWidget()
            button = getattr(host, "_play_btn", None) if host is not None else None
            mapped = self._map_widget(button)
            if mapped.isValid() and not mapped.isEmpty():
                return mapped
        if self._hole.isValid() and not self._hole.isEmpty():
            return QRect(self._hole)
        return QRect(self.rect().center(), self.rect().center())

    def _stage_rect(self) -> QRect:
        host = self.parentWidget()
        stage = getattr(host, "_stage", None) if host is not None else None
        mapped = self._map_widget(stage)
        if mapped.isValid() and mapped.width() > 40 and mapped.height() > 40:
            return mapped
        return self.rect().adjusted(
            int(self.width() * 0.12),
            int(self.height() * 0.22),
            -int(self.width() * 0.38),
            -int(self.height() * 0.28),
        )

    def _picture_rect(self) -> QRect:
        video = self._video_widget()
        dest = video.content_rect() if video is not None else None
        if dest is not None and dest.width() > 40 and dest.height() > 40:
            top_left = self.mapFromGlobal(video.mapToGlobal(dest.topLeft().toPoint()))
            return QRect(top_left, dest.size().toSize())
        return self._stage_rect()

    def _approach_start(self, dest: QPoint) -> QPoint:
        return QPoint(dest.x() - 56, dest.y() - 42)

    def _cursor_at(self, t: float) -> QPoint:
        demo = self.current_demo()
        aim = self._aim_rect()
        dest = aim.center()
        start = self._approach_start(dest)
        if demo is DemoKind.DROP:
            hole = self._hole if self._hole.isValid() else aim
            origin = QPoint(hole.center().x() - 24, hole.top() - 90)
            land = hole.center()
            if t < 0.72:
                return _along(origin, land, t / 0.72)
            return land
        if demo is DemoKind.SCRUB:
            hole = self._hole if self._hole.isValid() else aim
            pad = min(28, max(10, hole.width() // 8))
            left = QPoint(hole.left() + pad, hole.center().y())
            right = QPoint(hole.right() - pad, hole.center().y())
            wave = 0.5 - 0.5 * math.cos(2.0 * math.pi * t)
            return _along(left, right, wave)
        if demo is DemoKind.BOX:
            if t < 0.34:
                return _along(start, dest, t / 0.34)
            stage = self._picture_rect()
            box = self._demo_box(stage, 1.0)
            grab = QPoint(box.left(), box.top())
            if t < 0.42:
                return _along(dest, grab, (t - 0.34) / 0.08)
            end = QPoint(box.right(), box.bottom())
            grow = _clamp01((t - 0.42) / 0.40)
            return _along(grab, end, grow)
        if demo is DemoKind.LINE:
            if t < 0.34:
                return _along(start, dest, t / 0.34)
            stage = self._picture_rect()
            a, b = self._demo_line(stage)
            if t < 0.42:
                return _along(dest, a, (t - 0.34) / 0.08)
            grow = _clamp01((t - 0.42) / 0.40)
            return _along(a, b, grow)
        if t < 0.46:
            return _along(start, dest, t / 0.46)
        return dest

    def _press_at(self, t: float) -> float:
        demo = self.current_demo()
        if demo in {DemoKind.DROP, DemoKind.SCRUB}:
            return 0.0
        if demo in {DemoKind.BOX, DemoKind.LINE}:
            t0, t1 = 0.28, 0.38
        else:
            t0, t1 = 0.46, 0.62
        if t < t0:
            return 0.0
        if t > t1:
            return max(0.0, 1.0 - (t - t1) / 0.12)
        return _smooth((t - t0) / (t1 - t0))

    def _demo_box(self, stage: QRect, grow: float) -> QRect:
        grow = _clamp01(grow)
        left = stage.left() + int(stage.width() * 0.22)
        top = stage.top() + int(stage.height() * 0.28)
        width = int(stage.width() * 0.38 * max(0.12, grow))
        height = int(stage.height() * 0.32 * max(0.12, grow))
        return QRect(left, top, max(8, width), max(8, height))

    def _demo_line(self, stage: QRect) -> tuple[QPoint, QPoint]:
        y = stage.top() + int(stage.height() * 0.58)
        a = QPoint(stage.left() + int(stage.width() * 0.18), y)
        b = QPoint(stage.left() + int(stage.width() * 0.62), y - int(stage.height() * 0.08))
        return a, b

    def _paint_pulse(self, painter: QPainter) -> None:
        rects = self._cutout_rects()
        if not rects:
            return
        pulse = 0.5 + 0.5 * math.sin(self._phase * 2.0 * math.pi)
        width = 2.0 + 2.2 * pulse
        alpha = int(110 + 120 * pulse)
        color = QColor(HIGHLIGHT)
        color.setAlpha(alpha)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, width))
        inflate = int(round(3 * pulse))
        for rect in rects:
            painter.drawRoundedRect(
                rect.adjusted(-inflate, -inflate, inflate, inflate),
                HOLE_RADIUS + inflate * 0.3,
                HOLE_RADIUS + inflate * 0.3,
            )

    def _paint_demo(self, painter: QPainter) -> None:
        demo = self.current_demo()
        t = self._phase
        if demo is DemoKind.DROP:
            chip = QPoint(self._cursor.x() - 6, self._cursor.y() - 22)
            fade = 1.0 if t < 0.78 else max(0.0, 1.0 - (t - 0.78) / 0.16)
            self._paint_file_chip(painter, chip, fade)

    def _paint_file_chip(self, painter: QPainter, pos: QPoint, fade: float) -> None:
        if fade <= 0.02:
            return
        rect = QRect(pos.x(), pos.y(), 52, 36)
        fill = QColor(45, 45, 45, int(230 * fade))
        edge = QColor(90, 90, 90, int(240 * fade))
        painter.setBrush(fill)
        painter.setPen(QPen(edge, 1))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QColor(230, 230, 230, int(255 * fade)))
        font = QFont()
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "mp4")

    def _paint_cursor(self, painter: QPainter) -> None:
        pos = self._cursor
        if pos.isNull() and self._phase == 0.0:
            pos = self._cursor_at(0.0)
        if self._press > 0.02:
            ring = QColor(255, 255, 255, int(160 * self._press))
            radius = 7 + int(10 * self._press)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(ring, 2))
            painter.drawEllipse(pos, radius, radius)
        body = QPolygon(
            [
                pos,
                pos + QPoint(2, 18),
                pos + QPoint(6, 14),
                pos + QPoint(11, 24),
                pos + QPoint(14, 22),
                pos + QPoint(8, 12),
                pos + QPoint(16, 12),
            ]
        )
        painter.setBrush(QColor("#f4f4f4"))
        painter.setPen(QPen(QColor("#1a1a1a"), 1))
        painter.drawPolygon(body)

    def _refresh_hole(self) -> None:
        target = self.current_target()
        if target is None or not target.isVisible():
            self._hole = QRect()
            return
        top_left = self.mapFromGlobal(target.mapToGlobal(QPoint(0, 0)))
        rect = QRect(top_left, target.size()).adjusted(
            -HOLE_PAD, -HOLE_PAD, HOLE_PAD, HOLE_PAD
        )
        bounds = self.rect().adjusted(4, 4, -4, -4)
        self._hole = rect.intersected(bounds)

    def _place_card(self) -> None:
        self._card.adjustSize()
        hint = self._card.sizeHint()
        width = max(CARD_WIDTH, hint.width())
        height = hint.height()
        self._card.setFixedWidth(width)
        hole = self._hole if self._hole.isValid() else QRect(self.rect().center(), self.rect().center())
        x = hole.center().x() - width // 2
        y = hole.bottom() + CARD_MARGIN
        max_x = max(CARD_MARGIN, self.width() - width - CARD_MARGIN)
        x = min(max(CARD_MARGIN, x), max_x)
        if y + height + CARD_MARGIN > self.height():
            y = hole.top() - height - CARD_MARGIN
        if y < CARD_MARGIN:
            y = CARD_MARGIN
        max_y = max(CARD_MARGIN, self.height() - height - CARD_MARGIN)
        y = min(y, max_y)
        self._card.setGeometry(x, y, width, height)
        app = QApplication.instance()
        if app is None or app.platformName() != "offscreen":
            self._card.raise_()

    def _complete(self) -> None:
        if self._done:
            return
        self._done = True
        self._stop_demo()
        self._release_demo_stage()
        self._forward_grab = None
        mark_tutorial_seen(self._settings)
        self.hide()
        self.finished.emit()


def start_tutorial(
    window: QWidget,
    *,
    settings: QSettings | None = None,
    steps: list[TourStep] | None = None,
) -> TutorialOverlay | None:
    tour = steps if steps is not None else default_steps(window)
    if not tour:
        return None
    existing = getattr(window, "_tutorial_overlay", None)
    if existing is not None:
        try:
            existing.finished.disconnect()
        except RuntimeError:
            pass
        existing.discard()
        existing.deleteLater()
        window._tutorial_overlay = None  # type: ignore[attr-defined]
    overlay = TutorialOverlay(window, tour, settings)
    overlay.finished.connect(lambda: _on_overlay_finished(window))
    window._tutorial_overlay = overlay  # type: ignore[attr-defined]
    overlay.begin()
    return overlay


def _on_overlay_finished(window: QWidget) -> None:
    handler = getattr(window, "_on_tutorial_finished", None)
    window._tutorial_overlay = None  # type: ignore[attr-defined]
    if callable(handler):
        handler()


def maybe_start_tutorial(
    window: QWidget,
    *,
    force: bool = False,
    settings: QSettings | None = None,
) -> bool:
    store = settings or QSettings()
    if not force:
        flag = os.environ.get("TRACKLAB_SKIP_TUTORIAL", "").strip().lower()
        if flag in {"1", "true", "yes"}:
            return False
        if "--smoke" in sys.argv:
            return False
        if tutorial_seen(store):
            return False
    overlay = start_tutorial(window, settings=store)
    return overlay is not None
