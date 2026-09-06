"""First-run spotlight tour over the main window."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QSettings, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
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


@dataclass(frozen=True)
class TourStep:
    title: str
    body: str
    target: Callable[[], QWidget | None]
    prepare: Callable[[], None] | None = None


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
            "把 mp4 / mov 拖进画面，或点这里打开文件，开始分析。",
            drop_target,
        ),
        TourStep(
            "打开视频",
            "也可以用顶栏按钮打开视频或项目。",
            tool("open"),
        ),
        TourStep(
            "跟踪目标",
            "打开轨迹管理器，框选或点击目标。按 T 开始自动跟踪：快速用 Tiny 隔帧，精准用 Small 逐帧。",
            tool("track"),
        ),
        TourStep(
            "标定",
            "用标定尺把像素换成真实长度，或在坐标系菜单里设置运动平面。",
            tool("ruler"),
        ),
        TourStep(
            "分图",
            "位移、速度等曲线画在这里。可拖动 scrub 对齐当前帧。",
            chart_dock,
            prepare=ensure_chart,
        ),
        TourStep(
            "数据表",
            "每帧数值在数据表里。需要时从文件菜单导出 CSV / JSON。",
            data_dock,
            prepare=ensure_data,
        ),
        TourStep(
            "AI 助手",
            "识别实验类型、问答和生成报告。首次使用请在菜单里填写 DeepSeek 密钥。",
            tool("ai"),
        ),
        TourStep(
            "播放与逐帧",
            "空格播放或暂停，左右方向键按底栏步长逐帧移动。",
            transport,
        ),
    ]


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

    def current_target(self) -> QWidget | None:
        if not self._steps:
            return None
        return self._steps[self._index].target()

    def discard(self) -> None:
        self._done = True
        self.hide()

    def begin(self) -> None:
        self._done = False
        self._index = 0
        self._show_step()
        self.show()
        self.reposition()
        app = QApplication.instance()
        if app is not None and app.platformName() != "offscreen":
            self.raise_()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def advance(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._complete()
            return
        self._index += 1
        self._show_step()

    def reposition(self) -> None:
        host = self.parentWidget()
        if host is None:
            return
        origin = host.mapToGlobal(QPoint(0, 0))
        self.setGeometry(QRect(origin, host.size()))
        self._refresh_hole()
        self._place_card()
        app = QApplication.instance()
        if app is not None and app.platformName() != "offscreen":
            self.raise_()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        mask = QPainterPath()
        mask.addRect(self.rect())
        if self._hole.isValid() and not self._hole.isEmpty():
            hole = QPainterPath()
            hole.addRoundedRect(self._hole, HOLE_RADIUS, HOLE_RADIUS)
            mask -= hole
        painter.fillPath(mask, QColor(0, 0, 0, MASK_ALPHA))
        if self._hole.isValid() and not self._hole.isEmpty():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(HIGHLIGHT, 2))
            painter.drawRoundedRect(self._hole, HOLE_RADIUS, HOLE_RADIUS)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        event.accept()

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
        self.reposition()

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
