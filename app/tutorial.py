"""First-run spotlight tour over the main window."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QShortcut,
    QKeySequence,
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
CARD_WIDTH = 420
CARD_MARGIN = 16
MASK_ALPHA = 150
HIGHLIGHT = QColor("#4da3ff")
BALL = QColor("#e07a3a")
BALL_HI = QColor("#f0b27a")
MASK = QColor(77, 163, 255, 70)
MASK_EDGE = QColor("#4da3ff")
TRAIL = QColor("#80cbc4")
WARN = QColor("#f0c14b")
STAGE_BG = QColor("#1a1c20")
STAGE_FLOOR = QColor("#24262c")
STAGE_W = 384
STAGE_H = 216
DEMO_CYCLE_MS = 3000
DEMO_TICK_MS = 16


class DemoKind(str, Enum):
    DROP = "drop"
    OPEN = "open"
    NEW_TRACK = "new_track"
    BOX = "box"
    SEED_POINT = "seed_point"
    TRACK_RUN = "track_run"
    FAST_PRECISE = "fast_precise"
    FIX_POINT = "fix_point"
    RANGE = "range"
    RULER = "ruler"
    AXIS = "axis"
    CHART = "chart"
    TABLE = "table"
    READOUT = "readout"
    ASSISTANT = "assistant"
    PLAY = "play"
    SAVE = "save"
    CLICK = "click"


@dataclass(frozen=True)
class TourStep:
    title: str
    body: str
    target: Callable[[], QWidget | None]
    prepare: Callable[[], None] | None = None
    demo: DemoKind = DemoKind.CLICK


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

    def ensure_tracks() -> None:
        toggle = getattr(window, "_toggle_track_window", None)
        if callable(toggle):
            toggle()
        action = getattr(window, "_track_window_action", None)
        if action is not None:
            action.setChecked(True)
        manager = getattr(window, "_track_window", None)
        if manager is not None:
            manager.show()
            manager.raise_()

    def chart_dock() -> QWidget | None:
        workspace = getattr(window, "_workspace", None)
        return None if workspace is None else workspace.chart_dock

    def data_dock() -> QWidget | None:
        workspace = getattr(window, "_workspace", None)
        return None if workspace is None else workspace.data_dock

    def transport() -> QWidget | None:
        return getattr(window, "_transport", None)

    def view_bar() -> QWidget | None:
        return getattr(window, "_view_bar", None)

    def track_button() -> QWidget | None:
        return tool("track")()

    def analysis_menu() -> QWidget | None:
        bar = window.menuBar() if hasattr(window, "menuBar") else None
        return bar if bar is not None else tool("track")()

    return [
        TourStep(
            "导入视频",
            "TrackLab 从实验录像里取出位移和速度。把 mp4 / mov 拖进这块区域，或点它选文件。"
            "打开只建帧索引，不会先转码。下面是动画示范，也可以自己拖入。翻页请点下一步。",
            drop_target,
            demo=DemoKind.DROP,
        ),
        TourStep(
            "打开视频",
            "也可以点顶栏这个打开按钮。文件菜单同样能打开视频，或打开上次保存的项目。"
            "打开后教程不会自动翻页，看完动画再点下一步。",
            tool("open"),
            demo=DemoKind.OPEN,
        ),
        TourStep(
            "新建轨迹",
            "点轨迹按钮打开管理器，再点新建。一条视频可以跟多个物体，每条轨迹单独框、单独跟。"
            "先建好轨迹，再在画面上指定目标。",
            track_button,
            prepare=ensure_tracks,
            demo=DemoKind.NEW_TRACK,
        ),
        TourStep(
            "框选目标",
            "按住 Control 拖出方框（macOS 也可用右键拖）。这个框是给 SAM 2 的目标提示："
            "告诉模型「跟这个物体」，不是搜索范围，也不会裁剪画面，所以框大框小几乎不影响速度。"
            "框要紧贴目标。想更快，缩小进度条上的分析区间，或改用快速模式。",
            tool("track"),
            demo=DemoKind.BOX,
        ),
        TourStep(
            "加提示点",
            "Shift+Control 点击可以在目标上加点，适合框不好画的小物体。"
            "当前界面只有正点（「就是这个」），没有负点入口。框和点可以一起用，然后按 T 跟踪。",
            tool("track"),
            demo=DemoKind.SEED_POINT,
        ),
        TourStep(
            "按 T 跟踪",
            "按 T 或点「SAM 自动跟踪」。推理在独立线程跑，画面还能拖进度条。"
            "再按一次 T 取消。状态栏会显示本次跟踪的帧范围。下面动画是小球抛出后掩膜跟着走。",
            tool("track"),
            demo=DemoKind.TRACK_RUN,
        ),
        TourStep(
            "快速 / 精准",
            "轨迹菜单里选：快速预览用 Tiny，隔帧推理再插值，适合先看跟没跟对；"
            "精准分析用 Small，逐帧更稳，适合最终数据。两条抛物线：上面点稀，下面点密。",
            analysis_menu,
            demo=DemoKind.FAST_PRECISE,
        ),
        TourStep(
            "跟丢了怎么修",
            "某一帧偏了，点那一帧画面上的正确位置。再按 T，会从这一帧往后重跟，前面的点保留。"
            "黄色是低可信，不要直接拿去算。",
            tool("track"),
            demo=DemoKind.FIX_POINT,
        ),
        TourStep(
            "跟踪范围与背景补偿",
            "进度条上方两个三角是分析区间，I / O 把它们设到当前帧。跟踪只跑到结束三角。"
            "背景补偿用来减镜头晃，质量不够会自动保持关闭，不要强开。",
            transport,
            demo=DemoKind.RANGE,
        ),
        TourStep(
            "标定尺",
            "点尺子，在画面上拖出一段已知长度，例如直尺或桌边。只画一把尺时，整张图用同一个比例。"
            "近、远各画一把，用来补偿透视。填真实长度后，坐标从像素变成米。没有标定，分图和表格仍是像素。",
            tool("ruler"),
            demo=DemoKind.RULER,
        ),
        TourStep(
            "坐标系",
            "点坐标轴工具，把原点拖到参考点，例如抛出点或桌面。X、Y 方向可以转动。"
            "原点和标定一起决定位移、速度的正方向。",
            tool("axis"),
            demo=DemoKind.AXIS,
        ),
        TourStep(
            "分图",
            "分图像 x(t)、y(t)、速度这样随时间变化。按住曲线左右拖，当前帧会跟着走。滚轮缩放。"
            "头部可选速度算法（稳健拟合 / Tracker 差分）、窗口和拟合。点图上的点也会跳帧。",
            chart_dock,
            prepare=ensure_chart,
            demo=DemoKind.CHART,
        ),
        TourStep(
            "数据表",
            "每一帧的 t、x、y 列在这里。点某一行跳到那一帧。黄色表示这一帧跟踪不可信。"
            "文件菜单可导出 CSV 或 JSON。",
            data_dock,
            prepare=ensure_data,
            demo=DemoKind.TABLE,
        ),
        TourStep(
            "当前读数",
            "第二栏是当前轨迹和这一帧的 t、x、y。标定后单位会从 px 变成 m。"
            "左右三角跳到上一个或下一个有效点。",
            view_bar,
            demo=DemoKind.READOUT,
        ),
        TourStep(
            "AI 助手",
            "助手根据轨迹和标定判断实验类型，再用 DeepSeek 讲解公式。"
            "首次使用请到编辑菜单填写密钥。不会上传原始视频，只发送结构化数据。",
            tool("ai"),
            demo=DemoKind.ASSISTANT,
        ),
        TourStep(
            "播放与保存",
            "空格播放或暂停。左右方向键按底栏步长逐帧。分析区间外的进度会被拉回。"
            "右下角循环按钮只决定播到终点是回绕还是停住。分析完点保存，下次打开项目时轨迹和标定都在。"
            "帮助菜单的「快速开始」可以再看一遍。",
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


def _projectile(t: float) -> tuple[float, float]:
    u = _clamp01(t)
    x = 0.16 + 0.68 * u
    y = 0.74 - 1.76 * u * (1.0 - u)
    return x, y


class TutorialStage(QWidget):
    """Self-contained animated demo. Never touches the real UI."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("tutorialStage")
        self.setFixedSize(STAGE_W, STAGE_H)
        self._kind = DemoKind.DROP
        self._phase = 0.0

    @property
    def demo_kind(self) -> DemoKind:
        return self._kind

    @property
    def demo_phase(self) -> float:
        return self._phase

    def set_demo(self, kind: DemoKind, phase: float) -> None:
        self._kind = kind
        self._phase = phase % 1.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), STAGE_BG)
        painters = {
            DemoKind.DROP: self._paint_drop,
            DemoKind.OPEN: self._paint_open,
            DemoKind.NEW_TRACK: self._paint_new_track,
            DemoKind.BOX: self._paint_box,
            DemoKind.SEED_POINT: self._paint_seed,
            DemoKind.TRACK_RUN: self._paint_track_run,
            DemoKind.FAST_PRECISE: self._paint_fast_precise,
            DemoKind.FIX_POINT: self._paint_fix,
            DemoKind.RANGE: self._paint_range,
            DemoKind.RULER: self._paint_ruler,
            DemoKind.AXIS: self._paint_axis,
            DemoKind.CHART: self._paint_chart,
            DemoKind.TABLE: self._paint_table,
            DemoKind.READOUT: self._paint_readout,
            DemoKind.ASSISTANT: self._paint_assistant,
            DemoKind.PLAY: self._paint_play,
            DemoKind.SAVE: self._paint_save,
            DemoKind.CLICK: self._paint_open,
        }
        painters.get(self._kind, self._paint_open)(painter)

    def _pt(self, x: float, y: float) -> QPointF:
        return QPointF(x * self.width(), y * self.height())

    def _rect(self, x: float, y: float, w: float, h: float) -> QRectF:
        return QRectF(x * self.width(), y * self.height(), w * self.width(), h * self.height())

    def _font(self, px: int, bold: bool = False) -> QFont:
        font = QFont()
        font.setPixelSize(px)
        font.setBold(bold)
        return font

    def _scene(self, painter: QPainter) -> None:
        painter.fillRect(self._rect(0.0, 0.76, 1.0, 0.24), STAGE_FLOOR)
        painter.setPen(QPen(QColor("#3a3d44"), 1))
        painter.drawLine(self._pt(0.0, 0.76), self._pt(1.0, 0.76))

    def _ball(self, painter: QPainter, x: float, y: float, r: float = 0.045) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 70))
        painter.drawEllipse(self._pt(x + 0.008, 0.76), r * self.width() * 0.9, 4)
        painter.setBrush(BALL)
        painter.drawEllipse(self._pt(x, y), r * self.width(), r * self.width())
        painter.setBrush(BALL_HI)
        painter.drawEllipse(self._pt(x - r * 0.35, y - r * 0.35), r * self.width() * 0.28, r * self.width() * 0.28)

    def _mask(self, painter: QPainter, x: float, y: float, r: float = 0.07) -> None:
        painter.setBrush(MASK)
        painter.setPen(QPen(MASK_EDGE, 1.6))
        painter.drawEllipse(self._pt(x, y), r * self.width(), r * self.width())

    def _cursor(self, painter: QPainter, x: float, y: float, press: float = 0.0) -> None:
        pos = self._pt(x, y)
        if press > 0.02:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, int(160 * press)), 2))
            radius = 6 + int(8 * press)
            painter.drawEllipse(pos, radius, radius)
        body = QPolygonF(
            [
                pos,
                pos + QPointF(2, 18),
                pos + QPointF(6, 14),
                pos + QPointF(11, 24),
                pos + QPointF(14, 22),
                pos + QPointF(8, 12),
                pos + QPointF(16, 12),
            ]
        )
        painter.setBrush(QColor("#f4f4f4"))
        painter.setPen(QPen(QColor("#1a1a1a"), 1))
        painter.drawPolygon(body)

    def _label(self, painter: QPainter, text: str, x: float, y: float, color: QColor | None = None) -> None:
        painter.setFont(self._font(11, True))
        painter.setPen(color or QColor("#e6e6e6"))
        painter.drawText(self._pt(x, y), text)

    def _chip(self, painter: QPainter, x: float, y: float, fade: float) -> None:
        if fade <= 0.02:
            return
        rect = self._rect(x, y, 0.16, 0.16)
        painter.setBrush(QColor(45, 45, 45, int(230 * fade)))
        painter.setPen(QPen(QColor(90, 90, 90, int(240 * fade)), 1))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setFont(self._font(12, True))
        painter.setPen(QColor(230, 230, 230, int(255 * fade)))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "mp4")

    def _timeline(self, painter: QPainter, start: float, end: float, play: float) -> None:
        groove = self._rect(0.08, 0.86, 0.84, 0.04)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#3a3a3a"))
        painter.drawRoundedRect(groove, 2, 2)
        inner = QRectF(
            groove.left() + groove.width() * start,
            groove.top(),
            groove.width() * max(0.02, end - start),
            groove.height(),
        )
        painter.setBrush(QColor("#4da3ff"))
        painter.drawRoundedRect(inner, 2, 2)
        for edge in (start, end):
            x = groove.left() + groove.width() * edge
            tri = QPainterPath()
            tri.moveTo(x - 5, groove.top() - 10)
            tri.lineTo(x + 5, groove.top() - 10)
            tri.lineTo(x, groove.top() - 2)
            tri.closeSubpath()
            painter.fillPath(tri, QColor("#c4c4c4"))
        px = groove.left() + groove.width() * play
        head = QPainterPath()
        head.moveTo(px - 5, groove.bottom() + 10)
        head.lineTo(px + 5, groove.bottom() + 10)
        head.lineTo(px, groove.bottom() + 2)
        head.closeSubpath()
        painter.fillPath(head, QColor("#e2e2e2"))

    def _paint_drop(self, painter: QPainter) -> None:
        t = self._phase
        box = self._rect(0.18, 0.22, 0.64, 0.52)
        hover = t < 0.78
        painter.setBrush(QColor("#2a3340") if hover else QColor("#232323"))
        pen = QPen(HIGHLIGHT if hover else QColor("#5a5a5a"), 1.6)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRoundedRect(box, 10, 10)
        self._label(painter, "拖入视频", 0.38, 0.50)
        land = 0.42
        y = _mix(0.02, land, _smooth(min(t / 0.72, 1.0)))
        fade = 1.0 if t < 0.82 else max(0.0, 1.0 - (t - 0.82) / 0.14)
        self._chip(painter, 0.42, y, fade)

    def _paint_open(self, painter: QPainter) -> None:
        t = self._phase
        btn = self._rect(0.12, 0.12, 0.22, 0.16)
        pressed = 0.42 <= t <= 0.62
        painter.setBrush(QColor("#191919") if pressed else QColor("#383838"))
        painter.setPen(QPen(QColor("#4da3ff" if pressed else "#5a5a5a"), 1))
        painter.drawRoundedRect(btn, 8, 8)
        self._label(painter, "打开", 0.17, 0.23)
        appear = _smooth((t - 0.55) / 0.25) if t > 0.55 else 0.0
        if appear > 0:
            painter.setOpacity(appear)
            self._scene(painter)
            self._ball(painter, 0.32, 0.62)
            painter.setOpacity(1.0)
        self._cursor(painter, _mix(0.06, 0.20, _smooth(min(t / 0.42, 1.0))), 0.18, 1.0 if pressed else 0.0)

    def _paint_new_track(self, painter: QPainter) -> None:
        t = self._phase
        panel = self._rect(0.08, 0.10, 0.84, 0.78)
        painter.setBrush(QColor("#232323"))
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        painter.drawRoundedRect(panel, 8, 8)
        self._label(painter, "轨迹", 0.14, 0.22)
        btn = self._rect(0.62, 0.14, 0.22, 0.12)
        painter.setBrush(QColor("#2d4a66"))
        painter.setPen(QPen(HIGHLIGHT, 1))
        painter.drawRoundedRect(btn, 4, 4)
        painter.setFont(self._font(11))
        painter.setPen(QColor("#f0f0f0"))
        painter.drawText(btn, int(Qt.AlignmentFlag.AlignCenter), "新建")
        if t > 0.45:
            row = self._rect(0.14, 0.36, 0.72, 0.18)
            painter.setBrush(QColor("#2d2d2d"))
            painter.setPen(QPen(QColor("#4da3ff"), 1))
            painter.drawRoundedRect(row, 4, 4)
            painter.setPen(QColor("#80cbc4"))
            painter.drawText(self._pt(0.18, 0.48), "轨迹 1")
        cx = _mix(0.18, 0.72, _smooth(min(t / 0.40, 1.0)))
        cy = _mix(0.70, 0.20, _smooth(min(t / 0.40, 1.0)))
        self._cursor(painter, cx, cy, 1.0 if 0.40 <= t <= 0.55 else 0.0)

    def _paint_box(self, painter: QPainter) -> None:
        t = self._phase
        self._scene(painter)
        self._ball(painter, 0.48, 0.62)
        grow = _smooth(_clamp01((t - 0.18) / 0.45))
        left = 0.48 - 0.10 * grow
        top = 0.62 - 0.12 * grow
        w = 0.20 * max(0.15, grow)
        h = 0.22 * max(0.15, grow)
        painter.setBrush(QColor(77, 163, 255, 35))
        painter.setPen(QPen(HIGHLIGHT, 1.6))
        painter.drawRect(self._rect(left, top, w, h))
        self._label(painter, "目标提示，不是范围", 0.28, 0.16, QColor("#b8d4ff"))
        self._cursor(
            painter,
            left + w,
            top + h if t > 0.18 else 0.28,
            1.0 if t > 0.18 else 0.0,
        )

    def _paint_seed(self, painter: QPainter) -> None:
        t = self._phase
        self._scene(painter)
        self._ball(painter, 0.48, 0.62)
        if t > 0.40:
            painter.setBrush(QColor("#4da3ff"))
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(self._pt(0.48, 0.62), 6, 6)
        self._label(painter, "Shift+Control 正点", 0.30, 0.16, QColor("#b8d4ff"))
        cx = _mix(0.20, 0.48, _smooth(min(t / 0.40, 1.0)))
        cy = _mix(0.22, 0.62, _smooth(min(t / 0.40, 1.0)))
        self._cursor(painter, cx, cy, 1.0 if 0.40 <= t <= 0.58 else 0.0)

    def _paint_track_run(self, painter: QPainter) -> None:
        t = self._phase
        self._scene(painter)
        flight = _smooth(t)
        x, y = _projectile(flight)
        trail_n = max(1, int(flight * 14))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(TRAIL)
        for i in range(trail_n):
            u = i / 14
            px, py = _projectile(u)
            painter.drawEllipse(self._pt(px, py), 3, 3)
        self._mask(painter, x, y)
        self._ball(painter, x, y)
        self._timeline(painter, 0.0, 1.0, flight)
        self._label(painter, "按 T 跟踪", 0.08, 0.10)

    def _paint_fast_precise(self, painter: QPainter) -> None:
        painter.fillRect(self._rect(0.0, 0.0, 1.0, 0.5), STAGE_BG)
        painter.fillRect(self._rect(0.0, 0.5, 1.0, 0.5), QColor("#16181c"))
        self._label(painter, "Tiny 隔帧", 0.06, 0.10, QColor("#8f8f8f"))
        self._label(painter, "Small 逐帧", 0.06, 0.60, QColor("#8f8f8f"))
        t = _smooth(self._phase)
        for row, step, dash in ((0.0, 2, True), (0.5, 1, False)):
            path = QPainterPath()
            first = True
            samples = 24 if step == 1 else 12
            for i in range(samples + 1):
                u = i / samples * t
                x, y = _projectile(u)
                pt = self._pt(x, row + y * 0.42)
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            pen = QPen(TRAIL if not dash else QColor("#80cbc4"), 1.6)
            if dash:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(BALL)
            count = 8 if step == 2 else 16
            shown = max(1, int(count * t))
            for i in range(shown):
                u = i / max(count - 1, 1) * t
                x, y = _projectile(u)
                painter.drawEllipse(self._pt(x, row + y * 0.42), 3.2, 3.2)

    def _paint_fix(self, painter: QPainter) -> None:
        t = self._phase
        self._scene(painter)
        x, y = _projectile(0.55)
        self._ball(painter, x, y)
        wrong = (x + 0.10, y - 0.08)
        if t < 0.48:
            painter.setBrush(WARN)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawEllipse(self._pt(*wrong), 6, 6)
            self._cursor(painter, _mix(0.18, wrong[0], _smooth(t / 0.48)), _mix(0.20, wrong[1], _smooth(t / 0.48)))
        else:
            u = _smooth((t - 0.48) / 0.30)
            cx = _mix(wrong[0], x, u)
            cy = _mix(wrong[1], y, u)
            painter.setBrush(HIGHLIGHT)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawEllipse(self._pt(cx, cy), 6, 6)
            self._cursor(painter, cx, cy, 1.0)
        self._label(painter, "点偏了的帧，再按 T", 0.08, 0.10, WARN)

    def _paint_range(self, painter: QPainter) -> None:
        t = self._phase
        self._scene(painter)
        start, end = 0.22, 0.78
        local = _clamp01((t - 0.08) / 0.70)
        x = _mix(start, end, local)
        y = 0.62 + 0.04 * math.sin(local * math.pi)
        if local < 0.98:
            self._mask(painter, x, y, 0.06)
            self._ball(painter, x, y, 0.04)
        self._timeline(painter, start, end, _mix(start, end, local))
        self._label(painter, "跟到结束三角停", 0.08, 0.10)

    def _paint_ruler(self, painter: QPainter) -> None:
        t = self._phase
        self._scene(painter)
        self._ball(painter, 0.28, 0.62)
        a = self._pt(0.18, 0.70)
        grow = _smooth(_clamp01((t - 0.2) / 0.45))
        b = self._pt(_mix(0.18, 0.72, grow), 0.70)
        painter.setPen(QPen(QColor("#80cbc4"), 2.4))
        painter.drawLine(a, b)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#80cbc4"))
        painter.drawEllipse(a, 4, 4)
        painter.drawEllipse(b, 4, 4)
        if grow > 0.85:
            self._label(painter, "1.00 m", 0.40, 0.58, QColor("#80cbc4"))
        self._cursor(painter, _mix(0.18, 0.72, grow), 0.70, 1.0 if t > 0.2 else 0.0)

    def _paint_axis(self, painter: QPainter) -> None:
        t = self._phase
        self._scene(painter)
        origin = (0.28, 0.70)
        self._ball(painter, 0.48, 0.48)
        angle = math.radians(-18 * _smooth(min(t / 0.7, 1.0)))
        length = 0.34
        ox, oy = origin
        x2 = ox + length * math.cos(angle)
        y2 = oy + length * math.sin(angle)
        yx = ox + length * 0.7 * math.cos(angle - math.pi / 2)
        yy = oy + length * 0.7 * math.sin(angle - math.pi / 2)
        painter.setPen(QPen(QColor("#e6e6e6"), 2))
        painter.drawLine(self._pt(ox, oy), self._pt(x2, y2))
        painter.drawLine(self._pt(ox, oy), self._pt(yx, yy))
        painter.setBrush(HIGHLIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(self._pt(ox, oy), 5, 5)
        self._label(painter, "X", x2, y2 - 0.04)
        self._label(painter, "Y", yx - 0.04, yy)
        self._cursor(
            painter,
            _mix(0.12, ox, _smooth(min(t / 0.4, 1.0))),
            _mix(0.20, oy, _smooth(min(t / 0.4, 1.0))),
            1.0 if 0.35 <= t <= 0.55 else 0.0,
        )

    def _paint_chart(self, painter: QPainter) -> None:
        t = _smooth(self._phase)
        frame = self._rect(0.08, 0.10, 0.84, 0.78)
        painter.setBrush(QColor("#232323"))
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        painter.drawRoundedRect(frame, 6, 6)
        painter.setPen(QPen(QColor("#5a5a5a"), 1))
        painter.drawLine(self._pt(0.14, 0.78), self._pt(0.88, 0.78))
        painter.drawLine(self._pt(0.14, 0.18), self._pt(0.14, 0.78))
        path = QPainterPath()
        first = True
        steps = 40
        shown = max(2, int(steps * t))
        for i in range(shown):
            u = i / (steps - 1)
            x = 0.14 + 0.70 * u
            y = 0.70 - 0.42 * math.sin(u * math.pi)
            pt = self._pt(x, y)
            if first:
                path.moveTo(pt)
                first = False
            else:
                path.lineTo(pt)
        painter.setPen(QPen(TRAIL, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        cx = 0.14 + 0.70 * t
        cy = 0.70 - 0.42 * math.sin(t * math.pi)
        painter.setPen(QPen(QColor("#f0f0f0"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(self._pt(cx, 0.18), self._pt(cx, 0.78))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(HIGHLIGHT)
        painter.drawEllipse(self._pt(cx, cy), 4, 4)
        self._label(painter, "x(t)", 0.16, 0.16)

    def _paint_table(self, painter: QPainter) -> None:
        t = self._phase
        rows = 6
        shown = max(1, int(min(t / 0.7, 1.0) * rows))
        headers = ("帧", "t / s", "x", "y")
        painter.setFont(self._font(11, True))
        painter.setPen(QColor("#8f8f8f"))
        for i, head in enumerate(headers):
            painter.drawText(self._pt(0.08 + i * 0.22, 0.12), head)
        painter.setFont(self._font(12))
        for row in range(shown):
            y = 0.22 + row * 0.12
            if row == 3:
                painter.setBrush(QColor(240, 193, 75, 50))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(self._rect(0.06, y - 0.08, 0.88, 0.11))
                painter.setPen(WARN)
            else:
                painter.setPen(QColor("#d6d6d6"))
            painter.drawText(self._pt(0.08, y), str(row + 1))
            painter.drawText(self._pt(0.30, y), f"{row * 0.04:.2f}")
            painter.drawText(self._pt(0.52, y), f"{80 + row * 12}")
            painter.drawText(self._pt(0.74, y), f"{40 + row * 3}")

    def _paint_readout(self, painter: QPainter) -> None:
        t = self._phase
        self._scene(painter)
        x, y = _projectile(_smooth(t))
        self._ball(painter, x, y)
        unit = "m" if t > 0.55 else "px"
        scale = 0.01 if unit == "m" else 1.0
        box = self._rect(0.08, 0.08, 0.84, 0.22)
        painter.setBrush(QColor("#2b2b2b"))
        painter.setPen(QPen(QColor("#3e3e3e"), 1))
        painter.drawRoundedRect(box, 6, 6)
        painter.setFont(self._font(13, True))
        painter.setPen(QColor("#f0f0f0"))
        painter.drawText(
            box,
            int(Qt.AlignmentFlag.AlignCenter),
            f"t={t * 1.2:.2f}s   x={x / scale:.1f} {unit}   y={y / scale:.1f} {unit}",
        )

    def _paint_assistant(self, painter: QPainter) -> None:
        t = self._phase
        bubble = self._rect(0.10, 0.16, 0.80, 0.46)
        painter.setBrush(QColor("#2b2b2b"))
        painter.setPen(QPen(HIGHLIGHT, 1))
        painter.drawRoundedRect(bubble, 10, 10)
        self._label(painter, "斜抛运动", 0.18, 0.30)
        if t > 0.28:
            painter.setFont(self._font(14))
            painter.setPen(QColor("#80cbc4"))
            painter.drawText(self._pt(0.18, 0.48), "y = v₀ t − ½ g t²")
        self._label(painter, "只发送结构化数据", 0.18, 0.78, QColor("#8f8f8f"))

    def _paint_play(self, painter: QPainter) -> None:
        t = self._phase
        if t > 0.72:
            self._paint_save(painter)
            return
        self._scene(painter)
        start, end = 0.18, 0.82
        span = end - start
        cycle = (t / 0.72) % 1.0
        play = start + span * cycle
        x, y = _projectile(cycle)
        self._ball(painter, x, y, 0.04)
        self._timeline(painter, start, end, play)
        btn = self._rect(0.08, 0.08, 0.14, 0.14)
        painter.setBrush(QColor("#303030"))
        painter.setPen(QPen(QColor("#4a4a4a"), 1))
        painter.drawRoundedRect(btn, 4, 4)
        tri = QPainterPath()
        tri.moveTo(self._pt(0.12, 0.11))
        tri.lineTo(self._pt(0.12, 0.19))
        tri.lineTo(self._pt(0.18, 0.15))
        painter.fillPath(tri, QColor("#e2e2e2"))

    def _paint_save(self, painter: QPainter) -> None:
        t = self._phase
        doc = self._rect(0.34, 0.16, 0.32, 0.52)
        painter.setBrush(QColor("#2d2d2d"))
        painter.setPen(QPen(QColor("#6a6a6a"), 1.4))
        painter.drawRoundedRect(doc, 6, 6)
        painter.setPen(QPen(QColor("#5a5a5a"), 1))
        for i in range(4):
            y = 0.28 + i * 0.08
            painter.drawLine(self._pt(0.40, y), self._pt(0.60, y))
        self._label(painter, "项目", 0.42, 0.22)
        if t > 0.45:
            painter.setPen(QPen(QColor("#80cbc4"), 3))
            path = QPainterPath()
            path.moveTo(self._pt(0.42, 0.78))
            path.lineTo(self._pt(0.48, 0.86))
            path.lineTo(self._pt(0.62, 0.70))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)


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
        self._live_pointer = False
        self._forward_grab: QWidget | None = None
        self._demo_timer = QTimer(self)
        self._demo_timer.setInterval(DEMO_TICK_MS)
        self._demo_timer.timeout.connect(lambda: self._demo_tick(float(DEMO_TICK_MS)))
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
        self._stage = TutorialStage(self._card)

        self._skip = QPushButton("跳过")
        self._skip.setObjectName("tutorialSkip")
        self._skip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip.clicked.connect(self._complete)
        self._back = QPushButton("上一步")
        self._back.setObjectName("tutorialBack")
        self._back.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back.clicked.connect(self.retreat)
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
        buttons.addWidget(self._back)
        buttons.addWidget(self._next)

        layout = QVBoxLayout(self._card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self._stage, 0, Qt.AlignmentFlag.AlignHCenter)
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
        return QPoint()

    @property
    def demo_phase(self) -> float:
        return self._phase

    @property
    def stage(self) -> TutorialStage:
        return self._stage

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
        self._demo_timer.stop()
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
        if self._index >= len(self._steps) - 1:
            self._complete()
            return
        self._index += 1
        self._show_step()

    def retreat(self) -> None:
        if self._index <= 0:
            return
        self._index -= 1
        self._show_step()

    def step_index(self, title: str) -> int:
        for index, step in enumerate(self._steps):
            if step.title == title:
                return index
        raise KeyError(title)

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

    def _demo_tick(self, dt_ms: float = 16.0) -> None:
        if self._done:
            return
        step = float(dt_ms)
        if 0.0 < step <= 5.0:
            step *= 1000.0
        self._phase = (self._phase + step / DEMO_CYCLE_MS) % 1.0
        self._stage.set_demo(self.current_demo(), self._phase)
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
        self._back.setVisible(self._index > 0)
        self._phase = 0.0
        self._live_pointer = False
        self._forward_grab = None
        self._stage.set_demo(step.demo, 0.0)
        self.reposition()
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
        self._stage.set_demo(self.current_demo(), self._phase)

    def _is_interactive_pos(self, pos: QPoint) -> bool:
        if self._card.geometry().contains(pos):
            return False
        return any(rect.contains(pos) for rect in self._cutout_rects())

    def _cutout_rects(self) -> list[QRect]:
        if self._hole.isValid() and not self._hole.isEmpty():
            return [QRect(self._hole)]
        return []

    def _update_live_pointer(self, pos: QPoint) -> None:
        live = self._is_interactive_pos(pos)
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
        event.accept()
        return True

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
        hole = self._hole if self._hole.isValid() else QRect()
        max_x = max(CARD_MARGIN, self.width() - width - CARD_MARGIN)
        max_y = max(CARD_MARGIN, self.height() - height - CARD_MARGIN)

        def clamp(x: int, y: int) -> tuple[int, int]:
            return min(max(CARD_MARGIN, x), max_x), min(max(CARD_MARGIN, y), max_y)

        def covered(x: int, y: int) -> int:
            if not hole.isValid():
                return 0
            inter = QRect(x, y, width, height).intersected(hole)
            if inter.isEmpty():
                return 0
            return inter.width() * inter.height()

        if hole.isValid():
            cx = hole.center().x() - width // 2
            cy = hole.center().y() - height // 2
            candidates = [
                clamp(cx, hole.bottom() + CARD_MARGIN),
                clamp(cx, hole.top() - height - CARD_MARGIN),
                clamp(hole.right() + CARD_MARGIN, cy),
                clamp(hole.left() - width - CARD_MARGIN, cy),
            ]
        else:
            candidates = [(CARD_MARGIN, CARD_MARGIN)]
        x, y = min(candidates, key=lambda point: (covered(*point), point[1], point[0]))
        self._card.setGeometry(x, y, width, height)
        app = QApplication.instance()
        if app is None or app.platformName() != "offscreen":
            self._card.raise_()

    def _complete(self) -> None:
        if self._done:
            return
        self._done = True
        self._demo_timer.stop()
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
