from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QContextMenuEvent,
    QFont,
    QImage,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QWidget,
)

from app.icons import icon_size, next_icon, prev_icon
from ai.contracts import LOW_CONFIDENCE, PromptKind, TrackPrompt
from engine.video_index import VideoInfo


MODE_TRACK = "track"
MODE_RULER = "ruler"
MODE_AXIS = "axis"
MODE_PLANE = "plane"

AXIS_MIN_LENGTH = 360.0
AXIS_SPAN_FRACTION = 0.62
AXIS_ROTATE_MIN = 18.0
AXIS_ROTATE_MAX = 30.0
AXIS_ROTATE_FRACTION = 0.04
AXIS_ROTATE_SPAN_DEG = 62.0
AXIS_INK = QColor("#e6e8ed")


def axis_display_length(width: float, height: float) -> float:
    """On-screen axis length in video pixels: large enough to grab and rotate."""
    short = min(max(width, 1.0), max(height, 1.0))
    return max(AXIS_MIN_LENGTH, AXIS_SPAN_FRACTION * short)


def axis_rotate_radius(length: float) -> float:
    return max(AXIS_ROTATE_MIN, min(AXIS_ROTATE_MAX, length * AXIS_ROTATE_FRACTION))


def snap_axis_angle(angle_deg: float) -> float:
    snapped = math.floor(angle_deg / 90.0 + 0.5) * 90.0
    return snapped % 360.0


def axis_arm_ends(
    ox: float,
    oy: float,
    length: float,
    angle_deg: float,
    *,
    y_up: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Video-pixel endpoints of +x / +y for the overlay."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    if y_up:
        return (
            (ox + c * length, oy - s * length),
            (ox - s * length, oy - c * length),
        )
    return (
        (ox + c * length, oy + s * length),
        (ox - s * length, oy + c * length),
    )


def axis_pointer_angle(
    ox: float, oy: float, x: float, y: float, *, y_up: bool
) -> float:
    if y_up:
        return math.degrees(math.atan2(-(y - oy), x - ox))
    return math.degrees(math.atan2(y - oy, x - ox))


def _point_seg_dist(
    px: float, py: float, x0: float, y0: float, x1: float, y1: float
) -> float:
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 < 1e-9:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length2))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


@dataclass
class OverlayTrack:
    # (frame, x, y, visible[, confidence]) so the marker follows the decoded frame.
    points: list[tuple]
    color: str
    active: bool = False
    contour: list[tuple[float, float]] = field(default_factory=list)
    prompts: list[TrackPrompt] = field(default_factory=list)


def _fmt_duration(ms: int) -> str:
    total = max(ms, 0)
    minutes, rest = divmod(total // 1000, 60)
    hours, minutes = divmod(minutes, 60)
    frac = (total % 1000) // 10
    if hours:
        return f"{hours:d}:{minutes:02d}:{rest:02d}.{frac:02d}"
    return f"{minutes:02d}:{rest:02d}.{frac:02d}"


class VideoInfoLabel(QLabel):
    """Borderless, eliding video summary for the main toolbar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoInfoLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._name = ""
        self._rest = ""

    def set_info(self, info: VideoInfo | None) -> None:
        if info is None:
            self._name = ""
            self._rest = ""
            self.setText("")
            self.setToolTip("")
            return
        self._name = info.path.name
        self._rest = (
            f"{info.width}×{info.height} · {info.fps:.2f} fps · "
            f"{_fmt_duration(info.duration_ms)} · {info.frame_count} 帧"
        )
        self.setToolTip(f"{self._name} · {self._rest}")
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if not self._rest:
            self.setText("")
            return
        full = f"{self._name} · {self._rest}" if self._name else self._rest
        width = max(self.width(), 1)
        fm = self.fontMetrics()
        if self._name and fm.horizontalAdvance(full) > width:
            self.setText(fm.elidedText(self._rest, Qt.TextElideMode.ElideRight, width))
            return
        self.setText(fm.elidedText(full, Qt.TextElideMode.ElideRight, width))


def _has_control(event) -> bool:  # noqa: ANN001
    mods = event.modifiers()
    return bool(
        mods & Qt.KeyboardModifier.ControlModifier
        or mods & Qt.KeyboardModifier.MetaModifier
    )


def _has_shift(event) -> bool:  # noqa: ANN001
    return bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)


def _is_track_edit(event) -> bool:  # noqa: ANN001
    """Control/Cmd+left, or right-click (macOS Control+click)."""
    button = event.button()
    if button == Qt.MouseButton.RightButton:
        return True
    return button == Qt.MouseButton.LeftButton and _has_control(event)


class VideoView(QWidget):
    clicked_at = Signal(float, float)
    prompted = Signal(float, float, str)
    boxed = Signal(float, float, float, float)
    zoom_changed = Signal(float)
    ruler_drawn = Signal(float, float, float, float)
    origin_picked = Signal(float, float)
    axis_picked = Signal(float, float)
    axis_dragged = Signal(str, float, float, bool)
    axis_drag_finished = Signal()
    axis_edit_requested = Signal()
    interaction_cancelled = Signal()
    plane_point_picked = Signal(float, float)
    plane_corner_dragged = Signal(int, float, float)
    plane_drag_finished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoView")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._scaled: QImage | None = None
        self._scaled_key: tuple[int, int, int, int] | None = None
        self._track: list[tuple] = []
        self._track_index = 0
        self._overlays: list[OverlayTrack] = []
        self._seed: tuple[float, float] | None = None
        self._show_contours = True
        self._show_prompts = True
        self._press: QPointF | None = None
        self._press_video: tuple[float, float] | None = None
        self._press_button = Qt.MouseButton.NoButton
        self._box: QRectF | None = None
        self._just_boxed = False
        self._just_prompted = False
        self._shift = False
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._panning: QPointF | None = None
        self._anchors: list[tuple[float, float]] = []
        self._mode = MODE_TRACK
        self._axis_step = 0
        self._axis_origin: tuple[float, float] | None = None
        self._axis_drag: str | None = None
        self._draft: tuple[float, float, float, float] | None = None
        self._rulers: list[tuple[float, float, float, float, str, str]] = []
        self._axis_overlay: tuple[float, float, float, float, float, float, str, str] | None = None
        self._show_calibration = True
        self._plane_corners: list[tuple[float, float]] = []
        self._plane_grid: list[tuple[float, float, float, float]] = []
        self._plane_label = ""
        self._plane_drag: int | None = None

    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, value: float, *, anchor: QPointF | None = None) -> None:
        new_zoom = max(0.25, min(8.0, value))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        before = None if anchor is None else self._video_xy(anchor)
        self._zoom = new_zoom
        self._scaled = None
        self._scaled_key = None
        if new_zoom <= 1.0 + 1e-6:
            self._pan = QPointF(0, 0)
        elif before is not None and anchor is not None:
            dest = self._dest_rect()
            if dest is not None and self._image is not None:
                sx = dest.width() / self._image.width()
                sy = dest.height() / self._image.height()
                now = QPointF(dest.x() + before[0] * sx, dest.y() + before[1] * sy)
                self._pan += anchor - now
        self.zoom_changed.emit(self._zoom)
        self.update()

    def reset_zoom(self) -> None:
        self._pan = QPointF(0, 0)
        self.set_zoom(1.0)

    def set_frame(self, image: QImage | None, *, repaint: bool = True) -> None:
        self._image = image
        self._scaled = None
        self._scaled_key = None
        if repaint:
            self.update()

    def set_track(self, points: list[tuple[int, float, float, bool]]) -> None:
        self._track = points
        self.update()

    def set_overlays(
        self,
        overlays: list[OverlayTrack],
        *,
        index: int = 0,
        seed: tuple[float, float] | None = None,
    ) -> None:
        self._overlays = overlays
        self._track_index = index
        self._seed = seed
        if overlays:
            active = next((o for o in overlays if o.active), overlays[0])
            self._track = active.points
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: ANN001
        if self._image is None or self._image.isNull():
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            super().wheelEvent(event)
            return
        factor = 1.12 ** steps
        self.set_zoom(self._zoom * factor, anchor=event.position())
        event.accept()

    def set_track_index(self, index: int) -> None:
        self._track_index = index
        self.update()

    def set_seed(self, seed: tuple[float, float] | None) -> None:
        self._seed = seed
        self.update()

    def set_display_options(self, *, contours: bool = True, prompts: bool = True) -> None:
        self._show_contours = contours
        self._show_prompts = prompts
        self.update()

    def set_anchors(self, points: list[tuple[float, float]]) -> None:
        self._anchors = list(points)
        self.update()

    def interaction_mode(self) -> str:
        return self._mode

    def set_interaction_mode(self, mode: str, *, axis_step: int = 0) -> None:
        self._mode = mode
        self._axis_step = axis_step
        self._draft = None
        self._box = None
        self._press = None
        self._press_video = None
        self._axis_drag = None
        self._plane_drag = None
        if mode == MODE_AXIS:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif mode == MODE_PLANE:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        if mode != MODE_TRACK:
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.update()

    def set_axis_origin(self, origin: tuple[float, float] | None) -> None:
        self._axis_origin = origin
        self.update()

    def set_show_calibration(self, visible: bool) -> None:
        self._show_calibration = visible
        self.update()

    def set_ruler_overlay(
        self, rulers: list[tuple[float, float, float, float, str, str]]
    ) -> None:
        self._rulers = list(rulers)
        self.update()

    def set_plane_overlay(
        self,
        corners: list[tuple[float, float]],
        grid: list[tuple[float, float, float, float]] | None = None,
        label: str = "",
    ) -> None:
        self._plane_corners = list(corners)
        self._plane_grid = list(grid or [])
        self._plane_label = label
        self.update()

    def set_axis_overlay(
        self,
        origin: tuple[float, float] | None,
        x_end: tuple[float, float] | None,
        y_end: tuple[float, float] | None,
        xlabel: str,
        ylabel: str,
    ) -> None:
        if origin is None or x_end is None or y_end is None:
            self._axis_overlay = None
        else:
            self._axis_overlay = (
                origin[0],
                origin[1],
                x_end[0],
                x_end[1],
                y_end[0],
                y_end[1],
                xlabel,
                ylabel,
            )
        self.update()

    def clear_calibration_overlay(self) -> None:
        self._rulers = []
        self._axis_overlay = None
        self._draft = None
        self._axis_origin = None
        self._plane_corners = []
        self._plane_grid = []
        self._plane_label = ""
        self._plane_drag = None
        self.update()

    def clear_track(self) -> None:
        self._track = []
        self._overlays = []
        self._track_index = 0
        self._seed = None
        self._anchors = []
        self.clear_calibration_overlay()
        self.set_interaction_mode(MODE_TRACK)
        self.update()

    def clear_scaled_cache(self) -> None:
        self._scaled = None
        self._scaled_key = None

    def _dest_rect(self) -> QRectF | None:
        if self._image is None or self._image.isNull():
            return None
        iw, ih = self._image.width(), self._image.height()
        if iw <= 0 or ih <= 0:
            return None
        scale = min(self.width() / iw, self.height() / ih) * self._zoom
        w, h = iw * scale, ih * scale
        x = (self.width() - w) / 2 + self._pan.x()
        y = (self.height() - h) / 2 + self._pan.y()
        return QRectF(x, y, w, h)

    def _video_xy(self, pos: QPointF) -> tuple[float, float] | None:
        dest = self._dest_rect()
        if dest is None or self._image is None or not dest.contains(pos):
            return None
        sx = (pos.x() - dest.x()) / dest.width() * self._image.width()
        sy = (pos.y() - dest.y()) / dest.height() * self._image.height()
        return float(sx), float(sy)

    def _hit_threshold_video(self) -> float:
        dest = self._dest_rect()
        if dest is None or self._image is None or dest.width() <= 1e-6:
            return 18.0
        return max(12.0, 11.0 * self._image.width() / dest.width())

    def _axis_geometry(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float, float] | None:
        if self._axis_overlay is None:
            return None
        ox, oy, xx, xy, yx, yy, _xlabel, _ylabel = self._axis_overlay
        lx = math.hypot(xx - ox, xy - oy) or 1.0
        ly = math.hypot(yx - ox, yy - oy) or 1.0
        ux, uy = (xx - ox) / lx, (xy - oy) / lx
        vx, vy = (yx - ox) / ly, (yy - oy) / ly
        radius = axis_rotate_radius(lx)
        hx = ox + ux * (lx - radius)
        hy = oy + uy * (lx - radius)
        return ox, oy, ux, uy, vx, vy, radius, hx, hy

    def _axis_hit(self, video: tuple[float, float]) -> str | None:
        geom = self._axis_geometry()
        if geom is None:
            return None
        ox, oy, ux, uy, vx, vy, radius, hx, hy = geom
        px, py = video
        threshold = self._hit_threshold_video()
        if math.hypot(px - ox, py - oy) <= threshold * 1.35:
            return "origin"
        dx, dy = px - hx, py - hy
        dist = math.hypot(dx, dy)
        ang = math.degrees(math.atan2(dx * vx + dy * vy, dx * ux + dy * uy))
        if abs(dist - radius) <= threshold * 1.5 and -12.0 <= ang <= AXIS_ROTATE_SPAN_DEG + 14.0:
            return "rotate"
        return None

    def _axis_frame_hit(self, video: tuple[float, float]) -> bool:
        if self._axis_overlay is None:
            return False
        if self._axis_hit(video) is not None:
            return True
        ox, oy, xx, xy, yx, yy, _xlabel, _ylabel = self._axis_overlay
        px, py = video
        threshold = self._hit_threshold_video()
        return (
            _point_seg_dist(px, py, ox, oy, xx, xy) <= threshold
            or _point_seg_dist(px, py, ox, oy, yx, yy) <= threshold
        )

    def _plane_hit(self, video: tuple[float, float]) -> int | None:
        if len(self._plane_corners) < 1:
            return None
        threshold = self._hit_threshold_video() * 1.4
        px, py = video
        best: tuple[float, int] | None = None
        for index, (x, y) in enumerate(self._plane_corners):
            dist = math.hypot(px - x, py - y)
            if dist <= threshold and (best is None or dist < best[0]):
                best = (dist, index)
        return None if best is None else best[1]

    def _update_axis_cursor(self, pos: QPointF) -> None:
        if self._mode != MODE_AXIS or self._axis_drag is not None:
            return
        video = self._video_xy(pos)
        if video is None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        hit = self._axis_hit(video)
        if hit == "origin":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif hit == "rotate":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._panning = event.position()
            return
        if self._mode == MODE_RULER:
            if event.button() != Qt.MouseButton.LeftButton:
                super().mousePressEvent(event)
                return
            video = self._video_xy(event.position())
            if video is None:
                super().mousePressEvent(event)
                return
            self._press = event.position()
            self._press_video = video
            self._press_button = event.button()
            return
        if self._mode == MODE_PLANE:
            if event.button() != Qt.MouseButton.LeftButton:
                super().mousePressEvent(event)
                return
            video = self._video_xy(event.position())
            if video is None:
                super().mousePressEvent(event)
                return
            self._press = event.position()
            self._press_video = video
            self._press_button = event.button()
            hit = self._plane_hit(video) if len(self._plane_corners) >= 4 else None
            if hit is not None:
                self._plane_drag = hit
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.plane_corner_dragged.emit(hit, video[0], video[1])
            return
        if self._mode == MODE_AXIS:
            if event.button() != Qt.MouseButton.LeftButton:
                super().mousePressEvent(event)
                return
            video = self._video_xy(event.position())
            if video is None:
                super().mousePressEvent(event)
                return
            self._press = event.position()
            self._press_video = video
            self._press_button = event.button()
            hit = self._axis_hit(video)
            if hit is None:
                self._press = None
                self._press_video = None
                self._press_button = Qt.MouseButton.NoButton
                return
            self._axis_drag = hit
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.axis_dragged.emit(hit, video[0], video[1], _has_shift(event))
            return
        if not _is_track_edit(event):
            super().mousePressEvent(event)
            return
        video = self._video_xy(event.position())
        if video is None:
            super().mousePressEvent(event)
            return
        self._press = event.position()
        self._press_video = video
        self._press_button = event.button()
        self._shift = _has_shift(event)
        self._box = None

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: ANN001
        # macOS Control+click arrives as a context-menu event, not a left click.
        event.accept()
        if self._mode != MODE_TRACK:
            return
        if self._just_boxed or self._just_prompted:
            self._just_boxed = False
            self._just_prompted = False
            return
        if self._box is not None:
            return
        if not _has_shift(event):
            return
        video = self._video_xy(QPointF(event.pos()))
        if video is None:
            return
        self._emit_point(video[0], video[1])

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._panning is not None:
            delta = event.position() - self._panning
            self._pan += delta
            self._panning = event.position()
            self.update()
            return
        if self._mode == MODE_RULER and self._press_video is not None:
            current = self._video_xy(event.position())
            if current is None:
                return
            self._draft = (*self._press_video, *current)
            self.update()
            return
        if self._mode == MODE_PLANE:
            if self._plane_drag is not None:
                current = self._video_xy(event.position())
                if current is None:
                    return
                self.plane_corner_dragged.emit(
                    self._plane_drag, current[0], current[1]
                )
                return
            return
        if self._mode == MODE_AXIS:
            if self._axis_drag is not None:
                current = self._video_xy(event.position())
                if current is None:
                    return
                self.axis_dragged.emit(
                    self._axis_drag, current[0], current[1], _has_shift(event)
                )
                return
            self._update_axis_cursor(event.position())
            return
        if self._press is None or self._shift:
            if self._press is None:
                super().mouseMoveEvent(event)
            return
        delta = event.position() - self._press
        if delta.manhattanLength() < 6:
            return
        current = self._video_xy(event.position())
        if current is None or self._press_video is None:
            return
        x0, y0 = self._press_video
        x1, y1 = current
        self._box = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._panning is not None:
            self._panning = None
            return
        if self._mode == MODE_RULER:
            if self._press_video is None or event.button() != self._press_button:
                super().mouseReleaseEvent(event)
                return
            current = self._video_xy(event.position()) or self._press_video
            x0, y0 = self._press_video
            x1, y1 = current
            if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > 4:
                self.ruler_drawn.emit(x0, y0, x1, y1)
            self._press = None
            self._press_video = None
            self._press_button = Qt.MouseButton.NoButton
            self._draft = None
            self.update()
            return
        if self._mode == MODE_PLANE:
            if self._press_video is None or event.button() != self._press_button:
                super().mouseReleaseEvent(event)
                return
            if self._plane_drag is not None:
                self.plane_drag_finished.emit()
            elif len(self._plane_corners) < 4:
                current = self._video_xy(event.position()) or self._press_video
                self.plane_point_picked.emit(current[0], current[1])
            self._plane_drag = None
            self._press = None
            self._press_video = None
            self._press_button = Qt.MouseButton.NoButton
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.update()
            return
        if self._mode == MODE_AXIS:
            if self._press_video is None or event.button() != self._press_button:
                super().mouseReleaseEvent(event)
                return
            if self._axis_drag is not None:
                self.axis_drag_finished.emit()
            self._axis_drag = None
            self._press = None
            self._press_video = None
            self._press_button = Qt.MouseButton.NoButton
            self._draft = None
            self._update_axis_cursor(event.position())
            self.update()
            return
        if self._press_video is None or event.button() != self._press_button:
            super().mouseReleaseEvent(event)
            return
        boxed = (
            not self._shift
            and self._box is not None
            and self._box.width() > 3
            and self._box.height() > 3
        )
        if boxed:
            self.boxed.emit(
                self._box.x(),
                self._box.y(),
                self._box.x() + self._box.width(),
                self._box.y() + self._box.height(),
            )
            self._just_boxed = True
        elif self._shift:
            x, y = self._press_video
            self._emit_point(x, y)
            self._just_prompted = True
        self._press = None
        self._press_video = None
        self._press_button = Qt.MouseButton.NoButton
        self._box = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        video = self._video_xy(event.position())
        if video is not None and self._axis_frame_hit(video):
            self.axis_edit_requested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Escape and self._mode != MODE_TRACK:
            self._draft = None
            self._press = None
            self._press_video = None
            self._axis_drag = None
            self._plane_drag = None
            self.interaction_cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _emit_point(self, x: float, y: float) -> None:
        self.clicked_at.emit(x, y)
        self.prompted.emit(x, y, "positive")

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1f1f1f"))
        if self._image is None or self._image.isNull():
            return

        dest = self._dest_rect()
        assert dest is not None
        scaled = self._scaled_for(QSize(max(1, int(dest.width())), max(1, int(dest.height()))))
        painter.drawImage(int(dest.x()), int(dest.y()), scaled)
        sx = dest.width() / self._image.width()
        sy = dest.height() / self._image.height()

        overlays = self._overlays
        if not overlays and self._track:
            overlays = [OverlayTrack(points=self._track, color="#f0c14b", active=True)]
        for overlay in overlays:
            color = QColor(overlay.color)
            trail = QPainterPath()
            started = False
            current: tuple[float, float, bool, float] | None = None
            for item in overlay.points:
                frame, x, y, visible = item[0], item[1], item[2], item[3]
                conf = float(item[4]) if len(item) > 4 else 1.0
                if frame == self._track_index:
                    current = (x, y, visible, conf)
                if not visible:
                    started = False
                    continue
                px = dest.x() + x * sx
                py = dest.y() + y * sy
                if not started:
                    trail.moveTo(px, py)
                    started = True
                else:
                    trail.lineTo(px, py)
            width = 2.2 if overlay.active else 1.3
            color.setAlpha(255 if overlay.active else 160)
            painter.setPen(QPen(color, width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(trail)
            if overlay.active and current is not None and current[2]:
                px = dest.x() + current[0] * sx
                py = dest.y() + current[1] * sy
                painter.setBrush(color)
                if current[3] < LOW_CONFIDENCE:
                    painter.setPen(QPen(QColor("#f0c14b"), 2.2))
                else:
                    painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QRectF(px - 4, py - 4, 8, 8))
            if self._show_contours and overlay.active and overlay.contour:
                path = QPainterPath()
                first = True
                for x, y in overlay.contour:
                    px = dest.x() + x * sx
                    py = dest.y() + y * sy
                    if first:
                        path.moveTo(px, py)
                        first = False
                    else:
                        path.lineTo(px, py)
                path.closeSubpath()
                fill = QColor(overlay.color)
                fill.setAlpha(50)
                painter.setBrush(fill)
                painter.setPen(QPen(color, 1.2, Qt.PenStyle.DashLine))
                painter.drawPath(path)
            if self._show_prompts:
                for prompt in overlay.prompts:
                    px = dest.x() + prompt.x * sx
                    py = dest.y() + prompt.y * sy
                    if prompt.kind == PromptKind.BOX and prompt.x2 is not None and prompt.y2 is not None:
                        rect = QRectF(
                            dest.x() + min(prompt.x, prompt.x2) * sx,
                            dest.y() + min(prompt.y, prompt.y2) * sy,
                            abs(prompt.x2 - prompt.x) * sx,
                            abs(prompt.y2 - prompt.y) * sy,
                        )
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.setPen(QPen(QColor("#6cb6ff"), 1.2, Qt.PenStyle.DashLine))
                        painter.drawRect(rect)
                    elif prompt.kind == PromptKind.NEGATIVE:
                        painter.setPen(QPen(QColor("#e07a5f"), 2))
                        painter.drawLine(QPointF(px - 5, py - 5), QPointF(px + 5, py + 5))
                        painter.drawLine(QPointF(px - 5, py + 5), QPointF(px + 5, py - 5))
                    else:
                        painter.setPen(QPen(QColor("#7dce82"), 2))
                        painter.drawLine(QPointF(px - 6, py), QPointF(px + 6, py))
                        painter.drawLine(QPointF(px, py - 6), QPointF(px, py + 6))

        if self._seed is not None:
            px = dest.x() + self._seed[0] * sx
            py = dest.y() + self._seed[1] * sy
            painter.setPen(QPen(QColor("#ffffff"), 1.4))
            painter.drawLine(QPointF(px - 8, py), QPointF(px + 8, py))
            painter.drawLine(QPointF(px, py - 8), QPointF(px, py + 8))

        if self._anchors:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#9ee7ff"), 1.4))
            for ax, ay in self._anchors:
                px = dest.x() + ax * sx
                py = dest.y() + ay * sy
                painter.drawLine(QPointF(px - 5, py), QPointF(px + 5, py))
                painter.drawLine(QPointF(px, py - 5), QPointF(px, py + 5))
                painter.drawEllipse(QRectF(px - 3.5, py - 3.5, 7, 7))

        if self._show_calibration:
            self._paint_calibration(painter, dest, sx, sy)

        if self._box is not None:
            rect = QRectF(
                dest.x() + self._box.x() * sx,
                dest.y() + self._box.y() * sy,
                self._box.width() * sx,
                self._box.height() * sy,
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#6cb6ff"), 1.4, Qt.PenStyle.DashLine))
            painter.drawRect(rect)

    def _scaled_for(self, size: QSize) -> QImage:
        assert self._image is not None
        key = (
            size.width(),
            size.height(),
            int(self._zoom * 1000),
            self._image.cacheKey() & 0xFFFFFFFF,
        )
        if self._scaled is not None and self._scaled_key == key:
            return self._scaled
        self._scaled = self._image.scaled(
            size,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._scaled_key = key
        return self._scaled

    def _map_video(self, dest: QRectF, sx: float, sy: float, x: float, y: float) -> QPointF:
        return QPointF(dest.x() + x * sx, dest.y() + y * sy)

    @staticmethod
    def _draw_halo_text(
        painter: QPainter,
        pos: QPointF,
        text: str,
        color: QColor,
        halo: QColor | None = None,
    ) -> None:
        outline = halo or QColor(0, 0, 0, 210)
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)):
            painter.setPen(outline)
            painter.drawText(pos + QPointF(dx, dy), text)
        painter.setPen(color)
        painter.drawText(pos, text)

    def _paint_calibration(
        self, painter: QPainter, dest: QRectF, sx: float, sy: float
    ) -> None:
        font = QFont(painter.font())
        font.setPixelSize(13)
        font.setBold(True)
        painter.setFont(font)
        accent = QColor("#80cbc4")
        for x0, y0, x1, y1, label, role in self._rulers:
            p0 = self._map_video(dest, sx, sy, x0, y0)
            p1 = self._map_video(dest, sx, sy, x1, y1)
            painter.setPen(QPen(accent, 2.0))
            painter.drawLine(p0, p1)
            painter.setBrush(accent)
            painter.drawEllipse(QRectF(p0.x() - 3.5, p0.y() - 3.5, 7, 7))
            painter.drawEllipse(QRectF(p1.x() - 3.5, p1.y() - 3.5, 7, 7))
            mid = QPointF((p0.x() + p1.x()) / 2, (p0.y() + p1.y()) / 2 - 8)
            text = label if not role else f"{role} {label}"
            self._draw_halo_text(painter, mid, text, QColor("#ffffff"))
        if self._axis_overlay is not None:
            ox, oy, xx, xy, yx, yy, xlabel, ylabel = self._axis_overlay
            origin = self._map_video(dest, sx, sy, ox, oy)
            x_end = self._map_video(dest, sx, sy, xx, xy)
            y_end = self._map_video(dest, sx, sy, yx, yy)
            interactive = self._mode == MODE_AXIS
            rotate_r = axis_rotate_radius(math.hypot(xx - ox, xy - oy)) * sx
            self._draw_axis_frame(
                painter,
                origin,
                x_end,
                y_end,
                xlabel,
                ylabel,
                rotate_r=rotate_r,
                interactive=interactive,
            )
        self._paint_plane(painter, dest, sx, sy)
        if self._draft is not None:
            x0, y0, x1, y1 = self._draft
            painter.setPen(QPen(QColor("#f0c14b"), 1.6, Qt.PenStyle.DashLine))
            painter.drawLine(
                self._map_video(dest, sx, sy, x0, y0),
                self._map_video(dest, sx, sy, x1, y1),
            )

    def _paint_plane(
        self, painter: QPainter, dest: QRectF, sx: float, sy: float
    ) -> None:
        corners = self._plane_corners
        if not corners:
            return
        mapped = [self._map_video(dest, sx, sy, x, y) for x, y in corners]
        fill = QColor(128, 203, 196, 36)
        edge = QColor("#80cbc4")
        if len(mapped) >= 4:
            path = QPainterPath()
            path.moveTo(mapped[0])
            for point in mapped[1:4]:
                path.lineTo(point)
            path.closeSubpath()
            painter.setPen(QPen(edge, 1.8))
            painter.setBrush(fill)
            painter.drawPath(path)
        painter.setPen(QPen(QColor("#4db6ac"), 1.0, Qt.PenStyle.DotLine))
        for x0, y0, x1, y1 in self._plane_grid:
            painter.drawLine(
                self._map_video(dest, sx, sy, x0, y0),
                self._map_video(dest, sx, sy, x1, y1),
            )
        labels = ("原点", "X 端", "对角点", "Y 端")
        painter.setBrush(edge)
        for index, point in enumerate(mapped):
            painter.setPen(QPen(QColor(0, 0, 0, 140), 2.0))
            painter.drawEllipse(QRectF(point.x() - 4.5, point.y() - 4.5, 9, 9))
            painter.setPen(QColor("#d8fff8"))
            name = labels[index] if index < len(labels) else str(index + 1)
            painter.drawText(QPointF(point.x() + 8, point.y() - 6), f"{index + 1} {name}")
        if self._plane_label:
            painter.setPen(QColor("#d8fff8"))
            painter.drawText(mapped[0] + QPointF(8, 16), self._plane_label)

    def _draw_axis_frame(
        self,
        painter: QPainter,
        origin: QPointF,
        x_end: QPointF,
        y_end: QPointF,
        xlabel: str,
        ylabel: str,
        *,
        rotate_r: float,
        interactive: bool = False,
    ) -> None:
        ink = AXIS_INK
        halo = QColor(0, 0, 0, 150)
        self._stroke_axis_arm(painter, origin, x_end, ink, halo, xlabel)
        self._stroke_axis_arm(painter, origin, y_end, ink, halo, ylabel)
        dx = x_end.x() - origin.x()
        dy = x_end.y() - origin.y()
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        ylen = math.hypot(y_end.x() - origin.x(), y_end.y() - origin.y()) or 1.0
        vx = (y_end.x() - origin.x()) / ylen
        vy = (y_end.y() - origin.y()) / ylen
        origin_r = 4.2 if interactive else 3.6
        painter.setPen(QPen(halo, 2.2))
        painter.setBrush(ink)
        painter.drawEllipse(
            QRectF(
                origin.x() - origin_r,
                origin.y() - origin_r,
                origin_r * 2.0,
                origin_r * 2.0,
            )
        )
        if not interactive:
            return
        radius = rotate_r
        cx = x_end.x() - ux * radius
        cy = x_end.y() - uy * radius
        start_qt = -math.degrees(math.atan2(uy, ux))
        # Qt positive arc is CCW (toward screen-up). Follow +y instead.
        sign = 1.0 if (ux * vy - uy * vx) < 0.0 else -1.0
        arc_start = start_qt + 8.0 * sign
        arc_span = AXIS_ROTATE_SPAN_DEG * sign
        rect = QRectF(cx - radius, cy - radius, radius * 2.0, radius * 2.0)
        path = QPainterPath()
        path.arcMoveTo(rect, arc_start)
        path.arcTo(rect, arc_start, arc_span)
        painter.setPen(QPen(halo, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.setPen(QPen(ink, 1.25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(path)
        end_a = math.radians(8.0 + AXIS_ROTATE_SPAN_DEG)
        ex = cx + radius * (math.cos(end_a) * ux + math.sin(end_a) * vx)
        ey = cy + radius * (math.cos(end_a) * uy + math.sin(end_a) * vy)
        tx = -math.sin(end_a) * ux + math.cos(end_a) * vx
        ty = -math.sin(end_a) * uy + math.cos(end_a) * vy
        tlen = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tlen, ty / tlen
        self._draw_small_arrow(painter, QPointF(ex, ey), tx, ty, ink, size=7.0)

    def _stroke_axis_arm(
        self,
        painter: QPainter,
        origin: QPointF,
        tip: QPointF,
        ink: QColor,
        halo: QColor,
        label: str,
    ) -> None:
        dx = tip.x() - origin.x()
        dy = tip.y() - origin.y()
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        neg = QPointF(origin.x() - ux * length * 0.22, origin.y() - uy * length * 0.22)
        painter.setPen(QPen(halo, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(neg, tip)
        faded = QColor(ink)
        faded.setAlpha(70)
        painter.setPen(QPen(faded, 1.0, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(origin, neg)
        painter.setPen(QPen(ink, 1.05, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(origin, tip)
        self._draw_small_arrow(painter, tip, ux, uy, ink, size=6.0)
        font = QFont(painter.font())
        font.setPixelSize(12)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(ink)
        painter.drawText(QPointF(tip.x() + ux * 8, tip.y() + uy * 8), label)

    def _draw_small_arrow(
        self,
        painter: QPainter,
        tip: QPointF,
        ux: float,
        uy: float,
        color: QColor,
        *,
        size: float,
    ) -> None:
        wing = size * 0.42
        left = QPointF(tip.x() - ux * size + uy * wing, tip.y() - uy * size - ux * wing)
        right = QPointF(tip.x() - ux * size - uy * wing, tip.y() - uy * size + ux * wing)
        painter.setBrush(color)
        painter.setPen(QPen(color, 1.0))
        painter.drawPolygon(QPolygonF([tip, left, right]))


class DropHint(QWidget):
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dropHint")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hover = False
        self._status = ""

    def set_status(self, text: str) -> None:
        self._status = text
        self.update()

    def set_hover(self, hover: bool) -> None:
        if self._hover == hover:
            return
        self._hover = hover
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        background = QColor("#1b1b1b" if not self._hover else "#202020")
        painter.setBrush(background)
        border = QPen(QColor("#474747" if not self._hover else "#777777"), 1.0)
        painter.setPen(border)
        painter.drawRoundedRect(outer, 6, 6)

        center_x = self.width() / 2
        center_y = self.height() / 2 - 24
        icon_box = QRectF(center_x - 34, center_y - 34, 68, 58)
        painter.setPen(QPen(QColor("#a5a5a5" if self._hover else "#737373"), 1.4))
        painter.setBrush(QColor(255, 255, 255, 7))
        painter.drawRoundedRect(icon_box, 9, 9)
        screen = icon_box.adjusted(14, 12, -14, -15)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(screen, 4, 4)
        painter.drawLine(
            int(screen.left() + 8),
            int(screen.bottom() + 7),
            int(screen.right() - 8),
            int(screen.bottom() + 7),
        )

        title_rect = QRectF(0, center_y + 38, self.width(), 26)
        painter.setPen(QColor("#dddddd"))
        font = painter.font()
        font.setPointSize(14)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "拖入视频开始分析")

        subtitle_rect = QRectF(0, center_y + 68, self.width(), 22)
        painter.setPen(QColor("#858585"))
        font.setPointSize(10)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.drawText(
            subtitle_rect,
            Qt.AlignmentFlag.AlignCenter,
            "或点击选择文件  ·  MP4  MOV  AVI  MKV",
        )

        if self._status:
            box = QRectF(
                self.width() * 0.30,
                center_y + 100,
                self.width() * 0.40,
                30,
            )
            painter.setBrush(QColor("#292929"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(box, 8, 8)
            painter.setPen(QColor("#c7c7c7"))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self._status)


class VideoStage(QWidget):
    """Video surface that fills the main work area."""

    def __init__(self, hint: DropHint, video: VideoView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoStage")
        self.stack = QStackedWidget(self)
        self.stack.setFrameShape(QFrame.Shape.NoFrame)
        self.stack.addWidget(hint)
        self.stack.addWidget(video)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        self.stack.setGeometry(QRect(0, 0, self.width(), self.height()))
        super().resizeEvent(event)


class TimelineSlider(QSlider):
    """Timeline with draggable loop-range markers above and a playhead below."""

    loop_range_changed = Signal(int, int)

    MARKER_GRAB_PX = 9

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("timeline")
        self.setFixedHeight(36)
        self._loop_start = 0
        self._loop_end = 0
        self._dragging: str | None = None

    def loop_range(self) -> tuple[int, int]:
        return self._loop_start, self._loop_end

    def set_loop_range(self, start: int, end: int) -> None:
        start = max(self.minimum(), min(start, self.maximum()))
        end = max(self.minimum(), min(end, self.maximum()))
        if start > end:
            start, end = end, start
        if (start, end) == (self._loop_start, self._loop_end):
            return
        self._loop_start, self._loop_end = start, end
        self.update()
        self.loop_range_changed.emit(start, end)

    def _groove(self):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        return self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            option,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )

    def _x_for(self, value: int) -> float:
        groove = self._groove()
        span = max(self.maximum() - self.minimum(), 1)
        fraction = (value - self.minimum()) / span
        return groove.left() + groove.width() * fraction

    def _value_at(self, x: float) -> int:
        groove = self._groove()
        if groove.width() <= 0:
            return self.minimum()
        fraction = (x - groove.left()) / groove.width()
        fraction = max(0.0, min(1.0, fraction))
        span = self.maximum() - self.minimum()
        return self.minimum() + round(fraction * span)

    def _marker_at(self, x: float) -> str | None:
        candidates = {
            "start": abs(x - self._x_for(self._loop_start)),
            "end": abs(x - self._x_for(self._loop_end)),
        }
        name = min(candidates, key=candidates.get)
        return name if candidates[name] <= self.MARKER_GRAB_PX else None

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if not self.isEnabled() or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        pos = event.position()
        groove = self._groove()
        if pos.y() < groove.center().y() - 2:
            marker = self._marker_at(pos.x())
            if marker is not None:
                self._dragging = marker
                return
        self.setSliderDown(True)
        self.setValue(self._value_at(pos.x()))

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._dragging is not None:
            value = self._value_at(event.position().x())
            if self._dragging == "start":
                self.set_loop_range(min(value, self._loop_end), self._loop_end)
            else:
                self.set_loop_range(self._loop_start, max(value, self._loop_start))
            return
        if self.isSliderDown():
            self.setValue(self._value_at(event.position().x()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if self._dragging is not None:
            self._dragging = None
            return
        if self.isSliderDown():
            self.setSliderDown(False)
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        groove = self._groove()
        if groove.width() <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        mid = groove.center().y()

        left_x = self._x_for(self._loop_start)
        right_x = self._x_for(self._loop_end)
        shade = QColor(18, 18, 18, 170)
        if left_x > groove.left():
            painter.fillRect(
                QRectF(groove.left(), mid - 3, left_x - groove.left(), 6), shade
            )
        if right_x < groove.right():
            painter.fillRect(QRectF(right_x, mid - 3, groove.right() - right_x, 6), shade)

        marker_color = QColor("#c4c4c4" if self.isEnabled() else "#6a6a6a")
        for value in (self._loop_start, self._loop_end):
            x = self._x_for(value)
            path = QPainterPath()
            path.moveTo(x - 5, mid - 12)
            path.lineTo(x + 5, mid - 12)
            path.lineTo(x, mid - 4)
            path.closeSubpath()
            painter.fillPath(path, marker_color)

        x = self._x_for(self.value())
        playhead = QPainterPath()
        playhead.moveTo(x - 6, mid + 13)
        playhead.lineTo(x + 6, mid + 13)
        playhead.lineTo(x, mid + 4)
        playhead.closeSubpath()
        painter.fillPath(playhead, QColor("#e2e2e2" if self.isEnabled() else "#767676"))


class StepStepper(QWidget):
    """Tracker-style frame step: |◀  N  ▶| with a fixed set of step sizes."""

    step_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepStepper")
        self.setFixedHeight(26)

        back = QToolButton()
        back.setObjectName("stepGlyph")
        back.setIcon(prev_icon())
        back.setIconSize(icon_size())
        back.setToolTip("按步进后退")
        back.clicked.connect(lambda: self.step_requested.emit(-self.value()))

        self._combo = QComboBox()
        self._combo.setObjectName("stepValue")
        self._combo.addItems(["1", "2", "3", "4", "5"])
        self._combo.setCurrentIndex(0)
        self._combo.setFixedWidth(34)
        self._combo.setToolTip("每次前进或后退的帧数")

        forward = QToolButton()
        forward.setObjectName("stepGlyph")
        forward.setIcon(next_icon())
        forward.setIconSize(icon_size())
        forward.setToolTip("按步进前进")
        forward.clicked.connect(lambda: self.step_requested.emit(self.value()))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(0)
        layout.addWidget(back)
        layout.addWidget(self._combo)
        layout.addWidget(forward)

    def value(self) -> int:
        return int(self._combo.currentText())
