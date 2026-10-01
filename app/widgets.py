from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
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
    QMenu,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QWidget,
)

from app.fit_prompt import FitPromptCard
from app.marks import PENDING_OUTLINE, PENDING_YELLOW, TRUSTED_GREEN
from app.gestures import clamp_pan, gesture_from_native, gesture_from_wheel, pointer_pans
from app.icons import icon_size, next_icon, prev_icon
from app.theme import is_light
from app.theme import qcolor as theme_qcolor
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
# Pending fit point: a gentle breathing pulse, not a hard blink.
FIT_PULSE_PERIOD_S = 1.6
FIT_PULSE_TICK_MS = 40


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
    # (frame, x, y, visible[, quality, status, source]) follows the decoded frame.
    points: list[tuple]
    color: str
    active: bool = False
    contour: list[tuple[float, float]] = field(default_factory=list)
    prompts: list[TrackPrompt] = field(default_factory=list)
    follow_box: tuple[float, float, float, float] | None = None
    # frame -> (x, y, sigma_px): trajectory-fit proposals for untrusted frames.
    suggestions: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    # Frames waiting for the user (drawn yellow); trusted measurements are green.
    pending: frozenset[int] = frozenset()


def clip_index(index: int, start: int, end: int) -> int:
    """Keep a frame index inside the analysis range marked on the timeline."""
    if start > end:
        start, end = end, start
    return max(start, min(index, end))


def object_follow_box(
    contour: list[tuple[float, float]],
    point: tuple[float, float] | None,
    template: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Box that moves with the tracked object.

    The contour's bounds win. Otherwise keep the size of the box the user
    drew and center it on the current point. The drawn box is the object,
    not a search window.
    """
    if len(contour) >= 2:
        xs = [item[0] for item in contour]
        ys = [item[1] for item in contour]
        return (min(xs), min(ys), max(xs), max(ys))
    if point is None:
        return None
    px, py = point
    if template is not None:
        x0, y0, x1, y1 = template
        width = abs(x1 - x0)
        height = abs(y1 - y0)
    else:
        width = height = 48.0
    width = width if width >= 4.0 else 48.0
    height = height if height >= 4.0 else 48.0
    return (px - width / 2.0, py - height / 2.0, px + width / 2.0, py + height / 2.0)


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
    """Control/Cmd+left, Shift+left (place a point), or right-click (macOS Control+click)."""
    button = event.button()
    if button == Qt.MouseButton.RightButton:
        return True
    return button == Qt.MouseButton.LeftButton and (_has_control(event) or _has_shift(event))


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
    axis_delete_requested = Signal()
    interaction_cancelled = Signal()
    plane_point_picked = Signal(float, float)
    plane_corner_dragged = Signal(int, float, float)
    plane_drag_finished = Signal()
    manual_pick_cancelled = Signal()
    manual_pick_needs_shift = Signal()

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
        self._fit_curves: list[list[tuple[float, float]]] = []
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
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(FIT_PULSE_TICK_MS)
        self._pulse_timer.timeout.connect(self._advance_pulse)
        self._manual_pick = False
        self._pending_region: QRect | None = None
        self._pan_origin: QPointF | None = None
        self.fit_card = FitPromptCard(self)

    def zoom(self) -> float:
        return self._zoom

    def content_rect(self) -> QRectF | None:
        return self._dest_rect()

    def cancel_stroke(self) -> None:
        self._press = None
        self._press_video = None
        self._press_button = Qt.MouseButton.NoButton
        self._box = None
        self._draft = None
        self._panning = None
        self.update()

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
        self._limit_pan()
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
        self._sync_pulse()
        self.place_fit_card()
        self.update()

    def set_fit_curves(self, curves: list[list[tuple[float, float]]]) -> None:
        self._fit_curves = curves
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: ANN001
        if self._image is None or self._image.isNull():
            super().wheelEvent(event)
            return
        gesture = gesture_from_wheel(event)
        if gesture.kind == "pan":
            self._pan += QPointF(gesture.dx, gesture.dy)
            self._limit_pan()
            self.update()
            event.accept()
            return
        if gesture.kind == "zoom":
            factor = gesture.factor if gesture.factor is not None else 1.12 ** gesture.steps
            self.set_zoom(self._zoom * factor, anchor=event.position())
            event.accept()
            return
        super().wheelEvent(event)

    def event(self, event: QEvent) -> bool:
        gesture = gesture_from_native(event)
        if gesture is not None and self._image is not None and not self._image.isNull():
            if gesture.kind == "pan":
                self._pan += QPointF(gesture.dx, gesture.dy)
                self._limit_pan()
                self.update()
                return True
            if gesture.kind == "zoom":
                factor = gesture.factor if gesture.factor is not None else 1.12 ** gesture.steps
                anchor = event.position() if hasattr(event, "position") else None
                self.set_zoom(self._zoom * factor, anchor=anchor)
                return True
            if gesture.kind == "reset":
                self.reset_zoom()
                return True
        return super().event(event)

    def set_track_index(self, index: int) -> None:
        self._track_index = index
        self._sync_pulse()
        self.place_fit_card()
        self.update()

    # -- pending fit point: pulse, manual pick, floating card -------------

    def current_suggestion(self) -> tuple[float, float, float] | None:
        """(x, y, sigma_px) of the fit suggestion on the shown frame, if any."""
        for overlay in self._overlays:
            if overlay.active and overlay.suggestions:
                return overlay.suggestions.get(self._track_index)
        return None

    def _has_pending(self) -> bool:
        for overlay in self._overlays:
            if overlay.active and (overlay.suggestions or overlay.pending):
                return True
        return False

    def _sync_pulse(self) -> None:
        """Breathe while any point waits for the user; idle otherwise."""
        wanted = self._has_pending() and self.isVisible()
        if wanted and not self._pulse_timer.isActive():
            self._pulse_timer.start()
        elif not wanted and self._pulse_timer.isActive():
            self._pulse_timer.stop()
            self._pulse_phase = 0.0

    def pulse_level(self) -> float:
        """0..1 breathing level of the pending points."""
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * self._pulse_phase)

    def _advance_pulse(self) -> None:
        if not self._has_pending():
            self._sync_pulse()
            return
        self._pulse_phase = (self._pulse_phase + FIT_PULSE_TICK_MS / 1000.0 / FIT_PULSE_PERIOD_S) % 1.0
        # Repaint only around the yellow points (collected by the last paint).
        region = getattr(self, "_pending_region", None)
        if region is None:
            self.update()
            return
        self.update(region)

    def _screen_point(self, x: float, y: float) -> QPointF | None:
        dest = self._dest_rect()
        if dest is None or self._image is None or self._image.isNull():
            return None
        sx = dest.width() / self._image.width()
        sy = dest.height() / self._image.height()
        return QPointF(dest.x() + x * sx, dest.y() + y * sy)

    def manual_pick_active(self) -> bool:
        return self._manual_pick

    def set_manual_pick(self, active: bool) -> None:
        """Guided manual point: crosshair cursor until Shift+click or Esc."""
        self._manual_pick = bool(active)
        if self._manual_pick:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        elif self._mode == MODE_TRACK:
            self.unsetCursor()
        self.update()

    def place_fit_card(self) -> None:
        """Keep the card beside the marker, inside the view."""
        card = self.fit_card
        if not card.isVisible():
            return
        here = self.current_suggestion()
        anchor = None if here is None else self._screen_point(here[0], here[1])
        if anchor is None:
            anchor = self._current_track_screen_point()
        margin = 10
        width, height = card.width(), card.height()
        if anchor is None:
            card.move(max(margin, self.width() - width - margin), margin)
            return
        gap = 26
        x = anchor.x() + gap
        if x + width > self.width() - margin:
            x = anchor.x() - gap - width
        y = anchor.y() - height / 2.0
        x = max(margin, min(x, self.width() - width - margin))
        y = max(margin, min(y, self.height() - height - margin))
        card.move(int(x), int(y))

    def _current_track_screen_point(self) -> QPointF | None:
        for item in self._track:
            if item[0] == self._track_index:
                return self._screen_point(item[1], item[2])
        return None

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        self._sync_pulse()

    def hideEvent(self, event) -> None:  # noqa: ANN001
        super().hideEvent(event)
        self._pulse_timer.stop()

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

    def _content_size(self) -> tuple[float, float] | None:
        if self._image is None or self._image.isNull():
            return None
        iw, ih = self._image.width(), self._image.height()
        if iw <= 0 or ih <= 0 or self.width() <= 1 or self.height() <= 1:
            return None
        scale = min(self.width() / iw, self.height() / ih) * self._zoom
        return iw * scale, ih * scale

    def _limit_pan(self) -> None:
        size = self._content_size()
        if size is None:
            return
        x, y = clamp_pan(
            self._pan.x(),
            self._pan.y(),
            float(self.width()),
            float(self.height()),
            size[0],
            size[1],
        )
        self._pan = QPointF(x, y)
        self.place_fit_card()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._limit_pan()

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
        plain_video_pan = self._mode == MODE_TRACK and pointer_pans(
            event.button(), event.modifiers(), surface="video"
        )
        if plain_video_pan or event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._panning = event.position()
            self._pan_origin = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
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
        video = self._video_xy(QPointF(event.pos()))
        if (
            video is not None
            and self._axis_overlay is not None
            and self._axis_frame_hit(video)
            and not _has_shift(event)
        ):
            menu = QMenu(self)
            menu.addAction("删除坐标轴", self.axis_delete_requested.emit)
            menu.exec(event.globalPos())
            return
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
            self._limit_pan()
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
                if self._mode == MODE_TRACK:
                    # Shift+左键 places a point: show it before the click.
                    if self._manual_pick or _has_shift(event):
                        self.setCursor(Qt.CursorShape.CrossCursor)
                    else:
                        self.unsetCursor()
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
            origin = getattr(self, "_pan_origin", None)
            self._pan_origin = None
            if self._mode == MODE_TRACK and self._manual_pick:
                self.setCursor(Qt.CursorShape.CrossCursor)
                moved = 99.0 if origin is None else (event.position() - origin).manhattanLength()
                if moved < 4 and event.button() == Qt.MouseButton.LeftButton:
                    # A plain click while picking: remind that Shift is needed.
                    self.manual_pick_needs_shift.emit()
            elif self._mode == MODE_TRACK:
                self.unsetCursor()
            elif self._mode == MODE_AXIS:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self._mode == MODE_PLANE:
                self.setCursor(Qt.CursorShape.CrossCursor)
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
        if (
            event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
            and self._mode == MODE_AXIS
            and self._axis_overlay is not None
        ):
            self.axis_delete_requested.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._mode == MODE_TRACK and self._manual_pick:
            self.manual_pick_cancelled.emit()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Shift and self._mode == MODE_TRACK:
            self.setCursor(Qt.CursorShape.CrossCursor)
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

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: ANN001
        if event.key() == Qt.Key.Key_Shift and self._mode == MODE_TRACK and not self._manual_pick:
            self.unsetCursor()
        super().keyReleaseEvent(event)

    def _emit_point(self, x: float, y: float) -> None:
        self.clicked_at.emit(x, y)
        self.prompted.emit(x, y, "positive")

    def paintEvent(self, event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme_qcolor("video_letterbox"))
        if self._image is None or self._image.isNull():
            return

        dest = self._dest_rect()
        assert dest is not None
        scaled = self._scaled_for(QSize(max(1, int(dest.width())), max(1, int(dest.height()))))
        painter.drawImage(int(dest.x()), int(dest.y()), scaled)
        sx = dest.width() / self._image.width()
        sy = dest.height() / self._image.height()

        if self._fit_curves:
            # The fitted curve is only a faint guide under the measured track:
            # thin, dotted, mostly transparent, painted before the trail.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 70), 1.0, Qt.PenStyle.DotLine))
            for curve in self._fit_curves:
                if len(curve) < 2:
                    continue
                path = QPainterPath()
                x0, y0 = curve[0]
                path.moveTo(dest.x() + x0 * sx, dest.y() + y0 * sy)
                for x, y in curve[1:]:
                    path.lineTo(dest.x() + x * sx, dest.y() + y * sy)
                painter.drawPath(path)

        overlays = self._overlays
        if not overlays and self._track:
            overlays = [OverlayTrack(points=self._track, color="#f0c14b", active=True)]
        for overlay in overlays:
            color = QColor(overlay.color)
            trail = QPainterPath()
            started = False
            current: tuple[float, float, bool, float, str] | None = None
            trusted_dots: list[QPointF] = []
            pending_dots: list[QPointF] = []
            suggestions = overlay.suggestions if overlay.active else {}
            for item in overlay.points:
                frame, x, y, visible = item[0], item[1], item[2], item[3]
                conf = float(item[4]) if len(item) > 4 else 1.0
                status = str(item[5]) if len(item) > 5 else ("trusted" if visible else "lost")
                source = str(item[6]) if len(item) > 6 else ""
                if frame == self._track_index:
                    current = (x, y, visible, conf, status)
                suggestion = suggestions.get(frame)
                pending = (
                    suggestion is not None
                    or frame in overlay.pending
                    or (status != "trusted" and visible)
                    or source in {"interpolated", "predicted"}
                )
                if suggestion is not None:
                    # A predicted point sits where the fit puts it, not on the
                    # tracker's doubtful candidate.
                    x, y = suggestion[0], suggestion[1]
                elif not visible:
                    started = False
                    continue
                px = dest.x() + x * sx
                py = dest.y() + y * sy
                if not started:
                    trail.moveTo(px, py)
                    started = True
                else:
                    trail.lineTo(px, py)
                if frame == self._track_index:
                    continue  # the current frame gets its own marker below
                (pending_dots if pending else trusted_dots).append(QPointF(px, py))
            width = 2.0 if overlay.active else 1.2
            color.setAlpha(210 if overlay.active else 140)
            painter.setPen(QPen(color, width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(trail)
            self._paint_trail_dots(painter, trusted_dots, pending_dots, overlay.active)
            if overlay.active:
                self._remember_pending_region(pending_dots, suggestions.get(self._track_index), dest, sx)
            here = suggestions.get(self._track_index)
            if overlay.active and here is not None:
                self._paint_pending_point(painter, dest.x() + here[0] * sx, dest.y() + here[1] * sy, here[2] * sx)
            elif overlay.active and current is not None and current[2] and current[4] == "trusted":
                px = dest.x() + current[0] * sx
                py = dest.y() + current[1] * sy
                painter.setBrush(QColor(TRUSTED_GREEN))
                ring = QColor(PENDING_YELLOW) if current[3] < LOW_CONFIDENCE else QColor(255, 255, 255, 230)
                painter.setPen(QPen(ring, 2.0))
                painter.drawEllipse(QRectF(px - 5, py - 5, 10, 10))
            elif overlay.active and current is not None and current[2]:
                px = dest.x() + current[0] * sx
                py = dest.y() + current[1] * sy
                warning = QColor(PENDING_YELLOW)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(PENDING_OUTLINE), 4.0))
                painter.drawEllipse(QRectF(px - 6, py - 6, 12, 12))
                painter.setPen(QPen(warning, 2.2))
                painter.drawEllipse(QRectF(px - 6, py - 6, 12, 12))
                painter.drawLine(QPointF(px - 4, py - 4), QPointF(px + 4, py + 4))
                painter.drawLine(QPointF(px - 4, py + 4), QPointF(px + 4, py - 4))
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
            if overlay.follow_box is not None:
                x0, y0, x1, y1 = overlay.follow_box
                rect = QRectF(
                    dest.x() + min(x0, x1) * sx,
                    dest.y() + min(y0, y1) * sy,
                    abs(x1 - x0) * sx,
                    abs(y1 - y0) * sy,
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(color, 1.5, Qt.PenStyle.SolidLine))
                painter.drawRect(rect)
            if self._show_prompts:
                for prompt in overlay.prompts:
                    if prompt.frame != self._track_index:
                        continue
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
            # Background reference points of the shake compensation: neutral,
            # so they are not mistaken for the (light-blue) fitted trajectory.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 150), 1.2))
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

    def _paint_trail_dots(
        self,
        painter: QPainter,
        trusted: list[QPointF],
        pending: list[QPointF],
        active: bool,
    ) -> None:
        """Green for measurements, bigger yellow (dark rim) for points waiting for review."""
        green = QColor(TRUSTED_GREEN)
        green.setAlpha(235 if active else 150)
        radius = 2.4 if active else 1.7
        stride = max(1, len(trusted) // 4000)  # very long tracks: thin the dots, keep the line
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(green)
        for point in trusted[::stride]:
            painter.drawEllipse(point, radius, radius)
        if not pending:
            return
        # Breathing: brightness and a soft halo follow the same slow pulse as
        # the current-frame ring, so every point waiting for review catches the eye.
        level = self.pulse_level() if active else 0.0
        if active:
            halo = QColor(PENDING_YELLOW)
            halo.setAlpha(int(25 + 70 * level))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            glow = 5.5 + 3.0 * level
            for point in pending:
                painter.drawEllipse(point, glow, glow)
        yellow = QColor(PENDING_YELLOW)
        yellow.setAlpha(int(170 + 85 * level) if active else 170)
        rim = QColor(PENDING_OUTLINE)
        rim.setAlpha(220 if active else 120)
        big = (3.4 + 0.8 * level) if active else 2.4
        painter.setPen(QPen(rim, 1.4))
        painter.setBrush(yellow)
        for point in pending:
            painter.drawEllipse(point, big, big)

    def _remember_pending_region(
        self,
        pending: list[QPointF],
        here: tuple[float, float, float] | None,
        dest: QRectF,
        scale: float,
    ) -> None:
        """Bounding box of the breathing points, so the pulse repaints only that."""
        xs = [p.x() for p in pending]
        ys = [p.y() for p in pending]
        pad = 12.0
        if here is not None:
            hx = dest.x() + here[0] * scale
            hy = dest.y() + here[1] * scale
            reach = max(14.0, here[2] * scale) + 16.0
            xs += [hx - reach, hx + reach]
            ys += [hy - reach, hy + reach]
        if not xs:
            self._pending_region = None
            return
        self._pending_region = QRect(
            int(min(xs) - pad), int(min(ys) - pad), int(max(xs) - min(xs) + 2 * pad), int(max(ys) - min(ys) + 2 * pad)
        )

    def _paint_pending_point(self, painter: QPainter, hx: float, hy: float, sigma_screen: float) -> None:
        """Yellow ring that breathes, so a point waiting for 保留/手动打点 is hard to miss."""
        level = self.pulse_level()
        radius = max(7.0, sigma_screen)
        halo = QColor(PENDING_YELLOW)
        halo.setAlpha(int(45 + 85 * level))
        grow = 3.0 + 4.0 * level
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(QRectF(hx - radius - grow, hy - radius - grow, 2 * (radius + grow), 2 * (radius + grow)))
        ring = QColor(PENDING_YELLOW)
        ring.setAlpha(int(170 + 85 * level))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(PENDING_OUTLINE), 4.2))
        painter.drawEllipse(QRectF(hx - radius, hy - radius, 2 * radius, 2 * radius))
        painter.setPen(QPen(ring, 2.4))
        painter.drawEllipse(QRectF(hx - radius, hy - radius, 2 * radius, 2 * radius))
        if self._manual_pick:
            dash = QPen(QColor(255, 255, 255, 200), 1.2, Qt.PenStyle.DashLine)
            painter.setPen(dash)
            painter.drawEllipse(QRectF(hx - radius - 9, hy - radius - 9, 2 * (radius + 9), 2 * (radius + 9)))
        painter.setPen(QPen(QColor(PENDING_YELLOW), 1.8))
        painter.drawLine(QPointF(hx - 4, hy), QPointF(hx + 4, hy))
        painter.drawLine(QPointF(hx, hy - 4), QPointF(hx, hy + 4))

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
        background = theme_qcolor("drop_bg" if not self._hover else "drop_bg_hover")
        painter.setBrush(background)
        border = QPen(theme_qcolor("drop_border" if not self._hover else "drop_border_hover"), 1.0)
        painter.setPen(border)
        painter.drawRoundedRect(outer, 6, 6)

        center_x = self.width() / 2
        center_y = self.height() / 2 - 24
        icon_box = QRectF(center_x - 34, center_y - 34, 68, 58)
        painter.setPen(QPen(theme_qcolor("drop_icon_hover" if self._hover else "drop_icon"), 1.4))
        painter.setBrush(QColor(0, 0, 0, 8) if is_light() else QColor(255, 255, 255, 7))
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
        painter.setPen(theme_qcolor("drop_title"))
        font = painter.font()
        font.setPointSize(14)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, "拖入视频开始分析")

        subtitle_rect = QRectF(0, center_y + 68, self.width(), 22)
        painter.setPen(theme_qcolor("drop_sub"))
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
            painter.setBrush(theme_qcolor("drop_status_bg"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(box, 8, 8)
            painter.setPen(theme_qcolor("drop_status"))
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
        self.setToolTip("上方两个三角是跟踪范围。播放和跟踪都不会跑出去。")
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
        shade = QColor(0, 0, 0, 28) if is_light() else QColor(18, 18, 18, 170)
        if left_x > groove.left():
            painter.fillRect(
                QRectF(groove.left(), mid - 3, left_x - groove.left(), 6), shade
            )
        if right_x < groove.right():
            painter.fillRect(QRectF(right_x, mid - 3, groove.right() - right_x, 6), shade)

        marker_color = theme_qcolor("marker" if self.isEnabled() else "marker_off")
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
        painter.fillPath(playhead, theme_qcolor("playhead" if self.isEnabled() else "playhead_off"))


class StepStepper(QWidget):
    """Tracker-style frame step: |◀  N  ▶| with a fixed set of step sizes."""

    step_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepStepper")
        self.setFixedHeight(26)

        back = QToolButton()
        self._back = back
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
        self._forward = forward
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

    def apply_theme(self) -> None:
        self._back.setIcon(prev_icon())
        self._forward.setIcon(next_icon())
