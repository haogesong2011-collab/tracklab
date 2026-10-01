"""Pan and zoom gestures for the video view.

A trackpad two-finger drag pans. A pinch zooms. A mouse wheel zooms.
A plain left drag pans the video. Control or Command left still draws a box.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QInputDevice, QNativeGestureEvent, QWheelEvent

# One mouse-wheel notch. Trackpads that only report angleDelta use this
# to turn the notch into a pan of about this many screen pixels.
_NOTCH = 120.0
_TRACKPAD_PIXELS_PER_NOTCH = 48.0


@dataclass(frozen=True)
class ViewGesture:
    """One input interpreted as a view change.

    Pan dx/dy move the picture on screen: +x right, +y down.
    Zoom uses ``steps`` (mouse wheel notches, positive zooms in) or, when
    ``factor`` is set, a direct scale such as a pinch.
    """

    kind: str
    dx: float = 0.0
    dy: float = 0.0
    steps: float = 0.0
    factor: float | None = None


def classify_scroll(
    *,
    pixel_dx: float,
    pixel_dy: float,
    angle_dx: float,
    angle_dy: float,
    touchpad: bool,
    scrolling: bool,
    inverted: bool,
) -> ViewGesture:
    """Turn a wheel/trackpad scroll into a pan or a zoom.

    ``pixel_*`` follows the fingers: positive y is up. Screen y grows
    downward, so a natural-scrolling (``inverted``) upward drag moves the
    picture up. Classic scrolling moves it the other way. A mouse notch
    stays a zoom.
    """
    pad = touchpad or (scrolling and (pixel_dx or pixel_dy))
    if pad:
        dx, dy = pixel_dx, pixel_dy
        if dx == 0.0 and dy == 0.0:
            dx = angle_dx / _NOTCH * _TRACKPAD_PIXELS_PER_NOTCH
            dy = angle_dy / _NOTCH * _TRACKPAD_PIXELS_PER_NOTCH
        if dx == 0.0 and dy == 0.0:
            return ViewGesture("none")
        # pixelDelta.x already matches the picture (right is right). The same
        # negation on y ran the picture the wrong way on a Mac trackpad, so y
        # is left as reported. Classic scrolling still flips both axes.
        screen_dx, screen_dy = dx, dy
        if not inverted:
            screen_dx, screen_dy = -screen_dx, -screen_dy
        return ViewGesture("pan", screen_dx, screen_dy)
    steps = angle_dy / _NOTCH
    if steps == 0.0:
        return ViewGesture("none")
    return ViewGesture("zoom", steps=steps)


def gesture_from_wheel(event: QWheelEvent) -> ViewGesture:
    pixel = event.pixelDelta()
    angle = event.angleDelta()
    touchpad = event.deviceType() == QInputDevice.DeviceType.TouchPad
    scrolling = event.phase() != Qt.ScrollPhase.NoScrollPhase
    return classify_scroll(
        pixel_dx=float(pixel.x()),
        pixel_dy=float(pixel.y()),
        angle_dx=float(angle.x()),
        angle_dy=float(angle.y()),
        touchpad=touchpad,
        scrolling=scrolling,
        inverted=bool(event.inverted()),
    )


def gesture_from_native(event: QEvent) -> ViewGesture | None:
    """Pinch, smart-zoom, and native pan. None for every other event."""
    if event.type() != QEvent.Type.NativeGesture:
        return None
    native = event
    if not isinstance(native, QNativeGestureEvent):
        return None
    kind = native.gestureType()
    if kind == Qt.NativeGestureType.ZoomNativeGesture:
        factor = 1.0 + float(native.value())
        if factor <= 0.05:
            return ViewGesture("none")
        return ViewGesture("zoom", factor=factor)
    if kind == Qt.NativeGestureType.SmartZoomNativeGesture:
        return ViewGesture("reset")
    if kind == Qt.NativeGestureType.PanNativeGesture:
        delta = native.delta()
        return ViewGesture("pan", float(delta.x()), float(delta.y()))
    return None


def pointer_pans(button: Qt.MouseButton, modifiers: Qt.KeyboardModifier, *, surface: str) -> bool:
    """True when this mouse button should drag the view rather than the tool.

    Video, in track mode: a plain left drag pans. Control/Cmd-left still
    draws a box, and Shift-left places a point.
    """
    if button == Qt.MouseButton.MiddleButton:
        return True
    alt = bool(modifiers & Qt.KeyboardModifier.AltModifier)
    if button == Qt.MouseButton.LeftButton and alt:
        return True
    if surface == "video" and button == Qt.MouseButton.LeftButton:
        control = bool(
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        )
        shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        return not control and not shift
    return False


def clamp_axis_offset(
    offset: float, view: float, content: float, keep_fraction: float = 0.5
) -> float:
    """Keep at least half of the smaller side of the content inside the view."""
    if view <= 1.0 or content <= 1.0:
        return offset
    keep = min(content, view) * keep_fraction
    keep = min(max(keep, 1.0), content, view)
    min_off = keep - content
    max_off = view - keep
    if min_off > max_off:
        if content <= view:
            min_off, max_off = 0.0, view - content
        else:
            pinned = (view - content) / 2.0
            min_off = max_off = pinned
    return min(max(offset, min_off), max_off)


def clamp_pan(
    pan_x: float,
    pan_y: float,
    view_w: float,
    view_h: float,
    content_w: float,
    content_h: float,
    keep_fraction: float = 0.5,
) -> tuple[float, float]:
    """Pan of a picture that is centered when the pan is zero."""
    base_x = (view_w - content_w) / 2.0
    base_y = (view_h - content_h) / 2.0
    left = clamp_axis_offset(base_x + pan_x, view_w, content_w, keep_fraction)
    top = clamp_axis_offset(base_y + pan_y, view_h, content_h, keep_fraction)
    return left - base_x, top - base_y
