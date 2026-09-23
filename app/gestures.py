"""Pan and zoom gestures shared by the video view and the function charts.

A trackpad two-finger drag pans. A pinch zooms. A mouse wheel zooms.
Dragging with the mouse pans when that button is not already the tool
for the surface (placing a point, drawing a ruler, or scrubbing a chart).
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
        # Fingers up (positive dy) -> picture up (negative screen y) when
        # natural scrolling is on. Classic scrolling flips both axes.
        screen_dx, screen_dy = dx, -dy
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
    draws a box. Chart: left drag scrubs the playhead, so pan is the right
    button, the middle button, or Alt+left.
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
        return not control
    if surface == "chart" and button == Qt.MouseButton.RightButton:
        return True
    return False


def shift_view_center(
    center_t: float,
    center_v: float,
    dx: float,
    dy: float,
    plot_w: float,
    plot_h: float,
    t0: float,
    t1: float,
    v0: float,
    v1: float,
) -> tuple[float, float]:
    """Move a chart window so the curve follows a screen drag.

    ``dx``/``dy`` are screen pixels, +x right, +y down. Time grows to the
    right and the value grows upward, so dragging right reveals earlier
    times and dragging down reveals higher values.
    """
    tspan = t1 - t0
    vspan = v1 - v0
    if plot_w < 1.0 or plot_h < 1.0 or tspan == 0.0 or vspan == 0.0:
        return center_t, center_v
    return (
        center_t - dx / plot_w * tspan,
        center_v + dy / plot_h * vspan,
    )
