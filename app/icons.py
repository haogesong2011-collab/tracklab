from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

IconDrawer = Callable[[QPainter, int, QColor], None]


def _icon(drawer: IconDrawer, color: str = "#bdbdbd", disabled: str = "#666666") -> QIcon:
    icon = QIcon()
    icon.addPixmap(_pix(drawer, QColor(color)), QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(_pix(drawer, QColor(disabled)), QIcon.Mode.Disabled, QIcon.State.Off)
    return icon


def _pix(drawer: IconDrawer, color: QColor, size: int = 20) -> QPixmap:
    dpr = 2
    pixmap = QPixmap(size * dpr, size * dpr)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    drawer(painter, size, color)
    painter.end()
    return pixmap


def _play(p: QPainter, s: int, c: QColor) -> None:
    path = QPainterPath()
    path.moveTo(s * 0.32, s * 0.22)
    path.lineTo(s * 0.32, s * 0.78)
    path.lineTo(s * 0.82, s * 0.50)
    path.closeSubpath()
    p.fillPath(path, c)


def _pause(p: QPainter, s: int, c: QColor) -> None:
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(int(s * 0.28), int(s * 0.22), int(s * 0.16), int(s * 0.56), 2, 2)
    p.drawRoundedRect(int(s * 0.56), int(s * 0.22), int(s * 0.16), int(s * 0.56), 2, 2)


def _prev(p: QPainter, s: int, c: QColor) -> None:
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(int(s * 0.22), int(s * 0.24), int(s * 0.12), int(s * 0.52), 1, 1)
    path = QPainterPath()
    path.moveTo(s * 0.78, s * 0.22)
    path.lineTo(s * 0.78, s * 0.78)
    path.lineTo(s * 0.38, s * 0.50)
    path.closeSubpath()
    p.fillPath(path, c)


def _next(p: QPainter, s: int, c: QColor) -> None:
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    path = QPainterPath()
    path.moveTo(s * 0.22, s * 0.22)
    path.lineTo(s * 0.22, s * 0.78)
    path.lineTo(s * 0.62, s * 0.50)
    path.closeSubpath()
    p.fillPath(path, c)
    p.drawRoundedRect(int(s * 0.66), int(s * 0.24), int(s * 0.12), int(s * 0.52), 1, 1)


def _loop(p: QPainter, s: int, c: QColor) -> None:
    pen = QPen(c, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    box = QRectF(s * 0.19, s * 0.19, s * 0.62, s * 0.62)
    p.drawArc(box, 35 * 16, 135 * 16)
    p.drawArc(box, 215 * 16, 135 * 16)
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    first = QPainterPath()
    first.moveTo(s * 0.77, s * 0.20)
    first.lineTo(s * 0.78, s * 0.40)
    first.lineTo(s * 0.60, s * 0.31)
    first.closeSubpath()
    second = QPainterPath()
    second.moveTo(s * 0.23, s * 0.80)
    second.lineTo(s * 0.22, s * 0.60)
    second.lineTo(s * 0.40, s * 0.69)
    second.closeSubpath()
    p.fillPath(first, c)
    p.fillPath(second, c)


def _folder(p: QPainter, s: int, c: QColor) -> None:
    pen = QPen(c, 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(s * 0.15, s * 0.34)
    path.lineTo(s * 0.39, s * 0.34)
    path.lineTo(s * 0.47, s * 0.25)
    path.lineTo(s * 0.82, s * 0.25)
    path.lineTo(s * 0.82, s * 0.73)
    path.lineTo(s * 0.15, s * 0.73)
    path.closeSubpath()
    p.drawPath(path)


def _save(p: QPainter, s: int, c: QColor) -> None:
    p.setPen(QPen(c, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(s * 0.20, s * 0.16, s * 0.60, s * 0.68), 2, 2)
    p.drawRect(QRectF(s * 0.31, s * 0.16, s * 0.36, s * 0.24))
    p.drawRect(QRectF(s * 0.30, s * 0.57, s * 0.40, s * 0.27))


def _film(p: QPainter, s: int, c: QColor) -> None:
    p.setPen(QPen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(s * 0.16, s * 0.22, s * 0.68, s * 0.56), 2, 2)
    for x in (0.24, 0.68):
        p.drawRect(QRectF(s * x, s * 0.27, s * 0.08, s * 0.09))
        p.drawRect(QRectF(s * x, s * 0.64, s * 0.08, s * 0.09))


def _axis(p: QPainter, s: int, c: QColor) -> None:
    pen = QPen(c, 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawLine(int(s * 0.20), int(s * 0.75), int(s * 0.80), int(s * 0.75))
    p.drawLine(int(s * 0.28), int(s * 0.84), int(s * 0.28), int(s * 0.18))
    p.drawLine(int(s * 0.80), int(s * 0.75), int(s * 0.70), int(s * 0.68))
    p.drawLine(int(s * 0.80), int(s * 0.75), int(s * 0.70), int(s * 0.82))
    p.drawLine(int(s * 0.28), int(s * 0.18), int(s * 0.21), int(s * 0.28))
    p.drawLine(int(s * 0.28), int(s * 0.18), int(s * 0.35), int(s * 0.28))


def _ruler(p: QPainter, s: int, c: QColor) -> None:
    p.setPen(QPen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.save()
    p.translate(s * 0.50, s * 0.50)
    p.rotate(-28)
    p.drawRoundedRect(QRectF(-s * 0.35, -s * 0.12, s * 0.70, s * 0.24), 2, 2)
    for x in (-0.22, -0.06, 0.10, 0.26):
        p.drawLine(int(s * x), int(-s * 0.12), int(s * x), int(-s * 0.02))
    p.restore()


def _track(p: QPainter, s: int, c: QColor) -> None:
    points = [(0.20, 0.70), (0.36, 0.52), (0.54, 0.58), (0.74, 0.28)]
    p.setPen(QPen(c, 1.3))
    for start, end in zip(points, points[1:]):
        p.drawLine(int(s * start[0]), int(s * start[1]), int(s * end[0]), int(s * end[1]))
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    for x, y in points:
        p.drawEllipse(QRectF(s * x - 2.2, s * y - 2.2, 4.4, 4.4))


def _spark(p: QPainter, s: int, c: QColor) -> None:
    pen = QPen(c, 1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    for x1, y1, x2, y2 in (
        (0.50, 0.16, 0.50, 0.38),
        (0.50, 0.62, 0.50, 0.84),
        (0.16, 0.50, 0.38, 0.50),
        (0.62, 0.50, 0.84, 0.50),
        (0.26, 0.26, 0.39, 0.39),
        (0.61, 0.61, 0.74, 0.74),
        (0.74, 0.26, 0.61, 0.39),
        (0.39, 0.61, 0.26, 0.74),
    ):
        p.drawLine(int(s * x1), int(s * y1), int(s * x2), int(s * y2))
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(s * 0.44, s * 0.44, s * 0.12, s * 0.12))


def _eye(p: QPainter, s: int, c: QColor) -> None:
    p.setPen(QPen(c, 1.4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(s * 0.12, s * 0.50)
    path.cubicTo(s * 0.30, s * 0.24, s * 0.70, s * 0.24, s * 0.88, s * 0.50)
    path.cubicTo(s * 0.70, s * 0.76, s * 0.30, s * 0.76, s * 0.12, s * 0.50)
    p.drawPath(path)
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QRectF(s * 0.43, s * 0.43, s * 0.14, s * 0.14))


def _broom(p: QPainter, s: int, c: QColor) -> None:
    pen = QPen(c, 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(int(s * 0.22), int(s * 0.28), int(s * 0.78), int(s * 0.28))
    p.drawLine(int(s * 0.36), int(s * 0.22), int(s * 0.64), int(s * 0.22))
    p.drawRoundedRect(int(s * 0.28), int(s * 0.32), int(s * 0.44), int(s * 0.48), 2, 2)
    p.drawLine(int(s * 0.40), int(s * 0.40), int(s * 0.40), int(s * 0.70))
    p.drawLine(int(s * 0.50), int(s * 0.40), int(s * 0.50), int(s * 0.70))
    p.drawLine(int(s * 0.60), int(s * 0.40), int(s * 0.60), int(s * 0.70))


def _zoom(p: QPainter, s: int, c: QColor) -> None:
    pen = QPen(c, 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(s * 0.18, s * 0.16, s * 0.46, s * 0.46))
    p.drawLine(int(s * 0.58), int(s * 0.58), int(s * 0.82), int(s * 0.82))
    p.drawLine(int(s * 0.31), int(s * 0.39), int(s * 0.51), int(s * 0.39))
    p.drawLine(int(s * 0.41), int(s * 0.29), int(s * 0.41), int(s * 0.49))


def play_icon() -> QIcon:
    return _icon(_play, "#dddddd")


def pause_icon() -> QIcon:
    return _icon(_pause, "#dddddd")


def prev_icon() -> QIcon:
    return _icon(_prev, "#c8c8c8")


def next_icon() -> QIcon:
    return _icon(_next, "#c8c8c8")


def loop_icon() -> QIcon:
    icon = _icon(_loop, "#cccccc")
    # Checked state paints on a light button, so the glyph has to flip dark.
    icon.addPixmap(_pix(_loop, QColor("#1d1d1d")), QIcon.Mode.Normal, QIcon.State.On)
    icon.addPixmap(_pix(_loop, QColor("#1d1d1d")), QIcon.Mode.Active, QIcon.State.On)
    return icon


TOOLBAR_COLORS = {
    "open": "#5aa3e8",
    "save": "#e0b04a",
    "video": "#62c47a",
    "ruler": "#e08a4a",
    "axis": "#b07ae0",
    "track": "#e05c5c",
    "ai": "#3dccc0",
    "view": "#6cb4e8",
    "zoom": "#d4c24a",
    "cache": "#c47a6a",
}


def toolbar_icon(name: str) -> QIcon:
    drawers: dict[str, IconDrawer] = {
        "open": _folder,
        "save": _save,
        "video": _film,
        "axis": _axis,
        "ruler": _ruler,
        "track": _track,
        "ai": _spark,
        "view": _eye,
        "zoom": _zoom,
        "cache": _broom,
    }
    return _icon(drawers[name], TOOLBAR_COLORS[name])


def icon_size() -> QSize:
    return QSize(18, 18)
