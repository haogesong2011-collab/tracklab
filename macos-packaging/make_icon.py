#!/usr/bin/env python3
"""Render TrackLab.icns from vector drawing (no binary asset in git)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)


SIZES = (16, 32, 64, 128, 256, 512, 1024)


def render_icon(size: int) -> QImage:
    image = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    margin = size * 0.06
    radius = size * 0.22
    background = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#252525"))
    painter.drawRoundedRect(background, radius, radius)

    accent = QColor("#80cbc4")
    path = QPainterPath()
    path.moveTo(size * 0.22, size * 0.68)
    path.quadTo(size * 0.38, size * 0.28, size * 0.55, size * 0.52)
    path.quadTo(size * 0.68, size * 0.72, size * 0.82, size * 0.36)
    pen = QPen(accent, max(1.5, size * 0.055), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(accent)
    painter.drawEllipse(QPointF(size * 0.22, size * 0.68), size * 0.045, size * 0.045)
    painter.drawEllipse(QPointF(size * 0.82, size * 0.36), size * 0.05, size * 0.05)

    font = QFont()
    font.setFamily("Helvetica Neue")
    font.setBold(True)
    font.setPixelSize(max(8, int(size * 0.42)))
    painter.setFont(font)
    painter.setPen(QColor("#f5f5f5"))
    painter.drawText(QRectF(0, size * 0.08, size, size * 0.55), Qt.AlignmentFlag.AlignCenter, "T")
    painter.end()
    return image


def main() -> int:
    QGuiApplication.instance() or QGuiApplication(sys.argv)
    out_dir = Path(__file__).resolve().parent
    iconset = out_dir / "TrackLab.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)
    mapping = {
        16: ["icon_16x16.png"],
        32: ["icon_16x16@2x.png", "icon_32x32.png"],
        64: ["icon_32x32@2x.png"],
        128: ["icon_128x128.png"],
        256: ["icon_128x128@2x.png", "icon_256x256.png"],
        512: ["icon_256x256@2x.png", "icon_512x512.png"],
        1024: ["icon_512x512@2x.png"],
    }
    for size in SIZES:
        image = render_icon(size)
        for name in mapping.get(size, []):
            if not image.save(str(iconset / name), "PNG"):
                print(f"failed to write {name}", file=sys.stderr)
                return 1
    icns = out_dir / "TrackLab.icns"
    iconutil = shutil.which("iconutil")
    if iconutil is None:
        print("iconutil not found; cannot write TrackLab.icns", file=sys.stderr)
        return 1
    completed = subprocess.run(
        [iconutil, "-c", "icns", "-o", str(icns), str(iconset)],
        check=False,
    )
    if completed.returncode != 0 or not icns.is_file():
        print("iconutil failed", file=sys.stderr)
        return completed.returncode or 1
    print(icns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
