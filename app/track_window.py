"""Floating Tracker-style track control window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QHideEvent, QShowEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.track_panels import TrackListPanel


class TrackManagerWindow(QWidget):
    def __init__(self, panel: TrackListPanel, parent: QWidget | None = None) -> None:
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        super().__init__(parent, flags)
        self.setObjectName("trackManagerWindow")
        self.setWindowTitle("轨迹")
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setMinimumSize(240, 360)
        self.resize(280, 480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(panel)
        self._panel = panel
        self._visibility_hook = None

    def set_visibility_hook(self, hook) -> None:  # noqa: ANN001
        self._visibility_hook = hook

    def showEvent(self, event: QShowEvent) -> None:  # noqa: ANN001
        super().showEvent(event)
        if self._visibility_hook is not None:
            self._visibility_hook(True)

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: ANN001
        super().hideEvent(event)
        if self._visibility_hook is not None:
            self._visibility_hook(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: ANN001
        event.ignore()
        self.hide()
