"""VS Code-style bottom-right download progress toast."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ai.model_manager import ModelSpec


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


class DownloadToast(QFrame):
    cancelled = Signal()

    def __init__(self, host: QWidget) -> None:
        super().__init__(
            host,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint,
        )
        self.setObjectName("downloadToast")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedWidth(360)
        host_window = host.window()
        if host_window is not None and host_window.styleSheet():
            self.setStyleSheet(host_window.styleSheet())
        self.hide()

        self._title = QLabel("正在下载")
        self._title.setObjectName("downloadToastTitle")
        self._detail = QLabel("")
        self._detail.setObjectName("downloadToastDetail")
        self._detail.setWordWrap(True)
        self._status = QLabel("准备中…")
        self._status.setObjectName("downloadToastStatus")
        self._bar = QProgressBar()
        self._bar.setObjectName("downloadToastBar")
        self._bar.setRange(0, 0)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._cancel = QPushButton("取消")
        self._cancel.setObjectName("downloadToastCancel")
        self._cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel.clicked.connect(self.cancelled.emit)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self._title, stretch=1)
        header.addWidget(self._cancel, stretch=0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self._detail)
        layout.addWidget(self._bar)
        layout.addWidget(self._status)

    def set_download(self, spec: ModelSpec) -> None:
        label = "SAM 2.1 Tiny" if spec.model_id.endswith("tiny") else "SAM 2.1 Small"
        self.begin(f"正在下载 {label}", spec.filename)

    def begin(self, title: str, detail: str = "") -> None:
        self._title.setText(title)
        self._detail.setText(detail)
        self._status.setText("准备中…")
        self._bar.setRange(0, 0)
        self._bar.setValue(0)
        self._cancel.setEnabled(True)
        self.adjustSize()
        self.reposition()

    def set_progress(self, received: int, total: int) -> None:
        received = max(0, int(received))
        total = max(0, int(total))
        if total > 0:
            self._bar.setRange(0, total)
            self._bar.setValue(min(received, total))
            pct = min(100, int(received * 100 / total)) if total else 0
            self._status.setText(f"{format_bytes(received)} / {format_bytes(total)}  ·  {pct}%")
        else:
            self._bar.setRange(0, 0)
            self._status.setText(f"已下载 {format_bytes(received)}…")
        self.reposition()

    def set_stage(self, stage: str) -> None:
        if stage == "verify":
            if self._bar.maximum() > 0:
                self._bar.setValue(self._bar.maximum())
            self._status.setText("正在校验 SHA-256…")
        elif stage == "download":
            self._status.setText("正在下载…")
        elif stage == "checksums":
            self._status.setText("正在读取校验和…")
        elif stage == "extract":
            self._status.setText("正在解包安装包…")

    def set_finished(self, status: str = "权重已就绪") -> None:
        self._title.setText("下载完成")
        if self._bar.maximum() > 0:
            self._bar.setValue(self._bar.maximum())
        else:
            self._bar.setRange(0, 1)
            self._bar.setValue(1)
        self._status.setText(status)
        self._cancel.setEnabled(False)
        self.reposition()

    def reposition(self) -> None:
        host = self.parentWidget()
        if host is None:
            return
        self.adjustSize()
        margin = 16
        status = 28
        corner = host.mapToGlobal(host.rect().bottomRight())
        self.move(
            corner.x() - self.width() - margin,
            corner.y() - self.height() - margin - status,
        )
        app = QApplication.instance()
        if app is not None and app.platformName() != "offscreen":
            self.raise_()
