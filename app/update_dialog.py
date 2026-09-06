"""Qt dialog and background worker for GitHub Releases update checks."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from app.update_checker import UpdateInfo, UpdateStatus, check_for_update, status_bar_message


class UpdateCheckWorker(QObject):
    finished = Signal(object)

    def __init__(
        self,
        current: str,
        *,
        skipped: str = "",
        honor_skip: bool = True,
        parent=None,  # noqa: ANN001
    ) -> None:
        super().__init__(parent)
        self._current = current
        self._skipped = skipped
        self._honor_skip = honor_skip

    def run(self) -> None:
        try:
            info = check_for_update(
                self._current,
                skipped=self._skipped,
                honor_skip=self._honor_skip,
            )
        except Exception as exc:  # noqa: BLE001
            from app.update_checker import UpdateInfo, UpdateStatus

            info = UpdateInfo(
                status=UpdateStatus.ERROR,
                current=self._current,
                message=f"检查更新失败：{exc}",
            )
        self.finished.emit(info)


def run_update_check_in_thread(
    current: str,
    *,
    skipped: str = "",
    honor_skip: bool = True,
    on_finished: Callable[[UpdateInfo], None] | None = None,
) -> tuple[QThread, UpdateCheckWorker]:
    thread = QThread()
    worker = UpdateCheckWorker(current, skipped=skipped, honor_skip=honor_skip)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_finished is not None:
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(thread.quit)
    return thread, worker


class UpdateDialog(QDialog):
    def __init__(self, parent: QWidget | None, info: UpdateInfo) -> None:
        super().__init__(parent)
        self._choice = "later"
        self.setWindowTitle("发现新版本")
        self.setModal(True)
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        latest = info.latest or ""
        summary = QLabel(f"当前版本 {info.current}，最新版本 {latest}。")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        if info.published_at:
            published = QLabel(f"发布时间：{info.published_at}")
            layout.addWidget(published)

        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setPlainText(info.notes or "（无更新说明）")
        notes.setMinimumHeight(160)
        layout.addWidget(notes)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        skip = QPushButton("跳过此版本")
        later = QPushButton("稍后提醒")
        download = QPushButton("前往下载")
        download.setDefault(True)
        skip.clicked.connect(self._skip)
        later.clicked.connect(self._later)
        download.clicked.connect(self._download)
        buttons.addWidget(skip)
        buttons.addWidget(later)
        buttons.addWidget(download)
        layout.addLayout(buttons)

    def choice(self) -> str:
        return self._choice

    def _download(self) -> None:
        self._choice = "download"
        self.accept()

    def _later(self) -> None:
        self._choice = "later"
        self.reject()

    def _skip(self) -> None:
        self._choice = "skip"
        self.accept()


def open_release_page(info: UpdateInfo) -> bool:
    url = info.download_url
    return QDesktopServices.openUrl(QUrl(url))


def show_update_result(parent: QWidget | None, info: UpdateInfo, *, manual: bool) -> str | None:
    """Return download/later/skip for an available update; otherwise None."""
    if info.status is UpdateStatus.AVAILABLE:
        dialog = UpdateDialog(parent, info)
        dialog.exec()
        choice = dialog.choice()
        if choice == "download":
            open_release_page(info)
        return choice
    if not manual:
        return None
    title = "检查更新"
    text = status_bar_message(info)
    if info.status is UpdateStatus.LATEST:
        QMessageBox.information(parent, title, text)
    else:
        QMessageBox.warning(parent, title, text)
    return None
