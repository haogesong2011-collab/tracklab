"""Qt dialog and background worker for GitHub Releases update checks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.contracts import CancelToken
from app.download_toast import format_bytes
from app.update_checker import (
    UpdateInfo,
    UpdateStatus,
    can_self_update,
    check_for_update,
    status_bar_message,
)


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


class SelfUpdateWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int, int)
    stage = Signal(str)

    def __init__(
        self,
        info: UpdateInfo,
        bundle: Path,
        current: str,
        parent=None,  # noqa: ANN001
    ) -> None:
        super().__init__(parent)
        self._info = info
        self._bundle = bundle
        self._current = current
        self._cancel = CancelToken()

    def cancel(self) -> None:
        self._cancel.cancel()

    def run(self) -> None:
        from app.self_update import SelfUpdateCancelled, SelfUpdateError, prepare_update

        try:
            new_app = prepare_update(
                self._info,
                bundle=self._bundle,
                current=self._current,
                progress=self.progress.emit,
                stage=self.stage.emit,
                cancel=self._cancel,
            )
        except SelfUpdateCancelled:
            self.cancelled.emit()
            return
        except SelfUpdateError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"更新失败：{exc}")
            return
        self.finished.emit(new_app)


def run_self_update_in_thread(
    info: UpdateInfo,
    bundle: Path,
    current: str,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    on_stage: Callable[[str], None] | None = None,
    on_finished: Callable[[Path], None] | None = None,
    on_failed: Callable[[str], None] | None = None,
    on_cancelled: Callable[[], None] | None = None,
) -> tuple[QThread, SelfUpdateWorker]:
    thread = QThread()
    worker = SelfUpdateWorker(info, bundle, current)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_progress is not None:
        worker.progress.connect(on_progress, Qt.ConnectionType.QueuedConnection)
    if on_stage is not None:
        worker.stage.connect(on_stage, Qt.ConnectionType.QueuedConnection)
    if on_finished is not None:
        worker.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
    if on_failed is not None:
        worker.failed.connect(on_failed, Qt.ConnectionType.QueuedConnection)
    if on_cancelled is not None:
        worker.cancelled.connect(on_cancelled, Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.cancelled.connect(thread.quit)
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

        self._self_update = can_self_update(info)
        if self._self_update:
            size = 0
            installer = info.installer
            if installer is not None:
                size = installer.size
            extra = "约 " + format_bytes(size) if size else "安装包"
            hint = QLabel(
                f"点击「立即更新」将下载{extra}，校验后替换当前应用并重启。"
            )
            hint.setWordWrap(True)
            hint.setObjectName("panelHint")
            layout.addWidget(hint)

        notes = QTextEdit()
        notes.setReadOnly(True)
        notes.setPlainText(info.notes or "（无更新说明）")
        notes.setMinimumHeight(160)
        layout.addWidget(notes)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        skip = QPushButton("跳过此版本")
        later = QPushButton("稍后提醒")
        download = QPushButton("立即更新" if self._self_update else "前往下载")
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
        if choice == "download" and not can_self_update(info):
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
