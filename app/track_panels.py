"""Tracker-style side list and data table for named trajectories."""

from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt, QMimeData, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ai.contracts import TrackLayer, TrackResult
from ai.kinematics import accel_unit_for, is_low_confidence, quality_tooltip, series_for_result
from app.data_views import AX_NAME, AY_NAME, VX_NAME, VY_NAME, speed_axis_label
from engine.video_index import VideoInfo


class TrackListPanel(QWidget):
    new_requested = Signal()
    delete_requested = Signal()
    selection_changed = Signal(str)
    rename_requested = Signal(str, str)
    visibility_toggled = Signal(str, bool)
    track_requested = Signal()
    cancel_requested = Signal()
    shake_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trackListPanel")
        self.setMinimumWidth(220)
        title = QLabel("轨迹")
        title.setObjectName("panelTitle")

        self._list = QListWidget()
        self._list.setObjectName("trackList")
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.currentItemChanged.connect(self._on_current)
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(self._begin_rename)

        new_btn = QPushButton("新建")
        new_btn.setObjectName("panelButton")
        new_btn.clicked.connect(self.new_requested.emit)
        del_btn = QPushButton("删除")
        del_btn.setObjectName("panelButton")
        del_btn.clicked.connect(self.delete_requested.emit)
        self._track_btn = QPushButton("自动跟踪")
        self._track_btn.setObjectName("panelButtonPrimary")
        self._track_btn.clicked.connect(self.track_requested.emit)
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setObjectName("panelButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)

        self._shake_box = QCheckBox("背景补偿")
        self._shake_box.setObjectName("panelCheck")
        self._shake_box.setChecked(True)
        self._shake_box.setToolTip("减镜头晃。不稳时自动用原来的坐标。")
        self._shake_box.toggled.connect(self.shake_toggled.emit)

        self._progress = QProgressBar()
        self._progress.setObjectName("trackProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._status = QLabel("Control 拖动框选目标，Shift+左键点击加点")
        self._status.setObjectName("panelHint")
        self._status.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(new_btn)
        buttons.addWidget(del_btn)
        run = QHBoxLayout()
        run.setSpacing(6)
        run.addWidget(self._track_btn)
        run.addWidget(self._cancel_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addLayout(buttons)
        layout.addWidget(self._list, stretch=1)
        layout.addLayout(run)
        layout.addWidget(self._shake_box)
        layout.addWidget(self._progress)
        layout.addWidget(self._status)

    def set_tracks(self, layers: list[TrackLayer], active_id: str | None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for layer in layers:
            item = QListWidgetItem(layer.name)
            item.setData(Qt.ItemDataRole.UserRole, layer.track_id)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsEditable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked
            )
            item.setForeground(QColor(layer.color))
            item.setText(layer.name)
            bits: list[str] = []
            bits.append(
                "指定表面点"
                if layer.tracking_target.value == "surface_point"
                else "物体中心"
            )
            if layer.result:
                n = sum(1 for p in layer.result.points if p.usable_for_measurement())
                bits.append(f"{n} 个点")
            if layer.status == "running":
                bits.append("跟踪中")
            elif layer.status == "error":
                bits.append("失败")
            item.setToolTip(" · ".join(bits) if bits else layer.name)
            self._list.addItem(item)
            if layer.track_id == active_id:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)

    def set_running(self, running: bool) -> None:
        self._track_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        self._track_btn.setText("跟踪中…" if running else "自动跟踪")

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        total = max(total, 1)
        self._progress.setValue(int(100 * current / total))
        if message:
            self._status.setText(message)
        else:
            self._status.setText(f"SAM 2 跟踪 {current} / {total}")

    def set_hint(self, text: str) -> None:
        self._status.setText(text)

    def set_shake_enabled(self, enabled: bool) -> None:
        self._shake_box.blockSignals(True)
        self._shake_box.setChecked(enabled)
        self._shake_box.blockSignals(False)

    def _on_current(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        track_id = current.data(Qt.ItemDataRole.UserRole)
        if track_id:
            self.selection_changed.emit(str(track_id))

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        track_id = item.data(Qt.ItemDataRole.UserRole)
        if not track_id:
            return
        self.visibility_toggled.emit(
            str(track_id), item.checkState() == Qt.CheckState.Checked
        )
        name = item.text().strip()
        if name:
            self.rename_requested.emit(str(track_id), name)

    def _begin_rename(self, item: QListWidgetItem) -> None:
        self._list.editItem(item)


class TrackDataPanel(QWidget):
    frame_activated = Signal(int)
    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trackDataPanel")
        self.setMinimumHeight(140)
        self._table = QTableWidget(0, 10)
        self._table.setObjectName("trackTable")
        self._position_unit = "px"
        self._speed_unit = "px/s"
        self._accel_unit = "px/s²"
        self._table.setHorizontalHeaderLabels(self._header_labels())
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setWordWrap(False)
        self._table.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_menu)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setMinimumSectionSize(40)
        header.setStretchLastSection(False)
        self._table.cellClicked.connect(self._on_cell)
        copy = QShortcut(QKeySequence.StandardKey.Copy, self._table)
        copy.setContext(Qt.ShortcutContext.WidgetShortcut)
        copy.activated.connect(self.copy_selection)
        export = QPushButton("导出 CSV")
        export.setObjectName("dataExportButton")
        export.setToolTip("把当前轨迹导出成 CSV，用 Numbers 或 Excel 打开")
        export.clicked.connect(self.export_requested.emit)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addStretch()
        bar.addWidget(export)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(4)
        layout.addLayout(bar)
        layout.addWidget(self._table)

    def _header_labels(self) -> list[str]:
        unit = self._position_unit
        speed = self._speed_unit
        accel = self._accel_unit
        return [
            "帧",
            "时间 (s)",
            f"x ({unit})",
            f"y ({unit})",
            f"{VX_NAME} ({speed_axis_label(speed)})",
            f"{VY_NAME} ({speed_axis_label(speed)})",
            f"{AX_NAME} ({accel})",
            f"{AY_NAME} ({accel})",
            "可见",
            "追踪质量",
        ]

    def set_units(
        self,
        position_unit: str,
        speed_unit: str,
        accel_unit: str | None = None,
    ) -> None:
        accel = accel_unit or accel_unit_for(speed_unit)
        if (
            position_unit == self._position_unit
            and speed_unit == self._speed_unit
            and accel == self._accel_unit
        ):
            return
        self._position_unit = position_unit
        self._speed_unit = speed_unit
        self._accel_unit = accel
        self._table.setHorizontalHeaderLabels(self._header_labels())

    def set_layer(
        self,
        layer: TrackLayer | None,
        info: VideoInfo | None,
        *,
        result: TrackResult | None = None,
        calibration=None,  # noqa: ANN001
    ) -> None:
        self._table.setRowCount(0)
        if layer is None:
            return
        track = layer.result if result is None else result
        if track is None:
            return
        self.set_samples(series_for_result(track, info, calibration=calibration))

    def set_samples(self, samples) -> None:  # noqa: ANN001
        if samples:
            self.set_units(
                samples[0].position_unit,
                samples[0].speed_unit,
                samples[0].accel_unit,
            )
        else:
            self._table.setHorizontalHeaderLabels(self._header_labels())
        self._table.setRowCount(len(samples))
        warn = QColor("#f0c14b")
        for row, sample in enumerate(samples):
            values = [
                str(sample.frame + 1),
                f"{sample.time_s:.4f}",
                "" if sample.x is None else f"{sample.x:.2f}",
                "" if sample.y is None else f"{sample.y:.2f}",
                "" if sample.vx is None else f"{sample.vx:.2f}",
                "" if sample.vy is None else f"{sample.vy:.2f}",
                "" if sample.ax is None else f"{sample.ax:.2f}",
                "" if sample.ay is None else f"{sample.ay:.2f}",
                "是" if sample.visible else "否",
                f"{sample.confidence:.2f}",
            ]
            low = is_low_confidence(sample)
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, sample.frame)
                if low:
                    item.setForeground(warn)
                    item.setToolTip(quality_tooltip(sample))
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()
        for col in range(self._table.columnCount()):
            width = self._table.columnWidth(col)
            self._table.setColumnWidth(col, min(max(width + 8, 48), 128))

    def copy_selection(self) -> None:
        rows = sorted({index.row() for index in self._table.selectedIndexes()})
        if not rows:
            return
        headers = [
            self._table.horizontalHeaderItem(col).text()
            for col in range(self._table.columnCount())
        ]
        lines = ["\t".join(headers)]
        html_rows = [
            "<tr>" + "".join(f"<th>{escape(text)}</th>" for text in headers) + "</tr>"
        ]
        for row in rows:
            cells = []
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                cells.append("" if item is None else item.text())
            lines.append("\t".join(cells))
            html_rows.append(
                "<tr>" + "".join(f"<td>{escape(text)}</td>" for text in cells) + "</tr>"
            )
        payload = QMimeData()
        payload.setText("\n".join(lines))
        payload.setHtml("<table>" + "".join(html_rows) + "</table>")
        QGuiApplication.clipboard().setMimeData(payload)

    def _show_menu(self, pos) -> None:  # noqa: ANN001
        menu = QMenu(self._table)
        menu.addAction("复制", self.copy_selection)
        menu.addAction("全选", self._table.selectAll)
        menu.addAction("导出 CSV…", self.export_requested.emit)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def highlight_frame(self, frame: int) -> None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) == frame:
                self._table.selectRow(row)
                self._table.scrollToItem(item)
                return

    def _on_cell(self, row: int, _column: int) -> None:
        item = self._table.item(row, 0)
        if item is None:
            return
        self.frame_activated.emit(int(item.data(Qt.ItemDataRole.UserRole)))
