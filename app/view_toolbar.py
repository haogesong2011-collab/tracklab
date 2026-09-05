"""Tracker-style secondary toolbar: current track, t/x/y, point stepping."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QWidget,
)

from ai.contracts import TrackLayer
from ai.kinematics import KinematicSample, is_low_confidence, quality_tooltip
from app.icons import icon_size, next_icon, prev_icon, toolbar_icon


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


class ViewToolbar(QWidget):
    track_selected = Signal(str)
    visibility_toggled = Signal(bool)
    position_edited = Signal(float, float)
    prev_point_requested = Signal()
    next_point_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("viewToolbar")
        self.setFixedHeight(36)
        self._updating = False
        self._x_field: QLineEdit | None = None
        self._y_field: QLineEdit | None = None
        self._position_unit = "px"

        self._track = QComboBox()
        self._track.setObjectName("viewTrackSelect")
        self._track.setMinimumWidth(140)
        self._track.setToolTip("当前轨迹")
        self._track.currentIndexChanged.connect(self._on_track_changed)

        self._visible = QToolButton()
        self._visible.setObjectName("viewToolButton")
        self._visible.setCheckable(True)
        self._visible.setChecked(True)
        self._visible.setIcon(toolbar_icon("view"))
        self._visible.setIconSize(icon_size())
        self._visible.setToolTip("显示 / 隐藏当前轨迹")
        self._visible.toggled.connect(self._on_visible_toggled)

        prev_btn = QPushButton()
        prev_btn.setObjectName("viewStepButton")
        prev_btn.setIcon(prev_icon())
        prev_btn.setIconSize(icon_size())
        prev_btn.setToolTip("上一有效轨迹点")
        prev_btn.clicked.connect(self.prev_point_requested.emit)

        next_btn = QPushButton()
        next_btn.setObjectName("viewStepButton")
        next_btn.setIcon(next_icon())
        next_btn.setIconSize(icon_size())
        next_btn.setToolTip("下一有效轨迹点")
        next_btn.clicked.connect(self.next_point_requested.emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 2, 12, 2)
        layout.setSpacing(6)
        layout.addWidget(self._track)
        layout.addWidget(self._visible)
        layout.addWidget(prev_btn)
        layout.addWidget(next_btn)
        layout.addSpacing(8)

        self._fields: dict[str, QLineEdit] = {}
        self._labels: dict[str, QLabel] = {}
        for key, label, editable, digits in (
            ("t", "t (s)", False, 4),
            ("x", "x (px)", True, 2),
            ("y", "y (px)", True, 2),
        ):
            caption = QLabel(label)
            caption.setObjectName("viewFieldLabel")
            field = QLineEdit()
            field.setObjectName("viewFieldEdit" if editable else "viewFieldRead")
            field.setReadOnly(not editable)
            field.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            field.setFixedWidth(68 if key != "t" else 74)
            field.setToolTip(label)
            if editable:
                validator = QDoubleValidator(-1e6, 1e6, 4, field)
                validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                field.setValidator(validator)
                field.editingFinished.connect(self._on_position_finished)
            field.setProperty("digits", digits)
            self._fields[key] = field
            self._labels[key] = caption
            layout.addWidget(caption)
            layout.addWidget(field)
        self._x_field = self._fields["x"]
        self._y_field = self._fields["y"]
        layout.addStretch()

    def set_units(self, position_unit: str) -> None:
        self._position_unit = position_unit
        self._labels["x"].setText(f"x ({position_unit})")
        self._labels["y"].setText(f"y ({position_unit})")
        self._fields["x"].setToolTip(f"x ({position_unit})")
        self._fields["y"].setToolTip(f"y ({position_unit})")

    def set_tracks(self, layers: list[TrackLayer], active_id: str | None) -> None:
        self._updating = True
        self._track.clear()
        active_index = 0
        for index, layer in enumerate(layers):
            self._track.addItem(layer.name, layer.track_id)
            self._track.setItemData(index, QColor(layer.color), Qt.ItemDataRole.ForegroundRole)
            if layer.track_id == active_id:
                active_index = index
        if layers:
            self._track.setCurrentIndex(active_index)
        layer = next((item for item in layers if item.track_id == active_id), None)
        self._visible.setChecked(True if layer is None else layer.visible)
        self._updating = False

    def set_sample(self, sample: KinematicSample | None) -> None:
        self._updating = True
        for field in self._fields.values():
            field.blockSignals(True)
        try:
            if sample is None:
                for field in self._fields.values():
                    field.setText("")
                self._set_low_confidence(False)
                return
            self.set_units(sample.position_unit)
            self._fields["t"].setText(_fmt(sample.time_s, 4))
            self._fields["x"].setText(_fmt(sample.x))
            self._fields["y"].setText(_fmt(sample.y))
            self._set_low_confidence(is_low_confidence(sample), quality_tooltip(sample))
        finally:
            for field in self._fields.values():
                field.blockSignals(False)
            self._updating = False

    def _set_low_confidence(self, enabled: bool, extra: str = "") -> None:
        color = "#f0c14b" if enabled else ""
        for key in ("t", "x", "y"):
            label_style = f"color: {color};" if color else ""
            field_style = f"color: {color};" if color else ""
            self._labels[key].setStyleSheet(label_style)
            self._fields[key].setStyleSheet(field_style)
            if enabled:
                tip = "低可信度" if not extra else f"低可信度 · {extra}"
                self._fields[key].setToolTip(tip)
            elif extra and key != "t":
                self._fields[key].setToolTip(extra)
            elif key == "t":
                self._fields[key].setToolTip("t (s)")
            else:
                self._fields[key].setToolTip(f"{key} ({self._position_unit})")

    def _on_track_changed(self, index: int) -> None:
        if self._updating or index < 0:
            return
        track_id = self._track.itemData(index)
        if track_id:
            self.track_selected.emit(str(track_id))

    def _on_visible_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self.visibility_toggled.emit(checked)

    def _on_position_finished(self) -> None:
        if self._updating or self._x_field is None or self._y_field is None:
            return
        try:
            x = float(self._x_field.text())
            y = float(self._y_field.text())
        except ValueError:
            return
        self.position_edited.emit(x, y)
