"""Compact guide for single-ruler / near-far calibration."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ai.calibration import CalibrationMode, DEFAULT_RULER_LENGTH_M


class CalibrationDialog(QWidget):
    mode_changed = Signal(str)
    apply_requested = Signal()
    redraw_requested = Signal()
    swap_requested = Signal()
    length_changed = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setObjectName("calibrationDialog")
        self.setWindowTitle("标定尺")
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setFixedWidth(280)

        self._uniform = QRadioButton("单尺（全图同一比例）")
        self._near_far = QRadioButton("双尺（近/远插值）")
        self._uniform.setChecked(True)
        self._uniform.toggled.connect(self._on_mode)

        self._near_spin = self._length_spin("近尺 / 单尺")
        self._far_spin = self._length_spin("远尺")
        self._far_spin.setEnabled(False)

        self._status = QLabel("拖动鼠标画出标定尺，默认 1.000 m。")
        self._status.setObjectName("panelHint")
        self._status.setWordWrap(True)

        swap = QPushButton("交换近/远")
        swap.setObjectName("panelButton")
        swap.clicked.connect(self.swap_requested.emit)
        apply_btn = QPushButton("应用")
        apply_btn.setObjectName("panelButtonPrimary")
        apply_btn.clicked.connect(self.apply_requested.emit)
        redraw = QPushButton("重画")
        redraw.setObjectName("panelButton")
        redraw.clicked.connect(self.redraw_requested.emit)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(swap)
        buttons.addStretch()
        buttons.addWidget(redraw)
        buttons.addWidget(apply_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._uniform)
        layout.addWidget(self._near_far)
        layout.addWidget(QLabel("实际长度 (m)"))
        layout.addWidget(self._near_spin)
        layout.addWidget(self._far_spin)
        layout.addWidget(self._status)
        layout.addLayout(buttons)

        self._near_spin.valueChanged.connect(self._emit_lengths)
        self._far_spin.valueChanged.connect(self._emit_lengths)

    def _length_spin(self, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setObjectName("calLengthSpin")
        spin.setDecimals(3)
        spin.setRange(0.001, 1000.0)
        spin.setSingleStep(0.001)
        spin.setValue(DEFAULT_RULER_LENGTH_M)
        spin.setSuffix(" m")
        spin.setToolTip(tooltip)
        return spin

    def mode(self) -> CalibrationMode:
        if self._near_far.isChecked():
            return CalibrationMode.NEAR_FAR
        return CalibrationMode.UNIFORM

    def set_mode(self, mode: CalibrationMode) -> None:
        self._uniform.blockSignals(True)
        self._near_far.blockSignals(True)
        self._near_far.setChecked(mode is CalibrationMode.NEAR_FAR)
        self._uniform.setChecked(mode is not CalibrationMode.NEAR_FAR)
        self._uniform.blockSignals(False)
        self._near_far.blockSignals(False)
        self._far_spin.setEnabled(mode is CalibrationMode.NEAR_FAR)

    def lengths(self) -> tuple[float, float]:
        return float(self._near_spin.value()), float(self._far_spin.value())

    def set_lengths(self, near_m: float, far_m: float | None = None) -> None:
        self._near_spin.blockSignals(True)
        self._far_spin.blockSignals(True)
        self._near_spin.setValue(near_m)
        if far_m is not None:
            self._far_spin.setValue(far_m)
        self._near_spin.blockSignals(False)
        self._far_spin.blockSignals(False)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setStyleSheet("color: #e07a5f;" if error else "")

    def _on_mode(self, _checked: bool) -> None:
        mode = self.mode()
        self._far_spin.setEnabled(mode is CalibrationMode.NEAR_FAR)
        self.mode_changed.emit(mode.value)

    def _emit_lengths(self, _value: float) -> None:
        near_m, far_m = self.lengths()
        self.length_changed.emit(near_m, far_m)
