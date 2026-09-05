"""Guide for single-ruler, near-far, and planar calibration."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ai.calibration import CalibrationMode, DEFAULT_RULER_LENGTH_M


PLANE_HINTS = ("原点", "X 端", "对角点", "Y 端")


class CalibrationDialog(QWidget):
    mode_changed = Signal(str)
    apply_requested = Signal()
    redraw_requested = Signal()
    swap_requested = Signal()
    length_changed = Signal(float, float)
    plane_size_changed = Signal(float, float)
    charuco_requested = Signal()
    camera_calib_requested = Signal()
    audit_requested = Signal()
    experimental_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setObjectName("calibrationDialog")
        self.setWindowTitle("标定")
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setFixedWidth(300)

        self._uniform = QRadioButton("单尺（全图同一比例）")
        self._near_far = QRadioButton("双尺（近/远插值）")
        self._planar = QRadioButton("运动平面（推荐）")
        self._uniform.setChecked(True)
        self._uniform.toggled.connect(self._on_mode)
        self._near_far.toggled.connect(self._on_mode)
        self._planar.toggled.connect(self._on_mode)

        self._near_spin = self._length_spin("近尺 / 单尺")
        self._far_spin = self._length_spin("远尺")
        self._far_spin.setEnabled(False)
        self._width_spin = self._length_spin("平面宽度")
        self._height_spin = self._length_spin("平面高度")
        self._width_spin.setValue(1.0)
        self._height_spin.setValue(1.0)

        self._status = QLabel("拖动鼠标画出标定尺，默认 1.000 m。")
        self._status.setObjectName("panelHint")
        self._status.setWordWrap(True)

        self._charuco_btn = QPushButton("检测棋盘格")
        self._charuco_btn.setObjectName("panelButton")
        self._charuco_btn.clicked.connect(self.charuco_requested.emit)
        self._camera_btn = QPushButton("镜头标定…")
        self._camera_btn.setObjectName("panelButton")
        self._camera_btn.clicked.connect(self.camera_calib_requested.emit)
        self._audit_btn = QPushButton("AI 离面抽检")
        self._audit_btn.setObjectName("panelButton")
        self._audit_btn.clicked.connect(self.audit_requested.emit)
        self._audit_check = QCheckBox("实验性深度修正（不进入拟合）")
        self._audit_check.setObjectName("panelCheck")
        self._audit_check.setToolTip("仅多一列模型深度残差，不会改写几何坐标或物理拟合。")
        self._audit_check.toggled.connect(self.experimental_changed.emit)

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

        extra = QHBoxLayout()
        extra.setSpacing(6)
        extra.addWidget(self._charuco_btn)
        extra.addWidget(self._camera_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(self._planar)
        layout.addWidget(self._uniform)
        layout.addWidget(self._near_far)
        layout.addWidget(QLabel("实际长度 (m)"))
        layout.addWidget(self._near_spin)
        layout.addWidget(self._far_spin)
        layout.addWidget(self._width_spin)
        layout.addWidget(self._height_spin)
        layout.addLayout(extra)
        layout.addWidget(self._audit_btn)
        layout.addWidget(self._audit_check)
        layout.addWidget(self._status)
        layout.addLayout(buttons)

        self._near_spin.valueChanged.connect(self._emit_lengths)
        self._far_spin.valueChanged.connect(self._emit_lengths)
        self._width_spin.valueChanged.connect(self._emit_plane)
        self._height_spin.valueChanged.connect(self._emit_plane)
        self._sync_mode_widgets()

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
        if self._planar.isChecked():
            return CalibrationMode.PLANAR
        if self._near_far.isChecked():
            return CalibrationMode.NEAR_FAR
        return CalibrationMode.UNIFORM

    def set_mode(self, mode: CalibrationMode) -> None:
        for box in (self._uniform, self._near_far, self._planar):
            box.blockSignals(True)
        self._planar.setChecked(mode is CalibrationMode.PLANAR)
        self._near_far.setChecked(mode is CalibrationMode.NEAR_FAR)
        self._uniform.setChecked(mode is CalibrationMode.UNIFORM)
        for box in (self._uniform, self._near_far, self._planar):
            box.blockSignals(False)
        self._sync_mode_widgets()

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

    def plane_size(self) -> tuple[float, float]:
        return float(self._width_spin.value()), float(self._height_spin.value())

    def set_plane_size(self, width_m: float, height_m: float) -> None:
        self._width_spin.blockSignals(True)
        self._height_spin.blockSignals(True)
        self._width_spin.setValue(width_m)
        self._height_spin.setValue(height_m)
        self._width_spin.blockSignals(False)
        self._height_spin.blockSignals(False)

    def experimental_correction(self) -> bool:
        return self._audit_check.isChecked()

    def set_experimental_correction(self, enabled: bool) -> None:
        self._audit_check.blockSignals(True)
        self._audit_check.setChecked(enabled)
        self._audit_check.blockSignals(False)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setStyleSheet("color: #e07a5f;" if error else "")

    def plane_step_hint(self, count: int) -> str:
        if count >= 4:
            return "四个角点已就绪。确认宽高后点击应用，可拖动角点微调。"
        label = PLANE_HINTS[count]
        return f"请点击{label}（{count + 1}/4）。顺序：原点 → X 端 → 对角点 → Y 端。"

    def _on_mode(self, _checked: bool) -> None:
        self._sync_mode_widgets()
        self.mode_changed.emit(self.mode().value)

    def _sync_mode_widgets(self) -> None:
        planar = self.mode() is CalibrationMode.PLANAR
        near_far = self.mode() is CalibrationMode.NEAR_FAR
        self._far_spin.setEnabled(near_far)
        self._near_spin.setVisible(not planar)
        self._far_spin.setVisible(not planar)
        self._width_spin.setVisible(planar)
        self._height_spin.setVisible(planar)
        self._charuco_btn.setVisible(planar)
        self._camera_btn.setVisible(planar)
        self._audit_btn.setVisible(planar)
        self._audit_check.setVisible(planar)

    def _emit_lengths(self, _value: float) -> None:
        near_m, far_m = self.lengths()
        self.length_changed.emit(near_m, far_m)

    def _emit_plane(self, _value: float) -> None:
        width_m, height_m = self.plane_size()
        self.plane_size_changed.emit(width_m, height_m)


class CameraCalibDialog(QWidget):
    capture_requested = Signal()
    compute_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setObjectName("calibrationDialog")
        self.setWindowTitle("镜头标定")
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setFixedWidth(280)
        self._status = QLabel("在不同角度显示 ChArUco 棋盘，采集至少 3 张后计算内参。")
        self._status.setObjectName("panelHint")
        self._status.setWordWrap(True)
        capture = QPushButton("采集当前帧")
        capture.setObjectName("panelButtonPrimary")
        capture.clicked.connect(self.capture_requested.emit)
        compute = QPushButton("计算内参")
        compute.setObjectName("panelButton")
        compute.clicked.connect(self.compute_requested.emit)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.addWidget(self._status)
        layout.addWidget(capture)
        layout.addWidget(compute)

    def set_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setStyleSheet("color: #e07a5f;" if error else "")
