"""Selectable x / y / vₓ / vᵧ plots for the active track."""

from __future__ import annotations

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtCore import QEvent, QMargins, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ai.kinematics import (
    DEFAULT_VELOCITY_STEP,
    MAX_VELOCITY_STEP,
    KinematicSample,
    QuantityFit,
    contiguous_segments,
    fit_quantity,
    is_low_confidence,
)
from app.gestures import gesture_from_native, gesture_from_wheel, pointer_pans, shift_view_center

WARN = QColor("#f0c14b")


VX_NAME = "vₓ"
VY_NAME = "vᵧ"


def chart_series(position_unit: str = "px", speed_unit: str = "px/s") -> dict[str, tuple[str, str, str, str]]:
    return {
        "x": ("x", "x", "#6cb6ff", f"x ({position_unit})"),
        "y": ("y", "y", "#7dce82", f"y ({position_unit})"),
        "vx": ("vx", VX_NAME, "#e07a5f", f"{VX_NAME} ({speed_unit})"),
        "vy": ("vy", VY_NAME, "#d4a373", f"{VY_NAME} ({speed_unit})"),
    }


CHART_SERIES = chart_series()


class TrackChartView(QChartView):
    frame_activated = Signal(int)

    def __init__(
        self,
        specs: list[tuple[str, str, str]],
        y_title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("trackChartView")
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: #232323; border: none;")
        self.setMinimumSize(150, 80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._specs = specs
        self._samples: list[KinematicSample] = []
        self._series: list[QLineSeries] = []
        self._highlight_t: float | None = None
        self._zoom = 1.0
        self._center_t: float | None = None
        self._center_v: float | None = None
        self._fitted_t = (0.0, 1.0)
        self._fitted_v = (-1.0, 1.0)
        self._fit_degree = 0
        self._fit_series: QLineSeries | None = None
        self._fit_result: QuantityFit | None = None
        self._panning: QPointF | None = None

        chart = QChart()
        panel = QColor("#232323")
        chart.setBackgroundBrush(panel)
        chart.setPlotAreaBackgroundBrush(panel)
        chart.setPlotAreaBackgroundVisible(True)
        chart.setDropShadowEnabled(False)
        chart.legend().hide()
        chart.setBackgroundRoundness(0)
        chart.setMargins(QMargins(8, 12, 10, 10))
        layout = chart.layout()
        if layout is not None:
            layout.setContentsMargins(4, 4, 4, 4)
        self.setChart(chart)
        self.viewport().installEventFilter(self)

        self._axis_time = QValueAxis()
        self._axis_value = QValueAxis()
        for axis, title in ((self._axis_time, "t (s)"), (self._axis_value, y_title)):
            axis.setLabelsColor(QColor("#9a9a9a"))
            axis.setTitleBrush(QColor("#bdbdbd"))
            axis.setGridLineColor(QColor("#2f2f2f"))
            axis.setLinePenColor(QColor("#4a4a4a"))
            axis.setTitleText(title)
            axis.setLabelFormat("%g")
            axis.setTickCount(5)
        chart.addAxis(self._axis_time, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self._axis_value, Qt.AlignmentFlag.AlignLeft)

        self._cursor = QLineSeries()
        self._cursor.setName("")
        pen = QPen(QColor("#f0c14b"))
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._cursor.setPen(pen)
        chart.addSeries(self._cursor)
        self._cursor.attachAxis(self._axis_time)
        self._cursor.attachAxis(self._axis_value)
        for marker in chart.legend().markers(self._cursor):
            marker.setVisible(False)

    def set_spec(self, attr: str, name: str, color: str, y_title: str) -> None:
        self._specs = [(attr, name, color)]
        self._axis_value.setTitleText(y_title)
        self._zoom = 1.0
        self._center_t = None
        self._center_v = None
        self.set_samples(self._samples)

    def set_fit_degree(self, degree: int) -> None:
        self._fit_degree = max(0, min(2, int(degree)))
        self.set_samples(self._samples)

    @property
    def fit_result(self) -> QuantityFit | None:
        return self._fit_result

    def set_samples(self, samples: list[KinematicSample]) -> None:
        chart = self.chart()
        for series in self._series:
            chart.removeSeries(series)
        self._series = []
        if self._fit_series is not None:
            chart.removeSeries(self._fit_series)
            self._fit_series = None
        self._fit_result = None
        self._samples = samples
        if not samples:
            self._fitted_t = (0.0, 1.0)
            self._fitted_v = (-1.0, 1.0)
            self._zoom = 1.0
            self._center_t = None
            self._center_v = None
            self._apply_view()
            self._cursor.clear()
            return

        for attr, name, color in self._specs:
            first = True
            for segment in contiguous_segments(samples, attr):
                series = QLineSeries()
                series.setName(name if first else "")
                series.setPen(QPen(QColor(color), 1.8))
                for sample in segment:
                    series.append(sample.time_s, float(getattr(sample, attr)))
                chart.addSeries(series)
                series.attachAxis(self._axis_time)
                series.attachAxis(self._axis_value)
                self._series.append(series)
                first = False
            scatter = QScatterSeries()
            scatter.setName("")
            scatter.setMarkerSize(7)
            scatter.setColor(WARN)
            scatter.setBorderColor(WARN)
            has_low = False
            for sample in samples:
                value = getattr(sample, attr)
                if value is None or not is_low_confidence(sample):
                    continue
                scatter.append(sample.time_s, float(value))
                has_low = True
            if has_low:
                chart.addSeries(scatter)
                scatter.attachAxis(self._axis_time)
                scatter.attachAxis(self._axis_value)
                self._series.append(scatter)  # type: ignore[arg-type]
            self._apply_fit(chart, samples, attr, color)

        self._fit_axes(samples)
        for marker in chart.legend().markers():
            if not marker.series().name():
                marker.setVisible(False)
        if self._highlight_t is not None:
            self.highlight_time(self._highlight_t)
        else:
            self.highlight_time(samples[0].time_s)

    def highlight_frame(self, frame: int) -> None:
        for sample in self._samples:
            if sample.frame == frame:
                self.highlight_time(sample.time_s)
                return

    def highlight_time(self, time_s: float) -> None:
        self._highlight_t = time_s
        self._cursor.replace(
            [
                QPointF(time_s, self._axis_value.min()),
                QPointF(time_s, self._axis_value.max()),
            ]
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: ANN001
        if self._handle_pointer_press(event):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: ANN001
        if self._handle_pointer_move(event):
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: ANN001
        if self._handle_pointer_release(event):
            return
        super().mouseReleaseEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: ANN001
        if watched is self.viewport():
            if event.type() == QEvent.Type.NativeGesture and self._apply_native_gesture(event):
                return True
            if isinstance(event, QMouseEvent):
                et = event.type()
                if et == QEvent.Type.MouseButtonPress and self._handle_pointer_press(event):
                    return True
                if et == QEvent.Type.MouseMove and self._handle_pointer_move(event):
                    return True
                if et == QEvent.Type.MouseButtonRelease and self._handle_pointer_release(event):
                    return True
        return super().eventFilter(watched, event)

    def event(self, event: QEvent) -> bool:
        if self._apply_native_gesture(event):
            return True
        return super().event(event)

    def _handle_pointer_press(self, event: QMouseEvent) -> bool:
        if not self._samples:
            return False
        if pointer_pans(event.button(), event.modifiers(), surface="chart"):
            self._panning = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return True
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        series = self._series[0] if self._series else self._cursor
        value = self.chart().mapToValue(event.position(), series)
        nearest = min(self._samples, key=lambda s: abs(s.time_s - value.x()))
        self.frame_activated.emit(nearest.frame)
        return True

    def _handle_pointer_move(self, event: QMouseEvent) -> bool:
        if self._panning is None:
            return False
        if not (
            event.buttons()
            & (Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton)
        ):
            self._stop_pan()
            return False
        delta = event.position() - self._panning
        self._panning = event.position()
        self.pan_by_pixels(delta.x(), delta.y())
        return True

    def _handle_pointer_release(self, event: QMouseEvent) -> bool:
        if self._panning is None:
            return False
        if event.button() not in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            return False
        self._stop_pan()
        return True

    def _stop_pan(self) -> None:
        self._panning = None
        self.viewport().unsetCursor()

    def pan_by_pixels(self, dx: float, dy: float) -> None:
        chart = self.chart()
        plot = chart.plotArea() if chart is not None else None
        width = 0.0 if plot is None else plot.width()
        height = 0.0 if plot is None else plot.height()
        t0, t1 = self._axis_time.min(), self._axis_time.max()
        v0, v1 = self._axis_value.min(), self._axis_value.max()
        center_t, center_v = self._view_center()
        self._center_t, self._center_v = shift_view_center(
            center_t, center_v, dx, dy, width, height, t0, t1, v0, v1
        )
        self._apply_view()

    def _apply_native_gesture(self, event: QEvent) -> bool:
        gesture = gesture_from_native(event)
        if gesture is None or not self._samples:
            return False
        if gesture.kind == "pan":
            self.pan_by_pixels(gesture.dx, gesture.dy)
            return True
        if gesture.kind == "zoom":
            series = self._series[0] if self._series else self._cursor
            pos = event.position() if hasattr(event, "position") else QPointF(self.width() / 2, self.height() / 2)
            value = self.chart().mapToValue(pos, series)
            factor = gesture.factor if gesture.factor is not None else 1.15 ** gesture.steps
            self.zoom_at(factor, value.x(), value.y())
            return True
        if gesture.kind == "reset":
            self.reset_zoom()
            return True
        return False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        narrow = self.width() < 280
        short = self.height() < 140
        self._axis_value.setTitleVisible(not narrow)
        self._axis_time.setTitleVisible(not short)
        self._axis_value.setTickCount(4 if short else 5)
        self._axis_time.setTickCount(4 if narrow else 5)
        chart = self.chart()
        if chart is not None:
            chart.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: ANN001
        if not self._samples:
            super().wheelEvent(event)
            return
        gesture = gesture_from_wheel(event)
        if gesture.kind == "pan":
            self.pan_by_pixels(gesture.dx, gesture.dy)
            event.accept()
            return
        if gesture.kind != "zoom":
            super().wheelEvent(event)
            return
        series = self._series[0] if self._series else self._cursor
        value = self.chart().mapToValue(event.position(), series)
        factor = gesture.factor if gesture.factor is not None else 1.15 ** gesture.steps
        self.zoom_at(factor, value.x(), value.y())
        event.accept()

    def zoom_in(self) -> None:
        t, v = self._view_center()
        self.zoom_at(1.25, t, v)

    def zoom_out(self) -> None:
        t, v = self._view_center()
        self.zoom_at(1.0 / 1.25, t, v)

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self._center_t = None
        self._center_v = None
        self._apply_view()

    def zoom_at(self, factor: float, t: float, v: float) -> None:
        old = self._zoom
        new = max(0.8, min(40.0, old * factor))
        if abs(new - old) < 1e-6:
            return
        t0, t1 = self._axis_time.min(), self._axis_time.max()
        v0, v1 = self._axis_value.min(), self._axis_value.max()
        rel_t = 0.5 if t1 <= t0 else (t - t0) / (t1 - t0)
        rel_v = 0.5 if v1 <= v0 else (v - v0) / (v1 - v0)
        self._zoom = new
        ft0, ft1 = self._fitted_t
        fv0, fv1 = self._fitted_v
        tspan = (ft1 - ft0) / new
        vspan = (fv1 - fv0) / new
        self._center_t = t - (rel_t - 0.5) * tspan
        self._center_v = v - (rel_v - 0.5) * vspan
        self._apply_view()

    def _view_center(self) -> tuple[float, float]:
        return (
            (self._axis_time.min() + self._axis_time.max()) / 2.0,
            (self._axis_value.min() + self._axis_value.max()) / 2.0,
        )

    def _apply_view(self) -> None:
        t0, t1 = self._fitted_t
        v0, v1 = self._fitted_v
        if self._zoom <= 1.0001 and self._center_t is None:
            self._axis_time.setRange(t0, t1)
            self._axis_value.setRange(v0, v1)
        else:
            tspan = (t1 - t0) / self._zoom
            vspan = (v1 - v0) / self._zoom
            ct = self._center_t if self._center_t is not None else (t0 + t1) / 2.0
            cv = self._center_v if self._center_v is not None else (v0 + v1) / 2.0
            self._axis_time.setRange(ct - tspan / 2.0, ct + tspan / 2.0)
            self._axis_value.setRange(cv - vspan / 2.0, cv + vspan / 2.0)
        if self._highlight_t is not None:
            self.highlight_time(self._highlight_t)

    def _fit_axes(self, samples: list[KinematicSample]) -> None:
        times = [s.time_s for s in samples]
        vals = [
            float(getattr(sample, attr))
            for sample in samples
            for attr, _name, _color in self._specs
            if getattr(sample, attr) is not None
        ]
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            t1 = t0 + 1.0
        pad = (t1 - t0) * 0.04
        self._fitted_t = (t0 - pad, t1 + pad)
        if self._fit_result is not None:
            vals.extend(self._fit_result.evaluate(t0 + (t1 - t0) * i / 20.0) for i in range(21))
        if vals:
            lo, hi = min(vals), max(vals)
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            span = hi - lo
            self._fitted_v = (lo - span * 0.08, hi + span * 0.08)
        else:
            self._fitted_v = (-1.0, 1.0)
        self._apply_view()

    def _apply_fit(
        self, chart: QChart, samples: list[KinematicSample], attr: str, color: str
    ) -> None:
        if self._fit_degree < 1:
            return
        fitted = fit_quantity(samples, attr, self._fit_degree)
        self._fit_result = fitted
        if fitted is None:
            return
        times = [sample.time_s for sample in samples]
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            t1 = t0 + 1.0
        series = QLineSeries()
        series.setName("拟合")
        pen = QPen(QColor(color).lighter(140))
        pen.setWidth(1.6)
        pen.setStyle(Qt.PenStyle.DashLine)
        series.setPen(pen)
        steps = 80
        for i in range(steps + 1):
            time_s = t0 + (t1 - t0) * i / steps
            series.append(time_s, fitted.evaluate(time_s))
        chart.addSeries(series)
        series.attachAxis(self._axis_time)
        series.attachAxis(self._axis_value)
        self._fit_series = series
        for marker in chart.legend().markers(series):
            marker.setVisible(False)


class TrackChartPanel(QWidget):
    frame_activated = Signal(int)
    velocity_step_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trackChartPanel")
        self._frame = 0
        self._position_unit = "px"
        self._speed_unit = "px/s"
        self._selects: list[QComboBox] = []
        self._charts: list[TrackChartView] = []
        self._fit_labels: list[QLabel] = []

        header = QHBoxLayout()
        header.setContentsMargins(6, 2, 6, 0)
        header.setSpacing(8)
        hint = QLabel("单击跳到该帧，双指或右键拖动平移，滚轮或捏合缩放，双击复位")
        hint.setObjectName("panelHint")
        step_label = QLabel("步长")
        step_label.setObjectName("panelHint")
        self._step = QSpinBox()
        self._step.setObjectName("chartStep")
        self._step.setRange(1, MAX_VELOCITY_STEP)
        self._step.setValue(DEFAULT_VELOCITY_STEP)
        self._step.setToolTip(
            "Tracker 速度步长 N：v(i)=(p[i+N]−p[i−N])/(t[i+N]−t[i−N])。增大可压跟踪抖动。"
        )
        self._step.valueChanged.connect(self.velocity_step_changed.emit)
        fit_label = QLabel("拟合")
        fit_label.setObjectName("panelHint")
        self._fit = QComboBox()
        self._fit.setObjectName("chartFit")
        self._fit.setToolTip("按当前分图数据做最小二乘拟合，并画虚线")
        self._fit.addItem("关闭", 0)
        self._fit.addItem("线性", 1)
        self._fit.addItem("二次", 2)
        self._fit.currentIndexChanged.connect(self._apply_fit_mode)
        reset = QPushButton("复位")
        reset.setObjectName("panelButton")
        reset.setToolTip("恢复分图默认显示范围")
        reset.clicked.connect(self.reset_zoom)
        header.addWidget(hint)
        header.addStretch()
        header.addWidget(step_label)
        header.addWidget(self._step)
        header.addWidget(fit_label)
        header.addWidget(self._fit)
        header.addWidget(reset)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addLayout(header)

        series = chart_series()
        for default in ("x", "y"):
            combo = QComboBox()
            combo.setObjectName("chartQuantity")
            combo.setToolTip("选择这条分图显示的物理量")
            for key, (_attr, _name, _color, title_text) in series.items():
                combo.addItem(title_text, key)
            combo.setCurrentIndex(list(series).index(default))
            attr, name, color, axis = series[default]
            chart = TrackChartView([(attr, name, color)], axis)
            chart.frame_activated.connect(self.frame_activated.emit)
            combo.currentIndexChanged.connect(
                lambda _i, view=chart, box=combo: self._apply_quantity(view, box)
            )
            minus = QPushButton("－")
            minus.setObjectName("panelButton")
            minus.setFixedWidth(28)
            minus.setToolTip("缩小")
            minus.clicked.connect(chart.zoom_out)
            plus = QPushButton("＋")
            plus.setObjectName("panelButton")
            plus.setFixedWidth(28)
            plus.setToolTip("放大")
            plus.clicked.connect(chart.zoom_in)
            self._selects.append(combo)
            self._charts.append(chart)

            row = QWidget()
            row.setObjectName("chartQuantityRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(6, 0, 4, 0)
            row_layout.setSpacing(6)
            equation = QLabel("")
            equation.setObjectName("chartFitLabel")
            equation.setWordWrap(True)
            equation.hide()
            row_layout.addWidget(combo, stretch=0)
            row_layout.addWidget(equation, stretch=1)
            row_layout.addWidget(minus)
            row_layout.addWidget(plus)
            self._fit_labels.append(equation)
            layout.addWidget(row)
            layout.addWidget(chart, stretch=1)

    def _series(self) -> dict[str, tuple[str, str, str, str]]:
        return chart_series(self._position_unit, self._speed_unit)

    def _apply_quantity(self, chart: TrackChartView, combo: QComboBox) -> None:
        key = str(combo.currentData() or "x")
        attr, name, color, axis = self._series().get(key, self._series()["x"])
        chart.set_spec(attr, name, color, axis)
        chart.highlight_frame(self._frame)
        self._refresh_fit_labels()

    def _apply_fit_mode(self) -> None:
        degree = int(self._fit.currentData() or 0)
        for chart in self._charts:
            chart.set_fit_degree(degree)
        self._refresh_fit_labels()

    def _refresh_fit_labels(self) -> None:
        for chart, combo, label in zip(self._charts, self._selects, self._fit_labels):
            fitted = chart.fit_result
            if fitted is None:
                label.hide()
                label.setText("")
                continue
            key = str(combo.currentData() or "x")
            name = self._series().get(key, self._series()["x"])[1]
            label.setText(fitted.equation(name))
            label.setToolTip(label.text())
            label.show()

    @property
    def velocity_step(self) -> int:
        return int(self._step.value())

    def set_units(self, position_unit: str, speed_unit: str) -> None:
        if position_unit == self._position_unit and speed_unit == self._speed_unit:
            return
        keys = [str(box.currentData() or "x") for box in self._selects]
        self._position_unit = position_unit
        self._speed_unit = speed_unit
        series = self._series()
        for box, key, chart in zip(self._selects, keys, self._charts):
            if key == "v":
                key = "vx"
            box.blockSignals(True)
            box.clear()
            for item_key, (_attr, _name, _color, title_text) in series.items():
                box.addItem(title_text, item_key)
            box.setCurrentIndex(list(series).index(key) if key in series else 0)
            box.blockSignals(False)
            self._apply_quantity(chart, box)

    def set_samples(self, samples: list[KinematicSample]) -> None:
        if samples:
            self.set_units(samples[0].position_unit, samples[0].speed_unit)
        for chart in self._charts:
            chart.set_samples(samples)
        self._refresh_fit_labels()
        self.highlight_frame(self._frame)

    def highlight_frame(self, frame: int) -> None:
        self._frame = frame
        for chart in self._charts:
            chart.highlight_frame(frame)

    def reset_zoom(self) -> None:
        for chart in self._charts:
            chart.reset_zoom()
