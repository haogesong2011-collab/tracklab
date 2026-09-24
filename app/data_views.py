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
    DEFAULT_VELOCITY_MODE,
    DEFAULT_VELOCITY_STEP,
    MAX_VELOCITY_STEP,
    VELOCITY_MODE_LOCAL,
    VELOCITY_MODE_TRACKER,
    KinematicSample,
    QuantityFit,
    contiguous_segments,
    fit_quantity,
    is_low_confidence,
)
from app.chart_ticks import nice_axis_ticks, nice_tick_interval

WARN = QColor("#f0c14b")
DOT_SIZE = 3.5
ACTIVE_DOT_SIZE = 6.0


def _configure_axis(
    axis: QValueAxis, lo: float, hi: float, target: int, *, expand: bool
) -> None:
    if expand:
        nmin, nmax, step, fmt = nice_axis_ticks(lo, hi, target)
        axis.setRange(nmin, nmax)
    else:
        step, fmt = nice_tick_interval(lo, hi, target)
        if hi <= lo:
            hi = lo + 1.0
        axis.setRange(lo, hi)
    axis.setLabelFormat(fmt)
    axis.setMinorTickCount(0)
    tick_type = getattr(QValueAxis, "TickType", None)
    if tick_type is not None:
        axis.setTickType(QValueAxis.TickType.TicksDynamic)
        axis.setTickInterval(step)
    else:
        span = axis.max() - axis.min()
        count = max(2, int(round(span / step)) + 1) if step else 5
        axis.setTickCount(min(count, 12))


VX_NAME = "vₓ"
VY_NAME = "vᵧ"


def speed_axis_label(speed_unit: str) -> str:
    if speed_unit == "px/s":
        return "图像投影速度 px/s"
    return speed_unit


def chart_series(position_unit: str = "px", speed_unit: str = "px/s") -> dict[str, tuple[str, str, str, str]]:
    speed = speed_axis_label(speed_unit)
    return {
        "x": ("x", "x", "#6cb6ff", f"x ({position_unit})"),
        "y": ("y", "y", "#7dce82", f"y ({position_unit})"),
        "vx": ("vx", VX_NAME, "#e07a5f", f"{VX_NAME} ({speed})"),
        "vy": ("vy", VY_NAME, "#d4a373", f"{VY_NAME} ({speed})"),
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
        self.setDragMode(QChartView.DragMode.NoDrag)
        self.setRubberBand(QChartView.RubberBand.NoRubberBand)
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
        self._dragging = False
        self._last_scrub_frame: int | None = None
        self._readout = QLabel(self.viewport())
        self._readout.setObjectName("chartScrubReadout")
        self._readout.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._readout.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._readout.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._readout.hide()

        chart = QChart()
        panel = QColor("#232323")
        chart.setBackgroundBrush(panel)
        chart.setPlotAreaBackgroundBrush(panel)
        chart.setPlotAreaBackgroundVisible(True)
        chart.setDropShadowEnabled(False)
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.legend().hide()
        chart.setBackgroundRoundness(0)
        chart.setMargins(QMargins(8, 12, 10, 10))
        layout = chart.layout()
        if layout is not None:
            layout.setContentsMargins(4, 4, 4, 4)
        self.setChart(chart)
        self._readout.setParent(self.viewport())
        self.viewport().installEventFilter(self)

        self._axis_time = QValueAxis()
        self._axis_value = QValueAxis()
        for axis, title in ((self._axis_time, "t (s)"), (self._axis_value, y_title)):
            axis.setLabelsColor(QColor("#9a9a9a"))
            axis.setTitleBrush(QColor("#bdbdbd"))
            axis.setGridLineColor(QColor("#2f2f2f"))
            axis.setLinePenColor(QColor("#4a4a4a"))
            axis.setTitleText(title)
            axis.setMinorTickCount(0)
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

        self._active_dot = QScatterSeries()
        self._active_dot.setName("")
        self._active_dot.setMarkerSize(ACTIVE_DOT_SIZE)
        color = QColor(self._specs[0][2])
        self._active_dot.setColor(color)
        self._active_dot.setBorderColor(color)
        chart.addSeries(self._active_dot)
        self._active_dot.attachAxis(self._axis_time)
        self._active_dot.attachAxis(self._axis_value)
        for marker in chart.legend().markers(self._active_dot):
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
            self._active_dot.clear()
            return

        for attr, name, color in self._specs:
            first = True
            qcolor = QColor(color)
            for segment in contiguous_segments(samples, attr):
                series = QLineSeries()
                series.setName(name if first else "")
                series.setPen(QPen(qcolor, 1.0))
                for sample in segment:
                    series.append(sample.time_s, float(getattr(sample, attr)))
                chart.addSeries(series)
                series.attachAxis(self._axis_time)
                series.attachAxis(self._axis_value)
                self._series.append(series)
                first = False
            dots = QScatterSeries()
            dots.setName("")
            dots.setMarkerSize(DOT_SIZE)
            dots.setColor(qcolor)
            dots.setBorderColor(qcolor)
            for sample in samples:
                value = getattr(sample, attr)
                if value is None:
                    continue
                dots.append(sample.time_s, float(value))
            if dots.count():
                chart.addSeries(dots)
                dots.attachAxis(self._axis_time)
                dots.attachAxis(self._axis_value)
                self._series.append(dots)
            warn = QScatterSeries()
            warn.setName("")
            warn.setMarkerSize(DOT_SIZE)
            warn.setColor(WARN)
            warn.setBorderColor(WARN)
            has_low = False
            for sample in samples:
                value = getattr(sample, attr)
                if value is None or not is_low_confidence(sample):
                    continue
                warn.append(sample.time_s, float(value))
                has_low = True
            if has_low:
                chart.addSeries(warn)
                warn.attachAxis(self._axis_time)
                warn.attachAxis(self._axis_value)
                self._series.append(warn)
            self._apply_model(chart, samples, attr, color)
            self._apply_fit(chart, samples, attr, color)

        self._fit_axes(samples)
        self._style_active_dot()
        self._raise_overlay_series()
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
        self._active_dot.clear()
        if not self._samples or not self._specs:
            return
        nearest = min(self._samples, key=lambda s: abs(s.time_s - time_s))
        attr = self._specs[0][0]
        value = getattr(nearest, attr)
        if value is None:
            return
        self._active_dot.append(nearest.time_s, float(value))

    def _style_active_dot(self) -> None:
        color = QColor(self._specs[0][2])
        self._active_dot.setColor(color)
        self._active_dot.setBorderColor(color)

    def _raise_overlay_series(self) -> None:
        chart = self.chart()
        for series in (self._active_dot, self._cursor):
            chart.removeSeries(series)
            chart.addSeries(series)
            series.attachAxis(self._axis_time)
            series.attachAxis(self._axis_value)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: ANN001
        if watched is self.viewport() and isinstance(event, QMouseEvent):
            et = event.type()
            if et == QEvent.Type.MouseButtonPress:
                return self._on_scrub_press(event)
            if et == QEvent.Type.MouseMove:
                return self._on_scrub_move(event)
            if et == QEvent.Type.MouseButtonRelease:
                return self._on_scrub_release(event)
            if et == QEvent.Type.MouseButtonDblClick:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._stop_scrub()
                    self.reset_zoom()
                    return True
        return super().eventFilter(watched, event)

    def _on_scrub_press(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton or not self._samples:
            return False
        sample = self._nearest_sample(event.position())
        if sample is None:
            return False
        self._dragging = True
        self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
        self._activate_sample(sample, event.position())
        return True

    def _on_scrub_move(self, event: QMouseEvent) -> bool:
        if not self._dragging or not self._samples:
            return False
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self._stop_scrub()
            return False
        sample = self._nearest_sample(event.position())
        if sample is not None:
            self._activate_sample(sample, event.position())
        return True

    def _on_scrub_release(self, event: QMouseEvent) -> bool:
        if not self._dragging or event.button() != Qt.MouseButton.LeftButton:
            return False
        sample = self._nearest_sample(event.position())
        if sample is not None:
            self._activate_sample(sample, event.position())
        self._stop_scrub()
        return True

    def _stop_scrub(self) -> None:
        self._dragging = False
        self._last_scrub_frame = None
        self.viewport().unsetCursor()
        self._hide_readout()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if getattr(self, "_axis_time", None) is None:
            return
        narrow = self.width() < 280
        short = self.height() < 140
        self._axis_value.setTitleVisible(not narrow)
        self._axis_time.setTitleVisible(not short)
        chart = self.chart()
        if chart is not None:
            chart.update()

    def _tick_targets(self) -> tuple[int, int]:
        time_n = 4 if self.width() < 240 else (5 if self.width() < 420 else 6)
        value_n = 3 if self.height() < 100 else (4 if self.height() < 160 else 5)
        return time_n, value_n

    def _nearest_sample(self, pos: QPointF) -> KinematicSample | None:
        if not self._samples:
            return None
        series = self._series[0] if self._series else self._cursor
        value = self.chart().mapToValue(pos, series)
        return min(self._samples, key=lambda s: abs(s.time_s - value.x()))

    def _activate_sample(self, sample: KinematicSample, pos: QPointF) -> None:
        self.highlight_time(sample.time_s)
        if sample.frame != self._last_scrub_frame:
            self._last_scrub_frame = sample.frame
            self.frame_activated.emit(sample.frame)
        self._show_readout(sample, pos)

    def _show_readout(self, sample: KinematicSample, pos: QPointF) -> None:
        attr, name, _color = self._specs[0]
        value = getattr(sample, attr)
        unit = sample.position_unit if attr in {"x", "y"} else sample.speed_unit
        if value is None:
            val_text = "—" if sample.visible else "—（不可见）"
        else:
            val_text = f"{value:.2f} {unit}"
            if not sample.visible:
                val_text += "（不可见）"
        self._readout.setText(
            f"帧 {sample.frame + 1}  ·  t = {sample.time_s:.2f} s\n{name} = {val_text}"
        )
        self._readout.adjustSize()
        host = self._readout.parentWidget() or self
        x = int(pos.x()) + 14
        y = int(pos.y()) - self._readout.height() - 10
        x = max(6, min(x, max(6, host.width() - self._readout.width() - 6)))
        y = max(6, min(y, max(6, host.height() - self._readout.height() - 6)))
        self._readout.move(x, y)
        self._readout.show()

    def _hide_readout(self) -> None:
        self._readout.hide()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: ANN001
        if not self._samples:
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            super().wheelEvent(event)
            return
        series = self._series[0] if self._series else self._cursor
        value = self.chart().mapToValue(event.position(), series)
        self.zoom_at(1.15 ** steps, value.x(), value.y())
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
        if getattr(self, "_axis_time", None) is None:
            return
        t0, t1 = self._fitted_t
        v0, v1 = self._fitted_v
        expand = self._zoom <= 1.0001 and self._center_t is None
        if expand:
            win_t0, win_t1 = t0, t1
            win_v0, win_v1 = v0, v1
        else:
            tspan = (t1 - t0) / self._zoom
            vspan = (v1 - v0) / self._zoom
            ct = self._center_t if self._center_t is not None else (t0 + t1) / 2.0
            cv = self._center_v if self._center_v is not None else (v0 + v1) / 2.0
            win_t0, win_t1 = ct - tspan / 2.0, ct + tspan / 2.0
            win_v0, win_v1 = cv - vspan / 2.0, cv + vspan / 2.0
        time_n, value_n = self._tick_targets()
        _configure_axis(self._axis_time, win_t0, win_t1, time_n, expand=expand)
        _configure_axis(self._axis_value, win_v0, win_v1, value_n, expand=expand)
        if self._highlight_t is not None:
            self.highlight_time(self._highlight_t)

    def _fit_axes(self, samples: list[KinematicSample]) -> None:
        times = [s.time_s for s in samples]
        vals: list[float] = []
        for sample in samples:
            for attr, _name, _color in self._specs:
                value = getattr(sample, attr)
                if value is not None:
                    vals.append(float(value))
                model_attr = {"vx": "model_vx", "vy": "model_vy"}.get(attr)
                if model_attr is None:
                    continue
                model_value = getattr(sample, model_attr)
                if model_value is not None:
                    vals.append(float(model_value))
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            t1 = t0 + 1.0
        self._fitted_t = (t0, t1)
        if self._fit_result is not None:
            vals.extend(self._fit_result.evaluate(t0 + (t1 - t0) * i / 20.0) for i in range(21))
        if vals:
            lo, hi = min(vals), max(vals)
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            self._fitted_v = (lo, hi)
        else:
            self._fitted_v = (-1.0, 1.0)
        self._apply_view()

    def _apply_model(
        self, chart: QChart, samples: list[KinematicSample], attr: str, color: str
    ) -> None:
        model_attr = {"vx": "model_vx", "vy": "model_vy"}.get(attr)
        if model_attr is None:
            return
        first = True
        for segment in contiguous_segments(samples, model_attr):
            series = QLineSeries()
            series.setName("斜抛模型" if first else "")
            pen = QPen(QColor(color).lighter(150))
            pen.setWidth(1.5)
            pen.setStyle(Qt.PenStyle.DotLine)
            series.setPen(pen)
            for sample in segment:
                series.append(sample.time_s, float(getattr(sample, model_attr)))
            chart.addSeries(series)
            series.attachAxis(self._axis_time)
            series.attachAxis(self._axis_value)
            self._series.append(series)
            for marker in chart.legend().markers(series):
                marker.setVisible(False)
            first = False

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
    velocity_mode_changed = Signal(str)

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
        hint = QLabel("按住拖动调进度，滚轮缩放，双击复位")
        hint.setObjectName("panelHint")
        mode_label = QLabel("速度")
        mode_label.setObjectName("panelHint")
        self._mode = QComboBox()
        self._mode.setObjectName("chartVelocityMode")
        self._mode.addItem("稳健拟合", VELOCITY_MODE_LOCAL)
        self._mode.addItem("Tracker 差分", VELOCITY_MODE_TRACKER)
        self._mode.setCurrentIndex(0 if DEFAULT_VELOCITY_MODE == VELOCITY_MODE_LOCAL else 1)
        self._mode.setToolTip(
            "稳健拟合：以当前时刻为中心，用真实 PTS 做局部二次多项式并解析求导。"
            "Tracker 差分：v(i)=(p[i+N]−p[i−N])/(t[i+N]−t[i−N])。"
        )
        self._mode.currentIndexChanged.connect(self._emit_velocity_mode)
        step_label = QLabel("窗口")
        step_label.setObjectName("panelHint")
        self._step = QSpinBox()
        self._step.setObjectName("chartStep")
        self._step.setRange(1, MAX_VELOCITY_STEP)
        self._step.setValue(DEFAULT_VELOCITY_STEP)
        self._step.setToolTip(
            "稳健拟合的时间半窗约为 0.10s × (N/3)，30/60/120 fps 对应同一物理时间。"
            "Tracker 差分模式则是 ±N 帧。"
        )
        self._step.valueChanged.connect(self.velocity_step_changed.emit)
        self._model_hint = QLabel("")
        self._model_hint.setObjectName("panelWarn")
        self._model_hint.setWordWrap(True)
        self._model_hint.hide()
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
        header.addWidget(mode_label)
        header.addWidget(self._mode)
        header.addWidget(step_label)
        header.addWidget(self._step)
        header.addWidget(fit_label)
        header.addWidget(self._fit)
        header.addWidget(reset)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addLayout(header)
        layout.addWidget(self._model_hint)

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

    @property
    def velocity_mode(self) -> str:
        return str(self._mode.currentData() or DEFAULT_VELOCITY_MODE)

    def _emit_velocity_mode(self) -> None:
        self.velocity_mode_changed.emit(self.velocity_mode)

    def set_model_warning(self, text: str) -> None:
        self._model_hint.setText(text)
        self._model_hint.setVisible(bool(text))

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
