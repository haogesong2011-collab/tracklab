"""Selectable x / y / vₓ / vᵧ plots for the active track."""

from __future__ import annotations

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
import math

from PySide6.QtCore import QEvent, QMargins, QObject, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.theme import color as theme_color
from app.theme import qcolor as theme_qcolor
from ai.kinematics import (
    AUTO_VELOCITY_STEP,
    FIT_AUTO,
    DEFAULT_VELOCITY_STEP,
    FIT_ACCEL,
    FIT_DRAG,
    FIT_MODELS,
    FIT_OFF,
    FIT_PROJECTILE,
    FIT_UNIFORM,
    MAX_VELOCITY_STEP,
    VELOCITY_MODE_LOCAL,
    KinematicSample,
    QuantityFit,
    contiguous_segments,
    is_low_confidence,
    accel_unit_for,
    format_uncertainty,
    law_caption,
    law_readout,
    law_runs,
)
from app.chart_ticks import nice_axis_ticks, nice_tick_interval
from app.marks import PENDING_YELLOW, TRUSTED_GREEN

WARN = QColor(PENDING_YELLOW)
DOT_SIZE = 4.5
PENDING_DOT_SIZE = 7.5
ACTIVE_DOT_SIZE = 6.0
CHART_PULSE_TICK_MS = 80
CHART_PULSE_PERIOD_S = 1.6
DERIV_LOCAL = "local"
DERIV_MODEL = "model"


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
AX_NAME = "aₓ"
AY_NAME = "aᵧ"


def speed_axis_label(speed_unit: str) -> str:
    if speed_unit == "px/s":
        return "图像投影速度 px/s"
    return speed_unit


def quantity_family(attr: str) -> str:
    if attr in {"x", "y"}:
        return "position"
    if attr in {"vx", "vy"}:
        return "velocity"
    if attr in {"ax", "ay"}:
        return "accel"
    return attr


def chart_series(
    position_unit: str = "px",
    speed_unit: str = "px/s",
    accel_unit: str = "px/s²",
) -> dict[str, tuple[str, str, str, str]]:
    speed = speed_axis_label(speed_unit)
    return {
        "x": ("x", "x", "#6cb6ff", f"x ({position_unit})"),
        "y": ("y", "y", "#7dce82", f"y ({position_unit})"),
        "vx": ("vx", VX_NAME, "#e07a5f", f"{VX_NAME} ({speed})"),
        "vy": ("vy", VY_NAME, "#d4a373", f"{VY_NAME} ({speed})"),
        "ax": ("ax", AX_NAME, "#c77dff", f"{AX_NAME} ({accel_unit})"),
        "ay": ("ay", AY_NAME, "#9b8cff", f"{AY_NAME} ({accel_unit})"),
    }


CHART_SERIES = chart_series()


def window_caption(samples: list[KinematicSample]) -> str:
    """'±0.44 / 0.60 s' for the longest measured run; empty when unknown."""
    best: tuple[int, float, float] | None = None
    counts: dict[tuple[float, float], int] = {}
    for sample in samples:
        if sample.window_v_s is None or sample.window_a_s is None:
            continue
        key = (round(sample.window_v_s, 3), round(sample.window_a_s, 3))
        counts[key] = counts.get(key, 0) + 1
    for (wv, wa), count in counts.items():
        if best is None or count > best[0]:
            best = (count, wv, wa)
    if best is None:
        return ""
    return f"±{best[1]:.2f} / {best[2]:.2f} s"


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
        self.setStyleSheet(f"background: {theme_color('panel')}; border: none;")
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
        self._data_t = (0.0, 1.0)
        self._data_v = (-1.0, 1.0)
        self._shared_span: float | None = None
        self._manual_value: tuple[float, float] | None = None
        self._manual_time: tuple[float, float] | None = None
        self._fit_model = FIT_OFF
        self._law_series: list[QLineSeries] = []
        self._law_runs: list[tuple[float, float, QuantityFit]] = []
        self._pending_series: list[QScatterSeries] = []
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(CHART_PULSE_TICK_MS)
        self._pulse_timer.timeout.connect(self._pulse_pending)
        self._dragging = False
        self._last_scrub_frame: int | None = None
        self._readout = QLabel(self.viewport())
        self._readout.setObjectName("chartScrubReadout")
        self._readout.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._readout.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._readout.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._readout.hide()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_axis_menu)

        chart = QChart()
        panel = theme_qcolor("panel")
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
            axis.setLabelsColor(theme_qcolor("chart_label"))
            axis.setTitleBrush(theme_qcolor("chart_title"))
            axis.setGridLineColor(theme_qcolor("chart_grid"))
            axis.setLinePenColor(theme_qcolor("chart_axis"))
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

        # Where the mouse is on the fitted line (hover to read values).
        self._hover_dot = QScatterSeries()
        self._hover_dot.setName("")
        self._hover_dot.setMarkerSize(ACTIVE_DOT_SIZE + 2)
        self._hover_dot.setColor(QColor(0, 0, 0, 0))
        self._hover_dot.setBorderColor(QColor("#ffffff"))
        chart.addSeries(self._hover_dot)
        self._hover_dot.attachAxis(self._axis_time)
        self._hover_dot.attachAxis(self._axis_value)
        for marker in chart.legend().markers(self._hover_dot):
            marker.setVisible(False)
        self.viewport().setMouseTracking(True)

    def set_spec(self, attr: str, name: str, color: str, y_title: str) -> None:
        self._specs = [(attr, name, color)]
        self._axis_value.setTitleText(y_title)
        self._zoom = 1.0
        self._center_t = None
        self._center_v = None
        self.set_samples(self._samples)

    def set_fit_model(self, model: str) -> None:
        self._fit_model = model if model in FIT_MODELS else FIT_OFF
        self.set_samples(self._samples)

    def fit_caption(self, name: str) -> str:
        if not self._law_runs:
            if self._fit_model == FIT_AUTO and self._samples:
                return f"{name}：整段不像匀速或匀加速，保留逐帧数值"
            return ""
        t0, _t1, primary = max(self._law_runs, key=lambda item: item[1] - item[0])
        sample = next((s for s in self._samples if s.x is not None), None) or (self._samples[0] if self._samples else None)
        if sample is None:
            return primary.equation(name)
        text = law_caption(
            primary,
            self._specs[0][0],
            name,
            t0=t0,
            position_unit=sample.position_unit,
            speed_unit=sample.speed_unit,
            accel_unit=sample.accel_unit,
        )
        if len(self._law_runs) > 1:
            text += f"  （共 {len(self._law_runs)} 段，目标丢失处断开）"
        return text

    def set_samples(self, samples: list[KinematicSample]) -> None:
        chart = self.chart()
        for series in self._series:
            chart.removeSeries(series)
        self._series = []
        for series in self._law_series:
            chart.removeSeries(series)
        self._law_series = []
        self._law_runs = []
        self._pending_series = []
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
            trusted = [
                (sample.time_s, float(getattr(sample, attr)))
                for sample in samples
                if getattr(sample, attr) is not None and not is_low_confidence(sample)
            ]
            self._add_dots(chart, trusted, QColor(TRUSTED_GREEN), DOT_SIZE, border=None)
            doubtful = [
                (sample.time_s, float(getattr(sample, attr)))
                for sample in samples
                if getattr(sample, attr) is not None and is_low_confidence(sample)
            ]
            waiting = [
                (sample.time_s, float(sample.review_value(attr)))
                for sample in samples
                if sample.pending and getattr(sample, attr) is None and sample.review_value(attr) is not None
            ]
            # Measured but doubtful (low quality, off-plane, outside the ruler): still
            # green because it is a measurement, with a yellow rim.
            self._add_dots(chart, doubtful, QColor(TRUSTED_GREEN), DOT_SIZE + 1.5, border=WARN)
            pending_series = self._add_dots(chart, waiting, WARN, PENDING_DOT_SIZE, border=QColor("#3A2E00"))
            if pending_series is not None:
                self._pending_series.append(pending_series)
            self._apply_model(chart, samples, attr, color)
            self._apply_fit(chart, samples, attr, color)

        self._fit_axes(samples)
        self._style_active_dot()
        self._raise_overlay_series()
        self._sync_pending_pulse()
        for marker in chart.legend().markers():
            if not marker.series().name():
                marker.setVisible(False)
        if self._highlight_t is not None:
            self.highlight_time(self._highlight_t)
        else:
            self.highlight_time(samples[0].time_s)

    def _add_dots(
        self,
        chart: QChart,
        points: list[tuple[float, float]],
        color: QColor,
        size: float,
        *,
        border: QColor | None,
    ) -> QScatterSeries | None:
        if not points:
            return None
        dots = QScatterSeries()
        dots.setName("")
        dots.setMarkerSize(size)
        dots.setColor(color)
        dots.setBorderColor(border if border is not None else color)
        for time_s, value in points:
            dots.append(time_s, value)
        chart.addSeries(dots)
        dots.attachAxis(self._axis_time)
        dots.attachAxis(self._axis_value)
        self._series.append(dots)
        return dots

    def _sync_pending_pulse(self) -> None:
        if self._pending_series and self.isVisible():
            if not self._pulse_timer.isActive():
                self._pulse_timer.start()
        elif self._pulse_timer.isActive():
            self._pulse_timer.stop()

    def _pulse_pending(self) -> None:
        """Yellow points waiting for review breathe (slow brightness and size swing)."""
        if not self._pending_series:
            self._pulse_timer.stop()
            return
        self._pulse_phase = (self._pulse_phase + CHART_PULSE_TICK_MS / 1000.0 / CHART_PULSE_PERIOD_S) % 1.0
        level = 0.5 - 0.5 * math.cos(2.0 * math.pi * self._pulse_phase)
        color = QColor(WARN)
        color.setAlpha(int(110 + 145 * level))
        for series in self._pending_series:
            series.setColor(color)
            series.setMarkerSize(PENDING_DOT_SIZE - 1.0 + 2.5 * level)

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        self._sync_pending_pulse()

    def hideEvent(self, event) -> None:  # noqa: ANN001
        super().hideEvent(event)
        self._pulse_timer.stop()

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
        if watched is self.viewport() and event.type() == QEvent.Type.Leave and not self._dragging:
            self._hide_readout()
        if watched is self.viewport() and isinstance(event, QMouseEvent):
            et = event.type()
            if et == QEvent.Type.MouseButtonPress:
                return self._on_scrub_press(event)
            if et == QEvent.Type.MouseMove:
                if not self._dragging and not (event.buttons() & Qt.MouseButton.LeftButton):
                    return self._on_hover(event)
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
        if attr in {"ax", "ay"}:
            unit = sample.accel_unit
        pending_value = sample.review_value(attr) if sample.pending and value is None else None
        if pending_value is not None:
            val_text = f"{pending_value:.2f} {unit}（待确认，不计入测量）"
        elif value is None:
            val_text = "—" if sample.visible else "—（不可见）"
        else:
            val_text = f"{value:.2f}"
            sigma = getattr(sample, f"sigma_{attr}", None) if attr in {"vx", "vy", "ax", "ay"} else None
            if sigma is not None:
                val_text += f" ± {format_uncertainty(sigma)}"
            val_text += f" {unit}"
            if not sample.visible:
                val_text += "（不可见）"
            if sample.kinematics_source != "local" and attr in {"vx", "vy", "ax", "ay"}:
                val_text += "（由函数求得）"
        lines = [f"帧 {sample.frame + 1}  ·  t = {sample.time_s:.2f} s", f"{name} = {val_text}"]
        fitted = self._fit_readout(sample)
        if fitted:
            lines.append(fitted)
        local_value = getattr(sample, f"local_{attr}", None) if attr in {"vx", "vy", "ax", "ay"} else None
        if sample.kinematics_source != "local" and local_value is not None:
            lines.append(f"逐帧估计 {name} = {local_value:.2f}".replace("= -", "= −"))
        self._readout.setText("\n".join(lines))
        self._readout.adjustSize()
        host = self._readout.parentWidget() or self
        x = int(pos.x()) + 14
        y = int(pos.y()) - self._readout.height() - 10
        x = max(6, min(x, max(6, host.width() - self._readout.width() - 6)))
        y = max(6, min(y, max(6, host.height() - self._readout.height() - 6)))
        self._readout.move(x, y)
        self._readout.show()

    def _law_at(self, time_s: float):  # noqa: ANN202
        for t0, t1, law in self._law_runs:
            if t0 - 1e-9 <= time_s <= t1 + 1e-9:
                return law
        return None

    def _fit_readout(self, sample: KinematicSample) -> str:
        """'拟合 x = 12.34 m · vₓ = −7.80 m/s' at this frame's time."""
        law = self._law_at(sample.time_s)
        if law is None:
            return ""
        units = {
            "x": sample.position_unit,
            "y": sample.position_unit,
            "vₓ": sample.speed_unit,
            "vᵧ": sample.speed_unit,
            "v": sample.speed_unit,
            "aₓ": sample.accel_unit,
            "aᵧ": sample.accel_unit,
        }
        parts = [
            f"{symbol} = {value:.2f} {units.get(symbol, '')}".rstrip()
            for symbol, value in law_readout(law, self._specs[0][0], sample.time_s)
        ]
        return ("拟合 " + " · ".join(parts)).replace("= -", "= −") if parts else ""

    def _on_hover(self, event: QMouseEvent) -> bool:
        """Mouse over the chart without pressing: show the numbers, keep the frame."""
        if not self._samples:
            return False
        sample = self._nearest_sample(event.position())
        if sample is None:
            return False
        self._show_readout(sample, event.position())
        law = self._law_at(sample.time_s)
        self._hover_dot.clear()
        if law is not None:
            try:
                self._hover_dot.append(sample.time_s, float(law.evaluate(sample.time_s)))
            except (KeyError, ValueError):
                pass
        return False

    def _hide_readout(self) -> None:
        self._readout.hide()
        if hasattr(self, "_hover_dot"):
            self._hover_dot.clear()

    def apply_theme(self) -> None:
        panel = theme_qcolor("panel")
        chart = self.chart()
        chart.setBackgroundBrush(panel)
        chart.setPlotAreaBackgroundBrush(panel)
        self.setStyleSheet(f"background: {theme_color('panel')}; border: none;")
        for axis in (self._axis_time, self._axis_value):
            axis.setLabelsColor(theme_qcolor("chart_label"))
            axis.setTitleBrush(theme_qcolor("chart_title"))
            axis.setGridLineColor(theme_qcolor("chart_grid"))
            axis.setLinePenColor(theme_qcolor("chart_axis"))
        self.update()

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
        self._manual_value = None
        self._manual_time = None
        self._compose_ranges()
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
        value_expand = expand and self._shared_span is None and self._manual_value is None
        _configure_axis(self._axis_time, win_t0, win_t1, time_n, expand=expand)
        _configure_axis(self._axis_value, win_v0, win_v1, value_n, expand=value_expand)
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
                elif sample.pending and sample.review_value(attr) is not None:
                    vals.append(float(sample.review_value(attr)))
                model_attr = {"vx": "model_vx", "vy": "model_vy"}.get(attr)
                if model_attr is None:
                    continue
                model_value = getattr(sample, model_attr)
                if model_value is not None:
                    vals.append(float(model_value))
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            t1 = t0 + 1.0
        self._data_t = (t0, t1)
        for run_t0, run_t1, law in self._law_runs:
            vals.extend(
                law.evaluate(run_t0 + (run_t1 - run_t0) * i / 20.0) for i in range(21)
            )
        if vals:
            lo, hi = min(vals), max(vals)
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            self._data_v = (lo, hi)
        else:
            self._data_v = (-1.0, 1.0)
        self._compose_ranges()
        self._apply_view()

    def data_value_span(self) -> float:
        lo, hi = self._data_v
        return max(hi - lo, 0.0)

    def quantity_key(self) -> str:
        return self._specs[0][0] if self._specs else "x"

    def set_shared_span(self, span: float | None) -> None:
        self._shared_span = None if span is None or span <= 0 else float(span)
        self._compose_ranges()
        self._apply_view()

    def set_manual_range(
        self,
        value: tuple[float, float] | None,
        time: tuple[float, float] | None,
    ) -> None:
        self._manual_value = value
        self._manual_time = time
        self._zoom = 1.0
        self._center_t = None
        self._center_v = None
        self._compose_ranges()
        self._apply_view()

    def manual_ranges(
        self,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        return self._manual_value, self._manual_time

    def _compose_ranges(self) -> None:
        lo, hi = self._data_v
        if self._manual_value is not None:
            vlo, vhi = self._manual_value
            if vhi <= vlo:
                vhi = vlo + 1.0
            self._fitted_v = (vlo, vhi)
        elif self._shared_span is not None:
            mid = (lo + hi) / 2.0
            half = self._shared_span / 2.0
            self._fitted_v = (mid - half, mid + half)
        else:
            self._fitted_v = (lo, hi)
        if self._manual_time is not None:
            t0, t1 = self._manual_time
            if t1 <= t0:
                t1 = t0 + 1.0
            self._fitted_t = (t0, t1)
        else:
            self._fitted_t = self._data_t

    def _show_axis_menu(self, pos) -> None:  # noqa: ANN001
        menu = QMenu(self)
        menu.addAction("设置坐标轴…", self._edit_axis_range)
        menu.addAction("恢复自动范围", self.reset_zoom)
        menu.exec(self.mapToGlobal(pos))

    def _edit_axis_range(self) -> None:
        dialog = AxisRangeDialog(
            self,
            value_range=self._data_v,
            time_range=self._data_t,
            manual_value=self._manual_value,
            manual_time=self._manual_time,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.set_manual_range(dialog.value_range(), dialog.time_range())

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
        runs = law_runs(samples, self._fit_model, attr)
        self._law_runs = runs
        first = True
        for t0, t1, law in runs:
            series = QLineSeries()
            series.setName("拟合" if first else "")
            pen = QPen(QColor(color).lighter(125))
            pen.setWidthF(2.4)
            pen.setStyle(Qt.PenStyle.SolidLine)
            series.setPen(pen)
            steps = 48
            for i in range(steps + 1):
                time_s = t0 + (t1 - t0) * i / steps
                series.append(time_s, law.evaluate(time_s))
            chart.addSeries(series)
            series.attachAxis(self._axis_time)
            series.attachAxis(self._axis_value)
            self._law_series.append(series)
            for marker in chart.legend().markers(series):
                marker.setVisible(False)
            first = False


class AxisRangeDialog(QDialog):
    """Numbers-style min/max for one chart. Auto leaves that axis to the panel."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        value_range: tuple[float, float],
        time_range: tuple[float, float],
        manual_value: tuple[float, float] | None,
        manual_time: tuple[float, float] | None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置坐标轴")
        form = QFormLayout(self)
        self._auto_y = QCheckBox("自动")
        self._auto_y.setChecked(manual_value is None)
        y_lo, y_hi = manual_value or value_range
        self._ymin = self._spin(y_lo)
        self._ymax = self._spin(y_hi)
        self._auto_x = QCheckBox("自动")
        self._auto_x.setChecked(manual_time is None)
        x_lo, x_hi = manual_time or time_range
        self._xmin = self._spin(x_lo)
        self._xmax = self._spin(x_hi)
        form.addRow("纵轴最小值", self._ymin)
        form.addRow("纵轴最大值", self._ymax)
        form.addRow("纵轴", self._auto_y)
        form.addRow("横轴最小值", self._xmin)
        form.addRow("横轴最大值", self._xmax)
        form.addRow("横轴", self._auto_x)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._auto_y.toggled.connect(self._sync_enabled)
        self._auto_x.toggled.connect(self._sync_enabled)
        self._sync_enabled()

    @staticmethod
    def _spin(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1.0e12, 1.0e12)
        box.setDecimals(4)
        box.setValue(float(value))
        return box

    def _sync_enabled(self) -> None:
        auto_y = self._auto_y.isChecked()
        self._ymin.setEnabled(not auto_y)
        self._ymax.setEnabled(not auto_y)
        auto_x = self._auto_x.isChecked()
        self._xmin.setEnabled(not auto_x)
        self._xmax.setEnabled(not auto_x)

    def value_range(self) -> tuple[float, float] | None:
        if self._auto_y.isChecked():
            return None
        return (float(self._ymin.value()), float(self._ymax.value()))

    def time_range(self) -> tuple[float, float] | None:
        if self._auto_x.isChecked():
            return None
        return (float(self._xmin.value()), float(self._xmax.value()))


class TrackChartPanel(QWidget):
    frame_activated = Signal(int)
    velocity_step_changed = Signal(int)
    fit_model_changed = Signal(str)
    derivative_source_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trackChartPanel")
        self._frame = 0
        self._position_unit = "px"
        self._speed_unit = "px/s"
        self._accel_unit = "px/s²"
        self._selects: list[QComboBox] = []
        self._charts: list[TrackChartView] = []
        self._fit_labels: list[QLabel] = []

        header = QHBoxLayout()
        header.setContentsMargins(6, 2, 6, 0)
        header.setSpacing(8)
        hint = QLabel("按住拖动调进度，滚轮缩放，双击复位")
        hint.setObjectName("panelHint")
        step_label = QLabel("窗口")
        step_label.setObjectName("panelHint")
        self._step = QSpinBox()
        self._step.setObjectName("chartStep")
        self._step.setRange(AUTO_VELOCITY_STEP, MAX_VELOCITY_STEP)
        self._step.setSpecialValueText("自动")
        self._step.setValue(DEFAULT_VELOCITY_STEP)
        self._step.setToolTip(
            "求速度、加速度时用多长一段时间来平滑。\n"
            "自动：按数据挑选——加速度取「匀加速模型仍然拟合得住」的最宽窗口，"
            "速度略窄；遇到摆动、碰撞会自动变窄。\n"
            "1–10：手动，1 最灵（噪声大），10 最平（细节少）。"
        )
        self._step.valueChanged.connect(self.velocity_step_changed.emit)
        self._window_info = QLabel("")
        self._window_info.setObjectName("panelHint")
        self._window_info.setToolTip("当前实际使用的半窗口：速度 / 加速度（秒）")
        self._model_hint = QLabel("")
        self._model_hint.setObjectName("panelWarn")
        self._model_hint.setWordWrap(True)
        self._model_hint.hide()
        fit_label = QLabel("函数")
        fit_label.setObjectName("panelHint")
        self._fit = QComboBox()
        self._fit.setObjectName("chartFit")
        self._fit.setToolTip("在测量点上画参考曲线，不改原始数据。")
        self._fit.addItem("关闭", FIT_OFF)
        self._fit.addItem("自动", FIT_AUTO)
        self._fit.setItemData(
            1,
            "每个方向各自选直线（匀速）或抛物线（匀加速），用整段数据拟合，给出总体速度或加速度。"
            "摆动、碰撞这种两者都不像的运动，保留逐帧数值。",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._fit.addItem("匀速", FIT_UNIFORM)
        self._fit.addItem("匀加速", FIT_ACCEL)
        self._fit.addItem("斜抛", FIT_PROJECTILE)
        self._fit.addItem("斜抛+空气阻力", FIT_DRAG)
        self._fit.setItemData(
            self._fit.count() - 1,
            "a = g − k|v|v：整段飞行一起拟合，分开给出 g 和阻力系数 k（都带误差）。"
            "轻的物体（泡沫镖、羽毛球）上升时 |aᵧ| 明显大于 g、下落时小于 g，就是阻力造成的。",
            Qt.ItemDataRole.ToolTipRole,
        )
        self._fit.currentIndexChanged.connect(self._apply_fit_mode)
        deriv_label = QLabel("v/a")
        deriv_label.setObjectName("panelHint")
        self._deriv = QComboBox()
        self._deriv.setObjectName("chartDerivSource")
        self._deriv.addItem("逐帧", DERIV_LOCAL)
        self._deriv.addItem("按函数", DERIV_MODEL)
        self._deriv.setToolTip(
            "逐帧：每一帧用前后一小段位置求导，能看出细节，也会带上跟踪误差。\n"
            "按函数：用「函数」里选的运动规律整段拟合后求导（位置仍是测量值）。"
            "细长、模糊、会翻转的物体，跟踪点会沿物体滑几个像素，逐帧加速度会大幅起伏，这时用按函数。\n"
            "选了函数会自动切到按函数；逐帧的结果不画在图上，鼠标停在点上能看到，导出的 CSV 里也有。"
        )
        self._deriv.setEnabled(False)
        self._deriv.currentIndexChanged.connect(lambda _i: self._on_deriv_changed())
        self._same_scale = QCheckBox("同尺度")
        self._same_scale.setObjectName("chartSameScale")
        self._same_scale.setChecked(True)
        self._same_scale.setToolTip("单位相同时，两张图纵轴一格代表的数值一样。")
        self._same_scale.toggled.connect(lambda _checked: self._apply_shared_scale())
        reset = QPushButton("复位")
        reset.setObjectName("panelButton")
        reset.setToolTip("恢复默认范围")
        reset.clicked.connect(self.reset_zoom)
        header.addWidget(hint)
        header.addStretch()
        header.addWidget(self._same_scale)
        header.addWidget(step_label)
        header.addWidget(self._step)
        header.addWidget(self._window_info)
        header.addWidget(fit_label)
        header.addWidget(self._fit)
        header.addWidget(deriv_label)
        header.addWidget(self._deriv)
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
        # Default: fit the whole run automatically (straight line or parabola
        # per axis) and read the overall velocity / acceleration off it.
        self._fit.setCurrentIndex(self._fit.findData(FIT_AUTO))

    def _series(self) -> dict[str, tuple[str, str, str, str]]:
        return chart_series(self._position_unit, self._speed_unit, self._accel_unit)

    def _apply_quantity(self, chart: TrackChartView, combo: QComboBox) -> None:
        key = str(combo.currentData() or "x")
        attr, name, color, axis = self._series().get(key, self._series()["x"])
        chart.set_spec(attr, name, color, axis)
        chart.highlight_frame(self._frame)
        self._apply_shared_scale()
        self._refresh_fit_labels()

    def _apply_shared_scale(self) -> None:
        charts = self._charts
        if not charts:
            return
        if not self._same_scale.isChecked():
            for chart in charts:
                chart.set_shared_span(None)
            return
        families = {quantity_family(chart.quantity_key()) for chart in charts}
        if len(families) != 1:
            for chart in charts:
                chart.set_shared_span(None)
            return
        span = max(chart.data_value_span() for chart in charts)
        for chart in charts:
            chart.set_shared_span(span)

    @property
    def fit_model(self) -> str:
        return str(self._fit.currentData() or FIT_OFF)

    def _apply_fit_mode(self) -> None:
        model = self.fit_model
        # Picking a motion law means analysing under it: v / a follow the law
        # unless the user switches back to 逐帧.
        self._deriv.blockSignals(True)
        self._deriv.setEnabled(model != FIT_OFF)
        self._deriv.setCurrentIndex(1 if model != FIT_OFF else 0)
        self._deriv.blockSignals(False)
        for chart in self._charts:
            chart.set_fit_model(model)
        self._apply_shared_scale()
        self._refresh_fit_labels()
        self.fit_model_changed.emit(model)

    def _on_deriv_changed(self) -> None:
        self.derivative_source_changed.emit()

    @property
    def derivative_model(self) -> str:
        """The law v / a are taken from, or FIT_OFF for per-frame estimates."""
        model = self.fit_model
        if model == FIT_OFF or self._deriv.currentData() != DERIV_MODEL:
            return FIT_OFF
        return model

    def _refresh_fit_labels(self) -> None:
        for chart, combo, label in zip(self._charts, self._selects, self._fit_labels):
            key = str(combo.currentData() or "x")
            name = self._series().get(key, self._series()["x"])[1]
            text = chart.fit_caption(name)
            if not text:
                label.hide()
                label.setText("")
                continue
            label.setText(text)
            label.setToolTip(text)
            label.show()

    def apply_theme(self) -> None:
        for chart in self._charts:
            chart.apply_theme()

    @property
    def velocity_step(self) -> int:
        return int(self._step.value())

    @property
    def velocity_mode(self) -> str:
        return VELOCITY_MODE_LOCAL

    def set_model_warning(self, text: str) -> None:
        self._model_hint.setText(text)
        self._model_hint.setVisible(bool(text))

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
        keys = [str(box.currentData() or "x") for box in self._selects]
        self._position_unit = position_unit
        self._speed_unit = speed_unit
        self._accel_unit = accel
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
            self.set_units(
                samples[0].position_unit,
                samples[0].speed_unit,
                samples[0].accel_unit,
            )
        source = next((s.kinematics_source for s in samples if s.x is not None), "local")
        if source != "local":
            name = self._fit.currentText() or "函数"
            self._window_info.setText(f"v、a 由「{name}」求得")
        else:
            self._window_info.setText(window_caption(samples))
        for chart in self._charts:
            chart.set_samples(samples)
        self._apply_shared_scale()
        self._refresh_fit_labels()
        self.highlight_frame(self._frame)

    def highlight_frame(self, frame: int) -> None:
        self._frame = frame
        for chart in self._charts:
            chart.highlight_frame(frame)

    def reset_zoom(self) -> None:
        for chart in self._charts:
            chart.reset_zoom()
