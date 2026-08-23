"""Nested workspace: video in the center, chart/data as right-only docks."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRect, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QCloseEvent, QGuiApplication, QShowEvent
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

CHART_DOCK = "chartDock"
DATA_DOCK = "dataDock"
RIGHT_RATIO = 0.31
CHART_RATIO = 0.60
# Old workspace/state blobs may include nested/tabified AI docks and
# SIGSEGV inside Qt restoreState. Bump this key after dock-structure changes.
STATE_KEY = "workspace/state_v2"


def _on_screen(rect: QRect) -> bool:
    if not rect.isValid() or rect.width() < 40 or rect.height() < 40:
        return False
    for screen in QGuiApplication.screens():
        if screen.availableGeometry().intersects(rect):
            return True
    return False


class _HomeDock(QDockWidget):
    """Float freely, but closing only hides so the window menu can restore it."""

    def __init__(self, title: str, name: str, parent: QMainWindow) -> None:
        super().__init__(title, parent)
        self.setObjectName(name)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.setMinimumSize(QSize(220, 140))
        self._last_float = QSize(380, 280)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: ANN001
        event.ignore()
        self.hide()

    def remember_float_size(self) -> None:
        if self.isFloating():
            self._last_float = self.size()

    def restore_float_size(self) -> None:
        if self.isFloating():
            self.resize(self._last_float)


class DockWorkspace(QMainWindow):
    chart_visibility_changed = Signal(bool)
    data_visibility_changed = Signal(bool)

    def __init__(
        self,
        video: QWidget,
        chart: QWidget,
        data: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("dockWorkspace")
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setCentralWidget(video)
        self._normalizing = False
        self._restored = False
        self._shown = False

        self.chart_dock = _HomeDock("分图", CHART_DOCK, self)
        self.data_dock = _HomeDock("数据表", DATA_DOCK, self)
        self.chart_dock.setWidget(chart)
        self.data_dock.setWidget(data)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.chart_dock)
        self.splitDockWidget(
            self.chart_dock, self.data_dock, Qt.Orientation.Vertical
        )

        for dock in (self.chart_dock, self.data_dock):
            dock.topLevelChanged.connect(self._on_top_level)
            dock.dockLocationChanged.connect(self._on_location)
            dock.visibilityChanged.connect(self._on_visibility)

    def createPopupMenu(self) -> None:  # type: ignore[override]
        return None

    @property
    def chart_visible(self) -> bool:
        return not self.chart_dock.isHidden()

    @property
    def data_visible(self) -> bool:
        return not self.data_dock.isHidden()

    def set_chart_visible(self, visible: bool) -> None:
        self.chart_dock.setVisible(visible)
        if visible and not self.chart_dock.isFloating():
            self._normalize()

    def set_data_visible(self, visible: bool) -> None:
        self.data_dock.setVisible(visible)
        if visible and not self.data_dock.isFloating():
            self._normalize()

    def dock_chart(self) -> None:
        self._dock_home(self.chart_dock)

    def dock_data(self) -> None:
        self._dock_home(self.data_dock)

    def dock_all(self) -> None:
        self.chart_dock.show()
        self.data_dock.show()
        self._dock_home(self.chart_dock)
        self._dock_home(self.data_dock)

    def restore_default(self) -> None:
        self.dock_all()
        self._apply_default_sizes()

    def save_prefs(self, settings: QSettings) -> None:
        self.chart_dock.remember_float_size()
        self.data_dock.remember_float_size()
        settings.setValue(STATE_KEY, self.saveState())
        settings.setValue("workspace/chart_float", self.chart_dock.isFloating())
        settings.setValue("workspace/data_float", self.data_dock.isFloating())
        settings.setValue("workspace/chart_visible", self.chart_visible)
        settings.setValue("workspace/data_visible", self.data_visible)
        settings.setValue("workspace/chart_geo", self.chart_dock.saveGeometry())
        settings.setValue("workspace/data_geo", self.data_dock.saveGeometry())

    def restore_prefs(self, settings: QSettings) -> None:
        state = settings.value(STATE_KEY)
        if isinstance(state, QByteArray) and not state.isEmpty():
            self.restoreState(state)
        for dock, prefix in (
            (self.chart_dock, "workspace/chart"),
            (self.data_dock, "workspace/data"),
        ):
            geo = settings.value(f"{prefix}_geo")
            if isinstance(geo, QByteArray) and not geo.isEmpty():
                dock.restoreGeometry(geo)
                if dock.isFloating() and not _on_screen(dock.frameGeometry()):
                    dock.setFloating(False)
            visible = settings.value(f"{prefix}_visible", True)
            if isinstance(visible, str):
                visible = visible.lower() not in {"false", "0"}
            dock.setVisible(bool(visible))
        self._restored = True
        for dock in (self.chart_dock, self.data_dock):
            if dock.isFloating() and not _on_screen(dock.frameGeometry()):
                dock.setFloating(False)
        if not self._layout_ok():
            self.restore_default()
            return
        chart_docked = self.chart_visible and not self.chart_dock.isFloating()
        data_docked = self.data_visible and not self.data_dock.isFloating()
        if chart_docked or data_docked:
            self._normalize()

    def _layout_ok(self) -> bool:
        if self.tabifiedDockWidgets(self.chart_dock) or self.tabifiedDockWidgets(self.data_dock):
            return False
        for dock in (self.chart_dock, self.data_dock):
            if dock.isHidden() or dock.isFloating():
                continue
            if self.dockWidgetArea(dock) != Qt.DockWidgetArea.RightDockWidgetArea:
                return False
        return True

    def _dock_home(self, dock: _HomeDock) -> None:
        dock.remember_float_size()
        dock.setFloating(False)
        dock.show()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._normalize()
        self._apply_default_sizes()

    def _on_top_level(self, floating: bool) -> None:
        dock = self.sender()
        if not isinstance(dock, _HomeDock):
            return
        if floating:
            dock.restore_float_size()
            if not _on_screen(dock.frameGeometry()):
                dock.move(80, 80)
        else:
            self._normalize()

    def _on_location(self, area: Qt.DockWidgetArea) -> None:
        if self._normalizing:
            return
        dock = self.sender()
        if not isinstance(dock, QDockWidget) or dock.isFloating():
            return
        if area != Qt.DockWidgetArea.RightDockWidgetArea:
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._normalize()

    def _on_visibility(self, visible: bool) -> None:
        dock = self.sender()
        if dock is self.chart_dock:
            self.chart_visibility_changed.emit(visible)
        elif dock is self.data_dock:
            self.data_visibility_changed.emit(visible)
        if visible and not (isinstance(dock, QDockWidget) and dock.isFloating()):
            self._normalize()

    def _normalize(self) -> None:
        if self._normalizing:
            return
        self._normalizing = True
        try:
            chart_docked = self.chart_visible and not self.chart_dock.isFloating()
            data_docked = self.data_visible and not self.data_dock.isFloating()
            if chart_docked:
                self.addDockWidget(
                    Qt.DockWidgetArea.RightDockWidgetArea, self.chart_dock
                )
            if data_docked:
                self.addDockWidget(
                    Qt.DockWidgetArea.RightDockWidgetArea, self.data_dock
                )
            if chart_docked and data_docked:
                self.splitDockWidget(
                    self.chart_dock, self.data_dock, Qt.Orientation.Vertical
                )
        finally:
            self._normalizing = False

    def _apply_default_sizes(self) -> None:
        chart_docked = self.chart_visible and not self.chart_dock.isFloating()
        data_docked = self.data_visible and not self.data_dock.isFloating()
        if not chart_docked and not data_docked:
            return
        width = max(240, min(int(self.width() * RIGHT_RATIO), 460))
        docks = [d for d, ok in ((self.chart_dock, chart_docked), (self.data_dock, data_docked)) if ok]
        if docks:
            self.resizeDocks(docks, [width] * len(docks), Qt.Orientation.Horizontal)
        if chart_docked and data_docked:
            height = max(self.height(), 1)
            self.resizeDocks(
                [self.chart_dock, self.data_dock],
                [int(height * CHART_RATIO), int(height * (1.0 - CHART_RATIO))],
                Qt.Orientation.Vertical,
            )

    def showEvent(self, event: QShowEvent) -> None:  # noqa: ANN001
        super().showEvent(event)
        if self._shown:
            return
        self._shown = True
        if not self._restored:
            self._apply_default_sizes()
