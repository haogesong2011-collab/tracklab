"""System light and dark chrome. Data colors stay put."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.data_views import TrackChartView, chart_series  # noqa: E402
from app.icons import toolbar_icon  # noqa: E402
from app.main_window import MainWindow  # noqa: E402
from app.theme import preference, set_preference, stylesheet, use  # noqa: E402


class ThemeSheetTests(unittest.TestCase):
    def test_light_window_is_not_the_dark_background(self) -> None:
        light = stylesheet("light")
        dark = stylesheet("dark")
        self.assertIn("#f4f5f7", light)
        self.assertNotIn("#1f1f1f", light)
        self.assertNotIn("{{", light)
        self.assertIn("#1f1f1f", dark)
        self.assertNotIn("{{", dark)
        self.assertIn("#f0c14b", light)
        self.assertIn("#f0c14b", dark)
        bar = light[light.index("QScrollBar:horizontal") : light.index("QScrollBar:horizontal") + 220]
        self.assertIn("#f7f8fa", bar)
        self.assertNotIn("#1c1c1c", bar)

    def test_manual_appearance_ignores_the_name_until_cleared(self) -> None:
        try:
            self.assertEqual(set_preference("light"), "light")
            self.assertEqual(preference(), "light")
            self.assertIn("#f4f5f7", stylesheet())
            self.assertEqual(set_preference("dark"), "dark")
            self.assertIn("#1f1f1f", stylesheet())
        finally:
            set_preference("system")

    def test_series_colors_are_not_theme_tokens(self) -> None:
        series = chart_series()
        self.assertEqual(series["x"][2], "#6cb6ff")
        self.assertEqual(series["y"][2], "#7dce82")

    def test_toolbar_icons_use_one_ink(self) -> None:
        app = QApplication.instance() or QApplication(sys.argv)
        del app
        use("dark")
        try:
            for name in ("open", "save", "video", "ruler", "axis", "track", "ai", "view", "zoom", "cache"):
                icon = toolbar_icon(name)
                self.assertFalse(icon.isNull())
                self.assertFalse(icon.pixmap(18, 18).isNull())
        finally:
            use("dark")


class ThemePaintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication(sys.argv)

    def tearDown(self) -> None:
        use("dark")

    def test_light_chart_background_keeps_series_color(self) -> None:
        use("light")
        chart = TrackChartView([("x", "x", "#6cb6ff")], "x (px)")
        try:
            self.assertEqual(chart.chart().backgroundBrush().color().name(), "#ffffff")
            self.assertIn("#ffffff", chart.styleSheet())
            self.assertEqual(chart._specs[0][2], "#6cb6ff")
            use("dark")
            chart.apply_theme()
            self.assertEqual(chart.chart().backgroundBrush().color().name(), "#232323")
        finally:
            chart.close()
            use("dark")

    def test_refresh_uses_the_chosen_theme(self) -> None:
        window = MainWindow()
        try:
            use("light")
            window._refresh_theme()
            sheet = window.styleSheet()
            self.assertIn("#f4f5f7", sheet)
            self.assertNotIn("#1f1f1f", sheet)
            background = window._chart_panel._charts[0].chart().backgroundBrush().color().name()
            self.assertEqual(background, "#ffffff")
            self.assertEqual(window._assistant_window.styleSheet(), sheet)
        finally:
            window.close()
            use("dark")
