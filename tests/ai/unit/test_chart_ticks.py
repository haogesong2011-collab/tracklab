"""Nice 1-2-5 chart axis ticks."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chart_ticks import nice_axis_ticks, nice_tick_interval  # noqa: E402


def _is_one_two_five(step: float) -> bool:
    if step <= 0 or not math.isfinite(step):
        return False
    exp = math.floor(math.log10(step))
    frac = step / (10**exp)
    return any(abs(frac - n) < 1e-6 for n in (1.0, 2.0, 5.0))


class ChartTickTests(unittest.TestCase):
    def test_pixel_range_uses_integer_arithmetic_ticks(self) -> None:
        lo, hi, step, fmt = nice_axis_ticks(103.2, 248.7, 5)
        self.assertTrue(_is_one_two_five(step))
        self.assertGreaterEqual(step, 1.0)
        self.assertEqual(fmt, "%.0f")
        n = round((hi - lo) / step)
        self.assertAlmostEqual(lo + n * step, hi, places=6)
        self.assertLessEqual(lo, 103.2)
        self.assertGreaterEqual(hi, 248.7)

    def test_short_time_axis_stays_arithmetic(self) -> None:
        lo, hi, step, fmt = nice_axis_ticks(0.0, 0.133, 5)
        self.assertTrue(_is_one_two_five(step))
        n = round((hi - lo) / step)
        self.assertGreaterEqual(n, 1)
        self.assertAlmostEqual(lo + n * step, hi, places=6)
        self.assertRegex(fmt, r"%\.\d+f")

    def test_interval_without_expand_matches_visible_span(self) -> None:
        step, fmt = nice_tick_interval(0.04, 0.11, 5)
        self.assertTrue(_is_one_two_five(step))
        self.assertTrue(fmt.startswith("%"))

    def test_equal_bounds_do_not_crash(self) -> None:
        lo, hi, step, _fmt = nice_axis_ticks(5.0, 5.0, 5)
        self.assertLess(lo, hi)
        self.assertGreater(step, 0)
