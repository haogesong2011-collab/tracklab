"""Unit tests for pixel/world transforms, calibration, and interpolation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.calibration import (  # noqa: E402
    CalibrationMode,
    RulerRole,
    RulerSegment,
    near_far_state,
    uniform_state,
)
from ai.geometry import interpolate_xy, pixel_to_world, world_to_pixel  # noqa: E402
from ai.schema import Point2D  # noqa: E402


class GeometryTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        origin = Point2D(40, 140)
        ppm = 200.0
        for angle in (0.0, 15.0, -30.0):
            pixel = Point2D(80.0, 90.0)
            world = pixel_to_world(
                pixel.x, pixel.y, origin=origin, pixels_per_meter=ppm, axis_angle_deg=angle
            )
            back = world_to_pixel(
                world.x, world.y, origin=origin, pixels_per_meter=ppm, axis_angle_deg=angle
            )
            self.assertAlmostEqual(back.x, pixel.x, places=6)
            self.assertAlmostEqual(back.y, pixel.y, places=6)

    def test_y_up(self) -> None:
        origin = Point2D(0, 100)
        world = pixel_to_world(0, 0, origin=origin, pixels_per_meter=50.0)
        self.assertAlmostEqual(world.x, 0.0)
        self.assertAlmostEqual(world.y, 2.0)

    def test_interpolate(self) -> None:
        series = [(0, 0.0, 0.0), (10, 10.0, 20.0)]
        x, y = interpolate_xy(series, 5)  # type: ignore[misc]
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 10.0)
        self.assertIsNone(interpolate_xy([], 0))

    def test_uniform_ruler_scale(self) -> None:
        state = uniform_state(
            Point2D(0, 0),
            Point2D(100, 0),
            length_m=1.0,
            origin=Point2D(0, 100),
            axis_angle_deg=0.0,
        )
        state.frame.y_up = True
        self.assertEqual(state.mode, CalibrationMode.UNIFORM)
        self.assertAlmostEqual(state.pixels_per_meter_at(10, 10) or 0.0, 100.0)
        world = state.pixel_to_world(0, 0)
        self.assertAlmostEqual(world.x, 0.0)
        self.assertAlmostEqual(world.y, 1.0)
        back = state.world_to_pixel(world.x, world.y)
        self.assertAlmostEqual(back.x, 0.0, places=5)
        self.assertAlmostEqual(back.y, 0.0, places=5)

    def test_near_far_interpolation_and_extrapolation(self) -> None:
        near = RulerSegment(
            Point2D(0, 100), Point2D(100, 100), length_m=1.0, role=RulerRole.NEAR
        )
        far = RulerSegment(
            Point2D(0, 300), Point2D(50, 300), length_m=1.0, role=RulerRole.FAR
        )
        state = near_far_state(near, far, origin=Point2D(50, 100), axis_angle_deg=0.0)
        state.frame.y_up = True
        self.assertEqual(state.mode, CalibrationMode.NEAR_FAR)
        self.assertAlmostEqual(state.pixels_per_meter_at(50, 100) or 0.0, 100.0, places=4)
        self.assertAlmostEqual(state.pixels_per_meter_at(25, 300) or 0.0, 50.0, places=4)
        self.assertAlmostEqual(state.pixels_per_meter_at(37.5, 200) or 0.0, 75.0, places=4)
        self.assertFalse(state.is_extrapolated(37.5, 200))
        self.assertTrue(state.is_extrapolated(40, 400))
        ppm_out = state.pixels_per_meter_at(40, 400)
        self.assertIsNotNone(ppm_out)
        assert ppm_out is not None
        self.assertGreater(ppm_out, 0.0)
        world = state.pixel_to_world(50, 200)
        back = state.world_to_pixel(world.x, world.y)
        self.assertAlmostEqual(back.x, 50.0, places=3)
        self.assertAlmostEqual(back.y, 200.0, places=3)

    def test_axis_rotation_and_reject_skewed_rulers(self) -> None:
        state = uniform_state(
            Point2D(10, 10),
            Point2D(110, 10),
            length_m=1.0,
            origin=Point2D(10, 10),
            axis_angle_deg=90.0,
        )
        state.frame.y_up = True
        world = state.pixel_to_world(10, 0)
        self.assertAlmostEqual(world.x, 0.1, places=5)
        self.assertAlmostEqual(world.y, 0.0, places=5)

        skewed = near_far_state(
            RulerSegment(Point2D(0, 0), Point2D(80, 0), length_m=1.0, role=RulerRole.NEAR),
            RulerSegment(Point2D(0, 80), Point2D(0, 160), length_m=1.0, role=RulerRole.FAR),
        )
        self.assertEqual(skewed.mode, CalibrationMode.NONE)
        self.assertIn("夹角", skewed.warning)

    def test_ruler_without_origin_flips_y(self) -> None:
        state = uniform_state(Point2D(0, 0), Point2D(100, 0), length_m=1.0)
        self.assertIsNone(state.frame.origin)
        self.assertTrue(state.frame.y_up)
        world = state.pixel_to_world(0, 100)
        self.assertAlmostEqual(world.x, 0.0, places=5)
        self.assertAlmostEqual(world.y, -1.0, places=5)
        back = state.world_to_pixel(world.x, world.y)
        self.assertAlmostEqual(back.x, 0.0, places=5)
        self.assertAlmostEqual(back.y, 100.0, places=5)

    def test_default_frame_y_up_from_origin(self) -> None:
        state = uniform_state(
            Point2D(0, 0),
            Point2D(100, 0),
            length_m=1.0,
            origin=Point2D(0, 0),
            axis_angle_deg=0.0,
        )
        self.assertTrue(state.frame.y_up)
        below = state.pixel_to_world(0, 100)
        self.assertAlmostEqual(below.x, 0.0, places=5)
        self.assertAlmostEqual(below.y, -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
