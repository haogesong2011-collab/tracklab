"""Tracker-style fast autotracker tests."""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.autotracker import (  # noqa: E402
    GOOD_MATCH,
    TrackerAutoTracker,
    _match,
    _predict,
    rgb_squared_difference_map,
)
from ai.contracts import (  # noqa: E402
    CancelToken,
    FailureReason,
    PromptKind,
    TrackMode,
    TrackPoint,
    TrackPrompt,
)
from tests.ai.generate_fixtures import _write_video  # noqa: E402


def _moving_disk_video(
    path: Path, frames: int = 24
) -> tuple[float, float, float, float]:
    x0, y0 = 48.0, 60.0
    dx, dy = 3.0, 1.0
    images = []
    for i in range(frames):
        image = np.full((120, 200, 3), (28, 30, 34), dtype=np.uint8)
        cx, cy = x0 + i * dx, y0 + i * dy
        yy, xx = np.ogrid[:120, :200]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 8 * 8
        image[mask] = (220, 80, 40)
        images.append(image)
    _write_video(path, images, fps=30.0)
    return x0, y0, dx, dy


class TemplateMatcherTests(unittest.TestCase):
    def test_rgb_squared_difference_has_exact_minimum(self) -> None:
        image = np.full((70, 90, 3), 25, dtype=np.uint8)
        yy, xx = np.mgrid[0:19, 0:21]
        template = np.stack(
            [
                (xx * 7 + yy * 3) % 255,
                (xx * 2 + yy * 11) % 255,
                (xx * 13 + yy * 5) % 255,
            ],
            axis=-1,
        ).astype(np.uint8)
        image[17:36, 29:50] = template
        diffs = rgb_squared_difference_map(image, template)
        row, col = np.unravel_index(int(np.argmin(diffs)), diffs.shape)
        self.assertEqual((int(row), int(col)), (17, 29))
        self.assertLess(float(diffs[row, col]), 1e-3)

    def test_match_uses_peak_height_and_subpixel_location(self) -> None:
        image = np.full((90, 120, 3), 20, dtype=np.uint8)
        yy, xx = np.mgrid[-9:10, -9:10]
        patch = np.stack(
            [
                np.clip(220 - (xx * xx + yy * yy) * 2, 20, 220),
                np.clip(150 - (xx * xx + yy * yy), 20, 150),
                np.clip(90 + xx * 3 - yy * 2, 20, 220),
            ],
            axis=-1,
        ).astype(np.uint8)
        image[31:50, 51:70] = patch
        x, y, peak, _width = _match(
            image,
            patch.astype(np.float32),
            np.ones((19, 19), dtype=np.float32),
            60,
            40,
            35,
        )
        self.assertAlmostEqual(x, 60.5, delta=0.6)
        self.assertAlmostEqual(y, 40.5, delta=0.6)
        self.assertGreater(peak, GOOD_MATCH)

    def test_look_ahead_uses_velocity_and_acceleration(self) -> None:
        linear = [
            TrackPoint(frame=i, x=10 + i * 3, y=20 - i * 2) for i in range(4)
        ]
        self.assertEqual(_predict(linear, (0, 0)), (22.0, 12.0))
        accelerated = [
            TrackPoint(frame=i, x=10 + i * i, y=30.0) for i in range(3)
        ]
        self.assertEqual(_predict(accelerated, (0, 0)), (19.0, 30.0))

    def test_fast_source_has_no_neural_or_color_blob_import(self) -> None:
        import ai.autotracker as module

        source = inspect.getsource(module)
        self.assertNotIn("import torch", source)
        self.assertNotIn("import sam2", source)
        self.assertNotIn("ColorBlobTracker", source)


class TrackerAutoTrackerTests(unittest.TestCase):
    def test_follows_moving_disk(self) -> None:
        from ai.models import load_video

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disk.mp4"
            x0, y0, dx, dy = _moving_disk_video(path)
            info = load_video(path)
            result = TrackerAutoTracker().track(
                info,
                (x0, y0),
                prompts=[
                    TrackPrompt(
                        frame=0, kind=PromptKind.POSITIVE, x=x0, y=y0
                    )
                ],
            )
        self.assertEqual(result.model_name, "tracker_autotracker")
        self.assertTrue(all(point.visible for point in result.points))
        last = result.points[-1]
        self.assertAlmostEqual(last.x, x0 + dx * last.frame, delta=3.0)
        self.assertAlmostEqual(last.y, y0 + dy * last.frame, delta=3.0)

    def test_cancel_stops(self) -> None:
        from ai.models import load_video

        token = CancelToken()

        def progress(event) -> None:  # noqa: ANN001
            if event.current >= 3:
                token.cancel()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disk.mp4"
            x0, y0, _dx, _dy = _moving_disk_video(path, frames=20)
            result = TrackerAutoTracker().track(
                load_video(path), (x0, y0), cancel=token, progress=progress
            )
        self.assertEqual(result.failure_reason, FailureReason.CANCELLED)
        self.assertLess(len(result.points), 20)

    def test_factory_modes(self) -> None:
        from ai.desktop import create_tracker
        from ai.model_manager import SAM21_SMALL, SAM21_TINY
        from ai.sam2_tracker import Sam2Tracker

        fast = create_tracker(TrackMode.FAST)
        precise = create_tracker(TrackMode.PRECISE)
        self.assertIsInstance(fast, Sam2Tracker)
        self.assertIsInstance(precise, Sam2Tracker)
        self.assertEqual(fast.name, SAM21_TINY.model_id)
        self.assertEqual(precise.name, SAM21_SMALL.model_id)
        self.assertEqual(TrackerAutoTracker().name, "tracker_autotracker")


if __name__ == "__main__":
    unittest.main()
