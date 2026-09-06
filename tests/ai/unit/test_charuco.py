"""Optional ChArUco helpers stay importable without OpenCV."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.charuco import (  # noqa: E402
    calibrate_camera,
    detect_plane_from_frame,
    opencv_available,
)


class CharucoTests(unittest.TestCase):
    def test_detect_without_opencv_explains_optional_dep(self) -> None:
        if opencv_available():
            self.skipTest("OpenCV is installed")
        detection, message = detect_plane_from_frame(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertIsNone(detection)
        self.assertIn("OpenCV", message)

    def test_calibrate_requires_views(self) -> None:
        profile, message = calibrate_camera([], (320, 180))
        self.assertIsNone(profile)
        self.assertTrue(message)


if __name__ == "__main__":
    unittest.main()
