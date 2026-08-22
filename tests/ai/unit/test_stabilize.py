"""Four-corner camera-shake compensation (analysis coords only)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.contracts import TrackPoint, TrackResult  # noqa: E402
from ai.stabilize import (  # noqa: E402
    ShakeCompensation,
    compensate_result,
    detect_anchor_points,
    estimate_shake_from_frames,
)


W, H = 200, 160
CORNERS = ((28, 28), (W - 29, 28), (28, H - 29), (W - 29, H - 29))


def _stamp(frame: np.ndarray, cx: int, cy: int, code: int) -> None:
    yy, xx = np.mgrid[-7:8, -7:8]
    base = ((xx * (code + 1) + yy * (code + 3)) % 17) * 14 + 40
    cross = (np.abs(xx) <= 1) | (np.abs(yy) <= 1)
    val = np.where(cross, 255, base)
    rgb = np.stack(
        [val, (val + code * 30) % 220 + 20, 255 - val], axis=-1
    ).astype(np.uint8)
    x0, y0 = int(cx) - 7, int(cy) - 7
    if y0 < 0 or x0 < 0 or y0 + 15 > frame.shape[0] or x0 + 15 > frame.shape[1]:
        return
    frame[y0 : y0 + 15, x0 : x0 + 15] = rgb


def _frame(dx: int, dy: int, *, width: int = W, height: int = H, corners=CORNERS) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for code, (cx, cy) in enumerate(corners):
        _stamp(img, cx + dx, cy + dy, code)
    return img


class StabilizeTests(unittest.TestCase):
    def test_detects_points_in_four_corners(self) -> None:
        anchors = detect_anchor_points(_frame(0, 0))
        self.assertGreaterEqual(len(anchors), 4)
        hit = [False, False, False, False]
        for x, y in anchors:
            if x < W * 0.3 and y < H * 0.3:
                hit[0] = True
            elif x > W * 0.7 and y < H * 0.3:
                hit[1] = True
            elif x < W * 0.3 and y > H * 0.7:
                hit[2] = True
            elif x > W * 0.7 and y > H * 0.7:
                hit[3] = True
        self.assertEqual(hit, [True, True, True, True])

    def test_stationary_scene_offset_is_near_zero(self) -> None:
        frames = [_frame(0, 0) for _ in range(6)]
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertEqual(len(shake.dx), 6)
        for dx, dy in zip(shake.dx, shake.dy):
            self.assertLess(abs(dx), 1.5)
            self.assertLess(abs(dy), 1.5)

    def test_global_translation_is_recovered(self) -> None:
        step_x, step_y = 1, 2
        frames = [_frame(i * step_x, i * step_y) for i in range(8)]
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertEqual(len(shake.dx), 8)
        self.assertAlmostEqual(shake.dx[0], 0.0, delta=0.6)
        self.assertAlmostEqual(shake.dy[0], 0.0, delta=0.6)
        self.assertAlmostEqual(shake.dx[-1], 7 * step_x, delta=1.5)
        self.assertAlmostEqual(shake.dy[-1], 7 * step_y, delta=1.5)

        ox, oy = 90.0, 80.0
        result = TrackResult(
            clip_id="shake",
            points=[
                TrackPoint(frame=i, x=ox + i * step_x, y=oy + i * step_y)
                for i in range(8)
            ],
        )
        fixed = compensate_result(result, shake)
        for point in fixed.points:
            self.assertAlmostEqual(point.x, ox, delta=1.6)
            self.assertAlmostEqual(point.y, oy, delta=1.6)

    def test_large_interframe_translation_is_recovered(self) -> None:
        width, height = 480, 360
        corners = ((52, 40), (428, 40), (52, 320), (428, 320))
        step_x, step_y = 40, 28
        frames = [
            _frame(i * step_x, i * step_y, width=width, height=height, corners=corners)
            for i in range(4)
        ]
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertEqual(len(shake.dx), 4)
        self.assertAlmostEqual(shake.dx[-1], 3 * step_x, delta=2.5)
        self.assertAlmostEqual(shake.dy[-1], 3 * step_y, delta=2.5)
        ox, oy = 240.0, 180.0
        result = TrackResult(
            clip_id="shake-large",
            points=[
                TrackPoint(frame=i, x=ox + i * step_x, y=oy + i * step_y)
                for i in range(4)
            ],
        )
        fixed = compensate_result(result, shake)
        for point in fixed.points:
            self.assertAlmostEqual(point.x, ox, delta=2.5)
            self.assertAlmostEqual(point.y, oy, delta=2.5)

    def test_very_large_interframe_jump_is_recovered(self) -> None:
        width, height = 480, 360
        corners = ((52, 40), (428, 40), (52, 320), (428, 320))
        step_x, step_y = 72, 50
        frames = [
            _frame(i * step_x, i * step_y, width=width, height=height, corners=corners)
            for i in range(3)
        ]
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertAlmostEqual(shake.dx[-1], 2 * step_x, delta=3.0)
        self.assertAlmostEqual(shake.dy[-1], 2 * step_y, delta=3.0)

    def test_compensate_result_subtracts_offset(self) -> None:
        result = TrackResult(
            clip_id="c",
            points=[
                TrackPoint(frame=0, x=10, y=20, manual=False),
                TrackPoint(frame=1, x=13, y=24, manual=True),
            ],
        )
        shake = ShakeCompensation(dx=(0.0, 3.0), dy=(0.0, 4.0), anchors=())
        out = compensate_result(result, shake)
        self.assertAlmostEqual(out.points[0].x, 10.0)
        self.assertAlmostEqual(out.points[0].y, 20.0)
        self.assertAlmostEqual(out.points[1].x, 10.0)
        self.assertAlmostEqual(out.points[1].y, 20.0)
        self.assertTrue(out.points[1].manual)
        self.assertIs(compensate_result(result, None), result)

    def test_offset_clamps_frame_index(self) -> None:
        shake = ShakeCompensation(dx=(1.0, 2.0), dy=(3.0, 4.0), anchors=())
        self.assertEqual(shake.offset(-3), (1.0, 3.0))
        self.assertEqual(shake.offset(99), (2.0, 4.0))


if __name__ == "__main__":
    unittest.main()
