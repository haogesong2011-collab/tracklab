"""Background similarity-transform camera-shake compensation."""

from __future__ import annotations

import math
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
    Similarity,
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
        self.assertTrue(shake.recommended)

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

    def test_to_stable_and_to_raw_roundtrip(self) -> None:
        shake = ShakeCompensation(dx=(0.0, 6.0), dy=(0.0, 7.0), anchors=())
        sx, sy = shake.to_stable(16.0, 27.0, 1)
        self.assertAlmostEqual(sx, 10.0)
        self.assertAlmostEqual(sy, 20.0)
        rx, ry = shake.to_raw(sx, sy, 1)
        self.assertAlmostEqual(rx, 16.0)
        self.assertAlmostEqual(ry, 27.0)

    def test_similarity_inverse_roundtrip(self) -> None:
        model = Similarity(a=0.98, b=0.05, tx=4.0, ty=-3.0)
        x, y = 40.0, 90.0
        xp, yp = model.apply(x, y)
        bx, by = model.inverse().apply(xp, yp)
        self.assertAlmostEqual(bx, x, delta=1e-6)
        self.assertAlmostEqual(by, y, delta=1e-6)

    def test_stationary_does_not_add_track_jitter(self) -> None:
        frames = [_frame(0, 0) for _ in range(10)]
        shake = estimate_shake_from_frames(frames, total=len(frames))
        raw = TrackResult(
            clip_id="static",
            points=[TrackPoint(frame=i, x=90.0, y=80.0) for i in range(10)],
        )
        fixed = compensate_result(result := raw, shake)

        def _mean_step(points: list[TrackPoint]) -> float:
            steps = [
                math.hypot(b.x - a.x, b.y - a.y)
                for a, b in zip(points, points[1:])
            ]
            return float(np.mean(steps)) if steps else 0.0

        self.assertLessEqual(_mean_step(fixed.points), _mean_step(raw.points) + 0.05)
        rms = math.sqrt(np.mean([(p.x - 90.0) ** 2 + (p.y - 80.0) ** 2 for p in fixed.points]))
        self.assertLess(rms, 0.5)

    def test_rotation_is_recovered(self) -> None:
        cx, cy = 100.0, 80.0
        theta = math.radians(2.0)
        frames = []
        for i in range(6):
            corners = []
            for x, y in CORNERS:
                dx, dy = x - cx, y - cy
                c, s = math.cos(i * theta), math.sin(i * theta)
                corners.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
            frames.append(_frame(0, 0, corners=corners))
        shake = estimate_shake_from_frames(frames, total=len(frames))
        ox, oy = 90.0, 50.0
        points = []
        for i in range(6):
            dx, dy = ox - cx, oy - cy
            c, s = math.cos(i * theta), math.sin(i * theta)
            points.append(
                TrackPoint(
                    frame=i,
                    x=cx + c * dx - s * dy,
                    y=cy + s * dx + c * dy,
                )
            )
        fixed = compensate_result(TrackResult(clip_id="rot", points=points), shake)
        rms = math.sqrt(
            np.mean([(p.x - ox) ** 2 + (p.y - oy) ** 2 for p in fixed.points])
        )
        self.assertLess(rms, 1.0)

    def test_scale_is_recovered(self) -> None:
        cx, cy = 100.0, 80.0
        frames = []
        for i in range(5):
            scale = 1.0 + 0.012 * i
            corners = [
                (cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in CORNERS
            ]
            frames.append(_frame(0, 0, corners=corners))
        shake = estimate_shake_from_frames(frames, total=len(frames))
        ox, oy = 70.0, 55.0
        points = []
        for i in range(5):
            scale = 1.0 + 0.012 * i
            points.append(
                TrackPoint(
                    frame=i,
                    x=cx + (ox - cx) * scale,
                    y=cy + (oy - cy) * scale,
                )
            )
        fixed = compensate_result(TrackResult(clip_id="scale", points=points), shake)
        rms = math.sqrt(
            np.mean([(p.x - ox) ** 2 + (p.y - oy) ** 2 for p in fixed.points])
        )
        self.assertLess(rms, 1.0)

    def test_moving_center_blob_does_not_break_translation(self) -> None:
        step_x, step_y = 1, 1
        frames = []
        for i in range(8):
            img = _frame(i * step_x, i * step_y)
            bx = 80 + i * 6
            by = 70
            img[by : by + 12, bx : bx + 12] = 255
            frames.append(img)
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertAlmostEqual(shake.dx[-1], 7 * step_x, delta=2.0)
        self.assertAlmostEqual(shake.dy[-1], 7 * step_y, delta=2.0)

    def test_one_frame_mismatch_does_not_spike(self) -> None:
        rng = np.random.default_rng(0)
        frames = [_frame(i, 0) for i in range(8)]
        frames[3] = rng.integers(0, 40, size=frames[0].shape, dtype=np.uint8)
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertLess(abs(shake.dx[3] - 3.0), 2.5)
        self.assertAlmostEqual(shake.dx[-1], 7.0, delta=2.0)
        result = TrackResult(
            clip_id="spike",
            points=[TrackPoint(frame=i, x=90.0 + i, y=80.0) for i in range(8)],
        )
        fixed = compensate_result(result, shake)
        vx = [
            (b.x - a.x)
            for a, b in zip(fixed.points, fixed.points[1:])
        ]
        self.assertLess(max(abs(v) for v in vx), 3.0)

    def test_repeated_texture_does_not_spike(self) -> None:
        frames = []
        for i in range(8):
            img = _frame(i, 0)
            _stamp(img, 52 + i, 28, 0)
            _stamp(img, 52 + i, H - 29, 2)
            frames.append(img)
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertAlmostEqual(shake.dx[-1], 7.0, delta=2.5)
        result = TrackResult(
            clip_id="repeat",
            points=[TrackPoint(frame=i, x=90.0 + i, y=80.0) for i in range(8)],
        )
        fixed = compensate_result(result, shake)
        vx = [abs(b.x - a.x) for a, b in zip(fixed.points, fixed.points[1:])]
        self.assertLess(max(vx), 3.0)

    def test_compensate_does_not_add_points(self) -> None:
        result = TrackResult(
            clip_id="n",
            points=[
                TrackPoint(frame=0, x=1, y=2, visible=False),
                TrackPoint(frame=1, x=3, y=4, manual=True),
            ],
        )
        shake = ShakeCompensation(dx=(1.0, 2.0), dy=(0.0, 0.0), anchors=())
        out = compensate_result(result, shake)
        self.assertEqual(len(out.points), 2)
        self.assertFalse(out.points[0].visible)
        self.assertTrue(out.points[1].manual)
        self.assertEqual(out.points[0].frame, 0)

    def test_lost_anchors_do_not_snap_to_identity(self) -> None:
        frames = [_frame(i, 0) for i in range(5)]
        blank = np.zeros_like(frames[0])
        frames.extend([blank, blank, blank])
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertAlmostEqual(shake.dx[4], 4.0, delta=2.0)
        self.assertAlmostEqual(shake.dx[-1], shake.dx[4], delta=2.5)
        result = TrackResult(
            clip_id="lost",
            points=[
                TrackPoint(frame=i, x=90.0 + min(i, 4), y=80.0)
                for i in range(8)
            ],
        )
        fixed = compensate_result(result, shake)
        steps = [abs(b.x - a.x) for a, b in zip(fixed.points, fixed.points[1:])]
        self.assertLess(max(steps), 3.0)

    def test_mostly_noise_is_not_recommended(self) -> None:
        rng = np.random.default_rng(2)
        frames = [_frame(0, 0)]
        for _ in range(8):
            frames.append(rng.integers(0, 40, size=frames[0].shape, dtype=np.uint8))
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertFalse(shake.recommended)
        result = TrackResult(
            clip_id="noise",
            points=[TrackPoint(frame=i, x=90.0, y=80.0) for i in range(len(frames))],
        )
        fixed = compensate_result(result, shake)
        steps = [
            math.hypot(b.x - a.x, b.y - a.y)
            for a, b in zip(fixed.points, fixed.points[1:])
        ]
        self.assertLess(max(steps), 4.0)

    def test_pendulum_shake_fixture_does_not_worsen_track(self) -> None:
        from tests.ai.generate_fixtures import FPS, HEIGHT, WIDTH, pendulum_track

        frames, ann = pendulum_track(24)
        corners = (
            (28, 28),
            (WIDTH - 29, 28),
            (28, HEIGHT - 29),
            (WIDTH - 29, HEIGHT - 29),
        )
        stamped: list[np.ndarray] = []
        for img in frames:
            copy = img.copy()
            for code, (cx, cy) in enumerate(corners):
                _stamp(copy, cx, cy, code)
            stamped.append(copy)
        rng = np.random.default_rng(1)
        shaken: list[np.ndarray] = []
        dxs: list[int] = []
        dys: list[int] = []
        for img in stamped:
            dx, dy = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
            shaken.append(np.roll(np.roll(img, dy, axis=0), dx, axis=1))
            dxs.append(dx)
            dys.append(dy)
        gt = [item.center for item in ann.track]
        raw = TrackResult(
            clip_id="track_pendulum_shake_hard",
            points=[
                TrackPoint(frame=i, x=gt[i].x + dxs[i], y=gt[i].y + dys[i])
                for i in range(len(gt))
            ],
        )
        shake = estimate_shake_from_frames(shaken, total=len(shaken))
        fixed = compensate_result(raw, shake)

        def _rms(points: list[TrackPoint]) -> float:
            return float(
                math.sqrt(
                    np.mean(
                        [(p.x - t.x) ** 2 + (p.y - t.y) ** 2 for p, t in zip(points, gt)]
                    )
                )
            )

        def _mean_step(points: list[TrackPoint]) -> float:
            steps = [
                math.hypot(b.x - a.x, b.y - a.y)
                for a, b in zip(points, points[1:])
            ]
            return float(np.mean(steps)) if steps else 0.0

        def _p95_speed(points: list[TrackPoint]) -> float:
            dt = 1.0 / FPS
            speeds = [
                math.hypot(b.x - a.x, b.y - a.y) / dt
                for a, b in zip(points, points[1:])
            ]
            return float(np.percentile(speeds, 95)) if speeds else 0.0

        self.assertLessEqual(_rms(fixed.points), _rms(raw.points) + 0.35)
        self.assertLess(_rms(fixed.points), 1.0)

        ox, oy = 90.0, 90.0
        raw_bg = TrackResult(
            clip_id="bg",
            points=[
                TrackPoint(frame=i, x=ox + dxs[i], y=oy + dys[i])
                for i in range(len(dxs))
            ],
        )
        fix_bg = compensate_result(raw_bg, shake)
        self.assertLessEqual(
            _mean_step(fix_bg.points), _mean_step(raw_bg.points) + 0.05
        )
        self.assertLessEqual(
            _p95_speed(fix_bg.points), _p95_speed(raw_bg.points) + 8.0
        )

    def test_translation_stays_translation_model(self) -> None:
        frames = [_frame(i, 0) for i in range(8)]
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertEqual(shake.model_kind, "translation")

    def test_uhd_scale_recovers_translation(self) -> None:
        width, height = 1280, 720
        corners = ((80, 60), (width - 90, 60), (80, height - 70), (width - 90, height - 70))
        step = 2
        frames = [
            _frame(i * step, 0, width=width, height=height, corners=corners)
            for i in range(6)
        ]
        shake = estimate_shake_from_frames(frames, total=len(frames))
        self.assertAlmostEqual(shake.dx[-1], 5 * step, delta=2.5)
        result = TrackResult(
            clip_id="uhd",
            points=[TrackPoint(frame=i, x=640.0 + i * step, y=360.0) for i in range(6)],
        )
        fixed = compensate_result(result, shake)
        self._assert_not_worse(result, fixed)
        for point in fixed.points:
            self.assertAlmostEqual(point.x, 640.0, delta=2.5)

    def test_edge_player_is_excluded(self) -> None:
        step_x = 1
        frames = []
        exclude: dict[int, list[tuple[float, float]]] = {}
        for i in range(8):
            img = _frame(i * step_x, 0)
            px, py = W - 24, 30 + i * 8
            img[py - 8 : py + 8, px - 8 : px + 8] = 255
            frames.append(img)
            exclude[i] = [(float(px), float(py))]
        shake = estimate_shake_from_frames(
            frames, total=len(frames), exclude_by_frame=exclude
        )
        self.assertAlmostEqual(shake.dx[-1], 7 * step_x, delta=2.0)
        result = TrackResult(
            clip_id="player",
            points=[TrackPoint(frame=i, x=90.0 + i * step_x, y=80.0) for i in range(8)],
        )
        fixed = compensate_result(result, shake)
        self._assert_not_worse(result, fixed)

    def test_repeating_court_texture_does_not_worsen(self) -> None:
        frames = []
        for i in range(8):
            img = np.zeros((H, W, 3), dtype=np.uint8)
            yy, xx = np.mgrid[0:H, 0:W]
            tile = ((((xx + i) // 12) + ((yy) // 12)) % 2) * 70 + 40
            img[..., 0] = tile
            img[..., 1] = 90
            img[..., 2] = 55
            for code, (cx, cy) in enumerate(CORNERS):
                _stamp(img, cx + i, cy, code)
            frames.append(img)
        shake = estimate_shake_from_frames(frames, total=len(frames))
        result = TrackResult(
            clip_id="court",
            points=[TrackPoint(frame=i, x=90.0 + i, y=80.0) for i in range(8)],
        )
        fixed = compensate_result(result, shake)
        self._assert_not_worse(result, fixed)

    def test_alternating_transform_noise_is_gated(self) -> None:
        n = 12
        result = TrackResult(
            clip_id="alt",
            points=[TrackPoint(frame=i, x=40.0 + 2.0 * i, y=50.0) for i in range(n)],
        )
        shake = ShakeCompensation(
            dx=tuple(18.0 if i % 2 else 0.0 for i in range(n)),
            dy=tuple(0.0 for _ in range(n)),
            anchors=(),
            recommended=True,
            gains=tuple(1.0 for _ in range(n)),
        )
        fixed = compensate_result(result, shake)
        for raw, out in zip(result.points, fixed.points):
            self.assertAlmostEqual(out.x, raw.x)
            self.assertAlmostEqual(out.y, raw.y)

    def test_camera_cut_does_not_worsen_later_segment(self) -> None:
        frames = [_frame(i, 0) for i in range(5)]
        rng = np.random.default_rng(4)
        frames.extend(rng.integers(0, 40, size=frames[0].shape, dtype=np.uint8) for _ in range(6))
        shake = estimate_shake_from_frames(frames, total=len(frames))
        result = TrackResult(
            clip_id="cut",
            points=[TrackPoint(frame=i, x=90.0 + min(i, 4), y=80.0) for i in range(11)],
        )
        fixed = compensate_result(result, shake)
        self._assert_not_worse(result, fixed)
        for raw, out in zip(result.points[6:], fixed.points[6:]):
            self.assertAlmostEqual(out.x, raw.x, delta=0.05)
            self.assertAlmostEqual(out.y, raw.y, delta=0.05)

    def test_projectile_with_known_camera_motion(self) -> None:
        from tests.ai.generate_fixtures import projectile_track

        frames, ann = projectile_track(24)
        corners = (
            (28, 28),
            (ann.width - 29, 28),
            (28, ann.height - 29),
            (ann.width - 29, ann.height - 29),
        )
        cam = [(i, 0) for i in range(len(frames))]
        shaken: list[np.ndarray] = []
        exclude: dict[int, list[tuple[float, float]]] = {}
        for i, (img, (dx, dy)) in enumerate(zip(frames, cam)):
            copy = img.copy()
            for code, (cx, cy) in enumerate(corners):
                _stamp(copy, cx, cy, code)
            shaken.append(np.roll(np.roll(copy, dy, axis=0), dx, axis=1))
            exclude[i] = [(gt_x + dx, gt_y + dy) for gt_x, gt_y in [(ann.track[i].center.x, ann.track[i].center.y)]]
        gt = [item.center for item in ann.track]
        raw = TrackResult(
            clip_id="proj-cam",
            points=[
                TrackPoint(frame=i, x=gt[i].x + cam[i][0], y=gt[i].y + cam[i][1])
                for i in range(len(gt))
                if 0 <= gt[i].x < ann.width and 0 <= gt[i].y < ann.height
            ],
        )
        shake = estimate_shake_from_frames(
            shaken, total=len(shaken), exclude_by_frame=exclude
        )
        fixed = compensate_result(raw, shake)
        self._assert_not_worse(raw, fixed)
        visible_gt = [gt[p.frame] for p in fixed.points]
        rms = math.sqrt(
            np.mean([(p.x - t.x) ** 2 + (p.y - t.y) ** 2 for p, t in zip(fixed.points, visible_gt)])
        )
        if shake.recommended and all(g > 0.5 for g in shake.gains[: len(fixed.points)]):
            self.assertLessEqual(rms, 1.0)

    def _assert_not_worse(self, raw: TrackResult, fixed: TrackResult) -> None:
        from ai.stabilize import _jitter_metrics

        raw_m = _jitter_metrics([p.x for p in raw.points], [p.y for p in raw.points])
        fix_m = _jitter_metrics([p.x for p in fixed.points], [p.y for p in fixed.points])
        self.assertLessEqual(fix_m[0], raw_m[0] * 1.05 + 0.45)
        self.assertLessEqual(fix_m[1], raw_m[1] * 1.05 + 0.45)
        self.assertLessEqual(fix_m[2], raw_m[2] * 1.05 + 0.45)


if __name__ == "__main__":
    unittest.main()
