"""Throughput smoke test for the offline analysis gate (>= 10 fps)."""

from __future__ import annotations

import resource
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.models import ColorBlobTracker, load_video  # noqa: E402
from tests.ai.dataset import load_annotation, load_manifest, resolve_video  # noqa: E402
from tests.ai.generate_fixtures import build_fixtures  # noqa: E402


class ThroughputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (ROOT / "datasets" / "manifest.json").exists():
            build_fixtures()

    def test_baseline_tracker_exceeds_ten_fps(self) -> None:
        manifest = load_manifest()
        entry = next(e for e in manifest.entries if e.clip_id == "track_ball_normal")
        ann = load_annotation(entry.annotation_path)
        info = load_video(resolve_video(entry))
        t0 = time.perf_counter()
        result = ColorBlobTracker().track(
            info, (ann.track[0].center.x, ann.track[0].center.y)
        )
        elapsed = max(time.perf_counter() - t0, 1e-6)
        fps = info.frame_count / elapsed
        self.assertGreaterEqual(fps, 10.0, msg=f"{fps:.1f} fps, {result.elapsed_s:.3f}s")
        self.assertLess(result.elapsed_s, 2.0)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)
        self.assertLess(rss_gb, 2.0, msg=f"peak RSS {rss_gb:.2f} GB")

    def test_1080p_offline_throughput_and_memory(self) -> None:
        import tempfile

        import numpy as np

        from tests.ai.generate_fixtures import _write_video

        n = 20
        height, width = 1080, 1920
        frames: list[np.ndarray] = []
        seed = (300.0, 540.0)
        for i in range(n):
            img = np.full((height, width, 3), 32, dtype=np.uint8)
            cx = seed[0] + i * 25.0
            cy = seed[1]
            yy, xx = np.ogrid[:height, :width]
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 16 ** 2] = (220, 80, 60)
            frames.append(img)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate_1080p.mp4"
            _write_video(path, frames, 30.0)
            info = load_video(path)
            t0 = time.perf_counter()
            result = ColorBlobTracker().track(info, seed)
            elapsed = max(time.perf_counter() - t0, 1e-6)
        fps = n / elapsed
        self.assertGreaterEqual(fps, 10.0, msg=f"{fps:.1f} fps @1080p")
        self.assertEqual(len(result.points), n)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss / (1024 ** 3) if sys.platform == "darwin" else rss / (1024 ** 2)
        self.assertLess(rss_gb, 2.0, msg=f"peak RSS {rss_gb:.2f} GB")
