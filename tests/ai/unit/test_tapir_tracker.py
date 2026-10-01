from __future__ import annotations

import unittest

import numpy as np

from ai.contracts import TrackPointSource, TrackPointStatus
from ai.models import load_video
from ai.tapir_tracker import MODEL_SIZE, BootsTapirTracker, CropTransform
from tests.ai.dataset import load_manifest, resolve_video


class _FakeTapir:
    def __init__(self) -> None:
        self.query = None

    def __call__(self, frames, query):  # noqa: ANN001
        import torch

        self.query = query.detach().cpu().numpy()
        count = frames.shape[1]
        xy = torch.zeros((1, 1, count, 2), device=frames.device)
        xy[0, 0, :, 0] = query[0, 0, 2] + torch.arange(count, device=frames.device)
        xy[0, 0, :, 1] = query[0, 0, 1]
        occ = torch.full((1, 1, count), -6.0, device=frames.device)
        uncertainty = torch.full((1, 1, count), -6.0, device=frames.device)
        occ[0, 0, -1] = 6.0
        return {"tracks": xy, "occlusion": occ, "expected_dist": uncertainty}


class BootsTapirTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_surface_point_keeps_query_identity_and_uncertainty(self) -> None:
        entry = next(
            item for item in load_manifest().entries if item.clip_id == "track_ball_normal"
        )
        info = load_video(resolve_video(entry))
        fake = _FakeTapir()
        seed = (31.5, 22.25)
        result = BootsTapirTracker(model=fake, device="cpu", local_refine=False).track(
            info, seed, start_frame=0, end_frame=3
        )
        self.assertEqual(len(result.points), 4)
        assert fake.query is not None
        u, v = CropTransform.full(info.width, info.height, MODEL_SIZE).to_model(*seed)
        self.assertAlmostEqual(float(fake.query[0, 0, 1]), v, places=4)
        self.assertAlmostEqual(float(fake.query[0, 0, 2]), u, places=4)
        self.assertEqual(result.points[0].source, TrackPointSource.AUTO)
        self.assertEqual(result.points[0].status, TrackPointStatus.TRUSTED)
        self.assertEqual(result.points[-1].status, TrackPointStatus.LOST)
        self.assertFalse(result.points[-1].visible)
        self.assertIn("position_uncertainty", result.points[-1].diagnostics)

    def test_output_does_not_use_mask_centres_or_interpolation(self) -> None:
        entry = next(
            item for item in load_manifest().entries if item.clip_id == "track_ball_normal"
        )
        info = load_video(resolve_video(entry))
        result = BootsTapirTracker(model=_FakeTapir(), device="cpu", local_refine=False).track(
            info, (10.0, 12.0), start_frame=0, end_frame=2
        )
        self.assertTrue(all(not point.interpolated for point in result.points))
        self.assertAlmostEqual(result.points[0].x, 10.0, places=4)
        self.assertAlmostEqual(result.points[0].y, 12.0, places=4)

    def test_query_frame_is_not_replaced_by_the_playback_start(self) -> None:
        from ai.contracts import PromptKind, TrackPrompt

        entry = next(
            item for item in load_manifest().entries if item.clip_id == "track_ball_normal"
        )
        info = load_video(resolve_video(entry))
        fake = _FakeTapir()
        BootsTapirTracker(model=fake, device="cpu", local_refine=False).track(
            info,
            (1.0, 1.0),
            start_frame=2,
            end_frame=4,
            prompts=[TrackPrompt(frame=0, kind=PromptKind.POSITIVE, x=40.0, y=18.0)],
        )
        assert fake.query is not None
        # The real query frame is prepended as t=0; the playback start is not
        # substituted for it.
        self.assertAlmostEqual(float(fake.query[0, 0, 0]), 0.0)
        u, _v = CropTransform.full(info.width, info.height, MODEL_SIZE).to_model(40.0, 18.0)
        self.assertAlmostEqual(float(fake.query[0, 0, 2]), u, places=4)


if __name__ == "__main__":
    unittest.main()
