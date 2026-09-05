"""Teaching context must stay compact and must not leak secrets or raw video."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.assistant_context import (  # noqa: E402
    MAX_REPRESENTATIVE_POINTS,
    assert_private_context,
    build_teaching_context,
    chat_messages,
    preview_payload,
    representative_points,
)
from ai.contracts import (  # noqa: E402
    ChatMessage,
    ExperimentAnalysis,
    ExperimentCandidate,
    ExperimentType,
    FitResult,
    TeachingLevel,
)
from ai.experiment_catalog import formulas_for  # noqa: E402
from ai.kinematics import KinematicSample  # noqa: E402


def _analysis() -> ExperimentAnalysis:
    fit = FitResult(
        model="s=s0+vt",
        formula_id="uniform.s",
        frame_start=0,
        frame_end=9,
        time_start_s=0.0,
        time_end_s=0.3,
        parameters={"v": 1.5, "s0": 0.0},
        units={"v": "m/s", "s0": "m"},
        r2=0.99,
        nrmse=0.01,
        n_samples=10,
    )
    candidate = ExperimentCandidate(
        experiment_type=ExperimentType.UNIFORM_LINEAR,
        label="匀速直线运动",
        confidence=0.9,
        fit=fit,
        evidence=["linear"],
        missing=[],
    )
    return ExperimentAnalysis(
        clip_id="demo",
        candidates=[candidate],
        selected=candidate,
        auto_confirmable=True,
        coverage=1.0,
        mean_track_confidence=0.95,
        calibration_active=True,
        position_unit="m",
        speed_unit="m/s",
        fingerprint="abc",
        missing=[],
    )


class AssistantContextTests(unittest.TestCase):
    def test_compresses_representative_points_and_keeps_extrema(self) -> None:
        samples = [
            KinematicSample(
                frame=i,
                time_s=i / 30.0,
                x=float(i),
                y=float((i - 20) ** 2),
                vx=1.0,
                vy=None,
                speed=1.0,
                visible=True,
                confidence=1.0,
                manual=False,
            )
            for i in range(80)
        ]
        samples[10] = KinematicSample(
            frame=10,
            time_s=10 / 30.0,
            x=-50.0,
            y=0.0,
            vx=None,
            vy=None,
            speed=None,
            visible=True,
            confidence=1.0,
            manual=False,
        )
        packed = representative_points(samples)
        self.assertLessEqual(len(packed), MAX_REPRESENTATIVE_POINTS)
        frames = {item["frame"] for item in packed}
        self.assertIn(0, frames)
        self.assertIn(79, frames)
        self.assertIn(10, frames)

    def test_formulas_come_from_catalog_not_model(self) -> None:
        context = build_teaching_context(
            _analysis(),
            confirmed_type=ExperimentType.UNIFORM_LINEAR,
            teaching_level=TeachingLevel.HIGH,
        )
        ids = [item["id"] for item in context["catalog"]["formulas"]]
        self.assertEqual(ids, [item["id"] for item in formulas_for(ExperimentType.UNIFORM_LINEAR)])
        self.assertEqual(context["results"]["parameters"]["v"], 1.5)

    def test_missing_conditions_are_listed(self) -> None:
        analysis = _analysis()
        analysis.missing = ["calibration"]
        analysis.calibration_active = False
        context = build_teaching_context(
            analysis,
            confirmed_type=ExperimentType.UNIFORM_LINEAR,
        )
        self.assertIn("calibration", context["missing"])

    def test_context_omits_paths_and_keys(self) -> None:
        context = build_teaching_context(
            _analysis(),
            confirmed_type=ExperimentType.PENDULUM,
            samples=[],
        )
        assert_private_context(context)
        blob = preview_payload(context)
        self.assertNotIn("api_key", blob.lower())
        self.assertNotIn("/Users", blob)
        messages = chat_messages(
            context,
            [ChatMessage(role="user", content="解释速度")],
            "v 是多少？",
        )
        dumped = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("sk-", dumped)
        self.assertIn("json", messages[-1]["content"].lower() + messages[0]["content"].lower())

    def test_unconfirmed_is_not_marked_as_fact(self) -> None:
        context = build_teaching_context(_analysis(), confirmed_type=None)
        self.assertFalse(context["confirmed"])
        self.assertIsNone(context["confirmed_type"])

    def test_measurement_block_has_no_paths(self) -> None:
        context = build_teaching_context(
            _analysis(),
            confirmed_type=ExperimentType.UNIFORM_LINEAR,
            calibration_mode="planar",
            reprojection_rms_px=0.4,
            off_plane_frames=2,
            camera_moved=True,
            quality_label="运动平面",
        )
        assert_private_context(context)
        self.assertEqual(context["measurement"]["mode"], "planar")
        self.assertEqual(context["measurement"]["off_plane_frames"], 2)
        self.assertTrue(context["measurement"]["camera_moved"])


if __name__ == "__main__":
    unittest.main()
