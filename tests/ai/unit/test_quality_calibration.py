from __future__ import annotations

import unittest

import numpy as np

from ai.quality_calibration import (
    QualityCalibrator,
    calibration_metrics,
    fit_quality_calibrator,
)


class QualityCalibrationTests(unittest.TestCase):
    def test_fit_orders_low_and_high_quality(self) -> None:
        scores = np.array([0.05, 0.1, 0.2, 0.35, 0.65, 0.8, 0.9, 0.98])
        correct = np.array([False, False, False, False, True, True, True, True])
        fitted = fit_quality_calibrator(scores, correct, target="object_center")
        self.assertLess(fitted.predict(0.2), fitted.predict(0.8))
        restored = QualityCalibrator.from_dict(fitted.to_dict())
        self.assertAlmostEqual(restored.predict(0.8), fitted.predict(0.8))

    def test_brier_and_ece_are_zero_for_perfect_binary_predictions(self) -> None:
        metrics = calibration_metrics([0.0, 0.0, 1.0, 1.0], [False, False, True, True])
        self.assertAlmostEqual(metrics["brier"], 0.0)
        self.assertAlmostEqual(metrics["ece"], 0.0)

    def test_fit_requires_both_outcomes(self) -> None:
        with self.assertRaises(ValueError):
            fit_quality_calibrator([0.1, 0.2, 0.3, 0.4], [True] * 4, target="surface_point")


if __name__ == "__main__":
    unittest.main()
