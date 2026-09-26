"""Calibration utilities for TrackLab tracking-quality experiments.

Runtime scores stay labelled as tracking quality until a calibrator fitted on a
separate calibration split is supplied. This module intentionally has no
automatic fitting path so evaluation videos cannot leak into model selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


@dataclass(frozen=True)
class QualityCalibrator:
    slope: float
    intercept: float
    target: str
    version: str = "platt-v1"

    def predict(self, quality: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(quality, dtype=np.float64)
        predicted = _sigmoid(self.slope * values + self.intercept)
        return float(predicted) if predicted.ndim == 0 else predicted

    def to_dict(self) -> dict[str, float | str]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "target": self.target,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QualityCalibrator":
        return cls(
            slope=float(data["slope"]),
            intercept=float(data["intercept"]),
            target=str(data["target"]),
            version=str(data.get("version", "platt-v1")),
        )


def fit_quality_calibrator(
    qualities: list[float] | np.ndarray,
    correct: list[bool] | np.ndarray,
    *,
    target: str,
    iterations: int = 40,
    l2: float = 1e-4,
) -> QualityCalibrator:
    """Fit one-dimensional Platt scaling with damped Newton updates."""
    x = np.asarray(qualities, dtype=np.float64).reshape(-1)
    y = np.asarray(correct, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 4 or not ({0.0, 1.0} <= set(np.unique(y))):
        raise ValueError("calibration requires matching scores with both outcomes")
    design = np.column_stack((x, np.ones_like(x)))
    params = np.array([1.0, np.log((y.mean() + 1e-3) / (1.0 - y.mean() + 1e-3))])
    for _ in range(max(1, iterations)):
        pred = _sigmoid(design @ params)
        gradient = design.T @ (pred - y) + l2 * params
        weights = np.clip(pred * (1.0 - pred), 1e-6, None)
        hessian = design.T @ (design * weights[:, None]) + l2 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        params -= step
        if float(np.linalg.norm(step)) < 1e-8:
            break
    return QualityCalibrator(float(params[0]), float(params[1]), target=target)


def calibration_metrics(
    probabilities: list[float] | np.ndarray,
    correct: list[bool] | np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float]:
    pred = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0, 1.0)
    truth = np.asarray(correct, dtype=np.float64).reshape(-1)
    if len(pred) != len(truth) or not len(pred):
        raise ValueError("metrics require matching non-empty arrays")
    brier = float(np.mean((pred - truth) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    for index in range(len(edges) - 1):
        if index == len(edges) - 2:
            selected = (pred >= edges[index]) & (pred <= edges[index + 1])
        else:
            selected = (pred >= edges[index]) & (pred < edges[index + 1])
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(pred[selected].mean()) - float(truth[selected].mean())
            )
    return {"brier": brier, "ece": float(ece)}
