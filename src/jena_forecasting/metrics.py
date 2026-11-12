from __future__ import annotations

import math

import numpy as np


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Metric inputs must have the same nonzero shape.")
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(np.square(errors))))
    denominator = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = (
        float("nan")
        if denominator == 0
        else float(1 - np.sum(np.square(errors)) / denominator)
    )
    return {"mae_c": mae, "rmse_c": rmse, "r2": r2, "n": int(actual.size)}


def percent_improvement(baseline: float, candidate: float) -> float:
    return float("nan") if baseline == 0 else 100.0 * (baseline - candidate) / baseline
