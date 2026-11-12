from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from jena_forecasting.config import ExperimentConfig
from jena_forecasting.pipeline import run_experiments


class _FakeModel:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> "_FakeModel":
        self.fallback = float(target.mean())
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features.iloc[:, 0].fillna(self.fallback).to_numpy()


class PipelineTests(unittest.TestCase):
    def test_run_produces_fifteen_matched_comparisons(self) -> None:
        index = pd.date_range("2026-01-01", periods=24 * 12, freq="1h")
        values = np.sin(np.arange(len(index)) / 24)
        hourly = pd.DataFrame(
            {"T (degC)": values, "rh (%)": 50 + values},
            index=index,
        )
        config = ExperimentConfig(
            target_column="T (degC)",
            horizons_hours=(1, 6, 24),
            history_hours=2,
            train_end="2026-01-05 23:00",
            validation_end="2026-01-08 23:00",
            test_end="2026-01-12 00:00",
            random_seed=6840,
            random_forest={},
            multivariate_columns=("T (degC)", "rh (%)"),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch(
                    "jena_forecasting.pipeline.make_model", return_value=_FakeModel()
                ),
                patch("jena_forecasting.pipeline._save_model"),
            ):
                results = run_experiments(hourly, config, Path(temporary_directory))

        self.assertEqual(len(results), 15)
        for _, horizon_results in results.groupby("horizon_hours"):
            self.assertEqual(len(horizon_results), 5)
            self.assertEqual(horizon_results["n"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
