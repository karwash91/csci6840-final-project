from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .features import SupervisedFrame, build_supervised_frame
from .metrics import percent_improvement, regression_metrics
from .models import make_model
from .splits import Partition, chronological_partitions


def run_experiments(
    hourly: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    predictions_path = output_path / "predictions"
    models_path = output_path / "models"
    predictions_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for horizon in config.horizons_hours:
        supervised_by_input = {
            "temperature_only": build_supervised_frame(
                hourly,
                (config.target_column,),
                config.target_column,
                horizon,
                config.history_hours,
            ),
            "multivariate": build_supervised_frame(
                hourly,
                config.multivariate_columns,
                config.target_column,
                horizon,
                config.history_hours,
            ),
        }
        common_index = supervised_by_input[
            "temperature_only"
        ].features.index.intersection(
            supervised_by_input["multivariate"].features.index
        )
        supervised_by_input = {
            name: _restrict_supervised(supervised, common_index)
            for name, supervised in supervised_by_input.items()
        }
        baseline_test = chronological_partitions(
            supervised_by_input["temperature_only"],
            config.train_end,
            config.validation_end,
            config.test_end,
        )["test"]
        baseline_prediction = (
            hourly[config.target_column]
            .reindex(baseline_test.features.index)
            .to_numpy()
        )
        baseline_metrics = regression_metrics(
            baseline_test.target.to_numpy(), baseline_prediction
        )
        _save_predictions(
            predictions_path / f"h{horizon:02d}_persistence.csv",
            baseline_test,
            baseline_prediction,
        )
        rows.append(
            {
                "horizon_hours": horizon,
                "model": "persistence",
                "input_set": "temperature_only",
                **baseline_metrics,
                "mae_improvement_over_persistence_pct": 0.0,
                "rmse_improvement_over_persistence_pct": 0.0,
            }
        )
        for input_name, supervised in supervised_by_input.items():
            partitions = chronological_partitions(
                supervised, config.train_end, config.validation_end, config.test_end
            )
            train, test = partitions["train"], partitions["test"]
            if train.target.empty or test.target.empty:
                raise ValueError(
                    f"Empty train or test partition for {horizon}h {input_name}."
                )
            for model_name in ("linear_regression", "random_forest"):
                model = make_model(model_name, config.random_seed, config.random_forest)
                model.fit(train.features, train.target)
                predicted = np.asarray(model.predict(test.features))
                _save_model(
                    model,
                    models_path / f"h{horizon:02d}_{model_name}_{input_name}.joblib",
                )
                result = regression_metrics(test.target.to_numpy(), predicted)
                result["mae_improvement_over_persistence_pct"] = percent_improvement(
                    float(baseline_metrics["mae_c"]), float(result["mae_c"])
                )
                result["rmse_improvement_over_persistence_pct"] = percent_improvement(
                    float(baseline_metrics["rmse_c"]), float(result["rmse_c"])
                )
                rows.append(
                    {
                        "horizon_hours": horizon,
                        "model": model_name,
                        "input_set": input_name,
                        **result,
                    }
                )
                _save_predictions(
                    predictions_path / f"h{horizon:02d}_{model_name}_{input_name}.csv",
                    test,
                    predicted,
                )
    results = pd.DataFrame(rows).sort_values(["horizon_hours", "model", "input_set"])
    results.to_csv(output_path / "test_metrics.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "comparison_rows": len(results),
        "experiment_config": asdict(config),
    }
    (output_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return results


def _restrict_supervised(
    supervised: SupervisedFrame, index: pd.DatetimeIndex
) -> SupervisedFrame:
    return SupervisedFrame(
        features=supervised.features.loc[index],
        target=supervised.target.loc[index],
        target_time=supervised.target_time.loc[index],
    )


def _save_predictions(path: Path, partition: Partition, predicted: np.ndarray) -> None:
    pd.DataFrame(
        {
            "forecast_origin": partition.features.index,
            "target_time": partition.target_time.to_numpy(),
            "actual_temperature_c": partition.target.to_numpy(),
            "predicted_temperature_c": predicted,
        }
    ).to_csv(path, index=False)


def _save_model(model: object, path: Path) -> None:
    import joblib

    joblib.dump(model, path)
