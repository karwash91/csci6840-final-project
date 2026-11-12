from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .features import SupervisedFrame, build_supervised_frame
from .metrics import percent_improvement, regression_metrics
from .models import make_model
from .splits import Partition, chronological_partitions

INPUT_NAMES = ("temperature_only", "multivariate")
MODEL_NAMES = ("linear_regression", "random_forest")


def tune_on_validation(
    hourly: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, ExperimentConfig]:
    """Select history and forest settings using validation data only."""
    output_path = _prepare_output_directory(output_dir)
    rows: list[dict[str, object]] = []
    selections: dict[str, dict[str, object]] = {}

    for horizon in config.horizons_hours:
        candidates: list[dict[str, object]] = []
        for input_name in INPUT_NAMES:
            columns = _columns_for_input(config, input_name)
            for history in config.history_candidates:
                supervised = build_supervised_frame(
                    hourly, columns, config.target_column, horizon, history
                )
                candidates.append(
                    {
                        "model": "linear_regression",
                        "input_set": input_name,
                        "history_hours": history,
                        "forest_index": None,
                        "forest_settings": config.random_forest,
                        "supervised": supervised,
                    }
                )
                for forest_index, forest_settings in enumerate(
                    config.forest_candidates
                ):
                    candidates.append(
                        {
                            "model": "random_forest",
                            "input_set": input_name,
                            "history_hours": history,
                            "forest_index": forest_index,
                            "forest_settings": forest_settings,
                            "supervised": supervised,
                        }
                    )

        common_index = _common_observed_index(
            hourly,
            config.target_column,
            [candidate["supervised"] for candidate in candidates],
        )
        baseline_supervised = _restrict_supervised(
            candidates[0]["supervised"], common_index
        )
        baseline_validation = chronological_partitions(
            baseline_supervised,
            config.train_end,
            config.validation_end,
            config.test_end,
        )["validation"]
        baseline_prediction = (
            hourly[config.target_column]
            .reindex(baseline_validation.features.index)
            .to_numpy()
        )
        baseline_metrics = regression_metrics(
            baseline_validation.target.to_numpy(), baseline_prediction
        )
        rows.append(
            {
                "horizon_hours": horizon,
                "model": "persistence",
                "input_set": "temperature_only",
                "history_hours": 1,
                "forest_candidate": None,
                **baseline_metrics,
            }
        )

        for candidate in candidates:
            supervised = _restrict_supervised(candidate["supervised"], common_index)
            partitions = chronological_partitions(
                supervised, config.train_end, config.validation_end, config.test_end
            )
            train, validation = partitions["train"], partitions["validation"]
            if train.target.empty or validation.target.empty:
                raise ValueError(
                    f"Empty train or validation partition for {horizon}h "
                    f"{candidate['model']} {candidate['input_set']}."
                )
            model = make_model(
                str(candidate["model"]),
                config.random_seed,
                dict(candidate["forest_settings"]),
            )
            model.fit(train.features, train.target)
            predicted = np.asarray(model.predict(validation.features))
            metrics = regression_metrics(validation.target.to_numpy(), predicted)
            rows.append(
                {
                    "horizon_hours": horizon,
                    "model": candidate["model"],
                    "input_set": candidate["input_set"],
                    "history_hours": candidate["history_hours"],
                    "forest_candidate": candidate["forest_index"],
                    **metrics,
                }
            )

    scores = pd.DataFrame(rows).sort_values(
        ["horizon_hours", "model", "input_set", "history_hours", "forest_candidate"],
        na_position="first",
    )
    learned = scores[scores["model"].isin(MODEL_NAMES)]
    for (horizon, model_name, input_name), group in learned.groupby(
        ["horizon_hours", "model", "input_set"], sort=True
    ):
        best = group.sort_values(["rmse_c", "mae_c", "history_hours"]).iloc[0]
        selection: dict[str, object] = {"history_hours": int(best["history_hours"])}
        if model_name == "random_forest":
            forest_index = int(best["forest_candidate"])
            selection["random_forest"] = dict(config.forest_candidates[forest_index])
        selections[
            config.selection_key(int(horizon), str(model_name), str(input_name))
        ] = selection

    frozen = config.with_selections(selections)
    scores.to_csv(output_path / "validation_scores.csv", index=False)
    (output_path / "frozen_config.json").write_text(
        json.dumps(frozen.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return scores, frozen


def run_experiments(
    hourly: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: str | Path,
    input_path: str | Path | None = None,
) -> pd.DataFrame:
    output_path = _prepare_output_directory(output_dir)
    predictions_path = output_path / "predictions"
    models_path = output_path / "models"
    predictions_path.mkdir()
    models_path.mkdir()
    rows: list[dict[str, object]] = []
    partition_summaries: dict[str, object] = {}

    for horizon in config.horizons_hours:
        supervised_by_model: dict[tuple[str, str], SupervisedFrame] = {}
        settings_by_model: dict[tuple[str, str], dict[str, int | None]] = {}
        for model_name in MODEL_NAMES:
            for input_name in INPUT_NAMES:
                history, forest_settings = config.selection_for(
                    horizon, model_name, input_name
                )
                supervised_by_model[(model_name, input_name)] = build_supervised_frame(
                    hourly,
                    _columns_for_input(config, input_name),
                    config.target_column,
                    horizon,
                    history,
                )
                settings_by_model[(model_name, input_name)] = forest_settings

        common_index = _common_observed_index(
            hourly, config.target_column, list(supervised_by_model.values())
        )
        supervised_by_model = {
            key: _restrict_supervised(supervised, common_index)
            for key, supervised in supervised_by_model.items()
        }
        reference = supervised_by_model[("linear_regression", "temperature_only")]
        reference_partitions = chronological_partitions(
            reference, config.train_end, config.validation_end, config.test_end
        )
        partition_summaries[f"h{horizon:02d}"] = {
            name: _partition_summary(partition)
            for name, partition in reference_partitions.items()
        }
        baseline_test = reference_partitions["test"]
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

        for (model_name, input_name), supervised in supervised_by_model.items():
            partitions = chronological_partitions(
                supervised, config.train_end, config.validation_end, config.test_end
            )
            train, test = partitions["train"], partitions["test"]
            if train.target.empty or test.target.empty:
                raise ValueError(
                    f"Empty train or test partition for {horizon}h {input_name}."
                )
            model = make_model(
                model_name,
                config.random_seed,
                settings_by_model[(model_name, input_name)],
            )
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
    manifest = _build_manifest(
        config, results, partition_summaries, Path(input_path) if input_path else None
    )
    (output_path / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return results


def _columns_for_input(config: ExperimentConfig, input_name: str) -> tuple[str, ...]:
    if input_name == "temperature_only":
        return (config.target_column,)
    if input_name == "multivariate":
        return config.multivariate_columns
    raise ValueError(f"Unknown input set: {input_name}")


def _common_observed_index(
    hourly: pd.DataFrame,
    target_column: str,
    supervised_frames: list[SupervisedFrame],
) -> pd.DatetimeIndex:
    if not supervised_frames:
        raise ValueError("At least one supervised frame is required.")
    common_index = supervised_frames[0].features.index
    for supervised in supervised_frames[1:]:
        common_index = common_index.intersection(supervised.features.index)
    origin_temperature = pd.to_numeric(
        hourly[target_column].reindex(common_index), errors="coerce"
    )
    return common_index[np.isfinite(origin_temperature.to_numpy())]


def _restrict_supervised(
    supervised: SupervisedFrame, index: pd.DatetimeIndex
) -> SupervisedFrame:
    return SupervisedFrame(
        features=supervised.features.loc[index],
        target=supervised.target.loc[index],
        target_time=supervised.target_time.loc[index],
    )


def _prepare_output_directory(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _partition_summary(partition: Partition) -> dict[str, object]:
    if partition.target.empty:
        return {"count": 0}
    return {
        "count": len(partition.target),
        "forecast_origin_start": partition.features.index.min().isoformat(),
        "forecast_origin_end": partition.features.index.max().isoformat(),
        "target_start": partition.target_time.min().isoformat(),
        "target_end": partition.target_time.max().isoformat(),
    }


def _build_manifest(
    config: ExperimentConfig,
    results: pd.DataFrame,
    partition_summaries: dict[str, object],
    input_path: Path | None,
) -> dict[str, object]:
    import sklearn

    commit, dirty = _git_state()
    input_details: dict[str, object] | None = None
    if input_path is not None:
        input_details = {
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "bytes": input_path.stat().st_size,
        }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "working_tree_dirty": dirty,
        "input_data": input_details,
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "comparison_rows": len(results),
        "partitions": partition_summaries,
        "experiment_config": config.to_dict(),
    }


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
