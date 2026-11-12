# CSCI 6840 Final Project

Short-term temperature forecasting with the Jena Climate dataset.

The project tests whether historical weather variables improve direct 1-hour,
6-hour, and 24-hour temperature forecasts beyond temperature history alone.
For each horizon, the experiment compares a persistence baseline with linear
regression and random forest models using temperature-only and multivariate
inputs. The final test table therefore contains 15 matched comparisons.

## Current status

This repository contains the initial reproducible experiment scaffold. The
chronological split in `config/experiment.json` is a proposed design and must
be confirmed by both team members before final test evaluation.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/download_data.py
jena-forecast audit --input data/raw/jena_climate_2009_2016.csv
jena-forecast preprocess --input data/raw/jena_climate_2009_2016.csv
jena-forecast tune --input data/processed/jena_hourly.csv
jena-forecast run \
  --input data/processed/jena_hourly.csv \
  --config outputs/tuning/<timestamp>/frozen_config.json
```

The tune command compares the configured history lengths and random-forest
settings using validation data, then writes `validation_scores.csv` and a
frozen configuration. The run command refuses an unfrozen configuration and
writes metrics, timestamped predictions, models, and a run manifest to a unique
timestamped directory beneath `outputs/runs/`. Do not edit those outputs by
hand. Preserve the exact directory used for the presentation recording.

## Experiment design

- Target: `T (degC)` at `t + 1h`, `t + 6h`, and `t + 24h`.
- Forecast origin: an hourly timestamp `t` containing only observations
  available by that time.
- History: the current completed hour and the preceding 23 hours.
- Input sets: temperature-only and all configured historical weather columns.
- Models: persistence, linear regression, and random forest.
- Selection: choose settings using validation data, then freeze the config.
- Final metrics: MAE and RMSE in degrees Celsius, R squared, sample count, and
  percent improvement in MAE and RMSE over persistence.
- Leakage control: rows are assigned to a partition using the target timestamp,
  and lag values are matched by timestamp rather than by row position.
- Persistence control: comparisons require an observed temperature at the
  forecast origin, and metric calculation rejects nonfinite values.
- Wind direction: hourly values use a circular mean so directions around zero
  degrees are not incorrectly averaged to 180 degrees.

The proposed split is training through 2014, validation during 2015, and test
during 2016. Confirm the loaded dataset range and split with the team before
freezing the experiment.

## Repository layout

```text
config/                 Experiment definition and provisional split dates
data/raw/               Downloaded source data, ignored by Git
data/processed/         Reproducible hourly data, ignored by Git
outputs/                Metrics, predictions, audits, and run manifests
reports/figures/        Selected presentation-ready figures
scripts/                Dataset download helper
src/jena_forecasting/   Data, feature, model, evaluation, and CLI code
tests/                  Leakage and preprocessing regression tests
docs/                   Experiment decisions and open review items
```

## Reproducibility rules

1. Commit configuration changes before a final run.
2. Never tune on the test partition.
3. Keep the same forecast origins and targets when comparing models.
4. Keep raw and generated data out of Git; save a manifest with every run. The
   manifest records the commit, dirty-tree state, input SHA-256, package
   versions, and partition counts and date ranges.
5. Trace every slide result to a committed config and a saved output file.

Run checks with:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
```

## Data source

The download helper retrieves the public `jena_climate_2009_2016.csv.zip`
archive distributed with the TensorFlow time-series tutorial. Verify the file,
date range, columns, row count, missing values, and invalid readings with the
audit command before modeling.
