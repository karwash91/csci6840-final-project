# Experiment Design Review

This file translates the November 12 presentation action plan into repository
decisions. It distinguishes implemented safeguards from choices the team must
confirm before the final test run.

## Implemented in the bootstrap

- Direct forecasts for 1-hour, 6-hour, and 24-hour horizons.
- Persistence, linear regression, and random forest comparisons.
- Temperature-only and multivariate feature sets.
- Twenty-four hours of timestamp-matched history, including the forecast
  origin hour.
- Partition assignment by target timestamp so a target cannot cross a train,
  validation, or test boundary.
- Training-only imputation and scaling through scikit-learn pipelines.
- Matching test origins across every model and input set at a given horizon.
- Timestamped prediction files, fitted models, test metrics, and a run manifest
  containing the exact configuration and software environment.

## Provisional choices to confirm

- [ ] Approve training through December 31, 2014, validation during 2015, and
  testing during 2016.
- [ ] Approve right-labeled hourly means from six completed 10-minute readings.
- [ ] Approve treating numeric readings at or below `-999` as sensor sentinels.
- [ ] Approve the 14 multivariate columns in `config/experiment.json` or remove
  variables that are redundant, unreliable, or unavailable at forecast time.
- [ ] Choose random forest settings using validation results, commit the frozen
  configuration, and do not revisit settings after viewing test results.
- [ ] Confirm Kareem and John’s work division and the presentation playback date.

## Required run evidence

The final presentation run should preserve:

1. The raw-data audit with source filename, dates, columns, row count, duplicate
   count, missing values, and sentinel counts.
2. The processed hourly CSV and the cleaning and aggregation rules used to
   create it.
3. Fifteen test-metric rows: one persistence result and four learned-model
   results for each horizon.
4. Timestamped actual and predicted temperatures for every test comparison.
5. The fitted models, exact configuration, run manifest, figures, and commit ID
   used in the recorded demonstration and slides.

## Review finding

The action plan is internally consistent and visually clean. Its main project
risk is not presentation formatting; it is running final evaluation before the
split, cleaning rules, feature list, and validation-selected model settings are
frozen. Resolve the provisional choices above before treating any test score as
a final result.
