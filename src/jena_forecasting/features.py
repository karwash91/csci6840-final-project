from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SupervisedFrame:
    features: pd.DataFrame
    target: pd.Series
    target_time: pd.Series


def build_supervised_frame(
    hourly: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str,
    horizon_hours: int,
    history_hours: int,
) -> SupervisedFrame:
    """Build exact timestamp-matched lag features and a direct forecast target."""
    required = set(feature_columns) | {target_column}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not hourly.index.is_unique:
        raise ValueError("Hourly timestamps must be unique.")
    hourly = hourly.sort_index()
    origin_index = hourly.index
    features: dict[str, pd.Series] = {}
    complete_history = pd.Series(True, index=origin_index)
    for column in feature_columns:
        source = hourly[column]
        for lag in range(history_hours):
            lookup = origin_index - pd.Timedelta(hours=lag)
            complete_history &= lookup.isin(origin_index)
            features[f"{column}__lag_{lag:02d}h"] = pd.Series(
                source.reindex(lookup).to_numpy(), index=origin_index
            )
    target_time = pd.Series(
        origin_index + pd.Timedelta(hours=horizon_hours),
        index=origin_index,
        name="target_time",
    )
    target = pd.Series(
        hourly[target_column].reindex(pd.DatetimeIndex(target_time)).to_numpy(),
        index=origin_index,
        name="target",
    )
    feature_frame = pd.DataFrame(features, index=origin_index)
    valid = complete_history & target.notna()
    return SupervisedFrame(
        features=feature_frame.loc[valid],
        target=target.loc[valid],
        target_time=target_time.loc[valid],
    )
