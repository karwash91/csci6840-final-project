from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

TIMESTAMP_COLUMN = "Date Time"
TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M:%S"
INVALID_READING_FLOOR = -999.0


@dataclass(frozen=True)
class DataAudit:
    source: str
    rows: int
    columns: list[str]
    start: str
    end: str
    duplicate_timestamps: int
    missing_by_column: dict[str, int]
    invalid_sentinel_by_column: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_jena_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"Expected timestamp column {TIMESTAMP_COLUMN!r}.")
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN], format=TIMESTAMP_FORMAT, errors="raise"
    )
    return frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)


def audit_frame(frame: pd.DataFrame, source: str = "") -> DataAudit:
    timestamps = frame[TIMESTAMP_COLUMN]
    numeric = frame.drop(columns=[TIMESTAMP_COLUMN]).select_dtypes(include="number")
    invalid = (numeric <= INVALID_READING_FLOOR).sum()
    return DataAudit(
        source=source,
        rows=len(frame),
        columns=list(frame.columns),
        start=timestamps.min().isoformat(),
        end=timestamps.max().isoformat(),
        duplicate_timestamps=int(timestamps.duplicated().sum()),
        missing_by_column={
            key: int(value) for key, value in frame.isna().sum().items()
        },
        invalid_sentinel_by_column={
            key: int(value) for key, value in invalid.items() if int(value) > 0
        },
    )


def clean_and_resample_hourly(
    frame: pd.DataFrame,
    expected_readings_per_hour: int = 6,
) -> pd.DataFrame:
    """Create right-labeled hourly means using only completed intervals.

    A timestamp such as 01:00 summarizes readings after 00:00 through 01:00.
    Hours with fewer than the expected six 10-minute timestamps are excluded.
    Sentinel values at or below -999 are treated as missing before aggregation.
    """
    if frame[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate timestamps must be resolved before hourly aggregation."
        )
    indexed = frame.set_index(TIMESTAMP_COLUMN).sort_index()
    numeric = indexed.select_dtypes(include="number").copy()
    numeric = numeric.mask(numeric <= INVALID_READING_FLOOR)
    resampler = numeric.resample("1h", closed="right", label="right")
    hourly = resampler.mean()
    timestamp_counts = indexed.resample("1h", closed="right", label="right").size()
    hourly = hourly.loc[timestamp_counts >= expected_readings_per_hour]
    hourly.index.name = "timestamp"
    return hourly


def load_hourly_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return frame.sort_index()
