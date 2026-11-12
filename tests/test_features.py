from __future__ import annotations

import unittest

import pandas as pd

from jena_forecasting.features import build_supervised_frame
from jena_forecasting.splits import chronological_partitions


class FeatureTests(unittest.TestCase):
    def test_gap_does_not_turn_previous_row_into_previous_hour(self) -> None:
        index = pd.DatetimeIndex(
            [
                "2026-01-01 00:00",
                "2026-01-01 01:00",
                "2026-01-01 03:00",
                "2026-01-01 04:00",
                "2026-01-01 05:00",
            ]
        )
        hourly = pd.DataFrame({"T (degC)": [0.0, 1.0, 3.0, 4.0, 5.0]}, index=index)
        supervised = build_supervised_frame(hourly, ["T (degC)"], "T (degC)", 1, 2)
        self.assertEqual(
            list(supervised.features.index), [pd.Timestamp("2026-01-01 04:00")]
        )
        self.assertEqual(supervised.target.iloc[0], 5.0)

    def test_target_timestamp_controls_partition(self) -> None:
        index = pd.date_range("2026-01-01 00:00", periods=6, freq="1h")
        hourly = pd.DataFrame({"T (degC)": range(6)}, index=index)
        supervised = build_supervised_frame(hourly, ["T (degC)"], "T (degC)", 2, 1)
        partitions = chronological_partitions(
            supervised,
            train_end="2026-01-01 02:00",
            validation_end="2026-01-01 04:00",
            test_end="2026-01-01 05:00",
        )
        self.assertEqual(
            list(partitions["train"].target_time), [pd.Timestamp("2026-01-01 02:00")]
        )
        self.assertEqual(
            list(partitions["validation"].target_time),
            [pd.Timestamp("2026-01-01 03:00"), pd.Timestamp("2026-01-01 04:00")],
        )
        self.assertEqual(
            list(partitions["test"].target_time), [pd.Timestamp("2026-01-01 05:00")]
        )


if __name__ == "__main__":
    unittest.main()
