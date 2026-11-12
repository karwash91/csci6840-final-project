from __future__ import annotations

import unittest

import pandas as pd

from jena_forecasting.data import TIMESTAMP_COLUMN, clean_and_resample_hourly


class HourlyAggregationTests(unittest.TestCase):
    def test_right_labeled_completed_hours_and_sentinel_cleanup(self) -> None:
        timestamps = pd.date_range("2026-01-01 00:10", periods=11, freq="10min")
        frame = pd.DataFrame(
            {
                TIMESTAMP_COLUMN: timestamps,
                "T (degC)": list(range(11)),
                "wv (m/s)": [1, 1, 1, 1, 1, -9999, 2, 2, 2, 2, 2],
            }
        )
        hourly = clean_and_resample_hourly(frame)
        self.assertEqual(list(hourly.index), [pd.Timestamp("2026-01-01 01:00")])
        self.assertAlmostEqual(hourly.loc["2026-01-01 01:00", "T (degC)"], 2.5)
        self.assertAlmostEqual(hourly.loc["2026-01-01 01:00", "wv (m/s)"], 1.0)

    def test_duplicate_timestamp_is_rejected(self) -> None:
        frame = pd.DataFrame(
            {TIMESTAMP_COLUMN: [pd.Timestamp("2026-01-01")] * 2, "T (degC)": [1.0, 2.0]}
        )
        with self.assertRaisesRegex(ValueError, "Duplicate timestamps"):
            clean_and_resample_hourly(frame)

    def test_wind_direction_uses_circular_mean(self) -> None:
        timestamps = pd.date_range("2026-01-01 00:10", periods=6, freq="10min")
        frame = pd.DataFrame(
            {
                TIMESTAMP_COLUMN: timestamps,
                "T (degC)": [10.0] * 6,
                "wd (deg)": [359.0, 1.0, 359.0, 1.0, 359.0, 1.0],
            }
        )
        hourly = clean_and_resample_hourly(frame)

        self.assertAlmostEqual(hourly.iloc[0]["wd (deg)"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
