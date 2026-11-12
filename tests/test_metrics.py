from __future__ import annotations

import unittest

import numpy as np

from jena_forecasting.metrics import regression_metrics


class MetricTests(unittest.TestCase):
    def test_nonfinite_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            regression_metrics(np.array([1.0, 2.0]), np.array([1.0, np.nan]))


if __name__ == "__main__":
    unittest.main()
