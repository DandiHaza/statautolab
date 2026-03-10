from __future__ import annotations

import unittest

import pandas as pd

from app.profiling import profile_dataset


class ProfilingTests(unittest.TestCase):
    def test_profile_dataset_builds_iqr_outlier_summary(self) -> None:
        df = pd.DataFrame(
            {
                "value": [10, 11, 12, 13, 100],
                "other": [1, 1, 1, 1, 1],
            }
        )

        profile = profile_dataset(df)
        outlier_row = profile.outliers.loc[profile.outliers["column"] == "value"].iloc[0]

        self.assertEqual(int(outlier_row["outlier_count"]), 1)
        self.assertGreater(float(outlier_row["outlier_ratio_pct"]), 0.0)
