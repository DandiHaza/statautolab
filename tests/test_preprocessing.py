from __future__ import annotations

import unittest

import pandas as pd

from app.preprocessing import build_preprocessing_pipeline, build_preprocessing_summary_markdown


class PreprocessingTests(unittest.TestCase):
    def test_build_preprocessing_pipeline_excludes_target_and_splits_columns(self) -> None:
        df = pd.DataFrame(
            {
                "num_col": [1.0, None, 3.0],
                "cat_col": ["a", None, "b"],
                "date_col": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"]),
                "target": [0, 1, 0],
            }
        )

        preprocessor, features, summary = build_preprocessing_pipeline(df, "target")

        self.assertNotIn("target", features.columns)
        self.assertEqual(summary.numeric_columns, ["num_col"])
        self.assertEqual(summary.categorical_columns, ["cat_col"])
        self.assertEqual(summary.datetime_columns, ["date_col"])
        self.assertEqual(len(preprocessor.transformers), 2)

    def test_preprocessing_summary_markdown_contains_column_groups(self) -> None:
        df = pd.DataFrame(
            {
                "num_col": [1.0, 2.0],
                "cat_col": ["a", "b"],
                "target": [1, 0],
            }
        )
        _, _, summary = build_preprocessing_pipeline(df, "target")

        content = build_preprocessing_summary_markdown(summary)
        self.assertIn("수치형 컬럼", content)
        self.assertIn("범주형 컬럼", content)
