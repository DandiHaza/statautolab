from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import pandas as pd

from app.preprocessing import build_preprocessing_pipeline
from app.profiling import profile_dataset
from app.warnings_log import collect_data_warnings, save_warnings_summary


class WarningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path("tests/.tmp_warnings")
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_collect_data_warnings_detects_datetime_missing_and_correlation(self) -> None:
        df = pd.DataFrame(
            {
                "date_col": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13"]),
                "value": [1.0, None, None, None],
                "target": ["yes", "no", "yes", "no"],
            }
        )
        profile = profile_dataset(df)
        _, _, summary = build_preprocessing_pipeline(df, "target")

        warnings = collect_data_warnings(profile, preprocessing_summary=summary)
        codes = {record.code for record in warnings}

        self.assertIn("datetime_columns_excluded", codes)
        self.assertIn("high_missing_ratio_columns", codes)
        self.assertIn("correlation_matrix_not_generated", codes)

    def test_collect_data_warnings_detects_class_imbalance(self) -> None:
        df = pd.DataFrame({"feature": range(10), "target": ["yes"] * 8 + ["no"] * 2})
        profile = profile_dataset(df)

        warnings = collect_data_warnings(
            profile,
            target_series=df["target"],
            problem_type="classification",
        )

        self.assertTrue(any(record.code == "severe_target_class_imbalance" for record in warnings))

    def test_save_warnings_summary_creates_markdown_and_json(self) -> None:
        markdown_path, json_path = save_warnings_summary([], self.tmpdir)
        self.assertTrue(markdown_path.exists())
        self.assertTrue(json_path.exists())
        content = json.loads(json_path.read_text(encoding="utf-8-sig"))
        self.assertEqual(content, [])
