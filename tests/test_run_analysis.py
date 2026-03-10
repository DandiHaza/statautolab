from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from run_analysis import build_output_dir, parse_args


class RunAnalysisTests(unittest.TestCase):
    def test_build_output_dir_uses_date_and_time_structure(self) -> None:
        output_dir = build_output_dir("outputs", datetime(2026, 3, 10, 11, 30, 45))
        self.assertEqual(output_dir, Path("outputs") / "20260310" / "113045")

    def test_parse_args_accepts_input_option(self) -> None:
        args = parse_args(["--input", "data/sample.csv"])
        self.assertEqual(args.input_file, "data/sample.csv")

    def test_parse_args_accepts_report_format_and_task_type(self) -> None:
        args = parse_args(
            ["--input", "data/sample.csv", "--target", "buy", "--report-format", "html", "--task-type", "classification"]
        )
        self.assertEqual(args.report_format, "html")
        self.assertEqual(args.task_type, "classification")

    def test_parse_args_accepts_config_option(self) -> None:
        args = parse_args(["--config", "configs/default.yaml"])
        self.assertEqual(args.config, "configs/default.yaml")

    def test_parse_args_accepts_eval_options(self) -> None:
        args = parse_args(["--input", "data/sample.csv", "--eval-method", "cv", "--cv-folds", "4"])
        self.assertEqual(args.eval_method, "cv")
        self.assertEqual(args.cv_folds, 4)
