from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import pandas as pd

from app.experiment import append_experiment_log, save_config_snapshot, save_data_summary
from app.profiling import profile_dataset


class ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path("tests/.tmp_experiment")
        self.tmpdir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_config_snapshot_creates_json(self) -> None:
        path = save_config_snapshot({"run_id": "20260310_120000", "target": "buy"}, self.tmpdir)
        self.assertTrue(path.exists())
        content = json.loads(path.read_text(encoding="utf-8-sig"))
        self.assertEqual(content["run_id"], "20260310_120000")

    def test_append_experiment_log_adds_row(self) -> None:
        log_path = append_experiment_log(
            base_output_dir=self.tmpdir,
            run_id="20260310_120000",
            timestamp="2026-03-10T12:00:00",
            input_file="data/sample.csv",
            target="buy",
            task_type="auto",
            model_result=None,
            output_path=self.tmpdir / "20260310" / "120000",
        )
        self.assertTrue(log_path.exists())
        df = pd.read_csv(log_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["run_id"], "20260310_120000")

    def test_save_data_summary_creates_json(self) -> None:
        profile = profile_dataset(pd.DataFrame({"value": [1, 2, 3]}))
        path = save_data_summary(profile, "data/sample.csv", self.tmpdir)
        self.assertTrue(path.exists())
        content = json.loads(path.read_text(encoding="utf-8-sig"))
        self.assertEqual(content["row_count"], 3)
