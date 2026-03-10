from __future__ import annotations

import unittest

from app.config import load_config_file, resolve_settings


class ConfigTests(unittest.TestCase):
    def test_load_config_file_reads_yaml(self) -> None:
        config = load_config_file("configs/default.yaml")
        self.assertEqual(config["input_path"], "data/sample.csv")
        self.assertEqual(config["report_format"], "md")

    def test_cli_values_override_config(self) -> None:
        config = {
            "input_path": "data/sample.csv",
            "target": "buy",
            "output_dir": "outputs",
            "report_format": "md",
            "task_type": "auto",
            "random_state": 42,
            "test_size": 0.2,
        }
        cli = {
            "input_path": None,
            "target": "other_target",
            "output_dir": None,
            "report_format": "html",
            "task_type": None,
            "random_state": 7,
            "test_size": None,
        }

        resolved = resolve_settings(cli, config)

        self.assertEqual(resolved["target"], "other_target")
        self.assertEqual(resolved["report_format"], "html")
        self.assertEqual(resolved["random_state"], 7)
