from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from app.io import load_dataset


class LoadDatasetTests(unittest.TestCase):
    def test_missing_file_error_is_friendly(self) -> None:
        with self.assertRaises(FileNotFoundError) as context:
            load_dataset("does_not_exist.csv")

        message = str(context.exception)
        self.assertIn("입력 파일을 찾을 수 없습니다", message)
        self.assertIn("python run_analysis.py --input", message)

    def test_unsupported_extension_error_is_friendly(self) -> None:
        tmpdir = Path("tests/.tmp")
        tmpdir.mkdir(parents=True, exist_ok=True)
        path = tmpdir / "sample.txt"
        try:
            path.write_text("hello", encoding="utf-8")

            with self.assertRaises(ValueError) as context:
                load_dataset(path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        self.assertIn("지원하지 않는 파일 형식", str(context.exception))
