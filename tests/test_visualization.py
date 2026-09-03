from __future__ import annotations

import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd

from app.visualization import (
    KOREAN_FONT_CANDIDATES,
    configure_chart_fonts,
    generate_boxplots,
    generate_histograms,
)


class ChartFontTests(unittest.TestCase):
    def test_configure_chart_fonts_disables_unicode_minus(self) -> None:
        # Korean fonts have no glyph for U+2212, so the ASCII hyphen must be used instead.
        configure_chart_fonts()

        self.assertFalse(plt.rcParams["axes.unicode_minus"])

    def test_configure_chart_fonts_keeps_default_family_as_fallback(self) -> None:
        selected = configure_chart_fonts()
        if selected is None:
            self.skipTest("이 환경에는 후보 한글 폰트가 설치되어 있지 않습니다.")

        self.assertIn(selected, KOREAN_FONT_CANDIDATES)
        self.assertEqual(selected, plt.rcParams["font.sans-serif"][0])
        self.assertIn("DejaVu Sans", plt.rcParams["font.sans-serif"])

    def test_korean_column_names_render_without_missing_glyphs(self) -> None:
        if configure_chart_fonts() is None:
            self.skipTest("이 환경에는 후보 한글 폰트가 설치되어 있지 않습니다.")

        df = pd.DataFrame({"매출액": [100, 250, -180, 320], "방문 횟수": [1, 5, 3, 7]})

        with TemporaryDirectory() as tmpdir, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_histograms(df, Path(tmpdir))
            generate_boxplots(df, Path(tmpdir))

        missing_glyphs = [record for record in caught if "missing from font" in str(record.message)]

        self.assertEqual(missing_glyphs, [], "한글 컬럼명이 차트에서 글리프 누락으로 깨졌습니다.")
