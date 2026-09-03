from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

# Force a non-GUI backend so chart generation works in headless CLI environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Axis labels are raw column names, so a CJK-capable font must be selected explicitly:
# matplotlib defaults to DejaVu Sans, which renders Hangul as empty boxes on every OS.
# Linux deploy targets additionally need the font installed (see packages.txt).
KOREAN_FONT_CANDIDATES = (
    "Malgun Gothic",     # Windows
    "AppleGothic",       # macOS
    "NanumGothic",       # Linux, via packages.txt
    "NanumBarunGothic",
    "Noto Sans CJK KR",
)


def _find_installed_korean_font() -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in KOREAN_FONT_CANDIDATES if name in available), None)


def _register_nanum_font_files() -> None:
    """Register font files matplotlib missed because its cache predates the install."""
    for font_path in font_manager.findSystemFonts():
        if "nanum" in Path(font_path).name.lower():
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                continue


def configure_chart_fonts() -> str | None:
    """Select the first available Korean font. Returns its name, or None if none is installed."""
    # A minus sign has no glyph in most Korean fonts, so render it with the ASCII hyphen.
    plt.rcParams["axes.unicode_minus"] = False

    selected = _find_installed_korean_font()
    if selected is None:
        # On a deploy image, apt may install the font after matplotlib cached its font
        # list, which leaves the font on disk but invisible to the name lookup above.
        _register_nanum_font_files()
        selected = _find_installed_korean_font()

    if selected is not None:
        # Prepend rather than replace the family, so glyphs the Korean font lacks
        # still fall back to matplotlib's bundled default.
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [selected, *plt.rcParams["font.sans-serif"]]
    return selected


SELECTED_CHART_FONT = configure_chart_fonts()


def generate_histograms(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    saved_files: list[Path] = []

    for column in numeric_columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Histogram: {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        fig.tight_layout()

        file_path = output_path / f"histogram_{column}.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_files.append(file_path)

    return saved_files


def generate_boxplots(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    saved_files: list[Path] = []

    for column in numeric_columns:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.boxplot(x=df[column].dropna(), ax=ax)
        ax.set_title(f"Boxplot: {column}")
        ax.set_xlabel(column)
        fig.tight_layout()

        file_path = output_path / f"boxplot_{column}.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_files.append(file_path)

    return saved_files


def generate_correlation_heatmap(correlation_df: pd.DataFrame, output_dir: str | Path) -> Path | None:
    if correlation_df.empty:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_df, annot=True, cmap="Blues", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()

    file_path = output_path / "correlation_matrix.png"
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return file_path
