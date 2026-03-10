from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.modeling import ModelResult
from app.profiling import ProfileResult


def _table_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=index)


def build_markdown_report(
    source_name: str,
    profile: ProfileResult,
    histogram_paths: list[Path],
    correlation_path: Path | None,
    model_result: ModelResult | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Data Analysis Report: {source_name}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Rows: {profile.row_count}")
    lines.append(f"- Columns: {profile.column_count}")
    lines.append("")
    lines.append("## Preview")
    lines.append("")
    lines.append(_table_to_markdown(profile.preview))
    lines.append("")
    lines.append("## Column Summary")
    lines.append("")
    lines.append(_table_to_markdown(profile.dtypes))
    lines.append("")
    lines.append("## Missing Value Analysis")
    lines.append("")
    lines.append(_table_to_markdown(profile.missing))
    lines.append("")
    lines.append("## Numeric Descriptive Statistics")
    lines.append("")
    lines.append(_table_to_markdown(profile.numeric_summary))
    lines.append("")
    lines.append("## Categorical Descriptive Statistics")
    lines.append("")
    lines.append(_table_to_markdown(profile.categorical_summary))
    lines.append("")
    lines.append("## Visualizations")
    lines.append("")

    if histogram_paths:
        lines.append("### Histograms")
        lines.append("")
        for path in histogram_paths:
            lines.append(f"![{path.stem}]({path.as_posix()})")
        lines.append("")
    else:
        lines.append("_No numeric columns available for histogram generation._")
        lines.append("")

    if correlation_path is not None:
        lines.append("### Correlation Matrix")
        lines.append("")
        lines.append(f"![correlation_matrix]({correlation_path.as_posix()})")
        lines.append("")
        lines.append(_table_to_markdown(profile.correlation.reset_index().rename(columns={"index": "column"})))
        lines.append("")
    else:
        lines.append("_Not enough numeric columns to generate a correlation matrix._")
        lines.append("")

    lines.append("## Model Results")
    lines.append("")
    if model_result is None:
        lines.append("_Model automation was not run. Pass `--target` to enable baseline training._")
        lines.append("")
    else:
        lines.append(f"- Target: {model_result.target}")
        lines.append(f"- Problem type: {model_result.problem_type}")
        lines.append(f"- Train rows: {model_result.train_rows}")
        lines.append(f"- Validation rows: {model_result.validation_rows}")
        lines.append(f"- Best model: {model_result.best_model_name}")
        lines.append("")
        lines.append(_table_to_markdown(model_result.metrics))
        lines.append("")

    return "\n".join(lines)


def save_markdown_report(content: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
