from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

