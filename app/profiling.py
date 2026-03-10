from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ProfileResult:
    row_count: int
    column_count: int
    dtypes: pd.DataFrame
    missing: pd.DataFrame
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    preview: pd.DataFrame
    correlation: pd.DataFrame


def _build_dtype_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": df.notna().sum().values,
            "unique_count": df.nunique(dropna=True).values,
        }
    )
    return summary.sort_values("column").reset_index(drop=True)


def _build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_ratio = (missing_count / len(df) * 100) if len(df) else 0
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_count.values,
            "missing_ratio_pct": missing_ratio.values if hasattr(missing_ratio, "values") else [0] * len(df.columns),
        }
    )
    return summary.sort_values(["missing_count", "column"], ascending=[False, True]).reset_index(drop=True)


def _build_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame(columns=["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])

    summary = numeric_df.describe().transpose().reset_index().rename(columns={"index": "column"})
    return summary


def _build_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    categorical_df = df.select_dtypes(exclude="number")
    if categorical_df.empty:
        return pd.DataFrame(columns=["column", "count", "unique", "top", "freq"])

    summary = categorical_df.describe().transpose().reset_index().rename(columns={"index": "column"})
    return summary


def _build_correlation(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def profile_dataset(df: pd.DataFrame) -> ProfileResult:
    return ProfileResult(
        row_count=len(df),
        column_count=len(df.columns),
        dtypes=_build_dtype_summary(df),
        missing=_build_missing_summary(df),
        numeric_summary=_build_numeric_summary(df),
        categorical_summary=_build_categorical_summary(df),
        preview=df.head(10),
        correlation=_build_correlation(df),
    )

