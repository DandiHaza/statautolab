"""표 컬럼명 한글화와 텍스트 가공."""

from __future__ import annotations

import pandas as pd


def strip_markdown_images(markdown_text: str) -> str:
    cleaned_lines: list[str] = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("![") and "](" in stripped and stripped.endswith(")"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def localize_profile_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "column": "컬럼명 (Column)",
            "dtype": "데이터 타입 (Dtype)",
            "non_null_count": "비결측 개수 (Non-null Count)",
            "unique_count": "고유값 개수 (Unique Count)",
        }
    )


def localize_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "column": "컬럼명 (Column)",
            "missing_count": "결측치 개수 (Missing Count)",
            "missing_ratio_pct": "결측치 비율 (Missing Ratio %)",
        }
    )


def localize_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "column": "컬럼명 (Column)",
            "outlier_count": "이상치 개수 (Outlier Count)",
            "outlier_ratio_pct": "이상치 비율 (Outlier Ratio %)",
            "lower_bound": "하한 (Lower Bound)",
            "upper_bound": "상한 (Upper Bound)",
        }
    )


def localize_combined_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.rename(
        columns={
            "feature": "변수 (Feature)",
            "coefficient": "계수",
            "std_error": "Std Err",
            "t_value": "t",
            "p_value": "p-value",
            "ci_lower": "0.025",
            "ci_upper": "0.975",
        }
    ).copy()
    if "section" in localized.columns:
        localized = localized.drop(columns=["section"])
    if "변수 (Feature)" in localized.columns:
        localized["변수 (Feature)"] = localized["변수 (Feature)"].replace({"const": "절편 (Intercept)"})
    return localized


def localize_corr_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "feature_1": "변수 1",
            "feature_2": "변수 2",
            "correlation": "상관계수",
        }
    )


def localize_vif_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "feature": "변수",
            "vif": "VIF",
        }
    )


def build_ols_display_text(dashboard_data) -> str | None:
    if dashboard_data.ols_summary_text:
        return f"=== 최종 회귀 분석 결과 ===\n\n{dashboard_data.ols_summary_text}"
    return None
