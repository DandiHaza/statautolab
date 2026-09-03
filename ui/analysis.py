"""화면에 필요한 계산. 무거운 작업은 Streamlit 캐시로 재사용한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.io import load_dataset
from app.profiling import profile_dataset
from ui.constants import NO_TARGET


@st.cache_data(show_spinner=False)
def load_preview_dataset(file_path: str) -> pd.DataFrame:
    return load_dataset(file_path)


@st.cache_data(show_spinner=False)
def build_profile(file_path: str):
    df = load_dataset(file_path)
    return profile_dataset(df)


@st.cache_data(show_spinner=False)
def analyze_feature_selection(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> dict[str, object]:
    numeric_df = df[list(feature_columns)].select_dtypes(include="number").copy()
    result = {
        "high_corr_pairs": pd.DataFrame(columns=["feature_1", "feature_2", "correlation"]),
        "vif_table": pd.DataFrame(columns=["feature", "vif"]),
        "max_vif": None,
    }
    if numeric_df.shape[1] < 2:
        return result

    corr = numeric_df.corr(numeric_only=True)
    pair_rows: list[dict[str, object]] = []
    columns = corr.columns.tolist()
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            corr_value = float(corr.loc[left, right])
            if abs(corr_value) >= 0.7:
                pair_rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": corr_value,
                        "abs_correlation": abs(corr_value),
                    }
                )
    if pair_rows:
        result["high_corr_pairs"] = (
            pd.DataFrame(pair_rows)
            .sort_values("abs_correlation", ascending=False)
            .drop(columns=["abs_correlation"])
            .reset_index(drop=True)
        )

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:
        return result

    filled_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))
    vif_rows: list[dict[str, object]] = []
    values = filled_df.astype(float).values
    for idx, column in enumerate(filled_df.columns):
        vif_rows.append(
            {
                "feature": column,
                "vif": float(variance_inflation_factor(values, idx)),
            }
        )
    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)
    result["vif_table"] = vif_df
    if not vif_df.empty:
        result["max_vif"] = float(vif_df["vif"].max())
    return result


def collect_download_files(output_dir: Path) -> list[Path]:
    return [path for path in sorted(output_dir.rglob("*")) if path.is_file()]


def build_recommended_removals(
    high_corr_pairs: pd.DataFrame,
    vif_table: pd.DataFrame,
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    vif_lookup = {}
    if not vif_table.empty:
        vif_lookup = {str(row["feature"]): float(row["vif"]) for _, row in vif_table.iterrows()}

    seen_features: set[str] = set()
    for _, row in high_corr_pairs.head(3).iterrows():
        left = str(row["feature_1"])
        right = str(row["feature_2"])
        corr_value = float(row["correlation"])
        left_vif = vif_lookup.get(left, 0.0)
        right_vif = vif_lookup.get(right, 0.0)
        remove_feature = left if left_vif >= right_vif else right
        keep_feature = right if remove_feature == left else left
        if remove_feature in seen_features:
            continue
        seen_features.add(remove_feature)
        recommendations.append(
            {
                "feature": remove_feature,
                "reason": f"{keep_feature}와 상관이 높음 (corr={corr_value:.2f})",
            }
        )

    for _, row in vif_table.iterrows():
        feature = str(row["feature"])
        vif_value = float(row["vif"])
        if vif_value < 5 or feature in seen_features:
            continue
        seen_features.add(feature)
        recommendations.append(
            {
                "feature": feature,
                "reason": f"VIF가 높음 ({vif_value:.2f})",
            }
        )
        if len(recommendations) >= 5:
            break

    return recommendations


def infer_problem_type(df: pd.DataFrame, target_value: str, task_type: str) -> str | None:
    if target_value == NO_TARGET:
        return None
    if task_type != "auto":
        return task_type
    return "regression" if pd.api.types.is_numeric_dtype(df[target_value]) else "classification"


def get_available_models(problem_type: str | None, app_mode: str) -> list[str]:
    if problem_type == "regression":
        if app_mode == "analysis":
            return ["LinearRegression"]
        return ["LinearRegression", "RandomForestRegressor"]
    if problem_type == "classification":
        if app_mode == "analysis":
            return ["LogisticRegression"]
        return ["LogisticRegression", "RandomForestClassifier"]
    return []
