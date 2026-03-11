from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


IDENTIFIER_NAME_PATTERNS = ("id", "_id", "id_", "uuid", "key")


@dataclass
class PreprocessingSummary:
    target_column: str
    selected_feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    identifier_columns: list[str]
    excluded_columns: list[str]


def _looks_like_identifier(column_name: str, series: pd.Series) -> bool:
    normalized = column_name.strip().lower()
    if normalized == "id" or normalized.endswith("_id") or normalized.startswith("id_"):
        return True
    if "customer_id" in normalized or "user_id" in normalized or "member_id" in normalized:
        return True
    if normalized.endswith("uuid") or normalized.endswith("_key") or normalized == "key":
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    if non_null.nunique(dropna=True) != len(non_null):
        return False

    return any(token in normalized for token in IDENTIFIER_NAME_PATTERNS)


def _detect_datetime_columns(features: pd.DataFrame) -> list[str]:
    datetime_columns: list[str] = []
    for column in features.columns:
        series = features[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        parsed = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed")
        if parsed.notna().all():
            datetime_columns.append(column)

    return datetime_columns


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> tuple[ColumnTransformer, pd.DataFrame, PreprocessingSummary]:
    if target_column not in df.columns:
        raise ValueError(f"타깃 컬럼을 찾을 수 없습니다: {target_column}")

    if feature_columns is None:
        features = df.drop(columns=[target_column]).copy()
        identifier_columns = [
            column for column in features.columns if _looks_like_identifier(column, features[column])
        ]
        if identifier_columns:
            features = features.drop(columns=identifier_columns)
    else:
        missing_columns = [column for column in feature_columns if column not in df.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"선택한 독립변수 컬럼을 찾을 수 없습니다: {missing}")

        if target_column in feature_columns:
            feature_columns = [column for column in feature_columns if column != target_column]

        if not feature_columns:
            raise ValueError("타깃을 제외하고 나면 사용할 독립변수가 없습니다.")

        features = df[feature_columns].copy()
        identifier_columns = []

    datetime_columns = _detect_datetime_columns(features)
    numeric_columns = [
        column
        for column in features.select_dtypes(include="number").columns.tolist()
        if column not in datetime_columns
    ]
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns and column not in datetime_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    summary = PreprocessingSummary(
        target_column=target_column,
        selected_feature_columns=features.columns.tolist(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        identifier_columns=identifier_columns,
        excluded_columns=[target_column, *identifier_columns, *datetime_columns],
    )
    return preprocessor, features, summary


def build_preprocessing_summary_markdown(summary: PreprocessingSummary) -> str:
    lines: list[str] = []
    lines.append("# 전처리 요약")
    lines.append("")
    lines.append(f"- 타깃 컬럼: {summary.target_column}")
    lines.append(f"- 선택된 독립변수 수: {len(summary.selected_feature_columns)}")
    lines.append(f"- 수치형 컬럼 수: {len(summary.numeric_columns)}")
    lines.append(f"- 범주형 컬럼 수: {len(summary.categorical_columns)}")
    lines.append(f"- 날짜형 컬럼 수: {len(summary.datetime_columns)}")
    lines.append("")
    lines.append("## 컬럼 분류")
    lines.append("")
    lines.append(f"- 선택된 독립변수: {', '.join(summary.selected_feature_columns) if summary.selected_feature_columns else '없음'}")
    lines.append(f"- 수치형 컬럼: {', '.join(summary.numeric_columns) if summary.numeric_columns else '없음'}")
    lines.append(f"- 범주형 컬럼: {', '.join(summary.categorical_columns) if summary.categorical_columns else '없음'}")
    lines.append(f"- 날짜형 컬럼: {', '.join(summary.datetime_columns) if summary.datetime_columns else '없음'}")
    lines.append(f"- 식별자 컬럼 자동 제외: {', '.join(summary.identifier_columns) if summary.identifier_columns else '없음'}")
    lines.append("")
    lines.append("## 적용 규칙")
    lines.append("")
    lines.append("- 수치형 결측치: 평균값 대체")
    lines.append("- 범주형 결측치: 최빈값 대체")
    lines.append("- 범주형 인코딩: OneHotEncoder")
    lines.append("- 타깃 컬럼: 전처리 대상에서 제외")
    if summary.identifier_columns:
        lines.append("- 식별자 컬럼: 기본적으로 모델 입력에서 제외")
    if summary.datetime_columns:
        lines.append("- 날짜형 컬럼: 자동 feature engineering 없이 제외")
    lines.append("")
    return "\n".join(lines)


def save_preprocessing_summary(summary: PreprocessingSummary, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "preprocessing_summary.md"
    summary_path.write_text(build_preprocessing_summary_markdown(summary), encoding="utf-8-sig")
    return summary_path
