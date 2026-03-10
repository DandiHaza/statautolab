from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.evaluate import add_classification_auc, evaluate_classification, evaluate_regression
from app.model_selection import detect_problem_type, get_baseline_models
from app.preprocessing import PreprocessingSummary, build_preprocessing_pipeline


@dataclass
class ModelResult:
    target: str
    problem_type: str
    train_rows: int
    validation_rows: int
    metrics: pd.DataFrame
    best_model_name: str
    preprocessing_summary: PreprocessingSummary


def train_and_compare_models(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    task_type: str = "auto",
    random_state: int = 42,
) -> ModelResult:
    if target_column not in df.columns:
        available_columns = ", ".join(df.columns.astype(str).tolist())
        raise ValueError(
            f"타깃 컬럼을 찾을 수 없습니다: {target_column}. "
            f"사용 가능한 컬럼: {available_columns}"
        )

    model_df = df.dropna(subset=[target_column]).copy()
    if model_df.empty:
        raise ValueError("타깃 결측치를 제거한 뒤 사용할 데이터가 없습니다.")

    target = model_df[target_column]
    preprocessor, features, preprocessing_summary = build_preprocessing_pipeline(model_df, target_column)

    if features.empty:
        raise ValueError("타깃 컬럼을 제외하고 남은 입력 피처가 없습니다.")

    if not preprocessing_summary.numeric_columns and not preprocessing_summary.categorical_columns:
        raise ValueError("전처리 후 사용할 수치형/범주형 입력 피처가 없습니다.")

    if task_type == "auto":
        problem_type = detect_problem_type(target)
    else:
        problem_type = task_type
    if problem_type == "classification" and target.nunique() < 2:
        raise ValueError("분류 문제는 최소 2개 이상의 클래스가 필요합니다.")

    if preprocessing_summary.datetime_columns:
        columns_text = ", ".join(preprocessing_summary.datetime_columns)
        print(
            f"경고: 날짜형 컬럼({columns_text})이 감지되었습니다. 자동 feature engineering은 하지 않고 학습에서 제외합니다."
        )

    stratify = None
    if problem_type == "classification":
        min_class_count = int(target.value_counts().min())
        if min_class_count >= 2:
            stratify = target

    X_train, X_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    models = get_baseline_models(problem_type)

    rows: list[dict[str, object]] = []
    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)

        if problem_type == "regression":
            metrics = evaluate_regression(y_valid, predictions)
        else:
            metrics = evaluate_classification(y_valid, predictions)
            metrics = add_classification_auc(metrics, pipeline, X_valid, y_valid)

        rows.append(
            {
                "model": model_name,
                "problem_type": problem_type,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)
    best_model_name = _select_best_model(metrics_df, problem_type)

    return ModelResult(
        target=target_column,
        problem_type=problem_type,
        train_rows=len(X_train),
        validation_rows=len(X_valid),
        metrics=metrics_df,
        best_model_name=best_model_name,
        preprocessing_summary=preprocessing_summary,
    )


def _select_best_model(metrics_df: pd.DataFrame, problem_type: str) -> str:
    if problem_type == "regression":
        ordered = metrics_df.sort_values(["rmse", "mae", "r2"], ascending=[True, True, False])
        return str(ordered.iloc[0]["model"])

    sortable = metrics_df.copy()
    sortable["roc_auc_filled"] = sortable["roc_auc"].fillna(-1.0)
    ordered = sortable.sort_values(["accuracy", "f1", "roc_auc_filled"], ascending=[False, False, False])
    return str(ordered.iloc[0]["model"])


def save_model_results(result: ModelResult, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_path = output_path / "model_comparison.csv"
    result.metrics.to_csv(comparison_path, index=False)

    problem_type_label = "회귀" if result.problem_type == "regression" else "분류"
    summary_lines = [
        "# 모델 성능 요약",
        "",
        f"- 타깃 컬럼: {result.target}",
        f"- 문제 유형: {problem_type_label}",
        f"- 학습 데이터 행 수: {result.train_rows}",
        f"- 검증 데이터 행 수: {result.validation_rows}",
        f"- 최고 성능 모델: {result.best_model_name}",
        "",
        "## 모델 비교",
        "",
        result.metrics.to_markdown(index=False),
        "",
        "## 전처리 요약",
        "",
        f"- 수치형 컬럼: {', '.join(result.preprocessing_summary.numeric_columns) if result.preprocessing_summary.numeric_columns else '없음'}",
        f"- 범주형 컬럼: {', '.join(result.preprocessing_summary.categorical_columns) if result.preprocessing_summary.categorical_columns else '없음'}",
        f"- 날짜형 컬럼: {', '.join(result.preprocessing_summary.datetime_columns) if result.preprocessing_summary.datetime_columns else '없음'}",
        "",
    ]

    summary_path = output_path / "model_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8-sig")
    return comparison_path, summary_path
