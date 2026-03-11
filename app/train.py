from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from app.evaluate import add_classification_auc, evaluate_classification, evaluate_regression
from app.model_selection import detect_problem_type, get_baseline_models
from app.preprocessing import PreprocessingSummary, build_preprocessing_pipeline
from app.warnings_log import WarningRecord


@dataclass
class ModelResult:
    target: str
    problem_type: str
    train_rows: int
    validation_rows: int
    eval_method: str
    cv_folds: int
    metrics: pd.DataFrame
    best_model_name: str
    preprocessing_summary: PreprocessingSummary
    warnings: list[WarningRecord]
    best_model_pipeline: Pipeline


def train_and_compare_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str] | None = None,
    selected_model: str | None = None,
    test_size: float = 0.2,
    task_type: str = "auto",
    random_state: int = 42,
    eval_method: str = "holdout",
    cv_folds: int = 5,
) -> ModelResult:
    warnings: list[WarningRecord] = []

    if target_column not in df.columns:
        available_columns = ", ".join(df.columns.astype(str).tolist())
        raise ValueError(f"타깃 컬럼을 찾을 수 없습니다: {target_column}. 사용 가능한 컬럼: {available_columns}")

    model_df = df.dropna(subset=[target_column]).copy()
    if model_df.empty:
        raise ValueError("타깃 결측치를 제거하고 나면 사용할 수 있는 데이터가 없습니다.")

    target = model_df[target_column]
    preprocessor, features, preprocessing_summary = build_preprocessing_pipeline(
        model_df,
        target_column,
        feature_columns=feature_columns,
    )

    if features.empty:
        raise ValueError("타깃 컬럼을 제외하고 사용할 입력 변수가 없습니다.")

    if not preprocessing_summary.numeric_columns and not preprocessing_summary.categorical_columns:
        raise ValueError("전처리에 사용할 수치형 또는 범주형 입력 변수가 없습니다.")

    problem_type = detect_problem_type(target) if task_type == "auto" else task_type

    if problem_type == "classification" and target.nunique() < 2:
        raise ValueError("분류 문제는 최소 2개 이상의 클래스가 필요합니다.")

    if preprocessing_summary.datetime_columns:
        warnings.append(
            WarningRecord(
                code="datetime_columns_excluded",
                level="warning",
                message="날짜형 컬럼은 감지되었지만 자동 feature engineering 없이 학습 대상에서 제외했습니다.",
                details={"columns": preprocessing_summary.datetime_columns},
            )
        )

    if preprocessing_summary.identifier_columns:
        warnings.append(
            WarningRecord(
                code="identifier_columns_excluded",
                level="warning",
                message="식별자 성격의 컬럼은 기본적으로 모델 입력에서 제외했습니다.",
                details={"columns": preprocessing_summary.identifier_columns},
            )
        )

    models = get_baseline_models(problem_type)
    if selected_model is not None:
        if selected_model not in models:
            available_models = ", ".join(models.keys())
            raise ValueError(f"선택한 모델을 찾을 수 없습니다: {selected_model}. 사용 가능한 모델: {available_models}")
        models = {selected_model: models[selected_model]}
    if eval_method == "cv":
        metrics_df, effective_cv_folds, model_warnings = _evaluate_with_cv(
            features=features,
            target=target,
            preprocessor=preprocessor,
            models=models,
            problem_type=problem_type,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        warnings.extend(model_warnings)
        train_rows = 0
        validation_rows = 0
    else:
        metrics_df, train_rows, validation_rows, model_warnings = _evaluate_with_holdout(
            features=features,
            target=target,
            preprocessor=preprocessor,
            models=models,
            problem_type=problem_type,
            test_size=test_size,
            random_state=random_state,
        )
        warnings.extend(model_warnings)
        effective_cv_folds = cv_folds

    if metrics_df.empty:
        raise ValueError("모든 baseline 모델 학습이 실패했습니다. warnings_summary를 확인해 주세요.")

    best_model_name = selected_model or _select_best_model(metrics_df, problem_type)
    best_model_pipeline = _fit_best_model_pipeline(
        features=features,
        target=target,
        preprocessor=preprocessor,
        models=models,
        best_model_name=best_model_name,
    )

    return ModelResult(
        target=target_column,
        problem_type=problem_type,
        train_rows=train_rows,
        validation_rows=validation_rows,
        eval_method=eval_method,
        cv_folds=effective_cv_folds,
        metrics=metrics_df,
        best_model_name=best_model_name,
        preprocessing_summary=preprocessing_summary,
        warnings=warnings,
        best_model_pipeline=best_model_pipeline,
    )


def _evaluate_with_holdout(
    features: pd.DataFrame,
    target: pd.Series,
    preprocessor: object,
    models: dict[str, object],
    problem_type: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, int, int, list[WarningRecord]]:
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

    rows: list[dict[str, object]] = []
    warnings: list[WarningRecord] = []
    for model_name, estimator in models.items():
        try:
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

            row = {"model": model_name, "problem_type": problem_type, **metrics}
            for metric_name in metrics.keys():
                row[f"{metric_name}_std"] = 0.0
            rows.append(row)
        except Exception as exc:
            warnings.append(
                WarningRecord(
                    code="model_training_exception",
                    level="warning",
                    message=f"{model_name} 학습 중 예외가 발생해 해당 모델 결과를 제외했습니다.",
                    details={"model": model_name, "error": str(exc), "eval_method": "holdout"},
                )
            )

    return pd.DataFrame(rows), len(X_train), len(X_valid), warnings


def _evaluate_with_cv(
    features: pd.DataFrame,
    target: pd.Series,
    preprocessor: object,
    models: dict[str, object],
    problem_type: str,
    cv_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, int, list[WarningRecord]]:
    splitter, effective_folds = _build_cv_splitter(problem_type, target, cv_folds, random_state)
    rows: list[dict[str, object]] = []
    warnings: list[WarningRecord] = []

    for model_name, estimator in models.items():
        fold_metrics: list[dict[str, float | None]] = []
        try:
            split_target = target if problem_type == "classification" else None
            for train_idx, valid_idx in splitter.split(features, split_target):
                X_train = features.iloc[train_idx]
                X_valid = features.iloc[valid_idx]
                y_train = target.iloc[train_idx]
                y_valid = target.iloc[valid_idx]

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
                fold_metrics.append(metrics)
        except Exception as exc:
            warnings.append(
                WarningRecord(
                    code="model_training_exception",
                    level="warning",
                    message=f"{model_name} 학습 중 예외가 발생해 해당 모델 결과를 제외했습니다.",
                    details={"model": model_name, "error": str(exc), "eval_method": "cv"},
                )
            )
            continue

        aggregated = _aggregate_fold_metrics(fold_metrics)
        rows.append(
            {
                "model": model_name,
                "problem_type": problem_type,
                "evaluated_folds": effective_folds,
                **aggregated,
            }
        )

    return pd.DataFrame(rows), effective_folds, warnings


def _build_cv_splitter(problem_type: str, target: pd.Series, cv_folds: int, random_state: int):
    if problem_type == "classification":
        min_class_count = int(target.value_counts().min())
        effective_folds = min(cv_folds, min_class_count)
        if effective_folds < 2:
            raise ValueError("분류 CV는 각 클래스에 최소 2개 이상의 샘플이 필요합니다.")
        return StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state), effective_folds

    effective_folds = min(cv_folds, len(target))
    if effective_folds < 2:
        raise ValueError("회귀 CV는 최소 2개 이상의 샘플이 필요합니다.")
    return KFold(n_splits=effective_folds, shuffle=True, random_state=random_state), effective_folds


def _aggregate_fold_metrics(fold_metrics: list[dict[str, float | None]]) -> dict[str, object]:
    metric_names = sorted({key for metrics in fold_metrics for key in metrics.keys()})
    aggregated: dict[str, object] = {}
    for name in metric_names:
        values = [metrics[name] for metrics in fold_metrics if metrics.get(name) is not None and pd.notna(metrics.get(name))]
        if not values:
            aggregated[name] = np.nan
            aggregated[f"{name}_std"] = np.nan
            continue
        numeric_values = [float(value) for value in values]
        aggregated[name] = float(np.mean(numeric_values))
        aggregated[f"{name}_std"] = float(np.std(numeric_values, ddof=0))
    return aggregated


def _select_best_model(metrics_df: pd.DataFrame, problem_type: str) -> str:
    if problem_type == "regression":
        ordered = metrics_df.sort_values(["rmse", "mae", "r2"], ascending=[True, True, False])
        return str(ordered.iloc[0]["model"])

    sortable = metrics_df.copy()
    sortable["roc_auc_filled"] = sortable["roc_auc"].fillna(-1.0)
    ordered = sortable.sort_values(["accuracy", "f1", "roc_auc_filled"], ascending=[False, False, False])
    return str(ordered.iloc[0]["model"])


def _fit_best_model_pipeline(
    features: pd.DataFrame,
    target: pd.Series,
    preprocessor: object,
    models: dict[str, object],
    best_model_name: str,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", models[best_model_name]),
        ]
    )
    pipeline.fit(features, target)
    return pipeline


def save_model_results(result: ModelResult, output_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_path = output_path / "model_comparison.csv"
    result.metrics.to_csv(comparison_path, index=False)

    model_path = output_path / "best_model.joblib"
    joblib.dump(result.best_model_pipeline, model_path)

    best_row = result.metrics.loc[result.metrics["model"] == result.best_model_name].iloc[0].to_dict()
    metadata = {
        "target": result.target,
        "problem_type": result.problem_type,
        "eval_method": result.eval_method,
        "cv_folds": result.cv_folds,
        "best_model_name": result.best_model_name,
        "artifact_path": str(model_path),
        "selected_feature_columns": result.preprocessing_summary.selected_feature_columns,
        "numeric_columns": result.preprocessing_summary.numeric_columns,
        "categorical_columns": result.preprocessing_summary.categorical_columns,
        "datetime_columns": result.preprocessing_summary.datetime_columns,
        "identifier_columns": result.preprocessing_summary.identifier_columns,
        "best_metrics": {
            key: (float(value) if pd.notna(value) and isinstance(value, (int, float, np.floating)) else value)
            for key, value in best_row.items()
        },
    }
    metadata_path = output_path / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8-sig")

    problem_type_label = "회귀" if result.problem_type == "regression" else "분류"
    summary_lines = [
        "# 모델 성능 요약",
        "",
        f"- 타깃 컬럼: {result.target}",
        f"- 문제 유형: {problem_type_label}",
        f"- 평가 방식: {result.eval_method}",
        f"- CV fold 수: {result.cv_folds if result.eval_method == 'cv' else '해당 없음'}",
        f"- 학습 데이터 수: {result.train_rows if result.eval_method == 'holdout' else 'fold별 분할'}",
        f"- 검증 데이터 수: {result.validation_rows if result.eval_method == 'holdout' else 'fold별 분할'}",
        f"- 최고 성능 모델: {result.best_model_name}",
        "- best model 저장: 완료",
        f"- 모델 artifact 경로: {model_path.name}",
        f"- 모델 metadata 경로: {metadata_path.name}",
        "",
        "## 모델 비교",
        "",
        result.metrics.to_markdown(index=False),
        "",
        "## 전처리 요약",
        "",
        f"- 선택된 독립변수: {', '.join(result.preprocessing_summary.selected_feature_columns) if result.preprocessing_summary.selected_feature_columns else '없음'}",
        f"- 수치형 컬럼: {', '.join(result.preprocessing_summary.numeric_columns) if result.preprocessing_summary.numeric_columns else '없음'}",
        f"- 범주형 컬럼: {', '.join(result.preprocessing_summary.categorical_columns) if result.preprocessing_summary.categorical_columns else '없음'}",
        f"- 날짜형 컬럼: {', '.join(result.preprocessing_summary.datetime_columns) if result.preprocessing_summary.datetime_columns else '없음'}",
        f"- 식별자 자동 제외: {', '.join(result.preprocessing_summary.identifier_columns) if result.preprocessing_summary.identifier_columns else '없음'}",
        "",
    ]

    summary_path = output_path / "model_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8-sig")
    return comparison_path, summary_path, model_path, metadata_path
