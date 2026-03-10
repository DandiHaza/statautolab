from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelResult:
    target: str
    problem_type: str
    train_rows: int
    validation_rows: int
    metrics: pd.DataFrame
    best_model_name: str


def detect_problem_type(target: pd.Series) -> str:
    return "regression" if pd.api.types.is_numeric_dtype(target) else "classification"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def _build_models(problem_type: str) -> dict[str, object]:
    if problem_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        }

    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def _evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _evaluate_classification(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def train_baseline_models(df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> ModelResult:
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    model_df = df.dropna(subset=[target_column]).copy()
    if model_df.empty:
        raise ValueError("No rows available after dropping missing target values.")

    X = model_df.drop(columns=[target_column])
    y = model_df[target_column]

    if X.empty:
        raise ValueError("No feature columns available after removing the target column.")

    problem_type = detect_problem_type(y)
    if problem_type == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    preprocessor = _build_preprocessor(X)
    models = _build_models(problem_type)

    stratify = None
    if problem_type == "classification":
        min_class_count = int(y.value_counts().min())
        if min_class_count >= 2:
            stratify = y

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

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
            metrics = _evaluate_regression(y_valid, predictions)
        else:
            metrics = _evaluate_classification(y_valid, predictions)

        rows.append(
            {
                "model": model_name,
                "problem_type": problem_type,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)
    if problem_type == "regression":
        best_model_name = metrics_df.sort_values(["rmse", "mae"], ascending=[True, True]).iloc[0]["model"]
    else:
        best_model_name = metrics_df.sort_values(["accuracy", "f1_weighted"], ascending=[False, False]).iloc[0]["model"]

    return ModelResult(
        target=target_column,
        problem_type=problem_type,
        train_rows=len(X_train),
        validation_rows=len(X_valid),
        metrics=metrics_df,
        best_model_name=str(best_model_name),
    )


def save_model_results(result: ModelResult, output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "model_comparison.csv"
    result.metrics.to_csv(csv_path, index=False)

    markdown_lines = [
        "# Model Comparison",
        "",
        f"- Target: {result.target}",
        f"- Problem type: {result.problem_type}",
        f"- Train rows: {result.train_rows}",
        f"- Validation rows: {result.validation_rows}",
        f"- Best model: {result.best_model_name}",
        "",
        result.metrics.to_markdown(index=False),
        "",
    ]
    markdown_path = output_path / "model_comparison.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    return csv_path, markdown_path
