from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.io import load_dataset
from app.model_selection import detect_problem_type
from app.preprocessing import build_preprocessing_pipeline
from app.profiling import profile_dataset
from app.train import save_model_results, train_and_compare_models


@pytest.fixture
def classification_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [21, 25, 28, 32, 36, 40, 22, 27, 31, 38],
            "score": [55, 60, 62, 70, 78, 85, 58, 65, 72, 88],
            "city": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
            "target": ["no", "no", "no", "yes", "yes", "yes", "no", "no", "yes", "yes"],
        }
    )


@pytest.fixture
def local_tmp_path() -> Path:
    tmp_path = Path("tests/.tmp_pytest_case")
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_load_dataset_reads_csv(local_tmp_path: Path) -> None:
    csv_path = local_tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")

    loaded = load_dataset(csv_path)

    assert loaded.shape == (2, 2)
    assert loaded.columns.tolist() == ["a", "b"]


def test_build_preprocessing_pipeline_classifies_columns() -> None:
    df = pd.DataFrame(
        {
            "num_col": [1.0, None, 3.0],
            "cat_col": ["a", None, "b"],
            "date_col": pd.to_datetime(["2026-03-10", "2026-03-11", "2026-03-12"]),
            "target": [0, 1, 0],
        }
    )

    _, features, summary = build_preprocessing_pipeline(df, "target")

    assert "target" not in features.columns
    assert summary.numeric_columns == ["num_col"]
    assert summary.categorical_columns == ["cat_col"]
    assert summary.datetime_columns == ["date_col"]


def test_profile_dataset_builds_missing_summary() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": ["x", "y", None],
        }
    )

    profile = profile_dataset(df)

    assert profile.missing.loc[profile.missing["column"] == "a", "missing_count"].iloc[0] == 1
    assert profile.missing.loc[profile.missing["column"] == "b", "missing_count"].iloc[0] == 1


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series([1.0, 2.0, 3.0]), "regression"),
        (pd.Series(["yes", "no", "yes"]), "classification"),
    ],
)
def test_detect_problem_type(series: pd.Series, expected: str) -> None:
    assert detect_problem_type(series) == expected


def test_train_and_compare_models_returns_comparison_result(classification_df: pd.DataFrame) -> None:
    result = train_and_compare_models(classification_df, "target")

    assert result.problem_type == "classification"
    assert result.best_model_name in {"LogisticRegression", "RandomForestClassifier"}
    assert set(result.metrics["model"]) == {"LogisticRegression", "RandomForestClassifier"}
    assert {"accuracy", "f1"}.issubset(result.metrics.columns)


def test_train_and_compare_models_supports_cv(classification_df: pd.DataFrame) -> None:
    result = train_and_compare_models(classification_df, "target", eval_method="cv", cv_folds=3)

    assert result.eval_method == "cv"
    assert result.cv_folds == 3
    assert {"accuracy", "accuracy_std", "f1", "f1_std"}.issubset(result.metrics.columns)


def test_save_model_results_creates_model_artifacts(
    classification_df: pd.DataFrame,
    local_tmp_path: Path,
) -> None:
    result = train_and_compare_models(classification_df, "target")

    comparison_path, summary_path, model_path, metadata_path = save_model_results(result, local_tmp_path)

    assert comparison_path.exists()
    assert summary_path.exists()
    assert model_path.exists()
    assert metadata_path.exists()
