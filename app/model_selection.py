from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression


def detect_problem_type(target: pd.Series) -> str:
    return "regression" if pd.api.types.is_numeric_dtype(target) else "classification"


def get_baseline_models(problem_type: str) -> dict[str, Any]:
    if problem_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        }

    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
    }


# Grids stay deliberately small: tuning runs inside every evaluation fold, and the
# hosted demo has one CPU. LinearRegression has nothing worth searching.
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "RandomForestRegressor": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
    },
    "RandomForestClassifier": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10],
    },
    "LogisticRegression": {
        "model__C": [0.1, 1.0, 10.0],
    },
}


def get_param_grid(model_name: str) -> dict[str, list[Any]]:
    return PARAM_GRIDS.get(model_name, {})
