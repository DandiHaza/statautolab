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
