from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def evaluate_regression(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)) if len(y_true) >= 2 else float("nan"),
    }


def evaluate_classification(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float | None]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, average="weighted")),
        "roc_auc": None,
    }


def add_classification_auc(
    metrics: dict[str, float | None],
    estimator: object,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> dict[str, float | None]:
    try:
        class_count = int(pd.Series(y_valid).nunique())
        if class_count < 2:
            return metrics

        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X_valid)
            if class_count == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_valid, probabilities[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_valid, probabilities, multi_class="ovr"))
        elif hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X_valid)
            metrics["roc_auc"] = float(roc_auc_score(y_valid, scores, multi_class="ovr"))
    except Exception:
        metrics["roc_auc"] = None

    return metrics
