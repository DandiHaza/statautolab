from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.evaluate import evaluate_regression
from app.model_selection import get_baseline_models
from app.preprocessing import build_preprocessing_pipeline
from app.train import ModelResult


@dataclass
class RegressionDashboardData:
    best_metrics: dict[str, float]
    intercept: float | None
    regression_equation: str | None
    coefficients: pd.DataFrame
    feature_importances: pd.DataFrame
    predictions_preview: pd.DataFrame
    residual_summary: dict[str, float]
    ols_overview: dict[str, float | int | str | None]
    ols_coefficients: pd.DataFrame
    ols_diagnostics: dict[str, float | None]
    ols_summary_text: str | None
    combined_summary_table: pd.DataFrame


def _clean_feature_name(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def _get_feature_names(preprocessor: object, fallback_count: int) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        return [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    return [f"feature_{index}" for index in range(fallback_count)]


def _build_ols_details(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[dict[str, float | int | str | None], pd.DataFrame, dict[str, float | None], str | None]:
    try:
        import statsmodels.api as sm
    except Exception:
        return {}, pd.DataFrame(), {}, None

    model_df = df.dropna(subset=[target_column]).copy()
    preprocessor, features, _ = build_preprocessing_pipeline(
        model_df,
        target_column,
        feature_columns=feature_columns,
    )
    transformed = preprocessor.fit_transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed_df = pd.DataFrame(
        transformed,
        columns=_get_feature_names(preprocessor, transformed.shape[1]),
        index=model_df.index,
    )
    transformed_df = sm.add_constant(transformed_df, has_constant="add")
    ols_model = sm.OLS(model_df[target_column].astype(float), transformed_df.astype(float)).fit()

    conf_int = ols_model.conf_int()
    coefficient_table = pd.DataFrame(
        {
            "feature": ols_model.params.index,
            "coefficient": ols_model.params.values,
            "std_error": ols_model.bse.values,
            "t_value": ols_model.tvalues.values,
            "p_value": ols_model.pvalues.values,
            "ci_lower": conf_int.iloc[:, 0].values,
            "ci_upper": conf_int.iloc[:, 1].values,
        }
    )

    overview = {
        "dependent_variable": target_column,
        "model": "OLS",
        "method": "Least Squares",
        "observations": int(ols_model.nobs),
        "df_model": float(ols_model.df_model),
        "df_resid": float(ols_model.df_resid),
        "r_squared": float(ols_model.rsquared),
        "adj_r_squared": float(ols_model.rsquared_adj),
        "f_statistic": float(ols_model.fvalue) if ols_model.fvalue is not None else None,
        "prob_f_statistic": float(ols_model.f_pvalue) if ols_model.f_pvalue is not None else None,
        "aic": float(ols_model.aic),
        "bic": float(ols_model.bic),
        "log_likelihood": float(ols_model.llf),
    }
    diagnostics = {
        "durbin_watson": float(sm.stats.stattools.durbin_watson(ols_model.resid)),
        "condition_number": float(ols_model.condition_number),
        "residual_skew": float(pd.Series(ols_model.resid).skew()),
        "residual_kurtosis": float(pd.Series(ols_model.resid).kurt()),
    }
    return overview, coefficient_table, diagnostics, ols_model.summary().as_text()


def _build_linear_regression_coefficients(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[float | None, str | None, pd.DataFrame]:
    model_df = df.dropna(subset=[target_column]).copy()
    preprocessor, features, _ = build_preprocessing_pipeline(
        model_df,
        target_column,
        feature_columns=feature_columns,
    )
    estimator = get_baseline_models("regression")["LinearRegression"]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(features, model_df[target_column])

    model = pipeline.named_steps["model"]
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        return None, None, pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    coefficient_values = np.ravel(coefficients)
    feature_names = _get_feature_names(pipeline.named_steps["preprocessor"], len(coefficient_values))
    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficient_values,
            "abs_coefficient": np.abs(coefficient_values),
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    intercept = getattr(model, "intercept_", None)
    intercept_value = float(np.ravel([intercept])[0]) if intercept is not None else None
    equation_terms: list[str] = []
    if intercept_value is not None:
        equation_terms.append(f"{target_column} = {intercept_value:.4f}")
    else:
        equation_terms.append(f"{target_column} =")

    ordered_terms = coefficient_df.sort_values("feature").reset_index(drop=True)
    for _, row in ordered_terms.iterrows():
        coefficient = float(row["coefficient"])
        feature_name = str(row["feature"])
        sign = "+" if coefficient >= 0 else "-"
        equation_terms.append(f" {sign} {abs(coefficient):.4f} * {feature_name}")

    equation = "".join(equation_terms) if len(equation_terms) > 1 else None
    return intercept_value, equation, coefficient_df


def _build_combined_summary_table(ols_coefficients: pd.DataFrame) -> pd.DataFrame:
    coefficient_rows = ols_coefficients.copy()
    if coefficient_rows.empty:
        return pd.DataFrame(
            columns=["section", "feature", "coefficient", "std_error", "t_value", "p_value", "ci_lower", "ci_upper"]
        )

    coefficient_rows.insert(0, "section", "회귀계수")
    return coefficient_rows.reset_index(drop=True)


def build_regression_dashboard_data(df: pd.DataFrame, model_result: ModelResult) -> RegressionDashboardData | None:
    if model_result.problem_type != "regression":
        return None

    target_column = model_result.target
    feature_columns = model_result.preprocessing_summary.selected_feature_columns
    model_df = df.dropna(subset=[target_column]).copy()
    if model_df.empty:
        return None

    features = model_df[feature_columns].copy()
    target = model_df[target_column]
    best_pipeline = model_result.best_model_pipeline
    predictions = best_pipeline.predict(features)
    evaluation = evaluate_regression(target, predictions)
    residuals = target - predictions

    best_model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_importances = pd.DataFrame(columns=["feature", "importance"])
    if hasattr(best_model, "feature_importances_"):
        importances = np.ravel(best_model.feature_importances_)
        feature_names = _get_feature_names(preprocessor, len(importances))
        feature_importances = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    intercept, regression_equation, coefficients = _build_linear_regression_coefficients(
        df,
        target_column,
        feature_columns,
    )
    predictions_preview = pd.DataFrame(
        {
            "actual": target,
            "predicted": predictions,
            "residual": residuals,
        }
    ).reset_index(drop=True)

    residual_summary = {
        "mean_residual": float(residuals.mean()),
        "residual_std": float(residuals.std(ddof=0)),
        "max_absolute_residual": float(np.abs(residuals).max()),
    }
    ols_overview, ols_coefficients, ols_diagnostics, ols_summary_text = _build_ols_details(
        df,
        target_column,
        feature_columns,
    )
    combined_summary_table = _build_combined_summary_table(ols_coefficients)

    return RegressionDashboardData(
        best_metrics=evaluation,
        intercept=intercept,
        regression_equation=regression_equation,
        coefficients=coefficients,
        feature_importances=feature_importances,
        predictions_preview=predictions_preview,
        residual_summary=residual_summary,
        ols_overview=ols_overview,
        ols_coefficients=ols_coefficients,
        ols_diagnostics=ols_diagnostics,
        ols_summary_text=ols_summary_text,
        combined_summary_table=combined_summary_table,
    )
