from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from app.preprocessing import build_preprocessing_pipeline, collect_reference_levels
from app.regression_insights import build_regression_dashboard_data
from app.train import train_and_compare_models

FEATURES = ["age", "income", "city", "visits"]


def _sample_df() -> pd.DataFrame:
    return pd.read_csv("data/examples/regression_sample.csv")


def _design_matrix(drop_first: bool) -> pd.DataFrame:
    df = _sample_df()
    preprocessor, features, _ = build_preprocessing_pipeline(
        df, "spending_score", feature_columns=FEATURES, drop_first_category=drop_first
    )
    transformed = preprocessor.fit_transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    frame = pd.DataFrame(transformed, index=df.index)
    return sm.add_constant(frame, has_constant="add")


class ReferenceLevelEncodingTests(unittest.TestCase):
    def test_full_dummy_encoding_with_an_intercept_is_rank_deficient(self) -> None:
        # This is why the OLS path needs a reference level: the dummies sum to the
        # intercept column, so the coefficients are not uniquely determined.
        matrix = _design_matrix(drop_first=False)

        self.assertLess(np.linalg.matrix_rank(matrix.values), matrix.shape[1])

    def test_reference_level_encoding_gives_a_full_rank_design(self) -> None:
        matrix = _design_matrix(drop_first=True)

        self.assertEqual(np.linalg.matrix_rank(matrix.values), matrix.shape[1])

    def test_prediction_pipeline_keeps_the_full_encoding(self) -> None:
        # Prediction models add no intercept column, so dropping a level would only
        # throw information away.
        preprocessor, features, _ = build_preprocessing_pipeline(
            _sample_df(), "spending_score", feature_columns=FEATURES
        )
        preprocessor.fit(features)

        self.assertEqual(collect_reference_levels(preprocessor), {})

    def test_reference_level_is_reported(self) -> None:
        preprocessor, features, _ = build_preprocessing_pipeline(
            _sample_df(), "spending_score", feature_columns=FEATURES, drop_first_category=True
        )
        preprocessor.fit(features)

        self.assertEqual(collect_reference_levels(preprocessor), {"city": "Busan"})


class RegressionDashboardTests(unittest.TestCase):
    def test_dashboard_builds_without_a_rank_deficiency_warning(self) -> None:
        df = _sample_df()
        model_result = train_and_compare_models(
            df, "spending_score", feature_columns=FEATURES, selected_model="LinearRegression"
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dashboard = build_regression_dashboard_data(df, model_result)

        rank_warnings = [record for record in caught if "rank-deficient" in str(record.message)]

        self.assertEqual(rank_warnings, [], "설계행렬이 여전히 랭크 부족입니다.")
        self.assertEqual(dashboard.reference_levels, {"city": "Busan"})
        self.assertFalse(np.isnan(dashboard.ols_overview["adj_r_squared"]))
