from __future__ import annotations

import unittest

import pandas as pd
from sklearn.pipeline import Pipeline

from app.model_selection import get_param_grid
from app.train import train_and_compare_models


def _regression_df(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [20 + (index % 40) for index in range(rows)],
            "income": [2000 + index * 37 for index in range(rows)],
            "target": [float(index % 17) for index in range(rows)],
        }
    )


class ParamGridTests(unittest.TestCase):
    def test_linear_regression_has_nothing_to_search(self) -> None:
        self.assertEqual(get_param_grid("LinearRegression"), {})

    def test_random_forest_grid_targets_pipeline_step(self) -> None:
        grid = get_param_grid("RandomForestRegressor")

        self.assertTrue(grid)
        # Keys must address the estimator inside the pipeline, not the bare model.
        self.assertTrue(all(key.startswith("model__") for key in grid))


class TuningTests(unittest.TestCase):
    def test_tuning_is_off_by_default(self) -> None:
        result = train_and_compare_models(_regression_df(), "target")

        self.assertFalse(result.tuned)
        self.assertEqual(result.best_params, {})

    def test_tuning_reports_the_chosen_parameters(self) -> None:
        result = train_and_compare_models(
            _regression_df(),
            "target",
            selected_model="RandomForestRegressor",
            tune=True,
        )

        self.assertTrue(result.tuned)
        self.assertIn("n_estimators", result.best_params)
        self.assertIn("max_depth", result.best_params)

    def test_saved_model_is_a_plain_pipeline_not_a_search_object(self) -> None:
        result = train_and_compare_models(
            _regression_df(),
            "target",
            selected_model="RandomForestRegressor",
            tune=True,
        )

        # The artifact must be directly usable for prediction without unwrapping.
        self.assertIsInstance(result.best_model_pipeline, Pipeline)
        predictions = result.best_model_pipeline.predict(_regression_df().drop(columns=["target"]))
        self.assertEqual(len(predictions), 60)

    def test_tiny_dataset_skips_tuning_with_a_warning(self) -> None:
        tiny = pd.DataFrame({"age": [20, 30, 40], "target": [1.0, 2.0, 3.0]})

        result = train_and_compare_models(tiny, "target", tune=True)

        self.assertFalse(result.tuned)
        codes = {record.code for record in result.warnings}
        self.assertIn("tuning_skipped_small_data", codes)
