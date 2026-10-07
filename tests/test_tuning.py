"""Checks for outer-fold isolation and positional nested OOF predictions."""

import unittest
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold

from src.model import build_pipeline
from src.preprocess import build_feature_preprocessor
from src.tuning import nested_cross_validate, tune_pipeline


class NestedTuningTests(unittest.TestCase):
    def test_search_only_receives_outer_training_rows_and_oof_is_aligned(self):
        X = pd.DataFrame({"age": np.arange(24, dtype=float)},
                         index=np.arange(100, 124)[::-1])
        X.iloc[3, 0] = np.nan
        y = pd.Series([0, 1] * 12, index=X.index)
        pipeline = build_pipeline(
            build_feature_preprocessor(X, {"scaler": "standard"}),
            {"type": "decision_tree", "params": {"random_state": 42}},
        )
        outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=67)
        inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=67)
        expected_folds = list(outer.split(X, y))
        fitted_pipelines = []

        def checked_tune(base, X_train, y_train, *args):
            train_idx, val_idx = expected_folds[len(fitted_pipelines)]
            self.assertEqual(list(X_train.index), list(X.iloc[train_idx].index))
            self.assertEqual(list(y_train.index), list(y.iloc[train_idx].index))
            self.assertTrue(set(X_train.index).isdisjoint(X.iloc[val_idx].index))
            best, study = tune_pipeline(base, X_train, y_train, *args)
            fitted_pipelines.append(best)
            return best, study

        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            with patch("src.tuning.tune_pipeline", side_effect=checked_tune):
                scores, oof = nested_cross_validate(
                    pipeline, X, y, outer, inner, "balanced_accuracy",
                    {"classifier__max_depth": {"type": "int", "low": 1, "high": 3}},
                    n_trials=2, random_state=67,
                )
        finally:
            optuna.logging.set_verbosity(previous_verbosity)

        scorer = get_scorer("balanced_accuracy")
        for row, (train_idx, val_idx), fitted in zip(
            scores.to_dict("records"), expected_folds, fitted_pipelines
        ):
            np.testing.assert_array_equal(oof[val_idx], fitted.predict(X.iloc[val_idx]))
            self.assertAlmostEqual(row["validation"], scorer(fitted, X.iloc[val_idx], y.iloc[val_idx]))
            scaler = fitted.named_steps["preprocessing"].named_transformers_["numerical"].named_steps["scaler"]
            self.assertAlmostEqual(scaler.mean_[0], X.iloc[train_idx]["age"].mean())
            self.assertAlmostEqual(row["gap"], row["train"] - row["validation"])
        self.assertEqual(len(scores), 3)
        self.assertFalse(hasattr(pipeline.named_steps["classifier"], "tree_"))

    def test_repeated_outer_validation_rows_are_rejected_before_tuning(self):
        class InvalidOuterCV:
            def split(self, X, y):
                yield np.array([1]), np.array([0])
                yield np.array([1]), np.array([0])

        with patch("src.tuning.tune_pipeline") as search:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                nested_cross_validate(
                    None, pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]),
                    InvalidOuterCV(), None, "accuracy", {}, 1, 67,
                )
            search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
