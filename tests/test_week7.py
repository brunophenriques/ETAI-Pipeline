"""OOF probability alignment, multimetric scoring and append-only run storage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.evaluate import build_scorers, cross_validate_pipeline, oof_classification_report
from src.model import build_pipeline
from src.preprocess import build_feature_preprocessor
from src.results import save_scores
from src.tuning import nested_cross_validate


class Week7Tests(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=48, n_features=4, n_informative=3,
                                  n_redundant=0, flip_y=.1, random_state=42)
        self.X = pd.DataFrame(X, index=np.arange(100, 148)[::-1])
        self.y = pd.Series(y, index=self.X.index)
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        self.scorers = build_scorers(["accuracy", "balanced_accuracy", "precision", "recall",
                                     "f1", "fbeta_2", "roc_auc", "average_precision"])
        self.pipeline = build_pipeline(build_feature_preprocessor(self.X, {"scaler": "standard"}),
                                       {"type": "logistic_regression", "params": {"max_iter": 1000}})

    def test_oof_probabilities_and_every_metric_use_the_correct_validation_rows(self):
        folds, proba = cross_validate_pipeline(self.pipeline, self.X, self.y,
                                               self.cv, self.scorers, "roc_auc")
        for row, (train, val) in zip(folds.to_dict("records"), self.cv.split(self.X, self.y)):
            fitted = clone(self.pipeline).fit(self.X.iloc[train], self.y.iloc[train])
            np.testing.assert_allclose(proba[val], fitted.predict_proba(self.X.iloc[val])[:, 1])
            self.assertAlmostEqual(row["roc_auc"], roc_auc_score(self.y.iloc[val], proba[val]))
            for name, scorer in self.scorers.items():
                self.assertAlmostEqual(row[name], scorer(fitted, self.X.iloc[val], self.y.iloc[val]))
        self.assertTrue(((proba > 0) & (proba < 1)).any())
        self.assertFalse(hasattr(self.pipeline.named_steps["preprocessing"], "transformers_"))
        report = oof_classification_report(self.y, proba, probabilities=True)
        self.assertIn("Confusion matrix", report)

    def test_nested_supports_custom_f2_selection_and_all_probability_metrics(self):
        verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            folds, proba = nested_cross_validate(
                self.pipeline, self.X, self.y, self.cv,
                StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
                "fbeta_2", {"classifier__C": {"type": "float", "low": .01, "high": 1, "log": True}},
                2, 42, scorers=self.scorers,
            )
        finally:
            optuna.logging.set_verbosity(verbosity)
        self.assertTrue(np.isfinite(proba).all())
        for row, (_, val) in zip(folds.to_dict("records"), self.cv.split(self.X, self.y)):
            self.assertAlmostEqual(row["fbeta_2"], fbeta_score(self.y.iloc[val], proba[val] >= .5, beta=2))
            self.assertAlmostEqual(row["roc_auc"], roc_auc_score(self.y.iloc[val], proba[val]))
        self.assertTrue(set(self.scorers).issubset(folds.columns))

    def test_scores_append_and_preserve_run_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "runs.csv")
            save_scores(path, {"run": "0001", "model": "dummy", "mean": .5})
            board = save_scores(path, {"run": "0002", "model": "naive_bayes", "mean": .7})
            self.assertEqual(board["run"].tolist(), ["0001", "0002"])
            self.assertEqual(board["model"].tolist(), ["dummy", "naive_bayes"])

    def test_new_models_fit_and_produce_probabilities_in_the_pipeline(self):
        for name, params in [("gradient_boosting", {"n_estimators": 3, "random_state": 42}),
                             ("hist_gradient_boosting", {"max_iter": 3, "random_state": 42}),
                             ("naive_bayes", {})]:
            with self.subTest(model=name), threadpool_limits(limits=1):
                model = build_pipeline(build_feature_preprocessor(self.X, {"scaler": "standard"}),
                                       {"type": name, "params": params})
                model.fit(self.X, self.y)
                self.assertEqual(model.predict_proba(self.X).shape, (48, 2))


if __name__ == "__main__":
    unittest.main()
