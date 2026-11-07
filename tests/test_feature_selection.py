"""Selection must be fold-local, clonable and safe to cache by training data."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from src.evaluate import build_scorers, corrected_paired_test, cross_validate_pipeline
from src.model import VoteSelector, build_pipeline, build_selector, selection_search_space
from src.preprocess import build_feature_preprocessor


class FeatureSelectionTests(unittest.TestCase):
    def test_selector_factory_and_method_specific_spaces(self):
        spaces = {
            "vote": {"select__min_votes": {"type": "int", "low": 1, "high": 3}},
            "kbest": {"select__k": {"type": "int", "low": 1, "high": 6}},
            "from_model": {"select__threshold": {"type": "categorical", "choices": ["mean", "median"]}},
        }
        for method in spaces:
            with self.subTest(method=method), threadpool_limits(limits=1):
                config = {"enabled": True, "method": method, "min_votes": 1,
                          "k": 3, "random_state": 42, "search_spaces": spaces}
                model = build_pipeline(
                    build_feature_preprocessor(self.X, {"scaler": "standard"}),
                    {"type": "logistic_regression", "params": {}}, config,
                )
                self.assertEqual(selection_search_space(config), spaces[method])
                self.assertTrue(set(spaces[method]).issubset(model.get_params(deep=True)))
                fitted = clone(model).fit(self.X, self.y)
                selector = fitted.named_steps["select"]
                names = fitted.named_steps["preprocessing"].get_feature_names_out()
                self.assertGreater(selector.get_support().sum(), 0)
                self.assertEqual(len(selector.get_feature_names_out(names)), selector.get_support().sum())
                self.assertFalse(hasattr(model.named_steps["select"], "n_features_in_"))
                self.assertEqual(len(fitted.predict(self.X)), len(self.X))
                if method == "kbest":
                    self.assertEqual(selector.get_support().sum(), 3)
        self.assertEqual(selection_search_space({"enabled": True, "search_space": spaces["vote"]}), spaces["vote"])
        with self.assertRaises(ValueError):
            build_selector({"method": "unknown"})

    def setUp(self):
        X, y = make_classification(n_samples=90, n_features=6, n_informative=3,
                                  n_redundant=0, random_state=42)
        self.X, self.y = pd.DataFrame(X), pd.Series(y)

    def pipeline(self, memory=None):
        return build_pipeline(
            build_feature_preprocessor(self.X, {"scaler": "standard"}),
            {"type": "logistic_regression", "params": {"max_iter": 2000}},
            {"enabled": True, "min_votes": 1, "random_state": 42}, memory=memory,
        )

    def test_selector_only_sees_each_fold_training_rows(self):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        expected = list(cv.split(self.X, self.y))
        observed = []
        original = VoteSelector.fit

        def checked_fit(selector, X, y):
            train, _ = expected[len(observed)]
            np.testing.assert_array_equal(y, self.y.iloc[train])
            self.assertEqual(len(X), len(train))
            np.testing.assert_allclose(np.asarray(X).mean(axis=0), 0, atol=1e-12)
            observed.append(len(X))
            return original(selector, X, y)

        model = self.pipeline()
        with threadpool_limits(limits=1), patch.object(VoteSelector, "fit", checked_fit):
            _, proba = cross_validate_pipeline(model, self.X, self.y, cv,
                                               build_scorers(["roc_auc"]), "roc_auc")
        self.assertEqual(observed, [60, 60, 60])
        self.assertTrue(np.isfinite(proba).all())
        self.assertFalse(hasattr(model.named_steps["select"], "support_"))

    def test_cache_reuses_same_data_but_refits_different_training_rows(self):
        calls = []
        original = VoteSelector.fit

        def record(selector, X, y):
            calls.append(len(X))
            return original(selector, X, y)

        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            model = self.pipeline(memory=directory)
            with patch.object(VoteSelector, "fit", record):
                first = clone(model).fit(self.X, self.y)
                second = clone(model).set_params(classifier__C=.5).fit(self.X, self.y)
                self.assertEqual(calls, [90])
                clone(model).fit(self.X.iloc[:-1], self.y.iloc[:-1])
                self.assertEqual(calls, [90, 89])
                clone(model).fit(self.X, 1 - self.y)
                self.assertEqual(calls, [90, 89, 90])
            selector = second.named_steps["select"]
            names = second.named_steps["preprocessing"].get_feature_names_out()
            self.assertEqual(len(selector.get_feature_names_out(names)), selector.get_support().sum())
            self.assertTrue((selector.votes_ >= 0).all() and (selector.votes_ <= 3).all())

    def test_corrected_test_uses_paired_differences_and_sign(self):
        result = corrected_paired_test([.70, .72, .74, .71, .73], [.69, .70, .71, .67, .68])
        self.assertAlmostEqual(result["mean_difference"], .03)
        self.assertAlmostEqual(result["corrected_standard_error"], np.sqrt((.2 + .25) * .00025))
        with self.assertRaises(ValueError):
            corrected_paired_test([.7, .8], [.7])


if __name__ == "__main__":
    unittest.main()
