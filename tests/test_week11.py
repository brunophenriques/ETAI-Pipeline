"""Anomaly scoring must remain fold-local and preserve inference rows."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.model import build_pipeline
from src.preprocess import AnomalyScore, build_feature_preprocessor
from src.outliers import QuantileClipper


class AnomalyTests(unittest.TestCase):
    def test_clipper_learns_training_limits_and_preserves_gaps(self):
        X = np.array([[0., np.nan], [1., np.nan], [2., np.nan], [100., np.nan]])
        clip = QuantileClipper(lower=.25, upper=.75).fit(X)
        limits = clip.upper_.copy()
        out = clip.transform(np.array([[10000., np.nan], [-10000., np.nan]]))
        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(np.isnan(out[:, 1]).all())
        self.assertEqual(out[0, 0], np.quantile(X[:, 0], .75))
        self.assertEqual(out[1, 0], np.quantile(X[:, 0], .25))
        np.testing.assert_array_equal(limits, clip.upper_)
        self.assertEqual(len(clip.get_feature_names_out()), 2)

    def test_training_removal_does_not_remove_validation_or_prediction_rows(self):
        X = pd.DataFrame(self.X, columns=["a", "b", "c"])
        pipeline = build_pipeline(build_feature_preprocessor(X, {"scaler": "standard"}),
            {"type": "decision_tree", "params": {"max_depth": 3}},
            {"enabled": True, "method": "kbest", "k": 2},
            anomaly_config={"method": "none", "training_filter": {"enabled": True, "fraction": .1}})
        fitted = clone(pipeline).fit(X, self.y)
        self.assertEqual(len(fitted.removed_indices_), 6)
        self.assertEqual(fitted.named_steps["classifier"].tree_.n_node_samples[0], 54)
        self.assertEqual(len(fitted.predict(X)), 60)
        query = pd.concat([X.iloc[:10], X.iloc[:10]])
        probabilities = fitted.predict_proba(query)
        self.assertEqual(len(probabilities), 20)
        np.testing.assert_array_equal(probabilities[:10], probabilities[10:])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "filtered.joblib"
            joblib.dump(fitted, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(query), probabilities)

    def setUp(self):
        self.X = np.random.default_rng(42).normal(size=(60, 3))
        self.y = np.tile([0, 1], 30)

    def test_lof_training_and_novelty_use_the_right_scores(self):
        detector = AnomalyScore(method="lof", n_neighbors=20)
        train = detector.fit_transform(self.X, self.y)
        np.testing.assert_array_equal(train[:, :-1], self.X)
        np.testing.assert_allclose(train[:, -1], -detector.detector_.negative_outlier_factor_)
        query = self.X[:2] + 10
        actual = detector.transform(query)
        np.testing.assert_allclose(actual[:, -1], -detector.detector_.score_samples(query))
        self.assertEqual(detector.detector_.n_samples_fit_, len(self.X))
        self.assertEqual(actual.shape, (2, 4))

    def test_detector_does_not_use_labels_or_refit_on_validation(self):
        for method in ("isolation_forest", "lof"):
            a = AnomalyScore(method=method, random_state=42).fit(self.X, self.y)
            b = clone(a).fit(self.X, 1-self.y)
            query = np.array([[1., 2., 3.], [1., 2., 3.]])
            np.testing.assert_array_equal(a.transform(query), b.transform(query))
            np.testing.assert_array_equal(a.transform(query)[0], a.transform(query)[1])
            np.testing.assert_allclose(a.transform(query[:1]), a.transform(query)[:1])

    def test_saved_pipeline_feature_names_and_selection_include_anomaly(self):
        X = pd.DataFrame(self.X, columns=["a", "b", "c"])
        for method in ("none", "isolation_forest", "lof"):
            pipeline = build_pipeline(build_feature_preprocessor(X, {"scaler": "standard"}),
                {"type": "decision_tree", "params": {"max_depth": 3, "random_state": 42}},
                {"enabled": True, "method": "kbest", "k": 2},
                anomaly_config={"method": method, "random_state": 42, "n_neighbors": 20})
            pipeline.fit(X, self.y)
            names = pipeline.named_steps["preprocessing"].get_feature_names_out()
            if method != "none":
                names = pipeline.named_steps["anomaly"].get_feature_names_out(names)
                self.assertEqual(names[-1], "anomaly_score")
                self.assertEqual(len(names), 4)
            self.assertEqual(len(pipeline.named_steps["select"].get_feature_names_out(names)), 2)
            with TemporaryDirectory() as directory:
                path = Path(directory) / "pipeline.joblib"
                joblib.dump(pipeline, path)
                loaded = joblib.load(path)
                np.testing.assert_allclose(loaded.predict_proba(X.iloc[:20]), pipeline.predict_proba(X.iloc[:20]))
                self.assertEqual(len(loaded.predict(X.iloc[:20])), 20)

    def test_invalid_method_and_unfitted_transform(self):
        with self.assertRaises(NotFittedError):
            AnomalyScore().transform(self.X)
        with self.assertRaisesRegex(ValueError, "Unknown anomaly"):
            AnomalyScore(method="bad").fit(self.X)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            AnomalyScore(method="lof", n_neighbors=1).fit(self.X)
