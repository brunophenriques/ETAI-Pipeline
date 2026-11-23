"""Imputer fitting and persisted prediction preserve the inference contract."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np
import pandas as pd

from main import load_config
from predict import predict
from src.model import build_pipeline
from src.preprocess import preprocess, build_feature_preprocessor
from src.imputation import ForestProximityImputer
from src.inference import save_library_versions, library_versions


class Week10Tests(unittest.TestCase):
    def test_categories_improve_a_predictable_missing_numeric_feature(self):
        train = pd.DataFrame({"age": [20., 21., 19., 60., 61., 59.],
                              "group": ["young"] * 3 + ["old"] * 3})
        query = pd.DataFrame({"age": [np.nan, np.nan], "group": ["young", "old"]})
        prep = build_feature_preprocessor(train, {"scaler": "standard", "imputation": {
            "numeric_strategy": "knn", "n_neighbors": 3, "include_categories": True}})
        filled = prep.fit(train).transform(query)
        self.assertLess(filled[0, 0], filled[1, 0])
        self.assertEqual(len(prep.get_feature_names_out()), filled.shape[1])
        self.assertTrue(np.isfinite(prep.transform(pd.DataFrame({"age": [np.nan], "group": ["unknown"]}))).all())

    def test_multiple_imputation_averages_independent_saved_members(self):
        X = pd.DataFrame({"age": [20., 22., np.nan, 35., 40., np.nan, 60., 65.],
                          "count": [0., 1., 2., 3., 4., 5., 6., 7.]})
        prep = build_feature_preprocessor(X, {"scaler": "standard", "imputation": {
            "numeric_strategy": "iterative", "sample_posterior": True, "max_iter": 3}})
        model = build_pipeline(prep, {"type": "decision_tree", "params": {"max_depth": 2}},
            multiple_imputation={"enabled": True, "n_imputations": 3, "random_state": 42})
        model.fit(X, np.array([0, 0, 1, 0, 1, 0, 1, 1]))
        probabilities = model.predict_proba(X)
        np.testing.assert_allclose(model.predict_proba(X), probabilities)
        seeds = [member.named_steps['preprocessing'].named_transformers_['numerical'].named_steps['imputer'].random_state
                 for member in model.pipelines_]
        self.assertEqual(seeds, [42, 43, 44])
        from sklearn.utils import check_random_state
        for member in model.pipelines_:
            imputer = member.named_steps['preprocessing'].named_transformers_['numerical'].named_steps['imputer']
            imputer.random_state_ = check_random_state(imputer.random_state)
        expected = np.mean([member.predict_proba(X) for member in model.pipelines_], axis=0)
        np.testing.assert_allclose(probabilities, expected)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict_proba(X), probabilities)

    def test_proximity_uses_training_donors_and_never_overwrites_observed_values(self):
        train = pd.DataFrame({"age": [20., 21., 60., 61.], "priors_count": [1., 2., 8., 9.]})
        imputer = ForestProximityImputer(n_estimators=10).fit(train, [0, 1, 0, 1])
        query = pd.DataFrame({"age": [20., 60.], "priors_count": [np.nan, 999.]})
        filled = imputer.transform(query)
        self.assertTrue(1 <= filled.priors_count.iloc[0] <= 9)
        self.assertEqual(filled.priors_count.iloc[1], 999.)
        self.assertEqual(imputer.predictors_, ["age"])
        pd.testing.assert_frame_equal(imputer.transform(query.iloc[:1]), filled.iloc[:1])

    def test_proximity_pipeline_predicts_new_people_after_reload(self):
        config = load_config()
        config['preprocessing']['imputation']['numeric_strategy'] = 'forest_proximity'
        config['preprocessing']['imputation']['n_estimators'] = 5
        with redirect_stdout(StringIO()):
            X, _, y, _, _, _, prep = preprocess(pd.read_csv(config['data']['path']),
                config['data']['target'], config['data']['sensitive_attr'],
                config['data']['drop_columns'], config['diagnostics'],
                config['preprocessing'], **config['split'])
        model = build_pipeline(prep, {'type': 'decision_tree', 'params': {'max_depth': 3}})
        model.fit(X, y)
        self.assertNotIn('age_x_priors', model.named_steps['preprocessing'].named_steps['proximity'].predictors_)
        self.assertEqual(len(prep.get_feature_names_out()), model.named_steps['classifier'].n_features_in_)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'proximity.joblib'
            joblib.dump(model, path)
            with redirect_stdout(StringIO()):
                output = predict(pd.read_csv('data/compas_new_people.csv'), joblib.load(path), config)
            self.assertEqual(len(output), 20)
            self.assertTrue(output.probability.between(0, 1).all())

    def test_library_versions_are_written_next_to_model(self):
        import json
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            save_library_versions(path)
            saved = json.loads(path.with_suffix('.versions.json').read_text())
            self.assertEqual(saved, library_versions())
            self.assertIn('scikit-learn', saved)

    def test_cli_rejects_version_mismatch_before_loading_model(self):
        import json
        import subprocess
        import sys
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            path.with_suffix('.config.json').write_text('{}')
            path.with_suffix('.versions.json').write_text(json.dumps({'scikit-learn': 'incompatible'}))
            result = subprocess.run([sys.executable, 'predict.py', 'unused.csv', '--model', str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Model library versions differ', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_all_imputers_handle_unseen_missing_values(self):
        train = pd.DataFrame({"a": [1., 2., np.nan, 4., 5., 6.],
                              "b": [2., 4., 6., 8., np.nan, 12.]})
        for strategy in ("median", "knn", "iterative", "random_forest"):
            with self.subTest(strategy=strategy):
                prep = build_feature_preprocessor(train, {"scaler": "standard",
                    "imputation": {"numeric_strategy": strategy, "n_estimators": 5}})
                prep.fit(train)
                result = prep.transform(pd.DataFrame({"a": [np.nan], "b": [10.]}))
                self.assertEqual(result.shape, (1, 2))
                self.assertTrue(np.isfinite(result).all())

    def test_saved_pipeline_predicts_twenty_rows_and_keeps_duplicates(self):
        config = load_config()
        with redirect_stdout(StringIO()):
            X, _, y, _, _, _, prep = preprocess(pd.read_csv(config["data"]["path"]),
                config["data"]["target"], config["data"]["sensitive_attr"],
                config["data"]["drop_columns"], config["diagnostics"],
                config["preprocessing"], **config["split"])
        pipeline = build_pipeline(prep, {"type": "decision_tree", "params": {"max_depth": 3}})
        pipeline.fit(X, y)
        raw = pd.read_csv("data/compas_new_people.csv")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(pipeline, path)
            with redirect_stdout(StringIO()):
                output = predict(raw, joblib.load(path), config)
                expected = predict(raw, pipeline, config)
            pd.testing.assert_frame_equal(output, expected)
        self.assertEqual(len(output), 20)
        self.assertEqual(output.id.tolist(), raw.id.tolist())
        self.assertTrue(output.probability.between(0, 1).all())
        duplicates = raw.drop(columns="id").duplicated(keep=False)
        self.assertTrue(duplicates.any())
        self.assertEqual(output.loc[duplicates, "probability"].nunique(), 1)
        with self.assertRaisesRegex(ValueError, "missing features"):
            with redirect_stdout(StringIO()):
                predict(raw.drop(columns="age"), pipeline, config)
