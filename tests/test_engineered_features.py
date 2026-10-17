"""Check fixed feature formulas, missingness and reserved split stability."""

import copy
import unittest

import numpy as np
import pandas as pd

from main import load_config
from src.data import load_data
from src.preprocess import add_engineered_features, preprocess


class EngineeredFeatureTests(unittest.TestCase):
    def test_formulas_missingness_and_row_independence(self):
        frame = pd.DataFrame({
            "age": [23., 47., 17., np.nan],
            "priors_count": [6., 6., np.nan, 2.],
            "juv_fel_count": [2., 0., np.nan, 1.],
            "juv_misd_count": [1., 0., 2., np.nan],
            "juv_other_count": [3., 0., 1., 2.],
        }, index=[8, 4, 2, 1])
        original = frame.copy(deep=True)
        result = add_engineered_features(frame)
        np.testing.assert_allclose(result["priors_per_adult_year"], [1., .2, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(result["juvenile_offenses_total"], [6., 0., np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(result["age_x_priors"], [138., 282., np.nan, np.nan], equal_nan=True)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(result.iloc[[1]], add_engineered_features(frame.iloc[[1]]))
        edge = frame.iloc[[0]].copy()
        edge["age"] = 17.
        self.assertEqual(add_engineered_features(edge)["priors_per_adult_year"].iloc[0], 6.)

    def test_switch_preserves_splits_and_returns_unfitted_preprocessor(self):
        config = load_config()
        raw = load_data(config["data"]["path"])
        outputs = []
        for enabled in [False, True]:
            prep = copy.deepcopy(config["preprocessing"])
            prep["engineered_features"] = {"enabled": enabled}
            outputs.append(preprocess(
                raw, config["data"]["target"], config["data"]["sensitive_attr"],
                config["data"]["drop_columns"], copy.deepcopy(config["diagnostics"]),
                prep, **config["split"],
            ))
        before, after = outputs
        self.assertEqual(set(after[0].columns) - set(before[0].columns), {
            "priors_per_adult_year", "juvenile_offenses_total", "age_x_priors"
        })
        for idx in [0, 1]:
            self.assertTrue(before[idx].index.equals(after[idx].index))
            pd.testing.assert_frame_equal(before[idx], after[idx][before[idx].columns])
        for idx in [2, 3]:
            pd.testing.assert_series_equal(before[idx], after[idx])
        for idx in [4, 5]:
            pd.testing.assert_frame_equal(before[idx], after[idx])
        self.assertFalse(hasattr(after[6], "transformers_"))
        numerical_columns = after[6].transformers[0][2]
        self.assertIn("juvenile_offenses_total", numerical_columns)


if __name__ == "__main__":
    unittest.main()
