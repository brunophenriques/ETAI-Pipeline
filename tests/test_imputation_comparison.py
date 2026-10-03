"""Checks for KNN donor isolation and the variance-corrected paired test."""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

from src.evaluate import corrected_paired_comparison
from src.preprocess import build_feature_preprocessor


class ImputationComparisonTests(unittest.TestCase):
    def test_validation_rows_do_not_supply_neighbors_or_scaling_statistics(self):
        train = pd.DataFrame({"age": [0., 10., 20.], "count": [0., 100., 200.]})
        prep = build_feature_preprocessor(train, {
            "scaler": "standard", "numerical_imputation": "knn", "n_neighbors": 1,
        })
        self.assertEqual(list(prep.transformers[0][1].named_steps), ["scaler", "imputer"])
        prep.fit(train)
        numeric = prep.named_transformers_["numerical"]
        validation = pd.DataFrame({"age": [10., 10.], "count": [np.nan, 99999.]})
        transformed = prep.transform(validation)
        # The fully observed validation row is not allowed to donate 99999.
        original_units = numeric.named_steps["scaler"].inverse_transform(transformed)
        self.assertAlmostEqual(original_units[0, 1], 100.)
        np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, [10., 100.])

    def test_invalid_knn_configuration(self):
        frame = pd.DataFrame({"age": [20., 30.]})
        with self.assertRaisesRegex(ValueError, "requires scaler"):
            build_feature_preprocessor(frame, {"numerical_imputation": "knn"})
        for neighbors in [0, -1, True, 1.5]:
            with self.assertRaisesRegex(ValueError, "positive integer"):
                build_feature_preprocessor(frame, {
                    "scaler": "standard", "numerical_imputation": "knn",
                    "n_neighbors": neighbors,
                })

    def test_corrected_uncertainty_exceeds_independent_fold_estimate(self):
        differences = np.array([.01, .02, .03, .04, .05])
        result = corrected_paired_comparison(differences, .25)
        # Mean=.03, sample variance=.00025, correction=1/5+1/4=.45.
        expected_se = np.sqrt(.00025 * .45)
        self.assertAlmostEqual(result["mean_difference_knn_minus_median"], .03)
        self.assertAlmostEqual(result["corrected_standard_error"], expected_se)
        self.assertGreater(expected_se, differences.std(ddof=1) / np.sqrt(5))
        self.assertAlmostEqual(result["two_sided_p_value"], 2 * t.sf(.03 / expected_se, 4))

    def test_identical_scores_and_invalid_differences(self):
        result = corrected_paired_comparison([0., 0., 0., 0., 0.], .25)
        self.assertEqual(result["two_sided_p_value"], 1.)
        self.assertEqual(result["ci95"], [0., 0.])
        for differences in [[.1], [.1, np.nan]]:
            with self.assertRaises(ValueError):
                corrected_paired_comparison(differences, .25)


if __name__ == "__main__":
    unittest.main()
