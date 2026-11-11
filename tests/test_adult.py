"""Adult cleaning, inference row preservation, and optional fairness baseline."""
import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd

from main import load_config
from src.cleaning import clean_dataset
from src.evaluate import fairness_report
from src.inference import prepare_new_rows
from src.preprocess import preprocess


class AdultTests(unittest.TestCase):
    def test_cleaning_and_feature_roles(self):
        config = load_config('config_adult.yaml')
        raw = pd.read_csv(config['data']['path'])
        cleaned = clean_dataset(raw, deepcopy(config['diagnostics']))
        self.assertEqual(cleaned[['workclass', 'occupation', 'native_country']].isna().sum().tolist(),
                         [379, 381, 86])
        self.assertEqual((cleaned.age == 17).sum(), 74)
        self.assertEqual((cleaned.capital_gain == 99999).sum(), 29)
        X, _, _, _, _, _, _ = preprocess(raw, config['data']['target'],
            config['data']['sensitive_attr'], config['data']['drop_columns'],
            deepcopy(config['diagnostics']), config['preprocessing'], **config['split'])
        self.assertTrue(all(f'v{i:02}' in X for i in range(1, 26)))
        self.assertTrue(set(['id', 'fnlwgt', 'education', 'sex', 'race']).isdisjoint(X.columns))
        new = pd.read_csv(config['data']['prediction_path'])
        # Repeated observations and identifiers must survive inference unchanged.
        new = pd.concat([new, new.iloc[[0]]], ignore_index=True)
        prepared = prepare_new_rows(new, X.columns, config)
        self.assertEqual(len(prepared), 2001)
        self.assertEqual(prepared.columns.tolist(), X.columns.tolist())

    def test_fairness_without_compas(self):
        with redirect_stdout(StringIO()):
            report = fairness_report([0, 0, 1, 1], [.1, .8, .9, .2],
                pd.DataFrame({'sex': ['female', 'male', 'female', 'male']}), sensitive_attr='sex')
        self.assertIn('by sex', report)
        self.assertNotIn("COMPAS's own score", report)


if __name__ == '__main__':
    unittest.main()
