"""Verify sequential model routing without training or modifying the YAML."""
import unittest
from copy import deepcopy
from unittest.mock import patch

from main import load_config
from scripts.run_adult import run


class AdultRunnerTests(unittest.TestCase):
    def test_model_settings_and_separate_predictions(self):
        base = load_config('config_adult.yaml')
        before = deepcopy(base)
        def result(path, *, config):
            return {'metric': 'roc_auc', 'mean': .9}
        with patch('scripts.run_adult.main.load_config', return_value=base), \
                patch('scripts.run_adult.main.main', side_effect=result) as execute:
            run(models=['dummy', 'logistic_regression', 'xgboost'], trials=2)
        configs = [call.kwargs['config'] for call in execute.call_args_list]
        self.assertEqual([c['model']['type'] for c in configs],
                         ['dummy', 'logistic_regression', 'xgboost'])
        self.assertFalse(configs[0]['tuning']['enabled'])
        self.assertEqual(configs[2]['model'], base['model'])
        self.assertNotIn('tree_method', configs[1]['model']['params'])
        self.assertEqual(len({c['output']['predictions'] for c in configs}), 3)
        for c in configs:
            self.assertEqual(c['preprocessing'], base['preprocessing'])
            self.assertEqual(c['cv'], base['cv'])
            self.assertEqual(c['feature_selection'], base['feature_selection'])
        self.assertEqual(base, before)


if __name__ == '__main__':
    unittest.main()
