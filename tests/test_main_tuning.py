"""Exercise both entry-point paths without loading or evaluating test rows."""

import copy
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import optuna
import pandas as pd

import main
from src.preprocess import build_feature_preprocessor


class MainTuningTests(unittest.TestCase):
    def test_enabled_and_disabled_paths(self):
        base_config = main.load_config()
        X = pd.DataFrame({"age": [float(i) for i in range(24)]})
        y = pd.Series([0, 1] * 12)
        extras = pd.DataFrame({"race": ["Other"] * 24, "score_text": ["Low"] * 24})

        class ReservedRows:
            # Only the size is available: attempting to use rows would fail.
            def __len__(self):
                return 4

        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        try:
            for enabled in [True, False]:
                with self.subTest(tuning_enabled=enabled):
                    config = copy.deepcopy(base_config)
                    config["cv"].update(n_splits=3, n_jobs=1, scoring="balanced_accuracy")
                    config["tuning"].update(enabled=enabled, n_trials=2, n_splits=2)
                    prep = build_feature_preprocessor(X, config["preprocessing"])
                    with (
                        patch("main.load_config", return_value=config),
                        patch("main.load_data", return_value=None),
                        patch("main.preprocess", return_value=(X, ReservedRows(), y, None, extras, None, prep)),
                        patch("main.save_run", return_value="synthetic-report.txt") as save,
                        patch("main.nested_cross_validate", wraps=main.nested_cross_validate) as nested,
                        patch("main.cross_validate_pipeline", wraps=main.cross_validate_pipeline) as ordinary,
                        patch("main.tune_pipeline", wraps=main.tune_pipeline) as final_search,
                        patch("main.build_pipeline", wraps=main.build_pipeline) as build,
                        redirect_stdout(io.StringIO()),
                    ):
                        main.main()

                    report = save.call_args.args[2]
                    self.assertIn("balanced_accuracy", report)
                    self.assertIn("Reserved test set: 4 rows (not evaluated)", report)
                    self.assertEqual(nested.call_count, int(enabled))
                    self.assertEqual(ordinary.call_count, int(not enabled))
                    self.assertEqual(final_search.call_count, int(enabled))
                    if enabled:
                        self.assertIs(final_search.call_args.args[1], X)
                        self.assertIs(final_search.call_args.args[2], y)
                        self.assertIn("nested out-of-fold predictions", report)
                        self.assertIn("Selected parameters:", report)
                        self.assertIn("Optimism check", report)
                    else:
                        self.assertNotIn("Optimism check", report)
                    self.assertEqual(build.call_count, 1)
        finally:
            optuna.logging.set_verbosity(previous_verbosity)


if __name__ == "__main__":
    unittest.main()
