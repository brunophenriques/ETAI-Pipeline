"""Run the Week 7 models through main.main(), preserving config.yaml.

Every tuned model receives 15 trials, with the same data, features and 5x5
stratified folds. The CSV is append-only; completed models can be resumed.
"""
import copy
import io
import json
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import optuna
import pandas as pd

import main
from src.tuning import tune_pipeline


MODELS = {
    "dummy": {"strategy": "prior"},
    "logistic_regression": {"max_iter": 2000},
    "naive_bayes": {},
    "decision_tree": {"random_state": 42},
    "hist_gradient_boosting": {"random_state": 42},
    "gradient_boosting": {"random_state": 42},
    "random_forest": {"random_state": 42, "n_jobs": 1},
    "oblique_random_forest": {"random_state": 42, "n_jobs": 1},
    "adaboost": {"random_state": 42},
    "adaboost_logistic_regression": {"random_state": 42, "max_iter": 2000},
}


def run():
    base = main.load_config()
    base["cv"].update(n_splits=5, shuffle=True, random_state=42, n_jobs=-1,
                      selection_metric="roc_auc")
    base["tuning"].update(n_trials=15, n_splits=5, random_state=42)
    base["output"] = {"results_dir": "results/week7", "scoreboard": "results/runs.csv"}
    folder = Path("results/week7")
    folder.mkdir(parents=True, exist_ok=True)
    plan = {"dataset_sha256": sha256(Path(base["data"]["path"]).read_bytes()).hexdigest(),
            "preprocessing": base["preprocessing"], "split": base["split"],
            "cv": base["cv"], "trials": 15, "models": MODELS,
            "search_spaces": base["tuning"]["search_spaces"]}
    manifest = folder / "experiment.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != plan:
        raise ValueError("The Week 7 experiment settings changed; archive its outputs before resuming.")
    manifest.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        for name, params in MODELS.items():
            if Path("results/runs.csv").exists():
                board = pd.read_csv("results/runs.csv")
                done = board[(board["model"] == name) & (board["metric"] == "roc_auc")]
                if len(done) and Path(done.iloc[-1]["report"]).exists():
                    print(f"Already saved: {name}", flush=True)
                    continue
            config = copy.deepcopy(base)
            config["model"] = {"type": name, "params": params}
            config["tuning"]["enabled"] = name != "dummy"
            started = perf_counter()
            studies = 0

            def progress(*args, **kwargs):
                nonlocal studies
                result = tune_pipeline(*args, **kwargs)
                studies += 1
                # stdout is redirected to keep the complete run log concise.
                import sys
                print(f"{name}: search {studies}/6 complete", file=sys.stderr, flush=True)
                return result

            print(f"Running {name} ({15 if name != 'dummy' else 0} trials)", flush=True)
            with (patch("main.load_config", return_value=config),
                  patch("main.tune_pipeline", side_effect=progress),
                  patch("src.tuning.tune_pipeline", side_effect=progress),
                  (folder / f"{name}_console.txt").open("w", encoding="utf-8") as log,
                  redirect_stdout(log)):
                row = main.main()
            print(f"Saved {name}: ROC-AUC {row['mean']:.4f} +/- {row['std']:.4f} "
                  f"({perf_counter() - started:.1f}s)", flush=True)
    finally:
        optuna.logging.set_verbosity(verbosity)


if __name__ == "__main__":
    run()
