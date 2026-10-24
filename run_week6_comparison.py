"""Run the Week 6 baseline and 50-trial comparisons through main.main().

Uses in-memory configuration overrides, preserving config.yaml. Reports and
the exact configurations are kept under results/week6_comparison/.
Existing results are never overwritten. Run with python run_week6_comparison.py.
"""

import copy
import ast
import io
import json
import re
from hashlib import sha256
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import optuna
import pandas as pd
import sklearn

import main


OUTPUT = Path("results/week6_comparison")
MODELS = {
    "logistic_regression": {"max_iter": 1000, "solver": "lbfgs"},
    "decision_tree": {"random_state": 42},
    "random_forest": {"random_state": 42, "n_jobs": 1},
    "oblique_random_forest": {"random_state": 42, "n_jobs": 1},
    "adaboost": {"random_state": 42},
}
TUNED_BASE_PARAMS = {
    "logistic_regression": MODELS["logistic_regression"],
    # Retain the Week 5 fixed min_samples_split setting during DT tuning.
    "decision_tree": {"min_samples_split": 10, "random_state": 42},
    "random_forest": {"n_estimators": 300, "random_state": 42, "n_jobs": 1},
    "oblique_random_forest": {"n_estimators": 100, "n_components": 10,
                              "random_state": 42, "n_jobs": 1},
    "adaboost": {"max_depth": 1, "n_estimators": 50,
                 "learning_rate": 1.0, "random_state": 42},
}


def run_one(base_config, name, params, tuned):
    label = f"{name}_{'tuned_50trials' if tuned else 'default_cv'}_seed42"
    report_path = OUTPUT / f"{label}.txt"
    config_path = OUTPUT / f"{label}.json"
    if report_path.exists():
        print(f"Already saved: {report_path}", flush=True)
        return
    config = copy.deepcopy(base_config)
    config["model"] = {"type": name, "params": copy.deepcopy(params)}
    config["split"].update(test_size=0.2, random_state=42)
    config["cv"].update(n_splits=5, shuffle=True, random_state=42,
                        scoring="accuracy", metrics=["accuracy"],
                        selection_metric="accuracy", n_jobs=-1)
    config["tuning"].update(enabled=tuned, n_trials=50, n_splits=5, random_state=42)
    config["output"]["results_dir"] = str(OUTPUT)
    config["output"]["scoreboard"] = str(OUTPUT / "runs.csv")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = perf_counter()

    def save_report(results_dir, run_config, report):
        header = (
            f"Model: {name}\n"
            f"Starting model parameters: {params}\n"
            f"CV settings: {run_config['cv']}\n"
            f"Tuning enabled: {tuned}; trials per study: {50 if tuned else 0}\n"
            f"Full configuration: {config_path.name}\n"
            f"Elapsed seconds: {perf_counter() - started:.2f}\n\n"
        )
        with report_path.open("x", encoding="utf-8") as file:
            file.write(header + report)
        return str(report_path)

    print(f"Running {label}", flush=True)
    with (patch("main.load_config", return_value=config),
          patch("main.save_run", side_effect=save_report),
          redirect_stdout(io.StringIO())):
        main.main()
    print(f"Saved {label} ({perf_counter() - started:.1f}s)", flush=True)


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = main.load_config()
    # This table compares model families before the feature-engineering challenge.
    config["preprocessing"]["engineered_features"] = {"enabled": False}
    fingerprint = sha256(Path(config["data"]["path"]).read_bytes()).hexdigest()
    provenance_path = OUTPUT / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance["dataset_sha256"] != fingerprint:
            raise ValueError("The dataset changed; archive the existing reports before rerunning.")
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        for name, params in MODELS.items():
            run_one(config, name, params, tuned=False)
        for name, params in TUNED_BASE_PARAMS.items():
            run_one(config, name, params, tuned=True)
    finally:
        optuna.logging.set_verbosity(verbosity)
    summarize_reports()
    if not provenance_path.exists():
        provenance_path.write_text(json.dumps({
            "dataset": config["data"]["path"],
            "dataset_sha256": fingerprint,
            "sklearn_version": sklearn.__version__,
            "optuna_version": optuna.__version__,
            "split": {"test_size": 0.2, "random_state": 42},
            "cv": {"n_splits": 5, "shuffle": True, "random_state": 42,
                   "scoring": "accuracy", "n_jobs": -1},
            "preprocessing": config["preprocessing"],
            "tuning_trials_per_study": 50,
            "tuning_inner_folds": 5,
            "sampler_seed": 42,
            "reserved_test_evaluated": False,
        }, indent=2), encoding="utf-8")


def summarize_reports():
    """Collect displayed report values without re-evaluating any model."""
    rows = []
    for name in MODELS:
        row = {"model": name}
        for tuned in [False, True]:
            mode = "tuned_50trials" if tuned else "default_cv"
            path = OUTPUT / f"{name}_{mode}_seed42.txt"
            report = path.read_text(encoding="utf-8")
            score = re.search(r"(?:Nested )?CV accuracy \(mean \+/- std\): ([\d.]+) \+/- ([\d.]+)", report)
            gap = re.search(r"Mean train-validation gap: ([+\-\d.]+)", report)
            if score is None or gap is None:
                raise ValueError(f"Missing scores in {path}")
            row[f"{mode}_mean"] = float(score[1])
            row[f"{mode}_std"] = float(score[2])
            row[f"{mode}_gap"] = float(gap[1])
            row[f"{mode}_report"] = str(path)
            if tuned:
                settings = re.search(r"Tuning settings: (.+)", report)
                if settings is None or ast.literal_eval(settings[1])["n_trials"] != 50:
                    raise ValueError(f"Expected 50 trials in {path}")
                row["best_trial"] = float(re.search(r"Best Optuna trial score: ([\d.]+)", report)[1])
                row["chosen_parameters"] = ast.literal_eval(re.search(r"Selected parameters: (.+)", report)[1])
            if "Reserved test set: 1443 rows (not evaluated)." not in report:
                raise ValueError(f"Unexpected test-set summary in {path}")
        rows.append(row)
    (OUTPUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUTPUT / "summary.csv", index=False)
    return rows


if __name__ == "__main__":
    run()
