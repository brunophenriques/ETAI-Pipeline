"""Development-only multimetric CV, nested tuning, and an append-only scoreboard."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.data import load_data
from src.preprocess import preprocess
from src.model import build_pipeline
from src.evaluate import (build_scorers, cross_validate_pipeline, cv_report,
                          fairness_report, oof_classification_report)
from src.results import save_run, save_scores
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@threadpool_limits.wrap(limits=1)
def main():
    config = load_config()
    config["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_type = config["model"]["type"]
    cv_config = config["cv"]
    selection = cv_config.get("selection_metric", cv_config.get("scoring", "accuracy"))
    metrics = cv_config.get("metrics", [selection])
    scorers = build_scorers(metrics)
    if selection not in scorers:
        raise ValueError("selection_metric must be included in cv.metrics.")
    tuning = config.get("tuning", {})
    enabled = tuning.get("enabled", False)
    space = tuning.get("search_spaces", {}).get(model_type)
    if enabled and not space:
        raise ValueError(f"No tuning search space configured for {model_type}; disable tuning for dummy.")

    raw = load_data(config["data"]["path"])
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test, preprocessor = preprocess(
        raw, config["data"]["target"], config["data"]["sensitive_attr"],
        config["data"]["drop_columns"], config["diagnostics"], config["preprocessing"],
        **config["split"],
    )
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                        random_state=cv_config.get("random_state") if shuffle else None)
    # Materialized once: baseline, floor and nested evaluation share validation rows.
    folds = list(cv.split(X_dev, y_dev))
    n_jobs = cv_config.get("n_jobs", 1)
    model = build_pipeline(preprocessor, config["model"])
    floor_pipeline = build_pipeline(clone(preprocessor), {"type": "dummy", "params": {"strategy": "prior"}})
    print(f"Model: {model_type}; selection: {selection}; development rows: {len(X_dev)}", flush=True)
    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline, X_dev, y_dev, folds, {selection: scorers[selection]}, selection, n_jobs,
    )
    baseline, proba = cross_validate_pipeline(model, X_dev, y_dev, folds, scorers, selection, n_jobs)
    floor = float(floor_scores["validation"].mean())
    report = cv_report(baseline, metrics, selection)
    report += f"\n\nNo-model floor (dummy, same folds): {selection} = {floor:.4f}"
    final_scores = baseline
    study = None
    context = "development set, out-of-fold predictions"
    if enabled:
        inner = StratifiedKFold(n_splits=tuning["n_splits"], shuffle=True,
                                random_state=tuning["random_state"])
        args = dict(scoring=selection, search_space=space, n_trials=tuning["n_trials"],
                    random_state=tuning["random_state"], n_jobs=n_jobs)
        print(f"Nested tuning: {tuning['n_trials']} trials per study", flush=True)
        final_scores, proba = nested_cross_validate(
            model, X_dev, y_dev, folds, inner, **args, scorers=scorers,
        )
        report += "\n\n" + cv_report(final_scores, metrics, selection, "Nested cross-validation")
        context = "development set, nested out-of-fold predictions"
        model, study = tune_pipeline(model, X_dev, y_dev, inner,
                                    **{**args, "scoring": scorers[selection]})
        report += "\n\n" + tuning_report(study, final_scores, selection)
    report += "\n\n" + oof_classification_report(y_dev, proba, context=context, probabilities=True)
    report += "\n\n" + fairness_report(y_dev, proba, extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"], context=context, probabilities=True)
    model.fit(X_dev, y_dev)
    report += (f"\n\nFinal pipeline refit on {len(X_dev)} development rows.\n"
               f"Reserved test set: {len(X_test)} rows (not evaluated).\n")
    results_dir = config.get("output", {}).get("results_dir", "results/week7")
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(results_dir) / f"{model_type}_{config['run_id']}"
    baseline.to_csv(str(stem) + "_default_folds.csv", index=False)
    if enabled:
        final_scores.to_csv(str(stem) + "_nested_folds.csv", index=False)
        study.trials_dataframe().to_csv(str(stem) + "_trials.csv", index=False)
    pd.DataFrame({"row_index": X_dev.index, "y_true": y_dev.to_numpy(),
                  "oof_probability": proba, "oof_prediction": (proba >= .5).astype(int)
                  }).to_csv(str(stem) + "_oof.csv", index=False)

    # Include recipe identity so runs with different data/features/folds are not ranked together.
    diagnostics = {k: v for k, v in config["diagnostics"].items() if k != "num_missing_rows"}
    data_path = Path(config["data"]["path"])
    identity = dict(data=config["data"], diagnostics=diagnostics,
                    preprocessing=config["preprocessing"], split=config["split"], cv=cv_config,
                    dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest() if data_path.exists() else "synthetic")
    comparison_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    row = dict(run=config["run_id"], model=model_type, tuned=enabled, metric=selection,
               comparison_id=comparison_id, data=config["data"]["path"],
               n_trials=tuning["n_trials"] if enabled else 0, floor=floor,
               floor_std=float(floor_scores["validation"].std(ddof=1)),
               untuned=float(baseline["validation"].mean()),
               untuned_std=float(baseline["validation"].std(ddof=1)),
               untuned_gap=float(baseline["gap"].mean()),
               mean=float(final_scores["validation"].mean()),
               std=float(final_scores["validation"].std(ddof=1)),
               gap=float(final_scores["gap"].mean()),
               best_trial=float(study.best_value) if study else None,
               params=json.dumps(config["model"].get("params", {}), sort_keys=True),
               tuned_params=json.dumps(study.best_params, sort_keys=True) if study else "",
               report=str(stem) + ".txt", test_evaluated=False)
    for metric in metrics:
        row[f"{metric}_mean"] = float(final_scores[metric].mean())
        row[f"{metric}_std"] = float(final_scores[metric].std(ddof=1))
        row[f"untuned_{metric}"] = float(baseline[metric].mean())
    for i, value in enumerate(final_scores["validation"], 1):
        row[f"fold_{i}"] = float(value)
    path = save_run(results_dir, config, report)
    row["report"] = Path(path).as_posix()
    board = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    board = board[board["comparison_id"] == comparison_id].sort_values("mean", ascending=False)
    print(report)
    print("\nScoreboard (same data, features, folds and selection metric):")
    print(board[["model", "tuned", "untuned", "mean", "std", "floor"]].to_string(index=False))
    print(f"Full results saved to {path}")
    return row


if __name__ == "__main__":
    main()
