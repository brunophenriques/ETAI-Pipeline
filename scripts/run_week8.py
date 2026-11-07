"""Compare tuned XGBoost with and without fold-local voting selection.

Same development data, 5x5 folds, ROC-AUC and 15 trials in each condition.
Reports append to the common scoreboard; existing completed runs can be resumed.
"""
import copy
import json
import sys
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import StratifiedKFold

import main
from src.evaluate import corrected_paired_test
from src.tuning import tune_pipeline


FOLDER = Path("results/week8")


def run():
    base = main.load_config()
    base["model"] = {"type": "xgboost", "params": {
        "random_state": 42, "n_jobs": 1, "tree_method": "hist", "eval_metric": "logloss"}}
    base["cv"].update(n_splits=5, shuffle=True, random_state=42,
                      selection_metric="roc_auc", n_jobs=-1)
    base["tuning"].update(enabled=True, n_trials=15, n_splits=5, random_state=42)
    base["output"] = {"results_dir": str(FOLDER), "scoreboard": "results/runs.csv", "layout": "selection"}
    FOLDER.mkdir(parents=True, exist_ok=True)
    plan = {"dataset_sha256": sha256(Path(base["data"]["path"]).read_bytes()).hexdigest(),
            "data": base["data"], "preprocessing": base["preprocessing"],
            "split": base["split"], "cv": base["cv"], "model": base["model"],
            "trials": 15, "inner_folds": 5, "sampler_seed": 42,
            "selection": base["feature_selection"],
            "search_space": base["tuning"]["search_spaces"]["xgboost"],
            "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                         "optuna": optuna.__version__}}
    manifest = FOLDER / "experiment.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != plan:
        raise ValueError("The Week 8 experiment changed; archive the existing outputs before resuming.")
    manifest.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    summary_path = FOLDER / "runs.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        for selected in [False, True]:
            label = "selection_on" if selected else "selection_off"
            if label in saved and Path(saved[label]["report"]).exists():
                print(f"Already saved: {label}", flush=True)
                continue
            config = copy.deepcopy(base)
            config["feature_selection"]["enabled"] = selected
            studies = 0

            def progress(*args, **kwargs):
                nonlocal studies
                result = tune_pipeline(*args, **kwargs)
                studies += 1
                print(f"{label}: search {studies}/6 completed", file=sys.stderr, flush=True)
                return result

            print(f"Running tuned XGBoost, {label}", flush=True)
            started = perf_counter()
            (FOLDER / label).mkdir(exist_ok=True)
            with (patch("main.load_config", return_value=config),
                  patch("src.tuning.tune_pipeline", side_effect=progress),
                  patch("main.tune_pipeline", side_effect=progress),
                  (FOLDER / label / "console.txt").open("w", encoding="utf-8") as log,
                  redirect_stdout(log)):
                row = main.main()
            saved[label] = row
            summary_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            print(f"Saved {label}: {row['mean']:.4f} +/- {row['std']:.4f}; "
                  f"{row['n_features_kept']}/{row['n_features_before']} columns; "
                  f"{perf_counter() - started:.1f}s", flush=True)
    finally:
        optuna.logging.set_verbosity(verbosity)

    reports = {key: Path(row["report"]).with_suffix("") for key, row in saved.items()}
    frames = {key: pd.read_csv(str(stem) + "_nested_folds.csv") for key, stem in reports.items()}
    oofs = {key: pd.read_csv(str(stem) + "_oof.csv") for key, stem in reports.items()}
    for column in ["row_index", "y_true"]:
        pd.testing.assert_series_equal(oofs["selection_on"][column], oofs["selection_off"][column])
    target = oofs["selection_off"]["y_true"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(target, target))
    ratio = sum(len(val) / len(train) for train, val in folds) / len(folds)
    corrected = corrected_paired_test(frames["selection_on"]["validation"],
                                      frames["selection_off"]["validation"], ratio)
    corrected.update(contrast="selection_on_minus_off", validation_train_ratio=ratio,
                     paired_outer_folds=5)
    (FOLDER / "corrected_paired_test.json").write_text(json.dumps(corrected, indent=2), encoding="utf-8")
    paired = pd.DataFrame({"fold": frames["selection_off"]["fold"],
                           "without_selection": frames["selection_off"]["validation"],
                           "with_selection": frames["selection_on"]["validation"]})
    paired["difference"] = paired["with_selection"] - paired["without_selection"]
    paired.to_csv(FOLDER / "paired_fold_scores.csv", index=False)
    print(json.dumps(corrected, indent=2))


if __name__ == "__main__":
    run()
