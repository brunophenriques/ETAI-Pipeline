"""Week 8 B1: cold-cache nested comparisons of vote, kbest and from_model.

Run with python -m scripts.run_selectors. Each method receives the same XGBoost
space, data, 5x5 folds and 15 trials. Timings are whole-run wall times, including
the baseline, tuning, refits and file output, measured once per method.
"""
import copy
import itertools
import json
import sys
from contextlib import redirect_stdout
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import StratifiedKFold

import main
from src.evaluate import corrected_paired_test
from src.tuning import tune_pipeline


FOLDER = Path("results/week8/b1")
METHODS = ["vote", "kbest", "from_model"]


def run():
    base = main.load_config()
    base["model"] = {"type": "xgboost", "params": {
        "random_state": 42, "n_jobs": 1, "tree_method": "hist", "eval_metric": "logloss"}}
    base["cv"].update(n_splits=5, shuffle=True, random_state=42,
                      selection_metric="roc_auc", n_jobs=-1)
    base["tuning"].update(enabled=True, n_trials=15, n_splits=5, random_state=42)
    base["feature_selection"]["enabled"] = True
    base["feature_selection"]["random_state"] = 42
    # Each method is overridden below; keep the comparison plan independent
    # of the user's chosen default selector.
    base["feature_selection"]["method"] = "vote"
    FOLDER.mkdir(parents=True, exist_ok=True)
    plan = {"dataset_sha256": sha256(Path(base["data"]["path"]).read_bytes()).hexdigest(),
            "preprocessing": base["preprocessing"], "split": base["split"], "cv": base["cv"],
            "model": base["model"], "selection": base["feature_selection"],
            "xgboost_space": base["tuning"]["search_spaces"]["xgboost"],
            "n_trials": 15, "inner_folds": 5, "sampler_seed": 42,
            "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                         "optuna": optuna.__version__}}
    manifest = FOLDER / "experiment.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != plan:
        raise ValueError("B1 settings changed; archive its experiment folder before resuming.")
    manifest.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    summary_path = FOLDER / "runs.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        for method in METHODS:
            if method in saved and Path(saved[method]["report"]).exists():
                print(f"Already saved: {method}", flush=True)
                continue
            config = copy.deepcopy(base)
            config["feature_selection"]["method"] = method
            config["output"] = {"results_dir": str(FOLDER / method),
                                "scoreboard": "results/runs.csv", "layout": "flat"}
            (FOLDER / method).mkdir(exist_ok=True)
            studies = 0

            def progress(*args, **kwargs):
                nonlocal studies
                result = tune_pipeline(*args, **kwargs)
                studies += 1
                print(f"{method}: search {studies}/6 complete", file=sys.stderr, flush=True)
                return result

            print(f"Running {method}, 15 trials per study", flush=True)
            started = perf_counter()
            with (patch("main.load_config", return_value=config),
                  patch("src.tuning.tune_pipeline", side_effect=progress),
                  patch("main.tune_pipeline", side_effect=progress),
                  (FOLDER / method / "console.txt").open("w", encoding="utf-8") as log,
                  redirect_stdout(log)):
                row = main.main()
            row["elapsed_seconds"] = perf_counter() - started
            board = pd.read_csv("results/runs.csv", dtype={"run": str})
            board.loc[board["run"] == row["run"], "elapsed_seconds"] = row["elapsed_seconds"]
            board.to_csv("results/runs.csv", index=False)
            saved[method] = row
            summary_path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            print(f"Saved {method}: {row['mean']:.4f} +/- {row['std']:.4f}, "
                  f"{row['n_features_kept']} columns, {row['elapsed_seconds']:.1f}s", flush=True)
    finally:
        optuna.logging.set_verbosity(verbosity)

    frames, oofs = {}, {}
    for method, row in saved.items():
        stem = Path(row["report"]).with_suffix("")
        frames[method] = pd.read_csv(str(stem) + "_nested_folds.csv")
        oofs[method] = pd.read_csv(str(stem) + "_oof.csv")
        pd.testing.assert_frame_equal(oofs[method][["row_index", "y_true"]],
                                      oofs[METHODS[0]][["row_index", "y_true"]])
    target = oofs[METHODS[0]]["y_true"]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(target, target))
    ratio = sum(len(val) / len(train) for train, val in folds) / len(folds)
    comparisons = []
    for first, second in itertools.combinations(METHODS, 2):
        result = corrected_paired_test(frames[first].validation, frames[second].validation, ratio)
        result.update(first=first, second=second, contrast=f"{first}_minus_{second}")
        comparisons.append(result)
    # Three comparisons: report Holm-adjusted p-values as well as raw corrected ones.
    ordering = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["two_sided_p_value"]
                      if comparisons[i]["two_sided_p_value"] is not None else 1.)
    previous = 0.
    for rank, index in enumerate(ordering):
        value = comparisons[index]["two_sided_p_value"]
        adjusted = None if value is None else min(1., max(previous, (len(comparisons) - rank) * value))
        comparisons[index]["holm_p_value"] = adjusted
        previous = adjusted if adjusted is not None else previous
    (FOLDER / "corrected_pairwise_tests.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    pd.DataFrame([{key: row[key] for key in ["selection_method", "mean", "std", "gap",
                                           "n_features_kept", "elapsed_seconds", "report"]}
                  for row in saved.values()]).to_csv(FOLDER / "summary.csv", index=False)
    pd.DataFrame({"fold": range(1, 6), **{method: frames[method].validation for method in METHODS}}
                 ).to_csv(FOLDER / "paired_fold_scores.csv", index=False)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    run()
