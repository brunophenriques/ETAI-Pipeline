"""Run four untuned imputers on identical development folds and compare them."""
from contextlib import redirect_stdout
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import joblib
import numpy as np
import platform
import sklearn
import xgboost
from sklearn.model_selection import StratifiedKFold

import main
from src.evaluate import corrected_paired_test


def run():
    folder = Path("results/week10")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "environment.json").write_text(json.dumps({
        "python": platform.python_version(), "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "joblib": joblib.__version__}, indent=2), encoding="utf-8")
    base = main.load_config()
    base["tuning"]["enabled"] = False
    base["cv"]["n_jobs"] = 1
    rows, frames, oofs = [], {}, {}
    for strategy in ("median", "knn", "iterative", "random_forest"):
        config = deepcopy(base)
        config["preprocessing"]["imputation"]["numeric_strategy"] = strategy
        config["output"].update(results_dir=str(folder / strategy), layout="flat",
                                model_path=f"models/week10/{strategy}.joblib")
        started = perf_counter()
        print(f"Running {strategy}", flush=True)
        with (folder / f"{strategy}_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
            row = main.main(config=config)
        row["seconds"] = perf_counter() - started
        rows.append(row)
        stem = Path(row["report"]).with_suffix("")
        frames[strategy] = pd.read_csv(str(stem) + "_default_folds.csv")
        oofs[strategy] = pd.read_csv(str(stem) + "_oof.csv")
        print(f"{strategy}: {row['mean']:.6f}; {row['seconds']:.1f}s", flush=True)
    target = oofs["median"].y_true
    cv = base["cv"]
    folds = list(StratifiedKFold(cv["n_splits"], shuffle=cv["shuffle"],
                               random_state=cv["random_state"]).split(target, target))
    ratio = sum(len(v) / len(t) for t, v in folds) / len(folds)
    for frame in oofs.values():
        pd.testing.assert_frame_equal(frame[["row_index", "y_true"]],
                                      oofs["median"][["row_index", "y_true"]])
    tests = {}
    for a, b in combinations(frames, 2):
        tests[f"{a}_minus_{b}"] = corrected_paired_test(
            frames[a].validation, frames[b].validation, ratio)
    # Holm adjustment across all six comparisons.
    ordered = sorted(tests, key=lambda key: tests[key]["two_sided_p_value"] or 0)
    adjusted = 0.
    for i, key in enumerate(ordered):
        p = tests[key]["two_sided_p_value"]
        if p is not None:
            adjusted = max(adjusted, min(1., p * (len(ordered) - i)))
            tests[key]["holm_p_value"] = adjusted
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    pd.DataFrame({key: frame.validation for key, frame in frames.items()}).to_csv(
        folder / "paired_fold_scores.csv", index=False)
    (folder / "corrected_tests.json").write_text(json.dumps({
        "validation_train_ratio": ratio, "comparisons": tests}, indent=2), encoding="utf-8")
    # Median is the conservative default; keep its model at the documented path.
    import shutil
    for suffix in (".joblib", ".config.json", ".versions.json"):
        shutil.copyfile(f"models/week10/median{suffix}", f"models/model{suffix}")


if __name__ == "__main__":
    run()
