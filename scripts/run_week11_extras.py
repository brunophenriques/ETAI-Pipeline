"""Compare train-only removal, quantile clipping and nested LOF-k tuning."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import main
from predict import predict
from src.evaluate import corrected_paired_test
from src.inference import library_versions


def run():
    folder = Path("results/week11/extras")
    folder.mkdir(parents=True, exist_ok=True)
    base = main.load_config()
    base["cv"]["n_jobs"] = 1
    base["tuning"]["enabled"] = False
    base["multiple_imputation"]["enabled"] = False
    base["preprocessing"]["quantile_clip"]["enabled"] = False
    base["anomaly"].update(method="none", n_neighbors=60)
    base["anomaly"]["training_filter"]["enabled"] = False
    recipes = ("baseline", "drop_training_5pct", "clip_1_99", "lof_fixed_nested", "lof_tuned_k")
    (folder / "experiment.json").write_text(json.dumps({
        "base": base, "recipes": recipes, "versions": library_versions(),
        "dataset_sha256": sha256(Path(base["data"]["path"]).read_bytes()).hexdigest(),
        "nested": {"outer_folds": 5, "inner_folds": 3, "fixed_k": 60,
                   "k_search": [20, 100], "trials_per_search": 6,
                   "classifier_and_selection": "fixed; classifier max_depth=6 singleton search"},
    }, indent=2), encoding="utf-8")
    rows, frames, oofs = [], {}, {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for label in recipes:
        config = deepcopy(base)
        if label == "drop_training_5pct":
            config["anomaly"]["training_filter"]["enabled"] = True
        if label == "clip_1_99":
            config["preprocessing"]["quantile_clip"]["enabled"] = True
        if label.startswith("lof_"):
            config["anomaly"]["method"] = "lof"
            config["tuning"].update(enabled=True, n_splits=3,
                n_trials=6 if label == "lof_tuned_k" else 1)
            # Fix the model/selector so the k search is the only recipe change.
            config["feature_selection"]["search_spaces"] = {}
            config["tuning"]["search_spaces"][config["model"]["type"]] = {
                "classifier__max_depth": {"type": "categorical", "choices": [6]}}
            if label == "lof_fixed_nested":
                config["anomaly"]["search_space"] = {}
        config["output"].update(results_dir=str(folder / label), layout="flat",
            model_path=f"models/week11/extras/{label}.joblib")
        print(f"Running {label}", flush=True)
        started = perf_counter()
        with (folder / f"{label}_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
            row = main.main(config=config)
        row.update(recipe=label, seconds=perf_counter() - started)
        rows.append(row)
        # Save progress after each completed recipe.
        pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
        stem = Path(row["report"]).with_suffix("")
        suffix = "_nested_folds.csv" if config["tuning"]["enabled"] else "_default_folds.csv"
        frames[label] = pd.read_csv(str(stem) + suffix)
        oofs[label] = pd.read_csv(str(stem) + "_oof.csv")
        model = joblib.load(config["output"]["model_path"])
        raw = pd.read_csv("data/compas_new_people.csv")
        with (folder / f"{label}_prediction_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
            output = predict(raw, model, config)
        assert len(output) == len(raw) == 20
        assert output.id.tolist() == raw.id.tolist()
        assert output.probability.between(0, 1).all()
        output.to_csv(folder / f"{label}_predictions.csv", index=False)
        if hasattr(model, "removed_indices_"):
            pd.DataFrame({"removed_training_position": model.removed_indices_}).to_csv(
                folder / "final_removed_training_positions.csv", index=False)
        print(f"{label}: {row['mean']:.6f} +/- {row['std']:.6f}; {row['seconds']:.1f}s; 20 predictions", flush=True)
    for frame in oofs.values():
        pd.testing.assert_frame_equal(frame[["row_index", "y_true"]], oofs["baseline"][["row_index", "y_true"]])
    target = oofs["baseline"].y_true
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(target, target))
    ratio = float(np.mean([len(v) / len(t) for t, v in folds]))
    pairs = [("drop_training_5pct", "baseline"), ("clip_1_99", "baseline"),
             ("lof_tuned_k", "lof_fixed_nested")]
    tests = {f"{a}_minus_{b}": corrected_paired_test(frames[a].validation, frames[b].validation, ratio)
             for a, b in pairs}
    (folder / "corrected_tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    pd.DataFrame({key: frame.validation for key, frame in frames.items()}).to_csv(
        folder / "paired_fold_scores.csv", index=False)


if __name__ == "__main__":
    run()
