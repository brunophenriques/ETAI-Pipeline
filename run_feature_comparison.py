"""Compare the three fixed engineered features with the Week 6 nested baselines.

Run with python run_feature_comparison.py. Existing matching 50-trial reports
are reused for the off condition. Config.yaml and original reports are preserved.
"""

import ast
import copy
import json
import re
import shutil
import sys
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import optuna
import pandas as pd

import main
import run_week6_comparison as comparison
from src.tuning import tune_pipeline


OUTPUT = Path("results/feature_engineering")
MODELS = ["logistic_regression", "adaboost", "random_forest"]


def metrics(report):
    score = re.search(r"Nested CV accuracy \(mean \+/- std\): ([\d.]+) \+/- ([\d.]+)", report)
    gap = re.search(r"Mean train-validation gap: ([+\-\d.]+)", report)
    return float(score[1]), float(score[2]), float(gap[1])


def fold_values(report):
    # Saved legacy baseline tables display scores to three decimal places.
    table = report.split("Classification report", 1)[0]
    values = re.findall(r"^[ \t]*([1-5])[ \t]+([\d.]+)[ \t]+([\d.]+)[ \t]+[+\-\d.]+[ \t]+[\d.]+[ \t]+", table, re.MULTILINE)
    if len(values) != 5:
        raise ValueError("Expected five nested fold rows in the saved report.")
    return [float(validation) for _, _, validation in values]


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = main.load_config()
    provenance = json.loads(Path("results/week6_comparison/provenance.json").read_text(encoding="utf-8"))
    if sha256(Path(config["data"]["path"]).read_bytes()).hexdigest() != provenance["dataset_sha256"]:
        raise ValueError("The dataset changed; regenerate the no-feature baselines first.")
    config["preprocessing"]["engineered_features"] = {"enabled": True}
    new_folder = OUTPUT / "with_features"
    old_folder = OUTPUT / "without_features"
    new_folder.mkdir(exist_ok=True)
    old_folder.mkdir(exist_ok=True)
    previous_output = comparison.OUTPUT
    verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    summary, paired_rows = [], []
    try:
        comparison.OUTPUT = new_folder
        for name in MODELS:
            filename = f"{name}_tuned_50trials_seed42.txt"
            source = Path("results/week6_comparison") / filename
            baseline = source.read_text(encoding="utf-8")
            settings = ast.literal_eval(re.search(r"Tuning settings: (.+)", baseline)[1])
            assert settings["n_trials"] == 50 and settings["n_splits"] == 5
            assert settings["random_state"] == 42
            assert settings["search_spaces"][name] == config["tuning"]["search_spaces"][name]
            sidecar = source.with_suffix(".json")
            if sidecar.exists():
                previous_config = json.loads(sidecar.read_text(encoding="utf-8"))
                previous_prep = previous_config["preprocessing"]
                assert previous_config["cv"] == config["cv"]
                assert previous_config["split"] == config["split"]
            else:
                previous_prep = ast.literal_eval(re.search(r"Preprocessing settings: (.+)", baseline)[1])
            assert not previous_prep.get("engineered_features", {}).get("enabled", False)
            current_prep = copy.deepcopy(config["preprocessing"])
            current_prep.pop("engineered_features", None)
            previous_prep.pop("engineered_features", None)
            assert previous_prep == current_prep
            if not (old_folder / filename).exists():
                shutil.copy2(source, old_folder / filename)
            if sidecar.exists() and not (old_folder / sidecar.name).exists():
                shutil.copy2(sidecar, old_folder / sidecar.name)

            searches = 0

            def logged_tune(*args, **kwargs):
                nonlocal searches
                result = tune_pipeline(*args, **kwargs)
                searches += 1
                print(f"{name}: search {searches}/6 finished", file=sys.stderr, flush=True)
                return result

            with (patch("src.tuning.tune_pipeline", side_effect=logged_tune),
                  patch("main.tune_pipeline", side_effect=logged_tune)):
                comparison.run_one(config, name, comparison.TUNED_BASE_PARAMS[name], tuned=True)

            engineered = (new_folder / filename).read_text(encoding="utf-8")
            old_mean, old_std, old_gap = metrics(baseline)
            new_mean, new_std, new_gap = metrics(engineered)
            summary.append({
                "model": name,
                "without_mean": old_mean, "without_std": old_std, "without_gap": old_gap,
                "with_mean": new_mean, "with_std": new_std, "with_gap": new_gap,
                "difference": round(new_mean - old_mean, 3),
                "difference_exceeds_both_fold_stds": new_mean - old_mean > max(old_std, new_std),
                "without_report": str(old_folder / filename),
                "with_report": str(new_folder / filename),
                "baseline_source": str(source),
            })
            for fold, (before, after) in enumerate(zip(fold_values(baseline), fold_values(engineered)), 1):
                paired_rows.append({"model": name, "fold": fold, "without": before,
                                    "with": after, "difference": round(after - before, 3)})
            (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            pd.DataFrame(summary).to_csv(OUTPUT / "summary.csv", index=False)
            pd.DataFrame(paired_rows).to_csv(OUTPUT / "fold_comparison.csv", index=False)
    finally:
        comparison.OUTPUT = previous_output
        optuna.logging.set_verbosity(verbosity)
    print("Comparison saved. Summary and paired values use the reports' three-decimal precision.")


if __name__ == "__main__":
    run()
