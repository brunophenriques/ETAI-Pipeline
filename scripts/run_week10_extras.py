"""Reproduce B1/B2 model comparisons and B2/B3 hidden-value experiments."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import main
from predict import predict
from src.evaluate import corrected_paired_test
from src.imputation import ForestProximityImputer
from src.inference import library_versions
from src.preprocess import preprocess, build_feature_preprocessor


@threadpool_limits.wrap(limits=1)
def run():
    folder = Path("results/week10/extras")
    folder.mkdir(parents=True, exist_ok=True)
    base = main.load_config()
    base["tuning"]["enabled"] = False
    base["cv"]["n_jobs"] = 1
    settings = base["preprocessing"]["imputation"]
    settings.update(numeric_strategy="iterative", sample_posterior=False, include_categories=False)
    plan = {"base": base, "versions": library_versions(),
            "dataset_sha256": sha256(Path(base["data"]["path"]).read_bytes()).hexdigest(),
            "hidden_seed": 42, "hidden_count_per_column": 200,
            "hidden_numeric_columns": ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]}
    (folder / "experiment.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    rows, frames, oofs = [], {}, {}
    recipes = {
        "single": {},
        "multiple_5": {"posterior": True, "multiple": True},
        "with_categories": {"categories": True},
    }
    for label, options in recipes.items():
        config = deepcopy(base)
        config["preprocessing"]["imputation"].update(
            sample_posterior=options.get("posterior", False),
            include_categories=options.get("categories", False))
        config["multiple_imputation"]["enabled"] = options.get("multiple", False)
        config["output"].update(results_dir=str(folder / label), layout="flat",
                                model_path=f"models/week10/extras/{label}.joblib")
        print(f"Running {label}", flush=True)
        started = perf_counter()
        with (folder / f"{label}_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
            row = main.main(config=config)
        row.update(recipe=label, seconds=perf_counter() - started)
        rows.append(row)
        stem = Path(row["report"]).with_suffix("")
        frames[label] = pd.read_csv(str(stem) + "_default_folds.csv")
        oofs[label] = pd.read_csv(str(stem) + "_oof.csv")
        # Also verify the persisted ensemble works with the prediction entry point.
        import joblib
        fitted = joblib.load(config["output"]["model_path"])
        with (folder / f"{label}_prediction_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
            predictions = predict(pd.read_csv("data/compas_new_people.csv"), fitted, config)
        assert len(predictions) == 20
        predictions.to_csv(folder / f"{label}_predictions.csv", index=False)
        print(f"{label}: {row['mean']:.6f}, {row['seconds']:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    target = oofs["single"].y_true
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(target, target))
    ratio = np.mean([len(v) / len(t) for t, v in folds])
    comparisons = {}
    for label in ("multiple_5", "with_categories"):
        pd.testing.assert_frame_equal(oofs[label][["row_index", "y_true"]],
                                      oofs["single"][["row_index", "y_true"]])
        comparisons[label + "_minus_single"] = corrected_paired_test(
            frames[label].validation, frames["single"].validation, ratio)
    (folder / "corrected_tests.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    pd.DataFrame({label: frame.validation for label, frame in frames.items()}).to_csv(
        folder / "paired_fold_scores.csv", index=False)

    # Artificial masking uses raw columns only: engineered age/priors formulas
    # would reveal the hidden truth and invalidate the imputation comparison.
    with (folder / "hidden_setup_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
        X, _, y, _, _, _, _ = preprocess(pd.read_csv(base["data"]["path"]),
            base["data"]["target"], base["data"]["sensitive_attr"],
            base["data"]["drop_columns"], base["diagnostics"], base["preprocessing"], **base["split"])
    numeric = plan["hidden_numeric_columns"]
    categorical = ["age_cat", "sex", "c_charge_degree"]
    train_idx, val_idx = folds[0]
    train, validation = X.iloc[train_idx][numeric + categorical], X.iloc[val_idx][numeric + categorical]
    masked = validation.copy()
    rng = np.random.default_rng(42)
    hidden = {column: rng.choice(np.flatnonzero(validation[column].notna()), 200, replace=False)
              for column in ("age", "priors_count")}
    for column, indices in hidden.items():
        masked.iloc[indices, masked.columns.get_loc(column)] = np.nan
    metrics, guesses = [], []
    for strategy in ("median", "knn", "iterative", "random_forest"):
        for categories in (False, True):
            if strategy == "median" and categories:
                continue
            columns = numeric + categorical if categories else numeric
            prep_config = deepcopy(base["preprocessing"])
            prep_config["imputation"].update(numeric_strategy=strategy, include_categories=categories)
            prep = build_feature_preprocessor(train[columns], prep_config)
            started = perf_counter()
            prep.fit(train[columns])
            filled = prep.transform(masked[columns])
            # Numerical outputs are first and remain in scaled units.
            scaler = (prep.named_steps["encode"].named_transformers_["numerical"] if categories
                      else prep.named_transformers_["numerical"].named_steps["scaler"])
            numeric_filled = scaler.inverse_transform(filled[:, :len(numeric)])
            elapsed = perf_counter() - started
            for column, indices in hidden.items():
                truth = validation[column].iloc[indices].to_numpy()
                values = numeric_filled[indices, numeric.index(column)]
                label = strategy + ("_categories" if categories else "_numeric")
                metrics.append(dict(method=label, column=column,
                    mae=mean_absolute_error(truth, values), seconds=elapsed, n_hidden=len(indices)))
                guesses.extend(dict(method=label, column=column, row_index=int(validation.index[i]),
                                    truth=float(t), guess=float(v)) for i, t, v in zip(indices, truth, values))
            print(f"Hidden values: {strategy}, categories={categories}", flush=True)
    proximity = ForestProximityImputer(n_estimators=50, random_state=42)
    started = perf_counter()
    proximity.fit(train[numeric])
    filled = proximity.transform(masked[numeric])
    indices = hidden["priors_count"]
    truth, values = validation.priors_count.iloc[indices].to_numpy(), filled.priors_count.iloc[indices].to_numpy()
    metrics.append(dict(method="forest_proximity", column="priors_count",
        mae=mean_absolute_error(truth, values), seconds=perf_counter()-started, n_hidden=len(indices)))
    guesses.extend(dict(method="forest_proximity", column="priors_count", row_index=int(validation.index[i]),
                        truth=float(t), guess=float(v)) for i, t, v in zip(indices, truth, values))
    import joblib
    joblib.dump(proximity, "models/week10/extras/proximity.joblib")
    # A target-free unseen row with missing priors can be transformed after reload.
    reloaded = joblib.load("models/week10/extras/proximity.joblib")
    pd.testing.assert_frame_equal(reloaded.transform(masked[numeric]), filled)
    pd.DataFrame(metrics).to_csv(folder / "hidden_value_metrics.csv", index=False)
    pd.DataFrame(guesses).to_csv(folder / "hidden_value_predictions.csv", index=False)


if __name__ == "__main__":
    run()
