"""Compare none/Isolation Forest/LOF on paired development folds."""
from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import main
from predict import predict
from src.evaluate import corrected_paired_test
from src.inference import library_versions
from src.model import build_selector
from src.preprocess import AnomalyScore, preprocess


@threadpool_limits.wrap(limits=1)
def run():
    folder = Path("results/week11")
    folder.mkdir(parents=True, exist_ok=True)
    base = main.load_config()
    base["tuning"]["enabled"] = False
    base["cv"]["n_jobs"] = 1
    base["multiple_imputation"]["enabled"] = False
    (folder / "experiment.json").write_text(json.dumps({
        "base": base, "versions": library_versions(),
        "dataset_sha256": sha256(Path(base["data"]["path"]).read_bytes()).hexdigest(),
        "unusual_group": "highest ceil(5% of validation rows) within each fold; novelty scores",
        "guess_before_runs": "Small gain at most; unusual does not necessarily mean reoffends; XGBoost already sees age/count patterns.",
    }, indent=2), encoding="utf-8")
    rows, scores, oofs = [], {}, {}
    for method in ("none", "isolation_forest", "lof"):
        config = deepcopy(base)
        config["anomaly"]["method"] = method
        config["output"].update(results_dir=str(folder / method), layout="flat",
            model_path=f"models/week11/{method}.joblib")
        print(f"Running {method}", flush=True)
        started = perf_counter()
        with (folder / f"{method}_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
            row = main.main(config=config)
        row["seconds"] = perf_counter() - started
        rows.append(row)
        stem = Path(row["report"]).with_suffix("")
        scores[method] = pd.read_csv(str(stem) + "_default_folds.csv")
        oofs[method] = pd.read_csv(str(stem) + "_oof.csv")
        pipeline = joblib.load(config["output"]["model_path"])
        raw = pd.read_csv("data/compas_new_people.csv")
        with (folder / f"{method}_prediction_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
            output = predict(raw, pipeline, config)
        assert len(output) == len(raw) == 20
        assert output.id.tolist() == raw.id.tolist()
        assert output.probability.between(0, 1).all()
        assert output.loc[raw.drop(columns="id").duplicated(keep=False), "probability"].nunique() == 1
        output.to_csv(folder / f"{method}_predictions.csv", index=False)
        print(f"{method}: ROC-AUC {row['mean']:.6f}, {row['seconds']:.1f}s; 20 predictions", flush=True)
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    for frame in oofs.values():
        pd.testing.assert_frame_equal(frame[["row_index", "y_true"]], oofs["none"][["row_index", "y_true"]])
    with (folder / "diagnostic_console.txt").open("w", encoding="utf-8") as log, redirect_stdout(log):
        X, _, y, _, _, _, prep = preprocess(pd.read_csv(base["data"]["path"]),
            base["data"]["target"], base["data"]["sensitive_attr"], base["data"]["drop_columns"],
            base["diagnostics"], base["preprocessing"], **base["split"])
    cv = base["cv"]
    folds = list(StratifiedKFold(cv["n_splits"], shuffle=cv["shuffle"],
                               random_state=cv["random_state"]).split(X, y))
    ratio = float(np.mean([len(val) / len(train) for train, val in folds]))
    comparisons = {f"{a}_minus_{b}": corrected_paired_test(scores[a].validation, scores[b].validation, ratio)
                   for a, b in combinations(scores, 2)}
    ordered = sorted(comparisons, key=lambda key: comparisons[key]["two_sided_p_value"]
                     if comparisons[key]["two_sided_p_value"] is not None else 1.)
    adjusted = 0.
    for i, key in enumerate(ordered):
        p = comparisons[key]["two_sided_p_value"]
        if p is not None:
            adjusted = max(adjusted, min(1., p * (len(ordered) - i)))
            comparisons[key]["holm_p_value"] = adjusted
    (folder / "corrected_tests.json").write_text(json.dumps({
        "validation_train_ratio": ratio, "comparisons": comparisons}, indent=2), encoding="utf-8")
    pd.DataFrame({key: frame.validation for key, frame in scores.items()}).to_csv(folder / "paired_fold_scores.csv", index=False)

    # Score held-out development people. Neither detector sees their labels.
    diagnostics, duplicate_groups, selections = [], [], []
    for fold, (train, val) in enumerate(folds, 1):
        fitted = clone(prep)
        Z_train = fitted.fit_transform(X.iloc[train], y.iloc[train])
        Z_val = fitted.transform(X.iloc[val])
        largest = int(np.unique(Z_train, axis=0, return_counts=True)[1].max())
        duplicate_groups.append({"fold": fold, "largest_identical_training_group": largest})
        for method in ("isolation_forest", "lof"):
            detector = AnomalyScore(method=method, n_neighbors=base["anomaly"]["n_neighbors"],
                                    random_state=base["anomaly"]["random_state"])
            training_features = detector.fit_transform(Z_train)
            if base.get("feature_selection", {}).get("enabled", False):
                selector = build_selector(base["feature_selection"])
                selector.fit(training_features, y.iloc[train])
                names = detector.get_feature_names_out(fitted.get_feature_names_out())
                kept = selector.get_feature_names_out(names).tolist()
                selections.append(dict(fold=fold, method=method,
                    anomaly_kept="anomaly_score" in kept, kept_features=json.dumps(kept)))
            unusual = detector.transform(Z_val)[:, -1]
            # Stable sorting gives an exact top 5% group even when scores tie.
            top = np.zeros(len(val), dtype=bool)
            top[np.argsort(-unusual, kind="stable")[:int(np.ceil(.05 * len(val)))]] = True
            diagnostics.extend(dict(method=method, fold=fold, row_index=int(X.index[index]),
                score=float(score), top_five_percent=bool(flag), y_true=int(y.iloc[index]))
                for index, score, flag in zip(val, unusual, top))
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(folder / "anomaly_oof_scores.csv", index=False)
    diagnostic.groupby(["method", "top_five_percent"]).agg(
        people=("y_true", "size"), reoffended=("y_true", "sum"),
        reoffending_rate=("y_true", "mean")).reset_index().to_csv(folder / "anomaly_group_rates.csv", index=False)
    pd.DataFrame(duplicate_groups).to_csv(folder / "duplicate_groups.csv", index=False)
    pd.DataFrame(selections).to_csv(folder / "fold_selections.csv", index=False)
    # Put the no-anomaly reference at the normal prediction path.
    import shutil
    for suffix in (".joblib", ".config.json", ".versions.json"):
        shutil.copyfile(f"models/week11/none{suffix}", f"models/model{suffix}")


if __name__ == "__main__":
    run()
