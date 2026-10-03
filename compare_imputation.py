"""Run with python compare_imputation.py; never evaluate the reserved test set."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from main import load_config
from src.data import load_data
from src.evaluate import corrected_paired_comparison
from src.model import build_model
from src.preprocess import build_feature_preprocessor, preprocess


def compare_imputation(config: dict) -> dict:
    """Compare only the imputer; materialize identical folds for both recipes."""
    X, X_test, y, _, _, _, _ = preprocess(
        load_data(config["data"]["path"]),
        config["data"]["target"],
        config["data"]["sensitive_attr"],
        config["data"]["drop_columns"],
        config["diagnostics"],
        config["preprocessing"],
        **config["split"],
    )
    cv_cfg = config["cv"]
    shuffle = cv_cfg.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"], shuffle=shuffle,
        random_state=cv_cfg.get("random_state") if shuffle else None,
    )
    folds = list(cv.split(X, y))
    comparison_cfg = config["imputation_comparison"]
    holdouts = {
        seed: train_test_split(
            np.arange(len(X)), test_size=comparison_cfg["validation_size"],
            random_state=seed, stratify=y,
        )
        for seed in comparison_cfg["holdout_seeds"]
    }
    recipes = {}
    for strategy in ["median", "knn"]:
        prep_cfg = copy.deepcopy(config["preprocessing"])
        prep_cfg["numerical_imputation"] = strategy
        pipe = Pipeline([
            ("preprocessing", build_feature_preprocessor(X, prep_cfg)),
            ("classifier", build_model(config["model"])),
        ])
        scores = cross_validate(
            pipe, X, y, cv=folds, scoring="accuracy",
            return_train_score=True, error_score="raise",
        )
        holdout_scores = {}
        for seed, (train_idx, val_idx) in holdouts.items():
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            holdout_scores[str(seed)] = accuracy_score(y.iloc[val_idx], pipe.predict(X.iloc[val_idx]))
        recipes[strategy] = {
            "preprocessing": prep_cfg,
            "cv_train": scores["train_score"].tolist(),
            "cv_validation": scores["test_score"].tolist(),
            "cv_mean": float(scores["test_score"].mean()),
            "cv_std": float(scores["test_score"].std(ddof=1)),
            "cv_gap": float((scores["train_score"] - scores["test_score"]).mean()),
            "holdout_accuracy": holdout_scores,
        }
        print(f"{strategy}: CV {recipes[strategy]['cv_mean']:.3f} +/- {recipes[strategy]['cv_std']:.3f}", flush=True)
    differences = np.array(recipes["knn"]["cv_validation"]) - recipes["median"]["cv_validation"]
    ratio = float(np.mean([len(val) / len(train) for train, val in folds]))
    paired = corrected_paired_comparison(differences, ratio)
    # Retain the simpler recipe unless the comparison supports a KNN gain.
    p_value = paired["two_sided_p_value"]
    selected = "knn" if differences.mean() > 0 and p_value is not None and p_value < 0.05 else "median"
    return {
        "model": copy.deepcopy(config["model"]),
        "split": config["split"], "cv": cv_cfg,
        "sklearn_version": sklearn.__version__,
        "development_rows": len(X), "reserved_test_rows": len(X_test),
        "test_evaluated": False,
        "validation_train_ratio": ratio,
        "fold_sizes": [{"train": len(train), "validation": len(val)} for train, val in folds],
        "holdout_validation_size": comparison_cfg["validation_size"],
        "recipes": recipes,
        "paired_differences_knn_minus_median": differences.tolist(),
        "corrected_paired_comparison": paired,
        "recommended_recipe": selected,
    }


def comparison_markdown(result: dict) -> str:
    seeds = list(result["recipes"]["median"]["holdout_accuracy"])
    lines = [
        "## Numeric imputation comparison",
        "",
        f"Model: `{result['model']['type']}`, parameters `{result['model']['params']}`. "
        f"Both recipes use `{result['recipes']['median']['preprocessing']['scaler']}` scaling "
        "before imputation, identical one-hot encoding and missingness flags.",
        "",
        "| Recipe | " + " | ".join(f"Holdout seed {s}" for s in seeds)
        + " | CV accuracy (mean ± std) | CV train–validation gap |",
        "|---|" + "---|" * (len(seeds) + 2),
    ]
    for name, recipe in result["recipes"].items():
        holds = " | ".join(f"{recipe['holdout_accuracy'][s]:.3f}" for s in seeds)
        lines.append(f"| {name} | {holds} | {recipe['cv_mean']:.3f} ± {recipe['cv_std']:.3f} | {recipe['cv_gap']:+.3f} |")
    paired = result["corrected_paired_comparison"]
    p_value = paired["two_sided_p_value"]
    lines.extend([
        "",
        "Holdouts are stratified 75/25 splits of the development set; they are not the Week 3 "
        "test results. Both recipes reuse the same five stratified CV folds (seed "
        f"{result['cv']['random_state']}); all statistics use training rows only.",
        "",
        f"Paired difference (KNN − median): {paired['mean_difference_knn_minus_median']:+.5f}; "
        f"corrected SE {paired['corrected_standard_error']:.5f}; "
        f"95% CI [{paired['ci95'][0]:+.5f}, {paired['ci95'][1]:+.5f}]; "
        + (f"two-sided p = {p_value:.4f}." if p_value is not None else "p-value undefined (zero variance)."),
        "The approximate Nadeau–Bengio correction uses "
        "`SE = sqrt((1/n_folds + n_validation/n_train) * sample_variance(paired_differences))` "
        "and a t distribution with four degrees of freedom. "
        "Only five paired scores are available, so a non-significant result does not establish equivalence.",
        "",
        "Method reference: [scikit-learn statistical comparison example]"
        "(https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html).",
        "",
        f"Recommended recipe: `{result['recommended_recipe']}`. "
        f"The {result['reserved_test_rows']:,}-row final test set was not evaluated.",
    ])
    return "\n".join(lines) + "\n"


def main():
    result = compare_imputation(load_config())
    output = Path("results/imputation_comparison")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "comparison.md").write_text(comparison_markdown(result), encoding="utf-8")
    rows = []
    for name, recipe in result["recipes"].items():
        for fold, (train, validation) in enumerate(zip(recipe["cv_train"], recipe["cv_validation"]), 1):
            rows.append({"recipe": name, "fold": fold, "train": train, "validation": validation, "gap": train - validation})
    pd.DataFrame(rows).to_csv(output / "fold_scores.csv", index=False)
    # Windows consoles may use cp1252, which cannot print a mathematical minus.
    print(comparison_markdown(result).replace("\u2212", "-"))
    print(f"Evidence saved to {output}. Configuration was not changed by this script.")


if __name__ == "__main__":
    main()
