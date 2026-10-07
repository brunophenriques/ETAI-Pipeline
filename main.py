"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Pipeline:
    load config -> load raw data -> clean and preprocess
    -> reserve test set -> cross-validate development set -> refit -> save results
"""

import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocess import preprocess
from src.model import build_pipeline
from src.evaluate import (
    cross_validate_pipeline,
    fairness_report,
    format_nested_cv_report,
    oof_classification_report,
)
from src.results import save_run
from src.tuning import nested_cross_validate, tune_pipeline


def load_config(path: str = "config.yaml") -> dict:
    """Load the pipeline configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    """Run the complete predictive pipeline."""
    config = load_config()

    # Load the raw dataset
    df_raw = load_data(config["data"]["path"])

    # preprocess() calls clean_dataset() internally
    (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
        feature_preprocessor,
    ) = preprocess(
        df=df_raw,
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        diagnostics_config=config["diagnostics"],
        preprocessing_config=config["preprocessing"],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # Build one estimator so each CV fold fits its own preprocessing.
    model = build_pipeline(feature_preprocessor, config["model"])
    # Compare recipes on development data; leave the final test set unused.
    cv_config = config["cv"]
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)
    context = "development set, out-of-fold predictions"
    if tuning_enabled:
        model_type = config["model"]["type"]
        search_space = tuning_config.get("search_spaces", {}).get(model_type)
        if not search_space:
            raise ValueError(f"No tuning search space configured for {model_type}.")
        shuffle = cv_config.get("shuffle", True)
        outer_cv = StratifiedKFold(
            n_splits=cv_config["n_splits"], shuffle=shuffle,
            random_state=cv_config.get("random_state") if shuffle else None,
        )
        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"], shuffle=True,
            random_state=tuning_config["random_state"],
        )
        tuning_args = {
            "scoring": scoring,
            "search_space": search_space,
            "n_trials": tuning_config["n_trials"],
            "random_state": tuning_config["random_state"],
            "n_jobs": n_jobs,
        }
        fold_scores, y_oof = nested_cross_validate(
            model, X_dev, y_dev, outer_cv, inner_cv, **tuning_args,
        )
        report = format_nested_cv_report(fold_scores, scoring)
        report += f"\nTuning settings: {tuning_config}"
        context = "development set, nested out-of-fold predictions"
    else:
        report, y_oof = cross_validate_pipeline(model, X_dev, y_dev, cv_config)

    report += "\n\n" + oof_classification_report(y_dev, y_oof, context=context)
    report += "\n\n" + fairness_report(
        y_dev,
        y_oof,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
        context=context,
    )

    # Select final parameters separately, after evaluating the tuning procedure.
    if tuning_enabled:
        model, study = tune_pipeline(model, X_dev, y_dev, inner_cv, **tuning_args)
        nested_mean = float(fold_scores["validation"].mean())
        tuning_summary = (
            f"Final search on all development rows (metric: {scoring}).\n"
            f"Selected parameters: {study.best_params}\n"
            f"Final classifier parameters: {model.named_steps['classifier'].get_params()}\n"
            f"Best Optuna trial score: {study.best_value:.3f}\n"
            f"Nested CV mean score: {nested_mean:.3f}\n"
            f"Optimism check (best trial - nested mean): {study.best_value - nested_mean:+.3f}\n"
            "This difference is descriptive; outer training sets are smaller, "
            "so it is not a pure estimate of selection bias."
        )
        print(tuning_summary)
        report += "\n\n" + tuning_summary

    # Refit on all development rows; never evaluate the reserved test set here.
    model.fit(X_dev, y_dev)
    summary = (
        f"Final pipeline refit on {len(X_dev)} development rows.\n"
        f"Reserved test set: {len(X_test)} rows (not evaluated)."
    )
    print(summary)
    report += "\n\n" + summary

    # Save the configuration and evaluation report
    results_dir = config.get("output", {}).get(
        "results_dir",
        "results",
    )

    path = save_run(
        results_dir,
        config,
        report,
    )

    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
