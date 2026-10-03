"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Pipeline:
    load config -> load raw data -> clean and preprocess
    -> reserve test set -> cross-validate development set -> refit -> save results
"""

import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocess import preprocess
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline,
    fairness_report,
    oof_classification_report,
)
from src.results import save_run


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
    model = Pipeline([
        ("preprocessing", feature_preprocessor),
        ("classifier", build_model(config["model"])),
    ])
    # Compare recipes on development data; leave the final test set unused.
    report, y_oof = cross_validate_pipeline(
        model,
        X_dev,
        y_dev,
        config["cv"],
    )
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n\n" + fairness_report(
        y_dev,
        y_oof,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
        context="development set, out-of-fold predictions",
    )

    # CV evaluates fresh copies; refit this pipeline on all development rows.
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
