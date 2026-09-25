"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Pipeline:
    load config -> load raw data -> clean and preprocess
    -> split -> train -> evaluate -> save results
"""

import yaml

from src.data import load_data
from src.preprocess import preprocess
from src.model import build_model
from src.evaluate import evaluate, fairness_report
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
    X_train, X_test, y_train, y_test, extras_test = preprocess(
        df=df_raw,
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        diagnostics_config=config["diagnostics"],
        preprocessing_config=config["preprocessing"],
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # Build and train the model
    model = build_model(config["model"])
    model.fit(X_train, y_train)

    # Generate predictions for both datasets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate predictive performance
    report = evaluate(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
    )

    # Evaluate fairness using the sensitive attribute
    report += "\n" + fairness_report(
        y_test,
        y_test_pred,
        extras_test,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

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