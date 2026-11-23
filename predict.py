"""Predict new CSV rows using a saved pipeline and its training configuration."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.evaluate import positive_proba
from src.inference import prepare_new_rows


def predict(raw, pipeline, config):
    features = prepare_new_rows(raw, pipeline.feature_names_in_, config)
    probability = positive_proba(pipeline, features)
    identifier = config["diagnostics"]["id_column"]
    if identifier not in raw:
        raise ValueError(f"Prediction data is missing identifier column: {identifier}")
    return pd.DataFrame({identifier: raw[identifier].to_numpy(),
                         "probability": probability,
                         "prediction": (probability >= .5).astype(int)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default="predictions.csv")
    parser.add_argument("--model", default="models/model.joblib")
    args = parser.parse_args()
    model_path = Path(args.model)
    try:
        config = json.loads(model_path.with_suffix(".config.json").read_text(encoding="utf-8"))
        versions_path = model_path.with_suffix(".versions.json")
        if versions_path.exists():
            from src.inference import library_versions
            saved = json.loads(versions_path.read_text(encoding="utf-8"))
            current = library_versions()
            differences = [f"{key}: trained {value}, installed {current.get(key)}"
                           for key, value in saved.items() if current.get(key) != value]
            if differences:
                raise ValueError("Model library versions differ: " + "; ".join(differences))
        pipeline = joblib.load(model_path)
        predictions = predict(pd.read_csv(args.input), pipeline, config)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"Prediction failed: {error}\n")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    print(f"{len(predictions)} rows predicted -> {destination}")


if __name__ == "__main__":
    main()
