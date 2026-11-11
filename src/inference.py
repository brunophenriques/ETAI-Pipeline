"""Apply the training recipe to new rows without removing any observations."""
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.cleaning import (missing_value_diagnostics, valid_numerical_ranges,
                          canonicalize_categories, drop_redundant_columns)
from src.preprocess import (add_engineered_features, add_missingness_indicators,
                            prepare_categorical_missing_values)


def prepare_new_rows(raw, feature_columns, config):
    """Return training-aligned features while preserving row order and count."""
    diagnostics = deepcopy(config["diagnostics"])
    out = missing_value_diagnostics(raw, diagnostics)
    out = valid_numerical_ranges(out, diagnostics)
    out = canonicalize_categories(out, diagnostics)
    out = drop_redundant_columns(out, diagnostics)
    settings = config["preprocessing"]
    if settings.get("engineered_features", {}).get("enabled", False):
        out = add_engineered_features(out)
    out = add_missingness_indicators(out, settings.get("missing_indicator_columns", []))
    missing = set(feature_columns) - set(out.columns)
    if missing:
        raise ValueError(f"Prediction data is missing features: {sorted(missing)}")
    return prepare_categorical_missing_values(out.loc[:, feature_columns])


def predict_new_rows(model, feature_columns, config):
    """Export IDs, class-1 probabilities and predictions at threshold 0.5."""
    raw = pd.read_csv(config["data"]["prediction_path"])
    identifier = config["diagnostics"]["id_column"]
    X = prepare_new_rows(raw, feature_columns, config)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, positive_index]
    predictions = pd.DataFrame({identifier: raw[identifier].to_numpy(),
                                "probability": probabilities,
                                "prediction": (probabilities >= .5).astype(int)})
    path = Path(config["output"]["predictions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return path
