"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# DIAGNOSTICS QUICK REFERENCE
# ============================================================
#
# Source file:
#   data/compas_two_year_recidivism.csv
#
# ID column:
#   id
#
# Placeholder / missing-value tokens:
#   "", "-", "?", "N/A", "NA", "n/a", "na"
#
# Valid numerical ranges:
#   age:            18 <= age <= 100
#   decile_score:    1 <= decile_score <= 10
#   juv_fel_count:   juv_fel_count >= 0
#   priors_count:    0 <= priors_count <= 60
#
# Numerical columns:
#   age
#   decile_score
#   juv_fel_count
#   priors_count
#
# Canonical categories:
#
#   sex:
#     male   -> Male
#     female -> Female
#
#   c_charge_degree:
#     f           -> F
#     felony      -> F
#     m           -> M
#     misdemeanor -> M
#
#   score_text:
#     low    -> Low
#     medium -> Medium
#     high   -> High
#
#   race:
#     african-american -> African-American
#     african american -> African-American
#     asian            -> Asian
#     caucasian        -> Caucasian
#     hispanic         -> Hispanic
#     native american  -> Native American
#     other            -> Other
#
# Columns to drop during preprocessing:
#   prior_offenses
#   age_in_months
#   juvenile_total
#   id
#
# Columns excluded from model features later:
#   id
#   decile_score
#   score_text
#
# Missingness findings:
#   age             -> MCAR
#   juv_fel_count   -> MCAR
#   priors_count    -> MNAR
#   c_charge_degree -> MNAR
#   race            -> MCAR
#   sex             -> MCAR
#
# Duplicate findings:
#   exact row duplicates: 72
#   repeated IDs:         72
#
# ============================================================

def missing_value_diagnostics(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Replace placeholder values with NaN and ensure numerical columns
    contain numeric values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        diagnostics_config (dict): Dictionary containing diagnostic information.

    Returns:
        pd.DataFrame: DataFrame with placeholder/invalid values converted to NaN.
    """
    out = df.copy()

    placeholder_tokens = diagnostics_config.get("placeholder_tokens", [])
    numerical_columns = diagnostics_config.get("numerical_columns", [])

    # Replace known placeholder tokens with NaN
    out.replace(placeholder_tokens, np.nan, inplace=True)

    # Ensure expected numerical columns are numeric.
    # Invalid values are converted to NaN.
    for col in numerical_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Count rows containing at least one missing value
    num_missing_rows = out.isna().any(axis=1).sum()
    diagnostics_config["num_missing_rows"] = int(num_missing_rows)

    return out


def valid_numerical_ranges(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Apply validity rules to numerical columns based on the diagnostics configuration.

    Args:
        df (pd.DataFrame): The input DataFrame.
        diagnostics_config (dict): Dictionary containing diagnostic information.

    Returns:
        pd.DataFrame: DataFrame with invalid numerical values replaced with NaN.
    """
    out = df.copy()
    validity_rules = diagnostics_config.get("validity_rules", {})


    for col, rules in validity_rules.items():
        if col not in out.columns:
            continue

        min_value = rules.get("min")
        max_value = rules.get("max")

        if min_value is not None:
            out.loc[out[col] < min_value, col] = np.nan

        if max_value is not None:
            out.loc[out[col] > max_value, col] = np.nan

    return out

def canonicalize_categories(
    df: pd.DataFrame,
    diagnostics_config: dict
) -> pd.DataFrame:
    """
    Normalize categorical values and convert them to their canonical form.

    Example:
        "  MALE " -> "male" -> "Male"
        " Felony " -> "felony" -> "F"
    """
    out = df.copy()
    canonical_categories = diagnostics_config.get(
        "canonical_categories", {}
    )

    for col, mapping in canonical_categories.items():
        if col not in out.columns:
            continue

        out[col] = (
            out[col]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .replace(mapping)
        )

    return out

def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # naive: just drop rows with any missing values
    df = df.dropna()

    y = df[target]

    # kept aside for fairness auditing after training -- never used as a model input
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in drop_columns if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)

    # naive: one-hot encode all non-numeric columns, no further thought
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test
