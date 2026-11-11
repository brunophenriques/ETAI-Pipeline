"""Configuration-driven, deterministic row cleaning for COMPAS and Adult.

Convert configured placeholders and invalid numeric values to missing values;
normalise configured categories; remove duplicate training records and excluded
columns. Keep learned imputation inside the fitted model pipeline.

Dataset-specific rules live in config.yaml and config_adult.yaml. Unlabelled
inference calls the individual cleaning steps without removing duplicate rows.
"""
import numpy as np
import pandas as pd


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

    for col in diagnostics_config.get("strip_columns", []):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()

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

def remove_duplicates(
    df: pd.DataFrame,
    diagnostics_config: dict
) -> pd.DataFrame:
    """
    Remove exact duplicate rows and rows with repeated identifiers.

    The first occurrence of each duplicate row or identifier is preserved.

    Args:
        df (pd.DataFrame): The input DataFrame.
        diagnostics_config (dict): Dictionary containing diagnostic information.

    Returns:
        pd.DataFrame: DataFrame without duplicate rows or repeated identifiers.
    """
    out = df.copy()

    # Remove exact duplicate rows
    out = out.drop_duplicates(keep="first")

    # Remove rows with repeated identifiers
    id_column = diagnostics_config.get("id_column")

    if id_column and id_column in out.columns:
        out = out.drop_duplicates(
            subset=id_column,
            keep="first"
        )

    return out

def drop_redundant_columns(
    df: pd.DataFrame,
    diagnostics_config: dict
) -> pd.DataFrame:
    """
    Remove redundant or unnecessary columns specified in the configuration.

    Args:
        df (pd.DataFrame): The input DataFrame.
        diagnostics_config (dict): Dictionary containing diagnostic information.

    Returns:
        pd.DataFrame: DataFrame without the configured redundant columns.
    """
    out = df.copy()

    columns_to_drop = diagnostics_config.get(
        "columns_to_drop", []
    )

    existing_columns = [
        col for col in columns_to_drop
        if col in out.columns
    ]

    out = out.drop(columns=existing_columns)

    return out


def clean_dataset(
    df: pd.DataFrame,
    diagnostics_config: dict
) -> pd.DataFrame:
    """
    Apply all configured cleaning operations to the dataset.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        diagnostics_config (dict): Configuration for the cleaning operations.

    Returns:
        pd.DataFrame: The cleaned DataFrame with missing values preserved
        for later imputation.
    """
    out = missing_value_diagnostics(df, diagnostics_config)
    out = valid_numerical_ranges(out, diagnostics_config)
    out = canonicalize_categories(out, diagnostics_config)
    out = remove_duplicates(out, diagnostics_config)
    out = drop_redundant_columns(out, diagnostics_config)

    return out
