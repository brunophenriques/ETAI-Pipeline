"""
Preprocessing functions for preparing cleaned data for model training.

The preprocessing process:
    - creates missingness indicators for MNAR columns;
    - separates features, target, and fairness columns;
    - creates stratified training and test sets;
    - imputes missing values using training data;
    - one-hot encodes categorical features.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.cleaning import clean_dataset


def add_missingness_indicators(
    df: pd.DataFrame,
    indicator_columns: list
) -> pd.DataFrame:
    """
    Create binary indicators showing whether selected values were missing.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        indicator_columns (list): Columns that require missingness indicators.

    Returns:
        pd.DataFrame: DataFrame containing the missingness indicators.
    """
    out = df.copy()

    for col in indicator_columns:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)

    return out


def prepare_categorical_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert pandas categorical missing values to NumPy NaN values.

    This ensures that scikit-learn's SimpleImputer can process categorical
    columns created using pandas string data types.

    Args:
        df (pd.DataFrame): The input feature DataFrame.

    Returns:
        pd.DataFrame: DataFrame with compatible categorical missing values.
    """
    out = df.copy()

    categorical_columns = out.select_dtypes(
        include=["object", "string"]
    ).columns

    out[categorical_columns] = out[categorical_columns].astype(object)
    out[categorical_columns] = out[categorical_columns].where(
        out[categorical_columns].notna(),
        np.nan
    )

    return out


def impute_extra_columns(
    extras_train: pd.DataFrame,
    extras_test: pd.DataFrame,
    columns: list
):
    """
    Impute fairness columns using modes calculated from training data.

    Args:
        extras_train (pd.DataFrame): Training fairness columns.
        extras_test (pd.DataFrame): Test fairness columns.
        columns (list): Columns that require mode imputation.

    Returns:
        tuple: Imputed training and test fairness DataFrames.
    """
    train_out = extras_train.copy()
    test_out = extras_test.copy()

    for col in columns:
        if col not in train_out.columns:
            continue

        modes = train_out[col].mode(dropna=True)

        if modes.empty:
            continue

        training_mode = modes.iloc[0]
        train_out[col] = train_out[col].fillna(training_mode)

        if col in test_out.columns:
            test_out[col] = test_out[col].fillna(training_mode)

    return train_out, test_out


def build_feature_preprocessor(
    X_train: pd.DataFrame,
    preprocessing_config: dict
) -> ColumnTransformer:
    """
    Build the transformer used to impute and encode model features.

    Numerical columns are imputed with the configured numerical strategy.
    Categorical columns are imputed with the configured categorical
    strategy and then one-hot encoded.

    Args:
        X_train (pd.DataFrame): Training features.
        preprocessing_config (dict): Preprocessing configuration.

    Returns:
        ColumnTransformer: Transformer for model features.
    """
    numerical_columns = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = [
        col for col in X_train.columns
        if col not in numerical_columns
    ]

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=preprocessing_config.get(
                    "numerical_imputation",
                    "median"
                )
            )
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=preprocessing_config.get(
                    "categorical_imputation",
                    "most_frequent"
                )
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ])


def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    diagnostics_config: dict,
    preprocessing_config: dict,
    test_size: float,
    random_state: int
):
    """
    Apply the complete cleaning and preprocessing process.

    The function cleans the dataset, creates MNAR indicators, separates
    features and target, splits the data, imputes missing values, and
    encodes categorical features.

    Args:
        df (pd.DataFrame): The raw input DataFrame.
        target (str): Name of the target column.
        sensitive_attr (str): Attribute preserved for fairness auditing.
        drop_columns (list): Columns excluded from model features.
        diagnostics_config (dict): Cleaning configuration.
        preprocessing_config (dict): Preprocessing configuration.
        test_size (float): Proportion of data used for testing.
        random_state (int): Random seed used for reproducibility.

    Returns:
        tuple: Training features, test features, training targets,
        test targets, and test fairness columns.
    """
    out = clean_dataset(df, diagnostics_config)

    indicator_columns = preprocessing_config.get(
        "missing_indicator_columns", []
    )
    out = add_missingness_indicators(out, indicator_columns)

    y = out[target]

    extras_columns = [
        col for col in [sensitive_attr, "score_text"]
        if col in out.columns
    ]
    extras = out[extras_columns].copy()

    columns_to_exclude = [
        target,
        sensitive_attr,
        *drop_columns
    ]
    columns_to_exclude = [
        col for col in columns_to_exclude
        if col in out.columns
    ]

    X = out.drop(columns=columns_to_exclude)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        extras_train,
        extras_test
    ) = train_test_split(
        X,
        y,
        extras,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train = prepare_categorical_missing_values(X_train)
    X_test = prepare_categorical_missing_values(X_test)

    extra_columns = preprocessing_config.get(
        "extra_columns_to_impute", []
    )
    extras_train, extras_test = impute_extra_columns(
        extras_train,
        extras_test,
        extra_columns
    )

    feature_preprocessor = build_feature_preprocessor(
        X_train,
        preprocessing_config
    )

    X_train = feature_preprocessor.fit_transform(X_train)
    X_test = feature_preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, extras_test