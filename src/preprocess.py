"""
Preprocessing functions for preparing cleaned data for model training.

The preprocessing process:
    - creates missingness indicators for MNAR columns;
    - separates features, target, and fairness columns;
    - creates stratified training and test sets;
    - builds an unfitted imputation, scaling, and encoding recipe.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

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

    Numerical columns are scaled before imputation using training-fold
    statistics. Scalers preserve NaN, allowing KNN distances to use comparable
    units. Scaling options are none, standard, and robust.
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

    scaler_name = preprocessing_config.get("scaler", "none")
    scaler_factories = {
        "none": lambda: "passthrough",
        "standard": StandardScaler,
        "robust": RobustScaler,
    }
    if scaler_name not in scaler_factories:
        raise ValueError(
            f"Unknown scaler: {scaler_name}. Options: {list(scaler_factories)}"
        )

    strategy = preprocessing_config.get("numerical_imputation", "median")
    if strategy == "knn":
        if scaler_name == "none":
            raise ValueError("KNN imputation requires scaler: standard or robust.")
        neighbors = preprocessing_config.get("n_neighbors", 5)
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        imputer = KNNImputer(n_neighbors=neighbors, keep_empty_features=True)
    else:
        imputer = SimpleImputer(strategy=strategy, keep_empty_features=True)

    numerical_pipeline = Pipeline([
        ("scaler", scaler_factories[scaler_name]()),
        ("imputer", imputer),
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
    Prepare the data splits and an unfitted feature preprocessor.

    The function cleans the dataset, creates MNAR indicators, separates
    features and target, and splits the data. Feature imputation and encoding
    are fitted later inside the model pipeline, using training rows only.

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
        tuple: Untransformed training features, test features, training targets,
        test targets, development fairness columns, test fairness columns,
        and the unfitted feature preprocessor.
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

    # Preserve recorded groups for OOF auditing, including unknown race values.
    extras_dev = extras_train.copy()

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

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        extras_dev,
        extras_test,
        feature_preprocessor,
    )
