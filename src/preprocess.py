"""
Preprocessing functions for preparing cleaned data for model training.

The preprocessing process:
    - optionally creates fixed engineered features after cleaning;
    - creates missingness indicators for MNAR columns;
    - separates features, target, and fairness columns;
    - creates stratified training and test sets;
    - builds an unfitted imputation, scaling, and encoding recipe.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.cleaning import clean_dataset


class AnomalyScore(TransformerMixin, BaseEstimator):
    """Append unusualness learned from training features, never from labels.

    LOF fit_transform uses training negative_outlier_factor_; transform uses
    novelty scores for validation and new rows, without refitting the detector.
    """

    def __init__(self, method="isolation_forest", n_neighbors=20, random_state=None):
        self.method = method
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_data(self, X, reset=True)
        if self.method == "isolation_forest":
            self.detector_ = IsolationForest(random_state=self.random_state)
        elif self.method == "lof":
            if isinstance(self.n_neighbors, bool) or not isinstance(self.n_neighbors, int) or self.n_neighbors < 2:
                raise ValueError("LOF n_neighbors must be an integer of at least 2.")
            self.detector_ = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        else:
            raise ValueError(f"Unknown anomaly method: {self.method}. Use isolation_forest or lof.")
        self.detector_.fit(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        if self.method == "lof":
            return np.column_stack([X, -self.detector_.negative_outlier_factor_])
        return self.transform(X)

    def transform(self, X):
        check_is_fitted(self, "detector_")
        X = validate_data(self, X, reset=False)
        return np.column_stack([X, -self.detector_.score_samples(X)])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "detector_")
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        if len(input_features) != self.n_features_in_:
            raise ValueError("Feature names must match the detector's input columns.")
        return np.append(np.asarray(input_features, dtype=object), "anomaly_score")


_NUMERIC_IMPUTERS = {
    "median": lambda cfg, seed: SimpleImputer(strategy="median", keep_empty_features=True),
    "mean": lambda cfg, seed: SimpleImputer(strategy="mean", keep_empty_features=True),
    "knn": lambda cfg, seed: KNNImputer(n_neighbors=cfg.get("n_neighbors", 5), keep_empty_features=True),
    "iterative": lambda cfg, seed: IterativeImputer(max_iter=cfg.get("max_iter", 10), random_state=seed,
        sample_posterior=cfg.get("sample_posterior", False), keep_empty_features=True),
    "random_forest": lambda cfg, seed: IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=cfg.get("n_estimators", 50), random_state=seed, n_jobs=1),
        max_iter=cfg.get("max_iter", 10), random_state=seed, keep_empty_features=True),
    "most_frequent": lambda cfg, seed: SimpleImputer(strategy="most_frequent", keep_empty_features=True),
    "constant": lambda cfg, seed: SimpleImputer(strategy="constant", keep_empty_features=True),
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fixed row-wise formulas to cleaned data, without learning statistics.

    Missing inputs remain missing; learned imputation is left to the pipeline.
    Call after cleaning/deduplication so feature creation cannot change which
    observations enter the reserved split.
    """
    out = df.copy()
    out["priors_per_adult_year"] = (
        out["priors_count"] / (out["age"] - 17).clip(lower=1)
    )
    juvenile_columns = ["juv_fel_count", "juv_misd_count", "juv_other_count"]
    out["juvenile_offenses_total"] = out[juvenile_columns].sum(
        axis=1, min_count=len(juvenile_columns)
    )
    out["age_x_priors"] = out["age"] * out["priors_count"]
    return out


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

    settings = {**preprocessing_config, **preprocessing_config.get("imputation", {})}
    strategy = settings.get("numeric_strategy", settings.get("numerical_imputation", "median"))
    if strategy == "forest_proximity":
        from copy import deepcopy
        from src.imputation import ForestProximityImputer
        remaining = deepcopy(preprocessing_config)
        remaining.setdefault("imputation", {}).update(numeric_strategy="median", include_categories=False)
        return Pipeline([
            ("proximity", ForestProximityImputer(n_estimators=settings.get("n_estimators", 50),
                                                random_state=preprocessing_config.get("random_state", 42))),
            ("remaining", build_feature_preprocessor(X_train, remaining)),
        ])
    if strategy == "knn":
        if scaler_name == "none":
            raise ValueError("KNN imputation requires scaler: standard or robust.")
        neighbors = settings.get("n_neighbors", 5)
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")
    if strategy not in _NUMERIC_IMPUTERS:
        raise ValueError(f"Unknown numeric imputer: {strategy}. Options: {list(_NUMERIC_IMPUTERS)}")
    imputer = _NUMERIC_IMPUTERS[strategy](settings, preprocessing_config.get("random_state", 42))

    numeric_steps = []
    clip = preprocessing_config.get("quantile_clip", {})
    if clip.get("enabled", False):
        from src.outliers import QuantileClipper
        numeric_steps.append(("clip", QuantileClipper(lower=clip.get("lower", .01), upper=clip.get("upper", .99))))
    numeric_steps += [
        ("scaler", scaler_factories[scaler_name]()),
    ]
    numerical_pipeline = Pipeline(numeric_steps + [("imputer", imputer)])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy=settings.get("categorical_strategy", preprocessing_config.get("categorical_imputation", "most_frequent"))
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

    if settings.get("include_categories", False):
        if strategy not in {"knn", "iterative", "random_forest"}:
            raise ValueError("include_categories requires knn, iterative or random_forest.")
        # Fit category modes/encoding on training rows before joint imputation.
        # Imputed numeric values are kept alongside the observed encoded categories.
        encoded = ColumnTransformer([
            ("numerical", Pipeline(numeric_steps), numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ])
        return Pipeline([("encode", encoded), ("imputer", imputer)])

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

    The function cleans the dataset, optionally adds fixed engineered features,
    creates MNAR indicators, separates
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
    if preprocessing_config.get("engineered_features", {}).get("enabled", False):
        out = add_engineered_features(out)

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



def recover_recency(
    df: pd.DataFrame,
    history: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Recover missing recency using strictly earlier purchases.

    history must contain only transactions available
    at prediction time.

    Does not learn imputation statistics.
    """
    data = df.copy()

    settings = config["preprocessing"]["recency"]

    if not settings["recover_from_previous_date"]:
        return data

    date_col = config["preprocessing"][
        "date_features"
    ]["source_column"]

    customer_col = config["columns"]["customer_id"]

    data[date_col] = pd.to_datetime(data[date_col])

    past = history[[customer_col, date_col]].copy()
    past[date_col] = pd.to_datetime(past[date_col])

    past = past.dropna(subset=[date_col])
    past = past.rename(
        columns={date_col: "historical_purchase_date"}
    )

    # merge_asof requires chronological sorting.
    left = data.assign(_row_order=range(len(data)))
    left = left.sort_values(date_col)

    right = past.sort_values("historical_purchase_date")

    merged = pd.merge_asof(
        left,
        right,
        left_on=date_col,
        right_on="historical_purchase_date",
        by=customer_col,
        direction="backward",
        allow_exact_matches=False,
    )

    missing = merged["days_since_previous_purchase"].isna()

    recoverable = (
        missing
        & (merged["prior_purchase_count"] >= 1)
        & merged["historical_purchase_date"].notna()
    )

    merged["recency_was_recovered"] = (
        recoverable.astype(int)
    )

    merged.loc[recoverable, "days_since_previous_purchase"] = (
        merged.loc[recoverable, date_col]
        - merged.loc[recoverable, "historical_purchase_date"]
    ).dt.days.to_numpy()

    merged = (
        merged.sort_values("_row_order")
        .drop(columns=["_row_order", "historical_purchase_date"])
    )

    merged.index = data.index

    return merged
