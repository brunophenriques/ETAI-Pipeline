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
from sklearn.model_selection import train_test_split

def standardize_categories(df: pd.DataFrame, columns_map: dict, placeholder_tokens: list) -> pd.DataFrame:
    """
    Standardize the categories in specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_map (dict): A dictionary mapping column names to their canonical categories.

    Returns:
        pd.DataFrame: The DataFrame with standardized categories in the specified column.
    """
    # Example standardization logic (to be replaced with actual logic)
    out =df.copy()

    for column, y in columns_map.items():
        if column not in out.columns:
            continue
        cleaned_column = out[column].astype(str).str.strip()
        lowercased_column = cleaned_column.str.lower()
        out[column] = lowercased_column.map(y).fillna(lowercased_column)
        out.loc[out[column].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan

    return out

def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Clean the dataset by removing rows with missing values and logging the number of removed rows. Also applies the diagnosis

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        diagnostics (dict): A dictionary to store diagnostic information.

    Returns:
        pd.DataFrame: The cleaned DataFrame with no missing values.
    """


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
