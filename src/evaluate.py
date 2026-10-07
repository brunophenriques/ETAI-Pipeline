"""Development-set cross-validation and final holdout evaluation helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from scipy.stats import t


def corrected_paired_comparison(differences, validation_train_ratio: float) -> dict:
    """Approximate Nadeau-Bengio paired test; positive means KNN wins.

    Corrected SE = sqrt((1/n + n_validation/n_train) * sample variance).
    Use a two-sided test and a 95% confidence interval. Folds overlap in
    training data, so their differences must not be treated as independent.
    Reference: sklearn's plot_grid_search_stats example.
    """
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("Provide at least two finite paired score differences.")
    if not np.isfinite(validation_train_ratio) or validation_train_ratio <= 0:
        raise ValueError("The validation/train size ratio must be positive.")
    mean = float(differences.mean())
    se = float(np.sqrt((1 / len(differences) + validation_train_ratio)
                       * differences.var(ddof=1)))
    df = len(differences) - 1
    # Zero variance gives no uncertainty estimate for a nonzero difference.
    p_value = (1.0 if mean == 0 else None) if se == 0 else float(2 * t.sf(abs(mean / se), df))
    margin = float(t.ppf(0.975, df) * se)
    return {
        "mean_difference_knn_minus_median": mean,
        "corrected_standard_error": se,
        "degrees_of_freedom": df,
        "two_sided_p_value": p_value,
        "ci95": [mean - margin, mean + margin],
    }


def cross_validate_pipeline(pipeline, X, y, cv_config: dict) -> tuple[str, np.ndarray]:
    """Refit the complete pipeline inside each development-set fold.

    Return a printable report with per-fold scores, sample standard
    deviations, and the training-validation gap, plus one out-of-fold
    prediction per development row in its original positional order.
    Predictions reuse the fold estimators; no test rows are used.
    """
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    scoring = cv_config.get("scoring", "accuracy")
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=cv_config.get("n_jobs", 1),
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        error_score="raise",
    )
    y_oof = np.empty(len(y), dtype=np.asarray(y).dtype)
    for estimator, validation_indices in zip(
        scores["estimator"], scores["indices"]["test"]
    ):
        y_oof[validation_indices] = estimator.predict(X.iloc[validation_indices])
    folds = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    folds["gap"] = folds["train"] - folds["validation"]

    lines = [
        f"Cross-validation ({len(folds)} stratified folds, metric: {scoring})",
        "",
        folds.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
    ]
    for column in ["train", "validation", "gap"]:
        lines.append(
            f"{column.capitalize():<11s} mean = {folds[column].mean():.3f}   "
            f"std = {folds[column].std(ddof=1):.3f}"
        )
    lines.extend([
        "",
        f"CV {scoring} (mean +/- std): {folds['validation'].mean():.3f} "
        f"+/- {folds['validation'].std(ddof=1):.3f}",
        f"Mean train-validation gap: {folds['gap'].mean():+.3f}",
    ])
    report = "\n".join(lines)
    print(report)
    return report, y_oof


def format_nested_cv_report(folds: pd.DataFrame, scoring: str) -> str:
    """Summarize outer evaluation scores and each fold's tuning choices."""
    lines = [
        f"Nested cross-validation ({len(folds)} stratified outer folds, metric: {scoring})",
        "Inner best scores selected parameters; outer validation scores evaluate tuning.",
        "",
        folds.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
    ]
    for column in ["train", "validation", "gap"]:
        lines.append(
            f"{column.capitalize():<11s} mean = {folds[column].mean():.3f}   "
            f"std = {folds[column].std(ddof=1):.3f}"
        )
    lines.extend([
        "",
        f"Nested CV {scoring} (mean +/- std): {folds['validation'].mean():.3f} "
        f"+/- {folds['validation'].std(ddof=1):.3f}",
        f"Mean train-validation gap: {folds['gap'].mean():+.3f}",
    ])
    report = "\n".join(lines)
    print(report)
    return report


def oof_classification_report(
    y_true, y_pred, context: str = "development set, out-of-fold predictions"
) -> str:
    """Report classification metrics on unseen-fold development predictions."""
    report = (
        f"Classification report ({context}):\n"
        + classification_report(y_true, y_pred, zero_division=0)
    )
    print(report)
    return report


def evaluate(y_train, y_train_pred, y_test, y_pred) -> str:
    """
    Prints -- and returns as text, so it can also be saved to disk -- train accuracy and test accuracy side by side, plus the usual classification report on the test set.
    """
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    gap = train_accuracy - test_accuracy

    lines = [
        f"Train accuracy: {train_accuracy:.3f}",
        f"Test accuracy:  {test_accuracy:.3f}",
        f"Gap (train - test): {gap:+.3f}",
    ]
    lines.append("")
    lines.append("Classification report (test set):")
    lines.append(classification_report(y_test, y_pred))

    text = "\n".join(lines)
    print(text)
    return text


def fairness_report(
    y_test,
    y_pred,
    extras_test: pd.DataFrame,
    sensitive_attr: str = "race",
    context: str = "test set",
) -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show that "accuracy" and "fair" are not the same thing.

    For each race group, prints (and returns as text) the false
    positive rate (share of people who did NOT reoffend but were
    predicted to) for:
        - our own model
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), for comparison
    """
    df = extras_test.copy()
    df["y_true"] = np.asarray(y_test)
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)
    # Keep unknown race separate instead of assigning people to another group.
    df[sensitive_attr] = df[sensitive_attr].fillna("Unknown")

    lines = [
        f"False positive rate by race ({context})",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
