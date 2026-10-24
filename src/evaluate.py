"""Development-set cross-validation and final holdout evaluation helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, get_scorer,
    make_scorer, precision_score, recall_score, f1_score, fbeta_score,
)
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


def build_scorers(metrics: list) -> dict:
    """Build the reported scorers; binary metrics use class 1 and zero_division=0."""
    scorers = {}
    safe = {"precision": precision_score, "recall": recall_score, "f1": f1_score}
    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_", 1)[1])
            if not np.isfinite(beta) or beta <= 0:
                raise ValueError("F-beta requires a positive finite beta.")
            scorers[name] = make_scorer(fbeta_score, beta=beta, zero_division=0)
        elif name in safe:
            scorers[name] = make_scorer(safe[name], zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    if not scorers:
        raise ValueError("Configure at least one metric.")
    return scorers


def positive_proba(estimator, X):
    """Return class-1 probabilities explicitly, independent of class-column order."""
    classes = np.asarray(estimator.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("This project expects binary labels 0 and 1.")
    values = estimator.predict_proba(X)[:, np.flatnonzero(classes == 1)[0]]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid positive-class probabilities.")
    return values


def partition_folds(cv, X, y):
    """Materialize a partition and reject overlap or missing/repeated OOF rows."""
    folds = list(cv.split(X, y)) if hasattr(cv, "split") else list(cv)
    counts = np.zeros(len(y), dtype=int)
    for train, val in folds:
        if np.intersect1d(train, val).size:
            raise ValueError("Training and validation rows must be disjoint.")
        np.add.at(counts, val, 1)
    if not np.all(counts == 1):
        raise ValueError("Each row must occur in exactly one validation fold.")
    return folds


def cross_validate_pipeline(
    pipeline, X, y, cv_config, scorers=None, selection_metric=None, n_jobs=1
):
    """Refit the complete pipeline inside each development-set fold.

    With scorers supplied, return (fold-score DataFrame, class-1 OOF
    probabilities). Every scorer is evaluated on validation rows; train and
    gap columns describe selection_metric. OOF order matches X and y.

    The legacy single-metric cv_config call still returns (text, hard labels).
    Both modes reuse the fitted fold estimators; no reserved test rows are used.
    """
    if scorers is not None:
        if selection_metric not in scorers:
            raise ValueError("selection_metric must be one of the reported metrics.")
        folds = partition_folds(cv_config, X, y)
        scores = cross_validate(
            pipeline, X, y, cv=folds, scoring=scorers, n_jobs=n_jobs,
            return_train_score=True, return_estimator=True,
            return_indices=True, error_score="raise",
        )
        frame = pd.DataFrame({"fold": range(1, len(folds) + 1)})
        frame["train"] = scores[f"train_{selection_metric}"]
        frame["validation"] = scores[f"test_{selection_metric}"]
        frame["gap"] = frame["train"] - frame["validation"]
        for name in scorers:
            frame[name] = scores[f"test_{name}"]
        probabilities = np.empty(len(y))
        for estimator, indices in zip(scores["estimator"], scores["indices"]["test"]):
            probabilities[indices] = positive_proba(estimator, X.iloc[indices])
        return frame, probabilities

    # Compatibility with the Week 4-6 single-metric API.
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
    y_true, y_pred, context: str = "development set, out-of-fold predictions",
    probabilities: bool = True, threshold: float = 0.5,
) -> str:
    """Threshold class-1 OOF probabilities at 0.5 and report a confusion matrix.

    Binary 0/1 hard predictions are also accepted, for the earlier API.
    """
    if probabilities:
        y_pred = (np.asarray(y_pred) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = (
        f"Classification report ({context}):\n"
        + classification_report(y_true, y_pred, zero_division=0)
        + "\nConfusion matrix (rows=true, columns=predicted; labels=[0, 1]):\n"
        + np.array2string(matrix)
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
    probabilities: bool = True,
    threshold: float = 0.5,
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
    if probabilities:
        y_pred = (np.asarray(y_pred) >= threshold).astype(int)
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


def cv_report(folds, metrics, selection_metric, title="Cross-validation"):
    """Format every metric and the selection metric's train-validation gap."""
    lines = [f"{title} ({len(folds)} stratified folds, selection metric: {selection_metric})",
             folds.to_string(index=False, float_format=lambda x: f"{x:.4f}"), ""]
    for name in metrics:
        lines.append(f"{name:<20s} {folds[name].mean():.4f} +/- {folds[name].std(ddof=1):.4f}")
    prefix = "Nested CV" if "Nested" in title else "CV"
    lines.extend([
        f"{prefix} {selection_metric} (mean +/- std): {folds['validation'].mean():.3f} +/- {folds['validation'].std(ddof=1):.3f}",
        f"Mean train-validation gap: {folds['gap'].mean():+.3f}",
    ])
    return "\n".join(lines)
