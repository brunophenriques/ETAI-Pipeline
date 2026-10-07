"""Hyperparameter search using cross-validation of the complete pipeline."""

import optuna
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


def tune_pipeline(
    pipeline,
    X,
    y,
    cv,
    scoring: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Return an unfitted best pipeline and its seeded Optuna study.

    Supply development rows, or only outer-training rows during nested CV.
    Each trial clones the full pipeline; CV fits preprocessing and classifier
    independently on each training fold. Search-space keys are sklearn
    parameter paths, such as ``classifier__max_depth``.

    The best trial's mean CV score is a selection score, not an independent
    performance estimate. Evaluate the tuning procedure with nested CV.
    Trials run sequentially for reproducibility; n_jobs parallelizes CV.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if not search_space:
        raise ValueError("search_space must contain at least one parameter.")

    available_params = pipeline.get_params(deep=True)
    for name, spec in search_space.items():
        if name not in available_params:
            raise ValueError(f"Unknown pipeline parameter: {name}")
        if spec["type"] not in {"int", "float", "categorical"}:
            raise ValueError(f"Unknown search-space type for {name}: {spec['type']}")

    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            else:
                params[name] = trial.suggest_categorical(name, spec["choices"])

        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(
            candidate, X, y, cv=cv, scoring=scoring,
            n_jobs=n_jobs, error_score="raise",
        )
        trial.set_user_attr("fold_scores", scores.tolist())
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(
    pipeline,
    X,
    y,
    outer_cv,
    inner_cv,
    scoring: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate tuning with untouched outer validation folds.

    X and y must be positionally aligned pandas objects containing only
    development data. Supply stratified outer and inner CV splitters.
    Inner splits are generated locally on each outer training subset.
    Each row must appear in exactly one outer validation fold.

    Return outer-fold scores and selected parameters, plus predictions in
    the original positional order for classification and fairness reports.
    This does not select or fit a final pipeline on all development rows.
    """
    scorer = get_scorer(scoring)
    folds = list(outer_cv.split(X, y))
    validation_counts = np.zeros(len(y), dtype=int)
    for train_idx, val_idx in folds:
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError("Outer training and validation rows must be disjoint.")
        np.add.at(validation_counts, val_idx, 1)
    if not np.all(validation_counts == 1):
        raise ValueError("Each row must occur in exactly one outer validation fold.")

    y_oof = np.empty(len(y), dtype=np.asarray(y).dtype)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        best_pipeline, study = tune_pipeline(
            pipeline, X_train, y_train, inner_cv, scoring,
            search_space, n_trials, random_state, n_jobs,
        )
        best_pipeline.fit(X_train, y_train)

        train_score = float(scorer(best_pipeline, X_train, y_train))
        val_score = float(scorer(best_pipeline, X_val, y_val))
        y_oof[val_idx] = best_pipeline.predict(X_val)
        rows.append({
            "fold": fold,
            "train": train_score,
            "validation": val_score,
            "gap": train_score - val_score,
            "inner_best": float(study.best_value),
            **study.best_params,
        })

    return pd.DataFrame(rows), y_oof
