"""Fold-local clipping and anomaly removal restricted to training."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, validate_data


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Learn numeric limits from training values; preserve missing entries."""

    def __init__(self, lower=.01, upper=.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
        X = validate_data(self, X, reset=True, ensure_all_finite="allow-nan")
        # All-missing columns stay missing and are left for the imputer.
        self.lower_ = np.full(X.shape[1], -np.inf)
        self.upper_ = np.full(X.shape[1], np.inf)
        observed = ~np.isnan(X).all(axis=0)
        self.lower_[observed], self.upper_[observed] = np.nanquantile(
            X[:, observed], [self.lower, self.upper], axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        X = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return np.clip(X, self.lower_, self.upper_)


class TrainingOnlyAnomalyPipeline(Pipeline):
    """Fit preprocessing on training rows, filter them, then fit selection/model.

    Inference uses the ordinary Pipeline steps and never removes observations.
    No imbalanced-learn dependency is needed for this fit-only operation.
    """

    def __init__(self, steps, *, memory=None, filter_method="isolation_forest",
                 fraction=.05, n_neighbors=60, random_state=42):
        super().__init__(steps, memory=memory)
        self.filter_method = filter_method
        self.fraction = fraction
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None, **params):
        from src.preprocess import AnomalyScore
        if params:
            raise ValueError("Training-only removal does not support fit metadata.")
        if not 0 < self.fraction < 1:
            raise ValueError("Removal fraction must be between 0 and 1.")
        boundary = next(i for i, (name, _) in enumerate(self.steps) if name in {"select", "classifier"})
        prep = Pipeline([(name, clone(step)) for name, step in self.steps[:boundary]], memory=self.memory)
        Z = prep.fit_transform(X, y)
        self.detector_ = AnomalyScore(method=self.filter_method, n_neighbors=self.n_neighbors,
                                     random_state=self.random_state)
        scores = self.detector_.fit_transform(Z)[:, -1]
        self.n_training_samples_ = len(Z)
        n_drop = int(np.ceil(self.fraction * len(Z)))
        self.removed_indices_ = np.argsort(-scores, kind="stable")[:n_drop]
        keep = np.ones(len(Z), dtype=bool)
        keep[self.removed_indices_] = False
        self.kept_indices_ = np.flatnonzero(keep)
        labels = np.asarray(y)
        if len(np.unique(labels[keep])) < 2:
            raise ValueError("Anomaly removal left fewer than two classes.")
        tail = Pipeline([(name, clone(step)) for name, step in self.steps[boundary:]], memory=self.memory)
        tail.fit(Z[keep], labels[keep])
        self.steps = prep.steps + tail.steps
        return self
