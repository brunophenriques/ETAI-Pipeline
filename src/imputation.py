"""Multiple-imputation prediction and a feature-only forest proximity imputer."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state


def reset_posterior_rng(estimator):
    """Reset fitted imputers, including fitted ColumnTransformer clones."""
    if getattr(estimator, "sample_posterior", False):
        estimator.random_state_ = check_random_state(estimator.random_state)
    children = getattr(estimator, "named_transformers_", getattr(estimator, "named_steps", {}))
    for child in children.values():
        if not isinstance(child, str):
            reset_posterior_rng(child)


class MultipleImputationPipeline(Pipeline):
    """Fit independent posterior-imputed pipelines and average probabilities."""

    def __init__(self, steps, *, memory=None, n_imputations=5, random_state=42):
        super().__init__(steps, memory=memory)
        self.n_imputations = n_imputations
        self.random_state = random_state

    def fit(self, X, y=None, **params):
        if isinstance(self.n_imputations, bool) or not isinstance(self.n_imputations, int) or self.n_imputations < 1:
            raise ValueError("n_imputations must be a positive integer.")
        self.pipelines_ = []
        for i in range(self.n_imputations):
            member = Pipeline([(name, clone(step)) for name, step in self.steps], memory=self.memory)
            seed_keys = [key for key in member.get_params()
                         if key.endswith("imputer__random_state")]
            if not seed_keys:
                raise ValueError("Multiple imputation requires an iterative numeric imputer.")
            member.set_params(**{key: self.random_state + i for key in seed_keys})
            member.fit(X, y, **params)
            self.pipelines_.append(member)
        # Pipeline's feature names/classes and the reporting API describe member 1.
        self.steps = self.pipelines_[0].steps
        return self

    def predict_proba(self, X, **params):
        check_is_fitted(self, "pipelines_")
        # Reproducible posterior draws for repeated predictions of the same CSV.
        for member in self.pipelines_:
            reset_posterior_rng(member)
        return np.mean([member.predict_proba(X, **params)
                        for member in self.pipelines_], axis=0)

    def predict(self, X, **params):
        return self.classes_[self.predict_proba(X, **params).argmax(axis=1)]


class ForestProximityImputer(TransformerMixin, BaseEstimator):
    """Impute one column from observed training donors sharing forest leaves.

    The forest predicts that feature, never the classification label. The target
    feature is excluded from the forest inputs, including at prediction time.
    """

    def __init__(self, target_column="priors_count", n_estimators=50, random_state=42):
        self.target_column = target_column
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        # Derived priors features would reveal the value being imputed.
        excluded = {self.target_column, "priors_per_adult_year", "age_x_priors"}
        self.predictors_ = [c for c in X.select_dtypes(include="number") if c not in excluded
                            and not c.endswith("_was_missing")]
        observed = X[self.target_column].notna()
        if not observed.any():
            raise ValueError("Proximity imputation needs observed training donors.")
        self.encoder_ = SimpleImputer(strategy="median", keep_empty_features=True)
        predictors = self.encoder_.fit_transform(X.loc[observed, self.predictors_])
        self.donor_values_ = X.loc[observed, self.target_column].to_numpy(dtype=float)
        self.fallback_ = float(np.median(self.donor_values_))
        self.forest_ = RandomForestRegressor(n_estimators=self.n_estimators,
                                            random_state=self.random_state, n_jobs=1)
        self.forest_.fit(predictors, self.donor_values_)
        self.donor_leaves_ = self.forest_.apply(predictors)
        return self

    def transform(self, X):
        check_is_fitted(self, "forest_")
        out = X.copy()
        missing = out[self.target_column].isna().to_numpy()
        leaves = self.forest_.apply(self.encoder_.transform(out.loc[missing, self.predictors_])) if missing.any() else []
        guesses = []
        for row in leaves:
            weights = (self.donor_leaves_ == row).mean(axis=1)
            guesses.append(float(weights @ self.donor_values_ / weights.sum())
                           if weights.sum() else self.fallback_)
        out.loc[missing, self.target_column] = guesses
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
