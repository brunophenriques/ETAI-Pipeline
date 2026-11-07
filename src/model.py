"""Model construction."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectorMixin, f_classif
from sklearn.utils.validation import check_is_fitted, validate_data
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, RandomForestClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier


class VoteSelector(SelectorMixin, BaseEstimator):
    """Keep encoded columns selected by at least min_votes of three voters.

    A forest and gradient booster vote above their mean importance; an L1
    logistic voter selects nonzero coefficients. Fit is called inside the
    training folds by Pipeline, never globally before CV.
    """

    def __init__(self, min_votes=2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        if (isinstance(self.min_votes, bool)
                or not isinstance(self.min_votes, (int, np.integer))
                or self.min_votes not in (1, 2, 3)):
            raise ValueError("min_votes must be 1, 2 or 3.")
        X, y = validate_data(self, X, y)
        forest = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20,
            random_state=self.random_state, n_jobs=1,
        ).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(
            l1_ratio=1.0, C=.01, solver="liblinear", max_iter=2000,
            random_state=self.random_state,
        ).fit(X, y)
        self.votes_ = (
            (forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
            + (booster.feature_importances_ > booster.feature_importances_.mean()).astype(int)
            + (np.abs(lasso.coef_).max(axis=0) > 1e-8).astype(int)
        )
        self.support_ = self.votes_ >= self.min_votes
        if not self.support_.any():
            raise ValueError("VoteSelector selected no features; reduce min_votes.")
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_


def adaboost(max_depth: int = 1, **params):
    """Boost shallow trees; tune depth as classifier__estimator__max_depth."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth), **params
    )


def adaboost_logistic_regression(C: float = 1.0, max_iter: int = 2000, **params):
    """Boost logistic classifiers; tune C as classifier__estimator__C."""
    return AdaBoostClassifier(
        estimator=LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs"),
        **params,
    )


def oblique_random_forest(
    n_estimators: int = 100,
    n_components: int = 10,
    density="auto",
    random_state=None,
    n_jobs=None,
    **tree_params,
):
    """Bag trees with a separate sparse random projection per tree.

    This follows the notebook's approximation: projections are drawn per
    tree, rather than per node. Bagging seeds each projection and tree.
    Tune tree parameters as classifier__estimator__tree__<parameter> and
    projection parameters as classifier__estimator__project__<parameter>.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(
        estimator=oblique_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
    "adaboost_logistic_regression": adaboost_logistic_regression,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "xgboost": XGBClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_selector(selection_config: dict):
    """Construct a learned selector; all fitting is left to Pipeline.fit."""
    method = selection_config.get("method", "vote")
    seed = selection_config.get("random_state")
    if method == "vote":
        return VoteSelector(min_votes=selection_config.get("min_votes", 2), random_state=seed)
    if method == "kbest":
        return SelectKBest(score_func=f_classif, k=selection_config.get("k", 5))
    if method == "from_model":
        return SelectFromModel(
            RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                   random_state=seed, n_jobs=1),
            threshold=selection_config.get("threshold", "mean"),
        )
    raise ValueError("feature_selection.method must be vote, kbest or from_model.")


def selection_search_space(selection_config: dict) -> dict:
    """Return only this method's tuning parameters; accept the earlier vote API."""
    if not selection_config.get("enabled", False):
        return {}
    method = selection_config.get("method", "vote")
    if method not in {"vote", "kbest", "from_model"}:
        raise ValueError("Unknown feature selection method.")
    if "search_spaces" in selection_config:
        return dict(selection_config["search_spaces"].get(method, {}))
    return dict(selection_config.get("search_space", {})) if method == "vote" else {}


def build_pipeline(feature_preprocessor, model_config: dict, selection_config=None, memory=None) -> Pipeline:
    """Combine an unfitted feature preprocessor and classifier."""
    steps = [("preprocessing", feature_preprocessor)]
    if selection_config and selection_config.get("enabled", False):
        steps.append(("select", build_selector(selection_config)))
    steps.append(("classifier", build_model(model_config)))
    return Pipeline(steps, memory=memory)
