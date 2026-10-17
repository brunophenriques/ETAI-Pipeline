"""Model construction."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier


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
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(feature_preprocessor, model_config: dict) -> Pipeline:
    """Combine an unfitted feature preprocessor and classifier."""
    return Pipeline([
        ("preprocessing", feature_preprocessor),
        ("classifier", build_model(model_config)),
    ])
