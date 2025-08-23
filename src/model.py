"""Model factory utilities."""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingClassifier,
    StackingRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:  # Optional dependencies
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - graceful fallback when not installed
    XGBClassifier = None

try:  # Optional dependencies
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - graceful fallback when not installed
    LGBMClassifier = None


def _make_xgboost(**kwargs):
    if XGBClassifier is None:
        raise ImportError("xgboost not installed.")
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **kwargs)


def _make_lightgbm(**kwargs):
    if LGBMClassifier is None:  # pragma: no cover - requires optional package
        raise ImportError(
            "lightgbm is not installed. Install lightgbm to use this model."
        )
    return LGBMClassifier(**kwargs)


def _make_bagging(task, **kwargs):
    base = kwargs.pop("base_estimator", "decision_tree")
    return create_bagging(base, task=task, **kwargs)


def _make_stacking(task, **kwargs):
    defaults = {
        "classification": ("logistic", ["logistic", "random_forest"]),
        "regression": ("linear", ["linear", "random_forest"]),
    }
    final_est, estimators = defaults[task]
    estimators = kwargs.pop("estimators", estimators)
    final_est = kwargs.pop("final_estimator", final_est)
    return create_stacking(estimators, final_est, task=task, **kwargs)


MODEL_CONSTRUCTORS = {
    ("classification", "logistic"): lambda **kw: LogisticRegression(max_iter=1000, **kw),
    ("classification", "random_forest"): RandomForestClassifier,
    ("classification", "gradient_boosting"): GradientBoostingClassifier,
    ("classification", "svm"): lambda **kw: SVC(probability=True, **kw),
    ("classification", "knn"): KNeighborsClassifier,
    ("classification", "naive_bayes"): GaussianNB,
    ("classification", "extra_trees"): ExtraTreesClassifier,
    ("classification", "mlp"): MLPClassifier,
    ("classification", "decision_tree"): DecisionTreeClassifier,
    ("classification", "bagging"): lambda **kw: _make_bagging("classification", **kw),
    ("classification", "stacking"): lambda **kw: _make_stacking("classification", **kw),
    ("classification", "xgboost"): _make_xgboost,
    ("classification", "lightgbm"): _make_lightgbm,
    ("regression", "linear"): LinearRegression,
    ("regression", "random_forest"): RandomForestRegressor,
    ("regression", "gradient_boosting"): GradientBoostingRegressor,
    ("regression", "svm"): SVR,
    ("regression", "knn"): KNeighborsRegressor,
    ("regression", "extra_trees"): ExtraTreesRegressor,
    ("regression", "decision_tree"): DecisionTreeRegressor,
    ("regression", "mlp"): MLPRegressor,
    ("regression", "bagging"): lambda **kw: _make_bagging("regression", **kw),
    ("regression", "stacking"): lambda **kw: _make_stacking("regression", **kw),
}


def create_model(model_type: str = "logistic", task: str = "classification", **kwargs):
    """Create and configure an estimator for the requested task.

    Parameters
    ----------
    model_type:
        The type of model to create. Supported options are ``"logistic"``,
        ``"random_forest"``, ``"gradient_boosting"``, ``"svm"``, ``"knn"``,
        ``"naive_bayes"``, ``"extra_trees"``, ``"mlp"``, and optionally
        ``"xgboost"`` or ``"lightgbm"`` if those packages are installed.
    task:
        ``"classification"`` or ``"regression"``.
    **kwargs:
        Additional keyword arguments passed to the model constructor. This
        allows tuning of hyperparameters such as ``C`` or ``class_weight`` for
        logistic regression, or ``n_estimators`` for tree-based models.

    Returns
    -------
    estimator:
        An instance of the requested scikit-learn estimator.

    Examples
    --------
    >>> create_model("random_forest")  # classification is default task
    RandomForestClassifier()
    >>> create_model("linear", task="regression")
    LinearRegression()
    """

    key = (task, model_type)
    try:
        constructor = MODEL_CONSTRUCTORS[key]
    except KeyError as exc:  # pragma: no cover - simple lookup
        raise ValueError(f"Unsupported task/model_type combination: {key}") from exc
    return constructor(**kwargs)


def _resolve_estimators(names, task="classification"):
    """Utility to build estimator list for stacking."""

    return [(name, create_model(name, task=task)) for name in names]


def create_stacking(
    estimators, final_estimator="logistic", task="classification", **kwargs
):
    """Build a stacking ensemble using existing base learners.

    Parameters
    ----------
    estimators:
        List of model type strings to use as base learners.
    final_estimator:
        Model type or estimator object used as the meta learner.
    task:
        "classification" or "regression". Determines which scikit-learn
        stacking class is used.
    **kwargs:
        Additional arguments passed to the stacking constructor.
    """

    base_estimators = _resolve_estimators(estimators, task=task)
    final = (
        create_model(final_estimator, task=task)
        if isinstance(final_estimator, str)
        else final_estimator
    )
    if task == "regression":
        return StackingRegressor(
            estimators=base_estimators, final_estimator=final, **kwargs
        )
    return StackingClassifier(
        estimators=base_estimators, final_estimator=final, **kwargs
    )


def create_bagging(base_estimator="decision_tree", task="classification", **kwargs):
    """Build a bagging ensemble around an existing learner."""

    estimator = (
        create_model(base_estimator, task=task)
        if isinstance(base_estimator, str)
        else base_estimator
    )
    if task == "regression":
        return BaggingRegressor(estimator=estimator, **kwargs)
    return BaggingClassifier(estimator=estimator, **kwargs)
