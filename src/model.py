"""Model factory utilities."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    StackingRegressor,
    BaggingClassifier,
    BaggingRegressor,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:  # Optional dependencies
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - graceful fallback when not installed
    XGBClassifier = None

try:  # Optional dependencies
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - graceful fallback when not installed
    LGBMClassifier = None


def create_model(model_type: str = "logistic", **kwargs):
    """Create and configure a classification model.

    Parameters
    ----------
    model_type:
        The type of model to create. Supported options are ``"logistic"``,
        ``"random_forest"``, ``"gradient_boosting"``, ``"svm"``, ``"knn"``,
        ``"naive_bayes"``, ``"extra_trees"``, and optionally ``"xgboost"`` or
        ``"lightgbm"`` if those packages are installed.
    **kwargs:
        Additional keyword arguments passed to the model constructor. This
        allows tuning of hyperparameters such as ``C`` or ``class_weight`` for
        logistic regression, or ``n_estimators`` for tree-based models.

    Returns
    -------
    estimator:
        An instance of the requested scikit-learn estimator.
    """

    if model_type == "logistic":
        return LogisticRegression(max_iter=1000, **kwargs)
    if model_type == "random_forest":
        return RandomForestClassifier(**kwargs)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(**kwargs)
    if model_type == "svm":
        # Enable probability estimates for downstream metrics
        return SVC(probability=True, **kwargs)
    if model_type == "knn":
        return KNeighborsClassifier(**kwargs)
    if model_type == "naive_bayes":
        return GaussianNB(**kwargs)
    if model_type == "extra_trees":
        return ExtraTreesClassifier(**kwargs)
    if model_type == "decision_tree":
        return DecisionTreeClassifier(**kwargs)
    if model_type == "bagging":
        base = kwargs.pop("base_estimator", "decision_tree")
        task = kwargs.pop("task", "classification")
        return create_bagging(base, task=task, **kwargs)
    if model_type == "stacking":
        estimators = kwargs.pop("estimators", ["logistic", "random_forest"])
        final_est = kwargs.pop("final_estimator", "logistic")
        task = kwargs.pop("task", "classification")
        return create_stacking(estimators, final_est, task=task, **kwargs)
    if model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost not installed.")
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **kwargs,
        )
    if model_type == "lightgbm":
        if LGBMClassifier is None:  # pragma: no cover - requires optional package
            raise ImportError(
                "lightgbm is not installed. Install lightgbm to use this model."
            )
        return LGBMClassifier(**kwargs)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _resolve_estimators(names):
    """Utility to build estimator list for stacking."""

    return [(name, create_model(name)) for name in names]


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

    base_estimators = _resolve_estimators(estimators)
    final = (
        create_model(final_estimator)
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
        create_model(base_estimator)
        if isinstance(base_estimator, str)
        else base_estimator
    )
    if task == "regression":
        return BaggingRegressor(estimator=estimator, **kwargs)
    return BaggingClassifier(estimator=estimator, **kwargs)