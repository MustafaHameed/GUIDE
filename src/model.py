"""Model factory utilities."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
    if model_type == "xgboost":
        if XGBClassifier is None:  # pragma: no cover - requires optional package
            raise ImportError(
                "xgboost is not installed. Install xgboost to use this model."
            )
        # Silence warning about label encoder for recent xgboost versions
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", **kwargs)
    if model_type == "lightgbm":
        if LGBMClassifier is None:  # pragma: no cover - requires optional package
            raise ImportError(
                "lightgbm is not installed. Install lightgbm to use this model."
            )
        return LGBMClassifier(**kwargs)
    raise ValueError(f"Unsupported model_type: {model_type}")