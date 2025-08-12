"""Model factory utilities."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def create_model(model_type: str = "logistic", **kwargs):
    """Create and configure a classification model.

    Parameters
    ----------
    model_type:
        The type of model to create. Supported options are ``"logistic"``,
        ``"random_forest"``, and ``"gradient_boosting"``.
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
    raise ValueError(f"Unsupported model_type: {model_type}")