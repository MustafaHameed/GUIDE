"""Preprocessing utilities for the student performance dataset."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .model import create_model


def build_pipeline(
    X, model_type: str = "logistic", model_params: dict | None = None
):
    """Build preprocessing and modeling pipeline.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix used to infer column types.
    model_type : str, default "logistic"
        Type of model to create. Passed to :func:`create_model`.
    model_params : dict | None, optional
        Keyword arguments supplied to :func:`create_model` for model
        hyperparameters.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline combining preprocessing and the model returned by
        :func:`create_model`.
    """

    # Explicitly defined ordinal columns and their orderings
    ordinal_info = {
        "studytime": [1, 2, 3, 4],
        "Dalc": [1, 2, 3, 4, 5],
        "Walc": [1, 2, 3, 4, 5],
    }
    ordinal_features = [col for col in ordinal_info if col in X.columns]
    ordinal_categories = [ordinal_info[col] for col in ordinal_features]

    numeric_features = [
        col
        for col in X.select_dtypes(include="number").columns
        if col not in ordinal_features
    ]
    categorical_features = X.select_dtypes(exclude="number").columns

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=ordinal_categories)),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("ord", ordinal_transformer, ordinal_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model_params = model_params or {}
    clf = create_model(model_type, **model_params)
    return Pipeline(steps=[("preprocess", preprocess), ("model", clf)])