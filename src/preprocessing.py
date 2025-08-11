from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from .model import create_model


def build_pipeline(X):
    """Build preprocessing and modeling pipeline."""
    numeric_features = X.select_dtypes(include='number').columns
    categorical_features = X.select_dtypes(exclude='number').columns

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    clf = create_model()
    return Pipeline(steps=[('preprocess', preprocess), ('model', clf)])
