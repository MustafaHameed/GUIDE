from src.data import load_data
from src.preprocessing import build_pipeline


def test_pipeline_prediction_shapes(csv_path):
    X, y = load_data(csv_path)
    X_sample = X.head(10)
    y_sample = y.head(10)
    pipeline = build_pipeline(X_sample)
    pipeline.fit(X_sample, y_sample)
    preds = pipeline.predict(X_sample)
    probs = pipeline.predict_proba(X_sample)
    assert preds.shape == (10,)
    assert probs.shape == (10, 2)

