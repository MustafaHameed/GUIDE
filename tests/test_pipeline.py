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


def test_stacking_and_bagging(csv_path):
    X, y = load_data(csv_path)
    X_sample = X.head(10)
    y_sample = y.head(10)
    stack_params = {
        "estimators": ["logistic", "random_forest"],
        "final_estimator": "logistic",
    }
    stack_pipe = build_pipeline(
        X_sample, model_type="stacking", model_params=stack_params
    )
    stack_pipe.fit(X_sample, y_sample)
    assert stack_pipe.predict(X_sample).shape == (10,)

    bag_pipe = build_pipeline(
        X_sample, model_type="bagging", model_params={"base_estimator": "decision_tree"}
    )
    bag_pipe.fit(X_sample, y_sample)
    assert bag_pipe.predict(X_sample).shape == (10,)
