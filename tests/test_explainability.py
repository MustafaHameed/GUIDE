import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.explain.importance import ExplainabilityAnalyzer


def test_explainability_analyzer_permutation_and_local():
    """Permutation importance and local explanations run without SHAP."""
    X = pd.DataFrame({
        "a": np.random.randn(30),
        "b": np.random.randn(30),
    })
    y = (X["a"] + X["b"] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X, y)

    analyzer = ExplainabilityAnalyzer(model, X, X, y, y)
    analyzer.setup_shap_explainer()
    shap_vals = analyzer.compute_shap_values()
    assert shap_vals is None or shap_vals.shape[0] <= len(X)

    importance = analyzer.global_feature_importance(method="permutation")
    assert set(importance.columns) == {"feature", "importance"}
    assert len(importance) == 2

    local = analyzer.local_explanations([0, 1], method="shap")
    assert isinstance(local, dict)

    stability = analyzer.stability_by_group()
    assert stability == {}

