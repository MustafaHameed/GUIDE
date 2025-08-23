import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src import concepts


def test_group_concepts_aggregates_means():
    X = pd.DataFrame(
        {
            "absences": [1, 2],
            "failures": [0, 1],
            "Medu": [2, 3],
            "Fedu": [2, 4],
            "traveltime": [1, 2],
            "studytime": [3, 4],
            "goout": [2, 2],
            "Dalc": [1, 2],
            "Walc": [1, 2],
            "freetime": [3, 5],
        }
    )

    expected = pd.DataFrame(
        {
            "attendance": [0.5, 1.5],
            "socio_economic_status": [(2 + 2 + 1) / 3, (3 + 4 + 2) / 3],
            "engagement": [(3 + 2 + 1 + 1 + 3) / 5, (4 + 2 + 2 + 2 + 5) / 5],
        }
    )

    result = concepts.group_concepts(X)
    assert_frame_equal(result, expected)


def test_estimate_concept_effects_fallback(monkeypatch):
    X = pd.DataFrame(
        {
            "absences": [1, 2, 3],
            "failures": [0, 1, 1],
            "Medu": [1, 1, 1],
            "Fedu": [2, 2, 2],
            "traveltime": [1, 2, 1],
            "studytime": [1, 2, 3],
            "goout": [2, 3, 2],
            "Dalc": [1, 1, 1],
            "Walc": [1, 2, 1],
            "freetime": [3, 2, 3],
        }
    )
    y = pd.Series([0, 1, 0])

    class DummyLogisticRegression:
        def __init__(self, max_iter=None):
            self.max_iter = max_iter

        def fit(self, X, y):
            self.coef_ = np.array([[0.2, 0.1, 0.3]])
            return self

    monkeypatch.setattr(
        "sklearn.linear_model.LogisticRegression", DummyLogisticRegression
    )
    monkeypatch.setattr(concepts, "HAS_DOWHY", False)

    effects = concepts.estimate_concept_effects(X, y)

    assert list(effects["concept"]) == [
        "engagement",
        "attendance",
        "socio_economic_status",
    ]
    assert list(effects["effect"]) == [0.3, 0.2, 0.1]
