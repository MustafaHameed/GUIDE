import importlib
import sys
import types
import pandas as pd
import numpy as np
import pytest


class StreamlitStop(Exception):
    """Exception raised to simulate ``st.stop``"""


# Provide minimal stubs only if dashboard dependencies are missing
try:  # pragma: no cover - we don't exercise real modules here
    import streamlit  # noqa: F401
    import altair  # noqa: F401
except Exception:  # pragma: no cover
    streamlit = types.ModuleType("streamlit")
    streamlit.set_page_config = lambda *a, **k: None
    streamlit.sidebar = types.SimpleNamespace(
        text_input=lambda *a, **k: "",
        radio=lambda *a, **k: "Demo dataset",
        file_uploader=lambda *a, **k: None,
        selectbox=lambda *a, **k: "0",
    )
    streamlit.warning = lambda *a, **k: None
    streamlit.stop = lambda *a, **k: (_ for _ in ()).throw(StreamlitStop())
    streamlit.title = lambda *a, **k: None
    streamlit.metric = lambda *a, **k: None
    streamlit.dataframe = lambda *a, **k: None
    streamlit.subheader = lambda *a, **k: None

    class _Tab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    streamlit.tabs = lambda labels: [_Tab() for _ in labels]
    streamlit.success = lambda *a, **k: None
    streamlit.info = lambda *a, **k: None
    streamlit.selectbox = lambda *a, **k: ""
    streamlit.altair_chart = lambda *a, **k: None
    streamlit.error = lambda *a, **k: None
    streamlit.write = lambda *a, **k: None
    streamlit.button = lambda *a, **k: None
    sys.modules.setdefault("streamlit", streamlit)

    altair = types.ModuleType("altair")

    class _Chart:
        def __init__(self, *a, **k):
            pass

        def mark_bar(self, *a, **k):
            return self

        def mark_line(self, *a, **k):
            return self

        def encode(self, *a, **k):
            return self

    altair.Chart = _Chart
    altair.X = altair.Y = lambda *a, **k: None
    sys.modules.setdefault("altair", altair)

_fake_raw_df = pd.DataFrame(
    {
        "G1": [10, 12],
        "G2": [11, 13],
        "G3": [12, 14],
        "gender": ["M", "F"],
    }
)


def _fake_load_data(*args, **kwargs):
    X = _fake_raw_df.drop(columns=["G3"]).copy()
    y = pd.Series([1, 0])
    return X, y


def _fake_build_pipeline(X):
    class Pipe:
        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.array([[0.4, 0.6]])

    return Pipe()


def _patch_common(monkeypatch, token=""):
    import streamlit as st

    monkeypatch.setattr(st.sidebar, "text_input", lambda *a, **k: token)
    monkeypatch.setattr(
        st.sidebar, "selectbox", lambda prompt, options, **k: options[0]
    )
    monkeypatch.setattr(
        st, "stop", lambda *a, **k: (_ for _ in ()).throw(StreamlitStop())
    )
    monkeypatch.setattr("src.data.load_data", _fake_load_data)
    monkeypatch.setattr("src.preprocessing.build_pipeline", _fake_build_pipeline)
    monkeypatch.setattr("src._safe_read_csv", lambda *a, **k: None)
    monkeypatch.setattr("src.clear_caches", lambda *a, **k: None)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: _fake_raw_df.copy())


def _reload(module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_import_student_dashboard(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.delenv("STUDENT_TOKENS", raising=False)
    mod = _reload("dashboard_student")
    assert mod.student_id is None


def test_student_token_mapping(monkeypatch):
    _patch_common(monkeypatch, token="tok1")
    monkeypatch.setenv("STUDENT_TOKENS", "0:tok1")
    mod = _reload("dashboard_student")
    assert mod._token_to_id["tok1"] == "0"
    assert mod.student_id == "0"


def test_student_invalid_token_stops(monkeypatch):
    _patch_common(monkeypatch, token="bad")
    monkeypatch.setenv("STUDENT_TOKENS", "0:tok1")
    with pytest.raises(StreamlitStop):
        _reload("dashboard_student")
