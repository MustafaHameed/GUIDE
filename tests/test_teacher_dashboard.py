import importlib
import sys
import types
import pandas as pd
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


def _fake_load_data(*args, **kwargs):
    X = pd.DataFrame({"gender": ["M", "F"], "G1": [10, 12]})
    y = pd.Series([1, 0])
    return X, y


def _patch_common(monkeypatch, token=""):
    import streamlit as st

    monkeypatch.setattr(st.sidebar, "text_input", lambda *a, **k: token)
    monkeypatch.setattr(st, "selectbox", lambda prompt, options, **k: options[0])
    monkeypatch.setattr(
        st, "stop", lambda *a, **k: (_ for _ in ()).throw(StreamlitStop())
    )
    monkeypatch.setattr("src.data.load_data", _fake_load_data)
    monkeypatch.setattr("src._list_images", lambda *a, **k: [])
    monkeypatch.setattr("src._show_images_grid", lambda *a, **k: None)
    monkeypatch.setattr("src._show_table", lambda *a, **k: None)


def _reload(module_name):
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_import_teacher_dashboard(monkeypatch):
    _patch_common(monkeypatch)
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    mod = _reload("dashboard_teacher")
    assert mod.AUTH_TOKEN is None


def test_teacher_requires_valid_token(monkeypatch):
    _patch_common(monkeypatch, token="wrong")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "secret")
    with pytest.raises(StreamlitStop):
        _reload("dashboard_teacher")
