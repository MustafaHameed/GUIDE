import logging
from pathlib import Path
import sys
import types

# Provide minimal stubs only if dashboard dependencies are missing
try:  # pragma: no cover - we don't exercise the real modules in tests
    import streamlit  # noqa: F401
    import altair  # noqa: F401
except Exception:  # pragma: no cover
    streamlit_stub = types.ModuleType("streamlit")
    streamlit_stub.set_page_config = lambda *a, **k: None
    streamlit_stub.sidebar = types.SimpleNamespace(text_input=lambda *a, **k: "")
    streamlit_stub.warning = lambda *a, **k: None
    streamlit_stub.stop = lambda *a, **k: None
    streamlit_stub.title = lambda *a, **k: None
    streamlit_stub.caption = lambda *a, **k: None
    streamlit_stub.write = lambda *a, **k: None
    streamlit_stub.cache_data = lambda *a, **k: (lambda f: f)

    components_stub = types.ModuleType("streamlit.components")
    components_v1_stub = types.ModuleType("streamlit.components.v1")
    components_v1_stub.html = lambda *a, **k: None
    components_stub.v1 = components_v1_stub
    streamlit_stub.components = components_stub

    sys.modules.setdefault("streamlit", streamlit_stub)
    sys.modules.setdefault("streamlit.components", components_stub)
    sys.modules.setdefault("streamlit.components.v1", components_v1_stub)
    sys.modules.setdefault("altair", types.ModuleType("altair"))

logging.getLogger("streamlit").setLevel("ERROR")

from dashboard import _list_images


def test_list_images_case_insensitive(tmp_path):
    _list_images.clear()
    lower = tmp_path / "foo.png"
    lower.write_text("fake")
    upper = tmp_path / "bar.PNG"
    upper.write_text("fake")
    jpeg_upper = tmp_path / "baz.JPEG"
    jpeg_upper.write_text("fake")
    mixed = tmp_path / "mix.PnG"
    mixed.write_text("fake")

    results = _list_images(tmp_path)

    assert lower in results
    assert upper in results
    assert jpeg_upper in results
    assert mixed in results
    assert len(results) == 4


def test_list_images_cached(tmp_path, monkeypatch):
    _list_images.clear()

    sample = tmp_path / "img.png"
    sample.write_text("fake")

    calls = {"count": 0}
    original_glob = Path.glob

    def counting_glob(self, pattern):
        calls["count"] += 1
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", counting_glob)

    first = _list_images(tmp_path)
    assert calls["count"] > 0
    before = calls["count"]
    second = _list_images(tmp_path)
    assert second == first
    assert calls["count"] == before
