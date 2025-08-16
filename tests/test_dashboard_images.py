import logging
from pathlib import Path

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