import logging
from pathlib import Path

logging.getLogger("streamlit").setLevel("ERROR")

from dashboard import _list_images


def test_list_images_case_insensitive(tmp_path):
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
