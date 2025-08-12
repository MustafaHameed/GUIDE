import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def csv_path():
    return str(Path(__file__).resolve().parent.parent / "student-mat.csv")
