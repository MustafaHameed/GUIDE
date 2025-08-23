import sys
import os
import random
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def set_seed():
    """Ensure deterministic tests by fixing random seeds."""
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def csv_path():
    return str(Path(__file__).resolve().parent.parent / "student-mat.csv")
