from src.data import load_data, load_early_data


def test_load_data_shapes_and_labels(csv_path):
    X, y = load_data(csv_path)
    assert X.shape == (395, 32)
    assert y.shape == (395,)
    assert set(y.unique()) <= {0, 1}


def test_load_early_data(csv_path):
    X, y = load_early_data(csv_path, upto_grade=1)
    assert X.shape == (395, 31)
    assert y.shape == (395,)
    assert set(y.unique()) <= {0, 1}


def test_pass_threshold(csv_path):
    _, y_default = load_data(csv_path)
    _, y_strict = load_data(csv_path, pass_threshold=15)
    assert y_strict.sum() < y_default.sum()
