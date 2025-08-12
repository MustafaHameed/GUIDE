from src.data import load_data


def test_load_data_shapes_and_labels(csv_path):
    X, y = load_data(csv_path)
    assert X.shape == (395, 32)
    assert y.shape == (395,)
    assert set(y.unique()) <= {0, 1}
