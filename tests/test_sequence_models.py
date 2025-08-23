from pathlib import Path

import pandas as pd

from src.sequence_models import evaluate_sequence_model


def _make_sequence_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "G1": [10, 8, 12, 7, 14, 9, 11, 6, 13, 5],
            "G2": [11, 9, 13, 8, 15, 10, 12, 7, 14, 6],
            "G3": [12, 9, 14, 8, 16, 9, 13, 7, 15, 5],
            "studytime": [2, 1, 3, 2, 4, 1, 2, 1, 3, 2],
        }
    )
    csv_path = tmp_path / "seq.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_evaluate_sequence_hmm(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    csv_path = _make_sequence_csv(tmp_path)
    df = evaluate_sequence_model(str(csv_path), model_type="hmm")
    assert list(df["steps"]) == [1, 2]
    assert Path("tables/sequence_hmm_performance.csv").exists()

