import pandas as pd
from pathlib import Path

from src.oulad.sequence_model import (
    load_click_sequences,
    pad_sequences,
    split_by_ids,
    train_sequence_model,
)


def _create_raw(tmp_path: Path) -> Path:
    student_vle = pd.DataFrame(
        {
            "id_student": [1, 1, 2, 2, 3, 3],
            "code_module": ["AAA"] * 6,
            "code_presentation": ["2013B"] * 6,
            "id_site": [11, 12, 11, 12, 11, 12],
            "date": [1, 2, 1, 2, 1, 2],
            "sum_click": [5, 6, 0, 1, 3, 4],
        }
    )
    student_registration = pd.DataFrame(
        {
            "id_student": [1, 2, 3],
            "code_module": ["AAA"] * 3,
            "code_presentation": ["2013B"] * 3,
            "final_result": ["Pass", "Fail", "Pass"],
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    student_vle.to_csv(raw_dir / "studentVle.csv", index=False)
    student_registration.to_csv(raw_dir / "studentRegistration.csv", index=False)
    return raw_dir


def test_load_and_pad(tmp_path):
    raw_dir = _create_raw(tmp_path)
    seqs, labels, ids = load_click_sequences(raw_dir)
    assert len(seqs) == 3
    assert labels == [1, 0, 1]
    X = pad_sequences(seqs, max_len=3)
    assert X.shape == (3, 3)


def test_train_sequence_model(tmp_path):
    raw_dir = _create_raw(tmp_path)
    splits = {"train": [1, 2], "val": [3], "test": [3]}
    acc = train_sequence_model(
        raw_dir,
        splits,
        max_seq_len=3,
        epochs=1,
        hidden_size=4,
        attention_output=tmp_path / "att.csv",
    )
    assert 0.0 <= acc <= 1.0
    assert (tmp_path / "att.csv").exists()
    df = pd.read_csv(tmp_path / "att.csv")
    assert set(df.columns) == {"timestep", "attention"}
