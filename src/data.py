"""Data loading utilities for the student performance dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(
    csv_path: str | Path,
    pass_threshold: int = 10,
    task: str = "classification",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and return features/target for the requested task

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file.
    pass_threshold : int, default 10
        Minimum ``G3`` grade considered a passing score when ``task`` is
        ``"classification"``.
    task : {"classification", "regression"}, default "classification"
        Which prediction task to prepare the data for. Classification creates a
        binary ``pass`` target while regression uses the raw ``G3`` grade.
    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If ``task`` is not ``"classification"`` or ``"regression"``.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if task not in {"classification", "regression"}:
        raise ValueError(
            "task must be either 'classification' or 'regression', "
            f"got {task!r}"
        )

    df = pd.read_csv(path)

    if task == "classification":
        df["pass"] = (df["G3"] >= pass_threshold).astype(int)
        X = df.drop(columns=["G3", "pass"])
        y = df["pass"]
        return X, y

    X = df.drop(columns=["G3"])
    y = df["G3"]
    return X, y


def load_early_data(
    csv_path: str | Path,
    upto_grade: int = 1,
    pass_threshold: int = 10,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset using only grades up to ``G{upto_grade}``.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to the CSV file.
    upto_grade : int, default 1
        Highest grade column to retain as a feature. Later grade columns are
        dropped to avoid using future information when training early warning
        models.
    pass_threshold : int, default 10
        Minimum ``G3`` grade considered a passing score

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If ``upto_grade`` is not ``1`` or ``2``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Feature matrix ``X`` containing only historical grades up to the
        specified grade, and binary target vector ``y`` indicating pass/fail
        based on ``G3``.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if upto_grade not in {1, 2}:
        raise ValueError("upto_grade must be either 1 or 2")

    df = pd.read_csv(path)
    df["pass"] = (df["G3"] >= pass_threshold).astype(int)

    future_grades = [f"G{i}" for i in range(upto_grade + 1, 4)]
    cols_to_drop = ["G3", "pass", *future_grades]
    X = df.drop(columns=cols_to_drop)
    y = df["pass"]
    return X, y

