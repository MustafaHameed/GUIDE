"""Data loading utilities for the student performance dataset."""

from __future__ import annotations

import pandas as pd


def load_data(csv_path: str):
    """Load dataset and create binary target indicating pass/fail."""
    df = pd.read_csv(csv_path)
    df["pass"] = (df["G3"] >= 10).astype(int)
    X = df.drop(columns=["G3", "pass"])
    y = df["pass"]
    return X, y


def load_early_data(csv_path: str, upto_grade: int = 1):
    """Load dataset using only grades up to ``G{upto_grade}``.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    upto_grade : int, default 1
        Highest grade column to retain as a feature. Later grade columns are
        dropped to avoid using future information when training early warning
        models.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Feature matrix ``X`` containing only historical grades up to the
        specified grade, and binary target vector ``y`` indicating pass/fail
        based on ``G3``.
    """

    df = pd.read_csv(csv_path)
    df["pass"] = (df["G3"] >= 10).astype(int)

    future_grades = [f"G{i}" for i in range(upto_grade + 1, 4)]
    cols_to_drop = ["G3", "pass", *future_grades]
    X = df.drop(columns=cols_to_drop)
    y = df["pass"]
    return X, y

