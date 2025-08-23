import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.segmentation import segment_students


@pytest.fixture
def small_csv(tmp_path, csv_path):
    """Create a tiny subset of the student dataset for fast tests."""
    df = pd.read_csv(csv_path).head(20)
    subset_path = tmp_path / "student_subset.csv"
    df.to_csv(subset_path, index=False)
    return subset_path


@pytest.fixture
def run_in_tmp(tmp_path, monkeypatch):
    """Run the segmentation in an isolated directory and clean up afterwards."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    # Remove generated output directories to keep repository tidy
    for directory in (tmp_path / "figures", tmp_path / "tables"):
        if directory.exists():
            shutil.rmtree(directory)


def test_segment_students_creates_outputs(small_csv, run_in_tmp):
    summary = segment_students(csv_path=str(small_csv), n_clusters=2)

    # Assert that each algorithm produced exactly two clusters
    assert summary.groupby("algorithm")["cluster"].nunique().eq(2).all()

    # Verify that expected output files were created
    assert Path("figures/segmentation_kmeans.png").exists()
    assert Path("tables/segmentation_summary.csv").exists()
