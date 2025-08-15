"""High level workflow for loading data, running EDA and preparing models.

This module is intended to be executed as a script (``python -m src.data_workflow``)
but its :func:`main` function can also be imported and reused.  The workflow is:

1. Load the dataset.
2. Run exploratory data analysis and save figures/tables.
3. Build the preprocessing/model pipeline and optionally fit it.

The CSV path as well as output directories for the EDA artifacts can be
configured via command line arguments when run as a module.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from .data import load_data
from .preprocessing import build_pipeline
from .eda import run_eda


def main(
    csv_path: str = "student-mat.csv",
    figures_dir: str | Path = "figures",
    tables_dir: str | Path = "tables",
    fit: bool = False,
):
    """Run the full data workflow.

    Parameters
    ----------
    csv_path:
        Location of the raw CSV file.
    figures_dir, tables_dir:
        Directories where EDA output is written.
    fit:
        If ``True`` the preprocessing/model pipeline is fitted to the entire
        dataset after construction.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The constructed (and optionally fitted) pipeline.
    """

    csv_path = Path(csv_path)
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)

    # Run EDA on the full dataframe
    df = pd.read_csv(csv_path)
    run_eda(df, fig_dir=figures_dir, table_dir=tables_dir)

    # Build pipeline using feature matrix returned by ``load_data``
    X, y = load_data(str(csv_path))
    pipeline = build_pipeline(X)
    if fit:
        pipeline.fit(X, y)
    return pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data workflow")
    parser.add_argument(
        "--csv-path",
        default="student-mat.csv",
        help="Path to CSV file containing the dataset",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory to store generated figures",
    )
    parser.add_argument(
        "--tables-dir",
        default="tables",
        help="Directory to store generated tables",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Fit the pipeline after constructing it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        csv_path=args.csv_path,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        fit=args.fit,
    )
