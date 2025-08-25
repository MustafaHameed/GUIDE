import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .build_dataset import build_oulad_dataset

logger = logging.getLogger(__name__)


def _load_or_build_dataset(
    dataset_path: Path, raw_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Load processed dataset or build it from raw files.

    Parameters
    ----------
    dataset_path:
        Path to the processed parquet file created by :mod:`build_dataset`.
    raw_dir:
        Optional path to the raw OULAD CSV files. If ``dataset_path`` does not
        exist and ``raw_dir`` is provided, the dataset will be built using
        :func:`build_oulad_dataset`.
    """
    if dataset_path.exists():
        return pd.read_parquet(dataset_path)

    if raw_dir is None:
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Provide `raw_dir` to build it."
        )

    logger.info("Processed dataset not found. Building using raw files...")
    df, _ = build_oulad_dataset(raw_dir, dataset_path)
    return df


def _select_target(df: pd.DataFrame, target_col: Optional[str]) -> str:
    """Determine which column to use as the regression target.

    Attempts to use ``target_col`` if provided, otherwise searches for common
    continuous outcome columns. Falls back to ``studied_credits`` which is
    present in the base dataset and numeric.
    """
    if target_col and target_col in df.columns:
        return target_col

    candidates: List[str] = [
        "final_score",
        "score",
        "assessment_mean_score",
        "assessment_last_score",
        "studied_credits",
    ]
    for col in candidates:
        if col in df.columns:
            logger.info("Using target column `%s`", col)
            return col

    raise ValueError(
        "No suitable target column found. Specify one with `--target-col`."
    )


def train_regression(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train a simple regression model and compute evaluation metrics."""

    df = df.dropna(subset=[target_col])

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "id_student",
            "code_module",
            "code_presentation",
            target_col,
            "label_pass",
            "label_fail_or_withdraw",
        }
    ]
    X = df[feature_cols]
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = LinearRegression()
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    parser = argparse.ArgumentParser(description="OULAD regression experiment")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/oulad/processed/oulad_ml.parquet"),
        help="Processed dataset path",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/oulad/raw"),
        help="Raw OULAD CSV directory (used if processed data is missing)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column to predict (continuous)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/oulad_regression_metrics.csv"),
        help="Path to save metrics CSV",
    )

    args = parser.parse_args()

    df = _load_or_build_dataset(args.data, args.raw_dir)
    target_col = _select_target(df, args.target_col)
    metrics = train_regression(df, target_col)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.output, index=False)

    logger.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main()
