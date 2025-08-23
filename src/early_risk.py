"""Train a classifier using early grade data for risk assessment.

References
----------
- StratifiedKFold documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
- Fairlearn metrics: https://fairlearn.org/
"""

from __future__ import annotations
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)

try:
    from .data import load_early_data
    from .preprocessing import build_pipeline
except ImportError:
    # Fallback for direct execution
    from data import load_early_data
    from preprocessing import build_pipeline


logger = logging.getLogger(__name__)

def train_early(
    csv_path: str = "student-mat.csv",
    upto_grade: int = 1,
    model_type: str = "logistic",
    pass_threshold: int = 10,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Train a classifier using early grade data and report metrics.

    Parameters
    ----------
    csv_path:
        Path to the CSV file containing student data.
    upto_grade:
        The highest early grade column (e.g. ``G1`` or ``G2``) to include.
    model_type:
        Which model pipeline to build. Options depend on ``build_pipeline``.
    pass_threshold:
        Minimum final grade considered a pass when creating the target.
    group_cols:
        Optional list of column names for which to compute fairness metrics.

    Returns
    -------
    pd.DataFrame
        Data frame of permutation feature importances.

    Side Effects
    ------------
    Saves ROC curve and feature-importance plots under ``figures`` and
    corresponding CSV reports under ``reports``. Metrics are logged using the
    module logger.
    """
    X, y = load_early_data(
        csv_path, upto_grade=upto_grade, pass_threshold=pass_threshold
    )
    pipeline = build_pipeline(X, model_type=model_type)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=["accuracy", "f1"])
    logger.info(
        "CV accuracy %.3f +/- %.3f; F1 %.3f",
        scores["test_accuracy"].mean(),
        scores["test_accuracy"].std(),
        scores["test_f1"].mean(),
    )
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    y_prob = (
        pipeline.predict_proba(X)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )
    fig_dir = Path("figures"); fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("reports"); report_dir.mkdir(parents=True, exist_ok=True)

    # Simple train ROC (no hold-out in this quick utility)
    if y_prob is not None:
        RocCurveDisplay.from_predictions(y, y_prob)
        plt.tight_layout()
        plt.savefig(fig_dir / f"early_roc_G{upto_grade}.png")
        plt.close()

    # Per-group fairness metrics via Fairlearn (https://fairlearn.org/)
    if group_cols:
        for col in group_cols:
            if col not in X.columns:
                logger.warning("Column '%s' not in dataset. Skipping.", col)
                continue
            mf = MetricFrame(
                metrics={
                    "selection_rate": selection_rate,
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate,
                },
                y_true=y,
                y_pred=y_pred,
                sensitive_features=X[col],
            )
            fairness_df = mf.by_group.reset_index().rename(
                columns={"index": col}
            )
            fairness_df.to_csv(report_dir / f"fairness_{col}.csv", index=False)
            logger.info(
                "Fairness metrics for '%s':\n%s",
                col,
                fairness_df.to_string(
                    index=False, float_format=lambda x: f"{x:.3f}"
                ),
            )

    # Feature importance (permutation)
    result = permutation_importance(pipeline, X, y, n_repeats=10, random_state=42)
    imp = (
        pd.DataFrame(
            {"feature": X.columns, "importance": result.importances_mean}
        ).sort_values("importance", ascending=False)
    )
    imp.to_csv(report_dir / f"early_feature_importance_G{upto_grade}.csv", index=False)
    sns.barplot(data=imp.head(20), x="importance", y="feature")
    plt.tight_layout()
    plt.savefig(fig_dir / f"early_feature_importance_G{upto_grade}.png")
    plt.close()
    return imp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a classifier using early grade data"
    )
    parser.add_argument("--csv-path", default="student-mat.csv")
    parser.add_argument("--upto-grade", type=int, default=1)
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--pass-threshold", type=int, default=10)
    parser.add_argument(
        "--group-cols",
        nargs="+",
        help="Columns for per-group fairness metrics",
    )
    args = parser.parse_args()
    train_early(
        csv_path=args.csv_path,
        upto_grade=args.upto_grade,
        model_type=args.model_type,
        pass_threshold=args.pass_threshold,
        group_cols=args.group_cols,
    )
