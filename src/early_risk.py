"""Train a classifier using early grade data for risk assessment.

Beyond exporting risk probabilities, the script computes feature importances
using SHAP when available and falls back to permutation importance otherwise.
Ranked importances are written to ``reports/`` and a corresponding plot is
saved in ``figures/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
)

from .data import load_early_data
from .preprocessing import build_pipeline


def main(
    csv_path: str = "student-mat.csv",
    upto_grade: int = 1,
    model_type: str = "logistic",
    estimators: list[str] | None = None,
    final_estimator: str = "logistic",
    base_estimator: str = "decision_tree",
    group_cols: list[str] | None = None,
):
    """Train model on truncated data and export risk probabilities."""

    X, y = load_early_data(csv_path, upto_grade=upto_grade)
    model_params: dict | None = None
    if model_type == "stacking":
        model_params = {
            "estimators": estimators or ["logistic", "random_forest"],
            "final_estimator": final_estimator,
        }
    elif model_type == "bagging":
        model_params = {"base_estimator": base_estimator}
    pipeline = build_pipeline(X, model_type=model_type, model_params=model_params)

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    mean_scores = {m: scores[f"test_{m}"].mean() for m in scoring}
    print("Cross-validation metrics:")
    for m, val in mean_scores.items():
        print(f"{m}: {val:.3f}")
    pd.DataFrame([mean_scores]).to_csv(
        report_dir / f"early_cv_metrics_G{upto_grade}.csv", index=False
    )

    y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(y, y_pred))

    report = classification_report(y, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        report_dir / f"early_classification_report_G{upto_grade}.csv", index=True
    )

    RocCurveDisplay.from_predictions(y, y_prob)
    plt.tight_layout()
    plt.savefig(fig_dir / f"early_roc_curve_G{upto_grade}.png")
    plt.close()

    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)[:, 1]
    pd.DataFrame({"risk_probability": probs}).to_csv(
        report_dir / f"early_risk_probabilities_G{upto_grade}.csv", index=False
    )

    if group_cols:
        df_pred = X.copy()
        df_pred["y_true"] = y
        df_pred["y_pred"] = y_pred
        for col in group_cols:
            if col not in df_pred.columns:
                print(f"Column '{col}' not in dataset. Skipping.")
                continue
            for group_val, grp in df_pred.groupby(col):
                if grp["y_true"].nunique() < 2:
                    print(
                        f"Skipping group {col}={group_val} due to single class in y_true."
                    )
                    continue
                grp_report = classification_report(
                    grp["y_true"], grp["y_pred"], output_dict=True
                )
                pd.DataFrame(grp_report).transpose().to_csv(
                    report_dir
                    / f"early_classification_report_G{upto_grade}_{col}_{group_val}.csv",
                    index=True,
                )

    fi_csv = report_dir / f"early_feature_importance_G{upto_grade}.csv"
    fi_fig = fig_dir / f"early_feature_importance_G{upto_grade}.png"
    try:
        import shap

        explainer = shap.Explainer(pipeline, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(fi_fig)
        plt.close()
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": np.abs(shap_values.values).mean(axis=0),
                }
            ).sort_values("importance", ascending=False)
        )
        importance_df.to_csv(fi_csv, index=False)
    except Exception:
        result = permutation_importance(
            pipeline, X, y, n_repeats=10, random_state=42
        )
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": result.importances_mean,
                }
            ).sort_values("importance", ascending=False)
        )
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.tight_layout()
        plt.savefig(fi_fig)
        plt.close()
        importance_df.to_csv(fi_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early risk model training")
    parser.add_argument("--csv_path", default="student-mat.csv")
    parser.add_argument("--upto_grade", type=int, default=1)
    parser.add_argument("--model_type", default="logistic")
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--final_estimator", default="logistic")
    parser.add_argument("--base_estimator", default="decision_tree")
    parser.add_argument(
        "--group-cols",
        nargs="*",
        default=None,
        help="Columns to compute group-level metrics for",
    )
    args = parser.parse_args()
    main(**vars(args))