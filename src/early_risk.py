"""Train a classifier using early grade data for risk assessment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import train_test_split

from .data import load_early_data
from .preprocessing import build_pipeline


def main(
    csv_path: str = "student-mat.csv",
    upto_grade: int = 1,
    model_type: str = "logistic",
    estimators: list[str] | None = None,
    final_estimator: str = "logistic",
    base_estimator: str = "decision_tree",
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        report_dir / f"early_classification_report_G{upto_grade}.csv", index=True
    )

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.tight_layout()
    plt.savefig(fig_dir / f"early_roc_curve_G{upto_grade}.png")
    plt.close()

    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)[:, 1]
    pd.DataFrame({"risk_probability": probs}).to_csv(
        report_dir / f"early_risk_probabilities_G{upto_grade}.csv", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early risk model training")
    parser.add_argument("--csv_path", default="student-mat.csv")
    parser.add_argument("--upto_grade", type=int, default=1)
    parser.add_argument("--model_type", default="logistic")
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument("--final_estimator", default="logistic")
    parser.add_argument("--base_estimator", default="decision_tree")
    args = parser.parse_args()
    main(**vars(args))
