GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json

from .data import load_data
from .preprocessing import build_pipeline
from .model import create_model


def main(csv_path: str = "student-mat.csv"):
    X, y = load_data(csv_path)
    pipeline = build_pipeline(X)

    param_grid = [
        {
            "model": [create_model("logistic")],
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        {
            "model": [create_model("random_forest")],
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
        },
        {
            "model": [create_model("gradient_boosting")],
            "model__learning_rate": [0.01, 0.1],
            "model__n_estimators": [100, 200],
        },
    ]

    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")

    # Prepare output directories
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    # Hold-out evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    print("Best parameters:")
    print(search.best_params_)
    print("Hold-out classification report:")
    print(classification_report(y_test, y_pred))

    # Export best parameters
    with open(report_dir / "best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)

    # Export classification report as a table
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        report_dir / "classification_report.csv", index=True
    )

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png")
    plt.close()

    # ROC curve visualization
    RocCurveDisplay.from_estimator(search.best_estimator_, X_test, y_test)
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png")
    plt.close()

    # Cross-validation for robustness of the best model
    cv_scores = cross_val_score(search.best_estimator_, X, y, cv=5, scoring="f1")
    print(f"5-fold CV F1-score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Export cross-validation scores and grid search results
    cv_df = pd.DataFrame({"fold": range(1, len(cv_scores) + 1), "f1_score": cv_scores})
    cv_df.to_csv(report_dir / "cv_scores.csv", index=False)
    pd.DataFrame(search.cv_results_).to_csv(report_dir / "grid_search_results.csv", index=False)


if __name__ == '__main__':
    main()
