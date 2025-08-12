from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)
from sklearn.inspection import permutation_importance
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .data import load_data
from .preprocessing import build_pipeline


PARAM_GRIDS: dict[str, dict[str, dict]] = {
    "logistic": {
        "default": {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        }
    },
    "random_forest": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
        }
    },
    "gradient_boosting": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
        }
    },
    "svm": {
        "default": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        }
    },
    "knn": {
        "default": {
            "model__n_neighbors": [5, 10, 15],
            "model__weights": ["uniform", "distance"],
        }
    },
    "naive_bayes": {
        "default": {
            "model__var_smoothing": [1e-09, 1e-08, 1e-07],
        }
    },
    "extra_trees": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
        }
    },
    "xgboost": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 6],
            "model__learning_rate": [0.05, 0.1],
        }
    },
    "lightgbm": {
        "default": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [-1, 5, 10],
            "model__learning_rate": [0.05, 0.1],
        }
    },
}
def main(
    csv_path: str = "student-mat.csv",
    group_cols: list[str] | None = None,
    model_type: str = "logistic",
    param_grid: str = "none",
):
    """Train model and generate evaluation artifacts.

    Parameters
    ----------
    csv_path : str, default 'student-mat.csv'
        Path to the input CSV file.
    group_cols : list[str] | None, optional
        Demographic columns to compute group-level metrics for. If ``None``,
        only overall metrics are produced.
    model_type : str, default "logistic"
        Type of model to train.
    param_grid : str, default "none"
        Preset name for the hyperparameter grid. Use "none" to skip
        hyperparameter tuning.
    """
    X, y = load_data(csv_path)
    pipeline = build_pipeline(X, model_type=model_type)

    # Prepare output directories
    fig_dir = Path('figures')
    fig_dir.mkdir(exist_ok=True)
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)

    # Hold-out evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    grid = PARAM_GRIDS.get(model_type, {}).get(param_grid)

    best_params: dict | None = None
    best_score: float | None = None

    if grid:
        search = GridSearchCV(pipeline, grid, cv=5, scoring="f1")
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
        best_score = search.best_score_
        print(f"Best params from search: {best_params} (score={best_score:.3f})")
    else:
        model = pipeline
        model.fit(X_train, y_train)
        best_params = model.named_steps["model"].get_params()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Hold-out classification report:")
    print(classification_report(y_test, y_pred))

    # Export classification report as a table
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        report_dir / 'classification_report.csv', index=True
    )

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(fig_dir / 'confusion_matrix.png')
    plt.close()

    # ROC curve visualization
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.tight_layout()
    plt.savefig(fig_dir / 'roc_curve.png')
    plt.close()

    # Export best parameters and search metrics
    best_params_df = pd.DataFrame([best_params or {}])
    best_params_df.insert(0, "best_score", best_score)
    best_params_df.to_csv(report_dir / "best_params.csv", index=False)

    # Per-group evaluations
    if group_cols:
        overall_positive_rate = (y_pred == 1).mean()
        for col in group_cols:
            if col not in X_test.columns:
                print(f"Column '{col}' not in dataset. Skipping.")
                continue
            fairness_records: list[dict[str, float | str]] = []
            for group_value in X_test[col].unique():
                mask = X_test[col] == group_value
                y_true_g = y_test[mask]
                y_pred_g = y_pred[mask]
                y_prob_g = y_prob[mask]
                if y_true_g.nunique() < 2:
                    # Skip groups with a single class; metrics not meaningful
                    print(
                        f"Skipping group {col}={group_value} due to single class in y_true."
                    )
                    continue

                # Classification report
                grp_report = classification_report(
                    y_true_g, y_pred_g, output_dict=True
                )
                pd.DataFrame(grp_report).transpose().to_csv(
                    report_dir
                    / f"classification_report_{col}_{group_value}.csv",
                    index=True,
                )

                # Confusion matrix
                cm_g = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1])
                plt.figure(figsize=(4, 4))
                sns.heatmap(cm_g, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(
                    fig_dir / f"confusion_matrix_{col}_{group_value}.png"
                )
                plt.close()

                # ROC curve
                RocCurveDisplay.from_predictions(y_true_g, y_prob_g)
                plt.tight_layout()
                plt.savefig(fig_dir / f"roc_curve_{col}_{group_value}.png")
                plt.close()

                # Fairness metrics
                pos_rate = (y_pred_g == 1).mean()
                disparity = abs(pos_rate - overall_positive_rate)
                fairness_records.append(
                    {
                        col: group_value,
                        "positive_rate": pos_rate,
                        "disparity": disparity,
                    }
                )
if fairness_records:
            fairness_df = pd.DataFrame(fairness_records)
            fairness_path = report_dir / f"fairness_{col}.csv"
            fairness_df.to_csv(fairness_path, index=False)
            print(f"Fairness metrics for '{col}':")
            print(
                fairness_df.to_string(
                    index=False, float_format=lambda x: f"{x:.3f}"
                )
            )

    # Feature importance
    fi_csv = report_dir / "feature_importance.csv"
    fi_fig = fig_dir / "feature_importance.png"
    try:
        import shap

        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, show=False)
        plt.tight_layout()
        plt.savefig(fi_fig)
        plt.close()
        importance = np.abs(shap_values.values).mean(axis=0)
        pd.DataFrame({
            "feature": X_train.columns,
            "importance": importance,
        }).sort_values("importance", ascending=False).to_csv(fi_csv, index=False)
    except Exception:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42
        )
        importance_df = (
            pd.DataFrame({
                "feature": X_test.columns,
                "importance": result.importances_mean,
            })
            .sort_values("importance", ascending=False)
        )
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.tight_layout()
        plt.savefig(fi_fig)
        plt.close()
        importance_df.to_csv(fi_csv, index=False)
    # Cross-validation for robustness
    cv_model = build_pipeline(X, model_type=model_type, model_params=best_params)
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='f1')
    print(f'5-fold CV F1-score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

    # Export cross-validation scores
    cv_df = pd.DataFrame({'fold': range(1, len(cv_scores) + 1), 'f1_score': cv_scores})
    cv_df.to_csv(report_dir / 'cv_scores.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model with fairness evaluation')
    parser.add_argument('--csv-path', default='student-mat.csv')
    parser.add_argument(
        '--group-cols',
        nargs='*',
        default=None,
        help='Demographic columns to evaluate',
    )
    parser.add_argument(
        '--model-type',
        choices=list(PARAM_GRIDS.keys()),
        default='logistic',
        help='Type of model to train',
    )
