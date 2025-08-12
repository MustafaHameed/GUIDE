from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        for col in group_cols:
            if col not in X_test.columns:
                print(f"Column '{col}' not in dataset. Skipping.")
                continue
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
    parser.add_argument(
        '--param-grid',
        choices=['none', 'default'],
        default='none',
        help='Preset hyperparameter grid to use',
    )
    args = parser.parse_args()
    main(
        csv_path=args.csv_path,
        group_cols=args.group_cols,
        model_type=args.model_type,
        param_grid=args.param_grid,
    )

