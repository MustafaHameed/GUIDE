from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from .data import load_data
from .preprocessing import build_pipeline


def main(csv_path: str = 'student-mat.csv'):
    X, y = load_data(csv_path)
    model = build_pipeline(X)

    # Prepare output directories
    fig_dir = Path('figures')
    fig_dir.mkdir(exist_ok=True)
    report_dir = Path('reports')
    report_dir.mkdir(exist_ok=True)

    # Hold-out evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Hold-out classification report:')
    print(classification_report(y_test, y_pred))

    # Export classification report as a table
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        report_dir / 'classification_report.csv', index=True
    )

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(fig_dir / 'confusion_matrix.png')
    plt.close()

    # ROC curve visualization
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout()
    plt.savefig(fig_dir / 'roc_curve.png')
    plt.close()

    # Cross-validation for robustness
    cv_model = build_pipeline(X)
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='f1')
    print(f'5-fold CV F1-score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

    # Export cross-validation scores
    cv_df = pd.DataFrame({'fold': range(1, len(cv_scores) + 1), 'f1_score': cv_scores})
    cv_df.to_csv(report_dir / 'cv_scores.csv', index=False)


if __name__ == '__main__':
    main()