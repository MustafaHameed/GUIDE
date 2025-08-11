from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

from .data import load_data
from .preprocessing import build_pipeline


def main(csv_path: str = 'student-mat.csv'):
    X, y = load_data(csv_path)
    model = build_pipeline(X)

    # Hold-out evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Hold-out classification report:')
    print(classification_report(y_test, y_pred))

    # Cross-validation for robustness
    cv_model = build_pipeline(X)
    cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring='f1')
    print(f'5-fold CV F1-score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')


if __name__ == '__main__':
    main()
