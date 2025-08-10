import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


def load_data(csv_path: str):
    """Load dataset and create binary target indicating pass/fail."""
    df = pd.read_csv(csv_path)
    df['pass'] = (df['G3'] >= 10).astype(int)
    X = df.drop(columns=['G3', 'pass'])
    y = df['pass']
    return X, y


def build_pipeline(X):
    """Build preprocessing and modeling pipeline."""
    numeric_features = X.select_dtypes(include='number').columns
    categorical_features = X.select_dtypes(exclude='number').columns

    preprocess = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    clf = LogisticRegression(max_iter=1000)
    model = Pipeline(steps=[('preprocess', preprocess), ('model', clf)])
    return model


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
