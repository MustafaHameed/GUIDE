#!/usr/bin/env python3
"""
Train ML/DL models on OULAD dataset and prepare for transfer learning to UCI dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import logging
import sys

sys.path.append(str(Path(__file__).resolve() / "src"))
try:
    from logging_config import setup_logging
except ImportError:
    def setup_logging():
        logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def prepare_oulad_data(data_path: str = "data/oulad/processed/oulad_ml.csv"):
    """Load and prepare OULAD data for training."""
    logger.info(f"Loading OULAD data from {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    
    # Drop ID columns and non-predictive features
    id_cols = ['id_student', 'code_module', 'code_presentation']
    target_col = 'label_pass'
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in id_cols + [target_col, 'label_fail_or_withdraw']]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle categorical features
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Encode categorical features
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Handle missing values
    X = X.fillna(X.median())
    
    logger.info(f"Prepared features: {X.shape[1]} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols, encoders


def train_models(X, y, test_size=0.2, random_state=42):
    """Train multiple ML models on OULAD data."""
    logger.info("Training models on OULAD data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    # Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_params = {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [1000]
    }
    lr = GridSearchCV(LogisticRegression(random_state=random_state), lr_params, cv=5, scoring='roc_auc')
    lr.fit(X_train_scaled, y_train)
    
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    
    models['logistic'] = {
        'model': lr.best_estimator_,
        'scaler': scaler,
        'params': lr.best_params_
    }
    
    results['logistic'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_prob),
        'classification_report': classification_report(y_test, lr_pred)
    }
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=random_state), rf_params, cv=5, scoring='roc_auc')
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    
    models['random_forest'] = {
        'model': rf.best_estimator_,
        'scaler': None,  # RF doesn't need scaling
        'params': rf.best_params_
    }
    
    results['random_forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_prob),
        'classification_report': classification_report(y_test, rf_pred)
    }
    
    # Neural Network (MLP)
    logger.info("Training Neural Network (MLP)...")
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'alpha': [0.0001, 0.001],
        'max_iter': [500]
    }
    mlp = GridSearchCV(MLPClassifier(random_state=random_state), mlp_params, cv=3, scoring='roc_auc')
    mlp.fit(X_train_scaled, y_train)
    
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    
    models['mlp'] = {
        'model': mlp.best_estimator_,
        'scaler': scaler,  # Use same scaler as logistic
        'params': mlp.best_params_
    }
    
    results['mlp'] = {
        'accuracy': accuracy_score(y_test, mlp_pred),
        'roc_auc': roc_auc_score(y_test, mlp_prob),
        'classification_report': classification_report(y_test, mlp_pred)
    }
    
    return models, results, (X_train, X_test, y_train, y_test)


def save_models(models, output_dir="models/oulad"):
    """Save trained models to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_data in models.items():
        model_path = output_dir / f"oulad_{model_name}.pkl"
        joblib.dump(model_data, model_path)
        logger.info(f"Saved {model_name} model to {model_path}")
    
    return output_dir


def print_results(results):
    """Print model performance results."""
    logger.info("\n" + "="*60)
    logger.info("OULAD MODEL TRAINING RESULTS")
    logger.info("="*60)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name.upper()} RESULTS:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Classification Report:\n{metrics['classification_report']}")


def main():
    """Main training pipeline."""
    setup_logging()
    
    # Prepare data
    X, y, feature_cols, encoders = prepare_oulad_data()
    
    # Train models
    models, results, train_test_data = train_models(X, y)
    
    # Save models
    model_dir = save_models(models)
    
    # Save metadata
    metadata = {
        'feature_columns': feature_cols,
        'encoders': encoders,
        'target_column': 'label_pass',
        'dataset_shape': X.shape,
        'class_distribution': y.value_counts().to_dict()
    }
    
    metadata_path = model_dir / "oulad_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Print results
    print_results(results)
    
    logger.info(f"\nOULAD model training completed!")
    logger.info(f"Models saved to: {model_dir}")
    
    return models, results, metadata


if __name__ == "__main__":
    main()