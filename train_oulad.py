#!/usr/bin/env python3
"""
Train ML/DL models on OULAD dataset and prepare for transfer learning to UCI dataset.
Enhanced with advanced PyTorch-based deep learning models.
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
import torch

sys.path.append(str(Path(__file__).resolve().parent / "src"))
try:
    from logging_config import setup_logging
except ImportError:
    def setup_logging():
        logging.basicConfig(level=logging.INFO)

# Import our advanced deep learning module
try:
    from src.oulad.deep_learning import train_deep_learning_models, create_ensemble_model
    from src.oulad.advanced_deep_learning import train_advanced_deep_learning_models, create_advanced_ensemble
    from src.oulad.optimized_deep_learning import train_optimized_models, create_optimized_ensemble
    from src.oulad.final_deep_learning import train_final_optimized_models, create_final_ensemble
    DEEP_LEARNING_AVAILABLE = True
    ADVANCED_DL_AVAILABLE = True
    OPTIMIZED_DL_AVAILABLE = True
    FINAL_DL_AVAILABLE = True
except ImportError as e:
    print(f"Deep learning module not available: {e}")
    DEEP_LEARNING_AVAILABLE = False
    ADVANCED_DL_AVAILABLE = False
    OPTIMIZED_DL_AVAILABLE = False
    FINAL_DL_AVAILABLE = False

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
        
        # Handle PyTorch models differently
        if hasattr(model_data, 'get') and 'model' in model_data and hasattr(model_data['model'], 'state_dict'):
            # This is a PyTorch model - save state dict separately
            torch_model_path = output_dir / f"oulad_{model_name}_model.pt"
            torch.save(model_data['model'].state_dict(), torch_model_path)
            
            # Save everything else (scaler, config, etc.) with joblib
            model_data_copy = model_data.copy()
            model_data_copy['model'] = None  # Remove model for joblib saving
            joblib.dump(model_data_copy, model_path)
            
            logger.info(f"Saved {model_name} PyTorch model to {torch_model_path}")
            logger.info(f"Saved {model_name} metadata to {model_path}")
        else:
            # Traditional sklearn model
            joblib.dump(model_data, model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
    
    return output_dir


def print_results(results):
    """Print model performance results."""
    logger.info("\n" + "="*60)
    logger.info("OULAD MODEL TRAINING RESULTS")
    logger.info("="*60)
    
    # Separate model types
    traditional_models = {}
    deep_models = {}
    optimized_models = {}
    final_models = {}
    advanced_models = {}
    
    for model_name, metrics in results.items():
        if model_name in ['logistic', 'random_forest', 'mlp']:
            traditional_models[model_name] = metrics
        elif 'final' in model_name:
            final_models[model_name] = metrics
        elif 'optimized' in model_name:
            optimized_models[model_name] = metrics
        elif model_name in ['attention_tabular', 'tabnet_like', 'advanced_ensemble']:
            advanced_models[model_name] = metrics
        else:
            deep_models[model_name] = metrics
    
    # Print traditional models
    if traditional_models:
        logger.info("\nTRADITIONAL MACHINE LEARNING MODELS:")
        logger.info("-" * 40)
        for model_name, metrics in traditional_models.items():
            logger.info(f"\n{model_name.upper()} RESULTS:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Print final optimized deep learning models
    if final_models:
        logger.info("\n\nFINAL OPTIMIZED DEEP LEARNING MODELS:")
        logger.info("-" * 40)
        for model_name, metrics in final_models.items():
            logger.info(f"\n{model_name.upper()} RESULTS:")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
            if 'f1_score' in metrics:
                logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
            
            # Print validation metrics if available
            if 'val_accuracy' in metrics:
                logger.info(f"Best Val Accuracy: {metrics['val_accuracy']:.4f}")
            if 'val_auc' in metrics:
                logger.info(f"Best Val AUC: {metrics['val_auc']:.4f}")
    
    # Print optimized deep learning models
    if optimized_models:
        logger.info("\n\nOPTIMIZED DEEP LEARNING MODELS:")
        logger.info("-" * 40)
        for model_name, metrics in optimized_models.items():
            logger.info(f"\n{model_name.upper()} RESULTS:")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
            if 'f1_score' in metrics:
                logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
            if 'balanced_accuracy' in metrics:
                logger.info(f"Test Balanced Acc: {metrics['balanced_accuracy']:.4f}")
            
            # Print CV results if available
            if 'cv_val_acc_mean' in metrics:
                logger.info(f"CV Val Accuracy: {metrics['cv_val_acc_mean']:.4f}±{metrics['cv_val_acc_std']:.4f}")
            if 'cv_val_auc_mean' in metrics:
                logger.info(f"CV Val AUC: {metrics['cv_val_auc_mean']:.4f}±{metrics['cv_val_auc_std']:.4f}")
    
    # Print regular deep learning models
    if deep_models:
        logger.info("\n\nDEEP LEARNING MODELS:")
        logger.info("-" * 40)
        for model_name, metrics in deep_models.items():
            logger.info(f"\n{model_name.upper()} RESULTS:")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Print validation metrics if available
            if 'val_accuracy' in metrics:
                logger.info(f"Best Val Accuracy: {metrics['val_accuracy']:.4f}")
            if 'val_auc' in metrics:
                logger.info(f"Best Val AUC: {metrics['val_auc']:.4f}")
    
    # Print advanced models
    if advanced_models:
        logger.info("\n\nADVANCED DEEP LEARNING MODELS:")
        logger.info("-" * 40)
        for model_name, metrics in advanced_models.items():
            logger.info(f"\n{model_name.upper()} RESULTS:")
            logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
            
            if 'val_accuracy' in metrics:
                logger.info(f"Best Val Accuracy: {metrics['val_accuracy']:.4f}")
            if 'val_auc' in metrics:
                logger.info(f"Best Val AUC: {metrics['val_auc']:.4f}")
    
    # Print comprehensive summary
    logger.info("\n\nCOMPREHENSIVE MODEL PERFORMANCE SUMMARY:")
    logger.info("-" * 50)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    logger.info(f"{'Model':<25} {'Accuracy':<10} {'ROC AUC':<10} {'F1 Score':<10}")
    logger.info("-" * 55)
    for model_name, metrics in sorted_results:
        f1_score = metrics.get('f1_score', metrics.get('roc_auc', 0))
        logger.info(f"{model_name:<25} {metrics['accuracy']:<10.4f} {metrics['roc_auc']:<10.4f} {f1_score:<10.4f}")
    
    # Highlight best performing model
    best_model, best_metrics = sorted_results[0]
    logger.info(f"\n🏆 BEST PERFORMING MODEL: {best_model.upper()}")
    logger.info(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"   ROC AUC: {best_metrics['roc_auc']:.4f}")
    
    # Performance improvement analysis
    if len(sorted_results) > 1:
        improvement = best_metrics['accuracy'] - sorted_results[-1][1]['accuracy']
        logger.info(f"   Improvement over worst: +{improvement:.4f} ({improvement*100:.2f}%)")
    
    # Check if deep learning improved over traditional methods
    traditional_best = max([metrics['accuracy'] for name, metrics in traditional_models.items()]) if traditional_models else 0
    if best_metrics['accuracy'] > traditional_best and best_model not in traditional_models:
        improvement_over_traditional = best_metrics['accuracy'] - traditional_best
        logger.info(f"   Deep learning improvement: +{improvement_over_traditional:.4f} ({improvement_over_traditional*100:.2f}%)")


def main():
    """Main training pipeline."""
    setup_logging()
    
    # Prepare data
    X, y, feature_cols, encoders = prepare_oulad_data()
    
    # Train traditional models
    models, results, train_test_data = train_models(X, y)
    X_train, X_test, y_train, y_test = train_test_data
    
    # Train advanced deep learning models if available
    if DEEP_LEARNING_AVAILABLE:
        logger.info("\n" + "="*60)
        logger.info("TRAINING ADVANCED DEEP LEARNING MODELS")
        logger.info("="*60)
        
        try:
            dl_models, dl_results = train_deep_learning_models(
                X_train.values, y_train.values, X_test.values, y_test.values
            )
            
            # Create ensemble
            ensemble_results = create_ensemble_model(dl_models, X_test.values, y_test.values)
            
            # Add to main results
            results.update(dl_results)
            results['deep_ensemble'] = ensemble_results
            
            # Add to models dict
            models.update(dl_models)
            
            logger.info("Deep learning models trained successfully!")
            
        except Exception as e:
            logger.error(f"Error training deep learning models: {e}")
            logger.info("Continuing with traditional models only...")
    
    # Train final optimized models for best results
    if FINAL_DL_AVAILABLE:
        logger.info("\n" + "="*60)
        logger.info("TRAINING FINAL OPTIMIZED DEEP LEARNING MODELS")
        logger.info("="*60)
        
        try:
            final_models, final_results = train_final_optimized_models(
                X_train.values, y_train.values, X_test.values, y_test.values
            )
            
            # Create final ensemble
            final_ensemble_results = create_final_ensemble(
                final_models, X_test.values, y_test.values
            )
            
            # Add to main results
            results.update(final_results)
            results['final_ensemble'] = final_ensemble_results
            
            # Add to models dict
            models.update(final_models)
            
            logger.info("Final optimized deep learning models trained successfully!")
            
        except Exception as e:
            logger.error(f"Error training final deep learning models: {e}")
            logger.info("Continuing without final models...")
    
    # Train optimized models for better results
    if OPTIMIZED_DL_AVAILABLE and not FINAL_DL_AVAILABLE:
        logger.info("\n" + "="*60)
        logger.info("TRAINING OPTIMIZED DEEP LEARNING MODELS")
        logger.info("="*60)
        
        try:
            opt_models, opt_results = train_optimized_models(
                X_train.values, y_train.values, X_test.values, y_test.values
            )
            
            # Create optimized ensemble
            opt_ensemble_results = create_optimized_ensemble(
                opt_models, X_test.values, y_test.values
            )
            
            # Add to main results
            results.update(opt_results)
            results['optimized_ensemble'] = opt_ensemble_results
            
            # Add to models dict
            models.update(opt_models)
            
            logger.info("Optimized deep learning models trained successfully!")
            
        except Exception as e:
            logger.error(f"Error training optimized deep learning models: {e}")
            logger.info("Continuing without optimized models...")
    
    if not DEEP_LEARNING_AVAILABLE:
        logger.info("Deep learning models not available. Using traditional models only.")
    
    # Save models
    model_dir = save_models(models)
    
    # Save metadata
    metadata = {
        'feature_columns': feature_cols,
        'encoders': encoders,
        'target_column': 'label_pass',
        'dataset_shape': X.shape,
        'class_distribution': y.value_counts().to_dict(),
        'deep_learning_available': DEEP_LEARNING_AVAILABLE,
        'advanced_dl_available': ADVANCED_DL_AVAILABLE,
        'optimized_dl_available': OPTIMIZED_DL_AVAILABLE,
        'final_dl_available': FINAL_DL_AVAILABLE
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