#!/usr/bin/env python3
"""
Transfer Learning from OULAD to UCI - Simplified Feature-based Approach

This script implements transfer learning by training new models on shared features
between OULAD and UCI datasets, focusing on demographic and behavioral patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_oulad_transfer_data():
    """Load OULAD data and extract features suitable for transfer learning."""
    logger.info("Loading OULAD data for transfer learning...")
    
    df = pd.read_csv("data/oulad/processed/oulad_ml.csv")
    logger.info(f"Loaded OULAD dataset with shape: {df.shape}")
    
    # Create transfer learning features
    transfer_features = pd.DataFrame({
        'gender': df['sex'],
        'age_group': df['age_band'],
        'education_level': df['highest_education'],
        'socioeconomic_status': df['imd_band'],
        'prior_attempts': df['num_of_prev_attempts'],
        'study_load': df['studied_credits'],
        'target': df['label_pass']
    })
    
    # Encode categorical variables
    categorical_cols = ['gender', 'age_group', 'education_level', 'socioeconomic_status']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        transfer_features[col] = le.fit_transform(transfer_features[col].astype(str))
        encoders[col] = le
    
    # Handle missing values
    transfer_features = transfer_features.fillna(transfer_features.median())
    
    logger.info(f"OULAD transfer features shape: {transfer_features.shape}")
    logger.info(f"OULAD target distribution: {transfer_features['target'].value_counts().to_dict()}")
    
    return transfer_features, encoders


def load_and_prepare_uci_transfer_data():
    """Load UCI data and create corresponding transfer features."""
    logger.info("Loading UCI data for transfer learning...")
    
    df = pd.read_csv("student-mat.csv")
    logger.info(f"Loaded UCI dataset with shape: {df.shape}")
    
    # Create binary target (pass/fail)
    df['target'] = (df['G3'] >= 10).astype(int)
    
    # Map UCI features to transfer feature space
    transfer_features = pd.DataFrame({
        'gender': df['sex'].map({'F': 'F', 'M': 'M'}),
        'age_group': df['age'].apply(lambda x: '15-17' if x <= 17 else ('18-20' if x <= 20 else '21+')),
        'education_level': df['Medu'].apply(lambda x: f'level_{x}'),  # Mother's education
        'socioeconomic_status': df['famrel'].apply(lambda x: f'socio_{x}'),  # Family relationships as proxy
        'prior_attempts': df['failures'],  # Number of past class failures
        'study_load': df['studytime'],  # Weekly study time
        'target': df['target']
    })
    
    logger.info(f"UCI transfer features shape: {transfer_features.shape}")
    logger.info(f"UCI target distribution: {transfer_features['target'].value_counts().to_dict()}")
    
    return transfer_features


def create_consistent_feature_space(oulad_features, uci_features, oulad_encoders):
    """Create consistent feature space between OULAD and UCI datasets."""
    logger.info("Creating consistent feature space...")
    
    # Process UCI features to match OULAD encoding
    uci_processed = uci_features.copy()
    
    categorical_cols = ['gender', 'age_group', 'education_level', 'socioeconomic_status']
    
    for col in categorical_cols:
        # Get unique values from both datasets
        oulad_values = set(oulad_features[col].unique())
        uci_values = set(uci_features[col].astype(str).unique())
        
        # Create mapping for UCI values to OULAD space
        if col == 'gender':
            # Direct mapping for gender
            mapping = {'F': 0, 'M': 1}
        else:
            # For other categorical features, map unknown values to most common OULAD value
            most_common_oulad = oulad_features[col].mode()[0]
            mapping = {val: most_common_oulad for val in uci_values}
            
            # Map known values correctly
            oulad_str_values = [str(val) for val in oulad_encoders[col].classes_]
            for i, oulad_val in enumerate(oulad_str_values):
                if oulad_val in uci_values:
                    mapping[oulad_val] = i
        
        # Apply mapping
        uci_processed[col] = uci_features[col].astype(str).map(mapping)
        uci_processed[col] = uci_processed[col].fillna(oulad_features[col].mode()[0])
    
    # Normalize numeric features to similar scales
    numeric_cols = ['prior_attempts', 'study_load']
    
    for col in numeric_cols:
        # Simple min-max normalization to 0-1
        oulad_min, oulad_max = oulad_features[col].min(), oulad_features[col].max()
        uci_min, uci_max = uci_features[col].min(), uci_features[col].max()
        
        # Normalize UCI to OULAD scale
        if uci_max > uci_min:
            uci_normalized = (uci_features[col] - uci_min) / (uci_max - uci_min)
            uci_processed[col] = uci_normalized * (oulad_max - oulad_min) + oulad_min
        else:
            uci_processed[col] = oulad_features[col].mean()
    
    return oulad_features, uci_processed


def train_transfer_models(oulad_features, test_size=0.2):
    """Train models on OULAD data for transfer learning."""
    logger.info("Training transfer learning models on OULAD data...")
    
    feature_cols = ['gender', 'age_group', 'education_level', 'socioeconomic_status', 'prior_attempts', 'study_load']
    
    X = oulad_features[feature_cols]
    y = oulad_features['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    oulad_results = {}
    
    # Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    
    models['logistic'] = {'model': lr, 'scaler': scaler}
    oulad_results['logistic'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'roc_auc': roc_auc_score(y_test, lr_prob)
    }
    
    # Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    
    models['random_forest'] = {'model': rf, 'scaler': None}
    oulad_results['random_forest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'roc_auc': roc_auc_score(y_test, rf_prob)
    }
    
    # Neural Network
    logger.info("Training Neural Network...")
    mlp = MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=1000)
    mlp.fit(X_train_scaled, y_train)
    
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    
    models['mlp'] = {'model': mlp, 'scaler': scaler}
    oulad_results['mlp'] = {
        'accuracy': accuracy_score(y_test, mlp_pred),
        'roc_auc': roc_auc_score(y_test, mlp_prob)
    }
    
    # Print OULAD results
    logger.info("\nOULAD Model Performance:")
    for model_name, metrics in oulad_results.items():
        logger.info(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, ROC AUC = {metrics['roc_auc']:.4f}")
    
    return models, oulad_results, feature_cols


def evaluate_transfer_performance(models, uci_features, feature_cols):
    """Evaluate transfer learning performance on UCI dataset."""
    logger.info("Evaluating transfer learning performance on UCI data...")
    
    X_uci = uci_features[feature_cols]
    y_uci = uci_features['target']
    
    transfer_results = {}
    
    logger.info("\n" + "="*60)
    logger.info("TRANSFER LEARNING RESULTS: OULAD → UCI")
    logger.info("="*60)
    
    for model_name, model_data in models.items():
        logger.info(f"\nEvaluating {model_name.upper()} on UCI data...")
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Prepare UCI features
        if scaler is not None:
            X_uci_scaled = scaler.transform(X_uci)
        else:
            X_uci_scaled = X_uci
        
        # Make predictions
        y_pred = model.predict(X_uci_scaled)
        y_prob = model.predict_proba(X_uci_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_uci, y_pred)
        roc_auc = roc_auc_score(y_uci, y_prob)
        
        transfer_results[model_name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_uci, y_pred)
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        logger.info(f"Classification Report:\n{classification_report(y_uci, y_pred)}")
    
    return transfer_results


def run_complete_transfer_learning_pipeline():
    """Run the complete transfer learning pipeline from OULAD to UCI."""
    logger.info("Starting complete transfer learning pipeline...")
    
    # Load and prepare datasets
    oulad_features, oulad_encoders = load_and_prepare_oulad_transfer_data()
    uci_features = load_and_prepare_uci_transfer_data()
    
    # Create consistent feature space
    oulad_features, uci_features = create_consistent_feature_space(
        oulad_features, uci_features, oulad_encoders
    )
    
    # Train models on OULAD
    models, oulad_results, feature_cols = train_transfer_models(oulad_features)
    
    # Evaluate on UCI
    transfer_results = evaluate_transfer_performance(models, uci_features, feature_cols)
    
    # Calculate baseline
    baseline_accuracy = max(uci_features['target'].value_counts()) / len(uci_features)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRANSFER LEARNING SUMMARY")
    logger.info("="*60)
    logger.info(f"UCI Baseline (majority class): {baseline_accuracy:.4f}")
    
    improvements = {}
    for model_name, metrics in transfer_results.items():
        improvement = metrics['accuracy'] - baseline_accuracy
        improvements[model_name] = improvement
        logger.info(f"{model_name}: {metrics['accuracy']:.4f} (Δ = {improvement:+.4f})")
    
    # Save results
    results_dir = Path("reports/transfer_learning")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive report
    report = f"""# Transfer Learning Report: OULAD → UCI

## Dataset Information
- **OULAD Dataset**: {oulad_features.shape[0]} samples, {len(feature_cols)} features
- **UCI Dataset**: {uci_features.shape[0]} samples, {len(feature_cols)} features
- **Shared Features**: {', '.join(feature_cols)}

## Model Performance

### OULAD (Source Domain) Performance
"""
    
    for model_name, metrics in oulad_results.items():
        report += f"- **{model_name}**: Accuracy = {metrics['accuracy']:.4f}, ROC AUC = {metrics['roc_auc']:.4f}\n"
    
    report += f"""
### UCI (Target Domain) Transfer Performance
- **Baseline (Majority Class)**: {baseline_accuracy:.4f}

"""
    
    for model_name, metrics in transfer_results.items():
        improvement = improvements[model_name]
        report += f"- **{model_name}**: Accuracy = {metrics['accuracy']:.4f} (Δ = {improvement:+.4f}), ROC AUC = {metrics['roc_auc']:.4f}\n"
    
    report += f"""
## Key Findings
1. **Best Transfer Model**: {max(improvements, key=improvements.get)} (improvement: {max(improvements.values()):+.4f})
2. **Transfer Success**: {'Yes' if max(improvements.values()) > 0 else 'No'} (models outperform baseline)
3. **Domain Gap**: Transfer learning effectiveness varies by model type

## Feature Mapping
The transfer learning uses these shared conceptual features:
- **Gender**: Direct mapping between datasets
- **Age Group**: Age ranges mapped to categories  
- **Education Level**: Educational background indicators
- **Socioeconomic Status**: Family and social context proxies
- **Prior Attempts**: Academic history indicators
- **Study Load**: Academic engagement measures

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_path = results_dir / "transfer_learning_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"\nDetailed report saved to: {report_path}")
    
    return {
        'oulad_results': oulad_results,
        'transfer_results': transfer_results,
        'baseline_accuracy': baseline_accuracy,
        'improvements': improvements
    }


if __name__ == "__main__":
    results = run_complete_transfer_learning_pipeline()