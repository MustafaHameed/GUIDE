#!/usr/bin/env python3
"""
Transfer Learning from OULAD to UCI Student Performance Dataset

This script applies models trained on OULAD data to the UCI student performance dataset,
implementing transfer learning with feature mapping and domain adaptation techniques.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_feature_mapping():
    """Define mapping between OULAD and UCI features for transfer learning."""
    return {
        "shared_features": {
            # Demographics
            "sex": {
                "oulad_col": "sex",
                "uci_col": "sex", 
                "mapping": {"F": "F", "M": "M"}
            },
            # Age (convert to bands for compatibility)
            "age_band": {
                "oulad_col": "age_band",
                "uci_col": "age",
                "transform": "age_to_band"
            },
            # Socioeconomic proxy
            "ses_proxy": {
                "oulad_col": "imd_band",  # Index of Multiple Deprivation
                "uci_col": "Medu",  # Mother's education as SES proxy
                "transform": "ses_standardize"
            },
            # Academic performance proxy
            "academic_proxy": {
                "oulad_col": "studied_credits",  # Credits studied
                "uci_col": "G1",  # First period grade
                "transform": "normalize"
            },
            # Engagement proxy
            "engagement_proxy": {
                "oulad_col": "vle_total_clicks",  # VLE engagement
                "uci_col": "studytime",  # Weekly study time
                "transform": "normalize"
            }
        }
    }


def load_uci_data(uci_path="student-mat.csv"):
    """Load and prepare UCI student performance data."""
    logger.info(f"Loading UCI data from {uci_path}")
    
    df = pd.read_csv(uci_path)
    logger.info(f"Loaded UCI dataset with shape: {df.shape}")
    
    # Create binary pass/fail label (G3 >= 10)
    df['label_pass'] = (df['G3'] >= 10).astype(int)
    
    return df


def prepare_oulad_features_for_transfer(oulad_data_path="data/oulad/processed/oulad_ml.csv"):
    """Prepare OULAD features for transfer learning."""
    logger.info("Preparing OULAD features for transfer...")
    
    df = pd.read_csv(oulad_data_path)
    
    # Select relevant features for transfer
    transfer_features = {
        'sex': df['sex'],
        'age_band': df['age_band'], 
        'ses_proxy': df['imd_band'],
        'academic_proxy': df['studied_credits'],
        'engagement_proxy': df['vle_total_clicks'],
        'label_pass': df['label_pass']
    }
    
    oulad_transfer_df = pd.DataFrame(transfer_features)
    
    # Encode categorical features
    le_sex = LabelEncoder()
    oulad_transfer_df['sex'] = le_sex.fit_transform(oulad_transfer_df['sex'].astype(str))
    
    le_age = LabelEncoder()
    oulad_transfer_df['age_band'] = le_age.fit_transform(oulad_transfer_df['age_band'].astype(str))
    
    le_ses = LabelEncoder()
    oulad_transfer_df['ses_proxy'] = le_ses.fit_transform(oulad_transfer_df['ses_proxy'].astype(str))
    
    # Normalize numeric features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    oulad_transfer_df[['academic_proxy', 'engagement_proxy']] = scaler.fit_transform(
        oulad_transfer_df[['academic_proxy', 'engagement_proxy']].fillna(0)
    )
    
    return oulad_transfer_df, {'sex': le_sex, 'age_band': le_age, 'ses_proxy': le_ses}, scaler


def prepare_uci_features_for_transfer(uci_df, oulad_encoders, oulad_scaler):
    """Prepare UCI features to match OULAD feature space."""
    logger.info("Preparing UCI features for transfer...")
    
    # Map UCI features to OULAD feature space
    uci_transfer = pd.DataFrame()
    
    # Sex mapping
    uci_transfer['sex'] = uci_df['sex'].map({'F': 0, 'M': 1}).fillna(0)
    
    # Age to age band mapping
    age_mapping = {age: 0 if age <= 18 else (1 if age <= 22 else 2) for age in range(15, 25)}
    uci_transfer['age_band'] = uci_df['age'].map(age_mapping).fillna(0)
    
    # Mother's education as SES proxy
    medu_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 0: 0}  # Map to ordinal scale
    uci_transfer['ses_proxy'] = uci_df['Medu'].map(medu_mapping).fillna(0)
    
    # Academic performance proxy (G1 normalized)
    academic_proxy = uci_df['G1'].fillna(uci_df['G1'].mean()) / 20.0  # Normalize to 0-1
    
    # Engagement proxy (study time normalized)
    engagement_proxy = uci_df['studytime'].fillna(uci_df['studytime'].mean()) / 4.0  # Normalize to 0-1
    
    # Apply same scaling as OULAD
    academic_engagement = np.column_stack([academic_proxy, engagement_proxy])
    academic_engagement_scaled = oulad_scaler.transform(academic_engagement)
    
    uci_transfer['academic_proxy'] = academic_engagement_scaled[:, 0]
    uci_transfer['engagement_proxy'] = academic_engagement_scaled[:, 1]
    
    # Target variable
    uci_transfer['label_pass'] = uci_df['label_pass']
    
    return uci_transfer


def run_transfer_learning():
    """Run complete transfer learning pipeline from OULAD to UCI."""
    logger.info("Starting transfer learning from OULAD to UCI...")
    
    # Load trained OULAD models
    model_dir = Path("models/oulad")
    if not model_dir.exists():
        raise FileNotFoundError("OULAD models not found. Please train OULAD models first.")
    
    # Load models
    models = {}
    for model_name in ['logistic', 'random_forest', 'mlp']:
        model_path = model_dir / f"oulad_{model_name}.pkl"
        if model_path.exists():
            models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded {model_name} model")
    
    # Prepare OULAD features for reference
    oulad_transfer_df, oulad_encoders, oulad_scaler = prepare_oulad_features_for_transfer()
    logger.info(f"OULAD transfer features shape: {oulad_transfer_df.shape}")
    
    # Load and prepare UCI data
    uci_df = load_uci_data()
    uci_transfer_df = prepare_uci_features_for_transfer(uci_df, oulad_encoders, oulad_scaler)
    logger.info(f"UCI transfer features shape: {uci_transfer_df.shape}")
    
    # Extract features and targets
    feature_cols = ['sex', 'age_band', 'ses_proxy', 'academic_proxy', 'engagement_proxy']
    
    X_uci = uci_transfer_df[feature_cols]
    y_uci = uci_transfer_df['label_pass']
    
    # Apply each trained model to UCI data
    results = {}
    
    logger.info("\n" + "="*60)
    logger.info("TRANSFER LEARNING RESULTS: OULAD → UCI")
    logger.info("="*60)
    
    for model_name, model_data in models.items():
        logger.info(f"\nTesting {model_name.upper()} on UCI data...")
        
        model = model_data['model']
        scaler = model_data.get('scaler')
        
        # Apply scaling if needed
        if scaler is not None:
            X_uci_scaled = scaler.transform(X_uci)
        else:
            X_uci_scaled = X_uci
        
        # Make predictions
        try:
            y_pred = model.predict(X_uci_scaled)
            y_prob = model.predict_proba(X_uci_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_uci, y_pred)
            roc_auc = roc_auc_score(y_uci, y_prob) if y_prob is not None else None
            
            results[model_name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'classification_report': classification_report(y_uci, y_pred)
            }
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            if roc_auc:
                logger.info(f"ROC AUC: {roc_auc:.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_uci, y_pred)}")
            
        except Exception as e:
            logger.error(f"Error applying {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Compare with baseline (random prediction)
    baseline_accuracy = max(y_uci.value_counts()) / len(y_uci)
    logger.info(f"\nBaseline (majority class) accuracy: {baseline_accuracy:.4f}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRANSFER LEARNING SUMMARY")
    logger.info("="*60)
    
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            improvement = metrics['accuracy'] - baseline_accuracy
            logger.info(f"{model_name}: Accuracy = {metrics['accuracy']:.4f} (Δ = {improvement:+.4f})")
    
    # Save results
    results_dir = Path("reports/transfer_learning")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    import json
    results_json = {}
    for model_name, metrics in results.items():
        if 'error' not in metrics:
            results_json[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'roc_auc': float(metrics['roc_auc']) if metrics['roc_auc'] else None
            }
        else:
            results_json[model_name] = {'error': metrics['error']}
    
    results_json['baseline_accuracy'] = float(baseline_accuracy)
    results_json['uci_dataset_shape'] = uci_transfer_df.shape
    results_json['oulad_dataset_shape'] = oulad_transfer_df.shape
    
    results_path = results_dir / "transfer_learning_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    results = run_transfer_learning()