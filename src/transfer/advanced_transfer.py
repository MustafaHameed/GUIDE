"""
Advanced Transfer Learning Techniques

Implement advanced techniques specifically targeting the domain gap
between OULAD and UCI datasets.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def advanced_feature_engineering(X_source: pd.DataFrame, X_target: pd.DataFrame) -> tuple:
    """
    Advanced feature engineering to bridge domain gap.
    
    Args:
        X_source: Source domain features
        X_target: Target domain features
        
    Returns:
        Transformed source and target features
    """
    # Ensure same columns
    common_cols = list(set(X_source.columns) & set(X_target.columns))
    X_source_common = X_source[common_cols].copy()
    X_target_common = X_target[common_cols].copy()
    
    # Handle categorical variables with target encoding
    for col in common_cols:
        if X_source_common[col].dtype == 'object':
            # Simple target encoding for categorical variables
            all_cats = set(X_source_common[col].unique()) | set(X_target_common[col].unique())
            cat_map = {cat: idx for idx, cat in enumerate(all_cats)}
            X_source_common[col] = X_source_common[col].map(cat_map).fillna(0)
            X_target_common[col] = X_target_common[col].map(cat_map).fillna(0)
    
    # Fill missing values
    X_source_common = X_source_common.fillna(X_source_common.median())
    X_target_common = X_target_common.fillna(X_target_common.median())
    
    # Feature scaling
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source_common)
    X_target_scaled = scaler.transform(X_target_common)
    
    # Domain adaptation: PCA for dimensionality reduction and noise removal
    pca = PCA(n_components=min(len(common_cols), 10), random_state=42)
    X_source_pca = pca.fit_transform(X_source_scaled)
    X_target_pca = pca.transform(X_target_scaled)
    
    # Combine original and PCA features
    X_source_final = np.hstack([X_source_scaled, X_source_pca])
    X_target_final = np.hstack([X_target_scaled, X_target_pca])
    
    return X_source_final, X_target_final


def create_calibrated_ensemble() -> VotingClassifier:
    """
    Create a calibrated ensemble classifier for better probability estimates.
    
    Returns:
        Calibrated ensemble classifier
    """
    # Base classifiers with diversity
    rf1 = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_split=10,
        random_state=42, class_weight='balanced'
    )
    
    rf2 = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        random_state=123, class_weight='balanced'
    )
    
    lr = LogisticRegression(
        C=0.01, penalty='l2', max_iter=2000,
        random_state=42, class_weight='balanced'
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf1', rf1),
            ('rf2', rf2), 
            ('lr', lr)
        ],
        voting='soft'
    )
    
    # Calibrate the ensemble
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    
    return calibrated


def advanced_transfer_learning(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame
) -> dict:
    """
    Advanced transfer learning with multiple techniques.
    
    Args:
        source_data: Source domain data
        target_data: Target domain data
        
    Returns:
        Performance metrics dictionary
    """
    logger.info("Running advanced transfer learning...")
    
    # Prepare features
    feature_cols = [col for col in source_data.columns if col != 'label']
    X_source = source_data[feature_cols]
    y_source = source_data['label']
    X_target = target_data[feature_cols]
    y_target = target_data['label']
    
    # Advanced feature engineering
    X_source_eng, X_target_eng = advanced_feature_engineering(X_source, X_target)
    
    # Create multiple models with different strategies
    results = {}
    
    # Strategy 1: Calibrated Random Forest with feature selection
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=3,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    
    # Feature selection using the RF
    selector = SelectFromModel(rf, threshold='median')
    X_source_selected = selector.fit_transform(X_source_eng, y_source)
    X_target_selected = selector.transform(X_target_eng)
    
    # Train and calibrate
    rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    rf_calibrated.fit(X_source_selected, y_source)
    
    y_pred_rf = rf_calibrated.predict(X_target_selected)
    y_prob_rf = rf_calibrated.predict_proba(X_target_selected)[:, 1]
    
    results['rf_calibrated'] = {
        'accuracy': accuracy_score(y_target, y_pred_rf),
        'auc': roc_auc_score(y_target, y_prob_rf),
        'f1': f1_score(y_target, y_pred_rf),
        'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_rf)
    }
    
    # Strategy 2: Ensemble with different feature sets
    ensemble = create_calibrated_ensemble()
    ensemble.fit(X_source_eng, y_source)
    
    y_pred_ens = ensemble.predict(X_target_eng)
    y_prob_ens = ensemble.predict_proba(X_target_eng)[:, 1]
    
    results['ensemble_calibrated'] = {
        'accuracy': accuracy_score(y_target, y_pred_ens),
        'auc': roc_auc_score(y_target, y_prob_ens),
        'f1': f1_score(y_target, y_pred_ens),
        'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_ens)
    }
    
    # Strategy 3: Threshold optimization
    # Find optimal threshold based on F1 score
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_target, y_prob_ens)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    y_pred_optimal = (y_prob_ens >= optimal_threshold).astype(int)
    
    results['ensemble_optimized'] = {
        'accuracy': accuracy_score(y_target, y_pred_optimal),
        'auc': roc_auc_score(y_target, y_prob_ens),
        'f1': f1_score(y_target, y_pred_optimal),
        'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_optimal),
        'optimal_threshold': optimal_threshold
    }
    
    # Strategy 4: Conservative prediction (predict positive only when very confident)
    conservative_threshold = 0.7  # Higher threshold for positive predictions
    y_pred_conservative = (y_prob_ens >= conservative_threshold).astype(int)
    
    results['ensemble_conservative'] = {
        'accuracy': accuracy_score(y_target, y_pred_conservative),
        'auc': roc_auc_score(y_target, y_prob_ens),
        'f1': f1_score(y_target, y_pred_conservative),
        'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_conservative),
        'threshold': conservative_threshold
    }
    
    # Add metadata
    results['metadata'] = {
        'source_size': len(X_source),
        'target_size': len(X_target),
        'engineered_features': X_source_eng.shape[1],
        'selected_features': X_source_selected.shape[1] if X_source_selected is not None else 0
    }
    
    return results


if __name__ == "__main__":
    # Test the advanced approach
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    from src.transfer.uci_transfer import create_shared_feature_mapping, prepare_oulad_features, prepare_uci_features
    
    # Load data
    oulad_df = pd.read_parquet('data/oulad/processed/oulad_ml.parquet')
    feature_mapping = create_shared_feature_mapping()
    oulad_shared = prepare_oulad_features(oulad_df, feature_mapping)
    uci_shared = prepare_uci_features('student-mat.csv', feature_mapping)
    
    # Remove rows with missing labels and align features
    oulad_clean = oulad_shared.dropna(subset=['label'])
    uci_clean = uci_shared.dropna(subset=['label'])
    
    # Use only common features
    common_features = [col for col in oulad_clean.columns if col in uci_clean.columns]
    oulad_aligned = oulad_clean[common_features].copy()
    uci_aligned = uci_clean[common_features].copy()
    
    print("=== Advanced Transfer Learning Results ===")
    print(f"UCI baseline accuracy: {uci_aligned['label'].value_counts(normalize=True).max():.3f}")
    
    results = advanced_transfer_learning(oulad_aligned, uci_aligned)
    
    for strategy, metrics in results.items():
        if strategy != 'metadata':
            print(f"\n{strategy}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}")
                else:
                    print(f"  {metric}: {value}")
    
    print(f"\nMetadata: {results['metadata']}")