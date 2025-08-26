"""
Enhanced Transfer Learning for OULAD → UCI Performance Improvement

This module implements key improvements to boost transfer learning performance:
1. Better feature preprocessing and alignment
2. Domain adaptation with CORAL
3. Ensemble methods with calibration
4. Threshold optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    roc_auc_score, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)


def enhanced_preprocessing(X_source: pd.DataFrame, X_target: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced preprocessing with robust scaling and missing value handling."""
    
    # Find common columns
    common_cols = list(set(X_source.columns) & set(X_target.columns))
    logger.info(f"Using {len(common_cols)} common features: {common_cols}")
    
    X_source_common = X_source[common_cols].copy()
    X_target_common = X_target[common_cols].copy()
    
    # Handle categorical columns by label encoding
    for col in common_cols:
        if X_source_common[col].dtype == 'object':
            # Combine categories from both datasets
            all_categories = set(X_source_common[col].dropna().unique()) | set(X_target_common[col].dropna().unique())
            category_map = {cat: i for i, cat in enumerate(sorted(all_categories))}
            
            X_source_common[col] = X_source_common[col].map(category_map).fillna(-1)
            X_target_common[col] = X_target_common[col].map(category_map).fillna(-1)
    
    # Fill missing numeric values with median
    for col in common_cols:
        if X_source_common[col].dtype in ['int64', 'float64']:
            median_val = X_source_common[col].median()
            X_source_common[col] = X_source_common[col].fillna(median_val)
            X_target_common[col] = X_target_common[col].fillna(median_val)
    
    # Scale features using RobustScaler
    scaler = RobustScaler()
    X_source_scaled = scaler.fit_transform(X_source_common)
    X_target_scaled = scaler.transform(X_target_common)
    
    return X_source_scaled, X_target_scaled


def coral_domain_adaptation(X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
    """Apply CORAL domain adaptation to align source with target domain."""
    
    # Compute covariance matrices
    source_cov = np.cov(X_source.T) + np.eye(X_source.shape[1]) * 1e-6
    target_cov = np.cov(X_target.T) + np.eye(X_target.shape[1]) * 1e-6
    
    try:
        # Compute transformation matrix (CORAL)
        source_cov_sqrt = np.linalg.cholesky(source_cov)
        target_cov_sqrt = np.linalg.cholesky(target_cov)
        
        transform_matrix = np.linalg.solve(source_cov_sqrt, target_cov_sqrt)
        
        # Apply transformation
        X_source_centered = X_source - np.mean(X_source, axis=0)
        X_source_aligned = X_source_centered @ transform_matrix.T
        
        logger.info("Applied CORAL domain adaptation")
        return X_source_aligned
        
    except np.linalg.LinAlgError:
        logger.warning("CORAL failed, returning original source data")
        return X_source


def create_ensemble_model(random_state: int = 42) -> VotingClassifier:
    """Create an optimized ensemble model."""
    
    models = [
        ('rf', RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=random_state, 
            class_weight='balanced'
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.1,
            random_state=random_state
        )),
        ('lr', LogisticRegression(
            random_state=random_state, class_weight='balanced', 
            max_iter=1000, C=0.1
        ))
    ]
    
    return VotingClassifier(models, voting='soft')


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal classification threshold using F1 score."""
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    logger.info(f"Optimal threshold: {optimal_threshold:.3f} (F1: {f1_scores[optimal_idx]:.3f})")
    return optimal_threshold


def enhanced_transfer_experiment(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    use_coral: bool = True,
    use_ensemble: bool = True,
    use_calibration: bool = True,
    use_feature_selection: bool = True,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Enhanced transfer learning with multiple improvements.
    
    Args:
        source_data: Source domain data with 'label' column
        target_data: Target domain data with 'label' column
        use_coral: Whether to apply CORAL domain adaptation
        use_ensemble: Whether to use ensemble instead of single model
        use_calibration: Whether to calibrate probabilities
        use_feature_selection: Whether to apply feature selection
        test_size: Fraction of target data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with performance metrics
    """
    
    logger.info("Starting enhanced transfer learning experiment...")
    
    # Separate features and labels
    feature_cols = [col for col in source_data.columns if col != 'label']
    X_source = source_data[feature_cols].copy()
    y_source = source_data['label'].copy().astype(int)
    
    target_feature_cols = [col for col in target_data.columns if col != 'label']
    X_target = target_data[target_feature_cols].copy()
    y_target = target_data['label'].copy().astype(int)
    
    # Split target data
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target, test_size=test_size, random_state=random_state, 
        stratify=y_target
    )
    
    # Enhanced preprocessing
    X_source_proc, X_target_train_proc = enhanced_preprocessing(X_source, X_target_train)
    _, X_target_test_proc = enhanced_preprocessing(X_source, X_target_test)
    
    logger.info(f"After preprocessing - Source: {X_source_proc.shape}, Target test: {X_target_test_proc.shape}")
    
    # Feature engineering - add PCA components
    if X_source_proc.shape[1] > 5:
        pca = PCA(n_components=min(5, X_source_proc.shape[1]), random_state=random_state)
        X_source_pca = pca.fit_transform(X_source_proc)
        X_target_test_pca = pca.transform(X_target_test_proc)
        
        # Combine original and PCA features
        X_source_proc = np.hstack([X_source_proc, X_source_pca])
        X_target_test_proc = np.hstack([X_target_test_proc, X_target_test_pca])
        logger.info(f"Added PCA features, new shape: {X_source_proc.shape}")
    
    # Feature selection
    if use_feature_selection and X_source_proc.shape[1] > 10:
        selector = SelectKBest(f_classif, k=min(15, X_source_proc.shape[1]))
        X_source_proc = selector.fit_transform(X_source_proc, y_source)
        X_target_test_proc = selector.transform(X_target_test_proc)
        logger.info(f"After feature selection: {X_source_proc.shape}")
    
    # Domain adaptation
    if use_coral:
        X_source_proc = coral_domain_adaptation(X_source_proc, X_target_test_proc)
    
    # Model training
    if use_ensemble:
        model = create_ensemble_model(random_state)
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=random_state,
            class_weight='balanced'
        )
    
    # Calibration
    if use_calibration:
        model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    
    # Train model
    model.fit(X_source_proc, y_source)
    
    # Predictions
    y_prob = model.predict_proba(X_target_test_proc)[:, 1]
    
    # Threshold optimization
    # Use small validation set from target for threshold tuning
    val_size = min(50, len(X_target_train) // 2)
    if val_size > 10:
        X_val = X_target_train.iloc[:val_size]
        y_val = y_target_train.iloc[:val_size]
        _, X_val_proc = enhanced_preprocessing(X_source, X_val)
        
        if X_source_proc.shape[1] > 5:
            X_val_pca = pca.transform(X_val_proc)
            X_val_proc = np.hstack([X_val_proc, X_val_pca])
        
        if use_feature_selection and 'selector' in locals():
            X_val_proc = selector.transform(X_val_proc)
        
        y_val_prob = model.predict_proba(X_val_proc)[:, 1]
        optimal_threshold = optimize_threshold(y_val, y_val_prob)
    else:
        optimal_threshold = 0.5
    
    # Final predictions
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # Compute metrics
    results = {
        'accuracy': accuracy_score(y_target_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_target_test, y_pred),
        'f1': f1_score(y_target_test, y_pred),
        'auc': roc_auc_score(y_target_test, y_prob),
        'brier_score': brier_score_loss(y_target_test, y_prob),
        'optimal_threshold': optimal_threshold,
        'n_features': X_source_proc.shape[1],
        'n_source_samples': len(y_source),
        'n_target_test_samples': len(y_target_test),
        'target_class_distribution': y_target_test.value_counts().to_dict(),
        'use_coral': use_coral,
        'use_ensemble': use_ensemble,
        'use_calibration': use_calibration,
        'use_feature_selection': use_feature_selection
    }
    
    # Source domain CV for reference
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=random_state),
        X_source_proc, y_source, cv=3, scoring='accuracy'
    )
    results['source_cv_accuracy'] = cv_scores.mean()
    results['source_cv_std'] = cv_scores.std()
    
    logger.info(f"Enhanced transfer results - Accuracy: {results['accuracy']:.3f}, AUC: {results['auc']:.3f}")
    
    return results


def main():
    """Test the enhanced transfer learning approach."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    from transfer.uci_transfer import create_shared_feature_mapping, prepare_oulad_features, prepare_uci_features
    
    # Load and prepare data
    feature_mapping = create_shared_feature_mapping()
    oulad_df = pd.read_csv('data/oulad/processed/oulad_ml.csv')
    oulad_shared = prepare_oulad_features(oulad_df, feature_mapping)
    uci_shared = prepare_uci_features('student-mat-fixed.csv', feature_mapping)
    
    # Clean data
    oulad_clean = oulad_shared.dropna(subset=['label'])
    uci_clean = uci_shared.dropna(subset=['label'])
    
    print("=== Enhanced Transfer Learning Results ===")
    print(f"OULAD: {oulad_clean.shape}, UCI: {uci_clean.shape}")
    print(f"OULAD pass rate: {oulad_clean['label'].mean():.3f}")
    print(f"UCI pass rate: {uci_clean['label'].mean():.3f}")
    
    # Test different configurations
    configs = [
        {'name': 'Baseline RF', 'use_coral': False, 'use_ensemble': False, 'use_calibration': False, 'use_feature_selection': False},
        {'name': 'RF + CORAL', 'use_coral': True, 'use_ensemble': False, 'use_calibration': False, 'use_feature_selection': False},
        {'name': 'RF + Features', 'use_coral': False, 'use_ensemble': False, 'use_calibration': False, 'use_feature_selection': True},
        {'name': 'RF + CORAL + Features', 'use_coral': True, 'use_ensemble': False, 'use_calibration': False, 'use_feature_selection': True},
        {'name': 'Ensemble', 'use_coral': False, 'use_ensemble': True, 'use_calibration': False, 'use_feature_selection': False},
        {'name': 'Full Enhanced', 'use_coral': True, 'use_ensemble': True, 'use_calibration': True, 'use_feature_selection': True},
    ]
    
    for config in configs:
        name = config.pop('name')
        print(f"\n{name}:")
        try:
            results = enhanced_transfer_experiment(oulad_clean, uci_clean, **config)
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  AUC: {results['auc']:.3f}")
            print(f"  F1: {results['f1']:.3f}")
            print(f"  Balanced Acc: {results['balanced_accuracy']:.3f}")
            print(f"  Features: {results['n_features']}")
            print(f"  Threshold: {results['optimal_threshold']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()