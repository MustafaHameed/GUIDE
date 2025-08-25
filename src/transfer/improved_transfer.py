"""
Improved Transfer Learning with Better Domain Adaptation

This module implements enhanced transfer learning techniques to improve
cross-domain performance on the OULAD to UCI transfer task.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def improved_feature_alignment(source_df: pd.DataFrame, target_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Improved feature alignment between source and target domains.
    
    Args:
        source_df: Source domain dataframe 
        target_df: Target domain dataframe
        
    Returns:
        Aligned source and target dataframes
    """
    # Find common features
    source_features = set(source_df.columns) - {'label'}
    target_features = set(target_df.columns) - {'label'}
    common_features = source_features & target_features
    
    logger.info(f"Common features: {len(common_features)}")
    logger.info(f"Source-only features: {len(source_features - common_features)}")
    logger.info(f"Target-only features: {len(target_features - common_features)}")
    
    # Use only common features
    common_features = list(common_features)
    
    source_aligned = source_df[common_features + ['label']].copy()
    target_aligned = target_df[common_features + ['label']].copy()
    
    # Feature engineering: create statistical features from target-only features
    # This helps capture information from features not in source domain
    target_only_features = list(target_features - set(common_features))
    if target_only_features:
        # Create aggregate features from target-only features
        for feature in target_only_features:
            if target_df[feature].dtype in ['int64', 'float64']:
                # Create bins for numerical features
                target_aligned[f'{feature}_binned'] = pd.cut(target_df[feature], bins=3, labels=['low', 'med', 'high'])
                # Add corresponding dummy feature for source (use median strategy)
                source_aligned[f'{feature}_binned'] = 'med'  # Use median as default
    
    return source_aligned, target_aligned


def create_ensemble_classifier(use_cv: bool = False) -> VotingClassifier:
    """
    Create an ensemble classifier optimized for transfer learning.
    
    Args:
        use_cv: Whether to use cross-validation for hyperparameter tuning
        
    Returns:
        Ensemble classifier
    """
    # Base classifiers optimized for transfer learning
    log_reg = LogisticRegression(
        C=0.1, 
        penalty='l2', 
        max_iter=2000, 
        random_state=42,
        class_weight='balanced'
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    ada = AdaBoostClassifier(
        n_estimators=50,
        learning_rate=0.8,
        random_state=42
    )
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('logistic', log_reg),
            ('random_forest', rf),
            ('gradient_boost', gb),
            ('ada_boost', ada)
        ],
        voting='soft'  # Use probability averages
    )
    
    return ensemble


def domain_adaptation_weights(source_data: pd.DataFrame, target_data: pd.DataFrame) -> np.ndarray:
    """
    Compute importance weights for source samples to better match target distribution.
    
    Args:
        source_data: Source domain data
        target_data: Target domain data
        
    Returns:
        Array of sample weights for source data
    """
    # Simple domain adaptation: weight source samples by their similarity to target distribution
    feature_cols = [col for col in source_data.columns if col != 'label']
    
    weights = np.ones(len(source_data))
    
    for col in feature_cols:
        if source_data[col].dtype in ['int64', 'float64']:
            # For numerical features, weight by distance to target mean
            source_values = source_data[col].fillna(source_data[col].median())
            target_values = target_data[col].fillna(target_data[col].median())
            
            source_mean = source_values.mean()
            target_mean = target_values.mean()
            source_std = source_values.std()
            
            if source_std > 0 and not np.isnan(target_mean):
                # Gaussian weighting based on distance from target mean
                distance = np.abs(source_values - target_mean) / source_std
                feature_weights = np.exp(-distance / 2)  # Gaussian-like weighting
                feature_weights = np.nan_to_num(feature_weights, nan=1.0)  # Handle NaN
                weights *= feature_weights
    
    # Normalize weights and handle any remaining NaN values
    weights = np.nan_to_num(weights, nan=1.0)
    if weights.sum() > 0:
        weights = weights / weights.mean()
    else:
        weights = np.ones(len(source_data))
    
    return weights


def improved_transfer_experiment(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    use_ensemble: bool = True,
    use_domain_adaptation: bool = True
) -> Dict[str, float]:
    """
    Run improved transfer learning experiment.
    
    Args:
        source_data: Source domain data
        target_data: Target domain data  
        use_ensemble: Whether to use ensemble methods
        use_domain_adaptation: Whether to apply domain adaptation
        
    Returns:
        Dictionary with performance metrics
    """
    logger.info("Running improved transfer learning experiment...")
    
    # Step 1: Improved feature alignment
    source_aligned, target_aligned = improved_feature_alignment(source_data, target_data)
    
    # Step 2: Prepare features and labels
    feature_cols = [col for col in source_aligned.columns if col != 'label']
    X_source = source_aligned[feature_cols]
    y_source = source_aligned['label']
    X_target = target_aligned[feature_cols]
    y_target = target_aligned['label']
    
    # Step 3: Handle categorical variables
    for col in feature_cols:
        if X_source[col].dtype == 'object':
            # Use label encoding for categorical variables
            all_categories = set(X_source[col].unique()) | set(X_target[col].unique())
            category_map = {cat: idx for idx, cat in enumerate(all_categories)}
            X_source[col] = X_source[col].map(category_map).fillna(0)
            X_target[col] = X_target[col].map(category_map).fillna(0)
    
    # Step 4: Feature scaling with robust scaler
    scaler = RobustScaler()
    X_source_scaled = scaler.fit_transform(X_source.fillna(0))
    X_target_scaled = scaler.transform(X_target.fillna(0))
    
    # Step 5: Feature selection - select most transferable features
    # Use mutual information to select features that are informative across domains
    selector = SelectKBest(score_func=mutual_info_classif, k=min(10, X_source_scaled.shape[1]))
    X_source_selected = selector.fit_transform(X_source_scaled, y_source)
    X_target_selected = selector.transform(X_target_scaled)
    
    # Step 6: Train model
    if use_ensemble:
        model = create_ensemble_classifier()
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    # Step 7: Apply domain adaptation weights if requested
    sample_weight = None
    if use_domain_adaptation:
        sample_weight = domain_adaptation_weights(source_aligned, target_aligned)
    
    # Step 8: Train the model
    if sample_weight is not None:
        model.fit(X_source_selected, y_source, sample_weight=sample_weight)
    else:
        model.fit(X_source_selected, y_source)
    
    # Step 9: Evaluate on target
    y_pred = model.predict(X_target_selected)
    y_prob = model.predict_proba(X_target_selected)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Step 10: Calculate metrics
    results = {
        'accuracy': accuracy_score(y_target, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_target, y_pred),
        'f1': f1_score(y_target, y_pred),
        'source_size': len(X_source),
        'target_size': len(X_target),
        'n_features': X_source_selected.shape[1]
    }
    
    if y_prob is not None:
        results['auc'] = roc_auc_score(y_target, y_prob)
    
    # Cross-validation on source domain for comparison
    cv_scores = cross_val_score(model, X_source_selected, y_source, cv=3, scoring='accuracy')
    results['source_cv_accuracy'] = cv_scores.mean()
    
    return results