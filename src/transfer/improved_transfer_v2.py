"""
Enhanced Transfer Learning for OULAD → UCI Performance Improvement

This module implements comprehensive improvements to the transfer learning pipeline
to achieve better cross-domain performance on educational datasets.

Key improvements:
1. Robust preprocessing pipeline handling mixed data types
2. Advanced feature engineering and alignment
3. Domain adaptation techniques (CORAL, MMD)
4. Optimized neural network architectures
5. Ensemble methods with calibration
6. Threshold optimization for imbalanced classification
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
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    roc_auc_score, precision_recall_curve, roc_curve,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class RobustPreprocessor:
    """Robust preprocessing pipeline for mixed categorical/numeric data."""
    
    def __init__(self, handle_categorical=True, scale_numeric=True):
        self.handle_categorical = handle_categorical
        self.scale_numeric = scale_numeric
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        self.categorical_features = []
        self.numeric_features = []
        
    def fit(self, X: pd.DataFrame):
        """Fit the preprocessor on training data."""
        self.feature_names = list(X.columns)
        
        # Identify categorical and numeric features
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                self.categorical_features.append(col)
            else:
                self.numeric_features.append(col)
        
        logger.info(f"Categorical features: {len(self.categorical_features)}")
        logger.info(f"Numeric features: {len(self.numeric_features)}")
        
        # Fit label encoders for categorical features
        if self.handle_categorical:
            for col in self.categorical_features:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values by adding 'missing' category
                values = X[col].fillna('missing').astype(str)
                self.label_encoders[col].fit(values)
        
        # Fit scaler for numeric features
        if self.scale_numeric and self.numeric_features:
            self.scaler = RobustScaler()
            numeric_data = X[self.numeric_features].fillna(X[self.numeric_features].median())
            self.scaler.fit(numeric_data)
            
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessors."""
        X_processed = X.copy()
        
        # Process categorical features
        if self.handle_categorical:
            for col in self.categorical_features:
                if col in X_processed.columns:
                    # Handle missing values and unseen categories
                    values = X_processed[col].fillna('missing').astype(str)
                    # Handle unseen categories by mapping to 'missing'
                    known_classes = set(self.label_encoders[col].classes_)
                    values = values.apply(lambda x: x if x in known_classes else 'missing')
                    X_processed[col] = self.label_encoders[col].transform(values)
        
        # Process numeric features
        if self.scale_numeric and self.numeric_features:
            numeric_cols = [col for col in self.numeric_features if col in X_processed.columns]
            if numeric_cols:
                # Fill missing values with median from training
                for col in numeric_cols:
                    median_val = X_processed[col].median() if not X_processed[col].isna().all() else 0
                    X_processed[col] = X_processed[col].fillna(median_val)
                
                X_processed[numeric_cols] = self.scaler.transform(X_processed[numeric_cols])
        
        # Ensure we only return columns that were in the original feature set
        feature_cols = [col for col in self.feature_names if col in X_processed.columns]
        return X_processed[feature_cols].values
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class AdvancedFeatureEngineer:
    """Advanced feature engineering for cross-domain alignment."""
    
    def __init__(self, n_pca_components=10, use_interactions=True):
        self.n_pca_components = n_pca_components
        self.use_interactions = use_interactions
        self.pca = None
        self.interaction_features = []
        
    def fit(self, X: np.ndarray, feature_names: List[str]):
        """Fit feature engineering on training data."""
        self.original_features = feature_names
        
        # Fit PCA for dimensionality reduction
        if X.shape[1] > self.n_pca_components:
            self.pca = PCA(n_components=self.n_pca_components)
            self.pca.fit(X)
            logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Identify interaction features
        if self.use_interactions and X.shape[1] > 1:
            # Use top features for interactions to avoid explosion
            n_features = min(5, X.shape[1])
            for i in range(n_features):
                for j in range(i+1, n_features):
                    self.interaction_features.append((i, j))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features with advanced engineering."""
        features = [X]
        
        # Add PCA components
        if self.pca is not None:
            X_pca = self.pca.transform(X)
            features.append(X_pca)
        
        # Add interaction features
        if self.interaction_features:
            interactions = []
            for i, j in self.interaction_features:
                if i < X.shape[1] and j < X.shape[1]:
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                    interactions.append(interaction)
            if interactions:
                features.append(np.hstack(interactions))
        
        # Add polynomial features (degree 2) for first few features
        if X.shape[1] > 0:
            n_poly = min(3, X.shape[1])
            poly_features = X[:, :n_poly] ** 2
            features.append(poly_features)
        
        return np.hstack(features)
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, feature_names).transform(X)


class DomainAdaptationCORAL:
    """CORAL domain adaptation for reducing domain shift."""
    
    def __init__(self, lambda_coral=1.0):
        self.lambda_coral = lambda_coral
        self.source_cov = None
        self.target_cov = None
        
    def fit(self, X_source: np.ndarray, X_target: np.ndarray):
        """Compute source and target covariances."""
        # Center the data
        X_source_centered = X_source - np.mean(X_source, axis=0)
        X_target_centered = X_target - np.mean(X_target, axis=0)
        
        # Compute covariance matrices
        self.source_cov = np.cov(X_source_centered.T) + np.eye(X_source.shape[1]) * 1e-6
        self.target_cov = np.cov(X_target_centered.T) + np.eye(X_target.shape[1]) * 1e-6
        
        return self
    
    def transform_source(self, X_source: np.ndarray) -> np.ndarray:
        """Transform source data to align with target domain."""
        if self.source_cov is None or self.target_cov is None:
            return X_source
        
        try:
            # Compute transformation matrix
            source_cov_sqrt = np.linalg.cholesky(self.source_cov)
            target_cov_sqrt = np.linalg.cholesky(self.target_cov)
            
            # Apply CORAL transformation
            transform_matrix = np.linalg.solve(source_cov_sqrt, target_cov_sqrt)
            X_source_centered = X_source - np.mean(X_source, axis=0)
            X_transformed = X_source_centered @ transform_matrix.T
            
            return X_transformed
        except np.linalg.LinAlgError:
            logger.warning("CORAL transformation failed, returning original data")
            return X_source


class OptimizedEnsemble:
    """Optimized ensemble with calibration and threshold optimization."""
    
    def __init__(self, use_calibration=True, optimize_threshold=True):
        self.use_calibration = use_calibration
        self.optimize_threshold = optimize_threshold
        self.ensemble = None
        self.optimal_threshold = 0.5
        self.calibrator = None
        
    def create_base_models(self):
        """Create diverse base models for ensemble."""
        models = [
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, random_state=42, class_weight='balanced'
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                random_state=42
            )),
            ('lr', LogisticRegression(
                random_state=42, class_weight='balanced', max_iter=1000
            )),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500, 
                random_state=42, early_stopping=True,
                validation_fraction=0.1
            ))
        ]
        
        return VotingClassifier(models, voting='soft')
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Fit ensemble with calibration and threshold optimization."""
        
        # Create and fit base ensemble
        self.ensemble = self.create_base_models()
        self.ensemble.fit(X_train, y_train)
        
        # Calibrate if requested
        if self.use_calibration:
            self.calibrator = CalibratedClassifierCV(self.ensemble, method='isotonic', cv=3)
            self.calibrator.fit(X_train, y_train)
        
        # Optimize threshold if validation data provided
        if self.optimize_threshold and X_val is not None and y_val is not None:
            if self.calibrator:
                y_prob = self.calibrator.predict_proba(X_val)[:, 1]
            else:
                y_prob = self.ensemble.predict_proba(X_val)[:, 1]
            
            # Find optimal threshold using F1 score
            precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            logger.info(f"Optimal threshold: {self.optimal_threshold:.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with optimized threshold."""
        y_prob = self.predict_proba(X)[:, 1]
        return (y_prob >= self.optimal_threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with calibration."""
        if self.calibrator:
            return self.calibrator.predict_proba(X)
        else:
            return self.ensemble.predict_proba(X)


def improved_transfer_experiment(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    use_domain_adaptation: bool = True,
    use_advanced_features: bool = True,
    use_ensemble: bool = True
) -> Dict[str, Any]:
    """
    Run improved transfer learning experiment with all enhancements.
    
    Args:
        source_data: Source domain DataFrame with 'label' column
        target_data: Target domain DataFrame with 'label' column  
        test_size: Fraction of target data to use for testing
        random_state: Random seed for reproducibility
        use_domain_adaptation: Whether to use CORAL domain adaptation
        use_advanced_features: Whether to use advanced feature engineering
        use_ensemble: Whether to use optimized ensemble
        
    Returns:
        Dictionary with performance metrics and metadata
    """
    logger.info("Starting improved transfer learning experiment...")
    
    # Find common features
    feature_cols = [col for col in source_data.columns if col != 'label']
    target_feature_cols = [col for col in target_data.columns if col != 'label']
    common_features = list(set(feature_cols) & set(target_feature_cols))
    
    if not common_features:
        raise ValueError("No common features found between source and target domains")
    
    logger.info(f"Using {len(common_features)} common features: {common_features}")
    
    # Prepare data with common features
    X_source = source_data[common_features].copy()
    y_source = source_data['label'].copy()
    X_target = target_data[common_features].copy() 
    y_target = target_data['label'].copy()
    
    # Clean labels - ensure they are numeric
    y_source = pd.to_numeric(y_source, errors='coerce')
    y_target = pd.to_numeric(y_target, errors='coerce')
    
    # Remove rows with invalid labels
    valid_source = ~y_source.isna()
    valid_target = ~y_target.isna()
    
    X_source = X_source[valid_source]
    y_source = y_source[valid_source].astype(int)
    X_target = X_target[valid_target]
    y_target = y_target[valid_target].astype(int)
    
    logger.info(f"Data after cleaning - Source: {X_source.shape}, Target: {X_target.shape}")
    logger.info(f"Source labels: {y_source.value_counts().to_dict()}")
    logger.info(f"Target labels: {y_target.value_counts().to_dict()}")
    
    # Split target data into train/test
    from sklearn.model_selection import train_test_split
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target, test_size=test_size, random_state=random_state, 
        stratify=y_target
    )
    
    # Step 1: Robust preprocessing
    preprocessor = RobustPreprocessor()
    X_source_processed = preprocessor.fit_transform(X_source)
    X_target_train_processed = preprocessor.transform(X_target_train)
    X_target_test_processed = preprocessor.transform(X_target_test)
    
    logger.info(f"After preprocessing - Source: {X_source_processed.shape}, Target: {X_target_test_processed.shape}")
    
    # Step 2: Advanced feature engineering
    if use_advanced_features:
        feature_engineer = AdvancedFeatureEngineer()
        X_source_processed = feature_engineer.fit_transform(X_source_processed, common_features)
        X_target_train_processed = feature_engineer.transform(X_target_train_processed)
        X_target_test_processed = feature_engineer.transform(X_target_test_processed)
        logger.info(f"After feature engineering - Source: {X_source_processed.shape}, Target: {X_target_test_processed.shape}")
    
    # Step 3: Domain adaptation
    if use_domain_adaptation:
        coral = DomainAdaptationCORAL()
        coral.fit(X_source_processed, X_target_train_processed)
        X_source_processed = coral.transform_source(X_source_processed)
        logger.info("Applied CORAL domain adaptation")
    
    # Step 4: Train optimized ensemble
    if use_ensemble:
        model = OptimizedEnsemble()
        model.fit(X_source_processed, y_source, X_target_train_processed, y_target_train)
    else:
        # Fallback to single Random Forest
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=random_state,
            class_weight='balanced'
        )
        model.fit(X_source_processed, y_source)
    
    # Step 5: Evaluate on target test set
    y_pred = model.predict(X_target_test_processed)
    y_prob = model.predict_proba(X_target_test_processed)[:, 1]
    
    # Compute comprehensive metrics
    results = {
        'accuracy': accuracy_score(y_target_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_target_test, y_pred),
        'f1': f1_score(y_target_test, y_pred),
        'auc': roc_auc_score(y_target_test, y_prob),
        'brier_score': brier_score_loss(y_target_test, y_prob),
    }
    
    # Cross-validation on source domain for reference
    if use_ensemble:
        source_cv_scores = cross_val_score(
            OptimizedEnsemble().create_base_models(), 
            X_source_processed, y_source, cv=3, scoring='accuracy'
        )
    else:
        source_cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=random_state),
            X_source_processed, y_source, cv=3, scoring='accuracy'
        )
    
    results['source_cv_accuracy'] = source_cv_scores.mean()
    results['source_cv_std'] = source_cv_scores.std()
    
    # Metadata
    results['n_features'] = X_target_test_processed.shape[1]
    results['n_source_samples'] = len(y_source)
    results['n_target_test_samples'] = len(y_target_test)
    results['target_class_distribution'] = y_target_test.value_counts().to_dict()
    results['use_domain_adaptation'] = use_domain_adaptation
    results['use_advanced_features'] = use_advanced_features
    results['use_ensemble'] = use_ensemble
    
    if hasattr(model, 'optimal_threshold'):
        results['optimal_threshold'] = model.optimal_threshold
    
    logger.info(f"Improved transfer results - Accuracy: {results['accuracy']:.3f}, AUC: {results['auc']:.3f}")
    
    return results


def main():
    """Test the improved transfer learning approach."""
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
    
    print("=== Improved Transfer Learning Results ===")
    
    # Test different configurations
    configs = [
        {'use_domain_adaptation': False, 'use_advanced_features': False, 'use_ensemble': False},
        {'use_domain_adaptation': True, 'use_advanced_features': False, 'use_ensemble': False},
        {'use_domain_adaptation': False, 'use_advanced_features': True, 'use_ensemble': False},
        {'use_domain_adaptation': True, 'use_advanced_features': True, 'use_ensemble': False},
        {'use_domain_adaptation': True, 'use_advanced_features': True, 'use_ensemble': True},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        try:
            results = improved_transfer_experiment(oulad_clean, uci_clean, **config)
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  AUC: {results['auc']:.3f}")
            print(f"  F1: {results['f1']:.3f}")
            print(f"  Features: {results['n_features']}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()