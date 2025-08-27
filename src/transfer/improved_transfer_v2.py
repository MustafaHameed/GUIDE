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
    """Enhanced robust preprocessing pipeline for mixed categorical/numeric data."""
    
    def __init__(self, handle_categorical=True, scale_numeric=True, 
                 detect_mixed_types=True, handle_high_cardinality=True,
                 max_cardinality=50):
        self.handle_categorical = handle_categorical
        self.scale_numeric = scale_numeric
        self.detect_mixed_types = detect_mixed_types
        self.handle_high_cardinality = handle_high_cardinality
        self.max_cardinality = max_cardinality
        
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        self.categorical_features = []
        self.numeric_features = []
        self.mixed_features = []  # Features that appear numeric but are categorical
        self.high_cardinality_features = []  # Categorical features with many categories
        self.feature_types = {}  # Store detected types for each feature
        self.medians = {}  # Store medians for imputation
        
    def _detect_feature_type(self, col_data: pd.Series, col_name: str) -> str:
        """Enhanced feature type detection for mixed data types."""
        
        # Remove missing values for analysis
        non_null_data = col_data.dropna()
        
        if len(non_null_data) == 0:
            return 'categorical'  # Default for all-null columns
        
        # Check explicit data types first
        if col_data.dtype == 'object' or col_data.dtype.name == 'category':
            return 'categorical'
        
        # For numeric types, check if they're actually categorical
        if self.detect_mixed_types and col_data.dtype in ['int64', 'float64']:
            unique_count = non_null_data.nunique()
            total_count = len(non_null_data)
            
            # Heuristics for detecting categorical data in numeric columns
            if unique_count <= 10:  # Low cardinality suggests categorical
                return 'categorical'
            elif unique_count / total_count < 0.05:  # Very few unique values
                return 'categorical'
            elif col_data.dtype == 'int64' and unique_count <= 20:
                # Integer with reasonable unique count could be categorical
                # Check if values look like IDs or categories
                if non_null_data.min() >= 0 and non_null_data.max() < 100:
                    return 'categorical'
        
        return 'numeric'
        
    def fit(self, X: pd.DataFrame):
        """Enhanced fit method with better mixed data type handling."""
        self.feature_names = list(X.columns)
        
        # Enhanced feature type detection
        for col in X.columns:
            feature_type = self._detect_feature_type(X[col], col)
            self.feature_types[col] = feature_type
            
            if feature_type == 'categorical':
                self.categorical_features.append(col)
                
                # Check for high cardinality
                unique_count = X[col].nunique()
                if self.handle_high_cardinality and unique_count > self.max_cardinality:
                    self.high_cardinality_features.append(col)
                    
            else:
                self.numeric_features.append(col)
                # Store median for imputation
                self.medians[col] = X[col].median()
        
        logger.info(f"Enhanced feature detection:")
        logger.info(f"  Categorical features: {len(self.categorical_features)}")
        logger.info(f"  Numeric features: {len(self.numeric_features)}")
        logger.info(f"  High cardinality features: {len(self.high_cardinality_features)}")
        
        # Fit label encoders for categorical features
        if self.handle_categorical:
            for col in self.categorical_features:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values by adding 'missing' category
                values = X[col].astype(str).fillna('missing')
                
                # For high cardinality features, keep only top categories
                if col in self.high_cardinality_features:
                    value_counts = values.value_counts()
                    top_categories = value_counts.head(self.max_cardinality - 1).index.tolist()
                    values = values.apply(lambda x: x if x in top_categories else 'other')
                
                self.label_encoders[col].fit(values)
        
        # Fit scaler for numeric features with robust imputation
        if self.scale_numeric and self.numeric_features:
            self.scaler = RobustScaler()
            numeric_data = X[self.numeric_features].copy()
            
            # Use stored medians for imputation
            for col in self.numeric_features:
                numeric_data[col] = numeric_data[col].fillna(self.medians[col])
            
            self.scaler.fit(numeric_data)
            
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Enhanced transform method with robust mixed data type handling."""
        X_processed = X.copy()
        
        # Process categorical features with enhanced handling
        if self.handle_categorical:
            for col in self.categorical_features:
                if col in X_processed.columns:
                    # Handle missing values and unseen categories
                    values = X_processed[col].astype(str).fillna('missing')
                    
                    # For high cardinality features, apply same logic as training
                    if col in self.high_cardinality_features:
                        known_classes = set(self.label_encoders[col].classes_)
                        top_categories = [c for c in known_classes if c not in ['missing', 'other']]
                        values = values.apply(lambda x: x if x in top_categories else 'other')
                    
                    # Handle completely unseen categories by mapping to 'missing'
                    known_classes = set(self.label_encoders[col].classes_)
                    values = values.apply(lambda x: x if x in known_classes else 'missing')
                    X_processed[col] = self.label_encoders[col].transform(values)
        
        # Process numeric features with robust imputation
        if self.scale_numeric and self.numeric_features:
            numeric_cols = [col for col in self.numeric_features if col in X_processed.columns]
            if numeric_cols:
                # Use stored medians for consistent imputation
                for col in numeric_cols:
                    if col in self.medians:
                        X_processed[col] = X_processed[col].fillna(self.medians[col])
                    else:
                        # Fallback to column median or 0
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
    """Enhanced advanced feature engineering for cross-domain alignment."""
    
    def __init__(self, n_pca_components=10, use_interactions=True, use_polynomial=True,
                 use_statistical_features=True, use_feature_selection=True,
                 interaction_threshold=0.1, max_interactions=20):
        self.n_pca_components = n_pca_components
        self.use_interactions = use_interactions
        self.use_polynomial = use_polynomial
        self.use_statistical_features = use_statistical_features
        self.use_feature_selection = use_feature_selection
        self.interaction_threshold = interaction_threshold
        self.max_interactions = max_interactions
        
        self.pca = None
        self.interaction_features = []
        self.feature_correlations = None
        self.feature_statistics = {}
        self.selected_features = None
        self.original_features = []
        
    def _compute_feature_correlations(self, X: np.ndarray) -> np.ndarray:
        """Compute feature correlation matrix for intelligent interaction selection."""
        return np.corrcoef(X.T)
        
    def _select_interaction_features(self, X: np.ndarray) -> List[Tuple[int, int]]:
        """Intelligently select feature interactions based on correlations."""
        if not self.use_interactions or X.shape[1] < 2:
            return []
        
        interactions = []
        n_features = X.shape[1]
        
        # Compute correlations if feature selection is enabled
        if self.use_feature_selection and n_features > 2:
            corr_matrix = self._compute_feature_correlations(X)
            self.feature_correlations = corr_matrix
            
            # Select interactions based on moderate correlations
            # (not too high to avoid redundancy, not too low to ensure relevance)
            for i in range(n_features):
                for j in range(i+1, n_features):
                    corr_val = abs(corr_matrix[i, j])
                    if self.interaction_threshold < corr_val < 0.8:  # Sweet spot for interactions
                        interactions.append((i, j))
                        if len(interactions) >= self.max_interactions:
                            break
                if len(interactions) >= self.max_interactions:
                    break
        else:
            # Fallback to original method for small feature sets
            n_features_to_use = min(5, n_features)
            for i in range(n_features_to_use):
                for j in range(i+1, n_features_to_use):
                    interactions.append((i, j))
        
        logger.info(f"Selected {len(interactions)} interaction features")
        return interactions
        
    def _compute_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """Compute statistical features for better domain alignment."""
        if not self.use_statistical_features:
            return np.array([]).reshape(X.shape[0], 0)
        
        statistical_features = []
        
        # Row-wise statistics
        if X.shape[1] > 1:
            # Mean and std per sample
            statistical_features.extend([
                np.mean(X, axis=1).reshape(-1, 1),
                np.std(X, axis=1).reshape(-1, 1),
                np.median(X, axis=1).reshape(-1, 1),
                np.max(X, axis=1).reshape(-1, 1),
                np.min(X, axis=1).reshape(-1, 1)
            ])
        
        if statistical_features:
            return np.hstack(statistical_features)
        else:
            return np.array([]).reshape(X.shape[0], 0)
        
    def fit(self, X: np.ndarray, feature_names: List[str]):
        """Enhanced fit method with intelligent feature engineering."""
        self.original_features = feature_names
        
        # Fit PCA for dimensionality reduction
        if X.shape[1] > self.n_pca_components:
            self.pca = PCA(n_components=self.n_pca_components)
            self.pca.fit(X)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance ratio: {explained_variance:.3f}")
        
        # Intelligently select interaction features
        self.interaction_features = self._select_interaction_features(X)
        
        # Compute and store feature statistics for statistical features
        if self.use_statistical_features:
            self.feature_statistics = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0),
                'median': np.median(X, axis=0)
            }
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Enhanced transform with comprehensive feature engineering."""
        features = [X]
        
        # Add PCA components
        if self.pca is not None:
            X_pca = self.pca.transform(X)
            features.append(X_pca)
        
        # Add intelligent interaction features
        if self.interaction_features:
            interactions = []
            for i, j in self.interaction_features:
                if i < X.shape[1] and j < X.shape[1]:
                    # Multiple types of interactions
                    interaction_product = (X[:, i] * X[:, j]).reshape(-1, 1)
                    interaction_sum = (X[:, i] + X[:, j]).reshape(-1, 1)
                    interactions.extend([interaction_product, interaction_sum])
            if interactions:
                features.append(np.hstack(interactions))
        
        # Add polynomial features (degree 2) for selected features
        if self.use_polynomial and X.shape[1] > 0:
            n_poly = min(3, X.shape[1])
            poly_features = X[:, :n_poly] ** 2
            features.append(poly_features)
        
        # Add statistical features
        stat_features = self._compute_statistical_features(X)
        if stat_features.shape[1] > 0:
            features.append(stat_features)
        
        # Domain-specific transformations
        if X.shape[1] > 2:
            # Add ratio features for numerical stability
            ratios = []
            for i in range(min(3, X.shape[1])):
                for j in range(i+1, min(3, X.shape[1])):
                    # Safe ratio computation
                    denominator = X[:, j] + 1e-8  # Add small constant to avoid division by zero
                    ratio = (X[:, i] / denominator).reshape(-1, 1)
                    ratios.append(ratio)
            if ratios:
                features.append(np.hstack(ratios))
        
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
    """Enhanced optimized ensemble with advanced neural architectures and calibration."""
    
    def __init__(self, use_calibration=True, optimize_threshold=True,
                 use_advanced_networks=True, use_stacking=True,
                 ensemble_diversity=True):
        self.use_calibration = use_calibration
        self.optimize_threshold = optimize_threshold
        self.use_advanced_networks = use_advanced_networks
        self.use_stacking = use_stacking
        self.ensemble_diversity = ensemble_diversity
        
        self.ensemble = None
        self.optimal_threshold = 0.5
        self.calibrator = None
        self.threshold_metrics = {}
        
    def _create_optimized_mlp(self, input_dim: Optional[int] = None) -> MLPClassifier:
        """Create an optimized MLP with adaptive architecture."""
        
        # Adaptive architecture based on input dimension
        if input_dim is not None:
            if input_dim <= 10:
                hidden_layers = (64, 32)
            elif input_dim <= 50:
                hidden_layers = (128, 64, 32)
            else:
                hidden_layers = (256, 128, 64, 32)
        else:
            hidden_layers = (128, 64, 32)  # Default
        
        return MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.001,  # L2 regularization
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
            batch_size='auto'
        )
    
    def _create_diverse_mlp_ensemble(self, input_dim: Optional[int] = None) -> List:
        """Create diverse MLP architectures for better ensemble diversity."""
        mlp_models = []
        
        # Deep narrow network
        mlp_models.append(('mlp_deep', MLPClassifier(
            hidden_layer_sizes=(64, 64, 64),
            activation='relu', solver='adam', alpha=0.01,
            learning_rate='adaptive', max_iter=300,
            early_stopping=True, random_state=42
        )))
        
        # Wide shallow network
        mlp_models.append(('mlp_wide', MLPClassifier(
            hidden_layer_sizes=(200,),
            activation='tanh', solver='adam', alpha=0.001,
            learning_rate='adaptive', max_iter=400,
            early_stopping=True, random_state=43
        )))
        
        # Optimized architecture
        mlp_models.append(('mlp_optimized', self._create_optimized_mlp(input_dim)))
        
        return mlp_models
        
    def create_base_models(self, input_dim: Optional[int] = None):
        """Create enhanced diverse base models for ensemble."""
        models = []
        
        # Tree-based models with different configurations
        models.extend([
            ('rf_balanced', RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=3,
                min_samples_leaf=1, random_state=42, class_weight='balanced',
                max_features='sqrt', bootstrap=True
            )),
            ('rf_depth', RandomForestClassifier(
                n_estimators=200, max_depth=25, min_samples_split=5,
                min_samples_leaf=2, random_state=43, class_weight='balanced_subsample',
                max_features='log2', bootstrap=True
            )),
            ('gb_adaptive', GradientBoostingClassifier(
                n_estimators=150, max_depth=8, learning_rate=0.08,
                subsample=0.8, random_state=42, validation_fraction=0.1,
                n_iter_no_change=10
            )),
            ('gb_robust', GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.9, random_state=43, max_features='sqrt'
            ))
        ])
        
        # Linear models with different regularization
        models.extend([
            ('lr_l1', LogisticRegression(
                penalty='l1', solver='liblinear', C=0.1,
                random_state=42, class_weight='balanced', max_iter=1000
            )),
            ('lr_l2', LogisticRegression(
                penalty='l2', solver='lbfgs', C=1.0,
                random_state=42, class_weight='balanced', max_iter=1000
            )),
            ('lr_elastic', LogisticRegression(
                penalty='elasticnet', solver='saga', C=0.5, l1_ratio=0.5,
                random_state=42, class_weight='balanced', max_iter=1000
            ))
        ])
        
        # Enhanced neural networks
        if self.use_advanced_networks:
            models.extend(self._create_diverse_mlp_ensemble(input_dim))
        else:
            # Single optimized MLP
            models.append(('mlp', self._create_optimized_mlp(input_dim)))
        
        # Use stacking or voting ensemble
        if self.use_stacking:
            from sklearn.ensemble import StackingClassifier
            return StackingClassifier(
                estimators=models,
                final_estimator=LogisticRegression(
                    random_state=42, class_weight='balanced', max_iter=1000
                ),
                cv=5,
                stack_method='predict_proba'
            )
        else:
            return VotingClassifier(models, voting='soft')
    
    def _optimize_threshold_advanced(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Advanced threshold optimization with multiple metrics."""
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        # F1-based optimization
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_optimal_idx = np.argmax(f1_scores)
        f1_threshold = pr_thresholds[f1_optimal_idx] if f1_optimal_idx < len(pr_thresholds) else 0.5
        
        # Youden's J statistic (sensitivity + specificity - 1)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        youden_j = tpr - fpr
        youden_optimal_idx = np.argmax(youden_j)
        youden_threshold = roc_thresholds[youden_optimal_idx]
        
        # Balanced accuracy optimization
        balanced_acc_scores = []
        for threshold in roc_thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            balanced_acc_scores.append(balanced_acc)
        
        balanced_acc_optimal_idx = np.argmax(balanced_acc_scores)
        balanced_acc_threshold = roc_thresholds[balanced_acc_optimal_idx]
        
        # Return the best performing threshold (using F1 as primary metric)
        optimal_threshold = f1_threshold
        
        return {
            'f1_threshold': f1_threshold,
            'f1_score': f1_scores[f1_optimal_idx],
            'youden_threshold': youden_threshold,
            'youden_j': youden_j[youden_optimal_idx],
            'balanced_acc_threshold': balanced_acc_threshold,
            'balanced_acc': max(balanced_acc_scores),
            'optimal_threshold': optimal_threshold
        }
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Enhanced fit method with advanced calibration and threshold optimization."""
        
        # Create and fit base ensemble with input dimension info
        input_dim = X_train.shape[1] if X_train.ndim > 1 else None
        self.ensemble = self.create_base_models(input_dim)
        self.ensemble.fit(X_train, y_train)
        
        # Enhanced calibration
        if self.use_calibration:
            # Use sigmoid calibration for larger datasets, isotonic for smaller
            calibration_method = 'sigmoid' if len(y_train) > 1000 else 'isotonic'
            self.calibrator = CalibratedClassifierCV(
                self.ensemble, 
                method=calibration_method, 
                cv=5 if len(y_train) > 500 else 3
            )
            self.calibrator.fit(X_train, y_train)
            logger.info(f"Applied {calibration_method} calibration")
        
        # Advanced threshold optimization if validation data provided
        if self.optimize_threshold and X_val is not None and y_val is not None:
            if self.calibrator:
                y_prob = self.calibrator.predict_proba(X_val)[:, 1]
            else:
                y_prob = self.ensemble.predict_proba(X_val)[:, 1]
            
            # Use advanced threshold optimization
            self.threshold_metrics = self._optimize_threshold_advanced(y_val, y_prob)
            self.optimal_threshold = self.threshold_metrics['optimal_threshold']
            
            logger.info(f"Advanced threshold optimization completed:")
            logger.info(f"  Optimal threshold: {self.optimal_threshold:.3f}")
            logger.info(f"  F1 score: {self.threshold_metrics['f1_score']:.3f}")
            logger.info(f"  Balanced accuracy: {self.threshold_metrics['balanced_acc']:.3f}")
        
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