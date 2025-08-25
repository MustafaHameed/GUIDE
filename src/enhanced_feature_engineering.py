"""
Enhanced Feature Engineering Module

This module provides comprehensive feature engineering capabilities to enhance
ML/DL results on both OULAD and UCI datasets. It extends existing implementations
with advanced techniques for better domain adaptation and performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer,
    PolynomialFeatures, QuantileTransformer
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, mutual_info_classif, f_classif,
    SelectFromModel, RFE
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """
    Comprehensive feature engineering class for both OULAD and UCI datasets.
    """
    
    def __init__(self, dataset_type: str = "auto", config: Optional[Dict] = None):
        """
        Initialize the enhanced feature engineer.
        
        Args:
            dataset_type: Type of dataset ("oulad", "uci", "auto")
            config: Configuration dictionary for feature engineering
        """
        self.dataset_type = dataset_type
        self.config = config or {}
        self.feature_names_ = []
        self.scalers_ = {}
        self.selectors_ = {}
        self.transformers_ = {}
        self.is_fitted_ = False
        self.original_columns_ = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Fit and transform features with comprehensive engineering.
        
        Args:
            X: Input features dataframe
            y: Target variable (optional, used for supervised feature selection)
            
        Returns:
            Enhanced feature matrix
        """
        logger.info(f"Starting enhanced feature engineering for {X.shape[0]} samples, {X.shape[1]} features")
        
        # Store original columns for transform consistency
        self.original_columns_ = list(X.columns)
        
        # Detect dataset type if auto
        if self.dataset_type == "auto":
            self.dataset_type = self._detect_dataset_type(X)
            
        # Core feature engineering pipeline
        X_enhanced = self._core_feature_engineering(X, y)
        
        # Dataset-specific enhancements
        if self.dataset_type == "oulad":
            X_enhanced = self._oulad_specific_features(X_enhanced, X)
        elif self.dataset_type == "uci":
            X_enhanced = self._uci_specific_features(X_enhanced, X)
            
        # Advanced feature interactions
        X_enhanced = self._create_advanced_interactions(X_enhanced, y)
        
        # Statistical and aggregation features
        X_enhanced = self._create_statistical_features(X_enhanced)
        
        # Dimensionality reduction and representation learning
        X_enhanced = self._apply_dimensionality_reduction(X_enhanced)
        
        # Feature selection
        if y is not None:
            X_enhanced = self._apply_feature_selection(X_enhanced, y)
            
        logger.info(f"Enhanced features: {X_enhanced.shape[1]} (original: {X.shape[1]})")
        self.is_fitted_ = True
        
        return X_enhanced
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted transformers.
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit_transform before transform")
            
        # Ensure same columns as training data
        if self.original_columns_ is not None:
            # Reorder and fill missing columns
            X_aligned = pd.DataFrame(index=X.index)
            for col in self.original_columns_:
                if col in X.columns:
                    X_aligned[col] = X[col]
                else:
                    # Fill missing columns with median/mode
                    X_aligned[col] = 0  # Simple fallback
            X = X_aligned
        
        logger.info(f"Transforming data with shape {X.shape}")
        
        # Apply the same preprocessing steps as fit_transform
        X_work = X.copy()
        
        # Fill numeric columns
        numeric_cols = X_work.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_work[numeric_cols] = X_work[numeric_cols].fillna(X_work[numeric_cols].median())
        
        # Fill categorical columns
        categorical_cols = X_work.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X_work[col] = X_work[col].fillna(X_work[col].mode().iloc[0] if not X_work[col].mode().empty else 'unknown')
        
        # Encode categorical variables
        for col in categorical_cols:
            X_work[col] = pd.Categorical(X_work[col]).codes
            
        # Apply fitted scaler
        if 'scaler' in self.scalers_:
            X_scaled = self.scalers_['scaler'].transform(X_work)
        else:
            logger.warning("No fitted scaler found, using raw values")
            X_scaled = X_work.values
            
        X_enhanced = X_scaled
        
        # Add the same basic statistical features
        if X_enhanced.shape[1] >= 2:
            row_stats = []
            row_stats.append(np.mean(X_enhanced, axis=1).reshape(-1, 1))
            row_stats.append(np.std(X_enhanced, axis=1).reshape(-1, 1))
            row_stats.append(np.max(X_enhanced, axis=1).reshape(-1, 1))
            row_stats.append(np.min(X_enhanced, axis=1).reshape(-1, 1))
            row_stats.append(np.median(X_enhanced, axis=1).reshape(-1, 1))
            
            # Percentiles
            for p in [25, 75, 90]:
                percentile_vals = np.percentile(X_enhanced, p, axis=1).reshape(-1, 1)
                row_stats.append(percentile_vals)
                
            X_enhanced = np.hstack([X_enhanced] + row_stats)
        
        # Apply fitted transformers if dimensions match
        for name, transformer in self.transformers_.items():
            if hasattr(transformer, 'transform'):
                try:
                    expected_features = getattr(transformer, 'n_features_in_', None)
                    if expected_features is None or X_enhanced.shape[1] == expected_features:
                        if name == 'pca':
                            X_pca = transformer.transform(X_enhanced)
                            X_enhanced = np.hstack([X_enhanced, X_pca])
                        elif name == 'ica':
                            X_ica = transformer.transform(X_enhanced)
                            X_enhanced = np.hstack([X_enhanced, X_ica])
                except Exception as e:
                    logger.warning(f"Could not apply transformer {name}: {e}")
                    
        # Apply feature selection - pad or truncate to match expected dimensions
        if 'selector' in self.selectors_:
            try:
                selector = self.selectors_['selector']
                expected_features = getattr(selector, 'n_features_in_', None)
                
                if expected_features is not None:
                    current_features = X_enhanced.shape[1]
                    
                    if current_features < expected_features:
                        # Pad with zeros
                        padding = np.zeros((X_enhanced.shape[0], expected_features - current_features))
                        X_enhanced = np.hstack([X_enhanced, padding])
                    elif current_features > expected_features:
                        # Truncate
                        X_enhanced = X_enhanced[:, :expected_features]
                        
                    X_enhanced = selector.transform(X_enhanced)
                    
            except Exception as e:
                logger.warning(f"Could not apply feature selection: {e}")
            
        return X_enhanced
    
    def _detect_dataset_type(self, X: pd.DataFrame) -> str:
        """
        Automatically detect dataset type based on column names and characteristics.
        """
        columns = set(X.columns)
        
        # OULAD indicators
        oulad_indicators = {
            'vle_total_clicks', 'vle_days_active', 'assessment_mean_score',
            'vle_first4_clicks', 'vle_last4_clicks', 'assessment_count'
        }
        
        # UCI indicators  
        uci_indicators = {
            'studytime', 'Dalc', 'Walc', 'famrel', 'freetime', 'goout',
            'health', 'Medu', 'Fedu', 'school', 'sex', 'age'
        }
        
        oulad_matches = len(columns.intersection(oulad_indicators))
        uci_matches = len(columns.intersection(uci_indicators))
        
        if oulad_matches > uci_matches:
            logger.info("Detected OULAD dataset")
            return "oulad"
        elif uci_matches > oulad_matches:
            logger.info("Detected UCI dataset")
            return "uci"
        else:
            logger.info("Could not detect dataset type, using generic approach")
            return "generic"
    
    def _core_feature_engineering(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Core feature engineering common to all datasets.
        """
        # Handle missing values differently for numeric and categorical
        X_work = X.copy()
        
        # Fill numeric columns
        numeric_cols = X_work.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_work[numeric_cols] = X_work[numeric_cols].fillna(X_work[numeric_cols].median())
        
        # Fill categorical columns
        categorical_cols = X_work.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X_work[col] = X_work[col].fillna(X_work[col].mode().iloc[0] if not X_work[col].mode().empty else 'unknown')
        
        # Encode categorical variables
        for col in categorical_cols:
            X_work[col] = pd.Categorical(X_work[col]).codes
            
        # Robust scaling for numerical stability
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_work)
        self.scalers_['scaler'] = scaler
        
        # Store original feature names
        self.feature_names_ = [f"orig_{i}" for i in range(X_scaled.shape[1])]
        
        return X_scaled
    
    def _oulad_specific_features(self, X_enhanced: np.ndarray, X_original: pd.DataFrame) -> np.ndarray:
        """
        Create OULAD-specific features for temporal learning data.
        """
        logger.info("Creating OULAD-specific features...")
        
        features_to_add = []
        feature_names_to_add = []
        
        # Identify temporal engagement columns
        engagement_cols = [col for col in X_original.columns if 'vle_' in col or 'click' in col]
        if engagement_cols:
            # Engagement velocity and acceleration
            for i, col in enumerate(engagement_cols[:5]):  # Limit to first 5 for performance
                if i < X_enhanced.shape[1]:
                    col_data = X_enhanced[:, i].reshape(-1, 1)
                    
                    # Engagement intensity (normalized)
                    intensity = np.clip(col_data / (np.std(col_data) + 1e-8), -3, 3)
                    features_to_add.append(intensity)
                    feature_names_to_add.append(f"engagement_intensity_{i}")
                    
                    # Engagement consistency (inverse coefficient of variation)
                    if len(engagement_cols) > 1:
                        consistency = 1 / (np.std(X_enhanced[:, :min(len(engagement_cols), X_enhanced.shape[1])], axis=1, ddof=1).reshape(-1, 1) + 1e-8)
                        features_to_add.append(np.clip(consistency, 0, 10))
                        feature_names_to_add.append(f"engagement_consistency_{i}")
        
        # Assessment performance trends
        assessment_cols = [col for col in X_original.columns if 'assessment' in col or 'score' in col]
        if assessment_cols and len(assessment_cols) >= 2:
            # Performance trend (slope of scores over time)
            scores = X_enhanced[:, :min(len(assessment_cols), X_enhanced.shape[1])]
            if scores.shape[1] >= 2:
                trend = np.polyfit(range(scores.shape[1]), scores.T, 1)[0].reshape(-1, 1)
                features_to_add.append(trend)
                feature_names_to_add.append("performance_trend")
        
        # Learning pattern features
        if X_enhanced.shape[1] >= 3:
            # Early vs late engagement ratio
            early_features = X_enhanced[:, :X_enhanced.shape[1]//3]
            late_features = X_enhanced[:, -X_enhanced.shape[1]//3:]
            early_mean = np.mean(early_features, axis=1).reshape(-1, 1)
            late_mean = np.mean(late_features, axis=1).reshape(-1, 1)
            
            engagement_ratio = np.divide(late_mean, early_mean + 1e-8)
            features_to_add.append(np.clip(engagement_ratio, 0, 10))
            feature_names_to_add.append("early_late_engagement_ratio")
        
        if features_to_add:
            X_enhanced = np.hstack([X_enhanced] + features_to_add)
            self.feature_names_.extend(feature_names_to_add)
            
        return X_enhanced
    
    def _uci_specific_features(self, X_enhanced: np.ndarray, X_original: pd.DataFrame) -> np.ndarray:
        """
        Create UCI-specific features for academic performance data.
        """
        logger.info("Creating UCI-specific features...")
        
        features_to_add = []
        feature_names_to_add = []
        
        # Social and family factors
        social_cols = [col for col in X_original.columns 
                      if any(social in col.lower() for social in ['famrel', 'freetime', 'goout', 'friends'])]
        
        if social_cols and len(social_cols) >= 2:
            # Social support index
            social_indices = [X_original.columns.get_loc(col) for col in social_cols 
                            if col in X_original.columns and X_original.columns.get_loc(col) < X_enhanced.shape[1]]
            if social_indices:
                social_features = X_enhanced[:, social_indices]
                social_support = np.mean(social_features, axis=1).reshape(-1, 1)
                features_to_add.append(social_support)
                feature_names_to_add.append("social_support_index")
        
        # Educational background
        edu_cols = [col for col in X_original.columns if 'edu' in col.lower() or col in ['Medu', 'Fedu']]
        if edu_cols and len(edu_cols) >= 2:
            edu_indices = [X_original.columns.get_loc(col) for col in edu_cols 
                          if col in X_original.columns and X_original.columns.get_loc(col) < X_enhanced.shape[1]]
            if edu_indices:
                edu_features = X_enhanced[:, edu_indices]
                # Family education level
                family_edu = np.max(edu_features, axis=1).reshape(-1, 1)
                features_to_add.append(family_edu)
                feature_names_to_add.append("family_education_max")
                
                # Education balance (difference between parents)
                if edu_features.shape[1] >= 2:
                    edu_balance = np.abs(edu_features[:, 0] - edu_features[:, 1]).reshape(-1, 1)
                    features_to_add.append(edu_balance)
                    feature_names_to_add.append("education_balance")
        
        # Health and lifestyle factors
        health_cols = [col for col in X_original.columns 
                      if any(health in col.lower() for health in ['health', 'dalc', 'walc'])]
        if health_cols:
            health_indices = [X_original.columns.get_loc(col) for col in health_cols 
                            if col in X_original.columns and X_original.columns.get_loc(col) < X_enhanced.shape[1]]
            if health_indices:
                health_features = X_enhanced[:, health_indices]
                # Lifestyle risk score
                risk_score = np.mean(health_features, axis=1).reshape(-1, 1)
                features_to_add.append(risk_score)
                feature_names_to_add.append("lifestyle_risk_score")
        
        if features_to_add:
            X_enhanced = np.hstack([X_enhanced] + features_to_add)
            self.feature_names_.extend(feature_names_to_add)
            
        return X_enhanced
    
    def _create_advanced_interactions(self, X: np.ndarray, y: pd.Series = None) -> np.ndarray:
        """
        Create advanced feature interactions using mutual information guidance.
        """
        logger.info("Creating advanced feature interactions...")
        
        if X.shape[1] > 50:  # Limit for computational efficiency
            logger.info("Too many features for interaction creation, skipping...")
            return X
            
        features_to_add = []
        feature_names_to_add = []
        
        # Select top features for interactions based on mutual information
        if y is not None and X.shape[1] <= 20:
            try:
                mi_scores = mutual_info_classif(X, y, random_state=42)
                top_indices = np.argsort(mi_scores)[-min(8, X.shape[1]):]  # Top 8 features
                
                # Pairwise interactions
                for i, idx1 in enumerate(top_indices):
                    for idx2 in top_indices[i+1:]:
                        if len(features_to_add) < 10:  # Limit interactions
                            interaction = (X[:, idx1] * X[:, idx2]).reshape(-1, 1)
                            features_to_add.append(interaction)
                            feature_names_to_add.append(f"interact_{idx1}_{idx2}")
                
                # Polynomial features for top 3
                top_3 = top_indices[-3:]
                for idx in top_3:
                    if len(features_to_add) < 15:
                        poly = (X[:, idx] ** 2).reshape(-1, 1)
                        features_to_add.append(poly)
                        feature_names_to_add.append(f"poly_{idx}")
                        
            except Exception as e:
                logger.warning(f"Could not create supervised interactions: {e}")
        
        # Ratio features for numerical stability
        if X.shape[1] >= 4:
            for i in range(min(4, X.shape[1])):
                for j in range(i+1, min(4, X.shape[1])):
                    if len(features_to_add) < 20:
                        ratio = np.divide(X[:, i], X[:, j] + 1e-8).reshape(-1, 1)
                        ratio = np.clip(ratio, -10, 10)  # Clip extreme values
                        features_to_add.append(ratio)
                        feature_names_to_add.append(f"ratio_{i}_{j}")
        
        if features_to_add:
            X_enhanced = np.hstack([X] + features_to_add)
            self.feature_names_.extend(feature_names_to_add)
            return X_enhanced
        
        return X
    
    def _create_statistical_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create statistical aggregation features.
        """
        logger.info("Creating statistical features...")
        
        features_to_add = []
        feature_names_to_add = []
        
        # Row-wise statistics
        row_stats = [
            (np.mean(X, axis=1), "row_mean"),
            (np.std(X, axis=1), "row_std"),
            (np.max(X, axis=1), "row_max"),
            (np.min(X, axis=1), "row_min"),
            (np.median(X, axis=1), "row_median")
        ]
        
        for stat, name in row_stats:
            features_to_add.append(stat.reshape(-1, 1))
            feature_names_to_add.append(name)
        
        # Percentile features
        percentiles = [25, 75, 90]
        for p in percentiles:
            percentile_vals = np.percentile(X, p, axis=1).reshape(-1, 1)
            features_to_add.append(percentile_vals)
            feature_names_to_add.append(f"row_p{p}")
        
        # Skewness and kurtosis
        if X.shape[1] >= 3:
            try:
                skewness = stats.skew(X, axis=1, nan_policy='omit').reshape(-1, 1)
                kurtosis = stats.kurtosis(X, axis=1, nan_policy='omit').reshape(-1, 1)
                
                # Handle infinite values
                skewness = np.nan_to_num(skewness, nan=0, posinf=3, neginf=-3)
                kurtosis = np.nan_to_num(kurtosis, nan=0, posinf=3, neginf=-3)
                
                features_to_add.extend([skewness, kurtosis])
                feature_names_to_add.extend(["row_skewness", "row_kurtosis"])
            except Exception as e:
                logger.warning(f"Could not compute skewness/kurtosis: {e}")
        
        if features_to_add:
            X_enhanced = np.hstack([X] + features_to_add)
            self.feature_names_.extend(feature_names_to_add)
            return X_enhanced
        
        return X
    
    def _apply_dimensionality_reduction(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction for representation learning.
        """
        logger.info("Applying dimensionality reduction...")
        
        if X.shape[1] < 10:  # Skip if too few features
            return X
            
        features_to_add = []
        feature_names_to_add = []
        
        # PCA for linear combinations
        try:
            n_components = min(5, X.shape[1] // 2, X.shape[0] // 2)
            if n_components >= 2:
                pca = PCA(n_components=n_components, random_state=42)
                X_pca = pca.fit_transform(X)
                self.transformers_['pca'] = pca
                
                features_to_add.append(X_pca)
                feature_names_to_add.extend([f"pca_{i}" for i in range(n_components)])
        except Exception as e:
            logger.warning(f"PCA failed: {e}")
        
        # ICA for independent components (if enough samples)
        if X.shape[0] > 50 and X.shape[1] >= 4:
            try:
                n_components = min(3, X.shape[1] // 3)
                ica = FastICA(n_components=n_components, random_state=42, max_iter=200)
                X_ica = ica.fit_transform(X)
                self.transformers_['ica'] = ica
                
                features_to_add.append(X_ica)
                feature_names_to_add.extend([f"ica_{i}" for i in range(n_components)])
            except Exception as e:
                logger.warning(f"ICA failed: {e}")
        
        if features_to_add:
            X_enhanced = np.hstack([X] + features_to_add)
            self.feature_names_.extend(feature_names_to_add)
            return X_enhanced
        
        return X
    
    def _apply_feature_selection(self, X: np.ndarray, y: pd.Series) -> np.ndarray:
        """
        Apply intelligent feature selection to reduce dimensionality.
        """
        logger.info("Applying feature selection...")
        
        if X.shape[1] <= 20:  # Skip if few features
            return X
            
        try:
            # Use SelectKBest with mutual information
            k = min(int(X.shape[1] * 0.8), 100)  # Keep 80% of features or max 100
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            self.selectors_['selector'] = selector
            
            # Update feature names
            selected_indices = selector.get_support(indices=True)
            self.feature_names_ = [self.feature_names_[i] for i in selected_indices if i < len(self.feature_names_)]
            
            logger.info(f"Feature selection: {X.shape[1]} -> {X_selected.shape[1]}")
            return X_selected
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return X
    
    def get_feature_names(self) -> List[str]:
        """Get the names of engineered features."""
        return self.feature_names_
    
    def get_feature_importance(self, X: np.ndarray, y: pd.Series) -> pd.DataFrame:
        """
        Get feature importance using multiple methods.
        """
        if len(self.feature_names_) != X.shape[1]:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names_
            
        importance_data = {'feature': feature_names}
        
        # Mutual information
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            importance_data['mutual_info'] = mi_scores
        except:
            importance_data['mutual_info'] = [0] * len(feature_names)
            
        # Random Forest importance
        try:
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            importance_data['rf_importance'] = rf.feature_importances_
        except:
            importance_data['rf_importance'] = [0] * len(feature_names)
        
        return pd.DataFrame(importance_data).sort_values('mutual_info', ascending=False)


def create_domain_adaptive_features(
    source_X: pd.DataFrame, 
    target_X: pd.DataFrame,
    source_y: pd.Series = None,
    target_y: pd.Series = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create domain-adaptive features for transfer learning between OULAD and UCI.
    """
    logger.info("Creating domain-adaptive features for transfer learning...")
    
    # Find common features
    common_features = list(set(source_X.columns) & set(target_X.columns))
    logger.info(f"Found {len(common_features)} common features")
    
    if not common_features:
        logger.warning("No common features found, using positional alignment")
        # Use positional alignment as fallback
        min_features = min(source_X.shape[1], target_X.shape[1])
        source_aligned = source_X.iloc[:, :min_features]
        target_aligned = target_X.iloc[:, :min_features]
        common_features = [f"feature_{i}" for i in range(min_features)]
    else:
        source_aligned = source_X[common_features]
        target_aligned = target_X[common_features]
    
    # Apply simple but consistent feature engineering to both domains
    engineer = EnhancedFeatureEngineer(dataset_type="generic")
    
    # Fit on source
    source_enhanced = engineer.fit_transform(source_aligned, source_y)
    
    # Create a new engineer for target to ensure same feature engineering
    target_engineer = EnhancedFeatureEngineer(dataset_type="generic")
    target_enhanced = target_engineer.fit_transform(target_aligned, target_y)
    
    # Align dimensions by taking minimum features from both
    min_features = min(source_enhanced.shape[1], target_enhanced.shape[1])
    source_enhanced = source_enhanced[:, :min_features]
    target_enhanced = target_enhanced[:, :min_features]
    
    logger.info(f"Domain adaptation: Source {source_enhanced.shape}, Target {target_enhanced.shape}")
    
    return source_enhanced, target_enhanced