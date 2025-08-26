"""
FeatureBridge: Unified Preprocessing and Feature Engineering Pipeline

This module implements a unified preprocessing pipeline for both OULAD and UCI datasets
to enable consistent feature engineering and domain adaptation for transfer learning.

Key responsibilities:
- Fix positive class convention (label_pass=1 positive everywhere)
- Single ColumnTransformer with StandardScaler for numeric, OneHotEncoder for categoricals
- Handle both OULAD and UCI datasets consistently using canonical feature schema
- Support domain-specific feature mappings and transformations
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Any
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureBridge(BaseEstimator, TransformerMixin):
    """
    Unified preprocessing and feature engineering pipeline for transfer learning.
    
    Implements a canonical feature schema that both OULAD and UCI datasets
    can be mapped to, enabling consistent preprocessing and domain adaptation.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 enforce_positive_class: bool = True,
                 handle_unknown_categories: str = 'ignore',
                 numeric_imputation: str = 'median',
                 categorical_imputation: str = 'most_frequent',
                 random_state: int = 42):
        """
        Initialize the FeatureBridge.
        
        Args:
            config_path: Path to feature bridge YAML configuration
            enforce_positive_class: Ensure label_pass=1 is positive class
            handle_unknown_categories: How to handle unknown categories ('ignore', 'error')
            numeric_imputation: Strategy for numeric missing values
            categorical_imputation: Strategy for categorical missing values
            random_state: Random seed for reproducibility
        """
        self.config_path = config_path
        self.enforce_positive_class = enforce_positive_class
        self.handle_unknown_categories = handle_unknown_categories
        self.numeric_imputation = numeric_imputation
        self.categorical_imputation = categorical_imputation
        self.random_state = random_state
        
        # Load configuration
        self.config = self._load_config()
        
        # Fitted attributes
        self.is_fitted_ = False
        self.canonical_features_ = None
        self.numeric_features_ = None
        self.categorical_features_ = None
        self.preprocessor_ = None
        self.feature_names_out_ = None
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if self.config_path is None:
            # Use default config from the repository
            config_path = Path(__file__).parent.parent.parent / "configs" / "feature_bridge.yaml"
        else:
            config_path = Path(self.config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config file {config_path} not found, using default schema")
            config = self._get_default_config()
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not available."""
        return {
            'canonical_schema': {
                'demographics': {
                    'sex': {'type': 'categorical', 'values': ['F', 'M']},
                    'age_group': {'type': 'categorical', 'values': ['young', 'middle', 'older']}
                },
                'socioeconomic': {
                    'ses_index': {'type': 'numeric', 'range': [0, 4]},
                    'family_support': {'type': 'numeric', 'range': [0, 4]}
                },
                'academic_background': {
                    'prior_performance': {'type': 'numeric', 'range': [0, 1]},
                    'study_load': {'type': 'numeric', 'range': [0, 1]}
                },
                'engagement': {
                    'activity_level': {'type': 'numeric', 'range': [0, 1]},
                    'study_time': {'type': 'numeric', 'range': [0, 4]}
                }
            }
        }
    
    def map_dataset_to_canonical(self, df: pd.DataFrame, 
                                source_type: str) -> pd.DataFrame:
        """
        Map a dataset (OULAD or UCI) to the canonical feature schema.
        
        Args:
            df: Input dataframe (OULAD or UCI)
            source_type: Dataset type ('oulad' or 'uci')
            
        Returns:
            DataFrame with canonical feature schema
        """
        logger.info(f"Mapping {source_type} dataset to canonical schema...")
        
        if source_type not in ['oulad', 'uci']:
            raise ValueError(f"Unsupported source_type: {source_type}")
        
        canonical_df = pd.DataFrame(index=df.index)
        
        # Apply dataset-specific mappings
        if source_type == 'uci':
            canonical_df = self._map_uci_to_canonical(df, canonical_df)
        else:  # oulad
            canonical_df = self._map_oulad_to_canonical(df, canonical_df)
        
        # Ensure positive class convention for target variable
        if self.enforce_positive_class and 'label_pass' in canonical_df.columns:
            canonical_df['label_pass'] = canonical_df['label_pass'].astype(int)
            logger.info(f"Enforced positive class convention: {canonical_df['label_pass'].value_counts().to_dict()}")
        
        logger.info(f"Mapped to canonical schema: {canonical_df.shape[1]} features, {canonical_df.shape[0]} samples")
        return canonical_df
    
    def _map_uci_to_canonical(self, uci_df: pd.DataFrame, 
                             canonical_df: pd.DataFrame) -> pd.DataFrame:
        """Map UCI dataset to canonical schema."""
        
        # Demographics
        if 'sex' in uci_df.columns:
            canonical_df['sex'] = uci_df['sex'].map({'F': 'F', 'M': 'M'}).fillna('M')
        
        if 'age' in uci_df.columns:
            age_map = {age: 'young' if age <= 17 else ('middle' if age <= 20 else 'older') 
                      for age in range(15, 26)}
            canonical_df['age_group'] = uci_df['age'].map(age_map).fillna('middle')
        
        # Socioeconomic
        if 'Medu' in uci_df.columns:  # Mother's education as SES proxy
            canonical_df['ses_index'] = uci_df['Medu'].fillna(2).astype(float)  # 2 = medium
        
        if 'famrel' in uci_df.columns:  # Family relationships
            # Scale from [1,5] to [0,4]
            canonical_df['family_support'] = (uci_df['famrel'] - 1).clip(0, 4).fillna(2.0)
        
        # Academic background
        if 'G1' in uci_df.columns:  # First period grade
            canonical_df['prior_performance'] = (uci_df['G1'] / 20.0).clip(0, 1).fillna(0.5)
        
        if 'studytime' in uci_df.columns:
            canonical_df['study_load'] = (uci_df['studytime'] / 4.0).clip(0, 1).fillna(0.5)
        
        # Engagement
        if 'studytime' in uci_df.columns:
            canonical_df['activity_level'] = (uci_df['studytime'] / 4.0).clip(0, 1).fillna(0.5)
            canonical_df['study_time'] = uci_df['studytime'].fillna(2).astype(float)
        
        # Target variable
        if 'G3' in uci_df.columns:
            canonical_df['label_pass'] = (uci_df['G3'] >= 10).astype(int)
        elif 'label_pass' in uci_df.columns:
            canonical_df['label_pass'] = uci_df['label_pass'].astype(int)
        
        return canonical_df
    
    def _map_oulad_to_canonical(self, oulad_df: pd.DataFrame, 
                               canonical_df: pd.DataFrame) -> pd.DataFrame:
        """Map OULAD dataset to canonical schema."""
        
        # Demographics
        if 'sex' in oulad_df.columns:
            canonical_df['sex'] = oulad_df['sex'].map({'F': 'F', 'M': 'M'}).fillna('M')
        
        if 'age_band' in oulad_df.columns:
            age_map = {'0-35': 'young', '35-55': 'middle', '55<=': 'older'}
            canonical_df['age_group'] = oulad_df['age_band'].map(age_map).fillna('middle')
        
        # Socioeconomic
        if 'imd_band' in oulad_df.columns:
            # Convert IMD percentile to SES index (higher IMD = lower SES, so reverse)
            imd_to_ses = lambda x: 4 - (x / 25.0) if pd.notna(x) else 2  # 4=high SES, 0=low SES
            canonical_df['ses_index'] = oulad_df['imd_band'].apply(imd_to_ses).clip(0, 4)
        
        if 'disability' in oulad_df.columns:
            # Use disability as inverse proxy for family support
            disability_map = {'N': 3.0, 'Y': 1.0}  # No disability = better support
            canonical_df['family_support'] = oulad_df['disability'].map(disability_map).fillna(2.0)
        
        # Academic background
        if 'prev_attempts' in oulad_df.columns:
            # Convert previous attempts to performance proxy (inverse relationship)
            attempts_to_perf = {0: 1.0, 1: 0.7, 2: 0.4}
            canonical_df['prior_performance'] = (
                oulad_df['prev_attempts']
                .map(lambda x: attempts_to_perf.get(x, 0.1 if x > 2 else 0.5))
                .fillna(0.5)
            )
        
        if 'studied_credits' in oulad_df.columns:
            # Normalize credits to [0,1]
            canonical_df['study_load'] = (
                (oulad_df['studied_credits'] - 30) / (240 - 30)
            ).clip(0, 1).fillna(0.5)
        
        # Engagement
        if 'vle_total_clicks' in oulad_df.columns:
            # Log-normalize VLE clicks to [0,1]
            clicks = oulad_df['vle_total_clicks'].fillna(0)
            log_clicks = np.log1p(clicks)  # log(1 + x)
            max_log_clicks = np.log1p(10000)  # reasonable upper bound
            canonical_df['activity_level'] = (log_clicks / max_log_clicks).clip(0, 1)
            
            # Convert clicks to study time proxy
            study_time_map = pd.cut(
                clicks, 
                bins=[0, 100, 500, 2000, np.inf], 
                labels=[0, 1, 2, 3],
                include_lowest=True
            ).astype(float)
            canonical_df['study_time'] = study_time_map.fillna(1.0)
        
        # Target variable
        if 'final_result' in oulad_df.columns:
            result_map = {'Pass': 1, 'Distinction': 1, 'Fail': 0, 'Withdrawn': 0}
            canonical_df['label_pass'] = oulad_df['final_result'].map(result_map).fillna(0).astype(int)
        elif 'label_pass' in oulad_df.columns:
            canonical_df['label_pass'] = oulad_df['label_pass'].astype(int)
        
        return canonical_df
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
            source_type: str = 'uci') -> 'FeatureBridge':
        """
        Fit the feature bridge on training data.
        
        Args:
            X: Training features
            y: Training labels (optional, can be included in X as 'label_pass')
            source_type: Type of source dataset ('uci' or 'oulad')
            
        Returns:
            self
        """
        logger.info(f"Fitting FeatureBridge on {source_type} data...")
        
        # Map to canonical schema
        canonical_df = self.map_dataset_to_canonical(X, source_type)
        
        # Separate features and target
        feature_cols = [col for col in canonical_df.columns if col != 'label_pass']
        self.canonical_features_ = feature_cols
        
        X_canonical = canonical_df[feature_cols]
        
        # Identify feature types
        self.numeric_features_ = []
        self.categorical_features_ = []
        
        for col in X_canonical.columns:
            if X_canonical[col].dtype in ['object', 'category']:
                self.categorical_features_.append(col)
            else:
                self.numeric_features_.append(col)
        
        logger.info(f"Identified {len(self.numeric_features_)} numeric and {len(self.categorical_features_)} categorical features")
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=self.numeric_imputation)),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=self.categorical_imputation)),
            ('onehot', OneHotEncoder(handle_unknown=self.handle_unknown_categories, sparse_output=False))
        ])
        
        self.preprocessor_ = ColumnTransformer([
            ('num', numeric_transformer, self.numeric_features_),
            ('cat', categorical_transformer, self.categorical_features_)
        ])
        
        # Fit preprocessor
        self.preprocessor_.fit(X_canonical)
        
        # Get feature names after transformation
        self.feature_names_out_ = self._get_feature_names()
        
        self.is_fitted_ = True
        logger.info(f"FeatureBridge fitted. Output features: {len(self.feature_names_out_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame, source_type: str = 'uci') -> np.ndarray:
        """
        Transform data using the fitted feature bridge.
        
        Args:
            X: Input features to transform
            source_type: Type of source dataset ('uci' or 'oulad')
            
        Returns:
            Transformed feature matrix
        """
        check_is_fitted(self, 'is_fitted_')
        
        # Map to canonical schema
        canonical_df = self.map_dataset_to_canonical(X, source_type)
        
        # Select features (exclude target if present)
        X_canonical = canonical_df[self.canonical_features_]
        
        # Apply preprocessing
        X_transformed = self.preprocessor_.transform(X_canonical)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                     source_type: str = 'uci') -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, source_type).transform(X, source_type)
    
    def get_target(self, X: pd.DataFrame, source_type: str = 'uci') -> pd.Series:
        """
        Extract and standardize target variable.
        
        Args:
            X: Input dataframe containing target variable
            source_type: Type of source dataset ('uci' or 'oulad')
            
        Returns:
            Standardized target variable (label_pass=1 for positive class)
        """
        canonical_df = self.map_dataset_to_canonical(X, source_type)
        
        if 'label_pass' not in canonical_df.columns:
            raise ValueError("Target variable 'label_pass' not found in canonical schema")
        
        target = canonical_df['label_pass'].astype(int)
        
        if self.enforce_positive_class:
            # Ensure 1 is positive class
            assert target.isin([0, 1]).all(), "Target must be binary (0/1)"
        
        return target
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        feature_names = []
        
        # Numeric features
        feature_names.extend([f"num__{col}" for col in self.numeric_features_])
        
        # Categorical features (after one-hot encoding)
        if self.categorical_features_:
            cat_transformer = self.preprocessor_.named_transformers_['cat']
            onehot_encoder = cat_transformer.named_steps['onehot']
            
            for i, col in enumerate(self.categorical_features_):
                categories = onehot_encoder.categories_[i]
                feature_names.extend([f"cat__{col}__{cat}" for cat in categories])
        
        return feature_names
    
    def get_feature_importance_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping from canonical features to transformed feature names.
        
        Returns:
            Dictionary mapping canonical feature names to list of transformed features
        """
        check_is_fitted(self, 'is_fitted_')
        
        mapping = {}
        feature_idx = 0
        
        # Numeric features (1-to-1 mapping)
        for col in self.numeric_features_:
            mapping[col] = [self.feature_names_out_[feature_idx]]
            feature_idx += 1
        
        # Categorical features (1-to-many mapping)
        if self.categorical_features_:
            cat_transformer = self.preprocessor_.named_transformers_['cat']
            onehot_encoder = cat_transformer.named_steps['onehot']
            
            for i, col in enumerate(self.categorical_features_):
                n_categories = len(onehot_encoder.categories_[i])
                mapping[col] = self.feature_names_out_[feature_idx:feature_idx + n_categories]
                feature_idx += n_categories
        
        return mapping
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing transformations applied.
        
        Returns:
            Dictionary with preprocessing summary
        """
        check_is_fitted(self, 'is_fitted_')
        
        summary = {
            'n_canonical_features': len(self.canonical_features_),
            'n_output_features': len(self.feature_names_out_),
            'numeric_features': self.numeric_features_,
            'categorical_features': self.categorical_features_,
            'feature_mapping': self.get_feature_importance_mapping(),
            'config_used': {
                'numeric_imputation': self.numeric_imputation,
                'categorical_imputation': self.categorical_imputation,
                'handle_unknown_categories': self.handle_unknown_categories,
                'enforce_positive_class': self.enforce_positive_class
            }
        }
        
        return summary


def create_feature_bridge(config_path: Optional[str] = None, **kwargs) -> FeatureBridge:
    """
    Convenience function to create a FeatureBridge instance.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments for FeatureBridge
        
    Returns:
        FeatureBridge instance
    """
    return FeatureBridge(config_path=config_path, **kwargs)


def demo_feature_bridge():
    """Demonstrate FeatureBridge with synthetic data."""
    # Create synthetic UCI-like data
    uci_data = pd.DataFrame({
        'sex': ['F', 'M', 'F', 'M', 'F'],
        'age': [18, 19, 20, 21, 22],
        'Medu': [1, 2, 3, 4, 2],
        'famrel': [3, 4, 2, 5, 3],
        'G1': [12, 8, 15, 18, 10],
        'studytime': [2, 1, 3, 4, 2],
        'G3': [14, 6, 16, 19, 11]
    })
    
    # Create synthetic OULAD-like data
    oulad_data = pd.DataFrame({
        'sex': ['F', 'M', 'F', 'M', 'F'],
        'age_band': ['0-35', '0-35', '35-55', '0-35', '35-55'],
        'imd_band': [20, 40, 60, 10, 80],
        'disability': ['N', 'N', 'Y', 'N', 'N'],
        'prev_attempts': [0, 1, 0, 2, 0],
        'studied_credits': [60, 120, 90, 180, 75],
        'vle_total_clicks': [500, 1200, 300, 2000, 800],
        'final_result': ['Pass', 'Fail', 'Pass', 'Distinction', 'Pass']
    })
    
    print("FeatureBridge Demo")
    print("=" * 30)
    
    # Initialize bridge
    bridge = FeatureBridge()
    
    # Fit on UCI data
    bridge.fit(uci_data, source_type='uci')
    
    # Transform UCI data
    X_uci_transformed = bridge.transform(uci_data, source_type='uci')
    y_uci = bridge.get_target(uci_data, source_type='uci')
    
    print(f"UCI data: {uci_data.shape} -> {X_uci_transformed.shape}")
    print(f"UCI target distribution: {y_uci.value_counts().to_dict()}")
    
    # Transform OULAD data using same bridge
    X_oulad_transformed = bridge.transform(oulad_data, source_type='oulad')
    y_oulad = bridge.get_target(oulad_data, source_type='oulad')
    
    print(f"OULAD data: {oulad_data.shape} -> {X_oulad_transformed.shape}")
    print(f"OULAD target distribution: {y_oulad.value_counts().to_dict()}")
    
    # Show preprocessing summary
    summary = bridge.get_preprocessing_summary()
    print(f"\nPreprocessing Summary:")
    print(f"  Canonical features: {summary['n_canonical_features']}")
    print(f"  Output features: {summary['n_output_features']}")
    print(f"  Numeric features: {summary['numeric_features']}")
    print(f"  Categorical features: {summary['categorical_features']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_feature_bridge()