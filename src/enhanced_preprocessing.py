"""
Enhanced preprocessing for uplift modeling with proper feature scaling.

Implements:
- StandardScaler only on numeric features
- One-hot encoding for categorical (sparse, unscaled)
- Optional ordinal encoding for ordered categories
- Configurable preprocessing based on dataset and config
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def detect_feature_types(
    X: pd.DataFrame, 
    config: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[str], List[str], Dict[str, List]]:
    """
    Detect numeric, categorical, and ordinal features from dataframe.
    
    Args:
        X: Input dataframe
        config: Preprocessing configuration with optional manual feature lists
        
    Returns:
        Tuple of (numeric_features, categorical_features, ordinal_features, ordinal_categories)
    """
    
    # Check if features are manually specified in config
    if config and config.get('preprocessing'):
        preproc_config = config['preprocessing']
        
        if preproc_config.get('numeric_features') != 'auto':
            numeric_features = preproc_config.get('numeric_features', [])
        else:
            numeric_features = list(X.select_dtypes(include=['int64', 'float64']).columns)
            
        if preproc_config.get('categorical_features') != 'auto':
            categorical_features = preproc_config.get('categorical_features', [])
        else:
            categorical_features = list(X.select_dtypes(include=['object', 'category']).columns)
            
        # Handle ordinal features
        ordinal_features = []
        ordinal_categories = {}
        if preproc_config.get('ordinal_features'):
            for feature_info in preproc_config['ordinal_features']:
                if isinstance(feature_info, dict):
                    for col, categories in feature_info.items():
                        if col in X.columns:
                            ordinal_features.append(col)
                            ordinal_categories[col] = categories
    else:
        # Auto-detect feature types
        numeric_features = []
        categorical_features = []
        ordinal_features = []
        ordinal_categories = {}
        
        # Dataset-specific feature detection
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Check if it's really ordinal (small number of unique integer values)
                unique_vals = X[col].nunique()
                if X[col].dtype == 'int64' and unique_vals <= 10 and unique_vals > 2:
                    # Potential ordinal feature - check if values are sequential
                    unique_sorted = sorted(X[col].dropna().unique())
                    if len(unique_sorted) > 1 and all(
                        unique_sorted[i] == unique_sorted[i-1] + 1 
                        for i in range(1, len(unique_sorted))
                    ):
                        # Sequential integers - treat as ordinal
                        ordinal_features.append(col)
                        ordinal_categories[col] = unique_sorted
                    else:
                        numeric_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        # Handle known ordinal features for UCI dataset
        uci_ordinal_mappings = {
            "studytime": [1, 2, 3, 4],
            "Dalc": [1, 2, 3, 4, 5],
            "Walc": [1, 2, 3, 4, 5],
            "famrel": [1, 2, 3, 4, 5],
            "freetime": [1, 2, 3, 4, 5],
            "goout": [1, 2, 3, 4, 5],
            "health": [1, 2, 3, 4, 5],
            "Medu": [0, 1, 2, 3, 4],
            "Fedu": [0, 1, 2, 3, 4],
        }
        
        for col, categories in uci_ordinal_mappings.items():
            if col in X.columns and col in numeric_features:
                numeric_features.remove(col)
                ordinal_features.append(col)
                ordinal_categories[col] = categories
    
    # Remove ordinal features from numeric/categorical lists
    numeric_features = [col for col in numeric_features if col not in ordinal_features]
    categorical_features = [col for col in categorical_features if col not in ordinal_features]
    
    logger.info(f"Feature detection: {len(numeric_features)} numeric, "
                f"{len(categorical_features)} categorical, {len(ordinal_features)} ordinal")
    
    return numeric_features, categorical_features, ordinal_features, ordinal_categories


def create_preprocessing_pipeline(
    X: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    sparse_categorical: bool = True
) -> ColumnTransformer:
    """
    Create preprocessing pipeline with proper feature scaling.
    
    Args:
        X: Input dataframe for feature detection
        config: Preprocessing configuration
        sparse_categorical: Whether to keep categorical features sparse
        
    Returns:
        ColumnTransformer for preprocessing
    """
    
    # Detect feature types
    numeric_features, categorical_features, ordinal_features, ordinal_categories = \
        detect_feature_types(X, config)
    
    transformers = []
    
    # Numeric features: StandardScaler with imputation
    if numeric_features:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_transformer, numeric_features))
    
    # Categorical features: OneHot (sparse, unscaled)
    if categorical_features:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(
                drop='first',  # Avoid multicollinearity
                sparse_output=sparse_categorical,
                handle_unknown='ignore'
            ))
        ])
        transformers.append(('categorical', categorical_transformer, categorical_features))
    
    # Ordinal features: OrdinalEncoder with imputation
    if ordinal_features:
        # Create list of categories in correct order
        ordinal_cats = [ordinal_categories[col] for col in ordinal_features]
        
        ordinal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(
                categories=ordinal_cats,
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ))
        ])
        transformers.append(('ordinal', ordinal_transformer, ordinal_features))
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any unspecified columns
        sparse_threshold=0.3 if sparse_categorical else 0.0,  # Control sparsity
        n_jobs=-1
    )
    
    logger.info(f"Created preprocessing pipeline with {len(transformers)} transformers")
    
    return preprocessor


def create_uplift_pipeline(
    X: pd.DataFrame,
    model,
    config: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Create complete pipeline for uplift modeling.
    
    Args:
        X: Input dataframe for preprocessing setup
        model: Model instance to use
        config: Configuration dictionary
        
    Returns:
        Complete pipeline with preprocessing and model
    """
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X, config)
    
    # Create complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline


def get_feature_names(preprocessor: ColumnTransformer, input_features: List[str]) -> List[str]:
    """
    Get feature names after preprocessing transformation.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        input_features: Original feature names
        
    Returns:
        List of transformed feature names
    """
    feature_names = []
    
    try:
        # Get feature names from each transformer
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder':
                continue
                
            if hasattr(transformer, 'get_feature_names_out'):
                # Newer sklearn versions
                trans_features = transformer.get_feature_names_out(columns)
            elif hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                # For OneHotEncoder in pipeline
                trans_features = transformer.named_steps['onehot'].get_feature_names_out(columns)
            elif hasattr(transformer.named_steps['onehot'], 'get_feature_names'):
                # Older sklearn versions
                trans_features = transformer.named_steps['onehot'].get_feature_names(columns)
            else:
                # Fallback for numeric/ordinal features
                trans_features = columns
            
            feature_names.extend(trans_features)
            
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        feature_names = [f"feature_{i}" for i in range(preprocessor.n_features_in_)]
    
    return feature_names


def validate_preprocessing(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate preprocessing pipeline and return diagnostics.
    
    Args:
        X: Input dataframe
        preprocessor: Fitted preprocessing pipeline
        config: Configuration used
        
    Returns:
        Dictionary with validation results
    """
    
    # Transform a small sample to check output
    X_sample = X.head(100)
    X_transformed = preprocessor.transform(X_sample)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor, list(X.columns))
    
    validation_results = {
        'input_shape': X.shape,
        'output_shape': X_transformed.shape,
        'n_features_in': preprocessor.n_features_in_,
        'n_features_out': X_transformed.shape[1],
        'is_sparse': hasattr(X_transformed, 'toarray'),
        'feature_names': feature_names[:10],  # First 10 for brevity
        'transformers': [name for name, _, _ in preprocessor.transformers_],
        'memory_usage_mb': X_transformed.nbytes / (1024 * 1024) if hasattr(X_transformed, 'nbytes') else 'unknown'
    }
    
    logger.info(f"Preprocessing validation: {X.shape} -> {X_transformed.shape}")
    
    return validation_results