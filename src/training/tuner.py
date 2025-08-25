"""
Hyperparameter tuning module for uplift modeling.

Implements randomized/grid search with proper cross-validation splits
for UCI (StratifiedKFold) and OULAD (GroupKFold) datasets.
"""

import logging
import yaml
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from splits import get_cv_splitter, encode_labels
from logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(config: Dict[str, Any]) -> ColumnTransformer:
    """
    Create preprocessing pipeline with proper feature scaling.
    
    Args:
        config: Preprocessing configuration
        
    Returns:
        ColumnTransformer for preprocessing
    """
    # For now, use auto-detection approach
    # In a real implementation, this would be dataset-specific
    
    # Numeric features: StandardScaler
    numeric_transformer = StandardScaler()
    
    # Categorical features: OneHot (sparse, unscaled)
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=True)
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, []),  # Will be filled by detect_feature_types
            ('cat', categorical_transformer, [])  # Will be filled by detect_feature_types
        ],
        remainder='passthrough'  # Pass through any remaining columns
    )
    
    return preprocessor


def detect_feature_types(X: pd.DataFrame) -> tuple[List[str], List[str]]:
    """
    Auto-detect numeric and categorical features.
    
    Args:
        X: Feature dataframe
        
    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    numeric_cols = []
    categorical_cols = []
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (small number of unique values)
            if X[col].nunique() <= 10 and X[col].dtype == 'int64':
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    logger.info(f"Detected {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
    return numeric_cols, categorical_cols


def create_model_pipeline(model_name: str, model_config: Dict[str, Any], 
                         preprocessor: ColumnTransformer) -> Pipeline:
    """
    Create model pipeline with preprocessing.
    
    Args:
        model_name: Name of the model
        model_config: Model configuration
        preprocessor: Preprocessing pipeline
        
    Returns:
        Complete pipeline
    """
    if model_name == 'logistic_regression':
        model = LogisticRegression(
            class_weight=model_config.get('class_weight', 'balanced'),
            random_state=42,
            max_iter=1000
        )
    elif model_name == 'random_forest':
        model = RandomForestClassifier(
            class_weight=model_config.get('class_weight', 'balanced'),
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline


def run_tuning(config: Dict[str, Any], output_dir: Path, n_jobs: int = -1) -> Dict[str, Any]:
    """
    Run hyperparameter tuning based on configuration.
    
    Args:
        config: Training configuration
        output_dir: Directory to save results
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with tuning results
    """
    logger.info("Starting hyperparameter tuning...")
    
    # For now, this is a stub that creates placeholder results
    # In a real implementation, this would:
    # 1. Load the dataset based on config['dataset']['name']
    # 2. Create appropriate CV splitter
    # 3. Run RandomizedSearchCV/GridSearchCV
    # 4. Save best models and parameters
    
    dataset_name = config['dataset']['name']
    logger.info(f"Dataset: {dataset_name}")
    
    # Create placeholder results
    results = {
        'dataset': dataset_name,
        'cv_strategy': config['dataset']['cv']['strategy'],
        'models_tuned': list(config['models'].keys()),
        'best_params': {},
        'best_scores': {},
        'status': 'completed_stub'
    }
    
    # Save placeholder results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration used
    config_path = output_dir / 'tuning_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save results
    results_path = output_dir / 'tuning_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"Tuning results saved to {output_dir}")
    logger.info("NOTE: This is currently a stub implementation")
    
    return results


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset based on name.
    
    Args:
        dataset_name: Name of dataset to load
        
    Returns:
        Tuple of (features, targets)
    """
    if dataset_name.lower() == 'uci':
        # Load UCI student performance data
        try:
            df = pd.read_csv('student-mat.csv', sep=';')
            # Create binary target (pass/fail)
            y = (df['G3'] >= 10).astype(int)  # Pass if final grade >= 10
            X = df.drop(['G1', 'G2', 'G3'], axis=1)  # Remove grade columns
            return X, y
        except FileNotFoundError:
            logger.error("UCI dataset not found. Please ensure student-mat.csv exists.")
            raise
    
    elif dataset_name.lower() == 'oulad':
        # Load OULAD data
        try:
            # This would load from processed OULAD data
            # For now, create a placeholder
            logger.warning("OULAD dataset loading not implemented yet")
            raise NotImplementedError("OULAD dataset loading pending")
        except Exception as e:
            logger.error(f"Failed to load OULAD dataset: {e}")
            raise
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    setup_logging()
    
    # Example usage
    config_path = Path("configs/train_uplift.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        output_dir = Path("models/tuning_test")
        results = run_tuning(config, output_dir)
        print(f"Tuning completed: {results}")
    else:
        print(f"Configuration file not found: {config_path}")