"""
Main integration script for OULAD deep learning experiments.

This script provides a unified interface for running all deep learning approaches
on the OULAD dataset with comprehensive evaluation and optimization.
"""

import torch
import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our modules
try:
    from .modern_deep_learning import train_modern_deep_learning_models
    from .hyperparameter_optimization import run_comprehensive_optimization
    from .advanced_training_techniques import train_with_advanced_techniques
    from .comprehensive_evaluation import run_comprehensive_evaluation
except ImportError:
    from modern_deep_learning import train_modern_deep_learning_models
    from hyperparameter_optimization import run_comprehensive_optimization
    from advanced_training_techniques import train_with_advanced_techniques
    from comprehensive_evaluation import run_comprehensive_evaluation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oulad_deep_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_oulad_data(data_path: str = 'data/oulad/processed/oulad_ml.csv') -> tuple:
    """
    Load and preprocess OULAD dataset.
    
    Args:
        data_path: Path to OULAD dataset
        
    Returns:
        Tuple of (X, y, sensitive_features, feature_names)
    """
    logger.info(f"Loading OULAD data from {data_path}")
    
    # Try different possible paths
    possible_paths = [
        data_path,
        'data/oulad/processed/oulad_ml.csv',
        'oulad_ml.csv',
        '../oulad_ml.csv',
        '../../oulad_ml.csv'
    ]
    
    df = None
    for path in possible_paths:
        try:
            if Path(path).exists():
                df = pd.read_csv(path)
                logger.info(f"Successfully loaded data from {path}")
                break
        except Exception as e:
            logger.debug(f"Could not load from {path}: {e}")
            continue
    
    if df is None:
        raise FileNotFoundError(f"Could not find OULAD dataset in any of the following paths: {possible_paths}")
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Identify target column (common names for OULAD)
    target_columns = ['final_result', 'result', 'target', 'label', 'outcome', 'pass']
    target_col = None
    
    for col in target_columns:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # If no standard target found, use the last column
        target_col = df.columns[-1]
        logger.warning(f"No standard target column found, using last column: {target_col}")
    
    logger.info(f"Using target column: {target_col}")
    
    # Separate features and target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    # Handle categorical target
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        logger.info(f"Encoded target classes: {le.classes_}")
    
    # Convert binary target if needed
    if len(np.unique(y)) > 2:
        # Convert to binary: pass/fail
        y = (y >= np.median(y)).astype(int)
        logger.info("Converted to binary classification (pass/fail)")
    
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Identify sensitive features (commonly used in OULAD)
    sensitive_feature_names = ['gender', 'age_band', 'region', 'highest_education', 
                              'imd_band', 'disability', 'ethnicity']
    
    sensitive_features = None
    sensitive_col = None
    
    for col in sensitive_feature_names:
        if col in X.columns:
            sensitive_col = col
            break
    
    if sensitive_col:
        sensitive_features = X[sensitive_col].copy()
        if sensitive_features.dtype == 'object':
            le_sensitive = LabelEncoder()
            sensitive_features = le_sensitive.fit_transform(sensitive_features)
        logger.info(f"Using sensitive feature: {sensitive_col}")
    else:
        logger.info("No sensitive features identified")
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        logger.info(f"Encoding categorical columns: {list(categorical_columns)}")
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        logger.info("Filling missing values with median/mode")
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)
    
    # Convert to numpy arrays
    X_array = X.values.astype(np.float32)
    y_array = y.astype(np.int64)
    
    feature_names = list(X.columns)
    
    logger.info(f"Final dataset shape: X={X_array.shape}, y={y_array.shape}")
    logger.info(f"Features: {len(feature_names)}")
    
    return X_array, y_array, sensitive_features, feature_names


def run_basic_training(X_train, y_train, X_test, y_test, output_dir):
    """Run basic training of all modern deep learning models."""
    logger.info("Starting basic training of modern deep learning models...")
    
    results = train_modern_deep_learning_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=42
    )
    
    # Save results
    output_path = Path(output_dir) / 'basic_training'
    output_path.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(output_path / 'basic_training_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Basic training completed. Results saved to {output_path}")
    return results


def run_hyperparameter_optimization(X_train, y_train, X_val, y_val, output_dir, n_trials=50):
    """Run comprehensive hyperparameter optimization."""
    logger.info("Starting hyperparameter optimization...")
    
    optimization_results = run_comprehensive_optimization(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_dir=str(Path(output_dir) / 'optimization'),
        n_trials=n_trials
    )
    
    logger.info("Hyperparameter optimization completed")
    return optimization_results


def run_advanced_training_experiments(X_train, y_train, X_val, y_val, output_dir):
    """Run advanced training techniques experiments."""
    logger.info("Starting advanced training techniques experiments...")
    
    from modern_deep_learning import TabNet, FTTransformer, SAINT
    
    models_to_test = {
        'TabNet': TabNet(input_dim=X_train.shape[1]),
        'FT-Transformer': FTTransformer(input_dim=X_train.shape[1]),
        'SAINT': SAINT(input_dim=X_train.shape[1])
    }
    
    advanced_results = {}
    
    for model_name, model in models_to_test.items():
        logger.info(f"Training {model_name} with advanced techniques...")
        
        results = train_with_advanced_techniques(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            use_pretraining=True,
            use_mixup=True,
            epochs=100
        )
        
        advanced_results[model_name] = results
    
    # Save results
    output_path = Path(output_dir) / 'advanced_training'
    output_path.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(output_path / 'advanced_training_results.pkl', 'wb') as f:
        pickle.dump(advanced_results, f)
    
    logger.info(f"Advanced training experiments completed. Results saved to {output_path}")
    return advanced_results


def main():
    """Main function for running OULAD deep learning experiments."""
    parser = argparse.ArgumentParser(description='OULAD Deep Learning Experiments')
    
    parser.add_argument('--data_path', type=str, default='data/oulad/processed/oulad_ml.csv',
                       help='Path to OULAD dataset')
    parser.add_argument('--output_dir', type=str, default='oulad_deep_learning_results',
                       help='Output directory for results')
    parser.add_argument('--mode', type=str, choices=['basic', 'optimize', 'advanced', 'comprehensive'],
                       default='comprehensive', help='Experiment mode')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of hyperparameter optimization trials')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        X, y, sensitive_features, feature_names = load_oulad_data(args.data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        
        # Further split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=args.random_state, stratify=y_train
        )
        
        logger.info(f"Data splits: Train={X_train_split.shape}, Val={X_val_split.shape}, Test={X_test.shape}")
        
        # Split sensitive features if available
        sensitive_test = None
        if sensitive_features is not None:
            _, sensitive_test, _, _ = train_test_split(
                sensitive_features, y, test_size=args.test_size, random_state=args.random_state, stratify=y
            )
        
        # Run experiments based on mode
        all_results = {}
        
        if args.mode == 'basic':
            # Basic training only
            basic_results = run_basic_training(X_train, y_train, X_test, y_test, output_dir)
            all_results['basic'] = basic_results
            
        elif args.mode == 'optimize':
            # Hyperparameter optimization only
            optimization_results = run_hyperparameter_optimization(
                X_train_split, y_train_split, X_val_split, y_val_split, 
                output_dir, args.n_trials
            )
            all_results['optimization'] = optimization_results
            
        elif args.mode == 'advanced':
            # Advanced training techniques only
            advanced_results = run_advanced_training_experiments(
                X_train_split, y_train_split, X_val_split, y_val_split, output_dir
            )
            all_results['advanced'] = advanced_results
            
        elif args.mode == 'comprehensive':
            # Run comprehensive evaluation
            comprehensive_results = run_comprehensive_evaluation(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                sensitive_features=sensitive_test,
                output_dir=str(output_dir / 'comprehensive'),
                optimize_hyperparameters=True,
                n_optimization_trials=args.n_trials
            )
            all_results['comprehensive'] = comprehensive_results
        
        # Save summary
        summary = {
            'args': vars(args),
            'data_info': {
                'total_samples': len(X),
                'features': len(feature_names),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'class_distribution': np.bincount(y).tolist(),
                'feature_names': feature_names
            },
            'results': all_results
        }
        
        import json
        with open(output_dir / 'experiment_summary.json', 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj
            
            json.dump(summary, f, indent=2, default=convert_numpy)
        
        logger.info(f"Experiments completed successfully! Results saved to {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("OULAD DEEP LEARNING EXPERIMENTS COMPLETED")
        print("="*60)
        print(f"Mode: {args.mode}")
        print(f"Dataset: {args.data_path}")
        print(f"Samples: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})")
        print(f"Features: {len(feature_names)}")
        print(f"Output directory: {output_dir}")
        print("="*60)
        
        if args.mode == 'comprehensive' and 'comprehensive' in all_results:
            comp_results = all_results['comprehensive']
            if 'evaluation_results' in comp_results:
                eval_results = comp_results['evaluation_results']
                print("\nBEST PERFORMERS:")
                
                if 'comparative_analysis' in eval_results and 'best_performers' in eval_results['comparative_analysis']:
                    best_performers = eval_results['comparative_analysis']['best_performers']
                    for metric, (model, score) in best_performers.items():
                        print(f"  {metric.upper()}: {model} ({score:.4f})")
                
                print("\nFor detailed results, check the generated report and visualizations.")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()