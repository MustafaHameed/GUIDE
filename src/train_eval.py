"""
Training and Fairness Evaluation Pipeline

Implements training pipeline with multiple models, calibration, and fairness
mitigation techniques including preprocessing reweighing and postprocessing
threshold optimization.

References:
- Fairlearn postprocessing: https://fairlearn.org/v0.10/user_guide/mitigation/postprocessing.html
- Fairlearn equalized odds: https://fairlearn.org/v0.10/user_guide/fairness_in_machine_learning.html
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, brier_score_loss,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Fairness imports
from fairlearn.metrics import (
    MetricFrame, 
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
    selection_rate
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.preprocessing import CorrelationRemover
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

# Optional imports for additional models
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_oulad_data(dataset_path: Path, split_path: Path) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Load OULAD dataset and split definitions.
    
    Args:
        dataset_path: Path to OULAD parquet file
        split_path: Path to split JSON file
        
    Returns:
        Tuple of (dataset, split_dict)
    """
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    logger.info(f"Loading splits from {split_path}")
    with open(split_path, 'r') as f:
        splits = json.load(f)
    
    return df, splits


def create_model(model_type: str, **kwargs) -> Any:
    """Create model instance.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000, **kwargs),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100, **kwargs),
    }
    
    if HAS_XGB:
        models['xgboost'] = xgb.XGBClassifier(random_state=42, **kwargs)
    
    if HAS_LGB:
        models['lightgbm'] = lgb.LGBMClassifier(random_state=42, **kwargs)
    
    if model_type not in models:
        available = list(models.keys())
        raise ValueError(f"Model {model_type} not available. Choose from: {available}")
    
    return models[model_type]


def preprocess_reweighing(X_train: pd.DataFrame, y_train: pd.Series, 
                         sensitive_features: pd.Series) -> np.ndarray:
    """Apply AIF360 reweighing preprocessing.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        sensitive_features: Sensitive attribute values
        
    Returns:
        Sample weights for training
    """
    logger.info("Applying preprocessing reweighing...")
    
    # Prepare data for AIF360
    df_train = X_train.copy()
    df_train['y'] = y_train
    df_train['sensitive'] = sensitive_features
    
    # Create AIF360 dataset
    dataset = BinaryLabelDataset(
        df=df_train,
        label_names=['y'],
        protected_attribute_names=['sensitive']
    )
    
    # Apply reweighing
    reweighing = Reweighing(
        unprivileged_groups=[{'sensitive': 0}],
        privileged_groups=[{'sensitive': 1}]
    )
    
    reweighed_dataset = reweighing.fit_transform(dataset)
    
    return reweighed_dataset.instance_weights


def apply_postprocessing(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, sensitive_features_train: pd.Series,
                        sensitive_features_test: pd.Series, constraint: str = 'equalized_odds') -> Tuple[np.ndarray, np.ndarray]:
    """Apply Fairlearn postprocessing threshold optimization.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features  
        sensitive_features_train: Training sensitive features
        sensitive_features_test: Test sensitive features
        constraint: Fairness constraint ('equalized_odds' or 'demographic_parity')
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    logger.info(f"Applying postprocessing with {constraint} constraint...")
    
    # Create threshold optimizer
    postprocessor = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        prefit=True
    )
    
    # Fit postprocessor
    postprocessor.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    
    # Make fair predictions
    y_pred = postprocessor.predict(X_test, sensitive_features=sensitive_features_test)
    
    # Get probabilities if available
    try:
        y_prob = postprocessor.predict_proba(X_test, sensitive_features=sensitive_features_test)[:, 1]
    except:
        y_prob = None
    
    return y_pred, y_prob


def calculate_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              sensitive_features: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive fairness metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values
        
    Returns:
        Dictionary of fairness metrics
    """
    metrics = {}
    
    # Create MetricFrame for group-wise metrics
    mf = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'tpr': true_positive_rate,
            'fpr': false_positive_rate,
            'selection_rate': selection_rate
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Demographic parity difference
    metrics['demographic_parity_diff'] = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    
    # Equalized odds difference
    metrics['equalized_odds_diff'] = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )
    
    # TPR and FPR gaps
    tpr_by_group = mf.by_group['tpr']
    fpr_by_group = mf.by_group['fpr']
    
    metrics['tpr_gap'] = tpr_by_group.max() - tpr_by_group.min()
    metrics['fpr_gap'] = fpr_by_group.max() - fpr_by_group.min()
    
    # Worst-group error (1 - accuracy)
    accuracy_by_group = mf.by_group['accuracy']
    metrics['worst_group_error'] = 1 - accuracy_by_group.min()
    
    return metrics


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray],
                     sensitive_features: pd.Series, student_ids: pd.Series, 
                     n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
    """Calculate bootstrap confidence intervals for metrics.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        y_prob: Prediction probabilities  
        sensitive_features: Sensitive attributes
        student_ids: Student identifiers for proper resampling
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with mean and CI for each metric
    """
    logger.info(f"Computing bootstrap CIs with {n_bootstrap} resamples...")
    
    metrics_list = []
    unique_students = student_ids.unique()
    
    for i in range(n_bootstrap):
        # Resample at student level
        bootstrap_students = np.random.choice(unique_students, size=len(unique_students), replace=True)
        bootstrap_mask = student_ids.isin(bootstrap_students)
        
        # Calculate metrics for this bootstrap sample
        sample_metrics = {
            'accuracy': accuracy_score(y_true[bootstrap_mask], y_pred[bootstrap_mask]),
            'auc': roc_auc_score(y_true[bootstrap_mask], y_prob[bootstrap_mask]) if y_prob is not None else np.nan,
            'f1': f1_score(y_true[bootstrap_mask], y_pred[bootstrap_mask]),
        }
        
        if y_prob is not None:
            sample_metrics['brier'] = brier_score_loss(y_true[bootstrap_mask], y_prob[bootstrap_mask])
            sample_metrics['ece'] = expected_calibration_error(y_true[bootstrap_mask], y_prob[bootstrap_mask])
        
        # Add fairness metrics
        fairness_metrics = calculate_fairness_metrics(
            y_true[bootstrap_mask], 
            y_pred[bootstrap_mask],
            sensitive_features[bootstrap_mask]
        )
        sample_metrics.update(fairness_metrics)
        
        metrics_list.append(sample_metrics)
    
    # Calculate confidence intervals
    results = {}
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list if not np.isnan(m[metric])]
        if values:
            results[metric] = {
                'mean': np.mean(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'std': np.std(values)
            }
    
    return results


def train_and_evaluate_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            sensitive_features_train: pd.Series, sensitive_features_test: pd.Series,
                            student_ids_test: pd.Series,
                            fairness_method: str = 'none', calibration_method: str = 'none',
                            results_dir: Path = None) -> Dict[str, Any]:
    """Train and evaluate a single model with fairness considerations.
    
    Args:
        model_type: Type of model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        sensitive_features_train/test: Sensitive attributes
        student_ids_test: Student IDs for bootstrap resampling
        fairness_method: Fairness mitigation method
        calibration_method: Calibration method
        results_dir: Directory to save results
        
    Returns:
        Dictionary with all evaluation results
    """
    logger.info(f"Training {model_type} with fairness method: {fairness_method}")
    
    results = {
        'model_type': model_type,
        'fairness_method': fairness_method,
        'calibration_method': calibration_method
    }
    
    # Create and train model
    model = create_model(model_type)
    
    # Apply preprocessing reweighing if requested
    sample_weight = None
    if fairness_method == 'reweighing':
        # Convert sensitive features to binary for AIF360
        sensitive_binary = (sensitive_features_train == sensitive_features_train.mode()[0]).astype(int)
        sample_weight = preprocess_reweighing(X_train, y_train, sensitive_binary)
    
    # Train model
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    
    # Apply calibration if requested
    if calibration_method in ['platt', 'isotonic']:
        logger.info(f"Applying {calibration_method} calibration...")
        calibrated_model = CalibratedClassifierCV(model, method=calibration_method, cv=3)
        calibrated_model.fit(X_train, y_train)
        model = calibrated_model
    
    # Make initial predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Apply postprocessing if requested
    if fairness_method in ['equalized_odds', 'demographic_parity']:
        y_pred, y_prob_post = apply_postprocessing(
            model, X_train, y_train, X_test,
            sensitive_features_train, sensitive_features_test,
            constraint=fairness_method
        )
        if y_prob_post is not None:
            y_prob = y_prob_post
    
    # Calculate basic metrics
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['f1'] = f1_score(y_test, y_pred)
    
    if y_prob is not None:
        results['auc'] = roc_auc_score(y_test, y_prob)
        results['brier'] = brier_score_loss(y_test, y_prob)
        results['ece'] = expected_calibration_error(y_test, y_prob)
    
    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(y_test, y_pred, sensitive_features_test)
    results.update(fairness_metrics)
    
    # Bootstrap confidence intervals
    bootstrap_results = bootstrap_metrics(
        y_test, y_pred, y_prob, sensitive_features_test, student_ids_test
    )
    results['bootstrap'] = bootstrap_results
    
    # Save reliability plot if probabilities available
    if y_prob is not None and results_dir:
        save_reliability_plot(y_test, y_prob, results_dir, f"{model_type}_{fairness_method}")
    
    return results


def save_reliability_plot(y_true: np.ndarray, y_prob: np.ndarray, 
                         output_dir: Path, filename_prefix: str) -> None:
    """Save reliability diagram for probability calibration.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_dir: Output directory
        filename_prefix: Prefix for filename
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate bin statistics
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            confidence_in_bin = y_prob[in_bin].mean()
            
            bin_centers.append(confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(prop_in_bin)
    
    # Plot reliability diagram
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.scatter(bin_centers, bin_accuracies, s=bin_counts, alpha=0.7, label='Observed')
    
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Reliability Diagram - {filename_prefix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'reliability_{filename_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """CLI interface for training and evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train models with fairness evaluation')
    parser.add_argument('--config', type=Path, help='Path to JSON config with dataset and split paths')
    parser.add_argument('--dataset', type=Path, default='data/oulad/processed/oulad_ml.parquet')
    parser.add_argument('--split', type=Path, default='data/oulad/splits/random_split.json')
    parser.add_argument('--results-dir', type=Path, default='results/oulad')
    parser.add_argument('--figures-dir', type=Path, default='figures/oulad')
    parser.add_argument('--models', nargs='+', default=['logistic', 'random_forest'])
    parser.add_argument('--fairness-methods', nargs='+', default=['none', 'reweighing', 'equalized_odds'])
    parser.add_argument('--sensitive-attr', default='sex', help='Sensitive attribute column')

    args = parser.parse_args()

    # Allow configuration file to override paths
    if args.config and args.config.exists():
        with open(args.config, 'r') as f:
            cfg = json.load(f)
        args.dataset = Path(cfg.get('dataset', args.dataset))
        args.split = Path(cfg.get('split', args.split))

    # Load data
    df, splits = load_oulad_data(args.dataset, args.split)
    
    # Prepare train/test splits
    train_mask = df['id_student'].isin(splits['train'])
    test_mask = df['id_student'].isin(splits['test'])
    
    X_train = df[train_mask].drop(columns=['id_student', 'label_pass', 'label_fail_or_withdraw'] + 
                                         [args.sensitive_attr])
    y_train = df[train_mask]['label_pass']
    sensitive_train = df[train_mask][args.sensitive_attr]
    
    X_test = df[test_mask].drop(columns=['id_student', 'label_pass', 'label_fail_or_withdraw'] + 
                                       [args.sensitive_attr])
    y_test = df[test_mask]['label_pass']
    sensitive_test = df[test_mask][args.sensitive_attr]
    student_ids_test = df[test_mask]['id_student']
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train all model combinations
    all_results = []
    
    for model_type in args.models:
        for fairness_method in args.fairness_methods:
            try:
                results = train_and_evaluate_model(
                    model_type, X_train, y_train, X_test, y_test,
                    sensitive_train, sensitive_test, student_ids_test,
                    fairness_method=fairness_method,
                    results_dir=args.figures_dir
                )
                all_results.append(results)
                
                logger.info(f"Completed {model_type} + {fairness_method}")
                logger.info(f"Accuracy: {results['accuracy']:.3f}, "
                           f"EO Diff: {results.get('equalized_odds_diff', 'N/A'):.3f}")
                
            except Exception as e:
                logger.error(f"Failed for {model_type} + {fairness_method}: {e}")
    
    # Save results
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'bootstrap'} 
        for r in all_results
    ])
    results_df.to_csv(args.results_dir / 'model_results.csv', index=False)
    
    logger.info(f"Results saved to {args.results_dir}")
    logger.info("Training and evaluation completed!")


if __name__ == '__main__':
    main()