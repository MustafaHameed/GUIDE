"""
Metrics and plotting utilities for uplift modeling.

Handles persistence of metrics to tables/ and plots to figures/
with proper organization and confidence intervals.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import json

logger = logging.getLogger(__name__)


def ensure_output_dirs(base_dir: Path = Path(".")) -> Dict[str, Path]:
    """
    Ensure output directories exist and return paths.
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Dictionary with directory paths
    """
    dirs = {
        'tables': base_dir / 'tables',
        'figures': base_dir / 'figures',
        'reports': base_dir / 'reports',
        'models': base_dir / 'models'
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
    
    return dirs


def save_metrics_with_ci(
    metrics: Dict[str, float],
    bootstrap_results: Optional[Dict[str, Dict[str, float]]] = None,
    output_path: Path = None,
    experiment_name: str = "experiment"
) -> None:
    """
    Save metrics with confidence intervals to CSV.
    
    Args:
        metrics: Dictionary of metric name -> value
        bootstrap_results: Bootstrap results with CIs
        output_path: Path to save CSV file
        experiment_name: Name of the experiment
    """
    if output_path is None:
        output_path = Path("tables") / f"{experiment_name}_metrics.csv"
    
    # Create metrics dataframe
    metrics_data = []
    
    for metric_name, value in metrics.items():
        row = {
            'experiment': experiment_name,
            'metric': metric_name,
            'value': value
        }
        
        # Add confidence intervals if available
        if bootstrap_results and metric_name in bootstrap_results:
            ci_data = bootstrap_results[metric_name]
            row.update({
                'ci_lower': ci_data.get('ci_lower', np.nan),
                'ci_upper': ci_data.get('ci_upper', np.nan),
                'std': ci_data.get('std', np.nan)
            })
        
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Metrics saved to {output_path}")


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path = Path("figures"),
    experiment_name: str = "experiment",
    pos_label: int = 1
) -> None:
    """
    Plot and save ROC and Precision-Recall curves.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        output_dir: Directory to save plots
        experiment_name: Name of experiment for filename
        pos_label: Positive class label
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
    roc_auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(12, 5))
    
    # ROC subplot
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {experiment_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
    pr_auc = np.trapz(precision, recall)
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})', linewidth=2)
    plt.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, 
                label=f'Baseline ({np.mean(y_true):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {experiment_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_name}_roc_pr_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC/PR curves saved to {output_dir}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path = Path("figures"),
    experiment_name: str = "experiment",
    n_bins: int = 10
) -> None:
    """
    Plot calibration curve to assess probability calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        output_dir: Directory to save plot
        experiment_name: Name of experiment
        n_bins: Number of bins for calibration
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=2, 
             label=f"Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", alpha=0.5)
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {experiment_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_name}_calibration.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Calibration curve saved to {output_dir}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path = Path("figures"),
    experiment_name: str = "experiment",
    normalize: str = 'true'
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save plot
        experiment_name: Name of experiment
        normalize: Normalization option ('true', 'pred', 'all', None)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                cmap='Blues', square=True, cbar_kws={'shrink': 0.8})
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {experiment_name}')
    
    # Add class labels
    labels = ['Not At-Risk (0)', 'At-Risk (1)']
    plt.gca().set_xticklabels(labels)
    plt.gca().set_yticklabels(labels)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {output_dir}")


def create_model_comparison_table(
    results_list: List[Dict[str, Any]],
    output_path: Path = Path("tables/model_comparison.csv")
) -> pd.DataFrame:
    """
    Create comparison table across different models/experiments.
    
    Args:
        results_list: List of result dictionaries
        output_path: Path to save comparison table
        
    Returns:
        Comparison dataframe
    """
    comparison_data = []
    
    for result in results_list:
        row = {
            'experiment': result.get('experiment_name', 'unknown'),
            'model': result.get('model_name', 'unknown'),
            'dataset': result.get('dataset', 'unknown'),
            'cv_strategy': result.get('cv_strategy', 'unknown'),
        }
        
        # Add metrics
        metrics = result.get('metrics', {})
        for metric_name, value in metrics.items():
            row[metric_name] = value
        
        # Add confidence intervals if available
        bootstrap = result.get('bootstrap', {})
        for metric_name, ci_data in bootstrap.items():
            if isinstance(ci_data, dict):
                row[f"{metric_name}_ci_lower"] = ci_data.get('ci_lower')
                row[f"{metric_name}_ci_upper"] = ci_data.get('ci_upper')
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Model comparison table saved to {output_path}")
    
    return df


def save_experiment_config(
    config: Dict[str, Any],
    output_path: Path = None,
    experiment_name: str = "experiment"
) -> None:
    """
    Save experiment configuration for reproducibility.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
        experiment_name: Name of experiment
    """
    if output_path is None:
        output_path = Path("reports") / f"{experiment_name}_config.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Experiment config saved to {output_path}")


def create_experiment_summary(
    metrics: Dict[str, float],
    config: Dict[str, Any],
    model_info: Dict[str, Any],
    output_path: Path = None,
    experiment_name: str = "experiment"
) -> None:
    """
    Create comprehensive experiment summary report.
    
    Args:
        metrics: Final metrics
        config: Configuration used
        model_info: Model information
        output_path: Path to save summary
        experiment_name: Name of experiment
    """
    if output_path is None:
        output_path = Path("reports") / f"{experiment_name}_summary.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create markdown summary
    summary = f"""# Experiment Summary: {experiment_name}

## Configuration
- Dataset: {config.get('dataset', {}).get('name', 'unknown')}
- CV Strategy: {config.get('dataset', {}).get('cv', {}).get('strategy', 'unknown')}
- Positive Label: {config.get('dataset', {}).get('pos_label', 1)}

## Model Information
- Model Type: {model_info.get('model_type', 'unknown')}
- Parameters: {model_info.get('best_params', 'default')}

## Performance Metrics
"""
    
    for metric_name, value in metrics.items():
        summary += f"- {metric_name}: {value:.4f}\n"
    
    summary += f"""
## Files Generated
- Metrics: tables/{experiment_name}_metrics.csv
- ROC/PR Curves: figures/{experiment_name}_roc_pr_curves.png
- Calibration: figures/{experiment_name}_calibration.png
- Confusion Matrix: figures/{experiment_name}_confusion_matrix.png
- Configuration: reports/{experiment_name}_config.json

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Experiment summary saved to {output_path}")


def persist_all_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    metrics: Dict[str, float],
    config: Dict[str, Any],
    model_info: Dict[str, Any],
    bootstrap_results: Optional[Dict[str, Dict[str, float]]] = None,
    experiment_name: str = "experiment",
    base_dir: Path = Path(".")
) -> None:
    """
    Persist all artifacts (metrics, plots, configs) for an experiment.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        metrics: Final metrics
        config: Configuration used
        model_info: Model information
        bootstrap_results: Bootstrap confidence intervals
        experiment_name: Name of experiment
        base_dir: Base directory for outputs
    """
    # Ensure output directories exist
    dirs = ensure_output_dirs(base_dir)
    
    # Save metrics
    save_metrics_with_ci(
        metrics, bootstrap_results, 
        dirs['tables'] / f"{experiment_name}_metrics.csv",
        experiment_name
    )
    
    # Save plots if probabilities are available
    if y_prob is not None:
        plot_roc_pr_curves(y_true, y_prob, dirs['figures'], experiment_name)
        plot_calibration_curve(y_true, y_prob, dirs['figures'], experiment_name)
    
    # Save confusion matrix
    plot_confusion_matrix(y_true, y_pred, dirs['figures'], experiment_name)
    
    # Save configuration
    save_experiment_config(config, dirs['reports'] / f"{experiment_name}_config.json", experiment_name)
    
    # Save summary
    create_experiment_summary(
        metrics, config, model_info,
        dirs['reports'] / f"{experiment_name}_summary.md",
        experiment_name
    )
    
    logger.info(f"All artifacts persisted for experiment: {experiment_name}")


# Example usage function
def example_usage():
    """Example of how to use the metrics persistence utilities."""
    
    # Simulate some results
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_prob = np.random.beta(2, 5, n_samples)  # Skewed probabilities
    y_pred = (y_prob > 0.5).astype(int)
    
    metrics = {
        'accuracy': np.mean(y_true == y_pred),
        'auc': 0.75,
        'f1': 0.65,
        'precision': 0.70,
        'recall': 0.60
    }
    
    config = {
        'dataset': {'name': 'uci', 'pos_label': 1},
        'models': {'logistic_regression': {'C': 1.0}}
    }
    
    model_info = {
        'model_type': 'logistic_regression',
        'best_params': {'C': 1.0}
    }
    
    # Persist all artifacts
    persist_all_artifacts(
        y_true, y_pred, y_prob, metrics, config, model_info,
        experiment_name="example_experiment"
    )


if __name__ == "__main__":
    example_usage()