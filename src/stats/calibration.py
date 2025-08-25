"""Model calibration analysis with reliability curves and Expected Calibration Error (ECE)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import sklearn for calibration tools
try:
    from sklearn.calibration import calibration_curve, CalibratedClassifierCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        strategy: Binning strategy ('uniform' or 'quantile')
        
    Returns:
        Expected calibration error
    """
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    elif strategy == 'quantile':
        bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    else:
        raise ValueError("Strategy must be 'uniform' or 'quantile'")
    
    ece = 0.0
    total_samples = len(y_prob)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        if bin_lower == 0:  # Include left boundary for first bin
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Weighted contribution to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        strategy: Binning strategy ('uniform' or 'quantile')
        
    Returns:
        Maximum calibration error
    """
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    elif strategy == 'quantile':
        bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    else:
        raise ValueError("Strategy must be 'uniform' or 'quantile'")
    
    max_error = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        if bin_lower == 0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            # Accuracy and confidence in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Update maximum error
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_error = max(max_error, bin_error)
    
    return max_error


def brier_score_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Decompose Brier score into reliability, resolution, and uncertainty.
    
    Brier Score = Reliability - Resolution + Uncertainty
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary with decomposed components
    """
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    # Base rate (overall frequency of positive class)
    base_rate = np.mean(y_true)
    uncertainty = base_rate * (1 - base_rate)
    
    # Group predictions by unique probability values
    unique_probs = np.unique(y_prob)
    reliability = 0.0
    resolution = 0.0
    
    for prob in unique_probs:
        # Find samples with this probability
        mask = (y_prob == prob)
        n_k = mask.sum()
        
        if n_k > 0:
            # Observed frequency for this probability
            o_k = y_true[mask].mean()
            
            # Contribution to reliability (calibration error)
            reliability += n_k * (prob - o_k) ** 2
            
            # Contribution to resolution (discrimination ability)
            resolution += n_k * (o_k - base_rate) ** 2
    
    n_total = len(y_true)
    reliability /= n_total
    resolution /= n_total
    
    return {
        'brier_score': brier_score,
        'reliability': reliability,
        'resolution': resolution, 
        'uncertainty': uncertainty,
        'verification': reliability - resolution + uncertainty  # Should equal Brier score
    }


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform',
    title: str = "Reliability Diagram",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create reliability diagram (calibration plot).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        strategy: Binning strategy ('uniform' or 'quantile')
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Figure object and calibration metrics
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for reliability diagrams")
    
    # Create calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )
    
    # Calculate calibration metrics
    ece = expected_calibration_error(y_true, y_prob, n_bins, strategy)
    mce = maximum_calibration_error(y_true, y_prob, n_bins, strategy)
    brier_decomp = brier_score_decomposition(y_true, y_prob)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Main reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', 
             label=f'Model (ECE: {ece:.3f})', markersize=8)
    
    # Add bin counts as bar chart background
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    else:
        bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    
    bin_counts = []
    for i in range(n_bins):
        if i == 0:
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        else:
            in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        bin_counts.append(in_bin.sum())
    
    # Normalize bin counts for visualization
    max_count = max(bin_counts) if bin_counts else 1
    normalized_counts = [count / max_count * 0.1 for count in bin_counts]
    
    # Add histogram as background
    ax1_twin = ax1.twinx()
    # Ensure we have the right number of bins for the bar chart
    if len(normalized_counts) == len(mean_predicted_value):
        ax1_twin.bar(mean_predicted_value, normalized_counts, alpha=0.3, 
                     width=0.08, color='gray', label='Sample density')
        ax1_twin.set_ylabel('Normalized sample density', alpha=0.7)
        ax1_twin.set_ylim(0, 0.15)
    
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Calibration error by bin
    calibration_errors = np.abs(mean_predicted_value - fraction_of_positives)
    ax2.bar(range(len(calibration_errors)), calibration_errors, 
            alpha=0.7, color='red')
    ax2.set_xlabel('Bin')
    ax2.set_ylabel('Calibration Error')
    ax2.set_title('Calibration Error by Bin')
    ax2.grid(True, alpha=0.3)
    
    # Add text with metrics
    metrics_text = f"""Calibration Metrics:
ECE: {ece:.4f}
MCE: {mce:.4f}
Brier Score: {brier_decomp['brier_score']:.4f}
Reliability: {brier_decomp['reliability']:.4f}
Resolution: {brier_decomp['resolution']:.4f}"""
    
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Reliability diagram saved to {save_path}")
    
    calibration_metrics = {
        'ece': ece,
        'mce': mce,
        'brier_decomposition': brier_decomp,
        'mean_predicted_value': mean_predicted_value,
        'fraction_of_positives': fraction_of_positives,
        'bin_counts': bin_counts
    }
    
    return fig, calibration_metrics


def confidence_histogram(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 20,
    title: str = "Confidence Histogram", 
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create confidence histogram showing distribution of predicted probabilities.
    
    Args:
        y_prob: Predicted probabilities
        y_true: True binary labels  
        n_bins: Number of bins for histogram
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate probabilities by true class
    prob_positive = y_prob[y_true == 1]
    prob_negative = y_prob[y_true == 0]
    
    # Create histogram
    bins = np.linspace(0, 1, n_bins + 1)
    ax.hist(prob_negative, bins=bins, alpha=0.6, label='Negative class', 
            color='red', density=True)
    ax.hist(prob_positive, bins=bins, alpha=0.6, label='Positive class',
            color='blue', density=True)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add vertical lines for key statistics
    ax.axvline(np.mean(y_prob), color='black', linestyle='--', 
               label=f'Mean: {np.mean(y_prob):.3f}')
    ax.axvline(np.mean(y_true), color='green', linestyle='--',
               label=f'Base rate: {np.mean(y_true):.3f}')
    
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confidence histogram saved to {save_path}")
    
    return fig


def calibrate_model_predictions(
    y_true_train: np.ndarray,
    y_prob_train: np.ndarray,
    y_prob_test: np.ndarray,
    method: str = 'isotonic'
) -> Tuple[np.ndarray, Any]:
    """
    Apply post-hoc calibration to model predictions.
    
    Args:
        y_true_train: True labels for calibration set
        y_prob_train: Predicted probabilities for calibration set
        y_prob_test: Predicted probabilities to calibrate
        method: Calibration method ('isotonic' or 'platt')
        
    Returns:
        Calibrated probabilities and fitted calibrator
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for calibration")
    
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    elif method == 'platt':
        calibrator = LogisticRegression()
    else:
        raise ValueError("Method must be 'isotonic' or 'platt'")
    
    # Fit calibrator
    if method == 'platt':
        # Platt scaling requires reshaping for sklearn
        calibrator.fit(y_prob_train.reshape(-1, 1), y_true_train)
        y_prob_calibrated = calibrator.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]
    else:
        # Isotonic regression
        calibrator.fit(y_prob_train, y_true_train)
        y_prob_calibrated = calibrator.predict(y_prob_test)
    
    return y_prob_calibrated, calibrator


def threshold_selection_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: List[str] = ['f1', 'precision', 'recall', 'accuracy'],
    title: str = "Threshold Selection Analysis",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Analyze model performance across different probability thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        metrics: List of metrics to compute
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Figure object and optimal thresholds
    """
    thresholds = np.linspace(0, 1, 101)
    metric_curves = {metric: [] for metric in metrics}
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        if 'accuracy' in metrics:
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            metric_curves['accuracy'].append(accuracy)
        
        if 'precision' in metrics:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            metric_curves['precision'].append(precision)
        
        if 'recall' in metrics:
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            metric_curves['recall'].append(recall)
        
        if 'f1' in metrics:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metric_curves['f1'].append(f1)
    
    # Find optimal thresholds
    optimal_thresholds = {}
    for metric in metrics:
        optimal_idx = np.argmax(metric_curves[metric])
        optimal_thresholds[metric] = thresholds[optimal_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, metric in enumerate(metrics):
        ax.plot(thresholds, metric_curves[metric], 
                label=f'{metric.capitalize()} (opt: {optimal_thresholds[metric]:.3f})',
                color=colors[i % len(colors)])
        
        # Mark optimal threshold
        ax.axvline(optimal_thresholds[metric], color=colors[i % len(colors)], 
                  linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Threshold analysis saved to {save_path}")
    
    return fig, optimal_thresholds


def comprehensive_calibration_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    output_dir: Path = Path("calibration_analysis"),
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Perform comprehensive calibration analysis.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        output_dir: Directory to save outputs
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with all calibration results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate calibration metrics
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    mce = maximum_calibration_error(y_true, y_prob, n_bins)
    brier_decomp = brier_score_decomposition(y_true, y_prob)
    
    # Create reliability diagram
    rel_fig, rel_metrics = reliability_diagram(
        y_true, y_prob, n_bins,
        title=f"{model_name} - Reliability Diagram",
        save_path=output_dir / f"reliability_{model_name.lower()}.png"
    )
    plt.close(rel_fig)
    
    # Create confidence histogram
    conf_fig = confidence_histogram(
        y_prob, y_true,
        title=f"{model_name} - Confidence Distribution",
        save_path=output_dir / f"confidence_{model_name.lower()}.png"
    )
    plt.close(conf_fig)
    
    # Threshold analysis
    thresh_fig, optimal_thresholds = threshold_selection_analysis(
        y_true, y_prob,
        title=f"{model_name} - Threshold Selection",
        save_path=output_dir / f"thresholds_{model_name.lower()}.png"
    )
    plt.close(thresh_fig)
    
    # Save calibration metrics to CSV
    calibration_df = pd.DataFrame({
        'model': [model_name],
        'ece': [ece],
        'mce': [mce],
        'brier_score': [brier_decomp['brier_score']],
        'reliability': [brier_decomp['reliability']], 
        'resolution': [brier_decomp['resolution']],
        'uncertainty': [brier_decomp['uncertainty']]
    })
    
    calibration_df.to_csv(output_dir / f"calibration_metrics_{model_name.lower()}.csv", index=False)
    
    # Save optimal thresholds
    threshold_df = pd.DataFrame(optimal_thresholds, index=[model_name])
    threshold_df.to_csv(output_dir / f"optimal_thresholds_{model_name.lower()}.csv")
    
    results = {
        'ece': ece,
        'mce': mce,
        'brier_decomposition': brier_decomp,
        'optimal_thresholds': optimal_thresholds,
        'reliability_metrics': rel_metrics
    }
    
    logger.info(f"Comprehensive calibration analysis completed for {model_name}")
    logger.info(f"Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate example data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)
    
    # Simulate poorly calibrated probabilities (overconfident)
    base_prob = y_true + np.random.normal(0, 0.3, n_samples)
    y_prob = np.clip(base_prob, 0.01, 0.99)
    
    # Make probabilities more extreme (less calibrated)
    y_prob = np.where(y_prob > 0.5, y_prob * 1.3, y_prob * 0.7)
    y_prob = np.clip(y_prob, 0.01, 0.99)
    
    # Run comprehensive analysis
    results = comprehensive_calibration_analysis(
        y_true, y_prob, 
        model_name="Example_Model",
        output_dir=Path("example_calibration")
    )
    
    print("Calibration Analysis Results:")
    print(f"ECE: {results['ece']:.4f}")
    print(f"MCE: {results['mce']:.4f}")
    print(f"Brier Score: {results['brier_decomposition']['brier_score']:.4f}")
    print(f"Optimal F1 Threshold: {results['optimal_thresholds']['f1']:.3f}")