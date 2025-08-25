"""Bootstrap methods for confidence intervals and hypothesis testing."""

import numpy as np
import pandas as pd
from typing import Callable, Tuple, Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import sklearn for metrics
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    **metric_kwargs
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        metric_func: Metric function to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for interval
        random_state: Random seed for reproducibility
        **metric_kwargs: Additional arguments for metric function
        
    Returns:
        Dictionary with metric statistics
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sampling with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        try:
            score = metric_func(y_true[indices], y_pred[indices], **metric_kwargs)
            bootstrap_scores.append(score)
        except Exception as e:
            # Skip bootstrap samples that cause errors (e.g., all one class)
            logger.debug(f"Bootstrap sample failed: {e}")
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    
    # Original metric value
    original_score = metric_func(y_true, y_pred, **metric_kwargs)
    
    return {
        'original_score': original_score,
        'bootstrap_mean': np.mean(bootstrap_scores),
        'bootstrap_std': np.std(bootstrap_scores),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': len(bootstrap_scores),
        'bootstrap_scores': bootstrap_scores
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for any statistic.
    
    Args:
        data: Input data array
        statistic_func: Function to compute statistic (e.g., np.mean, np.median)
        confidence_level: Confidence level for interval
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with statistic and confidence interval
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sampling
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    # Original statistic
    original_stat = statistic_func(data)
    
    return {
        'statistic': original_stat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_mean': np.mean(bootstrap_stats),
        'bootstrap_std': np.std(bootstrap_stats),
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap
    }


def bootstrap_hypothesis_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    statistic_func: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 10000,
    alternative: str = 'two-sided',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform bootstrap hypothesis test comparing two samples.
    
    Args:
        sample1: First sample
        sample2: Second sample  
        statistic_func: Function computing test statistic
        n_bootstrap: Number of bootstrap resamples
        alternative: Type of test ('two-sided', 'greater', 'less')
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with test results
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Observed test statistic
    observed_stat = statistic_func(sample1, sample2)
    
    # Pool samples under null hypothesis (no difference)
    pooled_sample = np.concatenate([sample1, sample2])
    n1, n2 = len(sample1), len(sample2)
    
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample from pooled distribution
        resampled = np.random.choice(pooled_sample, size=n1+n2, replace=True)
        boot_sample1 = resampled[:n1]
        boot_sample2 = resampled[n1:]
        
        boot_stat = statistic_func(boot_sample1, boot_sample2)
        bootstrap_stats.append(boot_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate p-value based on alternative hypothesis
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
    elif alternative == 'greater':
        p_value = np.mean(bootstrap_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(bootstrap_stats <= observed_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    return {
        'observed_statistic': observed_stat,
        'p_value': p_value,
        'alternative': alternative,
        'n_bootstrap': n_bootstrap,
        'bootstrap_distribution': bootstrap_stats
    }


def bootstrap_model_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Compare multiple models using bootstrap confidence intervals.
    
    Args:
        y_true: True labels
        predictions_dict: Dictionary mapping model names to predictions
        metrics: List of metrics to compute
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with model comparison results
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn is required for model comparison")
    
    # Define metric functions
    metric_functions = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='binary', zero_division=0),
        'auc': roc_auc_score
    }
    
    results = []
    
    for model_name, predictions in predictions_dict.items():
        model_results = {'model': model_name}
        
        for metric_name in metrics:
            if metric_name not in metric_functions:
                logger.warning(f"Unknown metric: {metric_name}")
                continue
                
            metric_func = metric_functions[metric_name]
            
            # Handle different prediction types
            if metric_name == 'auc':
                # AUC expects probabilities
                if np.all(np.isin(predictions, [0, 1])):
                    logger.warning(f"AUC requires probabilities, skipping for {model_name}")
                    continue
                y_pred_for_metric = predictions
            else:
                # Other metrics expect binary predictions
                if not np.all(np.isin(predictions, [0, 1])):
                    # Convert probabilities to binary predictions
                    y_pred_for_metric = (predictions >= 0.5).astype(int)
                else:
                    y_pred_for_metric = predictions
            
            # Bootstrap the metric
            bootstrap_result = bootstrap_metric(
                y_true, y_pred_for_metric, metric_func,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                random_state=random_state
            )
            
            # Add results to dictionary
            model_results[f'{metric_name}_score'] = bootstrap_result['original_score']
            model_results[f'{metric_name}_ci_lower'] = bootstrap_result['ci_lower']
            model_results[f'{metric_name}_ci_upper'] = bootstrap_result['ci_upper']
            model_results[f'{metric_name}_ci_width'] = (
                bootstrap_result['ci_upper'] - bootstrap_result['ci_lower']
            )
        
        results.append(model_results)
    
    return pd.DataFrame(results)


def permutation_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    statistic_func: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 10000,
    alternative: str = 'two-sided',
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform permutation test for comparing two samples.
    
    Args:
        sample1: First sample
        sample2: Second sample
        statistic_func: Function computing test statistic
        n_permutations: Number of permutations
        alternative: Type of test ('two-sided', 'greater', 'less')
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with test results
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Observed test statistic
    observed_stat = statistic_func(sample1, sample2)
    
    # Combine samples
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    
    permutation_stats = []
    
    for _ in range(n_permutations):
        # Randomly permute the combined sample
        permuted = np.random.permutation(combined)
        perm_sample1 = permuted[:n1]
        perm_sample2 = permuted[n1:]
        
        perm_stat = statistic_func(perm_sample1, perm_sample2)
        permutation_stats.append(perm_stat)
    
    permutation_stats = np.array(permutation_stats)
    
    # Calculate p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(permutation_stats) >= np.abs(observed_stat))
    elif alternative == 'greater':
        p_value = np.mean(permutation_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(permutation_stats <= observed_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    return {
        'observed_statistic': observed_stat,
        'p_value': p_value,
        'alternative': alternative,
        'n_permutations': n_permutations,
        'permutation_distribution': permutation_stats
    }


def comprehensive_bootstrap_analysis(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    output_dir: Path = Path("bootstrap_analysis"),
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive bootstrap analysis for model comparison.
    
    Args:
        y_true: True labels
        predictions_dict: Dictionary mapping model names to predictions  
        output_dir: Directory to save results
        metrics: List of metrics to analyze
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with all bootstrap results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model comparison with confidence intervals
    comparison_df = bootstrap_model_comparison(
        y_true, predictions_dict, metrics,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state
    )
    
    # Save comparison results
    comparison_df.to_csv(output_dir / "bootstrap_model_comparison.csv", index=False)
    
    # Pairwise bootstrap tests
    model_names = list(predictions_dict.keys())
    pairwise_results = []
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            
            # Define difference function for bootstrap test
            def accuracy_difference(pred1, pred2):
                acc1 = accuracy_score(y_true, (pred1 >= 0.5).astype(int))
                acc2 = accuracy_score(y_true, (pred2 >= 0.5).astype(int))
                return acc1 - acc2
            
            # Bootstrap hypothesis test
            bootstrap_test = bootstrap_hypothesis_test(
                predictions_dict[model1],
                predictions_dict[model2],
                statistic_func=lambda s1, s2: accuracy_difference(s1, s2),
                n_bootstrap=n_bootstrap,
                random_state=random_state
            )
            
            pairwise_results.append({
                'model_1': model1,
                'model_2': model2,
                'accuracy_difference': bootstrap_test['observed_statistic'],
                'p_value': bootstrap_test['p_value'],
                'significant': bootstrap_test['p_value'] < 0.05
            })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv(output_dir / "bootstrap_pairwise_tests.csv", index=False)
    
    # Generate summary report
    _generate_bootstrap_report(
        comparison_df, pairwise_df, output_dir,
        confidence_level, n_bootstrap
    )
    
    results = {
        'model_comparison': comparison_df,
        'pairwise_tests': pairwise_df,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap
    }
    
    logger.info(f"Bootstrap analysis completed. Results saved to {output_dir}")
    
    return results


def _generate_bootstrap_report(
    comparison_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    output_dir: Path,
    confidence_level: float,
    n_bootstrap: int
) -> None:
    """Generate markdown report for bootstrap analysis."""
    
    report_path = output_dir / "bootstrap_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Bootstrap Analysis Report\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Bootstrap Samples:** {n_bootstrap}\n")
        f.write(f"**Confidence Level:** {confidence_level*100:.1f}%\n\n")
        
        # Model performance with confidence intervals
        f.write("## Model Performance Comparison\n\n")
        f.write("| Model | Accuracy | 95% CI | Precision | 95% CI | Recall | 95% CI | F1 | 95% CI |\n")
        f.write("|-------|----------|--------|-----------|--------|--------|--------|----|---------|\n")
        
        for _, row in comparison_df.iterrows():
            model = row['model']
            
            metrics_text = []
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if f'{metric}_score' in row:
                    score = row[f'{metric}_score']
                    ci_lower = row[f'{metric}_ci_lower']
                    ci_upper = row[f'{metric}_ci_upper']
                    metrics_text.extend([
                        f"{score:.3f}",
                        f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                    ])
                else:
                    metrics_text.extend(["N/A", "N/A"])
            
            f.write(f"| {model} | " + " | ".join(metrics_text) + " |\n")
        
        f.write("\n")
        
        # Pairwise comparisons
        f.write("## Pairwise Statistical Tests\n\n")
        f.write("Bootstrap hypothesis tests for accuracy differences:\n\n")
        
        for _, row in pairwise_df.iterrows():
            model1, model2 = row['model_1'], row['model_2']
            diff = row['accuracy_difference']
            p_val = row['p_value']
            significant = row['significant']
            
            status = "✅ Significant" if significant else "❌ Not significant"
            direction = ">" if diff > 0 else "<"
            
            f.write(f"**{model1} vs {model2}**\n")
            f.write(f"- Accuracy difference: {diff:.4f} ({model1} {direction} {model2})\n")
            f.write(f"- Bootstrap p-value: {p_val:.4f}\n")
            f.write(f"- Status: {status}\n\n")
        
        # Interpretation
        f.write("## Interpretation\n\n")
        f.write("- **Confidence Intervals**: Bootstrap 95% confidence intervals for each metric\n")
        f.write("- **Statistical Tests**: Bootstrap hypothesis tests for pairwise accuracy comparisons\n")
        f.write("- **Significance**: p < 0.05 indicates statistically significant difference\n")
        f.write("- **Bootstrap Method**: Provides robust estimates without distributional assumptions\n\n")
    
    logger.info(f"Bootstrap report saved to {report_path}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate example data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.4, n_samples)
    
    # Simulate predictions from two models
    pred1 = np.where(np.random.random(n_samples) < 0.8, y_true, 1 - y_true)
    pred2 = np.where(np.random.random(n_samples) < 0.75, y_true, 1 - y_true)
    
    # Bootstrap accuracy comparison
    accuracy_bootstrap = bootstrap_metric(
        y_true, pred1, accuracy_score,
        n_bootstrap=1000, random_state=42
    )
    
    print("Bootstrap Accuracy Results:")
    print(f"Original accuracy: {accuracy_bootstrap['original_score']:.4f}")
    print(f"Bootstrap mean: {accuracy_bootstrap['bootstrap_mean']:.4f}")
    print(f"95% CI: [{accuracy_bootstrap['ci_lower']:.4f}, {accuracy_bootstrap['ci_upper']:.4f}]")
    
    # Model comparison
    predictions_dict = {'Model_1': pred1, 'Model_2': pred2}
    comparison = bootstrap_model_comparison(y_true, predictions_dict, random_state=42)
    print("\nModel Comparison:")
    print(comparison[['model', 'accuracy_score', 'accuracy_ci_lower', 'accuracy_ci_upper']])