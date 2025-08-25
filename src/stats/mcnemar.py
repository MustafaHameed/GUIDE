"""Statistical tests for model comparison and validation."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Attempt to import optional statistical libraries
try:
    from scipy import stats
    from scipy.stats import chi2_contingency
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

try:
    import statsmodels.api as sm
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    mcnemar = None


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray, 
    y_pred2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform McNemar's test for comparing two binary classifiers.
    
    Args:
        y_true: True binary labels
        y_pred1: Predictions from first classifier
        y_pred2: Predictions from second classifier
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for McNemar's test")
    
    # Create contingency table for McNemar's test
    # Format: [[both_correct, model1_correct_model2_wrong],
    #          [model1_wrong_model2_correct, both_wrong]]
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    both_correct = np.sum(correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)
    model1_only = np.sum(correct1 & ~correct2)
    model2_only = np.sum(~correct1 & correct2)
    
    # Contingency table
    contingency_table = np.array([
        [both_correct, model1_only],
        [model2_only, both_wrong]
    ])
    
    # McNemar's test statistic
    # Uses continuity correction for small samples
    b = model1_only  # Model 1 correct, Model 2 wrong
    c = model2_only  # Model 1 wrong, Model 2 correct
    
    if b + c == 0:
        # No disagreement between models
        statistic = 0
        p_value = 1.0
    else:
        # McNemar's test with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    # Effect size (difference in accuracy)
    acc1 = np.mean(correct1)
    acc2 = np.mean(correct2)
    effect_size = acc1 - acc2
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'contingency_table': contingency_table,
        'accuracy_model1': acc1,
        'accuracy_model2': acc2, 
        'effect_size': effect_size,
        'disagreement_rate': (b + c) / len(y_true),
        'interpretation': _interpret_mcnemar_result(p_value, effect_size, alpha)
    }


def paired_t_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform paired t-test for comparing model performance scores.
    
    Args:
        scores1: Performance scores from first model (e.g., CV scores)
        scores2: Performance scores from second model 
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for paired t-test")
    
    # Check if arrays have same length
    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have the same length")
    
    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(scores1, scores2)
    
    # Calculate effect size (Cohen's d for paired samples)
    differences = scores1 - scores2
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    if std_diff == 0:
        cohens_d = 0
    else:
        cohens_d = mean_diff / std_diff
    
    # Confidence interval for the difference
    n = len(differences)
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    se_diff = std_diff / np.sqrt(n)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'mean_difference': mean_diff,
        'cohens_d': cohens_d,
        'confidence_interval': (ci_lower, ci_upper),
        'degrees_freedom': n - 1,
        'interpretation': _interpret_ttest_result(p_value, cohens_d, alpha)
    }


def bootstrap_difference_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Bootstrap test for comparing model performance differences.
    
    Args:
        scores1: Performance scores from first model
        scores2: Performance scores from second model
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with test results
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Original difference
    observed_diff = np.mean(scores1) - np.mean(scores2)
    
    # Bootstrap sampling
    n = len(scores1)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        boot_scores1 = scores1[indices]
        boot_scores2 = scores2[indices]
        
        boot_diff = np.mean(boot_scores1) - np.mean(boot_scores2)
        bootstrap_diffs.append(boot_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha/2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha/2))
    
    # P-value calculation (two-tailed)
    # Proportion of bootstrap samples where difference is as extreme as observed
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    
    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'confidence_interval': (ci_lower, ci_upper),
        'bootstrap_differences': bootstrap_diffs,
        'n_bootstrap': n_bootstrap,
        'interpretation': _interpret_bootstrap_result(p_value, observed_diff, alpha)
    }


def compare_multiple_models(
    predictions_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    alpha: float = 0.05,
    correction: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Compare multiple models using pairwise statistical tests.
    
    Args:
        predictions_dict: Dictionary mapping model names to predictions
        y_true: True labels
        alpha: Significance level
        correction: Multiple testing correction method
        
    Returns:
        DataFrame with pairwise comparison results
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    # Calculate number of comparisons for correction
    n_comparisons = n_models * (n_models - 1) // 2
    
    if correction == 'bonferroni':
        adjusted_alpha = alpha / n_comparisons
    elif correction == 'holm':
        # Holm-Bonferroni will be applied after sorting p-values
        adjusted_alpha = alpha
    else:
        adjusted_alpha = alpha
    
    results = []
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            
            # McNemar's test for classification comparison
            mcnemar_result = mcnemar_test(
                y_true, 
                predictions_dict[model1], 
                predictions_dict[model2],
                alpha=adjusted_alpha
            )
            
            results.append({
                'model_1': model1,
                'model_2': model2,
                'mcnemar_statistic': mcnemar_result['statistic'],
                'mcnemar_p_value': mcnemar_result['p_value'],
                'mcnemar_significant': mcnemar_result['significant'],
                'accuracy_diff': mcnemar_result['effect_size'],
                'model_1_accuracy': mcnemar_result['accuracy_model1'],
                'model_2_accuracy': mcnemar_result['accuracy_model2']
            })
    
    results_df = pd.DataFrame(results)
    
    # Apply Holm-Bonferroni correction if requested
    if correction == 'holm' and len(results_df) > 0:
        sorted_indices = np.argsort(results_df['mcnemar_p_value'])
        holm_significant = np.zeros(len(results_df), dtype=bool)
        
        for i, idx in enumerate(sorted_indices):
            adjusted_alpha_holm = alpha / (n_comparisons - i)
            if results_df.loc[idx, 'mcnemar_p_value'] <= adjusted_alpha_holm:
                holm_significant[idx] = True
            else:
                break  # Stop at first non-significant result
        
        results_df['holm_significant'] = holm_significant
    
    return results_df


def _interpret_mcnemar_result(p_value: float, effect_size: float, alpha: float) -> str:
    """Interpret McNemar's test results."""
    if p_value >= alpha:
        return f"No significant difference between models (p={p_value:.4f})"
    
    direction = "Model 1 performs better" if effect_size > 0 else "Model 2 performs better"
    magnitude = abs(effect_size)
    
    if magnitude < 0.01:
        effect_desc = "very small"
    elif magnitude < 0.05:
        effect_desc = "small"
    elif magnitude < 0.10:
        effect_desc = "moderate"
    else:
        effect_desc = "large"
    
    return f"{direction} with {effect_desc} effect (p={p_value:.4f}, diff={effect_size:.4f})"


def _interpret_ttest_result(p_value: float, cohens_d: float, alpha: float) -> str:
    """Interpret paired t-test results."""
    if p_value >= alpha:
        return f"No significant difference between models (p={p_value:.4f})"
    
    direction = "Model 1 performs better" if cohens_d > 0 else "Model 2 performs better"
    magnitude = abs(cohens_d)
    
    if magnitude < 0.2:
        effect_desc = "very small"
    elif magnitude < 0.5:
        effect_desc = "small"
    elif magnitude < 0.8:
        effect_desc = "moderate"
    else:
        effect_desc = "large"
    
    return f"{direction} with {effect_desc} effect (p={p_value:.4f}, d={cohens_d:.4f})"


def _interpret_bootstrap_result(p_value: float, diff: float, alpha: float) -> str:
    """Interpret bootstrap test results."""
    if p_value >= alpha:
        return f"No significant difference between models (p={p_value:.4f})"
    
    direction = "Model 1 performs better" if diff > 0 else "Model 2 performs better"
    return f"{direction} (p={p_value:.4f}, diff={diff:.4f})"


def save_comparison_results(
    results: pd.DataFrame,
    output_path: Path,
    title: str = "Model Comparison Results"
) -> None:
    """Save statistical comparison results to CSV and generate summary."""
    
    # Save detailed results
    results.to_csv(output_path, index=False)
    logger.info(f"Statistical comparison results saved to {output_path}")
    
    # Generate summary report
    summary_path = output_path.parent / f"stats_summary_{output_path.stem}.md"
    
    with open(summary_path, 'w') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Statistical Comparison Results\n\n")
        f.write("### Pairwise Model Comparisons\n\n")
        
        for _, row in results.iterrows():
            model1, model2 = row['model_1'], row['model_2']
            p_val = row['mcnemar_p_value']
            significant = row['mcnemar_significant']
            acc_diff = row['accuracy_diff']
            
            status = "✅ Significant" if significant else "❌ Not significant"
            f.write(f"**{model1} vs {model2}**\n")
            f.write(f"- Status: {status} (p = {p_val:.4f})\n")
            f.write(f"- Accuracy difference: {acc_diff:.4f}\n")
            f.write(f"- {model1} accuracy: {row['model_1_accuracy']:.4f}\n")
            f.write(f"- {model2} accuracy: {row['model_2_accuracy']:.4f}\n\n")
        
        f.write("## Interpretation Guidelines\n\n")
        f.write("- **McNemar's Test**: Tests if two classifiers have significantly different error rates\n")
        f.write("- **p < 0.05**: Statistically significant difference between models\n")
        f.write("- **Accuracy Difference**: Positive values favor the first model\n")
        f.write("- **Effect Size**: |diff| < 0.01 (very small), < 0.05 (small), < 0.10 (moderate), ≥ 0.10 (large)\n\n")
    
    logger.info(f"Statistical summary saved to {summary_path}")


if __name__ == "__main__":
    # Example usage and testing
    np.random.seed(42)
    
    # Generate example data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.6, n_samples)
    
    # Simulate predictions from two models
    # Model 1: 85% accuracy
    y_pred1 = np.where(np.random.random(n_samples) < 0.85, y_true, 1 - y_true)
    
    # Model 2: 82% accuracy  
    y_pred2 = np.where(np.random.random(n_samples) < 0.82, y_true, 1 - y_true)
    
    # Perform McNemar's test
    result = mcnemar_test(y_true, y_pred1, y_pred2)
    print("McNemar's Test Results:")
    print(f"Statistic: {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Significant: {result['significant']}")
    print(f"Interpretation: {result['interpretation']}")