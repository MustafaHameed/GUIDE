"""ANOVA and post-hoc analysis for multiple model comparison."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Attempt to import required statistical libraries
try:
    from scipy import stats
    from scipy.stats import f_oneway
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

try:
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import anova_lm
    from statsmodels.formula.api import ols
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    pairwise_tukeyhsd = None


def one_way_anova(
    scores_dict: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform one-way ANOVA to test if model performances differ significantly.
    
    Args:
        scores_dict: Dictionary mapping model names to performance scores
        alpha: Significance level
        
    Returns:
        Dictionary with ANOVA results
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for ANOVA")
    
    model_names = list(scores_dict.keys())
    scores_arrays = [scores_dict[name] for name in model_names]
    
    # Check that all arrays have data
    if any(len(arr) == 0 for arr in scores_arrays):
        raise ValueError("All score arrays must be non-empty")
    
    # Perform one-way ANOVA
    f_statistic, p_value = f_oneway(*scores_arrays)
    
    # Calculate effect size (eta-squared)
    # eta² = SSbetween / SStotal
    all_scores = np.concatenate(scores_arrays)
    grand_mean = np.mean(all_scores)
    total_n = len(all_scores)
    
    # Sum of squares between groups
    ss_between = 0
    for scores in scores_arrays:
        group_mean = np.mean(scores)
        ss_between += len(scores) * (group_mean - grand_mean) ** 2
    
    # Total sum of squares
    ss_total = np.sum((all_scores - grand_mean) ** 2)
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    # Degrees of freedom
    df_between = len(model_names) - 1
    df_within = total_n - len(model_names)
    df_total = total_n - 1
    
    # Calculate descriptive statistics
    descriptives = {}
    for name, scores in scores_dict.items():
        descriptives[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores, ddof=1),
            'n': len(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    
    return {
        'f_statistic': f_statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'eta_squared': eta_squared,
        'df_between': df_between,
        'df_within': df_within,
        'df_total': df_total,
        'descriptives': descriptives,
        'interpretation': _interpret_anova_result(p_value, eta_squared, alpha)
    }


def tukey_hsd_test(
    scores_dict: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform Tukey HSD post-hoc test for pairwise comparisons.
    
    Args:
        scores_dict: Dictionary mapping model names to performance scores
        alpha: Significance level for family-wise error rate
        
    Returns:
        Dictionary with Tukey HSD results
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required for Tukey HSD")
    
    # Prepare data for statsmodels
    data_list = []
    for model_name, scores in scores_dict.items():
        for score in scores:
            data_list.append({'model': model_name, 'score': score})
    
    df = pd.DataFrame(data_list)
    
    # Perform Tukey HSD test
    tukey_result = pairwise_tukeyhsd(
        endog=df['score'],
        groups=df['model'],
        alpha=alpha
    )
    
    # Extract pairwise comparison results
    comparisons = []
    for i, (group1, group2) in enumerate(zip(tukey_result.groupsunique[tukey_result.pairindices[0]], 
                                            tukey_result.groupsunique[tukey_result.pairindices[1]])):
        mean_diff = tukey_result.meandiffs[i]
        ci_lower = tukey_result.confint[i][0]
        ci_upper = tukey_result.confint[i][1]
        p_adj = tukey_result.pvalues[i]
        reject = tukey_result.reject[i]
        
        comparisons.append({
            'group_1': group1,
            'group_2': group2,
            'mean_diff': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_adj': p_adj,
            'significant': reject,
            'effect_size': _calculate_cohens_d(
                scores_dict[group1], 
                scores_dict[group2]
            )
        })
    
    comparisons_df = pd.DataFrame(comparisons)
    
    return {
        'tukey_result': tukey_result,
        'comparisons': comparisons_df,
        'alpha': alpha,
        'summary': tukey_result.summary(),
        'interpretation': _interpret_tukey_results(comparisons_df, alpha)
    }


def repeated_measures_anova(
    scores_df: pd.DataFrame,
    subject_col: str = 'fold',
    model_col: str = 'model',
    score_col: str = 'score',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform repeated measures ANOVA for cross-validation results.
    
    Args:
        scores_df: DataFrame with columns for subjects (folds), models, and scores
        subject_col: Column name for subjects (e.g., CV folds)
        model_col: Column name for models
        score_col: Column name for performance scores
        alpha: Significance level
        
    Returns:
        Dictionary with repeated measures ANOVA results
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required for repeated measures ANOVA")
    
    # Ensure data is properly formatted
    if not all(col in scores_df.columns for col in [subject_col, model_col, score_col]):
        raise ValueError(f"DataFrame must contain columns: {subject_col}, {model_col}, {score_col}")
    
    # Fit the model using OLS with repeated measures
    formula = f'{score_col} ~ C({model_col}) + C({subject_col})'
    model = ols(formula, data=scores_df).fit()
    
    # Perform ANOVA
    anova_results = anova_lm(model, typ=2)
    
    # Extract relevant statistics
    model_effect = anova_results.loc[f'C({model_col})']
    subject_effect = anova_results.loc[f'C({subject_col})']
    
    # Calculate effect sizes (partial eta-squared)
    ss_model = model_effect['sum_sq']
    ss_error = anova_results.loc['Residual']['sum_sq']
    partial_eta_sq_model = ss_model / (ss_model + ss_error)
    
    return {
        'anova_table': anova_results,
        'model_f_statistic': model_effect['F'],
        'model_p_value': model_effect['PR(>F)'],
        'model_significant': model_effect['PR(>F)'] < alpha,
        'partial_eta_squared': partial_eta_sq_model,
        'subject_f_statistic': subject_effect['F'],
        'subject_p_value': subject_effect['PR(>F)'],
        'alpha': alpha,
        'fitted_model': model,
        'interpretation': _interpret_rm_anova_result(
            model_effect['PR(>F)'], 
            partial_eta_sq_model, 
            alpha
        )
    }


def friedman_test(
    scores_dict: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform Friedman test (non-parametric alternative to repeated measures ANOVA).
    
    Args:
        scores_dict: Dictionary mapping model names to performance scores
        alpha: Significance level
        
    Returns:
        Dictionary with Friedman test results
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for Friedman test")
    
    # Convert to matrix format (subjects x treatments)
    model_names = list(scores_dict.keys())
    scores_arrays = [scores_dict[name] for name in model_names]
    
    # Check that all arrays have the same length
    lengths = [len(arr) for arr in scores_arrays]
    if len(set(lengths)) > 1:
        raise ValueError("All score arrays must have the same length for Friedman test")
    
    # Perform Friedman test
    statistic, p_value = stats.friedmanchisquare(*scores_arrays)
    
    # Calculate effect size (Kendall's W)
    n_subjects = lengths[0]
    k_treatments = len(model_names)
    
    # Rank data for Kendall's W calculation
    ranks_matrix = []
    for i in range(n_subjects):
        subject_scores = [scores_arrays[j][i] for j in range(k_treatments)]
        ranks = stats.rankdata(subject_scores)
        ranks_matrix.append(ranks)
    
    ranks_matrix = np.array(ranks_matrix)
    rank_sums = np.sum(ranks_matrix, axis=0)
    
    # Kendall's W (effect size)
    mean_rank_sum = np.mean(rank_sums)
    ss_ranks = np.sum((rank_sums - mean_rank_sum) ** 2)
    kendalls_w = (12 * ss_ranks) / (n_subjects**2 * (k_treatments**3 - k_treatments))
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'kendalls_w': kendalls_w,
        'n_subjects': n_subjects,
        'k_treatments': k_treatments,
        'rank_sums': rank_sums,
        'model_names': model_names,
        'interpretation': _interpret_friedman_result(p_value, kendalls_w, alpha)
    }


def nemenyi_post_hoc_test(
    scores_dict: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform Nemenyi post-hoc test following Friedman test.
    
    Args:
        scores_dict: Dictionary mapping model names to performance scores
        alpha: Significance level
        
    Returns:
        DataFrame with pairwise comparison results
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for Nemenyi test")
    
    model_names = list(scores_dict.keys())
    scores_arrays = [scores_dict[name] for name in model_names]
    
    # Check array lengths
    lengths = [len(arr) for arr in scores_arrays]
    if len(set(lengths)) > 1:
        raise ValueError("All score arrays must have the same length")
    
    n_subjects = lengths[0]
    k_treatments = len(model_names)
    
    # Calculate average ranks for each treatment
    ranks_matrix = []
    for i in range(n_subjects):
        subject_scores = [scores_arrays[j][i] for j in range(k_treatments)]
        ranks = stats.rankdata(subject_scores)
        ranks_matrix.append(ranks)
    
    ranks_matrix = np.array(ranks_matrix)
    avg_ranks = np.mean(ranks_matrix, axis=0)
    
    # Critical difference for Nemenyi test
    q_alpha = _get_studentized_range_statistic(k_treatments, alpha)
    critical_diff = q_alpha * np.sqrt(k_treatments * (k_treatments + 1) / (6 * n_subjects))
    
    # Pairwise comparisons
    comparisons = []
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            rank_diff = abs(avg_ranks[i] - avg_ranks[j])
            significant = rank_diff > critical_diff
            
            comparisons.append({
                'model_1': model1,
                'model_2': model2,
                'rank_diff': rank_diff,
                'critical_diff': critical_diff,
                'significant': significant,
                'avg_rank_1': avg_ranks[i],
                'avg_rank_2': avg_ranks[j]
            })
    
    return pd.DataFrame(comparisons)


def _calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def _get_studentized_range_statistic(k: int, alpha: float) -> float:
    """Get critical value for studentized range statistic."""
    # Simplified lookup table for common values
    # In practice, you would use a more comprehensive table or scipy
    q_values = {
        (3, 0.05): 3.314,
        (4, 0.05): 3.633,
        (5, 0.05): 3.858,
        (6, 0.05): 4.030,
        (7, 0.05): 4.170,
        (8, 0.05): 4.286,
        (9, 0.05): 4.387,
        (10, 0.05): 4.474
    }
    
    return q_values.get((k, alpha), 3.858)  # Default to k=5 if not found


def _interpret_anova_result(p_value: float, eta_squared: float, alpha: float) -> str:
    """Interpret ANOVA results."""
    if p_value >= alpha:
        return f"No significant difference between models (p={p_value:.4f})"
    
    if eta_squared < 0.01:
        effect = "very small"
    elif eta_squared < 0.06:
        effect = "small"
    elif eta_squared < 0.14:
        effect = "moderate"
    else:
        effect = "large"
    
    return f"Significant difference between models (p={p_value:.4f}) with {effect} effect (η²={eta_squared:.4f})"


def _interpret_tukey_results(comparisons_df: pd.DataFrame, alpha: float) -> str:
    """Interpret Tukey HSD results."""
    n_significant = comparisons_df['significant'].sum()
    n_total = len(comparisons_df)
    
    if n_significant == 0:
        return f"No significant pairwise differences found (α={alpha})"
    
    return f"{n_significant}/{n_total} pairwise comparisons are significant (α={alpha})"


def _interpret_rm_anova_result(p_value: float, eta_squared: float, alpha: float) -> str:
    """Interpret repeated measures ANOVA results."""
    if p_value >= alpha:
        return f"No significant model effect (p={p_value:.4f})"
    
    if eta_squared < 0.01:
        effect = "very small"
    elif eta_squared < 0.06:
        effect = "small"
    elif eta_squared < 0.14:
        effect = "moderate"
    else:
        effect = "large"
    
    return f"Significant model effect (p={p_value:.4f}) with {effect} effect (partial η²={eta_squared:.4f})"


def _interpret_friedman_result(p_value: float, kendalls_w: float, alpha: float) -> str:
    """Interpret Friedman test results."""
    if p_value >= alpha:
        return f"No significant difference between models (p={p_value:.4f})"
    
    if kendalls_w < 0.1:
        effect = "small"
    elif kendalls_w < 0.3:
        effect = "moderate"
    else:
        effect = "large"
    
    return f"Significant difference between models (p={p_value:.4f}) with {effect} effect (W={kendalls_w:.4f})"


def save_anova_results(
    anova_results: Dict[str, Any],
    tukey_results: Dict[str, Any],
    output_dir: Path,
    title: str = "Model Comparison ANOVA Results"
) -> None:
    """Save ANOVA and Tukey HSD results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Tukey comparisons
    tukey_df = tukey_results['comparisons']
    tukey_path = output_dir / "tukey_comparisons.csv"
    tukey_df.to_csv(tukey_path, index=False)
    
    # Save descriptive statistics
    descriptives_df = pd.DataFrame(anova_results['descriptives']).T
    descriptives_path = output_dir / "model_descriptives.csv"
    descriptives_df.to_csv(descriptives_path)
    
    # Generate summary report
    summary_path = output_dir / "anova_summary.md"
    
    with open(summary_path, 'w') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # ANOVA results
        f.write("## One-Way ANOVA Results\n\n")
        f.write(f"- **F-statistic:** {anova_results['f_statistic']:.4f}\n")
        f.write(f"- **p-value:** {anova_results['p_value']:.4f}\n")
        f.write(f"- **Effect size (η²):** {anova_results['eta_squared']:.4f}\n")
        f.write(f"- **Interpretation:** {anova_results['interpretation']}\n\n")
        
        # Descriptive statistics
        f.write("## Descriptive Statistics\n\n")
        f.write("| Model | Mean | Std | N | Min | Max |\n")
        f.write("|-------|------|-----|---|-----|-----|\n")
        for model, stats in anova_results['descriptives'].items():
            f.write(f"| {model} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['n']} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
        f.write("\n")
        
        # Tukey HSD results
        f.write("## Tukey HSD Post-Hoc Results\n\n")
        f.write(f"**Interpretation:** {tukey_results['interpretation']}\n\n")
        
        for _, row in tukey_df.iterrows():
            status = "✅ Significant" if row['significant'] else "❌ Not significant"
            f.write(f"**{row['group_1']} vs {row['group_2']}**\n")
            f.write(f"- Status: {status} (p = {row['p_adj']:.4f})\n")
            f.write(f"- Mean difference: {row['mean_diff']:.4f}\n")
            f.write(f"- 95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]\n")
            f.write(f"- Effect size (Cohen's d): {row['effect_size']:.4f}\n\n")
    
    logger.info(f"ANOVA results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate example data for 4 models with 10 CV folds each
    models = ['Logistic', 'Random_Forest', 'SVM', 'Neural_Net']
    n_folds = 10
    
    scores_dict = {}
    for i, model in enumerate(models):
        # Generate scores with different means but same variance
        base_score = 0.75 + i * 0.02  # Incrementally better performance
        scores = np.random.normal(base_score, 0.05, n_folds)
        scores = np.clip(scores, 0, 1)  # Ensure valid accuracy scores
        scores_dict[model] = scores
    
    # Perform ANOVA
    anova_result = one_way_anova(scores_dict)
    print("ANOVA Results:")
    print(f"F-statistic: {anova_result['f_statistic']:.4f}")
    print(f"p-value: {anova_result['p_value']:.4f}")
    print(f"Significant: {anova_result['significant']}")
    print(f"Interpretation: {anova_result['interpretation']}\n")
    
    # Perform Tukey HSD if ANOVA is significant
    if anova_result['significant']:
        tukey_result = tukey_hsd_test(scores_dict)
        print("Tukey HSD Results:")
        print(tukey_result['interpretation'])
        print("\nSignificant pairwise comparisons:")
        significant_pairs = tukey_result['comparisons'][tukey_result['comparisons']['significant']]
        for _, row in significant_pairs.iterrows():
            print(f"  {row['group_1']} vs {row['group_2']}: p = {row['p_adj']:.4f}")
    
    # Also test Friedman test
    friedman_result = friedman_test(scores_dict)
    print(f"\nFriedman Test Results:")
    print(f"Statistic: {friedman_result['statistic']:.4f}")
    print(f"p-value: {friedman_result['p_value']:.4f}")
    print(f"Interpretation: {friedman_result['interpretation']}")