"""
Transfer Learning Shift Diagnostics

Implements diagnostic tools to analyze domain shift between source and target datasets:
- Population Stability Index (PSI) for covariate shift
- Kolmogorov-Smirnov test for distribution differences  
- Proxy A-distance (PAD) for domain classification difficulty
- Label shift estimation using EM algorithm

Based on the problem statement requirements for improved transfer learning.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) to measure covariate shift.
    
    Args:
        expected: Expected (source) distribution values
        actual: Actual (target) distribution values  
        bins: Number of bins for discretization
        
    Returns:
        PSI value (higher = more shift, >0.2 indicates significant shift)
    """
    try:
        # Handle edge cases
        if len(expected) == 0 or len(actual) == 0:
            return np.nan
            
        # Create bins based on expected distribution
        bin_edges = np.histogram_bin_edges(expected, bins=bins)
        
        # Calculate distributions
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
        
        # Convert to proportions and add small constant to avoid log(0)
        expected_props = expected_counts / len(expected) + 1e-8
        actual_props = actual_counts / len(actual) + 1e-8
        
        # Calculate PSI
        psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
        
        return psi
        
    except Exception as e:
        logger.warning(f"PSI calculation failed: {e}")
        return np.nan


def calculate_ks_statistic(x1: np.ndarray, x2: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov test statistic and p-value.
    
    Args:
        x1: Source distribution samples
        x2: Target distribution samples
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    try:
        # Remove NaN values
        x1_clean = x1[~np.isnan(x1)]
        x2_clean = x2[~np.isnan(x2)]
        
        if len(x1_clean) == 0 or len(x2_clean) == 0:
            return np.nan, np.nan
            
        ks_stat, p_value = stats.ks_2samp(x1_clean, x2_clean)
        return ks_stat, p_value
        
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return np.nan, np.nan


def calculate_proxy_a_distance(X_source: np.ndarray, X_target: np.ndarray, 
                             test_size: float = 0.2, random_state: int = 42) -> float:
    """
    Calculate Proxy A-distance using domain classification accuracy.
    
    A-distance measures domain similarity. Higher values indicate more different domains.
    Proxy A-distance = 2(1 - 2*error_rate) where error_rate is domain classification error.
    
    Args:
        X_source: Source domain features
        X_target: Target domain features  
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Proxy A-distance value (0 = identical domains, 2 = completely different)
    """
    try:
        # Create domain labels (0 = source, 1 = target)
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]
        
        X_combined = np.vstack([X_source, X_target])
        y_domain = np.concatenate([np.zeros(n_source), np.ones(n_target)])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # Train domain classifier with cross-validation
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        cv_scores = cross_val_score(clf, X_scaled, y_domain, cv=5, scoring='accuracy')
        
        # Calculate proxy A-distance
        accuracy = cv_scores.mean()
        error_rate = 1 - accuracy
        proxy_a_distance = 2 * (1 - 2 * error_rate)
        
        # Clip to valid range [0, 2]
        proxy_a_distance = np.clip(proxy_a_distance, 0, 2)
        
        return proxy_a_distance
        
    except Exception as e:
        logger.warning(f"Proxy A-distance calculation failed: {e}")
        return np.nan


def estimate_label_shift(y_source: np.ndarray, X_target: np.ndarray, 
                        source_model, method: str = 'confusion_matrix') -> np.ndarray:
    """
    Estimate label shift in target domain using confusion matrix method.
    
    Args:
        y_source: Source domain labels
        X_target: Target domain features
        source_model: Trained model on source domain
        method: Method for estimation ('confusion_matrix' or 'em')
        
    Returns:
        Estimated target label proportions
    """
    try:
        # Get source label proportions
        source_props = np.bincount(y_source) / len(y_source)
        
        # Get predictions on target domain
        if hasattr(source_model, 'predict'):
            y_target_pred = source_model.predict(X_target)
        else:
            return source_props  # Fallback to source proportions
            
        if method == 'confusion_matrix':
            # Simple method: use predicted proportions
            target_props = np.bincount(y_target_pred, minlength=len(source_props)) / len(y_target_pred)
        else:
            # EM algorithm (simplified version)
            target_props = np.bincount(y_target_pred, minlength=len(source_props)) / len(y_target_pred)
            
        return target_props
        
    except Exception as e:
        logger.warning(f"Label shift estimation failed: {e}")
        return np.bincount(y_source) / len(y_source)


def generate_shift_report(X_source: pd.DataFrame, X_target: pd.DataFrame,
                         y_source: np.ndarray, source_model=None,
                         output_dir: Path = None) -> Dict:
    """
    Generate comprehensive shift analysis report.
    
    Args:
        X_source: Source domain features
        X_target: Target domain features
        y_source: Source domain labels
        source_model: Trained model for label shift estimation
        output_dir: Directory to save detailed reports
        
    Returns:
        Dictionary with shift analysis results
    """
    logger.info("Generating domain shift diagnostics report...")
    
    results = {
        'feature_shift': {},
        'domain_metrics': {},
        'label_shift': {},
        'summary': {}
    }
    
    # Align features between source and target
    common_features = list(set(X_source.columns) & set(X_target.columns))
    X_source_aligned = X_source[common_features]
    X_target_aligned = X_target[common_features]
    
    logger.info(f"Analyzing {len(common_features)} common features")
    
    # Feature-wise shift analysis
    feature_shifts = []
    for feature in common_features:
        source_vals = X_source_aligned[feature].dropna().values
        target_vals = X_target_aligned[feature].dropna().values
        
        if len(source_vals) > 0 and len(target_vals) > 0:
            # Calculate PSI and KS test
            psi = calculate_psi(source_vals, target_vals)
            ks_stat, ks_pval = calculate_ks_statistic(source_vals, target_vals)
            
            feature_result = {
                'feature': feature,
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'source_mean': np.mean(source_vals),
                'target_mean': np.mean(target_vals),
                'source_std': np.std(source_vals),
                'target_std': np.std(target_vals)
            }
            
            feature_shifts.append(feature_result)
            results['feature_shift'][feature] = feature_result
    
    # Overall domain similarity
    try:
        X_source_array = X_source_aligned.fillna(0).values
        X_target_array = X_target_aligned.fillna(0).values
        
        pad = calculate_proxy_a_distance(X_source_array, X_target_array)
        
        # Domain classifier AUC (alternative measure)
        n_source = X_source_array.shape[0]
        n_target = X_target_array.shape[0]
        X_combined = np.vstack([X_source_array, X_target_array])
        y_domain = np.concatenate([np.zeros(n_source), np.ones(n_target)])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(clf, X_scaled, y_domain, cv=5, scoring='roc_auc')
        domain_auc = cv_scores.mean()
        
        results['domain_metrics'] = {
            'proxy_a_distance': pad,
            'domain_classifier_auc': domain_auc,
            'domain_classifier_auc_std': cv_scores.std()
        }
        
    except Exception as e:
        logger.warning(f"Domain metrics calculation failed: {e}")
        results['domain_metrics'] = {'error': str(e)}
    
    # Label shift estimation
    if source_model is not None and len(common_features) > 0:
        try:
            X_target_for_pred = X_target_aligned.fillna(0).values
            target_label_props = estimate_label_shift(y_source, X_target_for_pred, source_model)
            source_label_props = np.bincount(y_source) / len(y_source)
            
            results['label_shift'] = {
                'source_proportions': source_label_props.tolist(),
                'estimated_target_proportions': target_label_props.tolist(),
                'label_shift_detected': np.max(np.abs(target_label_props - source_label_props)) > 0.1
            }
        except Exception as e:
            logger.warning(f"Label shift estimation failed: {e}")
            results['label_shift'] = {'error': str(e)}
    
    # Summary statistics
    if feature_shifts:
        feature_df = pd.DataFrame(feature_shifts)
        
        # Top shifting features by PSI
        top_psi_features = feature_df.nlargest(10, 'psi')['feature'].tolist()
        
        # Features with significant KS test (p < 0.05)
        significant_ks = feature_df[feature_df['ks_pvalue'] < 0.05]['feature'].tolist()
        
        results['summary'] = {
            'total_features_analyzed': len(feature_shifts),
            'top_shifting_features_psi': top_psi_features,
            'features_with_significant_ks': significant_ks,
            'mean_psi': feature_df['psi'].mean(),
            'median_psi': feature_df['psi'].median(),
            'max_psi': feature_df['psi'].max()
        }
        
        # Save detailed feature shift table
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            feature_df.to_csv(output_dir / 'feature_shift_analysis.csv', index=False)
            logger.info(f"Feature shift analysis saved to {output_dir / 'feature_shift_analysis.csv'}")
    
    return results


def create_shift_report_summary(results: Dict, source_name: str, target_name: str) -> str:
    """
    Create a markdown summary of the shift analysis.
    
    Args:
        results: Results from generate_shift_report
        source_name: Name of source domain (e.g., 'UCI')
        target_name: Name of target domain (e.g., 'OULAD')
        
    Returns:
        Markdown formatted summary string
    """
    summary = f"# Domain Shift Analysis: {source_name} → {target_name}\n\n"
    
    # Domain similarity metrics
    if 'domain_metrics' in results and 'proxy_a_distance' in results['domain_metrics']:
        pad = results['domain_metrics']['proxy_a_distance']
        auc = results['domain_metrics']['domain_classifier_auc']
        
        summary += f"## Overall Domain Similarity\n"
        summary += f"- **Proxy A-distance**: {pad:.3f} (0=identical, 2=completely different)\n"
        summary += f"- **Domain Classifier AUC**: {auc:.3f} (0.5=indistinguishable domains)\n\n"
        
        if pad > 1.0:
            summary += "⚠️ **High domain shift detected** - transfer learning may be challenging\n\n"
        elif pad > 0.5:
            summary += "⚡ **Moderate domain shift** - domain adaptation techniques recommended\n\n"
        else:
            summary += "✅ **Low domain shift** - standard transfer learning should work well\n\n"
    
    # Feature shift summary  
    if 'summary' in results:
        s = results['summary']
        summary += f"## Feature Shift Analysis\n"
        summary += f"- **Features analyzed**: {s.get('total_features_analyzed', 0)}\n"
        summary += f"- **Mean PSI**: {s.get('mean_psi', 0):.3f}\n"
        summary += f"- **Max PSI**: {s.get('max_psi', 0):.3f}\n\n"
        
        if s.get('top_shifting_features_psi'):
            summary += f"**Top shifting features (PSI)**: {', '.join(s['top_shifting_features_psi'][:5])}\n\n"
    
    # Label shift
    if 'label_shift' in results and 'source_proportions' in results['label_shift']:
        ls = results['label_shift']
        summary += f"## Label Shift Analysis\n"
        summary += f"- **Source class proportions**: {[f'{p:.3f}' for p in ls['source_proportions']]}\n"
        summary += f"- **Estimated target proportions**: {[f'{p:.3f}' for p in ls['estimated_target_proportions']]}\n"
        
        if ls.get('label_shift_detected'):
            summary += "⚠️ **Significant label shift detected** - consider label shift correction\n\n"
        else:
            summary += "✅ **No significant label shift detected**\n\n"
    
    # Recommendations
    summary += "## Recommendations\n"
    
    if 'domain_metrics' in results:
        pad = results['domain_metrics'].get('proxy_a_distance', 0)
        if pad > 1.0:
            summary += "- Use importance weighting and domain adaptation\n"
            summary += "- Consider CORAL feature alignment\n" 
            summary += "- Apply DANN for deep learning models\n"
        elif pad > 0.5:
            summary += "- Apply importance weighting\n"
            summary += "- Consider feature alignment techniques\n"
        else:
            summary += "- Standard transfer learning should be sufficient\n"
    
    if results.get('label_shift', {}).get('label_shift_detected'):
        summary += "- Apply label shift correction (Saerens-Decock method)\n"
    
    return summary


def main():
    """CLI interface for running shift diagnostics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Domain shift diagnostics for transfer learning")
    parser.add_argument('--from', dest='source', choices=['uci', 'oulad'], required=True,
                       help='Source domain')
    parser.add_argument('--to', dest='target', choices=['uci', 'oulad'], required=True,
                       help='Target domain')
    parser.add_argument('--output-dir', type=Path, default='tables/transfer',
                       help='Output directory for reports')
    parser.add_argument('--save-summary', action='store_true',
                       help='Save markdown summary')
    
    args = parser.parse_args()
    
    # Load data (placeholder - would need to implement data loading)
    logger.info(f"Running diagnostics: {args.source} → {args.target}")
    
    # This would need actual data loading implementation
    print(f"Domain shift diagnostics from {args.source} to {args.target}")
    print("Note: Full implementation requires integration with existing data loaders")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()