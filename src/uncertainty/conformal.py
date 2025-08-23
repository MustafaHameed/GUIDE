"""
Conformal Prediction for Binary Classification

Implements inductive conformal prediction for binary classification with
calibrated prediction sets and group-wise coverage analysis.

References:
- Conformal Prediction: https://en.wikipedia.org/wiki/Conformal_prediction
- Algorithmic Learning Theory: Vovk et al. "Algorithmic Learning in a Random World"
"""

import logging
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)


class InductiveConformalPredictor:
    """Inductive conformal prediction for binary classification."""
    
    def __init__(self, alpha: float = 0.1):
        """Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (1-alpha is target coverage)
        """
        self.alpha = alpha
        self.target_coverage = 1 - alpha
        self.calibration_scores = None
        self.threshold = None
        
        logger.info(f"Initialized conformal predictor with alpha={alpha}, target coverage={self.target_coverage}")
    
    def _nonconformity_score(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores as 1 - true class probability.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities for positive class
            
        Returns:
            Nonconformity scores
        """
        # For binary classification: 1 - probability of true class
        true_class_probs = np.where(y_true == 1, y_prob, 1 - y_prob)
        return 1 - true_class_probs
    
    def calibrate(self, y_cal: np.ndarray, y_prob_cal: np.ndarray) -> None:
        """Calibrate the conformal predictor using calibration set.
        
        Args:
            y_cal: Calibration set true labels
            y_prob_cal: Calibration set predicted probabilities
        """
        logger.info(f"Calibrating with {len(y_cal)} samples")
        
        # Compute nonconformity scores on calibration set
        self.calibration_scores = self._nonconformity_score(y_cal, y_prob_cal)
        
        # Compute threshold for desired coverage
        # Use (n+1)(1-alpha)/n quantile for finite sample correction
        n_cal = len(self.calibration_scores)
        quantile_level = (n_cal + 1) * (1 - self.alpha) / n_cal
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0
        
        self.threshold = np.quantile(self.calibration_scores, quantile_level)
        
        logger.info(f"Calibration threshold: {self.threshold:.4f}")
        logger.info(f"Empirical coverage on calibration set: "
                   f"{(self.calibration_scores <= self.threshold).mean():.4f}")
    
    def predict_sets(self, y_prob_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction sets for test data.
        
        Args:
            y_prob_test: Test set predicted probabilities
            
        Returns:
            Tuple of (prediction_sets, set_sizes, coverage_flags)
            - prediction_sets: Array where each row is [prob_class_0, prob_class_1]
            - set_sizes: Number of classes in each prediction set
            - coverage_flags: Binary array for coverage check (when y_test provided)
        """
        if self.threshold is None:
            raise ValueError("Must call calibrate() before predict_sets()")
        
        n_test = len(y_prob_test)
        prediction_sets = np.zeros((n_test, 2))  # [class_0, class_1]
        
        # For each test instance, include classes with nonconformity <= threshold
        for i, prob_pos in enumerate(y_prob_test):
            prob_neg = 1 - prob_pos
            
            # Check if class 0 (negative) should be included
            score_class_0 = 1 - prob_neg  # Nonconformity if true class is 0
            if score_class_0 <= self.threshold:
                prediction_sets[i, 0] = 1
            
            # Check if class 1 (positive) should be included  
            score_class_1 = 1 - prob_pos  # Nonconformity if true class is 1
            if score_class_1 <= self.threshold:
                prediction_sets[i, 1] = 1
        
        # Calculate set sizes
        set_sizes = prediction_sets.sum(axis=1)
        
        # Ensure at least one class is always included (for very miscalibrated models)
        empty_sets = (set_sizes == 0)
        if empty_sets.any():
            logger.warning(f"Found {empty_sets.sum()} empty prediction sets, "
                          f"including most likely class")
            for i in np.where(empty_sets)[0]:
                most_likely_class = int(y_prob_test[i] > 0.5)
                prediction_sets[i, most_likely_class] = 1
            set_sizes = prediction_sets.sum(axis=1)
        
        return prediction_sets, set_sizes, np.zeros(n_test)  # coverage_flags computed separately
    
    def evaluate_coverage(self, y_test: np.ndarray, prediction_sets: np.ndarray) -> Dict[str, float]:
        """Evaluate coverage and other metrics on test set.
        
        Args:
            y_test: Test set true labels
            prediction_sets: Prediction sets from predict_sets()
            
        Returns:
            Dictionary with coverage metrics
        """
        # Check which predictions are covered (true class in prediction set)
        coverage_flags = np.array([prediction_sets[i, int(y_test[i])] for i in range(len(y_test))])
        
        results = {
            'coverage': coverage_flags.mean(),
            'target_coverage': self.target_coverage,
            'coverage_gap': coverage_flags.mean() - self.target_coverage,
            'average_set_size': prediction_sets.sum(axis=1).mean(),
            'singleton_rate': (prediction_sets.sum(axis=1) == 1).mean(),
            'empty_set_rate': (prediction_sets.sum(axis=1) == 0).mean(),
            'full_set_rate': (prediction_sets.sum(axis=1) == 2).mean()
        }
        
        return results
    
    def evaluate_coverage_by_group(self, y_test: np.ndarray, prediction_sets: np.ndarray,
                                  groups: np.ndarray) -> pd.DataFrame:
        """Evaluate coverage separately for each demographic group.
        
        Args:
            y_test: Test set true labels
            prediction_sets: Prediction sets from predict_sets()
            groups: Group identifiers for each test sample
            
        Returns:
            DataFrame with coverage metrics by group
        """
        results = []
        
        for group_value in np.unique(groups):
            if pd.isna(group_value):
                continue
                
            group_mask = (groups == group_value)
            group_y = y_test[group_mask]
            group_sets = prediction_sets[group_mask]
            
            if len(group_y) > 0:
                group_results = self.evaluate_coverage(group_y, group_sets)
                group_results['group'] = group_value
                group_results['n_samples'] = len(group_y)
                results.append(group_results)
        
        return pd.DataFrame(results)


def run_conformal_prediction(y_val: np.ndarray, y_prob_val: np.ndarray,
                           y_test: np.ndarray, y_prob_test: np.ndarray,
                           groups_test: Optional[np.ndarray] = None,
                           alphas: List[float] = [0.10, 0.05]) -> Dict[float, Dict]:
    """Run conformal prediction for multiple alpha values.
    
    Args:
        y_val: Validation set labels (for calibration)
        y_prob_val: Validation set probabilities
        y_test: Test set labels
        y_prob_test: Test set probabilities  
        groups_test: Test set group identifiers
        alphas: List of miscoverage rates to evaluate
        
    Returns:
        Dictionary mapping alpha to results
    """
    logger.info(f"Running conformal prediction for alphas: {alphas}")
    
    all_results = {}
    
    for alpha in alphas:
        logger.info(f"Processing alpha = {alpha}")
        
        # Initialize and calibrate predictor
        cp = InductiveConformalPredictor(alpha=alpha)
        cp.calibrate(y_val, y_prob_val)
        
        # Generate prediction sets
        pred_sets, set_sizes, _ = cp.predict_sets(y_prob_test)
        
        # Evaluate overall coverage
        overall_results = cp.evaluate_coverage(y_test, pred_sets)
        
        # Evaluate coverage by group if groups provided
        group_results = None
        if groups_test is not None:
            group_results = cp.evaluate_coverage_by_group(y_test, pred_sets, groups_test)
        
        all_results[alpha] = {
            'overall': overall_results,
            'by_group': group_results,
            'predictor': cp,
            'prediction_sets': pred_sets,
            'set_sizes': set_sizes
        }
        
        logger.info(f"Alpha {alpha}: Coverage = {overall_results['coverage']:.3f}, "
                   f"Avg set size = {overall_results['average_set_size']:.3f}")
    
    return all_results


def test_conformal_prediction():
    """Simple unit tests for conformal prediction."""
    logger.info("Running conformal prediction tests...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate reasonably calibrated probabilities
    y_true = np.random.binomial(1, 0.3, n_samples)
    noise = np.random.normal(0, 0.1, n_samples)
    y_prob = np.clip(y_true * 0.7 + (1 - y_true) * 0.2 + noise, 0.01, 0.99)
    
    # Split into calibration and test
    y_cal, y_test, prob_cal, prob_test = train_test_split(
        y_true, y_prob, test_size=0.5, random_state=42
    )
    
    # Add synthetic groups
    groups_test = np.random.choice(['A', 'B'], size=len(y_test))
    
    # Run conformal prediction
    results = run_conformal_prediction(
        y_cal, prob_cal, y_test, prob_test, groups_test
    )
    
    # Check results
    for alpha, alpha_results in results.items():
        coverage = alpha_results['overall']['coverage']
        target = alpha_results['overall']['target_coverage']
        
        logger.info(f"Test alpha {alpha}: Coverage = {coverage:.3f}, Target = {target:.3f}")
        
        # Coverage should be close to target (within reasonable margin for finite samples)
        assert abs(coverage - target) < 0.1, f"Coverage too far from target for alpha {alpha}"
        
        # Set sizes should be reasonable
        avg_size = alpha_results['overall']['average_set_size']
        assert 1.0 <= avg_size <= 2.0, f"Unreasonable average set size: {avg_size}"
        
        # Group coverage should exist
        assert alpha_results['by_group'] is not None
        assert len(alpha_results['by_group']) == 2  # Two groups A, B
    
    logger.info("All conformal prediction tests passed!")


def main():
    """CLI interface for conformal prediction."""
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Conformal prediction evaluation')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--data-path', type=str, help='Path to dataset')
    parser.add_argument('--split-path', type=str, help='Path to data splits')
    parser.add_argument('--output-dir', type=str, default='conformal_results')
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.10, 0.05])
    parser.add_argument('--test-only', action='store_true', help='Run unit tests only')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_conformal_prediction()
        return
    
    # Load model and data (placeholder - would need actual implementation)
    logger.info("Loading model and data...")
    
    # This would be replaced with actual data loading logic
    # For now, generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_prob = np.clip(y_true * 0.8 + np.random.normal(0, 0.1, n_samples), 0.01, 0.99)
    groups = np.random.choice(['Group1', 'Group2'], size=n_samples)
    
    # Split data
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    
    y_val = y_true[val_idx]
    y_prob_val = y_prob[val_idx]
    y_test = y_true[test_idx]
    y_prob_test = y_prob[test_idx]
    groups_test = groups[test_idx]
    
    # Run conformal prediction
    results = run_conformal_prediction(
        y_val, y_prob_val, y_test, y_prob_test, groups_test, args.alphas
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for alpha, alpha_results in results.items():
        # Save overall results
        overall_df = pd.DataFrame([alpha_results['overall']])
        overall_df.to_csv(output_dir / f'conformal_overall_alpha_{alpha}.csv', index=False)
        
        # Save group results
        if alpha_results['by_group'] is not None:
            alpha_results['by_group'].to_csv(
                output_dir / f'conformal_by_group_alpha_{alpha}.csv', index=False
            )
    
    logger.info(f"Conformal prediction results saved to {output_dir}")


if __name__ == '__main__':
    setup_logging()
    main()
