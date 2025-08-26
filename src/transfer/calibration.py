"""
Calibration and Threshold Tuning for Transfer Learning

Implements model calibration and optimal threshold tuning techniques
to improve prediction reliability and performance in transfer learning scenarios.

Key components:
- Model calibration (Platt scaling, isotonic regression)
- Threshold optimization for different metrics
- Expected Calibration Error (ECE) computation
- Reliability diagrams and calibration evaluation
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, roc_curve, f1_score, 
    precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, 
                              n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy
    across probability bins.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of probability bins
        
    Returns:
        Expected Calibration Error
    """
    # Create probability bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in current bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in bin
            accuracy_in_bin = y_true[in_bin].mean()
            
            # Average confidence in bin
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add weighted contribution to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                             n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between confidence and accuracy across bins.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        Maximum Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_error = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_error = max(max_error, error)
    
    return max_error


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        
    Returns:
        Brier score
    """
    return np.mean((y_prob - y_true) ** 2)


class OptimalThresholdFinder:
    """
    Find optimal decision thresholds for different metrics.
    
    Supports optimization for various classification metrics including
    F1-score, balanced accuracy, Youden's J statistic, etc.
    """
    
    def __init__(self, metric: str = 'f1', pos_label: int = 1):
        """
        Initialize threshold finder.
        
        Args:
            metric: Metric to optimize ('f1', 'balanced_accuracy', 'youden', 
                   'precision', 'recall', 'accuracy')
            pos_label: Positive class label
        """
        self.metric = metric
        self.pos_label = pos_label
        self.optimal_threshold_ = None
        self.metric_scores_ = None
        self.thresholds_ = None
        
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'OptimalThresholdFinder':
        """
        Find optimal threshold for the specified metric.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            self
        """
        if self.metric in ['f1', 'precision', 'recall']:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            
            if self.metric == 'f1':
                # F1 score
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores)
                self.optimal_threshold_ = thresholds[best_idx]
                self.metric_scores_ = f1_scores
                
            elif self.metric == 'precision':
                # Maximize precision at high recall
                high_recall_mask = recall >= 0.7
                if high_recall_mask.any():
                    best_idx = np.argmax(precision[high_recall_mask])
                    # Adjust index for original arrays
                    best_idx = np.where(high_recall_mask)[0][best_idx]
                else:
                    best_idx = np.argmax(precision)
                self.optimal_threshold_ = thresholds[best_idx]
                self.metric_scores_ = precision
                
            elif self.metric == 'recall':
                # Maximize recall at high precision
                high_precision_mask = precision >= 0.7
                if high_precision_mask.any():
                    best_idx = np.argmax(recall[high_precision_mask])
                    best_idx = np.where(high_precision_mask)[0][best_idx]
                else:
                    best_idx = np.argmax(recall)
                self.optimal_threshold_ = thresholds[best_idx]
                self.metric_scores_ = recall
                
            self.thresholds_ = thresholds
            
        elif self.metric in ['balanced_accuracy', 'youden', 'accuracy']:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            
            if self.metric == 'youden':
                # Youden's J statistic (sensitivity + specificity - 1)
                youden_scores = tpr - fpr
                best_idx = np.argmax(youden_scores)
                self.metric_scores_ = youden_scores
                
            elif self.metric == 'balanced_accuracy':
                # Balanced accuracy = (sensitivity + specificity) / 2
                specificity = 1 - fpr
                balanced_acc = (tpr + specificity) / 2
                best_idx = np.argmax(balanced_acc)
                self.metric_scores_ = balanced_acc
                
            elif self.metric == 'accuracy':
                # Overall accuracy for each threshold
                accuracies = []
                for threshold in thresholds:
                    y_pred = (y_prob >= threshold).astype(int)
                    acc = accuracy_score(y_true, y_pred)
                    accuracies.append(acc)
                accuracies = np.array(accuracies)
                best_idx = np.argmax(accuracies)
                self.metric_scores_ = accuracies
                
            self.optimal_threshold_ = thresholds[best_idx]
            self.thresholds_ = thresholds
            
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        logger.info(f"Optimal threshold for {self.metric}: {self.optimal_threshold_:.4f}")
        return self
    
    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Make predictions using the optimal threshold.
        
        Args:
            y_prob: Predicted probabilities
            
        Returns:
            Binary predictions
        """
        if self.optimal_threshold_ is None:
            raise ValueError("Must fit threshold finder before prediction")
        
        return (y_prob >= self.optimal_threshold_).astype(int)
    
    def get_threshold_analysis(self) -> Dict:
        """
        Get analysis of threshold optimization.
        
        Returns:
            Dictionary with threshold analysis
        """
        if self.optimal_threshold_ is None:
            raise ValueError("Must fit threshold finder before analysis")
        
        best_score = np.max(self.metric_scores_)
        
        return {
            'optimal_threshold': self.optimal_threshold_,
            'best_score': best_score,
            'metric': self.metric,
            'n_thresholds_evaluated': len(self.thresholds_)
        }


class CalibratedTransferClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier with calibration and optimal threshold tuning for transfer learning.
    
    Combines a base classifier with:
    - Calibration (Platt scaling or isotonic regression)
    - Optimal threshold tuning
    - Calibration quality metrics
    """
    
    def __init__(self, base_classifier, calibration_method: str = 'platt',
                 threshold_metric: str = 'f1', cv_folds: int = 3,
                 random_state: int = 42):
        """
        Initialize calibrated transfer classifier.
        
        Args:
            base_classifier: Base classifier to calibrate
            calibration_method: Calibration method ('platt', 'isotonic')
            threshold_metric: Metric for threshold optimization
            cv_folds: Cross-validation folds for calibration
            random_state: Random seed
        """
        self.base_classifier = base_classifier
        self.calibration_method = calibration_method
        self.threshold_metric = threshold_metric
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Fitted components
        self.calibrated_classifier_ = None
        self.threshold_finder_ = None
        self.calibration_metrics_ = None
        self.is_fitted_ = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CalibratedTransferClassifier':
        """
        Fit the calibrated classifier.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        logger.info("Fitting calibrated transfer classifier...")
        
        # Calibrate the base classifier
        if self.calibration_method == 'platt':
            method = 'sigmoid'
        elif self.calibration_method == 'isotonic':
            method = 'isotonic'
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        self.calibrated_classifier_ = CalibratedClassifierCV(
            self.base_classifier,
            method=method,
            cv=self.cv_folds
        )
        
        self.calibrated_classifier_.fit(X, y)
        
        # Get calibrated probabilities for threshold optimization
        cv_probs = cross_val_predict(
            self.calibrated_classifier_, X, y,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                              random_state=self.random_state),
            method='predict_proba'
        )[:, 1]  # Probabilities for positive class
        
        # Find optimal threshold
        self.threshold_finder_ = OptimalThresholdFinder(metric=self.threshold_metric)
        self.threshold_finder_.fit(y, cv_probs)
        
        # Compute calibration metrics
        self.calibration_metrics_ = self._compute_calibration_metrics(y, cv_probs)
        
        self.is_fitted_ = True
        logger.info("Calibrated classifier fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using calibrated classifier and optimal threshold."""
        if not self.is_fitted_:
            raise ValueError("Must fit classifier before prediction")
        
        y_prob = self.predict_proba(X)[:, 1]
        return self.threshold_finder_.predict(y_prob)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        if not self.is_fitted_:
            raise ValueError("Must fit classifier before prediction")
        
        return self.calibrated_classifier_.predict_proba(X)
    
    def _compute_calibration_metrics(self, y_true: np.ndarray, 
                                   y_prob: np.ndarray) -> Dict:
        """Compute calibration quality metrics."""
        metrics = {
            'ece': expected_calibration_error(y_true, y_prob),
            'mce': maximum_calibration_error(y_true, y_prob),
            'brier_score': brier_score(y_true, y_prob)
        }
        
        # Reliability metrics
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10
        )
        
        # Perfect calibration would have fraction_of_positives == mean_predicted_value
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        metrics['reliability_error'] = calibration_error
        
        return metrics
    
    def get_calibration_summary(self) -> Dict:
        """
        Get summary of calibration and threshold optimization.
        
        Returns:
            Dictionary with calibration summary
        """
        if not self.is_fitted_:
            raise ValueError("Must fit classifier before getting summary")
        
        summary = {
            'calibration_method': self.calibration_method,
            'threshold_metric': self.threshold_metric,
            'calibration_metrics': self.calibration_metrics_,
            'threshold_analysis': self.threshold_finder_.get_threshold_analysis()
        }
        
        return summary
    
    def plot_calibration_curve(self, X: np.ndarray, y: np.ndarray, 
                              n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for plotting calibration curve.
        
        Args:
            X: Features for evaluation
            y: True labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            Tuple of (fraction_of_positives, mean_predicted_value)
        """
        if not self.is_fitted_:
            raise ValueError("Must fit classifier before plotting")
        
        y_prob = self.predict_proba(X)[:, 1]
        return calibration_curve(y, y_prob, n_bins=n_bins)


def evaluate_calibration_transfer(source_classifier, X_source: np.ndarray, y_source: np.ndarray,
                                X_target: np.ndarray, y_target: np.ndarray,
                                calibration_methods: List[str] = ['platt', 'isotonic'],
                                threshold_metrics: List[str] = ['f1', 'balanced_accuracy']) -> pd.DataFrame:
    """
    Evaluate different calibration and threshold combinations for transfer learning.
    
    Args:
        source_classifier: Classifier trained on source domain
        X_source: Source domain features
        y_source: Source domain labels
        X_target: Target domain features
        y_target: Target domain labels
        calibration_methods: List of calibration methods to evaluate
        threshold_metrics: List of threshold metrics to evaluate
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for calib_method in calibration_methods:
        for thresh_metric in threshold_metrics:
            logger.info(f"Evaluating {calib_method} calibration with {thresh_metric} threshold...")
            
            # Create calibrated classifier
            calib_clf = CalibratedTransferClassifier(
                clone(source_classifier),
                calibration_method=calib_method,
                threshold_metric=thresh_metric
            )
            
            # Fit on source domain
            calib_clf.fit(X_source, y_source)
            
            # Evaluate on target domain
            y_pred_target = calib_clf.predict(X_target)
            y_prob_target = calib_clf.predict_proba(X_target)[:, 1]
            
            # Compute metrics
            metrics = {
                'calibration_method': calib_method,
                'threshold_metric': thresh_metric,
                'accuracy': accuracy_score(y_target, y_pred_target),
                'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_target),
                'f1_score': f1_score(y_target, y_pred_target),
                'precision': precision_score(y_target, y_pred_target, zero_division=0),
                'recall': recall_score(y_target, y_pred_target),
                'roc_auc': roc_auc_score(y_target, y_prob_target),
                'ece': expected_calibration_error(y_target, y_prob_target),
                'brier_score': brier_score(y_target, y_prob_target)
            }
            
            # Add threshold info
            threshold_info = calib_clf.threshold_finder_.get_threshold_analysis()
            metrics['optimal_threshold'] = threshold_info['optimal_threshold']
            
            results.append(metrics)
    
    return pd.DataFrame(results)


def demo_calibration_threshold():
    """Demonstrate calibration and threshold tuning."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate synthetic data with class imbalance
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                             weights=[0.7, 0.3], random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Calibration and Threshold Tuning Demo")
    print("=" * 45)
    
    # Train base classifier
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    base_clf.fit(X_train, y_train)
    
    # Base classifier performance
    y_pred_base = base_clf.predict(X_test)
    y_prob_base = base_clf.predict_proba(X_test)[:, 1]
    
    print(f"Base classifier:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_base):.3f}")
    print(f"  F1-score: {f1_score(y_test, y_pred_base):.3f}")
    print(f"  ECE: {expected_calibration_error(y_test, y_prob_base):.3f}")
    
    # Calibrated classifier
    calib_clf = CalibratedTransferClassifier(
        clone(base_clf),
        calibration_method='platt',
        threshold_metric='f1'
    )
    calib_clf.fit(X_train, y_train)
    
    y_pred_calib = calib_clf.predict(X_test)
    y_prob_calib = calib_clf.predict_proba(X_test)[:, 1]
    
    print(f"\nCalibrated classifier:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_calib):.3f}")
    print(f"  F1-score: {f1_score(y_test, y_pred_calib):.3f}")
    print(f"  ECE: {expected_calibration_error(y_test, y_prob_calib):.3f}")
    
    # Show calibration summary
    summary = calib_clf.get_calibration_summary()
    print(f"\nCalibration Summary:")
    print(f"  Method: {summary['calibration_method']}")
    print(f"  Optimal threshold: {summary['threshold_analysis']['optimal_threshold']:.3f}")
    print(f"  ECE: {summary['calibration_metrics']['ece']:.3f}")
    print(f"  Brier score: {summary['calibration_metrics']['brier_score']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_calibration_threshold()