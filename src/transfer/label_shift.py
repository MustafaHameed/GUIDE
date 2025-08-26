"""
Label Shift Correction for Transfer Learning

Implements the Saerens-Decock method for correcting label shift between domains.
Adjusts predicted posteriors on target domain using estimated class priors.

Reference: "Adjusting the Outputs of a Classifier to New a Priori Probabilities: 
A Simple Procedure" by Saerens et al.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LabelShiftCorrector(BaseEstimator):
    """
    Corrects for label shift using the Saerens-Decock method.
    
    Estimates target domain class priors and adjusts model predictions accordingly.
    """
    
    def __init__(self, base_classifier, method: str = 'saerens_decock', 
                 max_iterations: int = 100, tolerance: float = 1e-6):
        """
        Initialize label shift corrector.
        
        Args:
            base_classifier: Base classifier trained on source domain
            method: Method for correction ('saerens_decock', 'confusion_matrix', 'em')
            max_iterations: Maximum iterations for iterative methods
            tolerance: Convergence tolerance
        """
        self.base_classifier = base_classifier
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Fitted parameters
        self.source_priors_ = None
        self.target_priors_ = None
        self.is_fitted_ = False
        
    def fit(self, X_source: np.ndarray, y_source: np.ndarray, 
            X_target: np.ndarray) -> 'LabelShiftCorrector':
        """
        Fit the label shift corrector.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features (unlabeled)
            
        Returns:
            self
        """
        logger.info(f"Fitting label shift corrector using {self.method} method...")
        
        # Compute source domain class priors
        unique_classes, class_counts = np.unique(y_source, return_counts=True)
        self.classes_ = unique_classes
        self.source_priors_ = class_counts / len(y_source)
        
        # Estimate target domain priors
        if self.method == 'saerens_decock':
            self.target_priors_ = self._estimate_target_priors_saerens_decock(
                X_source, y_source, X_target
            )
        elif self.method == 'confusion_matrix':
            self.target_priors_ = self._estimate_target_priors_confusion_matrix(
                X_source, y_source, X_target
            )
        elif self.method == 'em':
            self.target_priors_ = self._estimate_target_priors_em(
                X_source, y_source, X_target
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"Source priors: {self.source_priors_}")
        logger.info(f"Estimated target priors: {self.target_priors_}")
        
        self.is_fitted_ = True
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities with label shift correction.
        
        Args:
            X: Features to predict on
            
        Returns:
            Corrected class probabilities
        """
        if not self.is_fitted_:
            raise ValueError("LabelShiftCorrector must be fitted before prediction")
        
        # Get uncorrected predictions from base classifier
        uncorrected_probs = self.base_classifier.predict_proba(X)
        
        # Apply Saerens-Decock correction
        corrected_probs = self._apply_saerens_decock_correction(uncorrected_probs)
        
        return corrected_probs
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels with label shift correction.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def _estimate_target_priors_saerens_decock(self, X_source: np.ndarray, 
                                              y_source: np.ndarray, 
                                              X_target: np.ndarray) -> np.ndarray:
        """
        Estimate target priors using Saerens-Decock iterative method.
        """
        # Get predictions on target domain
        target_predictions = self.base_classifier.predict_proba(X_target)
        
        # Initialize target priors with source priors
        current_priors = self.source_priors_.copy()
        
        for iteration in range(self.max_iterations):
            # Update predictions using current prior estimates
            corrected_probs = self._apply_saerens_decock_correction(
                target_predictions, current_priors
            )
            
            # Estimate new priors from corrected predictions
            new_priors = np.mean(corrected_probs, axis=0)
            
            # Check convergence
            prior_change = np.linalg.norm(new_priors - current_priors)
            if prior_change < self.tolerance:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
                
            current_priors = new_priors
        else:
            logger.warning(f"Did not converge after {self.max_iterations} iterations")
        
        return current_priors
    
    def _estimate_target_priors_confusion_matrix(self, X_source: np.ndarray,
                                               y_source: np.ndarray,
                                               X_target: np.ndarray) -> np.ndarray:
        """
        Estimate target priors using confusion matrix approach.
        """
        # Get cross-validated predictions on source to estimate confusion matrix
        cv_predictions = cross_val_predict(
            self.base_classifier, X_source, y_source, 
            method='predict', cv=5
        )
        
        # Compute confusion matrix
        cm = confusion_matrix(y_source, cv_predictions, labels=self.classes_)
        
        # Normalize to get conditional probabilities P(predicted|true)
        cm_normalized = cm / cm.sum(axis=1, keepdims=True)
        
        # Get predictions on target
        target_predictions = self.base_classifier.predict(X_target)
        
        # Count predicted classes in target
        target_pred_counts = np.bincount(target_predictions, minlength=len(self.classes_))
        target_pred_probs = target_pred_counts / len(X_target)
        
        # Solve linear system: P(predicted) = CM * P(true)
        try:
            estimated_priors = np.linalg.solve(cm_normalized.T, target_pred_probs)
            
            # Ensure valid probabilities
            estimated_priors = np.clip(estimated_priors, 0, 1)
            estimated_priors = estimated_priors / estimated_priors.sum()
            
        except np.linalg.LinAlgError:
            logger.warning("Confusion matrix approach failed, using source priors")
            estimated_priors = self.source_priors_
        
        return estimated_priors
    
    def _estimate_target_priors_em(self, X_source: np.ndarray,
                                 y_source: np.ndarray, 
                                 X_target: np.ndarray) -> np.ndarray:
        """
        Estimate target priors using EM algorithm.
        """
        # Get probabilistic predictions on target
        target_probs = self.base_classifier.predict_proba(X_target)
        
        # Initialize with source priors
        current_priors = self.source_priors_.copy()
        
        for iteration in range(self.max_iterations):
            # E-step: compute responsibilities
            weighted_probs = target_probs * current_priors[np.newaxis, :]
            responsibilities = weighted_probs / weighted_probs.sum(axis=1, keepdims=True)
            
            # M-step: update priors
            new_priors = responsibilities.mean(axis=0)
            
            # Check convergence
            prior_change = np.linalg.norm(new_priors - current_priors)
            if prior_change < self.tolerance:
                logger.info(f"EM converged after {iteration + 1} iterations")
                break
                
            current_priors = new_priors
        else:
            logger.warning(f"EM did not converge after {self.max_iterations} iterations")
        
        return current_priors
    
    def _apply_saerens_decock_correction(self, predictions: np.ndarray,
                                       target_priors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Saerens-Decock correction to predictions.
        
        Args:
            predictions: Uncorrected class probabilities
            target_priors: Target domain priors (uses fitted if None)
            
        Returns:
            Corrected class probabilities
        """
        if target_priors is None:
            target_priors = self.target_priors_
        
        # Apply Bayes rule correction
        # P_target(y|x) = P_source(y|x) * P_target(y) / P_source(y)
        correction_factors = target_priors / self.source_priors_
        corrected = predictions * correction_factors[np.newaxis, :]
        
        # Normalize to ensure probabilities sum to 1
        corrected = corrected / corrected.sum(axis=1, keepdims=True)
        
        return corrected
    
    def get_shift_metrics(self) -> Dict:
        """
        Get metrics quantifying the amount of label shift.
        
        Returns:
            Dictionary with shift metrics
        """
        if not self.is_fitted_:
            raise ValueError("Must be fitted to compute shift metrics")
        
        # Total variation distance
        tv_distance = 0.5 * np.sum(np.abs(self.target_priors_ - self.source_priors_))
        
        # KL divergence
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        source_priors_safe = self.source_priors_ + eps
        target_priors_safe = self.target_priors_ + eps
        
        kl_divergence = np.sum(target_priors_safe * np.log(target_priors_safe / source_priors_safe))
        
        # Maximum difference
        max_difference = np.max(np.abs(self.target_priors_ - self.source_priors_))
        
        # Chi-square statistic
        chi2_stat = np.sum((self.target_priors_ - self.source_priors_)**2 / self.source_priors_)
        
        return {
            'total_variation_distance': tv_distance,
            'kl_divergence': kl_divergence,
            'max_class_difference': max_difference,
            'chi_square_statistic': chi2_stat,
            'shift_detected': tv_distance > 0.1  # Threshold for significant shift
        }


def estimate_label_shift_simple(y_source: np.ndarray, model, X_target: np.ndarray) -> np.ndarray:
    """
    Simple label shift estimation using predicted proportions.
    
    Args:
        y_source: Source domain labels
        model: Trained classifier
        X_target: Target domain features
        
    Returns:
        Estimated target domain class proportions
    """
    # Get source proportions
    source_classes, source_counts = np.unique(y_source, return_counts=True)
    source_priors = source_counts / len(y_source)
    
    # Get target predictions
    target_predictions = model.predict(X_target)
    target_counts = np.bincount(target_predictions, minlength=len(source_classes))
    target_priors = target_counts / len(X_target)
    
    return target_priors


def apply_label_shift_correction(model, X_source: np.ndarray, y_source: np.ndarray,
                               X_target: np.ndarray, method: str = 'saerens_decock'):
    """
    Convenience function to apply label shift correction.
    
    Args:
        model: Trained classifier
        X_source: Source domain features
        y_source: Source domain labels
        X_target: Target domain features
        method: Correction method
        
    Returns:
        LabelShiftCorrector instance
    """
    corrector = LabelShiftCorrector(model, method=method)
    corrector.fit(X_source, y_source, X_target)
    return corrector


def demo_label_shift_correction():
    """
    Demonstrate label shift correction with synthetic data.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    
    # Generate synthetic data
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=3, 
                             n_informative=8, random_state=42)
    
    # Split into source and target
    X_source, X_target = X[:1000], X[1000:]
    y_source, y_target = y[:1000], y[1000:]
    
    # Introduce label shift in target
    # Make class 0 more common, class 2 less common
    target_mask = (y_target == 0) | (np.random.random(len(y_target)) < 0.3)
    X_target_shifted = X_target[target_mask]
    y_target_shifted = y_target[target_mask]
    
    print("Label Shift Correction Demo")
    print("=" * 35)
    
    # Train base classifier on source
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_source, y_source)
    
    # Original predictions (without correction)
    y_pred_original = clf.predict(X_target_shifted)
    acc_original = accuracy_score(y_target_shifted, y_pred_original)
    
    # Apply label shift correction
    corrector = LabelShiftCorrector(clf, method='saerens_decock')
    corrector.fit(X_source, y_source, X_target_shifted)
    
    y_pred_corrected = corrector.predict(X_target_shifted)
    acc_corrected = accuracy_score(y_target_shifted, y_pred_corrected)
    
    print(f"Original accuracy: {acc_original:.3f}")
    print(f"Corrected accuracy: {acc_corrected:.3f}")
    print(f"Improvement: {acc_corrected - acc_original:.3f}")
    
    # Show shift metrics
    shift_metrics = corrector.get_shift_metrics()
    print(f"\nShift Metrics:")
    for key, value in shift_metrics.items():
        if isinstance(value, bool):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_label_shift_correction()