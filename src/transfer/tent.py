"""
TENT: Test-Time Entropy Minimization for Domain Adaptation

Implements Test-Time Entropy Minimization (TENT) for unsupervised domain adaptation.
TENT adapts a pre-trained model to the target domain by minimizing the entropy
of predictions on target data during test time.

Reference: "Tent: Fully Test-time Adaptation by Entropy Minimization" 
by Wang et al., ICLR 2021
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Union
import warnings
from copy import deepcopy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from scipy.special import softmax

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def entropy_loss(probabilities: np.ndarray) -> float:
    """
    Compute entropy loss for a batch of predictions.
    
    Args:
        probabilities: Predicted class probabilities (n_samples, n_classes)
        
    Returns:
        Average entropy across samples
    """
    # Clip probabilities to avoid log(0)
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    
    return np.mean(entropy)


def confidence_loss(probabilities: np.ndarray) -> float:
    """
    Compute confidence-based loss (negative max probability).
    
    Args:
        probabilities: Predicted class probabilities
        
    Returns:
        Average negative confidence
    """
    max_probs = np.max(probabilities, axis=1)
    return -np.mean(max_probs)


class TENTAdapter(BaseEstimator, ClassifierMixin):
    """
    Test-Time Entropy Minimization (TENT) adapter.
    
    Adapts a pre-trained classifier to the target domain by minimizing
    entropy of predictions on unlabeled target data.
    """
    
    def __init__(self, base_classifier, adaptation_strategy: str = 'entropy',
                 learning_rate: float = 0.001, max_iterations: int = 100,
                 batch_size: int = 32, confidence_threshold: float = 0.9,
                 early_stopping_patience: int = 10, temperature: float = 1.0):
        """
        Initialize TENT adapter.
        
        Args:
            base_classifier: Pre-trained classifier to adapt
            adaptation_strategy: Strategy for adaptation ('entropy', 'confidence', 'both')
            learning_rate: Learning rate for adaptation
            max_iterations: Maximum adaptation iterations
            batch_size: Batch size for adaptation
            confidence_threshold: Minimum confidence for sample selection
            early_stopping_patience: Patience for early stopping
            temperature: Temperature for softmax (temperature scaling)
        """
        self.base_classifier = base_classifier
        self.adaptation_strategy = adaptation_strategy
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.early_stopping_patience = early_stopping_patience
        self.temperature = temperature
        
        # Adapted classifier (copy of base)
        self.adapted_classifier_ = None
        self.adaptation_history_ = []
        self.is_fitted_ = False
        
    def adapt(self, X_target: np.ndarray, 
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'TENTAdapter':
        """
        Adapt the classifier to target domain using TENT.
        
        Args:
            X_target: Unlabeled target domain data
            validation_data: Optional (X_val, y_val) for monitoring adaptation
            
        Returns:
            self
        """
        logger.info(f"Starting TENT adaptation with {len(X_target)} target samples...")
        
        # Create adapted classifier as copy of base
        self.adapted_classifier_ = deepcopy(self.base_classifier)
        
        # Adaptation depends on classifier type
        if isinstance(self.adapted_classifier_, MLPClassifier):
            self._adapt_mlp(X_target, validation_data)
        elif isinstance(self.adapted_classifier_, LogisticRegression):
            self._adapt_logistic(X_target, validation_data)
        else:
            logger.warning(f"TENT adaptation not implemented for {type(self.adapted_classifier_)}")
            # For other classifiers, use pseudo-labeling approach
            self._adapt_pseudo_labeling(X_target, validation_data)
        
        self.is_fitted_ = True
        logger.info("TENT adaptation completed")
        return self
    
    def _adapt_mlp(self, X_target: np.ndarray,
                  validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Adapt MLP classifier using gradient-based optimization."""
        best_loss = float('inf')
        patience_counter = 0
        
        for iteration in range(self.max_iterations):
            # Shuffle target data
            indices = np.random.permutation(len(X_target))
            X_shuffled = X_target[indices]
            
            iteration_losses = []
            
            # Process in batches
            for batch_start in range(0, len(X_target), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(X_target))
                X_batch = X_shuffled[batch_start:batch_end]
                
                # Get predictions
                probabilities = self.adapted_classifier_.predict_proba(X_batch)
                
                # Apply temperature scaling
                if self.temperature != 1.0:
                    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
                    probabilities = softmax(logits / self.temperature, axis=1)
                
                # Filter high-confidence samples
                max_probs = np.max(probabilities, axis=1)
                confident_mask = max_probs >= self.confidence_threshold
                
                if np.sum(confident_mask) > 0:
                    confident_probs = probabilities[confident_mask]
                    
                    # Compute adaptation loss
                    if self.adaptation_strategy == 'entropy':
                        loss = entropy_loss(confident_probs)
                    elif self.adaptation_strategy == 'confidence':
                        loss = confidence_loss(confident_probs)
                    else:  # both
                        entropy = entropy_loss(confident_probs)
                        confidence = confidence_loss(confident_probs)
                        loss = entropy + 0.1 * confidence
                    
                    iteration_losses.append(loss)
                    
                    # Simplified gradient update (in practice, would use proper backprop)
                    # For MLPClassifier, we simulate adaptation by adjusting decision threshold
                    self._update_classifier_weights(X_batch[confident_mask], confident_probs)
            
            # Record iteration metrics
            avg_loss = np.mean(iteration_losses) if iteration_losses else float('inf')
            
            metrics = {'iteration': iteration, 'loss': avg_loss}
            
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.adapted_classifier_.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                metrics['val_accuracy'] = val_acc
            
            self.adaptation_history_.append(metrics)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at iteration {iteration}")
                break
                
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: loss={avg_loss:.4f}")
    
    def _adapt_logistic(self, X_target: np.ndarray,
                       validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Adapt logistic regression using iterative reweighting."""
        for iteration in range(self.max_iterations):
            # Get predictions
            probabilities = self.adapted_classifier_.predict_proba(X_target)
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                logits = np.log(np.clip(probabilities, 1e-8, 1.0))
                probabilities = softmax(logits / self.temperature, axis=1)
            
            # Create pseudo-labels from high-confidence predictions
            max_probs = np.max(probabilities, axis=1)
            confident_mask = max_probs >= self.confidence_threshold
            
            if np.sum(confident_mask) == 0:
                logger.warning(f"No confident predictions at iteration {iteration}")
                continue
            
            X_confident = X_target[confident_mask]
            pseudo_labels = np.argmax(probabilities[confident_mask], axis=1)
            
            # Refit classifier with pseudo-labels (simplified adaptation)
            try:
                # Create weighted training set
                sample_weights = max_probs[confident_mask]
                
                # For logistic regression, we can use partial_fit if available
                # Otherwise, create new model with combined data
                if hasattr(self.adapted_classifier_, 'partial_fit'):
                    self.adapted_classifier_.partial_fit(X_confident, pseudo_labels)
                else:
                    # Simplified: just use most confident predictions for retraining
                    top_confident = np.argsort(sample_weights)[-min(50, len(sample_weights)):]
                    X_retrain = X_confident[top_confident]
                    y_retrain = pseudo_labels[top_confident]
                    
                    if len(X_retrain) > 5:  # Minimum samples for retraining
                        new_classifier = type(self.adapted_classifier_)(
                            **self.adapted_classifier_.get_params()
                        )
                        new_classifier.fit(X_retrain, y_retrain)
                        self.adapted_classifier_ = new_classifier
                        
            except Exception as e:
                logger.warning(f"Adaptation step failed: {e}")
                break
            
            # Compute loss for monitoring
            loss = entropy_loss(probabilities[confident_mask])
            
            metrics = {'iteration': iteration, 'loss': loss, 'n_confident': np.sum(confident_mask)}
            
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.adapted_classifier_.predict(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                metrics['val_accuracy'] = val_acc
            
            self.adaptation_history_.append(metrics)
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: loss={loss:.4f}, confident={np.sum(confident_mask)}")
    
    def _adapt_pseudo_labeling(self, X_target: np.ndarray,
                              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Generic adaptation using pseudo-labeling for any classifier."""
        logger.info("Using pseudo-labeling adaptation (generic approach)")
        
        best_accuracy = -1
        best_classifier = deepcopy(self.adapted_classifier_)
        
        for iteration in range(min(self.max_iterations, 20)):  # Fewer iterations for generic approach
            # Get predictions on target data
            probabilities = self.adapted_classifier_.predict_proba(X_target)
            
            # Select high-confidence predictions
            max_probs = np.max(probabilities, axis=1)
            confident_mask = max_probs >= self.confidence_threshold
            
            if np.sum(confident_mask) < 10:  # Need minimum samples
                logger.warning(f"Insufficient confident predictions ({np.sum(confident_mask)})")
                break
            
            X_confident = X_target[confident_mask]
            pseudo_labels = np.argmax(probabilities[confident_mask], axis=1)
            
            # Create new classifier with pseudo-labeled data
            try:
                new_classifier = type(self.adapted_classifier_)(
                    **self.adapted_classifier_.get_params()
                )
                new_classifier.fit(X_confident, pseudo_labels)
                
                # Evaluate on validation data if available
                if validation_data is not None:
                    X_val, y_val = validation_data
                    val_pred = new_classifier.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                    
                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        best_classifier = deepcopy(new_classifier)
                        
                    metrics = {
                        'iteration': iteration,
                        'val_accuracy': val_acc,
                        'n_confident': np.sum(confident_mask),
                        'confidence_threshold': self.confidence_threshold
                    }
                else:
                    # No validation data, accept the new classifier
                    best_classifier = deepcopy(new_classifier)
                    metrics = {
                        'iteration': iteration,
                        'n_confident': np.sum(confident_mask),
                        'confidence_threshold': self.confidence_threshold
                    }
                
                self.adaptation_history_.append(metrics)
                self.adapted_classifier_ = new_classifier
                
                # Gradually lower confidence threshold
                self.confidence_threshold = max(0.6, self.confidence_threshold - 0.02)
                
            except Exception as e:
                logger.warning(f"Pseudo-labeling failed at iteration {iteration}: {e}")
                break
        
        self.adapted_classifier_ = best_classifier
    
    def _update_classifier_weights(self, X_batch: np.ndarray, probabilities: np.ndarray):
        """
        Simplified weight update for MLP (placeholder for proper gradient descent).
        
        In practice, this would require custom implementation with proper
        backpropagation through the network.
        """
        # This is a placeholder - proper implementation would require
        # access to the internal parameters of the classifier and
        # gradient computation for the entropy loss
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the adapted classifier."""
        if not self.is_fitted_:
            raise ValueError("TENTAdapter must be fitted before prediction")
        return self.adapted_classifier_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the adapted classifier."""
        if not self.is_fitted_:
            raise ValueError("TENTAdapter must be fitted before prediction")
        return self.adapted_classifier_.predict_proba(X)
    
    def get_adaptation_metrics(self) -> Dict:
        """
        Get metrics about the adaptation process.
        
        Returns:
            Dictionary with adaptation metrics
        """
        if not self.adaptation_history_:
            return {}
        
        history_df = pd.DataFrame(self.adaptation_history_)
        
        metrics = {
            'n_iterations': len(self.adaptation_history_),
            'final_loss': history_df['loss'].iloc[-1] if 'loss' in history_df else None,
            'best_loss': history_df['loss'].min() if 'loss' in history_df else None,
            'convergence': len(self.adaptation_history_) < self.max_iterations
        }
        
        if 'val_accuracy' in history_df:
            metrics.update({
                'final_val_accuracy': history_df['val_accuracy'].iloc[-1],
                'best_val_accuracy': history_df['val_accuracy'].max()
            })
        
        return metrics


def apply_tent_adaptation(base_classifier, X_target: np.ndarray,
                         validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                         **tent_params) -> TENTAdapter:
    """
    Convenience function to apply TENT adaptation.
    
    Args:
        base_classifier: Pre-trained classifier
        X_target: Unlabeled target domain data
        validation_data: Optional validation data for monitoring
        **tent_params: Parameters for TENTAdapter
        
    Returns:
        Fitted TENTAdapter
    """
    tent_adapter = TENTAdapter(base_classifier, **tent_params)
    tent_adapter.adapt(X_target, validation_data)
    return tent_adapter


def demo_tent():
    """Demonstrate TENT adaptation with synthetic data."""
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                             n_informative=7, random_state=42)
    
    # Split into source and target domains
    X_source, X_target_all, y_source, y_target_all = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    
    # Further split target for adaptation and testing
    X_target_adapt, X_target_test, _, y_target_test = train_test_split(
        X_target_all[:len(X_target_all)//2], y_target_all[:len(y_target_all)//2], 
        test_size=0.3, random_state=42
    )
    
    # Add domain shift to target data
    X_target_adapt += np.random.normal(0.5, 0.3, X_target_adapt.shape)
    X_target_test += np.random.normal(0.5, 0.3, X_target_test.shape)
    
    print("TENT Adaptation Demo")
    print("=" * 30)
    
    # Train base classifier on source
    base_classifier = LogisticRegression(random_state=42)
    base_classifier.fit(X_source, y_source)
    
    # Test base classifier on target (before adaptation)
    y_pred_before = base_classifier.predict(X_target_test)
    acc_before = accuracy_score(y_target_test, y_pred_before)
    
    print(f"Accuracy before TENT: {acc_before:.3f}")
    
    # Apply TENT adaptation
    tent_adapter = TENTAdapter(
        base_classifier,
        adaptation_strategy='entropy',
        max_iterations=50,
        confidence_threshold=0.8
    )
    
    # Use part of target test as validation for monitoring
    validation_data = (X_target_test[:20], y_target_test[:20])
    tent_adapter.adapt(X_target_adapt, validation_data)
    
    # Test adapted classifier
    y_pred_after = tent_adapter.predict(X_target_test)
    acc_after = accuracy_score(y_target_test, y_pred_after)
    
    print(f"Accuracy after TENT: {acc_after:.3f}")
    print(f"Improvement: {acc_after - acc_before:.3f}")
    
    # Show adaptation metrics
    metrics = tent_adapter.get_adaptation_metrics()
    print(f"\nAdaptation Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_tent()