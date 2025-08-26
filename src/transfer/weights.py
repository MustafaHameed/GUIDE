"""
Importance Weighting for Transfer Learning

Implements importance weighting to correct for covariate shift between domains.
Uses a domain classifier to estimate density ratios for reweighting source samples.

Based on the approach: w(x) = p(target|x) / p(source|x)
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ImportanceWeighter(BaseEstimator, TransformerMixin):
    """
    Compute importance weights for source samples to match target distribution.
    
    Uses a probabilistic classifier to estimate the likelihood that each source
    sample belongs to the target domain, then uses this for reweighting.
    """
    
    def __init__(self, classifier='logistic', clip_weights: bool = True, 
                 weight_clip_quantile: float = 0.95, random_state: int = 42):
        """
        Initialize the importance weighter.
        
        Args:
            classifier: Type of classifier ('logistic', 'random_forest')
            clip_weights: Whether to clip extreme weights
            weight_clip_quantile: Quantile for weight clipping
            random_state: Random seed for reproducibility
        """
        self.classifier = classifier
        self.clip_weights = clip_weights
        self.weight_clip_quantile = weight_clip_quantile
        self.random_state = random_state
        self.domain_classifier_ = None
        self.scaler_ = None
        self.is_fitted_ = False
        
    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'ImportanceWeighter':
        """
        Fit the domain classifier to estimate importance weights.
        
        Args:
            X_source: Source domain features
            X_target: Target domain features
            
        Returns:
            self
        """
        logger.info("Training domain classifier for importance weighting...")
        
        # Combine source and target data
        n_source = X_source.shape[0]
        n_target = X_target.shape[0]
        
        X_combined = np.vstack([X_source, X_target])
        # Domain labels: 0 = source, 1 = target
        y_domain = np.concatenate([np.zeros(n_source), np.ones(n_target)])
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_combined)
        
        # Initialize classifier
        if self.classifier == 'logistic':
            self.domain_classifier_ = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.classifier == 'random_forest':
            self.domain_classifier_ = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier}")
        
        # Fit domain classifier
        self.domain_classifier_.fit(X_scaled, y_domain)
        
        # Evaluate domain classification performance
        cv_scores = cross_val_score(
            self.domain_classifier_, X_scaled, y_domain, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc'
        )
        
        domain_auc = cv_scores.mean()
        logger.info(f"Domain classifier AUC: {domain_auc:.3f} (±{cv_scores.std():.3f})")
        
        if domain_auc < 0.6:
            logger.warning("Low domain classification AUC - domains may be too similar for importance weighting")
        elif domain_auc > 0.9:
            logger.warning("High domain classification AUC - domains may be too different, consider other adaptation methods")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X_source: np.ndarray) -> np.ndarray:
        """
        Compute importance weights for source samples.
        
        Args:
            X_source: Source domain features
            
        Returns:
            Array of importance weights
        """
        if not self.is_fitted_:
            raise ValueError("ImportanceWeighter must be fitted before transform")
        
        # Scale source features
        X_source_scaled = self.scaler_.transform(X_source)
        
        # Get probability of belonging to target domain
        # p(target|x) from domain classifier
        target_probs = self.domain_classifier_.predict_proba(X_source_scaled)[:, 1]
        
        # Calculate importance weights: w(x) = p(target|x) / p(source|x)
        # p(source|x) = 1 - p(target|x)
        source_probs = 1 - target_probs
        
        # Avoid division by zero
        source_probs = np.clip(source_probs, 1e-8, 1.0)
        weights = target_probs / source_probs
        
        # Handle extreme weights
        if self.clip_weights:
            weight_threshold = np.quantile(weights, self.weight_clip_quantile)
            weights = np.clip(weights, 0, weight_threshold)
            logger.info(f"Clipped weights above {weight_threshold:.3f} (quantile {self.weight_clip_quantile})")
        
        # Normalize weights to have mean 1
        weights = weights / np.mean(weights)
        
        logger.info(f"Computed {len(weights)} importance weights")
        logger.info(f"Weight statistics - Mean: {np.mean(weights):.3f}, Std: {np.std(weights):.3f}")
        logger.info(f"Weight range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")
        
        return weights
    
    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """
        Fit the weighter and compute importance weights.
        
        Args:
            X_source: Source domain features
            X_target: Target domain features
            
        Returns:
            Array of importance weights for source samples
        """
        return self.fit(X_source, X_target).transform(X_source)


def compute_importance_weights(X_source: np.ndarray, X_target: np.ndarray,
                             method: str = 'logistic', **kwargs) -> np.ndarray:
    """
    Convenience function to compute importance weights.
    
    Args:
        X_source: Source domain features
        X_target: Target domain features  
        method: Method for domain classification ('logistic', 'random_forest')
        **kwargs: Additional arguments for ImportanceWeighter
        
    Returns:
        Array of importance weights for source samples
    """
    weighter = ImportanceWeighter(classifier=method, **kwargs)
    return weighter.fit_transform(X_source, X_target)


def evaluate_weight_quality(weights: np.ndarray, X_source: np.ndarray, X_target: np.ndarray) -> dict:
    """
    Evaluate the quality of importance weights.
    
    Args:
        weights: Computed importance weights
        X_source: Source domain features
        X_target: Target domain features
        
    Returns:
        Dictionary with quality metrics
    """
    results = {}
    
    # Weight statistics
    results['weight_stats'] = {
        'mean': float(np.mean(weights)),
        'std': float(np.std(weights)),
        'min': float(np.min(weights)),
        'max': float(np.max(weights)),
        'median': float(np.median(weights)),
        'effective_sample_size': float(len(weights) * np.mean(weights)**2 / np.mean(weights**2))
    }
    
    # Covariate balance after weighting (simplified check)
    try:
        # Compare weighted source mean to target mean for each feature
        weighted_source_means = np.average(X_source, weights=weights, axis=0)
        target_means = np.mean(X_target, axis=0)
        
        # Mean absolute difference in standardized units
        source_stds = np.std(X_source, axis=0)
        target_stds = np.std(X_target, axis=0)
        
        # Avoid division by zero
        combined_stds = (source_stds + target_stds) / 2
        combined_stds[combined_stds == 0] = 1
        
        standardized_diff = np.abs(weighted_source_means - target_means) / combined_stds
        
        results['covariate_balance'] = {
            'mean_standardized_diff': float(np.mean(standardized_diff)),
            'max_standardized_diff': float(np.max(standardized_diff)),
            'features_with_large_diff': int(np.sum(standardized_diff > 0.2))
        }
        
    except Exception as e:
        logger.warning(f"Covariate balance evaluation failed: {e}")
        results['covariate_balance'] = {'error': str(e)}
    
    return results


class WeightedClassifierWrapper:
    """
    Wrapper to apply importance weights during training of classifiers.
    """
    
    def __init__(self, base_classifier, use_weights: bool = True):
        """
        Initialize weighted classifier wrapper.
        
        Args:
            base_classifier: Base sklearn classifier
            use_weights: Whether to use sample weights (if supported)
        """
        self.base_classifier = base_classifier
        self.use_weights = use_weights
        self.is_fitted_ = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """
        Fit the classifier with optional sample weights.
        
        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        if self.use_weights and sample_weight is not None:
            # Check if classifier supports sample weights
            try:
                self.base_classifier.fit(X, y, sample_weight=sample_weight)
                logger.info("Applied importance weights during training")
            except TypeError:
                logger.warning("Classifier doesn't support sample weights, training without weights")
                self.base_classifier.fit(X, y)
        else:
            self.base_classifier.fit(X, y)
            
        self.is_fitted_ = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted classifier."""
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before prediction")
        return self.base_classifier.predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the fitted classifier."""
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before prediction")
        return self.base_classifier.predict_proba(X)
        
    def __getattr__(self, name):
        """Delegate attribute access to base classifier."""
        return getattr(self.base_classifier, name)


def demo_importance_weighting():
    """
    Demonstrate importance weighting with synthetic data.
    """
    # Generate synthetic data with covariate shift
    np.random.seed(42)
    
    # Source domain: normal distribution
    X_source = np.random.normal(0, 1, (1000, 5))
    y_source = (X_source[:, 0] + X_source[:, 1] > 0).astype(int)
    
    # Target domain: shifted distribution
    X_target = np.random.normal(0.5, 1.2, (500, 5))
    
    print("Importance Weighting Demo")
    print("=" * 40)
    
    # Compute importance weights
    weighter = ImportanceWeighter(classifier='logistic')
    weights = weighter.fit_transform(X_source, X_target)
    
    # Evaluate weight quality
    quality = evaluate_weight_quality(weights, X_source, X_target)
    
    print(f"Weight Statistics:")
    for key, value in quality['weight_stats'].items():
        print(f"  {key}: {value:.3f}")
    
    print(f"\nCovariate Balance:")
    for key, value in quality['covariate_balance'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_importance_weighting()