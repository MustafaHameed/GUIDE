"""
CORAL (CORrelation ALignment) for Domain Adaptation

Implements CORAL feature alignment to reduce domain shift by aligning
the covariance matrices of source and target domains.

CORAL transforms source features by:
1. Whitening source features (zero mean, identity covariance)
2. "Recoloring" with target covariance structure

Reference: "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm, inv

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class CORALTransformer(BaseEstimator, TransformerMixin):
    """
    CORAL transformation for domain adaptation.
    
    Aligns the second-order statistics (covariances) of source and target domains
    by whitening source features and recoloring with target covariance.
    """
    
    def __init__(self, lambda_coral: float = 1.0, center_data: bool = True,
                 regularization: float = 1e-6):
        """
        Initialize CORAL transformer.
        
        Args:
            lambda_coral: Strength of CORAL alignment (0 = no alignment, 1 = full alignment)
            center_data: Whether to center data before alignment
            regularization: Regularization parameter for matrix inversion
        """
        self.lambda_coral = lambda_coral
        self.center_data = center_data
        self.regularization = regularization
        self.is_fitted_ = False
        
        # Fitted parameters
        self.source_mean_ = None
        self.target_mean_ = None
        self.source_cov_ = None
        self.target_cov_ = None
        self.transform_matrix_ = None
        
    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> 'CORALTransformer':
        """
        Fit the CORAL transformation.
        
        Args:
            X_source: Source domain features (n_samples_source, n_features)
            X_target: Target domain features (n_samples_target, n_features)
            
        Returns:
            self
        """
        logger.info("Fitting CORAL transformation...")
        
        # Ensure inputs are numpy arrays
        X_source = np.asarray(X_source)
        X_target = np.asarray(X_target)
        
        if X_source.shape[1] != X_target.shape[1]:
            raise ValueError("Source and target must have same number of features")
        
        n_features = X_source.shape[1]
        
        # Center the data if requested
        if self.center_data:
            self.source_mean_ = np.mean(X_source, axis=0)
            self.target_mean_ = np.mean(X_target, axis=0)
            
            X_source_centered = X_source - self.source_mean_
            X_target_centered = X_target - self.target_mean_
        else:
            self.source_mean_ = np.zeros(n_features)
            self.target_mean_ = np.zeros(n_features)
            
            X_source_centered = X_source
            X_target_centered = X_target
        
        # Compute covariance matrices
        self.source_cov_ = np.cov(X_source_centered, rowvar=False)
        self.target_cov_ = np.cov(X_target_centered, rowvar=False)
        
        # Add regularization to ensure numerical stability
        regularization_matrix = self.regularization * np.eye(n_features)
        self.source_cov_ += regularization_matrix
        self.target_cov_ += regularization_matrix
        
        # Compute CORAL transformation matrix
        try:
            # A = Cs^(-1/2) * Ct^(1/2)
            # where Cs is source covariance, Ct is target covariance
            
            # Compute matrix square roots using eigendecomposition for stability
            source_cov_inv_sqrt = self._matrix_power(self.source_cov_, -0.5)
            target_cov_sqrt = self._matrix_power(self.target_cov_, 0.5)
            
            # Full CORAL transformation
            coral_transform = target_cov_sqrt @ source_cov_inv_sqrt
            
            # Apply lambda weighting (interpolate between identity and full CORAL)
            identity = np.eye(n_features)
            self.transform_matrix_ = (
                (1 - self.lambda_coral) * identity + 
                self.lambda_coral * coral_transform
            )
            
            logger.info(f"CORAL transformation fitted with λ={self.lambda_coral}")
            
        except Exception as e:
            logger.warning(f"CORAL transformation failed, using identity: {e}")
            self.transform_matrix_ = np.eye(n_features)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: np.ndarray, domain: str = 'source') -> np.ndarray:
        """
        Apply CORAL transformation to features.
        
        Args:
            X: Input features to transform
            domain: Which domain the features belong to ('source' or 'target')
            
        Returns:
            Transformed features
        """
        if not self.is_fitted_:
            raise ValueError("CORALTransformer must be fitted before transform")
        
        X = np.asarray(X)
        
        # Center the data
        if domain == 'source':
            X_centered = X - self.source_mean_
        else:
            X_centered = X - self.target_mean_
        
        # Apply CORAL transformation (only to source domain)
        if domain == 'source':
            X_transformed = X_centered @ self.transform_matrix_.T
        else:
            # Target domain features are only centered
            X_transformed = X_centered
        
        return X_transformed
    
    def fit_transform(self, X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit CORAL transformation and transform both domains.
        
        Args:
            X_source: Source domain features
            X_target: Target domain features
            
        Returns:
            Tuple of (transformed_source, transformed_target)
        """
        self.fit(X_source, X_target)
        
        X_source_transformed = self.transform(X_source, domain='source')
        X_target_transformed = self.transform(X_target, domain='target')
        
        return X_source_transformed, X_target_transformed
    
    def _matrix_power(self, matrix: np.ndarray, power: float) -> np.ndarray:
        """
        Compute matrix power using eigendecomposition for numerical stability.
        
        Args:
            matrix: Input matrix (must be symmetric positive definite)
            power: Power to raise matrix to
            
        Returns:
            matrix^power
        """
        try:
            # Eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            
            # Clip negative eigenvalues to small positive values
            eigenvals = np.maximum(eigenvals, 1e-12)
            
            # Compute matrix power
            eigenvals_powered = np.power(eigenvals, power)
            
            # Reconstruct matrix
            result = eigenvecs @ np.diag(eigenvals_powered) @ eigenvecs.T
            
            return result
            
        except Exception as e:
            logger.warning(f"Matrix power computation failed: {e}")
            return np.eye(matrix.shape[0])
    
    def get_alignment_metrics(self) -> dict:
        """
        Compute metrics to evaluate CORAL alignment quality.
        
        Returns:
            Dictionary with alignment metrics
        """
        if not self.is_fitted_:
            raise ValueError("CORALTransformer must be fitted before computing metrics")
        
        metrics = {}
        
        # Frobenius norm of covariance difference (before alignment)
        cov_diff_before = self.source_cov_ - self.target_cov_
        frobenius_before = np.linalg.norm(cov_diff_before, 'fro')
        
        # Covariance after CORAL transformation
        source_cov_after = (
            self.transform_matrix_ @ self.source_cov_ @ self.transform_matrix_.T
        )
        cov_diff_after = source_cov_after - self.target_cov_
        frobenius_after = np.linalg.norm(cov_diff_after, 'fro')
        
        metrics['frobenius_norm_before'] = frobenius_before
        metrics['frobenius_norm_after'] = frobenius_after
        metrics['alignment_improvement'] = frobenius_before - frobenius_after
        metrics['relative_improvement'] = (
            (frobenius_before - frobenius_after) / frobenius_before
            if frobenius_before > 0 else 0
        )
        
        # Condition numbers
        metrics['source_cov_condition'] = np.linalg.cond(self.source_cov_)
        metrics['target_cov_condition'] = np.linalg.cond(self.target_cov_)
        
        return metrics


def coral_loss(source_features: np.ndarray, target_features: np.ndarray) -> float:
    """
    Compute CORAL loss (covariance alignment loss).
    
    Args:
        source_features: Source domain features
        target_features: Target domain features
        
    Returns:
        CORAL loss (Frobenius norm of covariance difference)
    """
    # Compute covariances
    source_cov = np.cov(source_features, rowvar=False)
    target_cov = np.cov(target_features, rowvar=False)
    
    # Compute Frobenius norm of difference
    cov_diff = source_cov - target_cov
    coral_loss_value = np.linalg.norm(cov_diff, 'fro')
    
    return coral_loss_value


def apply_coral_alignment(X_source: np.ndarray, X_target: np.ndarray,
                         lambda_coral: float = 1.0, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to apply CORAL alignment.
    
    Args:
        X_source: Source domain features
        X_target: Target domain features
        lambda_coral: Strength of alignment
        **kwargs: Additional arguments for CORALTransformer
        
    Returns:
        Tuple of (aligned_source, aligned_target)
    """
    coral = CORALTransformer(lambda_coral=lambda_coral, **kwargs)
    return coral.fit_transform(X_source, X_target)


class DeepCORAL:
    """
    Deep CORAL implementation for neural networks (placeholder for future extension).
    
    This would be integrated with PyTorch models to apply CORAL loss
    during deep learning training.
    """
    
    def __init__(self, lambda_coral: float = 1.0):
        self.lambda_coral = lambda_coral
    
    def coral_loss_torch(self, source_features, target_features):
        """
        Compute CORAL loss for PyTorch tensors.
        
        Args:
            source_features: Source domain feature tensor
            target_features: Target domain feature tensor
            
        Returns:
            CORAL loss tensor
        """
        # This would require PyTorch implementation
        # Placeholder for future integration with deep learning models
        raise NotImplementedError("Deep CORAL requires PyTorch integration")


def demo_coral():
    """
    Demonstrate CORAL transformation with synthetic data.
    """
    # Generate synthetic data with different covariance structures
    np.random.seed(42)
    
    # Source domain: identity covariance
    X_source = np.random.multivariate_normal([0, 0, 0], np.eye(3), 1000)
    
    # Target domain: different covariance structure
    target_cov = np.array([[2, 0.5, 0.1],
                          [0.5, 1.5, 0.3], 
                          [0.1, 0.3, 0.8]])
    X_target = np.random.multivariate_normal([1, -0.5, 0.2], target_cov, 500)
    
    print("CORAL Alignment Demo")
    print("=" * 30)
    
    # Apply CORAL transformation
    coral = CORALTransformer(lambda_coral=1.0)
    X_source_aligned, X_target_aligned = coral.fit_transform(X_source, X_target)
    
    # Compute alignment metrics
    metrics = coral.get_alignment_metrics()
    
    print("Alignment Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Compare covariances before and after
    print(f"\nSource covariance before alignment:")
    print(np.cov(X_source, rowvar=False))
    
    print(f"\nSource covariance after alignment:")
    print(np.cov(X_source_aligned, rowvar=False))
    
    print(f"\nTarget covariance:")
    print(np.cov(X_target, rowvar=False))
    
    # CORAL loss before and after
    loss_before = coral_loss(X_source, X_target)
    loss_after = coral_loss(X_source_aligned, X_target_aligned)
    
    print(f"\nCORAL loss before: {loss_before:.4f}")
    print(f"CORAL loss after: {loss_after:.4f}")
    print(f"Improvement: {loss_before - loss_after:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_coral()