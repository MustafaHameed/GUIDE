"""
Theoretical Transfer Learning Improvements

This module implements theoretical advances in transfer learning including
H-divergence minimization, optimal transport via Wasserstein distance,
information-theoretic methods, and causal transfer learning approaches.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class HDivergenceMinimizer(BaseEstimator, TransformerMixin):
    """
    H-divergence minimization for domain adaptation based on theory.
    
    Implements the theoretical framework from Ben-David et al. for
    minimizing the H-divergence between source and target domains.
    """
    
    def __init__(self, 
                 classifier_class=LogisticRegression,
                 n_iterations: int = 100,
                 learning_rate: float = 0.01,
                 lambda_reg: float = 0.1,
                 random_state: int = 42):
        """
        Initialize H-divergence minimizer.
        
        Args:
            classifier_class: Base classifier for H-divergence computation
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            lambda_reg: Regularization parameter
            random_state: Random seed
        """
        self.classifier_class = classifier_class
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        
    def fit(self, X_source, X_target):
        """
        Fit H-divergence minimizer to align domains.
        
        Args:
            X_source: Source domain features
            X_target: Target domain features
        """
        X_source = np.array(X_source)
        X_target = np.array(X_target)
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_source_scaled = self.scaler_.fit_transform(X_source)
        X_target_scaled = self.scaler_.transform(X_target)
        
        # Initialize transformation matrix
        n_features = X_source_scaled.shape[1]
        self.W_ = np.eye(n_features)
        
        # Compute initial H-divergence
        h_div = self._compute_h_divergence(X_source_scaled, X_target_scaled)
        logger.info(f"Initial H-divergence: {h_div:.4f}")
        
        # Optimize transformation to minimize H-divergence
        best_W = self.W_.copy()
        best_h_div = h_div
        
        for iteration in range(self.n_iterations):
            # Compute gradient (approximated via finite differences)
            grad_W = self._compute_gradient(X_source_scaled, X_target_scaled)
            
            # Update transformation matrix
            self.W_ = self.W_ - self.learning_rate * grad_W
            
            # Add regularization (encourage orthogonality)
            if self.lambda_reg > 0:
                # Orthogonality regularization
                orth_reg = self.lambda_reg * (self.W_ @ self.W_.T - np.eye(n_features))
                self.W_ = self.W_ - self.learning_rate * orth_reg
            
            # Evaluate current H-divergence
            if iteration % 20 == 0:
                current_h_div = self._compute_h_divergence(
                    X_source_scaled @ self.W_, X_target_scaled @ self.W_
                )
                
                if current_h_div < best_h_div:
                    best_h_div = current_h_div
                    best_W = self.W_.copy()
                
                logger.info(f"Iteration {iteration}: H-divergence = {current_h_div:.4f}")
        
        self.W_ = best_W
        final_h_div = self._compute_h_divergence(
            X_source_scaled @ self.W_, X_target_scaled @ self.W_
        )
        logger.info(f"Final H-divergence: {final_h_div:.4f} (improvement: {h_div - final_h_div:.4f})")
        
        return self
    
    def transform(self, X):
        """Transform features to minimize H-divergence."""
        X_scaled = self.scaler_.transform(X)
        return X_scaled @ self.W_
    
    def _compute_h_divergence(self, X_source, X_target):
        """Compute H-divergence between domains."""
        # Create domain labels
        domain_labels = np.concatenate([
            np.zeros(len(X_source)),  # Source = 0
            np.ones(len(X_target))    # Target = 1
        ])
        
        # Combine data
        X_combined = np.vstack([X_source, X_target])
        
        # Train domain classifier
        try:
            domain_classifier = self.classifier_class(random_state=self.random_state, max_iter=1000)
            domain_classifier.fit(X_combined, domain_labels)
            
            # Get predictions
            domain_predictions = domain_classifier.predict(X_combined)
            
            # H-divergence is 2 * (1 - domain_classification_error)
            domain_accuracy = accuracy_score(domain_labels, domain_predictions)
            h_divergence = 2 * (domain_accuracy - 0.5)  # Centered at 0.5 for random guessing
            
            return max(0, h_divergence)  # Ensure non-negative
        
        except:
            # Fallback to simple distance measure
            mean_source = np.mean(X_source, axis=0)
            mean_target = np.mean(X_target, axis=0)
            return np.linalg.norm(mean_source - mean_target)
    
    def _compute_gradient(self, X_source, X_target):
        """Compute gradient of H-divergence with respect to transformation matrix."""
        # Finite difference approximation
        epsilon = 1e-6
        n_features = self.W_.shape[0]
        grad_W = np.zeros_like(self.W_)
        
        # Current H-divergence
        current_h_div = self._compute_h_divergence(
            X_source @ self.W_, X_target @ self.W_
        )
        
        # Compute partial derivatives
        for i in range(n_features):
            for j in range(n_features):
                # Perturb W
                W_perturbed = self.W_.copy()
                W_perturbed[i, j] += epsilon
                
                # Compute perturbed H-divergence
                perturbed_h_div = self._compute_h_divergence(
                    X_source @ W_perturbed, X_target @ W_perturbed
                )
                
                # Finite difference
                grad_W[i, j] = (perturbed_h_div - current_h_div) / epsilon
        
        return grad_W


class WassersteinDomainAlignment(BaseEstimator, TransformerMixin):
    """
    Optimal transport using Wasserstein distance for domain alignment.
    """
    
    def __init__(self, 
                 reg_param: float = 0.1,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6):
        """
        Initialize Wasserstein domain alignment.
        
        Args:
            reg_param: Regularization parameter for Sinkhorn algorithm
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance
        """
        self.reg_param = reg_param
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def fit(self, X_source, X_target):
        """
        Fit optimal transport mapping between domains.
        
        Args:
            X_source: Source domain features
            X_target: Target domain features
        """
        X_source = np.array(X_source)
        X_target = np.array(X_target)
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_source_scaled = self.scaler_.fit_transform(X_source)
        X_target_scaled = self.scaler_.transform(X_target)
        
        # Compute optimal transport plan
        self.transport_plan_ = self._compute_optimal_transport(
            X_source_scaled, X_target_scaled
        )
        
        # Store source data for transformation
        self.X_source_fitted_ = X_source_scaled
        self.X_target_fitted_ = X_target_scaled
        
        return self
    
    def transform(self, X):
        """Transform source domain data using optimal transport."""
        X_scaled = self.scaler_.transform(X)
        
        # For each point in X, find the optimal transport-based transformation
        transformed_X = np.zeros_like(X_scaled)
        
        for i, x in enumerate(X_scaled):
            # Find nearest source point
            distances = cdist([x], self.X_source_fitted_)[0]
            nearest_idx = np.argmin(distances)
            
            # Apply transport plan
            transport_weights = self.transport_plan_[nearest_idx, :]
            
            # Weighted combination of target points
            transformed_x = np.average(self.X_target_fitted_, axis=0, weights=transport_weights)
            transformed_X[i] = transformed_x
        
        return transformed_X
    
    def _compute_optimal_transport(self, X_source, X_target):
        """Compute optimal transport plan using Sinkhorn algorithm."""
        n_source = len(X_source)
        n_target = len(X_target)
        
        # Compute cost matrix (Euclidean distances)
        cost_matrix = cdist(X_source, X_target, metric='euclidean')
        
        # Normalize cost matrix
        cost_matrix = cost_matrix / np.max(cost_matrix)
        
        # Uniform distributions
        a = np.ones(n_source) / n_source
        b = np.ones(n_target) / n_target
        
        # Sinkhorn algorithm
        K = np.exp(-cost_matrix / self.reg_param)
        u = np.ones(n_source) / n_source
        
        for iteration in range(self.max_iterations):
            u_prev = u.copy()
            v = b / (K.T @ u)
            u = a / (K @ v)
            
            # Check convergence
            if np.sum(np.abs(u - u_prev)) < self.tolerance:
                logger.info(f"Sinkhorn converged after {iteration} iterations")
                break
        
        # Compute transport plan
        transport_plan = np.diag(u) @ K @ np.diag(v)
        
        # Normalize rows to ensure they sum to 1
        row_sums = transport_plan.sum(axis=1, keepdims=True)
        transport_plan = transport_plan / np.maximum(row_sums, 1e-10)
        
        logger.info(f"Computed optimal transport plan: {transport_plan.shape}")
        
        return transport_plan


class InformationTheoreticTransfer(BaseEstimator, TransformerMixin):
    """
    Information-theoretic transfer learning using mutual information maximization.
    """
    
    def __init__(self, 
                 n_bins: int = 50,
                 mi_method: str = 'histogram',
                 alpha: float = 0.5):
        """
        Initialize information-theoretic transfer.
        
        Args:
            n_bins: Number of bins for histogram-based MI estimation
            mi_method: Method for MI estimation ('histogram', 'knn')
            alpha: Weight for balancing MI objectives
        """
        self.n_bins = n_bins
        self.mi_method = mi_method
        self.alpha = alpha
        
    def fit(self, X_source, y_source, X_target):
        """
        Fit information-theoretic transformation.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features
        """
        X_source = np.array(X_source)
        y_source = np.array(y_source)
        X_target = np.array(X_target)
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_source_scaled = self.scaler_.fit_transform(X_source)
        X_target_scaled = self.scaler_.transform(X_target)
        
        # Use PCA to find information-preserving projections
        self.pca_source_ = PCA(random_state=42)
        self.pca_target_ = PCA(random_state=42)
        
        # Fit PCAs
        source_components = self.pca_source_.fit_transform(X_source_scaled)
        target_components = self.pca_target_.fit_transform(X_target_scaled)
        
        # Find alignment between PCA spaces that maximizes MI
        n_components = min(source_components.shape[1], target_components.shape[1], 10)
        
        # Compute mutual information between aligned components
        mi_scores = []
        best_alignment = None
        best_mi = -np.inf
        
        # Try different alignments
        for rotation_angle in np.linspace(0, 2*np.pi, 20):
            # Create rotation matrix for alignment
            rotation_matrix = self._create_rotation_matrix(n_components, rotation_angle)
            
            # Apply rotation to target components
            rotated_target = target_components[:, :n_components] @ rotation_matrix
            
            # Compute MI between source and rotated target
            mi_total = 0
            for i in range(n_components):
                source_comp = source_components[:len(rotated_target), i]
                target_comp = rotated_target[:, i]
                
                mi = self._compute_mutual_information(source_comp, target_comp)
                mi_total += mi
            
            if mi_total > best_mi:
                best_mi = mi_total
                best_alignment = rotation_matrix
        
        self.alignment_matrix_ = best_alignment
        self.n_components_ = n_components
        
        logger.info(f"Best mutual information alignment: {best_mi:.4f}")
        
        return self
    
    def transform(self, X):
        """Transform features using information-theoretic alignment."""
        X_scaled = self.scaler_.transform(X)
        
        # Project to PCA space
        components = self.pca_target_.transform(X_scaled)
        
        # Apply alignment
        if self.alignment_matrix_ is not None:
            aligned_components = components[:, :self.n_components_] @ self.alignment_matrix_
            
            # Combine with remaining components
            if components.shape[1] > self.n_components_:
                remaining_components = components[:, self.n_components_:]
                final_components = np.hstack([aligned_components, remaining_components])
            else:
                final_components = aligned_components
        else:
            final_components = components
        
        # Transform back to original space
        return self.pca_target_.inverse_transform(final_components)
    
    def _create_rotation_matrix(self, n_dim, angle):
        """Create rotation matrix for component alignment."""
        if n_dim == 1:
            return np.array([[1.0]])
        elif n_dim == 2:
            return np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
        else:
            # For higher dimensions, create rotation in first 2 dimensions
            R = np.eye(n_dim)
            R[0, 0] = np.cos(angle)
            R[0, 1] = -np.sin(angle)
            R[1, 0] = np.sin(angle)
            R[1, 1] = np.cos(angle)
            return R
    
    def _compute_mutual_information(self, x, y):
        """Compute mutual information between two variables."""
        # Discretize continuous variables
        x_bins = np.linspace(np.min(x), np.max(x), self.n_bins)
        y_bins = np.linspace(np.min(y), np.max(y), self.n_bins)
        
        x_discrete = np.digitize(x, x_bins)
        y_discrete = np.digitize(y, y_bins)
        
        # Compute joint and marginal histograms
        joint_hist, _, _ = np.histogram2d(x_discrete, y_discrete, bins=self.n_bins)
        
        # Add small constant to avoid log(0)
        joint_hist = joint_hist + 1e-10
        
        # Normalize to get probabilities
        joint_prob = joint_hist / np.sum(joint_hist)
        
        # Marginal probabilities
        marginal_x = np.sum(joint_prob, axis=1)
        marginal_y = np.sum(joint_prob, axis=0)
        
        # Compute mutual information
        mi = 0
        for i in range(self.n_bins):
            for j in range(self.n_bins):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (marginal_x[i] * marginal_y[j])
                    )
        
        return mi


class CausalTransferLearning(BaseEstimator, ClassifierMixin):
    """
    Causal transfer learning that leverages causal relationships.
    """
    
    def __init__(self, 
                 causal_method: str = 'correlation',
                 n_causal_features: int = 5,
                 causal_threshold: float = 0.3):
        """
        Initialize causal transfer learning.
        
        Args:
            causal_method: Method for causal discovery ('correlation', 'mi')
            n_causal_features: Number of causal features to identify
            causal_threshold: Threshold for causal relationship strength
        """
        self.causal_method = causal_method
        self.n_causal_features = n_causal_features
        self.causal_threshold = causal_threshold
        
    def fit(self, X_source, y_source, X_target=None):
        """
        Fit causal transfer learning model.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features (optional)
        """
        X_source = np.array(X_source)
        y_source = np.array(y_source)
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_source_scaled = self.scaler_.fit_transform(X_source)
        
        # Identify causal features
        self.causal_features_ = self._identify_causal_features(X_source_scaled, y_source)
        
        # Train model on causal features
        self.causal_model_ = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_features='sqrt'
        )
        
        X_causal = X_source_scaled[:, self.causal_features_]
        self.causal_model_.fit(X_causal, y_source)
        
        # Train full model for comparison
        self.full_model_ = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.full_model_.fit(X_source_scaled, y_source)
        
        logger.info(f"Identified {len(self.causal_features_)} causal features: {self.causal_features_}")
        
        return self
    
    def predict(self, X):
        """Make predictions using causal model."""
        X_scaled = self.scaler_.transform(X)
        X_causal = X_scaled[:, self.causal_features_]
        return self.causal_model_.predict(X_causal)
    
    def predict_proba(self, X):
        """Predict probabilities using causal model."""
        X_scaled = self.scaler_.transform(X)
        X_causal = X_scaled[:, self.causal_features_]
        return self.causal_model_.predict_proba(X_causal)
    
    def _identify_causal_features(self, X, y):
        """Identify features with causal relationships to target."""
        n_features = X.shape[1]
        causal_scores = np.zeros(n_features)
        
        for i in range(n_features):
            if self.causal_method == 'correlation':
                # Simple correlation-based causality proxy
                causal_scores[i] = abs(np.corrcoef(X[:, i], y)[0, 1])
            
            elif self.causal_method == 'mi':
                # Mutual information-based causality
                causal_scores[i] = self._compute_mutual_information_simple(X[:, i], y)
        
        # Select top causal features
        causal_indices = np.argsort(causal_scores)[::-1]
        
        # Filter by threshold
        strong_causal = causal_indices[causal_scores[causal_indices] > self.causal_threshold]
        
        # Take top N features
        selected_features = strong_causal[:self.n_causal_features]
        
        # Ensure we have at least some features
        if len(selected_features) == 0:
            selected_features = causal_indices[:self.n_causal_features]
        
        return selected_features
    
    def _compute_mutual_information_simple(self, x, y):
        """Simple MI computation for feature selection."""
        # Discretize
        x_discrete = np.digitize(x, np.percentile(x, np.linspace(0, 100, 10)))
        y_discrete = y.astype(int)
        
        # Compute MI
        from sklearn.metrics import mutual_info_score
        return mutual_info_score(x_discrete, y_discrete)


class TheoreticalTransferEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble combining multiple theoretical transfer learning approaches.
    """
    
    def __init__(self, 
                 use_h_divergence: bool = True,
                 use_wasserstein: bool = True,
                 use_information_theory: bool = True,
                 use_causal: bool = True,
                 ensemble_method: str = 'voting'):
        """
        Initialize theoretical transfer ensemble.
        
        Args:
            use_h_divergence: Whether to use H-divergence minimization
            use_wasserstein: Whether to use Wasserstein domain alignment
            use_information_theory: Whether to use information-theoretic methods
            use_causal: Whether to use causal transfer learning
            ensemble_method: Method for combining predictions ('voting', 'weighted')
        """
        self.use_h_divergence = use_h_divergence
        self.use_wasserstein = use_wasserstein
        self.use_information_theory = use_information_theory
        self.use_causal = use_causal
        self.ensemble_method = ensemble_method
        
    def fit(self, X_source, y_source, X_target=None):
        """Fit theoretical transfer ensemble."""
        self.models_ = {}
        self.transformers_ = {}
        
        # H-divergence approach
        if self.use_h_divergence and X_target is not None:
            logger.info("Training H-divergence minimization...")
            h_div_transformer = HDivergenceMinimizer()
            h_div_transformer.fit(X_source, X_target)
            
            X_source_h = h_div_transformer.transform(X_source)
            h_div_model = RandomForestClassifier(n_estimators=100, random_state=42)
            h_div_model.fit(X_source_h, y_source)
            
            self.transformers_['h_divergence'] = h_div_transformer
            self.models_['h_divergence'] = h_div_model
        
        # Wasserstein approach
        if self.use_wasserstein and X_target is not None:
            logger.info("Training Wasserstein domain alignment...")
            wasserstein_transformer = WassersteinDomainAlignment()
            wasserstein_transformer.fit(X_source, X_target)
            
            X_source_w = wasserstein_transformer.transform(X_source)
            wasserstein_model = RandomForestClassifier(n_estimators=100, random_state=42)
            wasserstein_model.fit(X_source_w, y_source)
            
            self.transformers_['wasserstein'] = wasserstein_transformer
            self.models_['wasserstein'] = wasserstein_model
        
        # Information-theoretic approach
        if self.use_information_theory and X_target is not None:
            logger.info("Training information-theoretic transfer...")
            it_transformer = InformationTheoreticTransfer()
            it_transformer.fit(X_source, y_source, X_target)
            
            X_source_it = it_transformer.transform(X_source)
            it_model = RandomForestClassifier(n_estimators=100, random_state=42)
            it_model.fit(X_source_it, y_source)
            
            self.transformers_['information_theory'] = it_transformer
            self.models_['information_theory'] = it_model
        
        # Causal approach
        if self.use_causal:
            logger.info("Training causal transfer learning...")
            causal_model = CausalTransferLearning()
            causal_model.fit(X_source, y_source, X_target)
            
            self.models_['causal'] = causal_model
        
        logger.info(f"Trained theoretical ensemble with {len(self.models_)} methods")
        
        return self
    
    def predict(self, X):
        """Make predictions using theoretical ensemble."""
        predictions = []
        
        for method_name, model in self.models_.items():
            if method_name in self.transformers_:
                # Apply transformation first
                X_transformed = self.transformers_[method_name].transform(X)
                pred = model.predict(X_transformed)
            else:
                # Direct prediction
                pred = model.predict(X)
            
            predictions.append(pred)
        
        if not predictions:
            return np.zeros(len(X))
        
        # Ensemble predictions
        predictions = np.array(predictions)
        
        if self.ensemble_method == 'voting':
            # Majority voting
            final_pred = []
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                unique_vals, counts = np.unique(votes, return_counts=True)
                final_pred.append(unique_vals[np.argmax(counts)])
            return np.array(final_pred)
        
        else:  # weighted
            # Simple average (could be improved with learned weights)
            return np.round(np.mean(predictions, axis=0)).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities using theoretical ensemble."""
        probabilities = []
        
        for method_name, model in self.models_.items():
            try:
                if method_name in self.transformers_:
                    X_transformed = self.transformers_[method_name].transform(X)
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_transformed)
                    else:
                        pred = model.predict(X_transformed)
                        proba = np.column_stack([1-pred, pred])
                else:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                    else:
                        pred = model.predict(X)
                        proba = np.column_stack([1-pred, pred])
                
                probabilities.append(proba)
            except:
                continue
        
        if probabilities:
            return np.mean(probabilities, axis=0)
        else:
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def evaluate_theoretical_methods(X_source, y_source, X_target, y_target):
    """
    Comprehensive evaluation of theoretical transfer learning methods.
    """
    results = {}
    
    methods = {
        'h_divergence_only': {
            'use_h_divergence': True, 'use_wasserstein': False,
            'use_information_theory': False, 'use_causal': False
        },
        'wasserstein_only': {
            'use_h_divergence': False, 'use_wasserstein': True,
            'use_information_theory': False, 'use_causal': False
        },
        'information_theory_only': {
            'use_h_divergence': False, 'use_wasserstein': False,
            'use_information_theory': True, 'use_causal': False
        },
        'causal_only': {
            'use_h_divergence': False, 'use_wasserstein': False,
            'use_information_theory': False, 'use_causal': True
        },
        'theoretical_ensemble': {
            'use_h_divergence': True, 'use_wasserstein': True,
            'use_information_theory': True, 'use_causal': True
        }
    }
    
    for method_name, config in methods.items():
        logger.info(f"Evaluating {method_name}...")
        
        try:
            model = TheoreticalTransferEnsemble(**config)
            model.fit(X_source, y_source, X_target)
            
            # Evaluate on target
            y_pred = model.predict(X_target)
            y_prob = model.predict_proba(X_target)[:, 1]
            
            results[method_name] = {
                'accuracy': accuracy_score(y_target, y_pred),
                'f1': f1_score(y_target, y_pred)
            }
            
            logger.info(f"{method_name} - Accuracy: {results[method_name]['accuracy']:.3f}, "
                       f"F1: {results[method_name]['f1']:.3f}")
        
        except Exception as e:
            logger.error(f"Error evaluating {method_name}: {e}")
            results[method_name] = {'accuracy': 0.0, 'f1': 0.0}
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_source, n_target = 800, 200
    n_features = 15
    
    X_source = np.random.randn(n_source, n_features)
    y_source = (X_source.sum(axis=1) + np.random.randn(n_source) * 0.1 > 0).astype(int)
    
    # Add domain shift to target
    X_target = np.random.randn(n_target, n_features) + 0.5
    y_target = (X_target.sum(axis=1) + np.random.randn(n_target) * 0.1 > 0).astype(int)
    
    # Evaluate methods
    results = evaluate_theoretical_methods(X_source, y_source, X_target, y_target)
    
    print("\nTheoretical Transfer Learning Results:")
    for method, metrics in results.items():
        print(f"{method:25} - Accuracy: {metrics['accuracy']:.3f}, "
              f"F1: {metrics['f1']:.3f}")