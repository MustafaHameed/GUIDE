"""
Advanced Data Augmentation for Transfer Learning

This module implements sophisticated data augmentation techniques specifically
designed for transfer learning, including transfer-aware SMOTE, domain adaptation
mixup, and adversarial training approaches.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TransferAwareSMOTE(BaseEstimator, TransformerMixin):
    """
    Transfer-aware SMOTE that considers domain shift when generating synthetic samples.
    """
    
    def __init__(self, 
                 k_neighbors: int = 5,
                 domain_weight: float = 0.3,
                 intra_domain_ratio: float = 0.7,
                 random_state: int = 42):
        """
        Initialize Transfer-aware SMOTE.
        
        Args:
            k_neighbors: Number of neighbors for SMOTE
            domain_weight: Weight for domain-aware interpolation
            intra_domain_ratio: Ratio of intra-domain vs cross-domain synthesis
            random_state: Random seed
        """
        self.k_neighbors = k_neighbors
        self.domain_weight = domain_weight
        self.intra_domain_ratio = intra_domain_ratio
        self.random_state = random_state
        
    def fit_resample(self, X_source, y_source, X_target=None, target_ratios=None):
        """
        Generate synthetic samples considering domain information.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features (optional)
            target_ratios: Desired class ratios for augmentation
        """
        X_source = np.array(X_source)
        y_source = np.array(y_source)
        
        if X_target is not None:
            X_target = np.array(X_target)
            
        # Scale features for distance computation
        self.scaler_ = StandardScaler()
        X_source_scaled = self.scaler_.fit_transform(X_source)
        if X_target is not None:
            X_target_scaled = self.scaler_.transform(X_target)
        
        # Identify minority classes
        unique_classes, class_counts = np.unique(y_source, return_counts=True)
        majority_count = np.max(class_counts)
        
        if target_ratios is None:
            target_ratios = {cls: majority_count for cls in unique_classes}
        
        augmented_X = [X_source]
        augmented_y = [y_source]
        
        for cls in unique_classes:
            cls_mask = y_source == cls
            cls_samples = X_source_scaled[cls_mask]
            current_count = np.sum(cls_mask)
            target_count = target_ratios.get(cls, current_count)
            
            if target_count > current_count:
                n_synthetic = target_count - current_count
                logger.info(f"Generating {n_synthetic} synthetic samples for class {cls}")
                
                # Generate synthetic samples
                synthetic_samples = self._generate_synthetic_samples(
                    cls_samples, X_source_scaled, X_target_scaled if X_target is not None else None,
                    n_synthetic
                )
                
                # Inverse transform to original scale
                synthetic_samples = self.scaler_.inverse_transform(synthetic_samples)
                
                augmented_X.append(synthetic_samples)
                augmented_y.append(np.full(n_synthetic, cls))
        
        # Combine all samples
        final_X = np.vstack(augmented_X)
        final_y = np.hstack(augmented_y)
        
        logger.info(f"Augmented dataset: {len(final_X)} samples (from {len(X_source)})")
        
        return final_X, final_y
    
    def _generate_synthetic_samples(self, cls_samples, X_source, X_target, n_synthetic):
        """Generate synthetic samples using transfer-aware interpolation."""
        if len(cls_samples) < 2:
            # Not enough samples for interpolation
            return np.tile(cls_samples[0], (n_synthetic, 1))
        
        synthetic_samples = []
        np.random.seed(self.random_state)
        
        # Build neighbor finder for class samples
        n_neighbors = min(self.k_neighbors, len(cls_samples) - 1)
        nn_clf = NearestNeighbors(n_neighbors=n_neighbors)
        nn_clf.fit(cls_samples)
        
        for _ in range(n_synthetic):
            # Select random sample from class
            sample_idx = np.random.randint(0, len(cls_samples))
            sample = cls_samples[sample_idx]
            
            # Find neighbors
            distances, indices = nn_clf.kneighbors([sample])
            neighbor_indices = indices[0]
            
            # Decide on interpolation strategy
            if np.random.rand() < self.intra_domain_ratio or X_target is None:
                # Intra-domain interpolation (standard SMOTE)
                neighbor_idx = np.random.choice(neighbor_indices)
                neighbor = cls_samples[neighbor_idx]
                
                # Linear interpolation
                alpha = np.random.rand()
                synthetic_sample = sample + alpha * (neighbor - sample)
            else:
                # Cross-domain interpolation
                if len(X_target) > 0:
                    # Find nearest target samples
                    target_nn = NearestNeighbors(n_neighbors=min(5, len(X_target)))
                    target_nn.fit(X_target)
                    
                    target_distances, target_indices = target_nn.kneighbors([sample])
                    target_neighbor_idx = np.random.choice(target_indices[0])
                    target_neighbor = X_target[target_neighbor_idx]
                    
                    # Domain-aware interpolation
                    source_neighbor_idx = np.random.choice(neighbor_indices)
                    source_neighbor = cls_samples[source_neighbor_idx]
                    
                    alpha = np.random.rand()
                    beta = self.domain_weight
                    
                    synthetic_sample = (
                        sample + 
                        alpha * (source_neighbor - sample) + 
                        beta * (target_neighbor - sample)
                    )
                else:
                    # Fallback to intra-domain
                    neighbor_idx = np.random.choice(neighbor_indices)
                    neighbor = cls_samples[neighbor_idx]
                    alpha = np.random.rand()
                    synthetic_sample = sample + alpha * (neighbor - sample)
            
            synthetic_samples.append(synthetic_sample)
        
        return np.array(synthetic_samples)


class DomainAdaptationMixup(BaseEstimator, TransformerMixin):
    """
    Domain adaptation mixup for transfer learning data augmentation.
    """
    
    def __init__(self, 
                 alpha: float = 0.2,
                 cross_domain_ratio: float = 0.3,
                 random_state: int = 42):
        """
        Initialize Domain Adaptation Mixup.
        
        Args:
            alpha: Beta distribution parameter for mixup ratio
            cross_domain_ratio: Ratio of cross-domain to intra-domain mixups
            random_state: Random seed
        """
        self.alpha = alpha
        self.cross_domain_ratio = cross_domain_ratio
        self.random_state = random_state
        
    def fit_transform(self, X_source, y_source, X_target, y_target=None):
        """
        Apply domain adaptation mixup augmentation.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features
            y_target: Target domain labels (optional)
        """
        X_source = np.array(X_source)
        y_source = np.array(y_source)
        X_target = np.array(X_target)
        
        np.random.seed(self.random_state)
        
        # Combine source and target for mixup
        n_source = len(X_source)
        n_target = len(X_target)
        n_mixups = min(n_source, n_target) // 2  # Conservative number of mixups
        
        mixup_X = []
        mixup_y = []
        mixup_domains = []  # Track domain mixing information
        
        for i in range(n_mixups):
            # Decide on mixup type
            if np.random.rand() < self.cross_domain_ratio and y_target is not None:
                # Cross-domain mixup
                source_idx = np.random.randint(0, n_source)
                target_idx = np.random.randint(0, n_target)
                
                x1, y1 = X_source[source_idx], y_source[source_idx]
                x2, y2 = X_target[target_idx], y_target[target_idx]
                
                # Only mix if same class (for supervised cross-domain)
                if y1 == y2:
                    lam = np.random.beta(self.alpha, self.alpha)
                    mixed_x = lam * x1 + (1 - lam) * x2
                    mixed_y = y1  # Same class
                    
                    mixup_X.append(mixed_x)
                    mixup_y.append(mixed_y)
                    mixup_domains.append('cross')
            else:
                # Intra-domain mixup (source domain)
                idx1 = np.random.randint(0, n_source)
                idx2 = np.random.randint(0, n_source)
                
                if idx1 != idx2:
                    x1, y1 = X_source[idx1], y_source[idx1]
                    x2, y2 = X_source[idx2], y_source[idx2]
                    
                    lam = np.random.beta(self.alpha, self.alpha)
                    mixed_x = lam * x1 + (1 - lam) * x2
                    
                    if y1 == y2:
                        mixed_y = y1
                    else:
                        # For different classes, use probabilistic label
                        mixed_y = y1 if np.random.rand() < lam else y2
                    
                    mixup_X.append(mixed_x)
                    mixup_y.append(mixed_y)
                    mixup_domains.append('intra')
        
        if mixup_X:
            # Combine original and mixed samples
            augmented_X = np.vstack([X_source, np.array(mixup_X)])
            augmented_y = np.hstack([y_source, np.array(mixup_y)])
            
            logger.info(f"Generated {len(mixup_X)} mixup samples "
                       f"({sum(1 for d in mixup_domains if d == 'cross')} cross-domain, "
                       f"{sum(1 for d in mixup_domains if d == 'intra')} intra-domain)")
            
            return augmented_X, augmented_y
        else:
            return X_source, y_source


class AdversarialAugmentation(BaseEstimator, TransformerMixin):
    """
    Adversarial augmentation for robust transfer learning.
    """
    
    def __init__(self, 
                 epsilon: float = 0.1,
                 n_adversarial: int = None,
                 attack_method: str = 'fgsm',
                 random_state: int = 42):
        """
        Initialize Adversarial Augmentation.
        
        Args:
            epsilon: Perturbation magnitude
            n_adversarial: Number of adversarial examples to generate
            attack_method: Attack method ('fgsm', 'pgd', 'random')
            random_state: Random seed
        """
        self.epsilon = epsilon
        self.n_adversarial = n_adversarial
        self.attack_method = attack_method
        self.random_state = random_state
        
    def fit_transform(self, X, y, model=None):
        """
        Generate adversarial examples for data augmentation.
        
        Args:
            X: Input features
            y: Input labels
            model: Trained model for gradient-based attacks (optional)
        """
        X = np.array(X)
        y = np.array(y)
        
        if self.n_adversarial is None:
            self.n_adversarial = len(X) // 2
        
        np.random.seed(self.random_state)
        
        # Select samples for adversarial generation
        n_samples = min(self.n_adversarial, len(X))
        selected_indices = np.random.choice(len(X), size=n_samples, replace=False)
        
        adversarial_X = []
        adversarial_y = []
        
        for idx in selected_indices:
            x_orig = X[idx]
            y_orig = y[idx]
            
            if self.attack_method == 'random':
                # Random perturbation
                perturbation = np.random.normal(0, self.epsilon, size=x_orig.shape)
                x_adv = x_orig + perturbation
            
            elif self.attack_method == 'fgsm' and model is not None:
                # Fast Gradient Sign Method (simplified version)
                x_adv = self._fgsm_attack(x_orig, y_orig, model)
            
            else:
                # Fallback to random perturbation
                perturbation = np.random.normal(0, self.epsilon, size=x_orig.shape)
                x_adv = x_orig + perturbation
            
            adversarial_X.append(x_adv)
            adversarial_y.append(y_orig)
        
        # Combine original and adversarial samples
        augmented_X = np.vstack([X, np.array(adversarial_X)])
        augmented_y = np.hstack([y, np.array(adversarial_y)])
        
        logger.info(f"Generated {len(adversarial_X)} adversarial examples")
        
        return augmented_X, augmented_y
    
    def _fgsm_attack(self, x, y, model):
        """Fast Gradient Sign Method attack (simplified)."""
        # This is a simplified version - full implementation would require
        # access to model gradients
        
        try:
            # Get model prediction
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba([x])[0]
                confidence = np.max(prob)
                
                # Create perturbation based on prediction confidence
                perturbation_scale = self.epsilon * (1 - confidence)
                perturbation = np.random.normal(0, perturbation_scale, size=x.shape)
                
                return x + perturbation
            else:
                # Fallback to random perturbation
                perturbation = np.random.normal(0, self.epsilon, size=x.shape)
                return x + perturbation
                
        except:
            # Fallback to random perturbation
            perturbation = np.random.normal(0, self.epsilon, size=x.shape)
            return x + perturbation


class ProgressiveAugmentation(BaseEstimator, TransformerMixin):
    """
    Progressive data augmentation that adapts augmentation strategy during training.
    """
    
    def __init__(self, 
                 stages: List[Dict] = None,
                 random_state: int = 42):
        """
        Initialize Progressive Augmentation.
        
        Args:
            stages: List of augmentation configurations per stage
            random_state: Random seed
        """
        self.stages = stages or self._default_stages()
        self.random_state = random_state
        self.current_stage = 0
        
    def _default_stages(self):
        """Define default progressive augmentation stages."""
        return [
            {'method': 'smote', 'params': {'k_neighbors': 3}},
            {'method': 'mixup', 'params': {'alpha': 0.2}},
            {'method': 'adversarial', 'params': {'epsilon': 0.05}},
            {'method': 'combined', 'params': {'all_methods': True}}
        ]
    
    def fit_transform_stage(self, X_source, y_source, X_target=None, y_target=None, stage=None):
        """
        Apply augmentation for specific stage.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features
            y_target: Target domain labels
            stage: Stage number (if None, uses current_stage)
        """
        if stage is None:
            stage = self.current_stage
        
        if stage >= len(self.stages):
            logger.warning(f"Stage {stage} exceeds available stages, using last stage")
            stage = len(self.stages) - 1
        
        stage_config = self.stages[stage]
        method = stage_config['method']
        params = stage_config.get('params', {})
        
        logger.info(f"Applying stage {stage} augmentation: {method}")
        
        if method == 'smote':
            augmenter = TransferAwareSMOTE(random_state=self.random_state, **params)
            return augmenter.fit_resample(X_source, y_source, X_target)
        
        elif method == 'mixup':
            if X_target is not None:
                augmenter = DomainAdaptationMixup(random_state=self.random_state, **params)
                return augmenter.fit_transform(X_source, y_source, X_target, y_target)
            else:
                return X_source, y_source
        
        elif method == 'adversarial':
            augmenter = AdversarialAugmentation(random_state=self.random_state, **params)
            return augmenter.fit_transform(X_source, y_source)
        
        elif method == 'combined':
            # Apply multiple augmentation methods
            current_X, current_y = X_source, y_source
            
            # SMOTE
            smote = TransferAwareSMOTE(random_state=self.random_state)
            current_X, current_y = smote.fit_resample(current_X, current_y, X_target)
            
            # Mixup (if target available)
            if X_target is not None and y_target is not None:
                mixup = DomainAdaptationMixup(random_state=self.random_state)
                current_X, current_y = mixup.fit_transform(current_X, current_y, X_target, y_target)
            
            # Adversarial
            adversarial = AdversarialAugmentation(
                epsilon=0.05, 
                n_adversarial=len(current_X)//4,
                random_state=self.random_state
            )
            current_X, current_y = adversarial.fit_transform(current_X, current_y)
            
            return current_X, current_y
        
        else:
            logger.warning(f"Unknown augmentation method: {method}")
            return X_source, y_source
    
    def advance_stage(self):
        """Advance to next augmentation stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            logger.info(f"Advanced to stage {self.current_stage}")
        return self.current_stage


class AdaptiveAugmentation(BaseEstimator, TransformerMixin):
    """
    Adaptive augmentation that adjusts strategy based on model performance.
    """
    
    def __init__(self, 
                 performance_threshold: float = 0.7,
                 adaptation_factor: float = 0.1,
                 random_state: int = 42):
        """
        Initialize Adaptive Augmentation.
        
        Args:
            performance_threshold: Performance threshold for adaptation
            adaptation_factor: Factor for adjusting augmentation intensity
            random_state: Random seed
        """
        self.performance_threshold = performance_threshold
        self.adaptation_factor = adaptation_factor
        self.random_state = random_state
        self.augmentation_intensity = 1.0
        
    def adapt_strategy(self, model_performance: float):
        """
        Adapt augmentation strategy based on model performance.
        
        Args:
            model_performance: Current model performance metric (0-1)
        """
        if model_performance < self.performance_threshold:
            # Increase augmentation intensity
            self.augmentation_intensity = min(2.0, self.augmentation_intensity + self.adaptation_factor)
            logger.info(f"Increased augmentation intensity to {self.augmentation_intensity:.2f}")
        else:
            # Decrease augmentation intensity
            self.augmentation_intensity = max(0.5, self.augmentation_intensity - self.adaptation_factor)
            logger.info(f"Decreased augmentation intensity to {self.augmentation_intensity:.2f}")
    
    def fit_transform(self, X_source, y_source, X_target=None, y_target=None, model_performance=None):
        """
        Apply adaptive augmentation based on current performance.
        """
        if model_performance is not None:
            self.adapt_strategy(model_performance)
        
        # Adjust augmentation parameters based on intensity
        smote_neighbors = max(1, int(5 * self.augmentation_intensity))
        mixup_alpha = 0.2 * self.augmentation_intensity
        adversarial_epsilon = 0.1 * self.augmentation_intensity
        
        current_X, current_y = X_source, y_source
        
        # Apply SMOTE with adaptive parameters
        if self.augmentation_intensity > 0.5:
            smote = TransferAwareSMOTE(
                k_neighbors=smote_neighbors,
                random_state=self.random_state
            )
            current_X, current_y = smote.fit_resample(current_X, current_y, X_target)
        
        # Apply mixup if target available and intensity high enough
        if X_target is not None and y_target is not None and self.augmentation_intensity > 0.7:
            mixup = DomainAdaptationMixup(
                alpha=mixup_alpha,
                random_state=self.random_state
            )
            current_X, current_y = mixup.fit_transform(current_X, current_y, X_target, y_target)
        
        # Apply adversarial augmentation if intensity is high
        if self.augmentation_intensity > 0.8:
            adversarial = AdversarialAugmentation(
                epsilon=adversarial_epsilon,
                n_adversarial=int(len(current_X) * 0.2),
                random_state=self.random_state
            )
            current_X, current_y = adversarial.fit_transform(current_X, current_y)
        
        logger.info(f"Adaptive augmentation: {len(X_source)} → {len(current_X)} samples "
                   f"(intensity: {self.augmentation_intensity:.2f})")
        
        return current_X, current_y


def comprehensive_augmentation_evaluation(X_source, y_source, X_target, y_target=None):
    """
    Comprehensive evaluation of data augmentation techniques.
    """
    results = {}
    
    augmentation_methods = {
        'transfer_smote': TransferAwareSMOTE(random_state=42),
        'domain_mixup': DomainAdaptationMixup(random_state=42),
        'adversarial': AdversarialAugmentation(random_state=42),
        'progressive': ProgressiveAugmentation(random_state=42),
        'adaptive': AdaptiveAugmentation(random_state=42)
    }
    
    for method_name, augmenter in augmentation_methods.items():
        logger.info(f"Evaluating {method_name} augmentation...")
        
        try:
            if method_name == 'transfer_smote':
                aug_X, aug_y = augmenter.fit_resample(X_source, y_source, X_target)
                
            elif method_name == 'domain_mixup':
                if X_target is not None:
                    aug_X, aug_y = augmenter.fit_transform(X_source, y_source, X_target, y_target)
                else:
                    aug_X, aug_y = X_source, y_source
                    
            elif method_name == 'adversarial':
                aug_X, aug_y = augmenter.fit_transform(X_source, y_source)
                
            elif method_name == 'progressive':
                aug_X, aug_y = augmenter.fit_transform_stage(X_source, y_source, X_target, y_target)
                
            elif method_name == 'adaptive':
                aug_X, aug_y = augmenter.fit_transform(X_source, y_source, X_target, y_target)
            
            results[method_name] = {
                'original_size': len(X_source),
                'augmented_size': len(aug_X),
                'augmentation_ratio': len(aug_X) / len(X_source),
                'class_distribution': dict(zip(*np.unique(aug_y, return_counts=True)))
            }
            
            logger.info(f"{method_name} - Size: {len(X_source)} → {len(aug_X)} "
                       f"(ratio: {results[method_name]['augmentation_ratio']:.2f})")
            
        except Exception as e:
            logger.error(f"Error evaluating {method_name}: {e}")
            results[method_name] = {
                'original_size': len(X_source),
                'augmented_size': len(X_source),
                'augmentation_ratio': 1.0,
                'error': str(e)
            }
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data with class imbalance
    np.random.seed(42)
    n_source, n_target = 800, 200
    n_features = 15
    
    # Create imbalanced source data
    X_source_1 = np.random.randn(600, n_features)  # Majority class
    y_source_1 = np.ones(600)
    X_source_0 = np.random.randn(200, n_features) + 1  # Minority class
    y_source_0 = np.zeros(200)
    
    X_source = np.vstack([X_source_1, X_source_0])
    y_source = np.hstack([y_source_1, y_source_0])
    
    # Create target data with domain shift
    X_target = np.random.randn(n_target, n_features) + 0.5
    y_target = (X_target.sum(axis=1) > 0).astype(int)
    
    # Evaluate augmentation methods
    results = comprehensive_augmentation_evaluation(X_source, y_source, X_target, y_target)
    
    print("\nData Augmentation Results:")
    for method, metrics in results.items():
        if 'error' not in metrics:
            print(f"{method:15} - Original: {metrics['original_size']:3d}, "
                  f"Augmented: {metrics['augmented_size']:4d}, "
                  f"Ratio: {metrics['augmentation_ratio']:.2f}")
        else:
            print(f"{method:15} - Error: {metrics['error']}")