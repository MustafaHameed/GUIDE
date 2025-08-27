"""
Advanced Ensemble and Multi-Model Transfer Learning

This module implements sophisticated ensemble techniques specifically designed
for transfer learning, including mixture of experts, neural architecture search,
and Bayesian model averaging.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import optuna
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MixtureOfExpertsTransfer(BaseEstimator, ClassifierMixin):
    """
    Mixture of Experts for transfer learning with domain-specific gating.
    """
    
    def __init__(self, 
                 experts: Optional[List] = None,
                 gating_features: Optional[List[str]] = None,
                 random_state: int = 42):
        """
        Initialize Mixture of Experts for transfer learning.
        
        Args:
            experts: List of expert classifiers (if None, uses default set)
            gating_features: Features to use for gating network
            random_state: Random seed
        """
        self.experts = experts
        self.gating_features = gating_features
        self.random_state = random_state
        
        if self.experts is None:
            self.experts = [
                RandomForestClassifier(n_estimators=100, random_state=random_state),
                GradientBoostingClassifier(random_state=random_state),
                LogisticRegression(random_state=random_state, max_iter=1000),
                MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state),
                SVC(probability=True, random_state=random_state)
            ]
        
        self.n_experts = len(self.experts)
        
    def fit(self, X, y, domain_labels=None):
        """
        Train mixture of experts with domain-aware gating.
        
        Args:
            X: Training features
            y: Training labels  
            domain_labels: Domain indicators (0=source, 1=target)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Train individual experts
        self.fitted_experts_ = []
        self.expert_weights_ = np.zeros(self.n_experts)
        
        logger.info(f"Training {self.n_experts} experts...")
        
        for i, expert in enumerate(self.experts):
            try:
                fitted_expert = clone(expert)
                fitted_expert.fit(X, y)
                self.fitted_experts_.append(fitted_expert)
                
                # Evaluate expert performance for weighting
                if len(X) > 100:  # Only do CV if we have enough data
                    cv_scores = cross_val_score(fitted_expert, X, y, cv=3, scoring='accuracy')
                    self.expert_weights_[i] = cv_scores.mean()
                else:
                    # Simple train accuracy for small datasets
                    train_pred = fitted_expert.predict(X)
                    self.expert_weights_[i] = accuracy_score(y, train_pred)
                    
                logger.info(f"Expert {i} ({type(expert).__name__}): {self.expert_weights_[i]:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to train expert {i}: {e}")
                # Use dummy expert
                self.fitted_experts_.append(None)
                self.expert_weights_[i] = 0.0
        
        # Normalize weights
        if self.expert_weights_.sum() > 0:
            self.expert_weights_ = self.expert_weights_ / self.expert_weights_.sum()
        else:
            self.expert_weights_ = np.ones(self.n_experts) / self.n_experts
            
        # Train gating network
        self._train_gating_network(X, y, domain_labels)
        
        return self
    
    def _train_gating_network(self, X, y, domain_labels):
        """Train gating network to select experts based on input."""
        # Simple gating network based on domain and feature statistics
        self.gating_network_ = LogisticRegression(
            multi_class='multinomial',
            random_state=self.random_state,
            max_iter=1000
        )
        
        # Create gating features
        gating_X = self._create_gating_features(X, domain_labels)
        
        # Create pseudo-labels for gating (which expert would be best)
        gating_y = self._create_expert_labels(X, y)
        
        try:
            self.gating_network_.fit(gating_X, gating_y)
        except:
            # Fallback to uniform gating
            logger.warning("Gating network training failed, using uniform weights")
            self.gating_network_ = None
    
    def _create_gating_features(self, X, domain_labels):
        """Create features for gating network."""
        gating_features = []
        
        # Basic statistics
        gating_features.append(np.mean(X, axis=1).reshape(-1, 1))
        gating_features.append(np.std(X, axis=1).reshape(-1, 1))
        gating_features.append(np.min(X, axis=1).reshape(-1, 1))
        gating_features.append(np.max(X, axis=1).reshape(-1, 1))
        
        # Domain information
        if domain_labels is not None:
            gating_features.append(domain_labels.reshape(-1, 1))
        else:
            gating_features.append(np.zeros((len(X), 1)))
        
        return np.hstack(gating_features)
    
    def _create_expert_labels(self, X, y):
        """Create labels indicating which expert would be best for each sample."""
        expert_labels = np.zeros(len(X), dtype=int)
        
        # For simplicity, assign based on feature statistics
        # In practice, this could be based on individual expert performance
        feature_means = np.mean(X, axis=1)
        percentiles = np.percentile(feature_means, [25, 50, 75])
        
        for i in range(len(X)):
            if feature_means[i] < percentiles[0]:
                expert_labels[i] = 0
            elif feature_means[i] < percentiles[1]:
                expert_labels[i] = 1
            elif feature_means[i] < percentiles[2]:
                expert_labels[i] = 2
            else:
                expert_labels[i] = min(3, self.n_experts - 1)
        
        return expert_labels
    
    def predict(self, X, domain_labels=None):
        """Make predictions using mixture of experts."""
        X = np.array(X)
        
        # Get gating weights
        if self.gating_network_ is not None:
            gating_X = self._create_gating_features(X, domain_labels)
            try:
                gating_probs = self.gating_network_.predict_proba(gating_X)
            except:
                gating_probs = np.tile(self.expert_weights_, (len(X), 1))
        else:
            gating_probs = np.tile(self.expert_weights_, (len(X), 1))
        
        # Get predictions from all experts
        expert_predictions = []
        for expert in self.fitted_experts_:
            if expert is not None:
                try:
                    pred = expert.predict(X)
                    expert_predictions.append(pred)
                except:
                    expert_predictions.append(np.zeros(len(X)))
            else:
                expert_predictions.append(np.zeros(len(X)))
        
        expert_predictions = np.array(expert_predictions).T  # Shape: (n_samples, n_experts)
        
        # Weighted voting
        final_predictions = np.zeros(len(X))
        for i in range(len(X)):
            weighted_votes = {}
            for j, pred in enumerate(expert_predictions[i]):
                weight = gating_probs[i, j] if j < gating_probs.shape[1] else 0
                if pred in weighted_votes:
                    weighted_votes[pred] += weight
                else:
                    weighted_votes[pred] = weight
            
            final_predictions[i] = max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        return final_predictions.astype(int)
    
    def predict_proba(self, X, domain_labels=None):
        """Predict class probabilities using mixture of experts."""
        X = np.array(X)
        
        # Get gating weights
        if self.gating_network_ is not None:
            gating_X = self._create_gating_features(X, domain_labels)
            try:
                gating_probs = self.gating_network_.predict_proba(gating_X)
            except:
                gating_probs = np.tile(self.expert_weights_, (len(X), 1))
        else:
            gating_probs = np.tile(self.expert_weights_, (len(X), 1))
        
        # Get probabilities from all experts
        expert_probabilities = []
        for expert in self.fitted_experts_:
            if expert is not None and hasattr(expert, 'predict_proba'):
                try:
                    proba = expert.predict_proba(X)
                    if proba.shape[1] == 2:  # Binary classification
                        expert_probabilities.append(proba)
                    else:
                        # Handle multi-class case
                        expert_probabilities.append(proba[:, :2])
                except:
                    expert_probabilities.append(np.column_stack([
                        np.full(len(X), 0.5), np.full(len(X), 0.5)
                    ]))
            else:
                expert_probabilities.append(np.column_stack([
                    np.full(len(X), 0.5), np.full(len(X), 0.5)
                ]))
        
        # Weighted average of probabilities
        final_probabilities = np.zeros((len(X), 2))
        for i in range(len(X)):
            for j, expert_proba in enumerate(expert_probabilities):
                weight = gating_probs[i, j] if j < gating_probs.shape[1] else 0
                final_probabilities[i] += weight * expert_proba[i]
        
        return final_probabilities


class NeuralArchitectureSearchTransfer:
    """
    Neural Architecture Search for optimal transfer learning models.
    """
    
    def __init__(self, 
                 search_space: Dict = None,
                 n_trials: int = 50,
                 random_state: int = 42):
        self.search_space = search_space or self._default_search_space()
        self.n_trials = n_trials
        self.random_state = random_state
        
    def _default_search_space(self):
        """Define default search space for neural architectures."""
        return {
            'n_layers': (1, 5),
            'layer_sizes': (32, 512),
            'dropout_rate': (0.0, 0.5),
            'activation': ['relu', 'tanh', 'sigmoid'],
            'learning_rate': (1e-4, 1e-1),
            'batch_size': [32, 64, 128, 256],
            'optimizer': ['adam', 'sgd', 'rmsprop']
        }
    
    def search(self, X_train, y_train, X_val, y_val):
        """
        Search for optimal neural architecture for transfer learning.
        """
        def objective(trial):
            # Sample hyperparameters
            n_layers = trial.suggest_int('n_layers', *self.search_space['n_layers'])
            layer_sizes = []
            for i in range(n_layers):
                size = trial.suggest_int(f'layer_size_{i}', *self.search_space['layer_sizes'])
                layer_sizes.append(size)
            
            dropout_rate = trial.suggest_float('dropout_rate', *self.search_space['dropout_rate'])
            activation = trial.suggest_categorical('activation', self.search_space['activation'])
            learning_rate = trial.suggest_float('learning_rate', *self.search_space['learning_rate'], log=True)
            
            # Create and train model
            try:
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(layer_sizes),
                    activation=activation,
                    learning_rate_init=learning_rate,
                    max_iter=200,
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.1
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                
                return accuracy
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        # Get best parameters
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        logger.info(f"Best NAS score: {self.best_score_:.3f}")
        logger.info(f"Best parameters: {self.best_params_}")
        
        return self.best_params_
    
    def create_best_model(self):
        """Create model with best found architecture."""
        if not hasattr(self, 'best_params_'):
            raise ValueError("Must run search() first")
        
        # Extract layer sizes
        n_layers = self.best_params_['n_layers']
        layer_sizes = []
        for i in range(n_layers):
            layer_sizes.append(self.best_params_[f'layer_size_{i}'])
        
        model = MLPClassifier(
            hidden_layer_sizes=tuple(layer_sizes),
            activation=self.best_params_['activation'],
            learning_rate_init=self.best_params_['learning_rate'],
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        return model


class BayesianModelAveraging(BaseEstimator, ClassifierMixin):
    """
    Bayesian Model Averaging for transfer learning with uncertainty quantification.
    """
    
    def __init__(self, 
                 models: Optional[List] = None,
                 n_bootstrap: int = 100,
                 confidence_threshold: float = 0.8,
                 random_state: int = 42):
        self.models = models
        self.n_bootstrap = n_bootstrap
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        
        if self.models is None:
            self.models = [
                RandomForestClassifier(n_estimators=50, random_state=random_state),
                GradientBoostingClassifier(n_estimators=50, random_state=random_state),
                LogisticRegression(random_state=random_state, max_iter=1000),
                MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=random_state)
            ]
    
    def fit(self, X, y):
        """Train ensemble with bootstrap sampling for uncertainty estimation."""
        X = np.array(X)
        y = np.array(y)
        
        self.bootstrap_models_ = []
        self.model_weights_ = []
        
        logger.info(f"Training Bayesian ensemble with {self.n_bootstrap} bootstrap samples...")
        
        np.random.seed(self.random_state)
        
        for i in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Select random model
            model_idx = np.random.randint(len(self.models))
            model = clone(self.models[model_idx])
            
            try:
                model.fit(X_boot, y_boot)
                
                # Evaluate model weight based on out-of-bag performance
                oob_indices = np.setdiff1d(np.arange(len(X)), indices)
                if len(oob_indices) > 0:
                    oob_pred = model.predict(X[oob_indices])
                    weight = accuracy_score(y[oob_indices], oob_pred)
                else:
                    weight = 1.0
                
                self.bootstrap_models_.append(model)
                self.model_weights_.append(weight)
                
            except Exception as e:
                logger.warning(f"Bootstrap model {i} failed: {e}")
        
        # Normalize weights
        if len(self.model_weights_) > 0:
            self.model_weights_ = np.array(self.model_weights_)
            self.model_weights_ = self.model_weights_ / self.model_weights_.sum()
        
        logger.info(f"Successfully trained {len(self.bootstrap_models_)} bootstrap models")
        
        return self
    
    def predict(self, X):
        """Make predictions with uncertainty estimation."""
        predictions, uncertainties = self.predict_with_uncertainty(X)
        return predictions
    
    def predict_proba(self, X):
        """Predict probabilities with uncertainty estimation."""
        X = np.array(X)
        
        all_probas = []
        for model, weight in zip(self.bootstrap_models_, self.model_weights_):
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if proba.shape[1] >= 2:
                        all_probas.append(weight * proba[:, :2])
                    else:
                        # Handle binary case
                        all_probas.append(weight * np.column_stack([1-proba[:, 0], proba[:, 0]]))
            except:
                continue
        
        if not all_probas:
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])
        
        # Weighted average
        avg_proba = np.sum(all_probas, axis=0)
        
        # Normalize
        row_sums = avg_proba.sum(axis=1, keepdims=True)
        avg_proba = avg_proba / np.maximum(row_sums, 1e-10)
        
        return avg_proba
    
    def predict_with_uncertainty(self, X):
        """Make predictions with uncertainty estimates."""
        X = np.array(X)
        
        all_predictions = []
        for model, weight in zip(self.bootstrap_models_, self.model_weights_):
            try:
                pred = model.predict(X)
                all_predictions.append(pred)
            except:
                continue
        
        if not all_predictions:
            return np.zeros(len(X)), np.ones(len(X))
        
        all_predictions = np.array(all_predictions)
        
        # Mode for final prediction
        final_predictions = []
        uncertainties = []
        
        for i in range(len(X)):
            sample_preds = all_predictions[:, i]
            unique_vals, counts = np.unique(sample_preds, return_counts=True)
            mode_idx = np.argmax(counts)
            final_pred = unique_vals[mode_idx]
            
            # Uncertainty as 1 - (confidence in mode)
            uncertainty = 1.0 - (counts[mode_idx] / len(sample_preds))
            
            final_predictions.append(final_pred)
            uncertainties.append(uncertainty)
        
        return np.array(final_predictions), np.array(uncertainties)


class AdvancedEnsembleTransfer(BaseEstimator, ClassifierMixin):
    """
    Advanced ensemble transfer learning combining multiple sophisticated techniques.
    """
    
    def __init__(self, 
                 use_mixture_experts: bool = True,
                 use_nas: bool = False,  # Disabled by default due to computational cost
                 use_bayesian: bool = True,
                 random_state: int = 42):
        self.use_mixture_experts = use_mixture_experts
        self.use_nas = use_nas
        self.use_bayesian = use_bayesian
        self.random_state = random_state
        
    def fit(self, X_source, y_source, X_target=None, y_target=None):
        """Train advanced ensemble for transfer learning."""
        X_source = np.array(X_source)
        y_source = np.array(y_source)
        
        self.ensemble_models_ = {}
        
        # Mixture of Experts
        if self.use_mixture_experts:
            logger.info("Training Mixture of Experts...")
            domain_labels = np.zeros(len(X_source))  # Source domain
            if X_target is not None:
                X_combined = np.vstack([X_source, X_target])
                y_combined = np.hstack([y_source, y_target if y_target is not None else np.full(len(X_target), -1)])
                domain_combined = np.hstack([np.zeros(len(X_source)), np.ones(len(X_target))])
                
                # Filter out unlabeled target samples for training
                labeled_mask = y_combined != -1
                X_labeled = X_combined[labeled_mask]
                y_labeled = y_combined[labeled_mask]
                domain_labeled = domain_combined[labeled_mask]
                
                self.mixture_experts_ = MixtureOfExpertsTransfer(random_state=self.random_state)
                self.mixture_experts_.fit(X_labeled, y_labeled, domain_labeled)
            else:
                self.mixture_experts_ = MixtureOfExpertsTransfer(random_state=self.random_state)
                self.mixture_experts_.fit(X_source, y_source, domain_labels)
            
            self.ensemble_models_['mixture_experts'] = self.mixture_experts_
        
        # Neural Architecture Search
        if self.use_nas and X_target is not None and y_target is not None:
            logger.info("Running Neural Architecture Search...")
            nas = NeuralArchitectureSearchTransfer(n_trials=20, random_state=self.random_state)
            
            # Use part of target data for validation
            split_idx = len(X_target) // 2
            X_train = np.vstack([X_source, X_target[:split_idx]])
            y_train = np.hstack([y_source, y_target[:split_idx]])
            X_val, y_val = X_target[split_idx:], y_target[split_idx:]
            
            if len(X_val) > 0:
                nas.search(X_train, y_train, X_val, y_val)
                self.nas_model_ = nas.create_best_model()
                self.nas_model_.fit(X_train, y_train)
                self.ensemble_models_['nas'] = self.nas_model_
        
        # Bayesian Model Averaging
        if self.use_bayesian:
            logger.info("Training Bayesian Model Averaging...")
            self.bayesian_ensemble_ = BayesianModelAveraging(
                n_bootstrap=50,  # Reduced for performance
                random_state=self.random_state
            )
            self.bayesian_ensemble_.fit(X_source, y_source)
            self.ensemble_models_['bayesian'] = self.bayesian_ensemble_
        
        # Meta-learner to combine ensemble predictions
        if len(self.ensemble_models_) > 1:
            self._train_meta_learner(X_source, y_source)
        
        return self
    
    def _train_meta_learner(self, X, y):
        """Train meta-learner to combine ensemble predictions."""
        # Generate meta-features from base models
        meta_features = []
        
        for name, model in self.ensemble_models_.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    meta_features.append(proba)
                else:
                    pred = model.predict(X).reshape(-1, 1)
                    meta_features.append(pred)
            except:
                logger.warning(f"Failed to get predictions from {name}")
        
        if meta_features:
            meta_X = np.hstack(meta_features)
            self.meta_learner_ = LogisticRegression(random_state=self.random_state, max_iter=1000)
            self.meta_learner_.fit(meta_X, y)
        else:
            self.meta_learner_ = None
    
    def predict(self, X, domain_labels=None):
        """Make predictions using advanced ensemble."""
        if len(self.ensemble_models_) == 1:
            # Single model
            model = list(self.ensemble_models_.values())[0]
            if hasattr(model, 'predict') and 'mixture_experts' in self.ensemble_models_:
                return model.predict(X, domain_labels)
            else:
                return model.predict(X)
        
        # Multiple models - use meta-learner if available
        if hasattr(self, 'meta_learner_') and self.meta_learner_ is not None:
            meta_features = []
            for name, model in self.ensemble_models_.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        meta_features.append(proba)
                    else:
                        if name == 'mixture_experts':
                            pred = model.predict(X, domain_labels).reshape(-1, 1)
                        else:
                            pred = model.predict(X).reshape(-1, 1)
                        meta_features.append(pred)
                except:
                    continue
            
            if meta_features:
                meta_X = np.hstack(meta_features)
                return self.meta_learner_.predict(meta_X)
        
        # Fallback to simple voting
        predictions = []
        for name, model in self.ensemble_models_.items():
            try:
                if name == 'mixture_experts':
                    pred = model.predict(X, domain_labels)
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            except:
                continue
        
        if predictions:
            predictions = np.array(predictions)
            # Majority voting
            final_pred = []
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                unique_vals, counts = np.unique(votes, return_counts=True)
                final_pred.append(unique_vals[np.argmax(counts)])
            return np.array(final_pred)
        else:
            return np.zeros(len(X))
    
    def predict_proba(self, X, domain_labels=None):
        """Predict probabilities using advanced ensemble."""
        if len(self.ensemble_models_) == 1:
            # Single model
            model = list(self.ensemble_models_.values())[0]
            if hasattr(model, 'predict_proba'):
                if 'mixture_experts' in self.ensemble_models_:
                    return model.predict_proba(X, domain_labels)
                else:
                    return model.predict_proba(X)
            else:
                pred = model.predict(X)
                return np.column_stack([1-pred, pred])
        
        # Average probabilities from all models
        all_probas = []
        for name, model in self.ensemble_models_.items():
            try:
                if hasattr(model, 'predict_proba'):
                    if name == 'mixture_experts':
                        proba = model.predict_proba(X, domain_labels)
                    else:
                        proba = model.predict_proba(X)
                    all_probas.append(proba)
            except:
                continue
        
        if all_probas:
            return np.mean(all_probas, axis=0)
        else:
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def comprehensive_ensemble_evaluation(X_source, y_source, X_target, y_target):
    """
    Comprehensive evaluation of advanced ensemble techniques.
    """
    results = {}
    
    # Standard ensemble methods
    methods = {
        'mixture_experts': {'use_mixture_experts': True, 'use_nas': False, 'use_bayesian': False},
        'bayesian_average': {'use_mixture_experts': False, 'use_nas': False, 'use_bayesian': True},
        'combined_ensemble': {'use_mixture_experts': True, 'use_nas': False, 'use_bayesian': True}
    }
    
    for method_name, config in methods.items():
        logger.info(f"Evaluating {method_name}...")
        
        try:
            model = AdvancedEnsembleTransfer(**config, random_state=42)
            model.fit(X_source, y_source, X_target, y_target)
            
            # Evaluate on target
            y_pred = model.predict(X_target)
            y_prob = model.predict_proba(X_target)[:, 1]
            
            results[method_name] = {
                'accuracy': accuracy_score(y_target, y_pred),
                'f1': f1_score(y_target, y_pred),
                'auc': roc_auc_score(y_target, y_prob) if len(np.unique(y_target)) > 1 else 0.5
            }
            
            logger.info(f"{method_name} - Accuracy: {results[method_name]['accuracy']:.3f}, "
                       f"F1: {results[method_name]['f1']:.3f}, AUC: {results[method_name]['auc']:.3f}")
        
        except Exception as e:
            logger.error(f"Error evaluating {method_name}: {e}")
            results[method_name] = {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5}
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_source, n_target = 1000, 200
    n_features = 20
    
    X_source = np.random.randn(n_source, n_features)
    y_source = (X_source.sum(axis=1) + np.random.randn(n_source) * 0.1 > 0).astype(int)
    
    # Add domain shift to target
    X_target = np.random.randn(n_target, n_features) + 0.5
    y_target = (X_target.sum(axis=1) + np.random.randn(n_target) * 0.1 > 0).astype(int)
    
    # Evaluate methods
    results = comprehensive_ensemble_evaluation(X_source, y_source, X_target, y_target)
    
    print("\nAdvanced Ensemble Transfer Learning Results:")
    for method, metrics in results.items():
        print(f"{method:20} - Accuracy: {metrics['accuracy']:.3f}, "
              f"F1: {metrics['f1']:.3f}, AUC: {metrics['auc']:.3f}")