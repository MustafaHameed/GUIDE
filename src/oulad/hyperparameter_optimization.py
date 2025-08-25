"""
Advanced Hyperparameter Optimization for OULAD Deep Learning Models

This module implements state-of-the-art hyperparameter optimization techniques:
- Bayesian optimization with Optuna
- Multi-objective optimization (accuracy vs fairness)
- Progressive training strategies
- AutoML approaches for neural architecture search
"""

import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import logging
from typing import Dict, Tuple, Optional, List, Callable, Any
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modern models
try:
    from .modern_deep_learning import TabNet, FTTransformer, NODE, SAINT, AutoInt
except ImportError:
    from modern_deep_learning import TabNet, FTTransformer, NODE, SAINT, AutoInt

logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    Advanced hyperparameter tuner using Optuna for deep learning models.
    """
    
    def __init__(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 100,
                 direction: str = 'maximize', timeout: Optional[int] = None,
                 sampler: str = 'tpe', pruner: str = 'median'):
        """
        Initialize hyperparameter tuner.
        
        Args:
            model_type: Type of model to tune ('tabnet', 'ft_transformer', 'node', 'saint', 'autoint')
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
            n_trials: Number of optimization trials
            direction: Optimization direction ('maximize' or 'minimize')
            timeout: Timeout in seconds
            sampler: Sampling strategy ('tpe', 'cmaes', 'random')
            pruner: Pruning strategy ('median', 'successive_halving', 'none')
        """
        self.model_type = model_type.lower()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.direction = direction
        self.timeout = timeout
        self.input_dim = X_train.shape[1]
        
        # Setup scaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_val_scaled = self.scaler.transform(X_val)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup sampler
        if sampler == 'tpe':
            self.sampler = TPESampler(seed=42)
        elif sampler == 'cmaes':
            self.sampler = CmaEsSampler(seed=42)
        else:
            self.sampler = optuna.samplers.RandomSampler(seed=42)
        
        # Setup pruner
        if pruner == 'median':
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == 'successive_halving':
            self.pruner = SuccessiveHalvingPruner()
        else:
            self.pruner = optuna.pruners.NopPruner()
    
    def suggest_tabnet_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for TabNet."""
        return {
            'n_d': trial.suggest_int('n_d', 8, 128, step=8),
            'n_a': trial.suggest_int('n_a', 8, 128, step=8),
            'n_steps': trial.suggest_int('n_steps', 2, 8),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'n_independent': trial.suggest_int('n_independent', 1, 4),
            'n_shared': trial.suggest_int('n_shared', 1, 4),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'epochs': trial.suggest_int('epochs', 50, 200)
        }
    
    def suggest_ft_transformer_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for FT-Transformer."""
        return {
            'd_token': trial.suggest_categorical('d_token', [64, 96, 128, 192, 256]),
            'n_blocks': trial.suggest_int('n_blocks', 1, 8),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.5),
            'ffn_dropout': trial.suggest_float('ffn_dropout', 0.0, 0.5),
            'residual_dropout': trial.suggest_float('residual_dropout', 0.0, 0.3),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
            'd_ffn_factor': trial.suggest_float('d_ffn_factor', 1.0, 4.0),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'epochs': trial.suggest_int('epochs', 50, 150)
        }
    
    def suggest_node_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for NODE."""
        return {
            'num_layers': trial.suggest_int('num_layers', 4, 12),
            'tree_dim': trial.suggest_int('tree_dim', 1, 8),
            'depth': trial.suggest_int('depth', 4, 8),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
            'epochs': trial.suggest_int('epochs', 100, 300)
        }
    
    def suggest_saint_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for SAINT."""
        return {
            'embed_dim': trial.suggest_categorical('embed_dim', [16, 32, 64, 128]),
            'depth': trial.suggest_int('depth', 2, 8),
            'heads': trial.suggest_categorical('heads', [2, 4, 8, 16]),
            'dim_head': trial.suggest_categorical('dim_head', [8, 16, 32, 64]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'use_intersample': trial.suggest_categorical('use_intersample', [True, False]),
            'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'epochs': trial.suggest_int('epochs', 50, 200)
        }
    
    def suggest_autoint_params(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for AutoInt."""
        return {
            'embed_dim': trial.suggest_categorical('embed_dim', [8, 16, 32, 64]),
            'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4, 8]),
            'num_layers': trial.suggest_int('num_layers', 1, 6),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            'epochs': trial.suggest_int('epochs', 50, 150)
        }
    
    def create_model(self, params: Dict) -> nn.Module:
        """Create model with given parameters."""
        if self.model_type == 'tabnet':
            return TabNet(
                input_dim=self.input_dim,
                n_d=params['n_d'],
                n_a=params['n_a'],
                n_steps=params['n_steps'],
                gamma=params['gamma'],
                n_independent=params['n_independent'],
                n_shared=params['n_shared']
            )
        elif self.model_type == 'ft_transformer':
            return FTTransformer(
                input_dim=self.input_dim,
                d_token=params['d_token'],
                n_blocks=params['n_blocks'],
                attention_dropout=params['attention_dropout'],
                ffn_dropout=params['ffn_dropout'],
                residual_dropout=params['residual_dropout'],
                n_heads=params['n_heads'],
                d_ffn_factor=params['d_ffn_factor']
            )
        elif self.model_type == 'node':
            return NODE(
                input_dim=self.input_dim,
                num_layers=params['num_layers'],
                tree_dim=params['tree_dim'],
                depth=params['depth']
            )
        elif self.model_type == 'saint':
            return SAINT(
                input_dim=self.input_dim,
                embed_dim=params['embed_dim'],
                depth=params['depth'],
                heads=params['heads'],
                dim_head=params['dim_head'],
                dropout=params['dropout'],
                use_intersample=params['use_intersample']
            )
        elif self.model_type == 'autoint':
            return AutoInt(
                input_dim=self.input_dim,
                embed_dim=params['embed_dim'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                use_residual=params['use_residual']
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        # Suggest parameters based on model type
        if self.model_type == 'tabnet':
            params = self.suggest_tabnet_params(trial)
        elif self.model_type == 'ft_transformer':
            params = self.suggest_ft_transformer_params(trial)
        elif self.model_type == 'node':
            params = self.suggest_node_params(trial)
        elif self.model_type == 'saint':
            params = self.suggest_saint_params(trial)
        elif self.model_type == 'autoint':
            params = self.suggest_autoint_params(trial)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create and train model
        model = self.create_model(params).to(self.device)
        
        # Setup data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train_scaled),
            torch.LongTensor(self.y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val_scaled),
            torch.LongTensor(self.y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Train model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        
        best_val_auc = 0.0
        patience_counter = 0
        patience = 15
        
        for epoch in range(params['epochs']):
            # Training phase
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validation phase (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                model.eval()
                val_probs = []
                val_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        probs = F.softmax(outputs, dim=1)[:, 1]
                        
                        val_probs.extend(probs.cpu().numpy())
                        val_labels.extend(batch_y.cpu().numpy())
                
                val_auc = roc_auc_score(val_labels, val_probs)
                
                # Report intermediate value for pruning
                trial.report(val_auc, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
        
        return best_val_auc
    
    def optimize(self) -> optuna.Study:
        """Run hyperparameter optimization."""
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=f"{self.model_type}_optimization"
        )
        
        logger.info(f"Starting hyperparameter optimization for {self.model_type}")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"Timeout: {self.timeout} seconds")
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"Optimization completed for {self.model_type}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        return study


class MultiObjectiveTuner:
    """
    Multi-objective hyperparameter optimization considering both accuracy and fairness.
    """
    
    def __init__(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, sensitive_features: np.ndarray,
                 n_trials: int = 100, timeout: Optional[int] = None):
        """
        Initialize multi-objective tuner.
        
        Args:
            model_type: Type of model to tune
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            sensitive_features: Sensitive features for fairness evaluation
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.model_type = model_type.lower()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.sensitive_features = sensitive_features
        self.n_trials = n_trials
        self.timeout = timeout
        self.input_dim = X_train.shape[1]
        
        # Setup scaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_val_scaled = self.scaler.transform(X_val)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def calculate_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 sensitive_features: np.ndarray) -> Dict[str, float]:
        """Calculate fairness metrics."""
        from sklearn.metrics import confusion_matrix
        
        # Demographic parity difference
        favorable_rate_overall = np.mean(y_pred)
        
        fairness_metrics = {}
        
        for group_value in np.unique(sensitive_features):
            group_mask = sensitive_features == group_value
            group_favorable_rate = np.mean(y_pred[group_mask])
            
            # Demographic parity
            dp_diff = abs(group_favorable_rate - favorable_rate_overall)
            fairness_metrics[f'dp_diff_group_{group_value}'] = dp_diff
            
            # Equalized odds
            if len(np.unique(y_true[group_mask])) > 1:
                tn, fp, fn, tp = confusion_matrix(y_true[group_mask], y_pred[group_mask]).ravel()
                
                if (tp + fn) > 0:
                    tpr = tp / (tp + fn)
                    fairness_metrics[f'tpr_group_{group_value}'] = tpr
                
                if (tn + fp) > 0:
                    fpr = fp / (tn + fp)
                    fairness_metrics[f'fpr_group_{group_value}'] = fpr
        
        # Overall fairness score (lower is better)
        dp_diffs = [v for k, v in fairness_metrics.items() if 'dp_diff' in k]
        fairness_score = np.mean(dp_diffs) if dp_diffs else 0.0
        
        return fairness_score
    
    def multi_objective_function(self, trial: optuna.Trial) -> Tuple[float, float]:
        """Multi-objective function returning (accuracy, fairness)."""
        # Use same parameter suggestion as single objective
        tuner = OptunaTuner(
            self.model_type, self.X_train, self.y_train,
            self.X_val, self.y_val, n_trials=1
        )
        
        # Get parameters
        if self.model_type == 'tabnet':
            params = tuner.suggest_tabnet_params(trial)
        elif self.model_type == 'ft_transformer':
            params = tuner.suggest_ft_transformer_params(trial)
        elif self.model_type == 'node':
            params = tuner.suggest_node_params(trial)
        elif self.model_type == 'saint':
            params = tuner.suggest_saint_params(trial)
        elif self.model_type == 'autoint':
            params = tuner.suggest_autoint_params(trial)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train model
        model = tuner.create_model(params).to(self.device)
        
        # Setup data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train_scaled),
            torch.LongTensor(self.y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val_scaled),
            torch.LongTensor(self.y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Train model (simplified for multi-objective)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train for fewer epochs in multi-objective setting
        epochs = min(params['epochs'], 50)
        
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Evaluate model
        model.eval()
        val_preds = []
        val_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                probs = F.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_val, val_preds)
        fairness_score = self.calculate_fairness_metrics(
            self.y_val, np.array(val_preds), self.sensitive_features
        )
        
        return accuracy, -fairness_score  # Negative because we want to minimize unfairness
    
    def optimize(self) -> optuna.Study:
        """Run multi-objective optimization."""
        study = optuna.create_study(
            directions=['maximize', 'maximize'],  # Maximize accuracy and minimize unfairness
            study_name=f"{self.model_type}_multi_objective"
        )
        
        logger.info(f"Starting multi-objective optimization for {self.model_type}")
        
        study.optimize(
            self.multi_objective_function,
            n_trials=self.n_trials,
            timeout=self.timeout
        )
        
        logger.info(f"Multi-objective optimization completed for {self.model_type}")
        logger.info(f"Number of Pareto optimal solutions: {len(study.best_trials)}")
        
        return study


class ProgressiveTrainer:
    """
    Progressive training strategies for deep learning models.
    """
    
    def __init__(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray):
        """
        Initialize progressive trainer.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        self.model_type = model_type.lower()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_dim = X_train.shape[1]
        
        # Setup scaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_val_scaled = self.scaler.transform(X_val)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def progressive_resizing(self, base_params: Dict, growth_factor: float = 1.5,
                           stages: int = 3) -> Dict:
        """
        Train model with progressive resizing strategy.
        
        Args:
            base_params: Base hyperparameters
            growth_factor: Factor by which to grow model size
            stages: Number of progressive stages
            
        Returns:
            Training results
        """
        logger.info(f"Starting progressive resizing training for {self.model_type}")
        
        results = {'stages': []}
        
        for stage in range(stages):
            logger.info(f"Training stage {stage + 1}/{stages}")
            
            # Adjust model size
            stage_params = base_params.copy()
            
            if self.model_type == 'tabnet':
                stage_params['n_d'] = int(stage_params['n_d'] * (growth_factor ** stage))
                stage_params['n_a'] = int(stage_params['n_a'] * (growth_factor ** stage))
            elif self.model_type == 'ft_transformer':
                stage_params['d_token'] = int(stage_params['d_token'] * (growth_factor ** stage))
            elif self.model_type == 'saint':
                stage_params['embed_dim'] = int(stage_params['embed_dim'] * (growth_factor ** stage))
            elif self.model_type == 'autoint':
                stage_params['embed_dim'] = int(stage_params['embed_dim'] * (growth_factor ** stage))
            
            # Train model for this stage
            tuner = OptunaTuner(
                self.model_type, self.X_train, self.y_train,
                self.X_val, self.y_val, n_trials=1
            )
            
            model = tuner.create_model(stage_params).to(self.device)
            
            # Train model
            stage_results = self._train_stage(model, stage_params, stage)
            results['stages'].append(stage_results)
        
        return results
    
    def _train_stage(self, model: nn.Module, params: Dict, stage: int) -> Dict:
        """Train a single stage."""
        # Setup data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train_scaled),
            torch.LongTensor(self.y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val_scaled),
            torch.LongTensor(self.y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=params.get('batch_size', 128), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params.get('batch_size', 128), shuffle=False)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get('lr', 1e-3),
            weight_decay=params.get('weight_decay', 1e-4)
        )
        criterion = nn.CrossEntropyLoss()
        
        # Train for fewer epochs per stage
        epochs = params.get('epochs', 100) // 3
        
        best_val_auc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            if (epoch + 1) % 10 == 0:
                model.eval()
                val_probs = []
                val_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        probs = F.softmax(outputs, dim=1)[:, 1]
                        
                        val_probs.extend(probs.cpu().numpy())
                        val_labels.extend(batch_y.cpu().numpy())
                
                val_auc = roc_auc_score(val_labels, val_probs)
                best_val_auc = max(best_val_auc, val_auc)
        
        return {
            'stage': stage,
            'best_val_auc': best_val_auc,
            'model_params': params
        }


def run_comprehensive_optimization(X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 output_dir: str = 'optimization_results',
                                 n_trials: int = 50) -> Dict:
    """
    Run comprehensive hyperparameter optimization for all models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        output_dir: Directory to save results
        n_trials: Number of trials per model
        
    Returns:
        Dictionary with optimization results
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    models = ['tabnet', 'ft_transformer', 'node', 'saint', 'autoint']
    results = {}
    
    for model_type in models:
        logger.info(f"Optimizing {model_type}...")
        
        # Single-objective optimization
        tuner = OptunaTuner(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
            direction='maximize'
        )
        
        study = tuner.optimize()
        
        # Save results
        results[model_type] = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'study': study
        }
        
        # Save study
        with open(f"{output_dir}/{model_type}_study.pkl", 'wb') as f:
            pickle.dump(study, f)
        
        # Save best parameters
        with open(f"{output_dir}/{model_type}_best_params.json", 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        logger.info(f"Completed optimization for {model_type}")
        logger.info(f"Best AUC: {study.best_value:.4f}")
    
    return results