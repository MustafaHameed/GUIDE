"""
Optimized Deep Learning Models for OULAD Dataset

This module focuses on addressing the specific challenges of the OULAD dataset:
- Class imbalance handling
- Proper cross-validation
- Optimized architectures for tabular data
- Better threshold selection strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OptimizedTabularNet(nn.Module):
    """
    Optimized neural network for tabular data with focus on OULAD characteristics.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], 
                 dropout: float = 0.25, use_batch_norm: bool = True):
        super(OptimizedTabularNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Input layer with batch norm
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 2)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        return self.output_layer(x)


class CrossValidationTrainer:
    """
    Cross-validation trainer for robust model evaluation.
    """
    
    def __init__(self, model_class, model_params: Dict, n_folds: int = 5, random_state: int = 42):
        self.model_class = model_class
        self.model_params = model_params
        self.n_folds = n_folds
        self.random_state = random_state
        
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray, 
                          training_params: Dict) -> Dict:
        """
        Perform cross-validation training and evaluation.
        """
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        fold_results = {
            'train_acc': [],
            'val_acc': [],
            'val_auc': [],
            'val_f1': [],
            'val_balanced_acc': []
        }
        
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{self.n_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Create model
            model = self.model_class(**self.model_params)
            
            # Train model
            history = self._train_single_model(
                model, X_train_scaled, y_train_fold, 
                X_val_scaled, y_val_fold, **training_params
            )
            
            # Evaluate
            model.eval()
            device = next(model.parameters()).device
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            
            with torch.no_grad():
                outputs = model(X_val_tensor)
                val_probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                val_preds = torch.max(outputs, 1)[1].cpu().numpy()
            
            # Store results
            train_acc = max(history['train_acc'])
            val_acc = accuracy_score(y_val_fold, val_preds)
            val_auc = roc_auc_score(y_val_fold, val_probs)
            val_f1 = f1_score(y_val_fold, val_preds)
            val_balanced_acc = balanced_accuracy_score(y_val_fold, val_preds)
            
            fold_results['train_acc'].append(train_acc)
            fold_results['val_acc'].append(val_acc)
            fold_results['val_auc'].append(val_auc)
            fold_results['val_f1'].append(val_f1)
            fold_results['val_balanced_acc'].append(val_balanced_acc)
            
            models.append({
                'model': model.cpu(),
                'scaler': scaler,
                'history': history
            })
            
            logger.info(f"Fold {fold + 1} - Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
        
        # Aggregate results
        cv_results = {}
        for metric in fold_results:
            values = fold_results[metric]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        return cv_results, models
    
    def _train_single_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100,
                           batch_size: int = 64, learning_rate: float = 0.001,
                           weight_decay: float = 1e-4, patience: int = 15,
                           device: str = 'auto') -> Dict:
        """
        Train a single model with early stopping.
        """
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = model.to(device)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        # Handle class imbalance with weighted sampling
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss with class weights
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }
        
        best_val_score = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                val_probs = F.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                val_preds = torch.max(val_outputs, 1)[1].cpu().numpy()
                
                val_acc = accuracy_score(y_val, val_preds)
                val_auc = roc_auc_score(y_val, val_probs)
            
            # Record history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_correct / train_total)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_auc'].append(val_auc)
            
            # Early stopping based on balanced score
            val_score = 0.7 * val_auc + 0.3 * val_acc
            
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                best_model_state = {name: param.clone() for name, param in model.named_parameters()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Restore best weights
        if best_model_state is not None:
            for name, param in model.named_parameters():
                param.data.copy_(best_model_state[name])
        
        return history


def train_optimized_models(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          random_state: int = 42) -> Dict:
    """
    Train optimized deep learning models with cross-validation.
    """
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    logger.info("Training optimized deep learning models with cross-validation...")
    
    input_dim = X_train.shape[1]
    results = {}
    all_models = {}
    
    # Model configurations optimized for OULAD
    model_configs = {
        'optimized_mlp_small': {
            'model_params': {
                'input_dim': input_dim,
                'hidden_dims': [64, 32],
                'dropout': 0.2,
                'use_batch_norm': True
            },
            'training_params': {
                'epochs': 150,
                'batch_size': 128,
                'learning_rate': 0.002,
                'weight_decay': 1e-4,
                'patience': 20
            }
        },
        'optimized_mlp_medium': {
            'model_params': {
                'input_dim': input_dim,
                'hidden_dims': [128, 64, 32],
                'dropout': 0.25,
                'use_batch_norm': True
            },
            'training_params': {
                'epochs': 150,
                'batch_size': 64,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'patience': 25
            }
        },
        'optimized_mlp_large': {
            'model_params': {
                'input_dim': input_dim,
                'hidden_dims': [256, 128, 64, 32],
                'dropout': 0.3,
                'use_batch_norm': True
            },
            'training_params': {
                'epochs': 200,
                'batch_size': 64,
                'learning_rate': 0.0008,
                'weight_decay': 2e-4,
                'patience': 30
            }
        }
    }
    
    for model_name, config in model_configs.items():
        logger.info(f"Training {model_name} with cross-validation...")
        
        # Create trainer
        trainer = CrossValidationTrainer(
            model_class=OptimizedTabularNet,
            model_params=config['model_params'],
            n_folds=5,
            random_state=random_state
        )
        
        # Train with cross-validation
        cv_results, fold_models = trainer.train_and_evaluate(
            X_train, y_train, config['training_params']
        )
        
        # Train final model on full training set
        logger.info(f"Training final {model_name} on full training set...")
        
        # Use best parameters from CV
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        final_model = OptimizedTabularNet(**config['model_params'])
        
        # Use 20% of training data for validation
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=random_state
        )
        
        history = trainer._train_single_model(
            final_model, X_train_final, y_train_final,
            X_val_final, y_val_final, **config['training_params']
        )
        
        # Evaluate on test set
        final_model.eval()
        device = next(final_model.parameters()).device
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        with torch.no_grad():
            test_outputs = final_model(X_test_tensor)
            test_probs = F.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
            test_preds = torch.max(test_outputs, 1)[1].cpu().numpy()
        
        test_acc = accuracy_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_probs)
        test_f1 = f1_score(y_test, test_preds)
        test_balanced_acc = balanced_accuracy_score(y_test, test_preds)
        
        # Store results
        all_models[model_name] = {
            'model': final_model.cpu(),
            'scaler': scaler,
            'cv_models': fold_models,
            'history': history,
            'config': config
        }
        
        results[model_name] = {
            'accuracy': test_acc,
            'roc_auc': test_auc,
            'f1_score': test_f1,
            'balanced_accuracy': test_balanced_acc,
            'classification_report': classification_report(y_test, test_preds),
            'cv_val_acc_mean': cv_results['val_acc_mean'],
            'cv_val_acc_std': cv_results['val_acc_std'],
            'cv_val_auc_mean': cv_results['val_auc_mean'],
            'cv_val_auc_std': cv_results['val_auc_std'],
            'cv_val_f1_mean': cv_results['val_f1_mean'],
            'cv_val_f1_std': cv_results['val_f1_std']
        }
        
        logger.info(f"{model_name} - Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, "
                   f"Test F1: {test_f1:.4f}, CV Val Acc: {cv_results['val_acc_mean']:.4f}±{cv_results['val_acc_std']:.4f}")
    
    return all_models, results


def create_optimized_ensemble(models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Create ensemble using weighted averaging based on CV performance.
    """
    
    logger.info("Creating optimized ensemble...")
    
    # Get predictions and weights
    all_probs = []
    weights = []
    
    for model_name, model_data in models.items():
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Get CV AUC as weight
        cv_auc = model_data.get('cv_val_auc_mean', 0.5)
        weight = max(0, cv_auc - 0.5) * 2  # Convert to 0-1 range
        
        model.eval()
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = F.softmax(outputs, dim=1)[:, 1].numpy()
            all_probs.append(probs)
            weights.append(weight)
    
    # Weighted average
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)
    
    ensemble_probs = np.average(all_probs, axis=0, weights=weights)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_f1 = f1_score(y_test, ensemble_preds)
    ensemble_balanced_acc = balanced_accuracy_score(y_test, ensemble_preds)
    
    ensemble_results = {
        'accuracy': ensemble_acc,
        'roc_auc': ensemble_auc,
        'f1_score': ensemble_f1,
        'balanced_accuracy': ensemble_balanced_acc,
        'classification_report': classification_report(y_test, ensemble_preds),
        'weights': weights.tolist()
    }
    
    logger.info(f"Optimized Ensemble - Test Acc: {ensemble_acc:.4f}, Test AUC: {ensemble_auc:.4f}, "
               f"Test F1: {ensemble_f1:.4f}")
    
    return ensemble_results