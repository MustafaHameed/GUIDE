"""
Final Optimized Deep Learning for OULAD Dataset

This module implements the most targeted approach for the OULAD dataset:
- Better understanding of the data distribution
- Proper feature preprocessing specifically for educational data
- Lightweight but effective architectures
- Advanced ensemble techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import logging
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def analyze_oulad_features(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Analyze OULAD features to understand the data characteristics.
    """
    logger.info("Analyzing OULAD dataset characteristics...")
    
    analysis = {
        'feature_stats': {},
        'correlation_with_target': {},
        'class_balance': {},
        'feature_importance_proxy': {}
    }
    
    # Basic statistics
    analysis['feature_stats'] = {
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'missing_values': X.isnull().sum().sum(),
        'numeric_features': X.select_dtypes(include=[np.number]).shape[1],
        'categorical_features': X.select_dtypes(include=['object']).shape[1]
    }
    
    # Class balance
    class_counts = y.value_counts()
    analysis['class_balance'] = {
        'class_0': class_counts[0],
        'class_1': class_counts[1],
        'imbalance_ratio': class_counts[0] / class_counts[1]
    }
    
    # Simple correlation analysis
    if isinstance(X, pd.DataFrame):
        numeric_X = X.select_dtypes(include=[np.number])
        if not numeric_X.empty:
            correlations = numeric_X.corrwith(y).abs().sort_values(ascending=False)
            analysis['correlation_with_target'] = correlations.head(10).to_dict()
    
    logger.info(f"Dataset analysis: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class imbalance ratio: {analysis['class_balance']['imbalance_ratio']:.2f}")
    
    return analysis


class LightweightTabularNet(nn.Module):
    """
    Lightweight neural network optimized for OULAD-like tabular data.
    Focus on simplicity and effectiveness rather than complexity.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2, activation: str = 'relu'):
        super(LightweightTabularNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization for better gradient flow
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x):
        return self.network(x)


class EnsembleTabularNet(nn.Module):
    """
    Ensemble of lightweight networks for better performance.
    """
    
    def __init__(self, input_dim: int, num_models: int = 3, 
                 hidden_dims: List[int] = [64, 48, 32],
                 activations: List[str] = ['relu', 'gelu', 'swish']):
        super(EnsembleTabularNet, self).__init__()
        
        self.num_models = num_models
        
        # Create multiple models with different architectures
        self.models = nn.ModuleList()
        for i in range(num_models):
            hidden_dim = hidden_dims[i % len(hidden_dims)]
            activation = activations[i % len(activations)]
            
            model = LightweightTabularNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=0.15 + 0.05 * i,  # Varying dropout
                activation=activation
            )
            self.models.append(model)
        
        # Combination weights (learnable)
        self.combination_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x):
        # Get predictions from all models
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Stack outputs
        stacked_outputs = torch.stack(outputs, dim=0)  # (num_models, batch_size, 2)
        
        # Apply softmax to combination weights
        weights = F.softmax(self.combination_weights, dim=0)
        
        # Weighted combination
        combined_output = torch.einsum('m,mbc->bc', weights, stacked_outputs)
        
        return combined_output


def train_final_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     epochs: int = 100, batch_size: int = 64,
                     learning_rate: float = 0.001, weight_decay: float = 1e-4,
                     patience: int = 20, device: str = 'auto') -> Dict:
    """
    Train the final optimized model with best practices.
    """
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Compute sample weights for imbalanced dataset
    sample_weights = compute_sample_weight('balanced', y_train)
    sample_weights_tensor = torch.FloatTensor(sample_weights).to(device)
    
    # Data loader with weighted sampling
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    # Use weighted random sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function with label smoothing
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super(LabelSmoothingCrossEntropy, self).__init__()
            self.smoothing = smoothing
        
        def forward(self, x, target):
            confidence = 1. - self.smoothing
            log_probs = F.log_softmax(x, dim=-1)
            nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -log_probs.mean(dim=-1)
            loss = confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'lr': []
    }
    
    best_val_score = 0
    patience_counter = 0
    best_model_state = None
    
    logger.info(f"Training final model on {device} for {epochs} epochs")
    
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
        
        # Update learning rate
        scheduler.step()
        
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
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Early stopping based on balanced score
        val_score = 0.6 * val_auc + 0.4 * val_acc
        
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_model_state = {name: param.clone() for name, param in model.named_parameters()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 25 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}: Train Acc: {history['train_acc'][-1]:.4f}, "
                       f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Restore best weights
    if best_model_state is not None:
        for name, param in model.named_parameters():
            param.data.copy_(best_model_state[name])
    
    return history


def train_final_optimized_models(X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                random_state: int = 42) -> Dict:
    """
    Train the final optimized models for OULAD dataset.
    """
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    logger.info("Training final optimized deep learning models...")
    
    # Analyze dataset first
    X_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    y_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
    analysis = analyze_oulad_features(X_df, y_series)
    
    # Enhanced preprocessing
    scaler = MinMaxScaler()  # Often works better for neural networks than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    
    input_dim = X_train_scaled.shape[1]
    
    models = {}
    results = {}
    
    # Model configurations
    model_configs = {
        'final_lightweight': {
            'model': LightweightTabularNet(
                input_dim=input_dim,
                hidden_dim=64,
                num_layers=2,
                dropout=0.2,
                activation='relu'
            ),
            'epochs': 150,
            'learning_rate': 0.002,
            'batch_size': 128
        },
        'final_ensemble': {
            'model': EnsembleTabularNet(
                input_dim=input_dim,
                num_models=3,
                hidden_dims=[48, 64, 80],
                activations=['relu', 'gelu', 'swish']
            ),
            'epochs': 120,
            'learning_rate': 0.001,
            'batch_size': 64
        }
    }
    
    for model_name, config in model_configs.items():
        logger.info(f"Training {model_name}...")
        
        model = config['model']
        
        # Train the model
        history = train_final_model(
            model=model,
            X_train=X_train_split,
            y_train=y_train_split,
            X_val=X_val_split,
            y_val=y_val_split,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            patience=25
        )
        
        # Evaluate on test set
        model.eval()
        device = next(model.parameters()).device
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_probs = F.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
            test_preds = torch.max(test_outputs, 1)[1].cpu().numpy()
        
        test_acc = accuracy_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_probs)
        test_f1 = f1_score(y_test, test_preds)
        
        # Store results
        models[model_name] = {
            'model': model.cpu(),
            'scaler': scaler,
            'history': history,
            'config': config,
            'analysis': analysis
        }
        
        results[model_name] = {
            'accuracy': test_acc,
            'roc_auc': test_auc,
            'f1_score': test_f1,
            'classification_report': classification_report(y_test, test_preds),
            'val_accuracy': max(history['val_acc']),
            'val_auc': max(history['val_auc'])
        }
        
        logger.info(f"{model_name} - Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, "
                   f"Test F1: {test_f1:.4f}")
    
    return models, results


def create_final_ensemble(models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Create a final ensemble of all models including traditional ones.
    """
    
    logger.info("Creating final comprehensive ensemble...")
    
    all_probs = []
    model_weights = []
    
    for model_name, model_data in models.items():
        model = model_data['model']
        scaler = model_data['scaler']
        
        model.eval()
        X_test_scaled = scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = F.softmax(outputs, dim=1)[:, 1].numpy()
            all_probs.append(probs)
        
        # Weight by validation performance
        val_auc = model_data.get('history', {}).get('val_auc', [0.5])
        weight = max(val_auc) if val_auc else 0.5
        model_weights.append(weight)
    
    # Normalize weights
    model_weights = np.array(model_weights)
    model_weights = model_weights / model_weights.sum()
    
    # Weighted average
    ensemble_probs = np.average(all_probs, axis=0, weights=model_weights)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    ensemble_f1 = f1_score(y_test, ensemble_preds)
    
    ensemble_results = {
        'accuracy': ensemble_acc,
        'roc_auc': ensemble_auc,
        'f1_score': ensemble_f1,
        'classification_report': classification_report(y_test, ensemble_preds),
        'model_weights': model_weights.tolist()
    }
    
    logger.info(f"Final Ensemble - Test Acc: {ensemble_acc:.4f}, Test AUC: {ensemble_auc:.4f}, "
               f"Test F1: {ensemble_f1:.4f}")
    
    return ensemble_results