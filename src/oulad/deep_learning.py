"""
Advanced Deep Learning Models for OULAD Dataset

This module implements sophisticated PyTorch-based neural networks for the OULAD dataset,
providing significant improvements over the basic sklearn MLPClassifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TabularMLP(nn.Module):
    """
    Advanced Multi-Layer Perceptron for tabular data with modern regularization techniques.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 dropout: float = 0.3, use_batch_norm: bool = True):
        super(TabularMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class ResidualMLP(nn.Module):
    """
    MLP with residual connections for better gradient flow and performance.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_blocks: int = 3,
                 dropout: float = 0.3, use_batch_norm: bool = True):
        super(ResidualMLP, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout, use_batch_norm) 
            for _ in range(num_blocks)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        return self.output_head(x)


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.3, use_batch_norm: bool = True):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
        )
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        return F.relu(out + residual)


class WideAndDeepModel(nn.Module):
    """
    Wide & Deep model architecture combining linear and deep components.
    """
    
    def __init__(self, input_dim: int, embedding_dims: Optional[Dict[str, int]] = None,
                 deep_hidden_dims: List[int] = [256, 128, 64], dropout: float = 0.3):
        super(WideAndDeepModel, self).__init__()
        
        self.input_dim = input_dim
        
        # Wide component (linear)
        self.wide = nn.Linear(input_dim, 1)
        
        # Deep component
        self.deep = TabularMLP(input_dim, deep_hidden_dims, dropout)
        
        # Final combination layer
        self.final = nn.Linear(3, 2)  # 1 from wide + 2 from deep
        
    def forward(self, x):
        # Wide component
        wide_out = self.wide(x)
        
        # Deep component  
        deep_out = self.deep(x)
        
        # Combine wide and deep
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.final(combined)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def save_checkpoint(self, model: nn.Module):
        if self.restore_best_weights:
            self.best_weights = {name: param.clone() for name, param in model.named_parameters()}
    
    def restore_best(self, model: nn.Module):
        if self.best_weights is not None:
            for name, param in model.named_parameters():
                param.data.copy_(self.best_weights[name])


def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 15,
    device: str = 'auto'
) -> Dict:
    """
    Train a PyTorch model with advanced training techniques.
    
    Returns:
        Dictionary containing training history and final metrics
    """
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Loss function with class weights for imbalanced data
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(class_counts) / class_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    logger.info(f"Training model on {device} with {len(X_train)} samples")
    
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
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Record history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Early stopping check
        if early_stopping(val_acc, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}: "
                       f"Train Acc: {history['train_acc'][-1]:.4f}, "
                       f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
    
    # Restore best weights
    early_stopping.restore_best(model)
    
    return history


def train_deep_learning_models(X_train: np.ndarray, y_train: np.ndarray, 
                              X_test: np.ndarray, y_test: np.ndarray,
                              random_state: int = 42) -> Dict:
    """
    Train multiple advanced deep learning models on OULAD data.
    
    Returns:
        Dictionary containing trained models and their performance metrics
    """
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_dim = X_train_scaled.shape[1]
    
    # Split training data for validation
    val_split = 0.2
    val_size = int(len(X_train_scaled) * val_split)
    
    # Use stratified split to maintain class distribution
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=val_split, stratify=y_train, random_state=random_state
    )
    
    models = {}
    results = {}
    
    # Model configurations
    model_configs = {
        'advanced_mlp': {
            'model': TabularMLP(input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.4),
            'epochs': 150,
            'learning_rate': 0.001,
            'batch_size': 64
        },
        'residual_mlp': {
            'model': ResidualMLP(input_dim, hidden_dim=256, num_blocks=4, dropout=0.3),
            'epochs': 150,
            'learning_rate': 0.001,
            'batch_size': 64
        },
        'wide_deep': {
            'model': WideAndDeepModel(input_dim, deep_hidden_dims=[256, 128, 64], dropout=0.3),
            'epochs': 120,
            'learning_rate': 0.0008,
            'batch_size': 64
        }
    }
    
    for model_name, config in model_configs.items():
        logger.info(f"Training {model_name}...")
        
        model = config['model']
        
        # Train the model
        history = train_pytorch_model(
            model=model,
            X_train=X_train_split,
            y_train=y_train_split,
            X_val=X_val_split,
            y_val=y_val_split,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            early_stopping_patience=20
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
        
        # Store results
        models[model_name] = {
            'model': model.cpu(),  # Move to CPU for storage
            'scaler': scaler,
            'history': history,
            'config': config
        }
        
        results[model_name] = {
            'accuracy': test_acc,
            'roc_auc': test_auc,
            'classification_report': classification_report(y_test, test_preds),
            'val_accuracy': max(history['val_acc']),
            'val_auc': max(history['val_auc'])
        }
        
        logger.info(f"{model_name} - Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    
    return models, results


def create_ensemble_model(models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Create an ensemble of the trained models using averaging.
    """
    
    logger.info("Creating ensemble model...")
    
    # Get predictions from all models
    all_probs = []
    
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
    
    # Average predictions
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    
    ensemble_results = {
        'accuracy': ensemble_acc,
        'roc_auc': ensemble_auc,
        'classification_report': classification_report(y_test, ensemble_preds)
    }
    
    logger.info(f"Ensemble - Test Accuracy: {ensemble_acc:.4f}, Test AUC: {ensemble_auc:.4f}")
    
    return ensemble_results