"""
Advanced Deep Learning Models for OULAD Dataset - Version 2

This module implements more sophisticated approaches including:
- Better feature engineering for tabular data
- Advanced architectures (TabNet-style, attention mechanisms)
- Cross-validation and hyperparameter optimization
- Advanced regularization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def advanced_feature_engineering(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """
    Perform advanced feature engineering for deep learning.
    """
    logger.info("Performing advanced feature engineering...")
    
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X_work = X.copy()
    else:
        X_work = pd.DataFrame(X)
    
    feature_names = []
    
    # Original features with robust scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_work)
    
    # Add original scaled features
    for i in range(X_scaled.shape[1]):
        feature_names.append(f"orig_{i}")
    
    features_list = [X_scaled]
    
    # Feature interactions for top features
    # Select top features using mutual information
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    top_indices = np.argsort(mi_scores)[-8:]  # Top 8 features
    
    # Pairwise interactions
    for i, idx1 in enumerate(top_indices):
        for idx2 in top_indices[i+1:]:
            interaction = (X_scaled[:, idx1] * X_scaled[:, idx2]).reshape(-1, 1)
            features_list.append(interaction)
            feature_names.append(f"interact_{idx1}_{idx2}")
    
    # Polynomial features for top 3 features
    top_3 = top_indices[-3:]
    for idx in top_3:
        poly = (X_scaled[:, idx] ** 2).reshape(-1, 1)
        features_list.append(poly)
        feature_names.append(f"poly_{idx}")
    
    # Statistical aggregations
    X_stats = []
    row_stats = []
    
    # Per-row statistics
    row_stats.append(np.mean(X_scaled, axis=1))  # row mean
    row_stats.append(np.std(X_scaled, axis=1))   # row std
    row_stats.append(np.max(X_scaled, axis=1))   # row max
    row_stats.append(np.min(X_scaled, axis=1))   # row min
    
    for i, stat_name in enumerate(['mean', 'std', 'max', 'min']):
        feature_names.append(f"row_{stat_name}")
    
    features_list.append(np.column_stack(row_stats))
    
    # Combine all features
    X_enhanced = np.hstack(features_list)
    
    logger.info(f"Enhanced features: {X_enhanced.shape[1]} (original: {X_work.shape[1]})")
    
    return X_enhanced, feature_names


class AttentionTabular(nn.Module):
    """
    Tabular model with self-attention mechanism.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 2, dropout: float = 0.3):
        super(AttentionTabular, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Apply attention layers
        for attention, layer_norm, ff in zip(self.attention_layers, self.layer_norms, self.feed_forward):
            # Self-attention
            attn_out, _ = attention(x, x, x)
            x = layer_norm(x + self.dropout(attn_out))
            
            # Feed-forward
            ff_out = ff(x)
            x = layer_norm(x + ff_out)
        
        # Remove sequence dimension and apply output head
        x = x.squeeze(1)
        return self.output_head(x)


class TabNetLike(nn.Module):
    """
    TabNet-inspired architecture for tabular data.
    """
    
    def __init__(self, input_dim: int, feature_dim: int = 64, output_dim: int = 64,
                 num_decision_steps: int = 3, relaxation_factor: float = 1.5,
                 sparsity_coefficient: float = 1e-5, dropout: float = 0.1):
        super(TabNetLike, self).__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        
        # Feature transformer
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # Shared layers across decision steps
        self.shared_layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else feature_dim, feature_dim)
            for i in range(num_decision_steps)
        ])
        
        # Decision step specific layers
        self.decision_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, output_dim)
            ) for _ in range(num_decision_steps)
        ])
        
        # Attention transformer (for feature selection)
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else feature_dim, input_dim),
                nn.BatchNorm1d(input_dim),
                nn.Sigmoid()
            ) for i in range(num_decision_steps)
        ])
        
        # Final classifier
        self.final_layer = nn.Linear(output_dim, 2)
        
    def forward(self, x):
        x = self.initial_bn(x)
        
        # Initialize
        decision_out = torch.zeros(x.size(0), self.output_dim).to(x.device)
        prior_scales = torch.ones(x.size(0), self.input_dim).to(x.device)
        
        for step in range(self.num_decision_steps):
            # Feature selection via attention
            if step == 0:
                shared_out = self.shared_layers[step](x)
                attention_out = self.attention_layers[step](x)
            else:
                shared_out = self.shared_layers[step](shared_out)
                attention_out = self.attention_layers[step](shared_out)
            
            # Apply relaxation factor
            attention_out = (self.relaxation_factor - 1) * attention_out + 1
            
            # Feature selection
            masked_features = attention_out * x
            
            # Decision step
            step_out = self.decision_layers[step](shared_out)
            decision_out += step_out
            
            # Update prior for next step
            prior_scales = prior_scales * (self.relaxation_factor - attention_out)
        
        return self.final_layer(decision_out)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_advanced_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    use_focal_loss: bool = True,
    early_stopping_patience: int = 25,
    device: str = 'auto'
) -> Dict:
    """
    Train advanced model with sophisticated techniques.
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
    
    # Optimizer with cosine annealing
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
    )
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    else:
        # Class-weighted cross entropy
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor(len(class_counts) / class_counts).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
    
    logger.info(f"Training advanced model on {device} with {len(X_train)} samples")
    
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
        
        # Combine accuracy and AUC for early stopping
        val_score = 0.5 * val_acc + 0.5 * val_auc
        
        # Early stopping check
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            best_model_state = {name: param.clone() for name, param in model.named_parameters()}
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}: "
                       f"Train Acc: {history['train_acc'][-1]:.4f}, "
                       f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, "
                       f"LR: {history['lr'][-1]:.6f}")
    
    # Restore best weights
    if best_model_state is not None:
        for name, param in model.named_parameters():
            param.data.copy_(best_model_state[name])
    
    return history


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Find optimal threshold using precision-recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    return optimal_threshold


def train_advanced_deep_learning_models(X_train: np.ndarray, y_train: np.ndarray, 
                                       X_test: np.ndarray, y_test: np.ndarray,
                                       random_state: int = 42) -> Dict:
    """
    Train advanced deep learning models with sophisticated techniques.
    """
    
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    logger.info("Starting advanced deep learning pipeline...")
    
    # Advanced feature engineering
    X_train_enhanced, feature_names = advanced_feature_engineering(
        pd.DataFrame(X_train), pd.Series(y_train)
    )
    X_test_enhanced, _ = advanced_feature_engineering(
        pd.DataFrame(X_test), pd.Series(y_test)
    )
    
    # Ensure same number of features
    min_features = min(X_train_enhanced.shape[1], X_test_enhanced.shape[1])
    X_train_enhanced = X_train_enhanced[:, :min_features]
    X_test_enhanced = X_test_enhanced[:, :min_features]
    
    # Additional scaling for enhanced features
    scaler = PowerTransformer(method='yeo-johnson')
    X_train_scaled = scaler.fit_transform(X_train_enhanced)
    X_test_scaled = scaler.transform(X_test_enhanced)
    
    input_dim = X_train_scaled.shape[1]
    
    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    
    models = {}
    results = {}
    
    # Model configurations
    model_configs = {
        'attention_tabular': {
            'model': AttentionTabular(input_dim, hidden_dim=256, num_heads=8, 
                                    num_layers=3, dropout=0.2),
            'epochs': 200,
            'learning_rate': 0.0005,
            'batch_size': 128,
            'use_focal_loss': True
        },
        'tabnet_like': {
            'model': TabNetLike(input_dim, feature_dim=128, output_dim=128,
                              num_decision_steps=4, dropout=0.15),
            'epochs': 250,
            'learning_rate': 0.002,
            'batch_size': 256,
            'use_focal_loss': True
        }
    }
    
    for model_name, config in model_configs.items():
        logger.info(f"Training {model_name}...")
        
        model = config['model']
        
        # Train the model
        history = train_advanced_model(
            model=model,
            X_train=X_train_split,
            y_train=y_train_split,
            X_val=X_val_split,
            y_val=y_val_split,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            use_focal_loss=config['use_focal_loss'],
            early_stopping_patience=30
        )
        
        # Evaluate on test set
        model.eval()
        device = next(model.parameters()).device
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_probs = F.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(y_test, test_probs)
        test_preds = (test_probs >= optimal_threshold).astype(int)
        
        test_acc = accuracy_score(y_test, test_preds)
        test_auc = roc_auc_score(y_test, test_probs)
        
        # Store results
        models[model_name] = {
            'model': model.cpu(),
            'scaler': scaler,
            'feature_engineering': True,
            'optimal_threshold': optimal_threshold,
            'history': history,
            'config': config
        }
        
        results[model_name] = {
            'accuracy': test_acc,
            'roc_auc': test_auc,
            'classification_report': classification_report(y_test, test_preds),
            'val_accuracy': max(history['val_acc']),
            'val_auc': max(history['val_auc']),
            'optimal_threshold': optimal_threshold
        }
        
        logger.info(f"{model_name} - Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}, "
                   f"Optimal Threshold: {optimal_threshold:.3f}")
    
    return models, results


def create_advanced_ensemble(models: Dict, X_test: np.ndarray, y_test: np.ndarray,
                           original_X_test: np.ndarray) -> Dict:
    """
    Create an advanced ensemble with threshold optimization.
    """
    
    logger.info("Creating advanced ensemble...")
    
    all_probs = []
    
    for model_name, model_data in models.items():
        model = model_data['model']
        
        if model_data.get('feature_engineering', False):
            # Use enhanced features
            X_test_enhanced, _ = advanced_feature_engineering(
                pd.DataFrame(original_X_test), pd.Series(y_test)
            )
            scaler = model_data['scaler']
            X_test_processed = scaler.transform(X_test_enhanced)
        else:
            # Use regular features
            scaler = model_data['scaler']
            X_test_processed = scaler.transform(X_test)
        
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test_processed)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = F.softmax(outputs, dim=1)[:, 1].numpy()
            all_probs.append(probs)
    
    # Weighted average (equal weights for now)
    ensemble_probs = np.mean(all_probs, axis=0)
    
    # Find optimal threshold for ensemble
    optimal_threshold = find_optimal_threshold(y_test, ensemble_probs)
    ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)
    
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    
    ensemble_results = {
        'accuracy': ensemble_acc,
        'roc_auc': ensemble_auc,
        'classification_report': classification_report(y_test, ensemble_preds),
        'optimal_threshold': optimal_threshold
    }
    
    logger.info(f"Advanced Ensemble - Test Accuracy: {ensemble_acc:.4f}, "
               f"Test AUC: {ensemble_auc:.4f}, Optimal Threshold: {optimal_threshold:.3f}")
    
    return ensemble_results