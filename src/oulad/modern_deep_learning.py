"""
Modern Deep Learning Architectures for OULAD Dataset

This module implements state-of-the-art deep learning architectures specifically
designed for tabular data, including:
- TabNet: Google's neural network for tabular data
- FT-Transformer: Feature Tokenizer + Transformer  
- NODE: Neural Oblivious Decision Trees
- SAINT: Self-Attention and Intersample Attention Transformer
- AutoInt: Automatic Feature Interaction Learning
- Advanced training techniques and optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import logging
from typing import Dict, Tuple, Optional, List, Union
import math
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TabNet(nn.Module):
    """
    Implementation of TabNet: Attentive Interpretable Tabular Learning
    https://arxiv.org/abs/1908.07442
    """
    
    def __init__(self, input_dim: int, output_dim: int = 2, n_d: int = 64, n_a: int = 64,
                 n_steps: int = 3, gamma: float = 1.3, n_independent: int = 2,
                 n_shared: int = 2, epsilon: float = 1e-15, virtual_batch_size: int = 128,
                 momentum: float = 0.02, mask_type: str = 'sparsemax'):
        super(TabNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        
        # Feature transformer
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=momentum)
        
        # Shared layers
        if n_shared > 0:
            self.shared = nn.ModuleList([
                nn.Linear(input_dim if i == 0 else n_d + n_a, n_d + n_a)
                for i in range(n_shared)
            ])
        else:
            self.shared = None
            
        # Independent layers for each step
        self.steps = nn.ModuleList([
            TabNetStep(input_dim, n_d, n_a, n_independent, virtual_batch_size, momentum)
            for _ in range(n_steps)
        ])
        
        # Final mapping
        self.final_mapping = nn.Linear(n_d, output_dim, bias=False)
        
        # Initialize weights
        self.initialize_non_glu()
        
    def initialize_non_glu(self):
        """Initialize non-GLU layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
                    
    def forward(self, x):
        # Apply initial batch norm
        x = self.initial_bn(x)
        
        # Apply shared layers
        if self.shared is not None:
            for shared_layer in self.shared:
                x = F.relu(shared_layer(x))
        
        # Process through decision steps
        output_aggregated = torch.zeros(x.shape[0], self.n_d, device=x.device)
        masked_features = x
        mask_values = torch.zeros(x.shape[0], self.input_dim, device=x.device)
        
        for step in self.steps:
            step_output, mask = step(masked_features, mask_values)
            output_aggregated += step_output
            mask_values = mask_values + mask
            
            # Update masked features
            masked_features = x * (self.gamma - mask_values)
            
        # Final output
        output = self.final_mapping(output_aggregated)
        
        return output


class TabNetStep(nn.Module):
    """Single step in TabNet architecture."""
    
    def __init__(self, input_dim: int, n_d: int, n_a: int, n_independent: int,
                 virtual_batch_size: int, momentum: float):
        super(TabNetStep, self).__init__()
        
        self.n_d = n_d
        self.n_a = n_a
        
        # Attentive transformer
        self.attentive_transformer = AttentiveTransformer(
            input_dim, n_a, virtual_batch_size, momentum
        )
        
        # Feature transformer
        self.feature_transformer = FeatureTransformer(
            input_dim, n_d + n_a, n_independent, virtual_batch_size, momentum
        )
        
    def forward(self, processed_feat, prior_mask):
        # Attentive transformer
        mask = self.attentive_transformer(processed_feat, prior_mask)
        
        # Apply mask
        masked_feat = processed_feat * mask
        
        # Feature transformer
        output = self.feature_transformer(masked_feat)
        
        # Split decision and attention
        decision_out = output[:, :self.n_d]
        attention_out = output[:, self.n_d:]
        
        return decision_out, mask


class AttentiveTransformer(nn.Module):
    """Attentive transformer for feature selection."""
    
    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int, momentum: float):
        super(AttentiveTransformer, self).__init__()
        
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = nn.BatchNorm1d(output_dim, momentum=momentum)
        
    def forward(self, processed_feat, prior_mask):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.sigmoid(x)  # Simplified sparsemax
        
        # Apply prior mask
        x = x * prior_mask
        
        return x


class FeatureTransformer(nn.Module):
    """Feature transformer with GLU activation."""
    
    def __init__(self, input_dim: int, output_dim: int, n_independent: int,
                 virtual_batch_size: int, momentum: float):
        super(FeatureTransformer, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else output_dim, output_dim * 2),
                nn.BatchNorm1d(output_dim * 2, momentum=momentum),
                GLULayer()
            ) for i in range(n_independent)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        return x


class GLULayer(nn.Module):
    """Gated Linear Unit layer."""
    
    def forward(self, x):
        return F.glu(x, dim=1)


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular data
    Based on "Revisiting Deep Learning Models for Tabular Data"
    """
    
    def __init__(self, input_dim: int, d_token: int = 192, n_blocks: int = 3,
                 attention_dropout: float = 0.2, ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0, n_heads: int = 8,
                 d_ffn_factor: float = 4/3, activation: str = 'reglu'):
        super(FTTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_token = d_token
        
        # Feature tokenizer - convert each feature to a token
        self.feature_tokenizer = nn.Linear(1, d_token)
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(input_dim + 1, d_token))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn_factor, attention_dropout,
                           ffn_dropout, residual_dropout, activation)
            for _ in range(n_blocks)
        ])
        
        # Layer normalization
        self.ln = nn.LayerNorm(d_token)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token // 2, 2)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Tokenize features
        x = x.unsqueeze(-1)  # (batch_size, n_features, 1)
        tokens = self.feature_tokenizer(x)  # (batch_size, n_features, d_token)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Add positional embeddings
        tokens += self.positional_embeddings.unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        # Use CLS token for classification
        cls_output = tokens[:, 0]  # (batch_size, d_token)
        cls_output = self.ln(cls_output)
        
        return self.head(cls_output)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, d_token: int, n_heads: int, d_ffn_factor: float,
                 attention_dropout: float, ffn_dropout: float,
                 residual_dropout: float, activation: str):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            d_token, n_heads, dropout=attention_dropout, batch_first=True
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(d_token)
        self.ln2 = nn.LayerNorm(d_token)
        
        # Feed-forward network
        d_ffn = int(d_token * d_ffn_factor)
        if activation == 'reglu':
            self.ffn = ReGLU(d_token, d_ffn, ffn_dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_token, d_ffn),
                nn.ReLU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(d_ffn, d_token)
            )
        
        self.residual_dropout = nn.Dropout(residual_dropout)
        
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = x + self.residual_dropout(attn_out)
        x = self.ln1(x)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = x + self.residual_dropout(ffn_out)
        x = self.ln2(x)
        
        return x


class ReGLU(nn.Module):
    """ReGLU activation function."""
    
    def __init__(self, d_in: int, d_out: int, dropout: float):
        super(ReGLU, self).__init__()
        self.linear = nn.Linear(d_in, d_out * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear(x)
        x1, x2 = x.chunk(2, dim=-1)
        return self.dropout(x1 * F.relu(x2))


class NODE(nn.Module):
    """
    Neural Oblivious Decision Trees for tabular data
    Based on "Neural Oblivious Decision Trees for Deep Learning on Tabular Data"
    """
    
    def __init__(self, input_dim: int, num_layers: int = 6, tree_dim: int = 2,
                 depth: int = 6, choice_function: str = 'entmax15',
                 bin_function: str = 'entmoid15'):
        super(NODE, self).__init__()
        
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.tree_dim = tree_dim
        self.depth = depth
        
        # Create oblivious decision trees
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(input_dim, tree_dim, depth, choice_function, bin_function)
            for _ in range(num_layers)
        ])
        
        # Final linear layer
        self.fc = nn.Linear(num_layers * tree_dim, 2)
        
    def forward(self, x):
        # Get outputs from all trees
        tree_outputs = []
        for tree in self.trees:
            tree_output = tree(x)
            tree_outputs.append(tree_output)
        
        # Concatenate tree outputs
        concatenated = torch.cat(tree_outputs, dim=1)
        
        # Final prediction
        return self.fc(concatenated)


class ObliviousDecisionTree(nn.Module):
    """Single oblivious decision tree."""
    
    def __init__(self, input_dim: int, output_dim: int, depth: int,
                 choice_function: str, bin_function: str):
        super(ObliviousDecisionTree, self).__init__()
        
        self.depth = depth
        self.output_dim = output_dim
        
        # Feature selection for each level
        self.feature_selectors = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(depth)
        ])
        
        # Leaf values
        num_leaves = 2 ** depth
        self.leaf_values = nn.Parameter(torch.randn(num_leaves, output_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initialize leaf indices
        leaf_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Traverse tree levels
        for level, selector in enumerate(self.feature_selectors):
            # Get decision values
            decisions = torch.sigmoid(selector(x)).squeeze(-1)
            
            # Convert to binary decisions
            binary_decisions = (decisions > 0.5).long()
            
            # Update leaf indices
            leaf_indices = leaf_indices * 2 + binary_decisions
        
        # Get leaf values
        output = self.leaf_values[leaf_indices]
        
        return output


class SAINT(nn.Module):
    """
    Self-Attention and Intersample Attention Transformer
    Based on "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training"
    """
    
    def __init__(self, input_dim: int, embed_dim: int = 32, depth: int = 6,
                 heads: int = 8, dim_head: int = 16, dropout: float = 0.1,
                 use_intersample: bool = True):
        super(SAINT, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_intersample = use_intersample
        
        # Feature embeddings
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])
        
        # Positional embeddings
        self.pos_embeddings = nn.Parameter(torch.randn(input_dim, embed_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SAINTLayer(embed_dim, heads, dim_head, dropout, use_intersample)
            for _ in range(depth)
        ])
        
        # Output layer
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2)
        )
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Embed each feature
        embedded_features = []
        for i, embedding in enumerate(self.feature_embeddings):
            feat_embed = embedding(x[:, i:i+1])
            embedded_features.append(feat_embed)
        
        # Stack embeddings
        x = torch.stack(embedded_features, dim=1).squeeze(-1)
        
        # Add positional embeddings
        x += self.pos_embeddings.unsqueeze(0)
        
        # Apply SAINT layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.mlp_head(x)


class SAINTLayer(nn.Module):
    """Single SAINT layer with self-attention and intersample attention."""
    
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float,
                 use_intersample: bool):
        super(SAINTLayer, self).__init__()
        
        self.use_intersample = use_intersample
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        
        # Intersample attention (if enabled)
        if use_intersample:
            self.intersample_attn = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        if use_intersample:
            self.ln3 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + self.dropout(attn_out))
        
        # Intersample attention
        if self.use_intersample:
            # Transpose for intersample attention
            x_t = x.transpose(0, 1)  # (seq_len, batch_size, dim)
            intersample_out, _ = self.intersample_attn(x_t, x_t, x_t)
            intersample_out = intersample_out.transpose(0, 1)
            x = self.ln2(x + self.dropout(intersample_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.ln3(x + self.dropout(ff_out)) if self.use_intersample else self.ln2(x + self.dropout(ff_out))
        
        return x


class AutoInt(nn.Module):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    """
    
    def __init__(self, input_dim: int, embed_dim: int = 16, num_heads: int = 2,
                 num_layers: int = 3, dropout: float = 0.1, use_residual: bool = True):
        super(AutoInt, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_residual = use_residual
        
        # Feature embeddings
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim * embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Embed each feature
        embedded_features = []
        for i, embedding in enumerate(self.embeddings):
            feat_embed = embedding(x[:, i:i+1])
            embedded_features.append(feat_embed)
        
        # Stack embeddings: (batch_size, num_features, embed_dim)
        embedded_x = torch.stack(embedded_features, dim=1).squeeze(-1)
        
        # Apply attention layers
        for attention, layer_norm in zip(self.attention_layers, self.layer_norms):
            attn_out, _ = attention(embedded_x, embedded_x, embedded_x)
            
            if self.use_residual:
                embedded_x = layer_norm(embedded_x + attn_out)
            else:
                embedded_x = layer_norm(attn_out)
        
        # Flatten for output
        flattened = embedded_x.view(batch_size, -1)
        
        return self.output_layer(flattened)


def train_modern_deep_learning_models(X_train: np.ndarray, y_train: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray,
                                     random_state: int = 42) -> Dict:
    """
    Train modern deep learning models on OULAD dataset.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        X_test: Test features
        y_test: Test labels
        random_state: Random seed
        
    Returns:
        Dictionary containing trained models and results
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_dim = X_train_scaled.shape[1]
    
    # Model configurations
    model_configs = {
        'TabNet': {
            'model': TabNet(input_dim=input_dim, n_d=32, n_a=32, n_steps=3),
            'epochs': 150,
            'batch_size': 256,
            'lr': 0.02,
            'weight_decay': 1e-5
        },
        'FT-Transformer': {
            'model': FTTransformer(input_dim=input_dim, d_token=64, n_blocks=3),
            'epochs': 100,
            'batch_size': 128,
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'NODE': {
            'model': NODE(input_dim=input_dim, num_layers=8, tree_dim=3, depth=6),
            'epochs': 200,
            'batch_size': 512,
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'SAINT': {
            'model': SAINT(input_dim=input_dim, embed_dim=32, depth=4, heads=4),
            'epochs': 120,
            'batch_size': 128,
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'AutoInt': {
            'model': AutoInt(input_dim=input_dim, embed_dim=16, num_heads=2, num_layers=3),
            'epochs': 100,
            'batch_size': 256,
            'lr': 1e-3,
            'weight_decay': 1e-4
        }
    }
    
    results = {}
    trained_models = {}
    
    for model_name, config in model_configs.items():
        logger.info(f"Training {model_name}...")
        
        model = config['model'].to(device)
        
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Train model
        model_results = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['epochs'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            device=device
        )
        
        results[model_name] = model_results
        trained_models[model_name] = {
            'model': model.cpu(),
            'scaler': scaler,
            'config': config
        }
        
        logger.info(f"{model_name} - Test Accuracy: {model_results['test_accuracy']:.4f}, "
                   f"Test AUC: {model_results['test_auc']:.4f}")
    
    return {
        'models': trained_models,
        'results': results,
        'scaler': scaler
    }


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
               epochs: int, lr: float, weight_decay: float, device: torch.device) -> Dict:
    """Train a single model."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_test_auc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
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
        
        scheduler.step()
        
        # Validation phase
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_probs = []
            test_preds = []
            test_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    probs = F.softmax(outputs, dim=1)[:, 1]
                    _, predicted = torch.max(outputs, 1)
                    
                    test_probs.extend(probs.cpu().numpy())
                    test_preds.extend(predicted.cpu().numpy())
                    test_labels.extend(batch_y.cpu().numpy())
            
            test_auc = roc_auc_score(test_labels, test_probs)
            
            if test_auc > best_test_auc:
                best_test_auc = test_auc
                best_model_state = {name: param.clone() for name, param in model.named_parameters()}
        
        if (epoch + 1) % 25 == 0:
            train_acc = train_correct / train_total
            logger.info(f"Epoch {epoch + 1}/{epochs}: Train Acc: {train_acc:.4f}, "
                       f"Train Loss: {train_loss/len(train_loader):.4f}")
    
    # Restore best model
    if best_model_state is not None:
        for name, param in model.named_parameters():
            param.data.copy_(best_model_state[name])
    
    # Final evaluation
    model.eval()
    test_probs = []
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_f1 = f1_score(test_labels, test_preds)
    
    return {
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_probs': test_probs,
        'test_preds': test_preds
    }