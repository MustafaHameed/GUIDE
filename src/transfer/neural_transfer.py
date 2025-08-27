"""
Enhanced Neural Network Transfer Learning for OULAD → UCI

This module implements advanced neural network approaches for transfer learning:
1. Pre-training on source domain with domain adaptation layers
2. Fine-tuning on target domain with careful regularization 
3. Adversarial training for domain-invariant features
4. Advanced architectures with attention mechanisms
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

logger = logging.getLogger(__name__)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class DomainAdversarialNet(nn.Module):
    """Domain Adversarial Neural Network for transfer learning."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.3, lambda_domain: float = 0.1):
        super().__init__()
        self.lambda_domain = lambda_domain
        
        # Feature extractor
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Class classifier 
        self.class_classifier = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """Forward pass with gradient reversal for domain classifier."""
        features = self.feature_extractor(x)
        
        # Class prediction
        class_output = self.class_classifier(features)
        
        # Domain prediction with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        
        return class_output, domain_output, features


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class AttentionTransferNet(nn.Module):
    """Transfer learning network with attention mechanism."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.3):
        super().__init__()
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.Tanh(),
            nn.Linear(prev_dim // 2, prev_dim),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        output = self.classifier(attended_features)
        return output, attended_features


def train_domain_adversarial_model(
    X_source: np.ndarray, y_source: np.ndarray,
    X_target: np.ndarray, y_target: np.ndarray,
    epochs: int = 100, batch_size: int = 64, 
    learning_rate: float = 0.001, lambda_domain: float = 0.1
) -> DomainAdversarialNet:
    """Train domain adversarial network."""
    
    # Prepare data
    X_source_tensor = torch.FloatTensor(X_source).to(device)
    y_source_tensor = torch.LongTensor(y_source).to(device)
    X_target_tensor = torch.FloatTensor(X_target).to(device)
    y_target_tensor = torch.LongTensor(y_target).to(device)
    
    # Create domain labels (0 for source, 1 for target)
    source_domain = torch.zeros(len(X_source), dtype=torch.long).to(device)
    target_domain = torch.ones(len(X_target), dtype=torch.long).to(device)
    
    # Initialize model
    model = DomainAdversarialNet(
        input_dim=X_source.shape[1],
        lambda_domain=lambda_domain
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        # Progress of training (for gradient reversal)
        alpha = 2.0 / (1.0 + np.exp(-10 * epoch / epochs)) - 1.0
        
        # Sample batches
        source_indices = torch.randperm(len(X_source))[:batch_size]
        target_indices = torch.randperm(len(X_target))[:batch_size]
        
        X_source_batch = X_source_tensor[source_indices]
        y_source_batch = y_source_tensor[source_indices]
        X_target_batch = X_target_tensor[target_indices]
        
        # Forward pass on source data
        class_output_source, domain_output_source, _ = model(X_source_batch, alpha)
        
        # Forward pass on target data  
        _, domain_output_target, _ = model(X_target_batch, alpha)
        
        # Compute losses
        class_loss = class_criterion(class_output_source, y_source_batch)
        domain_loss_source = domain_criterion(domain_output_source, source_domain[:len(source_indices)])
        domain_loss_target = domain_criterion(domain_output_target, target_domain[:len(target_indices)])
        domain_loss = domain_loss_source + domain_loss_target
        
        total_loss = class_loss + lambda_domain * domain_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Class Loss = {class_loss:.4f}, Domain Loss = {domain_loss:.4f}")
    
    return model


def train_attention_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 100, batch_size: int = 64,
    learning_rate: float = 0.001, early_stopping_patience: int = 10
) -> AttentionTransferNet:
    """Train attention-based transfer model."""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Initialize model
    model = AttentionTransferNet(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        
        # Training loop
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs, _ = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            
            _, predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (predicted == y_val_tensor).float().mean()
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def neural_transfer_experiment(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    method: str = 'domain_adversarial',  # 'domain_adversarial', 'attention', 'standard'
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Neural network transfer learning experiment.
    
    Args:
        source_data: Source domain data with 'label' column
        target_data: Target domain data with 'label' column
        method: Neural network method to use
        test_size: Fraction of target data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with performance metrics
    """
    
    logger.info(f"Starting neural transfer experiment with method: {method}")
    
    # Prepare data
    feature_cols = [col for col in source_data.columns if col != 'label']
    target_feature_cols = [col for col in target_data.columns if col != 'label']
    common_features = list(set(feature_cols) & set(target_feature_cols))
    
    X_source = source_data[common_features].values
    y_source = source_data['label'].values.astype(int)
    X_target = target_data[common_features].values
    y_target = target_data['label'].values.astype(int)
    
    # Split target data
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target, test_size=test_size, random_state=random_state, 
        stratify=y_target
    )
    
    # Preprocessing
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_train_scaled = scaler.transform(X_target_train)
    X_target_test_scaled = scaler.transform(X_target_test)
    
    logger.info(f"Data shapes - Source: {X_source_scaled.shape}, Target: {X_target_test_scaled.shape}")
    
    # Train model based on method
    if method == 'domain_adversarial':
        model = train_domain_adversarial_model(
            X_source_scaled, y_source,
            X_target_train_scaled, y_target_train,
            epochs=150, lambda_domain=0.1
        )
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_target_test_scaled).to(device)
            class_output, _, _ = model(X_test_tensor)
            y_prob = torch.softmax(class_output, dim=1)[:, 1].cpu().numpy()
            y_pred = torch.argmax(class_output, dim=1).cpu().numpy()
            
    elif method == 'attention':
        # Use some target data for training
        X_combined = np.vstack([X_source_scaled, X_target_train_scaled])
        y_combined = np.hstack([y_source, y_target_train])
        
        # Split combined data for training/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=random_state,
            stratify=y_combined
        )
        
        model = train_attention_model(X_train, y_train, X_val, y_val, epochs=150)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_target_test_scaled).to(device)
            outputs, _ = model(X_test_tensor)
            y_prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            
    else:  # standard neural network
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            alpha=0.01
        )
        model.fit(X_source_scaled, y_source)
        y_prob = model.predict_proba(X_target_test_scaled)[:, 1]
        y_pred = model.predict(X_target_test_scaled)
    
    # Compute metrics
    results = {
        'method': method,
        'accuracy': accuracy_score(y_target_test, y_pred),
        'auc': roc_auc_score(y_target_test, y_prob),
        'f1': f1_score(y_target_test, y_pred),
        'n_features': len(common_features),
        'n_source_samples': len(y_source),
        'n_target_test_samples': len(y_target_test),
        'target_class_distribution': pd.Series(y_target_test).value_counts().to_dict()
    }
    
    logger.info(f"Neural {method} results - Accuracy: {results['accuracy']:.3f}, AUC: {results['auc']:.3f}")
    
    return results


def main():
    """Test neural network transfer learning approaches."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    from transfer.uci_transfer import create_shared_feature_mapping, prepare_oulad_features, prepare_uci_features
    
    # Load and prepare data
    feature_mapping = create_shared_feature_mapping()
    oulad_df = pd.read_csv('data/oulad/processed/oulad_ml.csv')
    oulad_shared = prepare_oulad_features(oulad_df, feature_mapping)
    uci_shared = prepare_uci_features('student-mat-fixed.csv', feature_mapping)
    
    # Clean data
    oulad_clean = oulad_shared.dropna(subset=['label'])
    uci_clean = uci_shared.dropna(subset=['label'])
    
    print("=== Neural Transfer Learning Results ===")
    print(f"OULAD: {oulad_clean.shape}, UCI: {uci_clean.shape}")
    
    # Test different neural approaches
    methods = ['standard', 'attention', 'domain_adversarial']
    
    for method in methods:
        print(f"\n{method.title()} Neural Network:")
        try:
            results = neural_transfer_experiment(oulad_clean, uci_clean, method=method)
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  AUC: {results['auc']:.3f}")
            print(f"  F1: {results['f1']:.3f}")
            print(f"  Features: {results['n_features']}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()