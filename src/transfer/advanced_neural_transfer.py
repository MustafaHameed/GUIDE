"""
Advanced Neural Transfer Learning Techniques

This module implements state-of-the-art neural transfer learning approaches
including transformer-based domain adaptation, contrastive learning, and
progressive domain adaptation for educational dataset transfer.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class TransformerDomainAdapter(nn.Module):
    """Transformer-based domain adaptation with attention mechanisms."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 3, 
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature embedding
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder for feature learning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Domain-invariant feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)  # Source vs Target
        )
        
    def forward(self, x, lambda_grl=1.0):
        batch_size = x.size(0)
        
        # Embed features
        x = self.feature_embedding(x)
        
        # Add positional information (simple approach)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Transformer encoding
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        
        # Extract domain-invariant features
        features = self.feature_extractor(x)
        
        # Classification
        class_logits = self.classifier(features)
        
        # Domain discrimination with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, lambda_grl)
        domain_logits = self.domain_discriminator(reversed_features)
        
        return class_logits, domain_logits, features


class ContrastiveDomainAdapter(nn.Module):
    """Contrastive learning for domain-invariant representations."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 projection_dim: int = 128, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 2)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        logits = self.classifier(features)
        return logits, features, projections
    
    def contrastive_loss(self, projections1, projections2, labels1, labels2):
        """Supervised contrastive loss for domain alignment."""
        # Normalize projections
        proj1 = F.normalize(projections1, dim=1)
        proj2 = F.normalize(projections2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(proj1, proj2.T) / self.temperature
        
        # Create positive mask (same class across domains)
        labels1 = labels1.unsqueeze(1)
        labels2 = labels2.unsqueeze(0)
        positive_mask = (labels1 == labels2).float()
        
        # Contrastive loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean log-likelihood of positive pairs
        loss = -(positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
        return loss.mean()


class ProgressiveDomainAdapter(nn.Module):
    """Progressive domain adaptation with curriculum learning."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        self.stages = len(hidden_dims)
        self.stage_models = nn.ModuleList()
        
        # Create progressive stages
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            stage = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.stage_models.append(stage)
            prev_dim = hidden_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1] // 2, 2)
        )
        
    def forward(self, x, stage: int = None):
        if stage is None:
            stage = self.stages
        
        # Progressive feature learning
        for i in range(min(stage, self.stages)):
            x = self.stage_models[i](x)
        
        # Classification
        if stage >= self.stages:
            x = self.classifier(x)
        
        return x


class NeuralTransferLearningClassifier(BaseEstimator, ClassifierMixin):
    """
    Advanced neural transfer learning classifier with multiple techniques.
    """
    
    def __init__(self, 
                 method: str = 'transformer',
                 hidden_dim: int = 256,
                 num_epochs: int = 100,
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 lambda_grl: float = 1.0,
                 device: str = 'auto',
                 random_state: int = 42):
        """
        Initialize neural transfer learning classifier.
        
        Args:
            method: Transfer method ('transformer', 'contrastive', 'progressive')
            hidden_dim: Hidden layer dimension
            num_epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            lambda_grl: Gradient reversal lambda for adversarial training
            device: Device for training ('auto', 'cpu', 'cuda')
            random_state: Random seed
        """
        self.method = method
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_grl = lambda_grl
        self.device = device
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Initialized {method} neural transfer classifier on {self.device}")
    
    def fit(self, X_source, y_source, X_target=None, y_target=None):
        """
        Train the neural transfer learning model.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features (optional)
            y_target: Target domain labels (optional, for validation)
        """
        # Prepare data
        X_source = np.array(X_source, dtype=np.float32)
        y_source = np.array(y_source, dtype=np.long)
        
        if X_target is not None:
            X_target = np.array(X_target, dtype=np.float32)
            if y_target is not None:
                y_target = np.array(y_target, dtype=np.long)
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_source_scaled = self.scaler_.fit_transform(X_source)
        if X_target is not None:
            X_target_scaled = self.scaler_.transform(X_target)
        
        input_dim = X_source_scaled.shape[1]
        
        # Initialize model
        if self.method == 'transformer':
            self.model_ = TransformerDomainAdapter(input_dim, self.hidden_dim)
        elif self.method == 'contrastive':
            self.model_ = ContrastiveDomainAdapter(input_dim, self.hidden_dim)
        elif self.method == 'progressive':
            self.model_ = ProgressiveDomainAdapter(input_dim)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.model_.to(self.device)
        
        # Prepare data loaders
        source_dataset = TensorDataset(
            torch.tensor(X_source_scaled),
            torch.tensor(y_source)
        )
        source_loader = DataLoader(
            source_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if X_target is not None:
            target_tensor = torch.tensor(X_target_scaled)
            if y_target is not None:
                target_dataset = TensorDataset(target_tensor, torch.tensor(y_target))
                target_loader = DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True)
            else:
                target_dataset = TensorDataset(target_tensor, torch.zeros(len(target_tensor), dtype=torch.long))
                target_loader = DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train model
        self._train_model(source_loader, target_loader if X_target is not None else None)
        
        return self
    
    def _train_model(self, source_loader, target_loader=None):
        """Train the neural transfer model."""
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        self.model_.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Source domain training
            for batch_idx, (x_source, y_source) in enumerate(source_loader):
                x_source = x_source.to(self.device)
                y_source = y_source.to(self.device)
                
                optimizer.zero_grad()
                
                if self.method == 'transformer':
                    class_logits, domain_logits, features = self.model_(x_source, self.lambda_grl)
                    
                    # Classification loss
                    class_loss = F.cross_entropy(class_logits, y_source)
                    
                    # Domain adversarial loss (all source samples are domain 0)
                    domain_labels = torch.zeros(x_source.size(0), dtype=torch.long, device=self.device)
                    domain_loss = F.cross_entropy(domain_logits, domain_labels)
                    
                    loss = class_loss + domain_loss
                
                elif self.method == 'contrastive':
                    logits, features, projections = self.model_(x_source)
                    loss = F.cross_entropy(logits, y_source)
                    
                    # Add contrastive loss if target data available
                    if target_loader is not None:
                        try:
                            x_target, y_target = next(iter(target_loader))
                            x_target = x_target.to(self.device)
                            y_target = y_target.to(self.device)
                            
                            _, _, proj_target = self.model_(x_target)
                            contrastive_loss = self.model_.contrastive_loss(
                                projections, proj_target, y_source, y_target
                            )
                            loss += 0.1 * contrastive_loss
                        except:
                            pass
                
                elif self.method == 'progressive':
                    # Progressive training - start with early stages
                    current_stage = min(epoch // 20 + 1, self.model_.stages + 1)
                    output = self.model_(x_source, current_stage)
                    
                    if current_stage > self.model_.stages:  # Final stage with classification
                        loss = F.cross_entropy(output, y_source)
                    else:  # Intermediate stages with reconstruction loss
                        # Simple MSE loss for feature preservation
                        loss = F.mse_loss(output, x_source[:, :output.size(1)])
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if epoch % 20 == 0:
                avg_loss = epoch_loss / num_batches
                logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    def predict(self, X):
        """Make predictions on new data."""
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            if self.method == 'transformer':
                logits, _, _ = self.model_(X_tensor, 0.0)  # No gradient reversal during inference
            elif self.method == 'contrastive':
                logits, _, _ = self.model_(X_tensor)
            elif self.method == 'progressive':
                logits = self.model_(X_tensor)
            
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            if self.method == 'transformer':
                logits, _, _ = self.model_(X_tensor, 0.0)
            elif self.method == 'contrastive':
                logits, _, _ = self.model_(X_tensor)
            elif self.method == 'progressive':
                logits = self.model_(X_tensor)
            
            probabilities = F.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()


class MetaTransferLearner(BaseEstimator, ClassifierMixin):
    """
    Meta-learning approach for transfer learning with few-shot adaptation.
    """
    
    def __init__(self, 
                 base_model_type: str = 'neural',
                 n_meta_epochs: int = 50,
                 n_adaptation_steps: int = 5,
                 meta_lr: float = 0.001,
                 adaptation_lr: float = 0.01,
                 hidden_dim: int = 128):
        self.base_model_type = base_model_type
        self.n_meta_epochs = n_meta_epochs
        self.n_adaptation_steps = n_adaptation_steps
        self.meta_lr = meta_lr
        self.adaptation_lr = adaptation_lr
        self.hidden_dim = hidden_dim
    
    def fit(self, X_source, y_source, X_target=None, y_target=None):
        """Train meta-learner for fast adaptation."""
        # Implementation of MAML-style meta-learning for transfer
        # This is a simplified version - full implementation would require
        # multiple source domains for proper meta-training
        
        logger.info("Training meta-transfer learner...")
        
        # For now, use standard neural network as base
        self.base_model_ = NeuralTransferLearningClassifier(
            method='transformer',
            hidden_dim=self.hidden_dim,
            num_epochs=self.n_meta_epochs
        )
        
        self.base_model_.fit(X_source, y_source, X_target, y_target)
        
        return self
    
    def adapt_to_target(self, X_target_support, y_target_support, X_target_query):
        """Fast adaptation to target domain with few examples."""
        # Simplified few-shot adaptation
        # In practice, this would use gradient-based meta-learning
        
        logger.info(f"Adapting to target domain with {len(X_target_support)} support examples")
        
        # Fine-tune on support set
        adapted_model = NeuralTransferLearningClassifier(
            method='transformer',
            hidden_dim=self.hidden_dim,
            num_epochs=self.n_adaptation_steps
        )
        
        adapted_model.fit(X_target_support, y_target_support)
        
        return adapted_model.predict(X_target_query)
    
    def predict(self, X):
        return self.base_model_.predict(X)
    
    def predict_proba(self, X):
        return self.base_model_.predict_proba(X)


def evaluate_neural_transfer_methods(X_source, y_source, X_target, y_target, 
                                   methods=['transformer', 'contrastive', 'progressive']):
    """
    Comprehensive evaluation of neural transfer learning methods.
    """
    results = {}
    
    for method in methods:
        logger.info(f"Evaluating {method} neural transfer learning...")
        
        try:
            # Initialize and train model
            model = NeuralTransferLearningClassifier(
                method=method,
                num_epochs=50,  # Reduced for quick evaluation
                hidden_dim=128
            )
            
            model.fit(X_source, y_source, X_target, y_target)
            
            # Evaluate
            y_pred = model.predict(X_target)
            y_prob = model.predict_proba(X_target)[:, 1]
            
            results[method] = {
                'accuracy': accuracy_score(y_target, y_pred),
                'f1': f1_score(y_target, y_pred),
                'auc': roc_auc_score(y_target, y_prob) if len(np.unique(y_target)) > 1 else 0.5
            }
            
            logger.info(f"{method} - Accuracy: {results[method]['accuracy']:.3f}, "
                       f"F1: {results[method]['f1']:.3f}, AUC: {results[method]['auc']:.3f}")
        
        except Exception as e:
            logger.error(f"Error evaluating {method}: {e}")
            results[method] = {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5}
    
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
    X_target = np.random.randn(n_target, n_features) + 0.5  # Domain shift
    y_target = (X_target.sum(axis=1) + np.random.randn(n_target) * 0.1 > 0).astype(int)
    
    # Evaluate methods
    results = evaluate_neural_transfer_methods(X_source, y_source, X_target, y_target)
    
    print("\nNeural Transfer Learning Results:")
    for method, metrics in results.items():
        print(f"{method:12} - Accuracy: {metrics['accuracy']:.3f}, "
              f"F1: {metrics['f1']:.3f}, AUC: {metrics['auc']:.3f}")