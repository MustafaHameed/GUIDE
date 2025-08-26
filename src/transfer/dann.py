"""
Domain Adversarial Neural Networks (DANN) for Transfer Learning

Implements DANN architecture with gradient reversal layer for domain adaptation.
Trains a feature extractor that produces domain-invariant representations.

Reference: "Domain-Adversarial Training of Neural Networks" by Ganin et al.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import warnings

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide fallback implementations

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer for domain adversarial training.
    
    Forward pass: identity function
    Backward pass: multiply gradients by -lambda
    """
    
    @staticmethod
    def forward(ctx, x, lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_p
        return output, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer module."""
    
    def __init__(self, lambda_p: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_p = lambda_p
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_p)
    
    def set_lambda(self, lambda_p: float):
        self.lambda_p = lambda_p


class DANNModel(nn.Module):
    """
    Domain Adversarial Neural Network architecture.
    
    Consists of:
    - Feature extractor (shared)
    - Label classifier head
    - Domain classifier head with gradient reversal
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 num_classes: int = 2, dropout: float = 0.2):
        """
        Initialize DANN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions for feature extractor
            num_classes: Number of target classes
            dropout: Dropout probability
        """
        super(DANNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Feature extractor
        feature_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # Domain classifier with gradient reversal
        self.gradient_reversal = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, 2)  # Binary domain classification
        )
        
    def forward(self, x, lambda_p: float = 1.0):
        """
        Forward pass through DANN model.
        
        Args:
            x: Input features
            lambda_p: Lambda parameter for gradient reversal
            
        Returns:
            Tuple of (label_logits, domain_logits)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Label prediction
        label_logits = self.label_classifier(features)
        
        # Domain prediction with gradient reversal
        reversed_features = self.gradient_reversal(features)
        domain_logits = self.domain_classifier(reversed_features)
        
        # Update lambda for gradient reversal
        self.gradient_reversal.set_lambda(lambda_p)
        
        return label_logits, domain_logits


class DANNClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible DANN classifier.
    """
    
    def __init__(self, hidden_dims: List[int] = [128, 64], num_epochs: int = 100,
                 batch_size: int = 32, learning_rate: float = 0.001,
                 lambda_schedule: str = 'grl', lambda_max: float = 1.0,
                 device: str = 'auto', random_state: int = 42):
        """
        Initialize DANN classifier.
        
        Args:
            hidden_dims: Hidden layer dimensions
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            lambda_schedule: Lambda scheduling ('grl', 'constant', 'linear')
            lambda_max: Maximum lambda value
            device: Device for training ('auto', 'cpu', 'cuda')
            random_state: Random seed
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DANN but not available")
        
        self.hidden_dims = hidden_dims
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_schedule = lambda_schedule
        self.lambda_max = lambda_max
        self.device = device
        self.random_state = random_state
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model components
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.classes_ = None
        self.is_fitted_ = False
        
    def fit(self, X_source: np.ndarray, y_source: np.ndarray,
            X_target: np.ndarray) -> 'DANNClassifier':
        """
        Fit DANN model with source and target data.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features (unlabeled)
            
        Returns:
            self
        """
        logger.info("Training DANN model...")
        
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Prepare data
        X_source = np.asarray(X_source)
        y_source = np.asarray(y_source)
        X_target = np.asarray(X_target)
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_source_scaled = self.scaler_.fit_transform(X_source)
        X_target_scaled = self.scaler_.transform(X_target)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_source_encoded = self.label_encoder_.fit_transform(y_source)
        self.classes_ = self.label_encoder_.classes_
        
        # Combine data for training
        X_combined = np.vstack([X_source_scaled, X_target_scaled])
        
        # Create domain labels (0 = source, 1 = target)
        domain_labels = np.concatenate([
            np.zeros(len(X_source_scaled)),
            np.ones(len(X_target_scaled))
        ])
        
        # Create labels for combined data (use -1 for unlabeled target)
        label_labels = np.concatenate([
            y_source_encoded,
            np.full(len(X_target_scaled), -1)
        ])
        
        # Initialize model
        input_dim = X_combined.shape[1]
        num_classes = len(self.classes_)
        
        self.model_ = DANNModel(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            num_classes=num_classes
        ).to(self.device)
        
        # Train model
        self._train_dann(X_combined, label_labels, domain_labels)
        
        self.is_fitted_ = True
        return self
    
    def _train_dann(self, X: np.ndarray, label_labels: np.ndarray, 
                   domain_labels: np.ndarray):
        """Train the DANN model."""
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        label_tensor = torch.LongTensor(label_labels).to(self.device)
        domain_tensor = torch.LongTensor(domain_labels).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, label_tensor, domain_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizers
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Loss functions
        label_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore unlabeled
        domain_criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model_.train()
        
        for epoch in range(self.num_epochs):
            epoch_label_loss = 0
            epoch_domain_loss = 0
            
            # Calculate lambda for this epoch
            lambda_p = self._calculate_lambda(epoch)
            
            for batch_x, batch_label, batch_domain in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                label_logits, domain_logits = self.model_(batch_x, lambda_p)
                
                # Calculate losses
                label_loss = label_criterion(label_logits, batch_label)
                domain_loss = domain_criterion(domain_logits, batch_domain)
                
                # Total loss
                total_loss = label_loss + domain_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_label_loss += label_loss.item()
                epoch_domain_loss += domain_loss.item()
            
            # Log progress
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs}: "
                          f"Label Loss: {epoch_label_loss:.4f}, "
                          f"Domain Loss: {epoch_domain_loss:.4f}, "
                          f"Lambda: {lambda_p:.4f}")
    
    def _calculate_lambda(self, epoch: int) -> float:
        """Calculate lambda parameter for gradient reversal."""
        progress = epoch / self.num_epochs
        
        if self.lambda_schedule == 'constant':
            return self.lambda_max
        elif self.lambda_schedule == 'linear':
            return self.lambda_max * progress
        elif self.lambda_schedule == 'grl':
            # Original GRL schedule: 2/(1+exp(-10*p)) - 1
            return self.lambda_max * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)
        else:
            return self.lambda_max
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            label_logits, _ = self.model_(X_tensor)
            probs = F.softmax(label_logits, dim=1).cpu().numpy()
        
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        predicted_indices = np.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(predicted_indices)


# Fallback implementation when PyTorch is not available
class DANNClassifierFallback(BaseEstimator, ClassifierMixin):
    """
    Fallback DANN implementation when PyTorch is not available.
    Uses a simple ensemble approach as approximation.
    """
    
    def __init__(self, **kwargs):
        logger.warning("PyTorch not available, using fallback DANN implementation")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        self.base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_fitted_ = False
        
    def fit(self, X_source: np.ndarray, y_source: np.ndarray, 
            X_target: np.ndarray) -> 'DANNClassifierFallback':
        """Fit using source data only (fallback)."""
        self.base_clf.fit(X_source, y_source)
        self.classes_ = self.base_clf.classes_
        self.is_fitted_ = True
        return self
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using base classifier."""
        return self.base_clf.predict_proba(X)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using base classifier."""
        return self.base_clf.predict(X)


def create_dann_classifier(**kwargs):
    """
    Factory function to create DANN classifier with fallback.
    
    Returns:
        DANN classifier (PyTorch version if available, fallback otherwise)
    """
    if TORCH_AVAILABLE:
        return DANNClassifier(**kwargs)
    else:
        return DANNClassifierFallback(**kwargs)


def demo_dann():
    """
    Demonstrate DANN with synthetic data.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available, cannot run DANN demo")
        return
    
    # Generate synthetic data with domain shift
    np.random.seed(42)
    
    # Source domain
    X_source = np.random.normal(0, 1, (1000, 10))
    y_source = (X_source[:, 0] + X_source[:, 1] > 0).astype(int)
    
    # Target domain (shifted)
    X_target = np.random.normal(0.5, 1.2, (500, 10))
    y_target = (X_target[:, 0] + X_target[:, 1] > 0.5).astype(int)
    
    print("DANN Demo")
    print("=" * 15)
    
    # Train DANN
    dann = DANNClassifier(num_epochs=50, batch_size=64)
    dann.fit(X_source, y_source, X_target)
    
    # Evaluate on target
    y_pred = dann.predict(X_target)
    y_prob = dann.predict_proba(X_target)[:, 1]
    
    accuracy = accuracy_score(y_target, y_pred)
    auc = roc_auc_score(y_target, y_prob)
    
    print(f"Target domain accuracy: {accuracy:.3f}")
    print(f"Target domain AUC: {auc:.3f}")
    
    # Compare with baseline (train on source only)
    from sklearn.ensemble import RandomForestClassifier
    baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline.fit(X_source, y_source)
    
    y_pred_baseline = baseline.predict(X_target)
    accuracy_baseline = accuracy_score(y_target, y_pred_baseline)
    
    print(f"Baseline accuracy: {accuracy_baseline:.3f}")
    print(f"DANN improvement: {accuracy - accuracy_baseline:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_dann()