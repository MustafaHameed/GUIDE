"""
Advanced Training Techniques for OULAD Deep Learning Models

This module implements state-of-the-art training techniques:
- Mixup and CutMix for tabular data
- Self-supervised pre-training  
- Contrastive learning
- Knowledge distillation
- Advanced regularization techniques
- Meta-learning approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import logging
from typing import Dict, Tuple, Optional, List, Union, Callable
import math
import random
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TabularMixup:
    """
    Mixup augmentation for tabular data.
    Based on "mixup: Beyond Empirical Risk Minimization"
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Initialize Mixup.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying mixup
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to batch.
        
        Args:
            x: Input features (batch_size, features)
            y: Target labels (batch_size,)
            
        Returns:
            Mixed features, original labels, mixed labels, mixing lambda
        """
        if random.random() > self.prob:
            return x, y, y, 1.0
        
        batch_size = x.size(0)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Generate random permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Mix features
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Return mixed features and both sets of labels
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class TabularCutMix:
    """
    CutMix-inspired technique for tabular data.
    Randomly masks features instead of spatial regions.
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying cutmix
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to batch.
        
        Args:
            x: Input features (batch_size, features)
            y: Target labels (batch_size,)
            
        Returns:
            Mixed features, original labels, mixed labels, mixing lambda
        """
        if random.random() > self.prob:
            return x, y, y, 1.0
        
        batch_size, num_features = x.size()
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Calculate number of features to replace
        cut_ratio = np.sqrt(1 - lam)
        cut_features = int(num_features * cut_ratio)
        
        # Generate random permutation
        index = torch.randperm(batch_size, device=x.device)
        
        # Create mixed features
        mixed_x = x.clone()
        
        # Randomly select features to replace
        for i in range(batch_size):
            feature_indices = torch.randperm(num_features)[:cut_features]
            mixed_x[i, feature_indices] = x[index[i], feature_indices]
        
        # Adjust lambda based on actual feature ratio
        lam = 1 - (cut_features / num_features)
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor,
                   y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Compute mixup loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing lambda
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class SelfSupervisedPretrainer:
    """
    Self-supervised pre-training for tabular data using masking and reconstruction.
    """
    
    def __init__(self, model: nn.Module, mask_ratio: float = 0.15, 
                 reconstruction_weight: float = 1.0):
        """
        Initialize self-supervised pre-trainer.
        
        Args:
            model: Model to pre-train
            mask_ratio: Ratio of features to mask
            reconstruction_weight: Weight for reconstruction loss
        """
        self.model = model
        self.mask_ratio = mask_ratio
        self.reconstruction_weight = reconstruction_weight
        
        # Add reconstruction head
        if hasattr(model, 'fc') or hasattr(model, 'final_mapping'):
            # Get the dimension before final layer
            if hasattr(model, 'fc'):
                hidden_dim = model.fc.in_features
            else:
                hidden_dim = model.final_mapping.in_features
        else:
            hidden_dim = 128  # Default
        
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model.input_dim if hasattr(model, 'input_dim') else 128)
        )
    
    def mask_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask features for reconstruction.
        
        Args:
            x: Input features
            
        Returns:
            Masked features, mask tensor
        """
        batch_size, num_features = x.size()
        num_masked = int(num_features * self.mask_ratio)
        
        # Create mask
        mask = torch.zeros_like(x)
        masked_x = x.clone()
        
        for i in range(batch_size):
            # Randomly select features to mask
            masked_indices = torch.randperm(num_features)[:num_masked]
            mask[i, masked_indices] = 1
            
            # Replace with random noise or zeros
            masked_x[i, masked_indices] = torch.randn_like(masked_x[i, masked_indices]) * 0.1
        
        return masked_x, mask
    
    def pretrain(self, data_loader: DataLoader, epochs: int = 50,
                lr: float = 1e-3, device: torch.device = None) -> Dict:
        """
        Pre-train model using self-supervised learning.
        
        Args:
            data_loader: Data loader with unlabeled data
            epochs: Number of pre-training epochs
            lr: Learning rate
            device: Device to use
            
        Returns:
            Pre-training results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(device)
        self.reconstruction_head.to(device)
        
        # Optimizer for both model and reconstruction head
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.reconstruction_head.parameters()),
            lr=lr
        )
        
        reconstruction_criterion = nn.MSELoss()
        
        results = {'reconstruction_losses': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                else:
                    x = batch_data
                
                x = x.to(device)
                
                # Mask features
                masked_x, mask = self.mask_features(x)
                
                # Forward pass through model
                if hasattr(self.model, 'forward_features'):
                    features = self.model.forward_features(masked_x)
                else:
                    # Try to get intermediate features
                    features = self.model(masked_x)
                    if len(features.shape) > 2:
                        features = features.mean(dim=1)  # Global average pooling
                
                # Reconstruct original features
                reconstructed = self.reconstruction_head(features)
                
                # Compute reconstruction loss only on masked features
                reconstruction_loss = reconstruction_criterion(
                    reconstructed * mask, x * mask
                )
                
                optimizer.zero_grad()
                reconstruction_loss.backward()
                optimizer.step()
                
                epoch_loss += reconstruction_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            results['reconstruction_losses'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: Reconstruction Loss: {avg_loss:.4f}")
        
        logger.info("Self-supervised pre-training completed")
        return results


class ContrastiveLearner:
    """
    Contrastive learning for tabular data using feature augmentation.
    """
    
    def __init__(self, model: nn.Module, projection_dim: int = 128,
                 temperature: float = 0.1):
        """
        Initialize contrastive learner.
        
        Args:
            model: Base model
            projection_dim: Dimension of projection head
            temperature: Temperature for contrastive loss
        """
        self.model = model
        self.temperature = temperature
        
        # Add projection head
        if hasattr(model, 'fc') or hasattr(model, 'final_mapping'):
            if hasattr(model, 'fc'):
                hidden_dim = model.fc.in_features
            else:
                hidden_dim = model.final_mapping.in_features
        else:
            hidden_dim = 128
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def augment_features(self, x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        """
        Augment features with noise.
        
        Args:
            x: Input features
            noise_std: Standard deviation of noise
            
        Returns:
            Augmented features
        """
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss (InfoNCE).
        
        Args:
            z1: First set of projections
            z2: Second set of projections
            
        Returns:
            Contrastive loss
        """
        batch_size = z1.size(0)
        
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size, device=z1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def train_contrastive(self, data_loader: DataLoader, epochs: int = 50,
                         lr: float = 1e-3, device: torch.device = None) -> Dict:
        """
        Train model using contrastive learning.
        
        Args:
            data_loader: Data loader
            epochs: Number of epochs
            lr: Learning rate
            device: Device to use
            
        Returns:
            Training results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(device)
        self.projection_head.to(device)
        
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.projection_head.parameters()),
            lr=lr
        )
        
        results = {'contrastive_losses': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                else:
                    x = batch_data
                
                x = x.to(device)
                
                # Create two augmented views
                x1 = self.augment_features(x)
                x2 = self.augment_features(x)
                
                # Forward pass through model
                if hasattr(self.model, 'forward_features'):
                    h1 = self.model.forward_features(x1)
                    h2 = self.model.forward_features(x2)
                else:
                    h1 = self.model(x1)
                    h2 = self.model(x2)
                    
                    if len(h1.shape) > 2:
                        h1 = h1.mean(dim=1)
                        h2 = h2.mean(dim=1)
                
                # Project to contrastive space
                z1 = self.projection_head(h1)
                z2 = self.projection_head(h2)
                
                # Compute contrastive loss
                loss = self.contrastive_loss(z1, z2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            results['contrastive_losses'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: Contrastive Loss: {avg_loss:.4f}")
        
        logger.info("Contrastive pre-training completed")
        return results


class KnowledgeDistiller:
    """
    Knowledge distillation for tabular data models.
    """
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 temperature: float = 4.0, alpha: float = 0.5):
        """
        Initialize knowledge distiller.
        
        Args:
            teacher_model: Larger, pre-trained teacher model
            student_model: Smaller student model to train
            temperature: Temperature for distillation
            alpha: Weight for distillation loss vs hard target loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         targets: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            targets: Ground truth targets
            
        Returns:
            Combined distillation loss
        """
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_loss *= (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, kl_loss, hard_loss
    
    def train_student(self, train_loader: DataLoader, val_loader: DataLoader,
                     epochs: int = 100, lr: float = 1e-3,
                     device: torch.device = None) -> Dict:
        """
        Train student model using knowledge distillation.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            device: Device to use
            
        Returns:
            Training results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.teacher_model.to(device)
        self.student_model.to(device)
        
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        results = {
            'total_losses': [],
            'kl_losses': [],
            'hard_losses': [],
            'val_accuracies': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.student_model.train()
            epoch_total_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_hard_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = self.teacher_model(batch_x)
                
                # Get student predictions
                student_logits = self.student_model(batch_x)
                
                # Compute distillation loss
                total_loss, kl_loss, hard_loss = self.distillation_loss(
                    student_logits, teacher_logits, batch_y
                )
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_total_loss += total_loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_hard_loss += hard_loss.item()
                num_batches += 1
            
            scheduler.step()
            
            # Validation phase
            self.student_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    outputs = self.student_model(batch_x)
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total
            
            # Record results
            results['total_losses'].append(epoch_total_loss / num_batches)
            results['kl_losses'].append(epoch_kl_loss / num_batches)
            results['hard_losses'].append(epoch_hard_loss / num_batches)
            results['val_accuracies'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                           f"Total Loss: {results['total_losses'][-1]:.4f}, "
                           f"Val Acc: {val_acc:.4f}")
        
        logger.info(f"Knowledge distillation completed. Best val accuracy: {best_val_acc:.4f}")
        results['best_val_accuracy'] = best_val_acc
        
        return results


class AdvancedTrainer:
    """
    Advanced trainer combining multiple training techniques.
    """
    
    def __init__(self, model: nn.Module, use_mixup: bool = True, use_cutmix: bool = False,
                 use_label_smoothing: bool = True, label_smoothing: float = 0.1,
                 use_gradient_clipping: bool = True, max_grad_norm: float = 1.0):
        """
        Initialize advanced trainer.
        
        Args:
            model: Model to train
            use_mixup: Whether to use mixup augmentation
            use_cutmix: Whether to use cutmix augmentation
            use_label_smoothing: Whether to use label smoothing
            label_smoothing: Label smoothing parameter
            use_gradient_clipping: Whether to use gradient clipping
            max_grad_norm: Maximum gradient norm
        """
        self.model = model
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing = label_smoothing
        self.use_gradient_clipping = use_gradient_clipping
        self.max_grad_norm = max_grad_norm
        
        # Initialize augmentation techniques
        if use_mixup:
            self.mixup = TabularMixup(alpha=1.0, prob=0.5)
        if use_cutmix:
            self.cutmix = TabularCutMix(alpha=1.0, prob=0.5)
        
        # Loss function
        if use_label_smoothing:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   device: torch.device) -> Dict:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Apply augmentation
            if self.use_mixup and random.random() < 0.5:
                mixed_x, y_a, y_b, lam = self.mixup(batch_x, batch_y)
                
                optimizer.zero_grad()
                outputs = self.model(mixed_x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                
            elif self.use_cutmix and random.random() < 0.5:
                mixed_x, y_a, y_b, lam = self.cutmix(batch_x, batch_y)
                
                optimizer.zero_grad()
                outputs = self.model(mixed_x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                
            else:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            
            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy (approximate for mixed samples)
            _, predicted = torch.max(outputs, 1)
            if self.use_mixup or self.use_cutmix:
                # For mixed samples, use original labels for simplicity
                epoch_acc += (predicted == batch_y).float().mean().item()
            else:
                epoch_acc += (predicted == batch_y).float().mean().item()
            
            num_batches += 1
        
        return {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_acc / num_batches
        }
    
    def validate(self, val_loader: DataLoader, device: torch.device) -> Dict:
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        val_probs = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = self.model(batch_x)
                loss = F.cross_entropy(outputs, batch_y)  # Use standard loss for validation
                
                probs = F.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs, 1)
                
                val_loss += loss.item()
                val_probs.extend(probs.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_f1 = f1_score(val_labels, val_preds)
        
        return {
            'loss': val_loss / len(val_loader),
            'accuracy': val_acc,
            'auc': val_auc,
            'f1': val_f1,
            'probs': val_probs,
            'preds': val_preds
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4,
             device: torch.device = None, patience: int = 20) -> Dict:
        """
        Train model with advanced techniques.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            device: Device to use
            patience: Early stopping patience
            
        Returns:
            Training results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        results = {
            'train_losses': [],
            'train_accuracies': [],
            'val_losses': [],
            'val_accuracies': [],
            'val_aucs': [],
            'val_f1s': []
        }
        
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            train_results = self.train_epoch(train_loader, optimizer, device)
            
            # Validation
            val_results = self.validate(val_loader, device)
            
            scheduler.step()
            
            # Record results
            results['train_losses'].append(train_results['loss'])
            results['train_accuracies'].append(train_results['accuracy'])
            results['val_losses'].append(val_results['loss'])
            results['val_accuracies'].append(val_results['accuracy'])
            results['val_aucs'].append(val_results['auc'])
            results['val_f1s'].append(val_results['f1'])
            
            # Early stopping
            if val_results['auc'] > best_val_auc:
                best_val_auc = val_results['auc']
                patience_counter = 0
                best_model_state = {name: param.clone() for name, param in self.model.named_parameters()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: "
                           f"Train Loss: {train_results['loss']:.4f}, "
                           f"Val AUC: {val_results['auc']:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            for name, param in self.model.named_parameters():
                param.data.copy_(best_model_state[name])
        
        results['best_val_auc'] = best_val_auc
        results['final_val_results'] = self.validate(val_loader, device)
        
        logger.info(f"Training completed. Best validation AUC: {best_val_auc:.4f}")
        
        return results


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross-entropy loss.
    """
    
    def __init__(self, smoothing: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross-entropy loss.
        
        Args:
            pred: Predictions (batch_size, num_classes)
            target: Targets (batch_size,)
            
        Returns:
            Smoothed cross-entropy loss
        """
        num_classes = pred.size(1)
        
        # Convert targets to one-hot
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        target_smooth = (1 - self.smoothing) * target_one_hot + \
                       self.smoothing / num_classes
        
        # Compute cross-entropy with smoothed targets
        log_pred = F.log_softmax(pred, dim=1)
        loss = -torch.sum(target_smooth * log_pred, dim=1).mean()
        
        return loss


def train_with_advanced_techniques(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray,
                                 batch_size: int = 128, epochs: int = 100,
                                 lr: float = 1e-3, use_pretraining: bool = True,
                                 use_mixup: bool = True, device: torch.device = None) -> Dict:
    """
    Train model with advanced techniques.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        use_pretraining: Whether to use self-supervised pre-training
        use_mixup: Whether to use mixup augmentation
        device: Device to use
        
    Returns:
        Training results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    results = {}
    
    # Self-supervised pre-training
    if use_pretraining:
        logger.info("Starting self-supervised pre-training...")
        pretrainer = SelfSupervisedPretrainer(model)
        
        # Create unlabeled data loader (without labels)
        unlabeled_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
        
        pretraining_results = pretrainer.pretrain(
            unlabeled_loader, epochs=20, lr=lr, device=device
        )
        results['pretraining'] = pretraining_results
    
    # Supervised training with advanced techniques
    logger.info("Starting supervised training with advanced techniques...")
    trainer = AdvancedTrainer(
        model, 
        use_mixup=use_mixup,
        use_label_smoothing=True,
        use_gradient_clipping=True
    )
    
    training_results = trainer.train(
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        device=device
    )
    
    results['training'] = training_results
    results['scaler'] = scaler
    
    return results