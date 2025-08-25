"""
Enhanced OULAD Deep Learning Integration

This module demonstrates how to integrate the enhanced feature engineering
with the existing OULAD deep learning pipeline to improve performance.
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.enhanced_feature_engineering import EnhancedFeatureEngineer

try:
    from src.oulad.advanced_deep_learning import train_advanced_deep_learning_models
except ImportError as e:
    logging.warning(f"Could not import OULAD modules: {e}")

logger = logging.getLogger(__name__)


def enhanced_oulad_feature_engineering(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, list]:
    """
    Apply enhanced feature engineering specifically optimized for OULAD data.
    
    Args:
        X: OULAD feature matrix
        y: Target labels
        
    Returns:
        Enhanced feature matrix and feature names
    """
    logger.info("Applying enhanced feature engineering for OULAD data...")
    
    # Use enhanced feature engineer with OULAD-specific optimizations
    engineer = EnhancedFeatureEngineer(dataset_type="oulad")
    
    # Apply comprehensive feature engineering
    X_enhanced = engineer.fit_transform(X, y)
    feature_names = engineer.get_feature_names()
    
    logger.info(f"OULAD feature engineering: {X.shape[1]} -> {X_enhanced.shape[1]} features")
    
    return X_enhanced, feature_names


def enhanced_transfer_feature_engineering(
    source_X: pd.DataFrame, 
    target_X: pd.DataFrame,
    source_y: pd.Series = None,
    target_y: pd.Series = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply enhanced feature engineering for transfer learning scenarios.
    
    Args:
        source_X: Source domain features (e.g., OULAD)
        target_X: Target domain features (e.g., UCI)
        source_y: Source domain labels
        target_y: Target domain labels
        
    Returns:
        Enhanced source and target feature matrices
    """
    logger.info("Applying enhanced feature engineering for transfer learning...")
    
    from src.enhanced_feature_engineering import create_domain_adaptive_features
    
    # Apply domain-adaptive feature engineering
    source_enhanced, target_enhanced = create_domain_adaptive_features(
        source_X, target_X, source_y, target_y
    )
    
    logger.info(f"Transfer learning feature engineering:")
    logger.info(f"  Source: {source_X.shape[1]} -> {source_enhanced.shape[1]} features")
    logger.info(f"  Target: {target_X.shape[1]} -> {target_enhanced.shape[1]} features")
    
    return source_enhanced, target_enhanced


def create_enhanced_deep_learning_model(input_size: int, num_classes: int = 2) -> nn.Module:
    """
    Create an enhanced deep learning model optimized for engineered features.
    
    Args:
        input_size: Number of input features
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    
    class EnhancedTabularNN(nn.Module):
        """Enhanced neural network for tabular data with engineered features."""
        
        def __init__(self, input_size: int, num_classes: int = 2):
            super().__init__()
            
            # Adaptive layer sizes based on input
            hidden1 = min(512, max(64, input_size * 2))
            hidden2 = min(256, max(32, input_size))
            hidden3 = min(128, max(16, input_size // 2))
            
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(hidden2, hidden3),
                nn.BatchNorm1d(hidden3),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(hidden3, num_classes)
            )
            
        def forward(self, x):
            return self.network(x)
    
    return EnhancedTabularNN(input_size, num_classes)


def train_enhanced_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001
) -> Dict:
    """
    Train an enhanced deep learning model on engineered features.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Training results dictionary
    """
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Create model
    model = create_enhanced_deep_learning_model(X_train.shape[1])
    model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    val_aucs = []
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
            
            val_acc = accuracy_score(y_val, val_preds)
            val_auc = roc_auc_score(y_val, val_probs)
            
            val_accuracies.append(val_acc)
            val_aucs.append(val_auc)
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    results = {
        'model': model,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_aucs': val_aucs,
        'best_val_auc': best_val_auc,
        'final_val_acc': val_accuracies[-1],
        'final_val_auc': val_aucs[-1]
    }
    
    return results


def compare_feature_engineering_impact(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Compare the impact of enhanced feature engineering on model performance.
    
    Args:
        X: Input features
        y: Target labels
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Comparison results
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    logger.info("Comparing feature engineering impact...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train_val, X_val, y_train_val, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    
    results = {}
    
    # 1. Baseline: Original features only
    logger.info("Training baseline model...")
    
    # Prepare baseline data
    X_train_baseline = X_train_val.copy()
    X_val_baseline = X_val.copy()
    X_test_baseline = X_test.copy()
    
    # Handle categorical variables for baseline
    for col in X_train_baseline.select_dtypes(include=['object', 'category']).columns:
        X_train_baseline[col] = pd.Categorical(X_train_baseline[col]).codes
        X_val_baseline[col] = pd.Categorical(X_val_baseline[col]).codes
        X_test_baseline[col] = pd.Categorical(X_test_baseline[col]).codes
    
    # Fill missing values
    X_train_baseline = X_train_baseline.fillna(0)
    X_val_baseline = X_val_baseline.fillna(0)
    X_test_baseline = X_test_baseline.fillna(0)
    
    # Scale features
    scaler_baseline = StandardScaler()
    X_train_baseline_scaled = scaler_baseline.fit_transform(X_train_baseline)
    X_val_baseline_scaled = scaler_baseline.transform(X_val_baseline)
    X_test_baseline_scaled = scaler_baseline.transform(X_test_baseline)
    
    # Train baseline model
    baseline_results = train_enhanced_model(
        X_train_baseline_scaled, y_train_val.values,
        X_val_baseline_scaled, y_val.values,
        epochs=30
    )
    
    # Test baseline model
    baseline_model = baseline_results['model']
    baseline_model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test_baseline_scaled)
        test_outputs = baseline_model(test_tensor)
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].numpy()
        test_preds = torch.argmax(test_outputs, dim=1).numpy()
    
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    results['baseline'] = {
        'train_history': baseline_results,
        'test_accuracy': accuracy_score(y_test, test_preds),
        'test_auc': roc_auc_score(y_test, test_probs),
        'test_f1': f1_score(y_test, test_preds),
        'n_features': X_train_baseline_scaled.shape[1]
    }
    
    # 2. Enhanced: With feature engineering
    logger.info("Training enhanced model...")
    
    # Apply enhanced feature engineering
    engineer = EnhancedFeatureEngineer(dataset_type="auto")
    X_train_enhanced = engineer.fit_transform(X_train_val, y_train_val)
    X_val_enhanced = engineer.transform(X_val)
    X_test_enhanced = engineer.transform(X_test)
    
    # Train enhanced model
    enhanced_results = train_enhanced_model(
        X_train_enhanced, y_train_val.values,
        X_val_enhanced, y_val.values,
        epochs=30
    )
    
    # Test enhanced model
    enhanced_model = enhanced_results['model']
    enhanced_model.eval()
    with torch.no_grad():
        test_tensor_enhanced = torch.FloatTensor(X_test_enhanced)
        test_outputs_enhanced = enhanced_model(test_tensor_enhanced)
        test_probs_enhanced = torch.softmax(test_outputs_enhanced, dim=1)[:, 1].numpy()
        test_preds_enhanced = torch.argmax(test_outputs_enhanced, dim=1).numpy()
    
    results['enhanced'] = {
        'train_history': enhanced_results,
        'test_accuracy': accuracy_score(y_test, test_preds_enhanced),
        'test_auc': roc_auc_score(y_test, test_probs_enhanced),
        'test_f1': f1_score(y_test, test_preds_enhanced),
        'n_features': X_train_enhanced.shape[1]
    }
    
    # Calculate improvements
    results['improvement'] = {
        'accuracy': results['enhanced']['test_accuracy'] - results['baseline']['test_accuracy'],
        'auc': results['enhanced']['test_auc'] - results['baseline']['test_auc'],
        'f1': results['enhanced']['test_f1'] - results['baseline']['test_f1'],
        'feature_ratio': results['enhanced']['n_features'] / results['baseline']['n_features']
    }
    
    return results


def run_enhanced_oulad_demo():
    """
    Run a demonstration of enhanced OULAD feature engineering.
    """
    from sklearn.datasets import make_classification
    
    print("="*80)
    print("ENHANCED OULAD FEATURE ENGINEERING DEMONSTRATION")
    print("="*80)
    
    # Create OULAD-like synthetic data
    print("\n1. Creating OULAD-like synthetic data...")
    
    # Generate base features
    np.random.seed(42)
    n_samples = 1000
    
    # VLE engagement features
    vle_features = {
        'vle_total_clicks': np.random.poisson(150, n_samples),
        'vle_days_active': np.random.randint(5, 35, n_samples),
        'vle_first4_clicks': np.random.poisson(30, n_samples),
        'vle_last4_clicks': np.random.poisson(40, n_samples),
        'vle_mean_clicks': np.random.gamma(2, 10, n_samples),
        'vle_max_clicks': np.random.gamma(3, 15, n_samples)
    }
    
    # Assessment features
    assessment_features = {
        'assessment_mean_score': np.random.normal(65, 20, n_samples),
        'assessment_count': np.random.randint(3, 12, n_samples),
        'assessment_last_score': np.random.normal(68, 25, n_samples),
        'assessment_ontime_rate': np.random.beta(2, 1, n_samples)
    }
    
    # Demographic features
    demo_features = {
        'age_band': np.random.choice(['0-35', '35-55', '55<='], n_samples),
        'highest_education': np.random.choice(['HE Qualification', 'A Level or Equivalent', 'Lower Than A Level'], n_samples),
        'imd_band': np.random.choice(['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                                     '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'], n_samples)
    }
    
    # Combine all features
    all_features = {**vle_features, **assessment_features, **demo_features}
    X_oulad = pd.DataFrame(all_features)
    
    # Create realistic target based on engagement and performance
    engagement_score = (
        (X_oulad['vle_total_clicks'] > 100).astype(int) +
        (X_oulad['vle_days_active'] > 20).astype(int) +
        (X_oulad['assessment_mean_score'] > 60).astype(int) +
        (X_oulad['assessment_ontime_rate'] > 0.7).astype(int)
    )
    
    # Add some noise and make it binary
    y_oulad = (engagement_score >= 2).astype(int)
    noise = np.random.binomial(1, 0.1, n_samples)  # 10% noise
    y_oulad = np.abs(y_oulad - noise)  # Flip 10% of labels
    
    y_oulad = pd.Series(y_oulad)
    
    print(f"   Data shape: {X_oulad.shape}")
    print(f"   Target distribution: {y_oulad.value_counts().to_dict()}")
    print(f"   VLE features: {len(vle_features)}")
    print(f"   Assessment features: {len(assessment_features)}")
    print(f"   Demographic features: {len(demo_features)}")
    
    # Apply enhanced feature engineering
    print("\n2. Applying enhanced OULAD feature engineering...")
    X_enhanced, feature_names = enhanced_oulad_feature_engineering(X_oulad, y_oulad)
    
    print(f"   Original features: {X_oulad.shape[1]}")
    print(f"   Enhanced features: {X_enhanced.shape[1]}")
    print(f"   Feature increase: {((X_enhanced.shape[1] / X_oulad.shape[1]) - 1) * 100:.1f}%")
    
    # Compare model performance
    print("\n3. Comparing model performance...")
    try:
        comparison_results = compare_feature_engineering_impact(X_oulad, y_oulad)
        
        baseline = comparison_results['baseline']
        enhanced = comparison_results['enhanced']
        improvement = comparison_results['improvement']
        
        print(f"\n   Baseline Model ({baseline['n_features']} features):")
        print(f"     Test Accuracy: {baseline['test_accuracy']:.4f}")
        print(f"     Test ROC-AUC: {baseline['test_auc']:.4f}")
        print(f"     Test F1-Score: {baseline['test_f1']:.4f}")
        
        print(f"\n   Enhanced Model ({enhanced['n_features']} features):")
        print(f"     Test Accuracy: {enhanced['test_accuracy']:.4f}")
        print(f"     Test ROC-AUC: {enhanced['test_auc']:.4f}")
        print(f"     Test F1-Score: {enhanced['test_f1']:.4f}")
        
        print(f"\n   Improvements:")
        print(f"     Accuracy: {improvement['accuracy']:+.4f}")
        print(f"     ROC-AUC: {improvement['auc']:+.4f}")
        print(f"     F1-Score: {improvement['f1']:+.4f}")
        print(f"     Feature Ratio: {improvement['feature_ratio']:.2f}x")
        
    except Exception as e:
        print(f"   Could not perform comparison: {e}")
    
    print("\n" + "="*80)
    print("ENHANCED OULAD DEMONSTRATION COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Run demonstration
    run_enhanced_oulad_demo()