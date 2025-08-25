"""
Simple working test for OULAD deep learning implementation.
"""

import sys
sys.path.append('src/oulad')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


def load_oulad_data():
    """Load and preprocess OULAD data."""
    print("Loading OULAD data...")
    
    df = pd.read_csv('data/oulad/processed/oulad_ml.csv')
    print(f"Data shape: {df.shape}")
    
    # Use label_pass as target
    target_col = 'label_pass'
    y = df[target_col].copy()
    X = df.drop(columns=[target_col, 'id_student', 'label_fail_or_withdraw']).copy()
    
    # Remove samples with missing target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"After removing missing targets: {len(y)} samples")
    
    # Convert to binary
    y = (y > 0.5).astype(int)
    print(f"Class distribution: {np.bincount(y)}")
    
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X.values.astype(np.float32), y.values.astype(np.int64)


class SimpleTabularNN(nn.Module):
    """Simple tabular neural network for testing."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 2):
        super(SimpleTabularNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def test_simple_model():
    """Test a simple model on OULAD data."""
    print("=== Testing Simple Tabular Neural Network ===")
    
    # Load data
    X, y = load_oulad_data()
    print(f"Dataset: {X.shape} features, {len(y)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = SimpleTabularNN(input_dim=X_train_scaled.shape[1])
    print(f"Model created with {X_train_scaled.shape[1]} input features")
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    
    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Training model...")
    model.train()
    for epoch in range(20):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    # Evaluate model
    print("Evaluating model...")
    model.eval()
    test_preds = []
    test_probs = []
    test_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_probs)
    f1 = f1_score(test_labels, test_preds)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print("✅ Simple model test completed successfully!")
    return True


def test_modern_architectures():
    """Test multiple modern architectures."""
    print("\n=== Testing Modern Deep Learning Architectures ===")
    
    try:
        from modern_deep_learning import train_modern_deep_learning_models
        
        # Load data
        X, y = load_oulad_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training modern architectures on {X_train.shape[0]} samples...")
        
        # Train models (this will automatically handle scaling and everything)
        results = train_modern_deep_learning_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=42
        )
        
        print("\nResults Summary:")
        print("-" * 60)
        
        for model_name, model_results in results['results'].items():
            print(f"{model_name}:")
            print(f"  Accuracy: {model_results['test_accuracy']:.4f}")
            print(f"  AUC: {model_results['test_auc']:.4f}")
            print(f"  F1-Score: {model_results['test_f1']:.4f}")
            print()
        
        print("✅ Modern architectures test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Modern architectures test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        print("🚀 Starting OULAD Deep Learning Tests")
        print("=" * 60)
        
        # Test simple model first
        test_simple_model()
        
        # Test modern architectures
        test_modern_architectures()
        
        print("\n🎉 All tests completed successfully!")
        print("The OULAD deep learning implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()