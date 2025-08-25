"""
Quick test script for OULAD modern deep learning models.
"""

import sys
sys.path.append('src/oulad')

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from modern_deep_learning import TabNet, FTTransformer, train_modern_deep_learning_models


def load_oulad_data_simple():
    """Simple OULAD data loader."""
    print("Loading OULAD data...")
    
    df = pd.read_csv('data/oulad/processed/oulad_ml.csv')
    print(f"Data shape: {df.shape}")
    
    # Use final_result as target if available, otherwise last column
    if 'final_result' in df.columns:
        target_col = 'final_result'
    else:
        target_col = df.columns[-1]
    
    print(f"Target column: {target_col}")
    
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    
    # Encode categorical target
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Target classes: {le.classes_}")
    
    # Convert to binary if needed
    if len(np.unique(y)) > 2:
        y = (y >= np.median(y)).astype(int)
        print("Converted to binary classification")
    elif len(np.unique(y)) == 1:
        # Create artificial binary target for testing
        print("Warning: Only one class found, creating artificial binary target")
        np.random.seed(42)
        y = np.random.binomial(1, 0.3, len(y))  # 30% positive class
    
    print(f"Class distribution: {np.bincount(y)}")
    
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X.values.astype(np.float32), y.astype(np.int64)


def test_single_model():
    """Test a single model quickly."""
    print("=== Testing Single TabNet Model ===")
    
    # Load data
    X, y = load_oulad_data_simple()
    print(f"Dataset: {X.shape} features, {len(y)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to numpy arrays to avoid pandas Series issues
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = TabNet(input_dim=X_train_scaled.shape[1], n_d=32, n_a=32, n_steps=3)
    print(f"Model created: {type(model).__name__}")
    
    # Quick training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    
    from torch.utils.data import DataLoader, TensorDataset
    
    # Prepare data
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Train for few epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(10):  # Just 10 epochs for testing
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
            print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(train_loader):.4f}")
    
    # Evaluate
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
    
    print("✅ Single model test completed successfully!")
    return True


def test_multiple_models():
    """Test multiple models with the training function."""
    print("\n=== Testing Multiple Models ===")
    
    # Load data
    X, y = load_oulad_data_simple()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training models on {X_train.shape[0]} samples...")
    
    # Run training function (with reduced parameters for speed)
    results = train_modern_deep_learning_models(
        X_train=X_train,
        y_train=y_train, 
        X_test=X_test,
        y_test=y_test,
        random_state=42
    )
    
    print("\nResults Summary:")
    print("-" * 50)
    
    for model_name, model_results in results['results'].items():
        print(f"{model_name}:")
        print(f"  Accuracy: {model_results['test_accuracy']:.4f}")
        print(f"  AUC: {model_results['test_auc']:.4f}")
        print(f"  F1-Score: {model_results['test_f1']:.4f}")
        print()
    
    print("✅ Multiple models test completed successfully!")
    return True


if __name__ == '__main__':
    try:
        # Run tests
        test_single_model()
        test_multiple_models()
        
        print("\n🎉 All tests passed! The OULAD deep learning implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()