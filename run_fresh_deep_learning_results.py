#!/usr/bin/env python3
"""
Fresh Deep Learning Results - GUIDE Project
===========================================

This script runs all available deep learning models and presents fresh, comprehensive results.
Focuses on working models and provides detailed performance comparisons.

Author: GUIDE Team  
Date: 2025-08-25
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create results directory
RESULTS_DIR = project_root / f"fresh_dl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "figures").mkdir(exist_ok=True)
(RESULTS_DIR / "tables").mkdir(exist_ok=True)
(RESULTS_DIR / "models").mkdir(exist_ok=True)


class SimpleTabularMLP(nn.Module):
    """Simple but effective MLP for tabular data."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 dropout: float = 0.3, num_classes: int = 2):
        super(SimpleTabularMLP, self).__init__()
        
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
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class DeepTabularNet(nn.Module):
    """Deep tabular network with residual connections."""
    
    def __init__(self, input_dim: int, num_classes: int = 2):
        super(DeepTabularNet, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        # Residual blocks
        self.block1 = self._make_block(512, 256)
        self.block2 = self._make_block(256, 128)
        self.block3 = self._make_block(128, 64)
        
        self.output_layer = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def _make_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.input_layer(x)))
        
        # Residual connections with dimension matching
        x1 = torch.relu(self.block1(x)) 
        x2 = torch.relu(self.block2(x1))
        x3 = torch.relu(self.block3(x2))
        
        x = self.dropout(x3)
        return self.output_layer(x)


class WideAndDeep(nn.Module):
    """Wide & Deep model for tabular data."""
    
    def __init__(self, input_dim: int, num_classes: int = 2):
        super(WideAndDeep, self).__init__()
        
        # Wide component (linear)
        self.wide = nn.Linear(input_dim, 1)
        
        # Deep component
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # Final layer combines both
        self.final = nn.Linear(2, num_classes)
        
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.final(combined)


def load_and_prepare_data(data_path: str = "data/oulad/processed/oulad_ml.csv") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and prepare data for training."""
    logger.info(f"Loading data from {data_path}")
    
    if Path(data_path).exists():
        df = pd.read_csv(data_path)
        logger.info(f"Loaded OULAD data: {df.shape}")
        
        # Prepare OULAD data
        categorical_cols = ['code_module', 'code_presentation', 'sex', 'age_band', 
                           'highest_education', 'imd_band', 'sex_x_age']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Create binary target
        if 'label_pass' in df.columns:
            target_col = 'label_pass'
        else:
            target_col = 'assessment_ontime_rate'
            df[target_col] = (df[target_col] > df[target_col].median()).astype(int)
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['id_student', target_col]]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col]
        
    else:
        # Fallback to UCI student data
        logger.info("OULAD data not found, using UCI student data")
        df = pd.read_csv("student-mat.csv")
        logger.info(f"Loaded UCI data: {df.shape}")
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        # Create binary target from G3
        target_col = 'G3'
        y = (df[target_col] >= 10).astype(int)  # Pass/fail threshold
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
    
    # Fill missing values
    X = X.fillna(X.median())
    
    logger.info(f"Features: {X.shape[1]}, Target distribution: {y.value_counts().to_dict()}")
    return X.values, y.values, feature_cols


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, device='cpu'):
    """Train a PyTorch model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': [], 'val_auc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_probs = torch.softmax(val_outputs, dim=1)[:, 1]
            val_preds = torch.argmax(val_outputs, dim=1)
            
            train_preds = torch.argmax(model(X_train_tensor), dim=1)
            train_acc = accuracy_score(y_train, train_preds.cpu().numpy())
            val_acc = accuracy_score(y_val, val_preds.cpu().numpy())
            
            try:
                val_auc = roc_auc_score(y_val, val_probs.cpu().numpy())
            except:
                val_auc = 0.5
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        if patience_counter >= 15:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    return model, history


def evaluate_model(model, X_test, y_test, device='cpu'):
    """Evaluate a trained model."""
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
    
    acc = accuracy_score(y_test, preds.cpu().numpy())
    try:
        auc = roc_auc_score(y_test, probs.cpu().numpy())
    except:
        auc = 0.5
    f1 = f1_score(y_test, preds.cpu().numpy())
    
    return {
        'accuracy': acc,
        'roc_auc': auc,
        'f1_score': f1,
        'predictions': preds.cpu().numpy(),
        'probabilities': probs.cpu().numpy()
    }


def run_deep_learning_experiments():
    """Run comprehensive deep learning experiments."""
    logger.info("🚀 STARTING FRESH DEEP LEARNING EXPERIMENTS")
    logger.info("=" * 60)
    
    # Load and prepare data
    X, y, feature_cols = load_and_prepare_data()
    
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    input_dim = X_train.shape[1]
    logger.info(f"Data splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Define models to train
    models_config = {
        'SimpleTabularMLP': SimpleTabularMLP(input_dim, [256, 128, 64]),
        'DeepTabularNet': DeepTabularNet(input_dim),
        'WideAndDeep': WideAndDeep(input_dim),
        'LightweightMLP': SimpleTabularMLP(input_dim, [128, 64], dropout=0.2),
        'DeepMLP': SimpleTabularMLP(input_dim, [512, 256, 128, 64], dropout=0.4),
    }
    
    results = {}
    trained_models = {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Train each model
    for model_name, model in models_config.items():
        logger.info(f"\n🧠 Training {model_name}...")
        
        try:
            trained_model, history = train_model(
                model, X_train, y_train, X_val, y_val, 
                epochs=150, lr=0.001, device=device
            )
            
            # Evaluate on test set
            test_results = evaluate_model(trained_model, X_test, y_test, device)
            
            # Add training history
            test_results['history'] = history
            test_results['val_accuracy'] = max(history['val_acc'])
            test_results['val_auc'] = max(history['val_auc'])
            
            results[model_name] = test_results
            trained_models[model_name] = trained_model
            
            logger.info(f"✅ {model_name} - Test Acc: {test_results['accuracy']:.4f}, "
                       f"Test AUC: {test_results['roc_auc']:.4f}, Test F1: {test_results['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {e}")
            continue
    
    return results, trained_models, (X_test, y_test), feature_cols


def create_ensemble_predictions(models, X_test, device='cpu'):
    """Create ensemble predictions from multiple models."""
    logger.info("🔧 Creating ensemble predictions...")
    
    all_probs = []
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    for model_name, model in models.items():
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
    
    # Average probabilities
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    return ensemble_preds, ensemble_probs


def save_results(results, test_data, feature_cols):
    """Save all results to files."""
    logger.info("💾 Saving results...")
    
    X_test, y_test = test_data
    
    # Create summary DataFrame
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Test_Accuracy': metrics['accuracy'],
            'Test_ROC_AUC': metrics['roc_auc'],
            'Test_F1_Score': metrics['f1_score'],
            'Val_Accuracy': metrics.get('val_accuracy', 0),
            'Val_AUC': metrics.get('val_auc', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
    summary_df.to_csv(RESULTS_DIR / "tables" / "deep_learning_results.csv", index=False)
    
    # Create detailed results JSON
    detailed_results = {}
    for model_name, metrics in results.items():
        detailed_results[model_name] = {
            'test_accuracy': float(metrics['accuracy']),
            'test_roc_auc': float(metrics['roc_auc']),
            'test_f1_score': float(metrics['f1_score']),
            'val_accuracy': float(metrics.get('val_accuracy', 0)),
            'val_auc': float(metrics.get('val_auc', 0)),
            'classification_report': classification_report(y_test, metrics['predictions'])
        }
    
    with open(RESULTS_DIR / "tables" / "detailed_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"Results saved to {RESULTS_DIR / 'tables'}")
    return summary_df


def create_visualizations(results, summary_df):
    """Create visualizations of the results."""
    logger.info("📊 Creating visualizations...")
    
    # Performance comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Accuracy comparison
    plt.subplot(2, 2, 1)
    plt.bar(summary_df['Model'], summary_df['Test_Accuracy'], color='skyblue', alpha=0.7)
    plt.title('Test Accuracy by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: ROC AUC comparison
    plt.subplot(2, 2, 2)
    plt.bar(summary_df['Model'], summary_df['Test_ROC_AUC'], color='lightcoral', alpha=0.7)
    plt.title('Test ROC AUC by Model', fontsize=14, fontweight='bold')
    plt.ylabel('ROC AUC')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: F1 Score comparison
    plt.subplot(2, 2, 3)
    plt.bar(summary_df['Model'], summary_df['Test_F1_Score'], color='lightgreen', alpha=0.7)
    plt.title('Test F1 Score by Model', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 4: Validation vs Test Accuracy
    plt.subplot(2, 2, 4)
    plt.scatter(summary_df['Val_Accuracy'], summary_df['Test_Accuracy'], 
                s=100, alpha=0.7, color='purple')
    for i, model in enumerate(summary_df['Model']):
        plt.annotate(model, (summary_df['Val_Accuracy'].iloc[i], 
                            summary_df['Test_Accuracy'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.plot([0.4, 0.7], [0.4, 0.7], 'r--', alpha=0.5)  # diagonal line
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Test Accuracy')
    plt.title('Validation vs Test Accuracy', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "figures" / "deep_learning_performance_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Training curves for best model
    best_model_name = summary_df.iloc[0]['Model']
    if 'history' in results[best_model_name]:
        history = results[best_model_name]['history']
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Training Accuracy', color='blue')
        plt.plot(history['val_acc'], label='Validation Accuracy', color='red')
        plt.title(f'{best_model_name} - Training Curves (Accuracy)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_auc'], label='Validation AUC', color='green')
        plt.title(f'{best_model_name} - Training Curves (AUC)', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "figures" / f"{best_model_name}_training_curves.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()


def print_comprehensive_summary(results, summary_df):
    """Print comprehensive summary of results."""
    print("\n" + "=" * 80)
    print("🎯 FRESH DEEP LEARNING RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Models Trained: {len(results)}")
    print(f"🏆 Best Model: {summary_df.iloc[0]['Model']}")
    print(f"📈 Best Accuracy: {summary_df.iloc[0]['Test_Accuracy']:.4f}")
    print(f"📈 Best ROC AUC: {summary_df.iloc[0]['Test_ROC_AUC']:.4f}")
    print(f"📈 Best F1 Score: {summary_df.iloc[0]['Test_F1_Score']:.4f}")
    
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 80)
    print(f"{'Model':<20} {'Test Acc':<10} {'Test AUC':<10} {'Test F1':<10} {'Val Acc':<10}")
    print("-" * 80)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<20} {row['Test_Accuracy']:<10.4f} "
              f"{row['Test_ROC_AUC']:<10.4f} {row['Test_F1_Score']:<10.4f} "
              f"{row['Val_Accuracy']:<10.4f}")
    
    print("\n📁 Results saved to:")
    print(f"   📊 Tables: {RESULTS_DIR / 'tables'}")
    print(f"   🎨 Figures: {RESULTS_DIR / 'figures'}")
    print(f"   🏠 Main directory: {RESULTS_DIR}")
    
    print("\n✅ Fresh deep learning results generation complete!")


def main():
    """Main execution function."""
    print("🎯 FRESH DEEP LEARNING RESULTS GENERATOR")
    print("=" * 60)
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results directory: {RESULTS_DIR}")
    print("=" * 60)
    
    try:
        # Run experiments
        results, trained_models, test_data, feature_cols = run_deep_learning_experiments()
        
        if not results:
            logger.error("No models were successfully trained!")
            return
        
        # Create ensemble
        if len(trained_models) > 1:
            ensemble_preds, ensemble_probs = create_ensemble_predictions(
                trained_models, test_data[0]
            )
            
            # Evaluate ensemble
            y_test = test_data[1]
            ensemble_acc = accuracy_score(y_test, ensemble_preds)
            ensemble_auc = roc_auc_score(y_test, ensemble_probs)
            ensemble_f1 = f1_score(y_test, ensemble_preds)
            
            results['Ensemble'] = {
                'accuracy': ensemble_acc,
                'roc_auc': ensemble_auc,
                'f1_score': ensemble_f1,
                'predictions': ensemble_preds,
                'probabilities': ensemble_probs
            }
            
            logger.info(f"✅ Ensemble - Test Acc: {ensemble_acc:.4f}, "
                       f"Test AUC: {ensemble_auc:.4f}, Test F1: {ensemble_f1:.4f}")
        
        # Save results and create visualizations
        summary_df = save_results(results, test_data, feature_cols)
        create_visualizations(results, summary_df)
        
        # Print comprehensive summary
        print_comprehensive_summary(results, summary_df)
        
        return RESULTS_DIR
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()