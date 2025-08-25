#!/usr/bin/env python3
"""
Test script to compare improved transfer learning with baseline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import logging
from src.transfer.improved_transfer import improved_transfer_experiment
from src.transfer.uci_transfer import transfer_experiment, create_shared_feature_mapping, prepare_oulad_features, prepare_uci_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_transfer_methods():
    """Compare baseline vs improved transfer learning."""
    
    # Load data
    oulad_df = pd.read_parquet('data/oulad/processed/oulad_ml.parquet')
    
    # Prepare data for baseline method
    feature_mapping = create_shared_feature_mapping()
    oulad_shared = prepare_oulad_features(oulad_df, feature_mapping)
    uci_shared = prepare_uci_features('student-mat.csv', feature_mapping)
    
    # Remove rows with missing labels
    oulad_clean = oulad_shared.dropna(subset=['label'])
    uci_clean = uci_shared.dropna(subset=['label'])
    
    # For baseline, need to ensure common feature set
    feature_cols = [col for col in oulad_clean.columns if col != 'label']
    uci_common = uci_clean[feature_cols + ['label']].copy()
    
    print("=== Dataset Information ===")
    print(f"OULAD shape: {oulad_clean.shape}")
    print(f"UCI common features shape: {uci_common.shape}")
    print(f"Common features: {feature_cols}")
    print(f"UCI baseline accuracy: {uci_clean['label'].value_counts(normalize=True).max():.3f}")
    
    print("\n=== Improved Transfer Learning ===")
    
    # Test improved method first (more robust)
    try:
        improved_results = improved_transfer_experiment(
            oulad_clean, uci_clean,
            use_ensemble=True,
            use_domain_adaptation=True
        )
        print("Improved Ensemble + Domain Adaptation:")
        print(f"  Accuracy: {improved_results['accuracy']:.3f}")
        print(f"  AUC: {improved_results.get('auc', 'N/A'):.3f}")
        print(f"  F1: {improved_results['f1']:.3f}")
        print(f"  Source CV Accuracy: {improved_results['source_cv_accuracy']:.3f}")
        print(f"  Features used: {improved_results['n_features']}")
        
        # Test without domain adaptation
        improved_no_da = improved_transfer_experiment(
            oulad_clean, uci_clean,
            use_ensemble=True,
            use_domain_adaptation=False
        )
        print("\nImproved Ensemble only:")
        print(f"  Accuracy: {improved_no_da['accuracy']:.3f}")
        print(f"  AUC: {improved_no_da.get('auc', 'N/A'):.3f}")
        
        # Test single model with improvements
        improved_single = improved_transfer_experiment(
            oulad_clean, uci_clean,
            use_ensemble=False,
            use_domain_adaptation=True
        )
        print("\nImproved Single Model + DA:")
        print(f"  Accuracy: {improved_single['accuracy']:.3f}")
        print(f"  AUC: {improved_single.get('auc', 'N/A'):.3f}")
        
    except Exception as e:
        print(f"Improved method failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_transfer_methods()