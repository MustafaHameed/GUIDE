#!/usr/bin/env python3
"""
Enhanced Data Loading for GUIDE Project

Provides unified loading functions for all three datasets:
- OULAD (Open University Learning Analytics Dataset)
- XuetangX (MOOC dataset)
- UCI (Student Performance dataset)

Usage:
    from scripts.enhanced_data_loader import load_dataset, load_all_datasets
    
    # Load individual datasets
    oulad_df = load_dataset('oulad')
    xuetangx_df = load_dataset('xuetangx')
    uci_df = load_dataset('uci')
    
    # Load all datasets
    datasets = load_all_datasets()
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


def load_oulad_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load OULAD dataset (real or synthetic)."""
    if data_path is None:
        # Try real dataset first, then fall back to synthetic
        real_path = "data/oulad/processed/oulad_ml.csv"
        synthetic_path = "data/oulad/processed/oulad_ml.csv"  # Currently contains synthetic
        data_path = real_path
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded OULAD dataset: {df.shape}")
        
        # Ensure required columns exist
        if 'label_pass' not in df.columns and 'final_result' in df.columns:
            df['label_pass'] = (df['final_result'] == 'Pass').astype(int)
        
        return df
        
    except FileNotFoundError:
        logger.error(f"OULAD dataset not found at {data_path}")
        logger.info("Run: python scripts/download_datasets.py --dataset oulad")
        logger.info("Then: python scripts/preprocess_oulad.py")
        raise


def load_xuetangx_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load XuetangX MOOC dataset."""
    if data_path is None:
        data_path = "data/xuetangx/processed/xuetangx_ml.csv"
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded XuetangX dataset: {df.shape}")
        return df
        
    except FileNotFoundError:
        logger.error(f"XuetangX dataset not found at {data_path}")
        logger.info("Run: python scripts/preprocess_xuetangx.py --synthetic")
        raise


def load_uci_dataset(data_path: Optional[str] = None, dataset_type: str = 'math') -> pd.DataFrame:
    """Load UCI Student Performance dataset."""
    if data_path is None:
        if dataset_type == 'math':
            # Try multiple possible locations
            possible_paths = [
                "data/uci/raw/student-mat.csv",
                "student-mat.csv",
                "student-mat-fixed.csv"
            ]
        else:  # portuguese
            possible_paths = [
                "data/uci/raw/student-por.csv", 
                "student-por.csv"
            ]
        
        data_path = None
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                break
    
    if data_path is None:
        raise FileNotFoundError("UCI dataset not found. Run: python scripts/download_datasets.py --dataset uci")
    
    try:
        df = pd.read_csv(data_path)
        
        # Create pass/fail label if not exists
        if 'label_pass' not in df.columns:
            if 'G3' in df.columns:
                df['label_pass'] = (df['G3'] >= 10).astype(int)
            else:
                logger.warning("No G3 column found for creating labels")
                
        logger.info(f"Loaded UCI dataset ({dataset_type}): {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading UCI dataset from {data_path}: {e}")
        raise


def load_dataset(dataset_name: str, **kwargs) -> pd.DataFrame:
    """Load a specific dataset by name."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'oulad':
        return load_oulad_dataset(**kwargs)
    elif dataset_name == 'xuetangx':
        return load_xuetangx_dataset(**kwargs)
    elif dataset_name == 'uci':
        return load_uci_dataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: oulad, xuetangx, uci")


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """Load all available datasets."""
    datasets = {}
    
    # Load OULAD
    try:
        datasets['oulad'] = load_oulad_dataset()
    except Exception as e:
        logger.warning(f"Could not load OULAD: {e}")
    
    # Load XuetangX
    try:
        datasets['xuetangx'] = load_xuetangx_dataset()
    except Exception as e:
        logger.warning(f"Could not load XuetangX: {e}")
    
    # Load UCI
    try:
        datasets['uci'] = load_uci_dataset()
    except Exception as e:
        logger.warning(f"Could not load UCI: {e}")
    
    logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    return datasets


def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a dataset."""
    try:
        df = load_dataset(dataset_name)
        
        info = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'target_columns': [col for col in df.columns if col.startswith('label_')],
            'missing_values': df.isnull().sum().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Add target distribution if available
        for target_col in info['target_columns']:
            if target_col in df.columns:
                info[f'{target_col}_distribution'] = df[target_col].value_counts().to_dict()
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting info for {dataset_name}: {e}")
        return {}


def create_transfer_learning_pairs() -> List[Tuple[str, str]]:
    """Create all possible transfer learning pairs."""
    datasets = ['oulad', 'xuetangx', 'uci']
    pairs = []
    
    for source in datasets:
        for target in datasets:
            if source != target:
                pairs.append((source, target))
    
    return pairs


def validate_datasets() -> Dict[str, bool]:
    """Validate that all datasets can be loaded."""
    validation = {}
    
    for dataset_name in ['oulad', 'xuetangx', 'uci']:
        try:
            df = load_dataset(dataset_name)
            
            # Basic validation checks
            checks = {
                'has_data': len(df) > 0,
                'has_target': any(col.startswith('label_') for col in df.columns),
                'no_all_missing': not df.isnull().all(axis=1).any(),
            }
            
            validation[dataset_name] = all(checks.values())
            
            if not validation[dataset_name]:
                logger.warning(f"Dataset {dataset_name} failed validation: {checks}")
                
        except Exception as e:
            logger.error(f"Dataset {dataset_name} failed to load: {e}")
            validation[dataset_name] = False
    
    return validation


def main():
    """Demonstrate the data loading functionality."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== GUIDE Dataset Loader Demo ===\n")
    
    # Validate all datasets
    print("1. Validating datasets...")
    validation = validate_datasets()
    for dataset, valid in validation.items():
        status = "✓" if valid else "✗"
        print(f"   {dataset.upper()}: {status}")
    
    print("\n2. Loading all datasets...")
    datasets = load_all_datasets()
    
    print("\n3. Dataset information:")
    for name, df in datasets.items():
        print(f"\n   {name.upper()}:")
        print(f"     Shape: {df.shape}")
        
        # Show target distribution
        target_cols = [col for col in df.columns if col.startswith('label_')]
        for target_col in target_cols:
            if target_col in df.columns:
                dist = df[target_col].value_counts()
                rate = df[target_col].mean()
                print(f"     {target_col}: {rate:.1%} positive rate")
    
    print("\n4. Transfer learning pairs:")
    pairs = create_transfer_learning_pairs()
    for i, (source, target) in enumerate(pairs, 1):
        print(f"   {i}. {source.upper()} → {target.upper()}")
    
    print(f"\nTotal: {len(datasets)} datasets, {len(pairs)} transfer pairs")


if __name__ == "__main__":
    main()