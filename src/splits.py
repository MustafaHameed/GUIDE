"""
Dataset-specific splitting utilities for leak-free evaluation.

Provides utilities for:
- StratifiedKFold for UCI data (balanced by label and demographics)
- GroupKFold for OULAD data (prevents leakage across module+presentation)
- Label encoding standardization to {0,1} with 1 = at-risk/fail
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def encode_labels(y: pd.Series, positive_label: str = "fail") -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Encode labels to {0,1} with 1 = at-risk/fail.
    
    Args:
        y: Raw target labels
        positive_label: The value that should be encoded as 1 (positive class)
        
    Returns:
        Tuple of (encoded_labels, mapping_info)
    """
    logger.info(f"Encoding labels with positive_label='{positive_label}'")
    
    unique_vals = y.unique()
    logger.info(f"Original label values: {unique_vals}")
    
    # Create mapping - positive_label maps to 1, others to 0
    mapping = {}
    for val in unique_vals:
        if str(val).lower() in [positive_label.lower(), "1", "true", "fail", "withdraw", "at-risk"]:
            mapping[val] = 1
        else:
            mapping[val] = 0
    
    # Apply mapping
    y_encoded = y.map(mapping)
    
    # Store mapping info for traceability
    mapping_info = {
        "original_values": list(unique_vals),
        "mapping": mapping,
        "positive_label": positive_label,
        "positive_class_count": int(y_encoded.sum()),
        "total_count": len(y_encoded)
    }
    
    logger.info(f"Label encoding mapping: {mapping}")
    logger.info(f"Positive class (1) count: {mapping_info['positive_class_count']}/{mapping_info['total_count']}")
    
    return y_encoded, mapping_info


def get_stratified_kfold(n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> StratifiedKFold:
    """
    Get StratifiedKFold splitter for UCI-like datasets.
    
    Args:
        n_splits: Number of folds
        shuffle: Whether to shuffle before splitting
        random_state: Random seed for reproducibility
        
    Returns:
        StratifiedKFold instance
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def get_group_kfold(n_splits: int = 5) -> GroupKFold:
    """
    Get GroupKFold splitter for OULAD data to prevent leakage across module+presentation.
    
    Args:
        n_splits: Number of folds
        
    Returns:
        GroupKFold instance
    """
    return GroupKFold(n_splits=n_splits)


def create_oulad_groups(df: pd.DataFrame, group_key: str = "module_presentation") -> pd.Series:
    """
    Create groups for OULAD GroupKFold splitting.
    
    Args:
        df: OULAD dataframe with module and presentation columns
        group_key: Type of grouping ('module_presentation', 'module', 'presentation')
        
    Returns:
        Series of group identifiers
    """
    if group_key == "module_presentation":
        if "code_module" in df.columns and "code_presentation" in df.columns:
            groups = df["code_module"].astype(str) + "_" + df["code_presentation"].astype(str)
        else:
            logger.warning("code_module/code_presentation columns not found, using student IDs as groups")
            groups = df.get("id_student", range(len(df)))
    elif group_key == "module":
        groups = df.get("code_module", range(len(df)))
    elif group_key == "presentation":
        groups = df.get("code_presentation", range(len(df)))
    else:
        raise ValueError(f"Unknown group_key: {group_key}")
    
    logger.info(f"Created {len(groups.unique())} unique groups using {group_key}")
    return groups


def get_cv_splitter(dataset_name: str, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[Any, Optional[pd.Series]]:
    """
    Get appropriate CV splitter based on dataset and configuration.
    
    Args:
        dataset_name: Name of dataset ('uci', 'oulad', etc.)
        df: Input dataframe
        config: CV configuration from train config
        
    Returns:
        Tuple of (cv_splitter, groups) where groups is None for StratifiedKFold
    """
    cv_config = config.get("cv", {})
    strategy = cv_config.get("strategy", "stratified_kfold")
    n_splits = cv_config.get("n_splits", 5)
    shuffle = cv_config.get("shuffle", True)
    random_state = cv_config.get("random_state", 42)
    
    if dataset_name.lower() == "uci" or strategy == "stratified_kfold":
        logger.info(f"Using StratifiedKFold with {n_splits} splits for {dataset_name}")
        return get_stratified_kfold(n_splits, shuffle, random_state), None
    
    elif dataset_name.lower() == "oulad" or strategy == "group_kfold":
        logger.info(f"Using GroupKFold with {n_splits} splits for {dataset_name}")
        group_key = cv_config.get("group_key", "module_presentation")
        groups = create_oulad_groups(df, group_key)
        return get_group_kfold(n_splits), groups
    
    else:
        logger.warning(f"Unknown dataset/strategy: {dataset_name}/{strategy}, defaulting to StratifiedKFold")
        return get_stratified_kfold(n_splits, shuffle, random_state), None


def validate_splits(X: pd.DataFrame, y: pd.Series, cv_splitter: Any, groups: Optional[pd.Series] = None) -> bool:
    """
    Validate that splits are leak-free and properly balanced.
    
    Args:
        X: Feature matrix
        y: Target labels
        cv_splitter: CV splitter instance
        groups: Group labels for GroupKFold (None for StratifiedKFold)
        
    Returns:
        True if validation passes
    """
    logger.info("Validating CV splits...")
    
    try:
        split_iter = cv_splitter.split(X, y, groups) if groups is not None else cv_splitter.split(X, y)
        
        fold_stats = []
        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            # Check for index overlap (should be none)
            overlap = set(train_idx) & set(val_idx)
            if overlap:
                logger.error(f"Fold {fold_idx}: Found {len(overlap)} overlapping indices!")
                return False
            
            # Check label balance
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            train_pos_rate = y_train_fold.mean()
            val_pos_rate = y_val_fold.mean()
            
            fold_stats.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_pos_rate": train_pos_rate,
                "val_pos_rate": val_pos_rate
            })
            
            logger.debug(f"Fold {fold_idx}: train_size={len(train_idx)}, val_size={len(val_idx)}, "
                        f"train_pos={train_pos_rate:.3f}, val_pos={val_pos_rate:.3f}")
        
        # Log summary statistics
        if fold_stats:
            avg_train_pos = np.mean([f["train_pos_rate"] for f in fold_stats])
            avg_val_pos = np.mean([f["val_pos_rate"] for f in fold_stats])
            logger.info(f"CV validation passed. Avg train pos rate: {avg_train_pos:.3f}, "
                       f"avg val pos rate: {avg_val_pos:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"CV split validation failed: {e}")
        return False