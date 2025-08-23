"""
OULAD Dataset Splitting

Defines three evaluation splits for OULAD dataset:
1. Chronological: Train on early presentations, test on later ones
2. Module holdout: Rotate each course module as test set  
3. Random: Stratified split by label and gender

References:
- OU Analyse: https://analyse.kmi.open.ac.uk/open-dataset
- OULAD presentations B (February) and J (October) represent chronological order
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_leakage(train_ids: List[int], val_ids: List[int], test_ids: List[int]) -> bool:
    """Check for student ID leakage between splits.
    
    Args:
        train_ids: Training set student IDs
        val_ids: Validation set student IDs  
        test_ids: Test set student IDs
        
    Returns:
        True if no leakage detected, False otherwise
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    # Check for overlaps
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    if train_val_overlap:
        logger.error(f"Train-Val overlap: {len(train_val_overlap)} students")
        return False
    if train_test_overlap:
        logger.error(f"Train-Test overlap: {len(train_test_overlap)} students") 
        return False
    if val_test_overlap:
        logger.error(f"Val-Test overlap: {len(val_test_overlap)} students")
        return False
        
    logger.info("No leakage detected between splits")
    return True


def save_split(split_data: Dict[str, List[int]], output_path: Path) -> None:
    """Save split data to JSON file.
    
    Args:
        split_data: Dictionary with train/val/test student ID lists
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    
    logger.info(f"Saved split to {output_path}")
    
    # Print split sizes
    for split_name, ids in split_data.items():
        logger.info(f"{split_name}: {len(ids)} students")


def chronological_split(df: pd.DataFrame, output_dir: Path) -> Dict[str, List[int]]:
    """Create chronological split based on presentation timing.
    
    Default order: [2013B, 2013J, 2014B, 2014J]
    Train on early presentations, validate on middle, test on latest.
    
    Args:
        df: OULAD dataset with code_presentation column
        output_dir: Directory to save split files
        
    Returns:
        Dictionary with train/val/test student ID lists
    """
    logger.info("Creating chronological split...")
    
    # Define chronological order (B = February, J = October)
    presentation_order = ['2013B', '2013J', '2014B', '2014J']
    
    # Filter to presentations that exist in data
    available_presentations = df['code_presentation'].unique()
    ordered_presentations = [p for p in presentation_order if p in available_presentations]
    
    if len(ordered_presentations) < 3:
        logger.warning(f"Only {len(ordered_presentations)} presentations available. Using all for training.")
        train_presentations = ordered_presentations
        val_presentations = []
        test_presentations = []
    else:
        # Split chronologically
        n_presentations = len(ordered_presentations)
        train_end = max(1, n_presentations - 2)
        val_end = max(train_end + 1, n_presentations - 1)
        
        train_presentations = ordered_presentations[:train_end]
        val_presentations = ordered_presentations[train_end:val_end]
        test_presentations = ordered_presentations[val_end:]
    
    logger.info(f"Train presentations: {train_presentations}")
    logger.info(f"Val presentations: {val_presentations}")
    logger.info(f"Test presentations: {test_presentations}")
    
    # Get student IDs for each split
    train_ids = df[df['code_presentation'].isin(train_presentations)]['id_student'].unique().tolist()
    val_ids = df[df['code_presentation'].isin(val_presentations)]['id_student'].unique().tolist()
    test_ids = df[df['code_presentation'].isin(test_presentations)]['id_student'].unique().tolist()
    
    # Check for leakage
    check_leakage(train_ids, val_ids, test_ids)
    
    split_data = {
        'train': train_ids,
        'val': val_ids, 
        'test': test_ids,
        'metadata': {
            'split_type': 'chronological',
            'train_presentations': train_presentations,
            'val_presentations': val_presentations,
            'test_presentations': test_presentations
        }
    }
    
    save_split(split_data, output_dir / 'chronological_split.json')
    return split_data


def module_holdout_split(df: pd.DataFrame, output_dir: Path) -> Dict[str, Dict[str, List[int]]]:
    """Create module holdout splits rotating each course module as test.
    
    Args:
        df: OULAD dataset with code_module column
        output_dir: Directory to save split files
        
    Returns:
        Dictionary mapping module names to train/val/test splits
    """
    logger.info("Creating module holdout splits...")
    
    modules = df['code_module'].unique()
    all_splits = {}
    
    for test_module in modules:
        logger.info(f"Creating split with {test_module} as test module")
        
        # Students in test module
        test_students = df[df['code_module'] == test_module]['id_student'].unique()
        
        # Students in other modules for train/val
        other_students = df[df['code_module'] != test_module]['id_student'].unique()
        
        # Split other students into train/val (stratified by label and sex if available)
        if len(other_students) > 0:
            # Get representative sample for stratification
            other_df = df[df['id_student'].isin(other_students)].drop_duplicates('id_student')
            
            try:
                # Try stratified split by label and sex
                stratify_cols = []
                if 'label_pass' in other_df.columns:
                    stratify_cols.append('label_pass')
                if 'sex' in other_df.columns:
                    stratify_cols.append('sex')
                
                if stratify_cols:
                    stratify_values = other_df[stratify_cols].fillna('unknown')
                    if len(stratify_cols) > 1:
                        stratify_values = stratify_values.apply(lambda x: '_'.join(x.astype(str)), axis=1)
                    else:
                        stratify_values = stratify_values.iloc[:, 0]
                    
                    train_students, val_students = train_test_split(
                        other_students,
                        test_size=0.2,
                        stratify=stratify_values,
                        random_state=42
                    )
                else:
                    train_students, val_students = train_test_split(
                        other_students,
                        test_size=0.2,
                        random_state=42
                    )
            except ValueError:
                # Fallback to simple split if stratification fails
                train_students, val_students = train_test_split(
                    other_students,
                    test_size=0.2,
                    random_state=42
                )
        else:
            train_students = val_students = []
        
        # Check for leakage
        check_leakage(train_students.tolist(), val_students.tolist(), test_students.tolist())
        
        split_data = {
            'train': train_students.tolist(),
            'val': val_students.tolist(),
            'test': test_students.tolist(),
            'metadata': {
                'split_type': 'module_holdout',
                'test_module': test_module,
                'other_modules': [m for m in modules if m != test_module]
            }
        }
        
        save_split(split_data, output_dir / f'module_holdout_{test_module}.json')
        all_splits[test_module] = split_data
    
    return all_splits


def random_split(df: pd.DataFrame, output_dir: Path, 
                train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1) -> Dict[str, List[int]]:
    """Create stratified random split preserving student grouping.
    
    Args:
        df: OULAD dataset  
        output_dir: Directory to save split files
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        
    Returns:
        Dictionary with train/val/test student ID lists
    """
    logger.info("Creating stratified random split...")
    
    # Ensure sizes sum to 1
    total_size = train_size + val_size + test_size
    if not np.isclose(total_size, 1.0):
        logger.warning(f"Split sizes sum to {total_size}, normalizing...")
        train_size /= total_size
        val_size /= total_size
        test_size /= total_size
    
    # Get unique students with their labels and demographics
    student_df = df.drop_duplicates('id_student')
    
    # Prepare stratification variables
    stratify_cols = []
    if 'label_pass' in student_df.columns:
        stratify_cols.append('label_pass')
    if 'sex' in student_df.columns:
        stratify_cols.append('sex')
    
    try:
        if stratify_cols:
            # Create combined stratification variable
            stratify_df = student_df[stratify_cols].fillna('unknown')
            if len(stratify_cols) > 1:
                stratify_var = stratify_df.apply(lambda x: '_'.join(x.astype(str)), axis=1)
            else:
                stratify_var = stratify_df.iloc[:, 0]
            
            # First split: separate out test set
            train_val_students, test_students = train_test_split(
                student_df['id_student'].values,
                test_size=test_size,
                stratify=stratify_var,
                random_state=42
            )
            
            # Second split: separate train and val
            val_size_adjusted = val_size / (train_size + val_size)
            train_stratify = stratify_var[student_df['id_student'].isin(train_val_students)]
            
            train_students, val_students = train_test_split(
                train_val_students,
                test_size=val_size_adjusted,
                stratify=train_stratify,
                random_state=42
            )
        else:
            logger.warning("No stratification variables available, using simple random split")
            # Simple random split
            train_val_students, test_students = train_test_split(
                student_df['id_student'].values,
                test_size=test_size,
                random_state=42
            )
            
            val_size_adjusted = val_size / (train_size + val_size)
            train_students, val_students = train_test_split(
                train_val_students,
                test_size=val_size_adjusted,
                random_state=42
            )
            
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using simple random split.")
        # Fallback to simple split
        train_val_students, test_students = train_test_split(
            student_df['id_student'].values,
            test_size=test_size,
            random_state=42
        )
        
        val_size_adjusted = val_size / (train_size + val_size)
        train_students, val_students = train_test_split(
            train_val_students,
            test_size=val_size_adjusted,
            random_state=42
        )
    
    # Check for leakage
    check_leakage(train_students.tolist(), val_students.tolist(), test_students.tolist())
    
    split_data = {
        'train': train_students.tolist(),
        'val': val_students.tolist(),
        'test': test_students.tolist(),
        'metadata': {
            'split_type': 'random',
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'stratify_cols': stratify_cols,
            'random_state': 42
        }
    }
    
    save_split(split_data, output_dir / 'random_split.json')
    return split_data


def create_all_splits(dataset_path: Path, output_dir: Path) -> Dict[str, any]:
    """Create all three types of splits for OULAD dataset.
    
    Args:
        dataset_path: Path to OULAD parquet file
        output_dir: Directory to save split JSON files
        
    Returns:
        Dictionary containing all splits
    """
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Unique students: {df['id_student'].nunique()}")
    logger.info(f"Unique modules: {df['code_module'].nunique()}")
    logger.info(f"Unique presentations: {df['code_presentation'].nunique()}")
    
    all_splits = {}
    
    # Create chronological split
    all_splits['chronological'] = chronological_split(df, output_dir)
    
    # Create module holdout splits
    all_splits['module_holdout'] = module_holdout_split(df, output_dir)
    
    # Create random split
    all_splits['random'] = random_split(df, output_dir)
    
    logger.info("All splits created successfully!")
    return all_splits


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for split creation."""
    parser = argparse.ArgumentParser(description='Create OULAD dataset splits')
    parser.add_argument(
        '--dataset',
        type=Path,
        default='data/oulad/processed/oulad_ml.parquet',
        help='Path to OULAD parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='data/oulad/splits',
        help='Directory to save split JSON files'
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """CLI entrypoint for creating OULAD splits."""
    args = parse_args(argv)

    try:
        create_all_splits(args.dataset, args.output_dir)
        logger.info("Split creation completed successfully!")
    except Exception as e:
        logger.error(f"Failed to create splits: {e}")
        raise


if __name__ == '__main__':
    main()