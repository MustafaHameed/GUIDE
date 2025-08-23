"""
OULAD Dataset Builder

Builds a unified machine learning dataset from OULAD raw CSV files.
Combines student demographics, VLE interactions, and assessment data into
a single row per student-module-presentation with features, labels, and 
sensitive attributes.

References:
- OULAD Documentation: https://analyse.kmi.open.ac.uk/open-dataset
- Nature Paper: https://www.nature.com/articles/sdata2017171
- PMC Technical Details: https://pmc.ncbi.nlm.nih.gov/articles/PMC5704676/
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_oulad_tables(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all OULAD CSV tables into memory.
    
    Args:
        raw_dir: Directory containing OULAD CSV files
        
    Returns:
        Dictionary mapping table names to DataFrames
    """
    tables = {}
    
    # Core tables from OULAD
    table_files = {
        'studentInfo': 'studentInfo.csv',
        'studentVle': 'studentVle.csv', 
        'vle': 'vle.csv',
        'studentRegistration': 'studentRegistration.csv',
        'studentAssessment': 'studentAssessment.csv',
        'assessments': 'assessments.csv'
    }
    
    for table_name, filename in table_files.items():
        file_path = raw_dir / filename
        if file_path.exists():
            logger.info(f"Loading {table_name} from {filename}")
            tables[table_name] = pd.read_csv(file_path)
            logger.info(f"{table_name}: {tables[table_name].shape[0]} rows, {tables[table_name].shape[1]} columns")
        else:
            logger.warning(f"File not found: {filename}")
            
    return tables


def create_vle_features(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    """Create VLE interaction features aggregated by student-presentation.
    
    Args:
        student_vle: Student VLE interaction records
        vle: VLE object metadata
        
    Returns:
        DataFrame with VLE features per student-presentation
    """
    # Merge VLE interactions with metadata
    vle_data = student_vle.merge(vle, on=['id_site'], how='left')
    
    # Calculate weekly aggregates
    vle_features = []
    
    for (code_module, code_presentation), group in vle_data.groupby(['code_module', 'code_presentation']):
        student_features = {}
        
        for (id_student,), student_group in group.groupby(['id_student']):
            features = {
                'id_student': id_student,
                'code_module': code_module,
                'code_presentation': code_presentation,
            }
            
            # Total clicks
            features['vle_total_clicks'] = student_group['sum_click'].sum()
            
            # Mean daily clicks
            features['vle_mean_clicks'] = student_group['sum_click'].mean()
            
            # Max daily clicks
            features['vle_max_clicks'] = student_group['sum_click'].max()
            
            # Early engagement (first 4 weeks)
            early_clicks = student_group[student_group['date'] <= 28]['sum_click'].sum()
            features['vle_first4_clicks'] = early_clicks
            
            # Late engagement (last 4 weeks, approximate)
            late_clicks = student_group[student_group['date'] >= -28]['sum_click'].sum()
            features['vle_last4_clicks'] = late_clicks
            
            # Cumulative engagement pattern
            sorted_data = student_group.sort_values('date')
            features['vle_cumulative_clicks'] = sorted_data['sum_click'].cumsum().iloc[-1] if len(sorted_data) > 0 else 0
            
            # Days active
            features['vle_days_active'] = student_group['date'].nunique()
            
            vle_features.append(features)
    
    return pd.DataFrame(vle_features)


def create_assessment_features(student_assessment: pd.DataFrame, assessments: pd.DataFrame) -> pd.DataFrame:
    """Create assessment submission features.
    
    Args:
        student_assessment: Student assessment submissions
        assessments: Assessment metadata
        
    Returns:
        DataFrame with assessment features per student-presentation
    """
    # Merge assessments with metadata
    assessment_data = student_assessment.merge(
        assessments, 
        on=['id_assessment'], 
        how='left'
    )
    
    assessment_features = []
    
    for (code_module, code_presentation), group in assessment_data.groupby(['code_module', 'code_presentation']):
        for (id_student,), student_group in group.groupby(['id_student']):
            features = {
                'id_student': id_student,
                'code_module': code_module,
                'code_presentation': code_presentation,
            }
            
            # Assessment count
            features['assessment_count'] = len(student_group)
            
            # Mean score
            valid_scores = student_group['score'].dropna()
            features['assessment_mean_score'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Last score
            features['assessment_last_score'] = student_group['score'].iloc[-1] if len(student_group) > 0 else np.nan
            
            # On-time submission rate (if date_submitted available)
            if 'date_submitted' in student_group.columns and 'date' in student_group.columns:
                on_time = (student_group['date_submitted'] <= student_group['date']).sum()
                features['assessment_ontime_rate'] = on_time / len(student_group) if len(student_group) > 0 else 0
            else:
                features['assessment_ontime_rate'] = np.nan
                
            assessment_features.append(features)
    
    return pd.DataFrame(assessment_features)


def create_labels_and_sensitive_attrs(student_info: pd.DataFrame, student_registration: pd.DataFrame) -> pd.DataFrame:
    """Create labels and sensitive attributes.
    
    Args:
        student_info: Student demographic information
        student_registration: Student registration and outcomes
        
    Returns:
        DataFrame with labels and sensitive attributes
    """
    # Merge student info with registration outcomes
    labels_data = student_registration.merge(
        student_info,
        on=['id_student'],
        how='left'
    )
    
    # Create binary pass/fail label
    labels_data['label_pass'] = (labels_data['final_result'] == 'Pass').astype(int)
    labels_data['label_fail_or_withdraw'] = (labels_data['final_result'].isin(['Fail', 'Withdrawn'])).astype(int)
    
    # Map sensitive attributes
    labels_data['sex'] = labels_data['gender'].map({'F': 'Female', 'M': 'Male'})
    
    # Create intersection features
    labels_data['sex_x_age'] = labels_data['sex'].astype(str) + '_x_' + labels_data['age_band'].astype(str)
    
    # Select relevant columns
    result_cols = [
        'id_student', 'code_module', 'code_presentation',
        'label_pass', 'label_fail_or_withdraw',
        'sex', 'age_band', 'highest_education', 'imd_band', 'sex_x_age',
        'studied_credits', 'num_of_prev_attempts'
    ]
    
    return labels_data[result_cols]


def build_oulad_dataset(raw_dir: Path, output_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build unified OULAD ML dataset.
    
    Args:
        raw_dir: Directory containing OULAD CSV files
        output_path: Path for output parquet file
        
    Returns:
        Tuple of (main_dataset, group_counts)
    """
    logger.info("Loading OULAD tables...")
    tables = load_oulad_tables(raw_dir)
    
    # Validate required tables exist
    required_tables = ['studentInfo', 'studentRegistration'] 
    for table in required_tables:
        if table not in tables:
            raise ValueError(f"Required table {table} not found in {raw_dir}")
    
    logger.info("Creating labels and sensitive attributes...")
    main_data = create_labels_and_sensitive_attrs(
        tables['studentInfo'], 
        tables['studentRegistration']
    )
    
    # Add VLE features if available
    if 'studentVle' in tables and 'vle' in tables:
        logger.info("Creating VLE features...")
        vle_features = create_vle_features(tables['studentVle'], tables['vle'])
        main_data = main_data.merge(
            vle_features,
            on=['id_student', 'code_module', 'code_presentation'],
            how='left'
        )
        logger.info(f"Added VLE features. Shape: {main_data.shape}")
    
    # Add assessment features if available
    if 'studentAssessment' in tables and 'assessments' in tables:
        logger.info("Creating assessment features...")
        assessment_features = create_assessment_features(
            tables['studentAssessment'], 
            tables['assessments']
        )
        main_data = main_data.merge(
            assessment_features,
            on=['id_student', 'code_module', 'code_presentation'],
            how='left'
        )
        logger.info(f"Added assessment features. Shape: {main_data.shape}")
    
    # Create group counts for fairness analysis
    sensitive_cols = ['sex', 'age_band', 'highest_education', 'imd_band', 'sex_x_age']
    group_counts = []
    
    for col in sensitive_cols:
        if col in main_data.columns:
            counts = main_data[col].value_counts().reset_index()
            counts.columns = ['group', 'count']
            counts['attribute'] = col
            counts['missingness'] = main_data[col].isna().sum()
            group_counts.append(counts)
    
    group_counts_df = pd.concat(group_counts, ignore_index=True) if group_counts else pd.DataFrame()
    
    # Validation
    logger.info("Validating dataset...")
    required_cols = ['id_student', 'code_module', 'code_presentation', 'label_pass']
    for col in required_cols:
        if col not in main_data.columns:
            raise ValueError(f"Required column {col} missing from final dataset")
    
    # Log dropped rows (never drop rows with keys and labels only due to missing VLE)
    initial_count = len(main_data)
    main_data_clean = main_data.dropna(subset=['id_student', 'label_pass'])
    final_count = len(main_data_clean)
    
    if initial_count != final_count:
        logger.warning(f"Dropped {initial_count - final_count} rows due to missing keys/labels")
    
    logger.info(f"Final dataset shape: {main_data_clean.shape}")
    
    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    main_data_clean.to_parquet(output_path, index=False)
    logger.info(f"Saved main dataset to {output_path}")
    
    if not group_counts_df.empty:
        group_counts_path = output_path.parent / 'group_counts.csv'
        group_counts_df.to_csv(group_counts_path, index=False)
        logger.info(f"Saved group counts to {group_counts_path}")
    
    return main_data_clean, group_counts_df


def main():
    """CLI interface for OULAD dataset builder."""
    parser = argparse.ArgumentParser(description='Build unified OULAD ML dataset')
    parser.add_argument(
        '--raw-dir', 
        type=Path, 
        default='data/oulad/raw',
        help='Directory containing OULAD CSV files'
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        default='data/oulad/processed/oulad_ml.parquet',
        help='Output path for processed parquet file'
    )
    
    args = parser.parse_args()
    
    try:
        dataset, group_counts = build_oulad_dataset(args.raw_dir, args.output)
        logger.info("OULAD dataset building completed successfully!")
        logger.info(f"Dataset shape: {dataset.shape}")
        if not group_counts.empty:
            logger.info(f"Group counts shape: {group_counts.shape}")
    except Exception as e:
        logger.error(f"Failed to build OULAD dataset: {e}")
        raise


if __name__ == '__main__':
    main()