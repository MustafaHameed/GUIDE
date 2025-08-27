#!/usr/bin/env python3
"""
OULAD Real Dataset Preprocessing Script

Processes the real OULAD dataset files to create the ML-ready dataset 
that's compatible with existing GUIDE transfer learning code.

Usage:
    python scripts/preprocess_oulad.py [--raw-dir RAW_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OULADPreprocessor:
    """Real OULAD dataset preprocessor."""
    
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.data = {}
        
    def load_raw_data(self) -> bool:
        """Load all OULAD CSV files."""
        required_files = [
            "studentInfo.csv",
            "studentVle.csv", 
            "vle.csv",
            "studentAssessment.csv",
            "assessments.csv",
            "studentRegistration.csv"
        ]
        
        missing_files = []
        for filename in required_files:
            file_path = self.raw_dir / filename
            if file_path.exists():
                try:
                    self.data[filename.replace('.csv', '')] = pd.read_csv(file_path)
                    logger.info(f"Loaded {filename}: {self.data[filename.replace('.csv', '')].shape}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    return False
            else:
                missing_files.append(filename)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
            
        return True
    
    def create_vle_features(self) -> pd.DataFrame:
        """Create VLE engagement features per student."""
        # Merge studentVle with vle to get activity types
        vle_data = self.data['studentVle'].merge(
            self.data['vle'][['id_site', 'activity_type', 'week_from', 'week_to']], 
            on='id_site', 
            how='left'
        )
        
        # Aggregate VLE features per student
        vle_features = vle_data.groupby(['code_module', 'code_presentation', 'id_student']).agg({
            'sum_click': ['sum', 'mean', 'max', 'std'],
            'date': ['min', 'max', 'nunique'],  # First access, last access, days active
            'id_site': 'nunique'  # Number of unique VLE items accessed
        }).reset_index()
        
        # Flatten column names
        vle_features.columns = [
            'code_module', 'code_presentation', 'id_student',
            'vle_total_clicks', 'vle_mean_clicks', 'vle_max_clicks', 'vle_std_clicks',
            'vle_first_access', 'vle_last_access', 'vle_days_active', 'vle_unique_items'
        ]
        
        # Calculate engagement duration
        vle_features['vle_engagement_duration'] = (
            vle_features['vle_last_access'] - vle_features['vle_first_access']
        )
        
        # Create early engagement features (first 4 weeks)
        early_vle = vle_data[vle_data['date'] <= 28]  # First 4 weeks
        early_features = early_vle.groupby(['code_module', 'code_presentation', 'id_student']).agg({
            'sum_click': 'sum'
        }).reset_index()
        early_features.columns = ['code_module', 'code_presentation', 'id_student', 'vle_first4_clicks']
        
        vle_features = vle_features.merge(early_features, on=['code_module', 'code_presentation', 'id_student'], how='left')
        vle_features['vle_first4_clicks'] = vle_features['vle_first4_clicks'].fillna(0)
        
        # Fill NaN values
        numeric_cols = vle_features.select_dtypes(include=[np.number]).columns
        vle_features[numeric_cols] = vle_features[numeric_cols].fillna(0)
        
        return vle_features
    
    def create_assessment_features(self) -> pd.DataFrame:
        """Create assessment performance features per student."""
        # Merge student assessments with assessment details
        assessment_data = self.data['studentAssessment'].merge(
            self.data['assessments'], on='id_assessment', how='left'
        )
        
        # The merge creates date_x (from studentAssessment) and date_y (from assessments)
        # Rename them for clarity
        assessment_data = assessment_data.rename(columns={
            'date_x': 'date_submitted',
            'date_y': 'date_deadline'
        })
        
        # Aggregate assessment features
        assessment_features = assessment_data.groupby(['code_module', 'code_presentation', 'id_student']).agg({
            'score': ['count', 'mean', 'max', 'min', 'std'],
            'date_submitted': ['min', 'max'],
            'is_banked': 'sum'
        }).reset_index()
        
        # Flatten column names
        assessment_features.columns = [
            'code_module', 'code_presentation', 'id_student',
            'assessment_count', 'assessment_mean_score', 'assessment_max_score', 
            'assessment_min_score', 'assessment_std_score',
            'assessment_first_submit', 'assessment_last_submit', 'assessment_banked_count'
        ]
        
        # Calculate submission timing features
        # Use the renamed columns for on-time calculation
        assessment_with_deadlines = assessment_data.copy()  # Already has date_submitted and date_deadline
        
        # On-time submission rate
        assessment_with_deadlines['on_time'] = (
            assessment_with_deadlines['date_submitted'] <= assessment_with_deadlines['date_deadline']
        )
        
        timing_features = assessment_with_deadlines.groupby(['code_module', 'code_presentation', 'id_student']).agg({
            'on_time': 'mean'
        }).reset_index()
        timing_features.columns = ['code_module', 'code_presentation', 'id_student', 'assessment_ontime_rate']
        
        assessment_features = assessment_features.merge(
            timing_features, on=['code_module', 'code_presentation', 'id_student'], how='left'
        )
        
        # Fill NaN values
        numeric_cols = assessment_features.select_dtypes(include=[np.number]).columns
        assessment_features[numeric_cols] = assessment_features[numeric_cols].fillna(0)
        
        return assessment_features
    
    def create_ml_dataset(self) -> pd.DataFrame:
        """Create the final ML dataset."""
        # Start with student info
        student_info = self.data['studentInfo'].copy()
        
        # Get final results from student registration
        student_registration = self.data['studentRegistration'].copy()
        
        # Merge student info with registration info
        ml_data = student_info.merge(
            student_registration[['code_module', 'code_presentation', 'id_student', 'final_result']],
            on=['code_module', 'code_presentation', 'id_student'],
            how='inner'
        )
        
        # Create target variables
        ml_data['label_pass'] = (ml_data['final_result'] == 'Pass').astype(int)
        ml_data['label_fail_or_withdraw'] = (
            ml_data['final_result'].isin(['Fail', 'Withdrawn'])
        ).astype(int)
        
        # Create VLE features
        vle_features = self.create_vle_features()
        ml_data = ml_data.merge(
            vle_features, on=['code_module', 'code_presentation', 'id_student'], how='left'
        )
        
        # Create assessment features
        assessment_features = self.create_assessment_features()
        ml_data = ml_data.merge(
            assessment_features, on=['code_module', 'code_presentation', 'id_student'], how='left'
        )
        
        # Create interaction features (but only if gender column exists)
        if 'gender' in ml_data.columns:
            ml_data['sex_x_age'] = ml_data['gender'] + '_x_' + ml_data['age_band']
        else:
            logger.warning("Gender column not found, skipping sex_x_age feature creation")
        
        # Select final columns to match existing format
        final_columns = [
            'id_student', 'code_module', 'code_presentation', 
            'label_pass', 'label_fail_or_withdraw',
            'sex', 'age_band', 'highest_education', 'imd_band', 'sex_x_age',
            'studied_credits', 'num_of_prev_attempts',
            'vle_total_clicks', 'vle_mean_clicks', 'vle_max_clicks', 
            'vle_first4_clicks', 'vle_last_access', 'vle_engagement_duration', 'vle_days_active',
            'assessment_count', 'assessment_mean_score', 'assessment_last_submit', 'assessment_ontime_rate'
        ]
        
        # Keep only columns that exist
        available_columns = [col for col in final_columns if col in ml_data.columns]
        ml_data = ml_data[available_columns]
        
        # Fill remaining NaN values
        numeric_cols = ml_data.select_dtypes(include=[np.number]).columns
        ml_data[numeric_cols] = ml_data[numeric_cols].fillna(0)
        
        categorical_cols = ml_data.select_dtypes(include=[object]).columns
        ml_data[categorical_cols] = ml_data[categorical_cols].fillna('Unknown')
        
        logger.info(f"Created ML dataset: {ml_data.shape}")
        logger.info(f"Pass rate: {ml_data['label_pass'].mean():.3f}")
        
        return ml_data
    
    def create_summary_stats(self, ml_data: pd.DataFrame) -> Dict:
        """Create summary statistics."""
        stats = {
            'total_students': len(ml_data),
            'pass_rate': ml_data['label_pass'].mean(),
            'modules': ml_data['code_module'].nunique(),
            'presentations': ml_data['code_presentation'].nunique(),
            'demographics': {
                'gender': ml_data['gender'].value_counts().to_dict() if 'gender' in ml_data.columns else {},
                'age_band': ml_data['age_band'].value_counts().to_dict(),
                'education': ml_data['highest_education'].value_counts().to_dict()
            }
        }
        return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess real OULAD dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/oulad/raw"),
        help="Directory containing raw OULAD CSV files"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path,
        default=Path("data/oulad/processed"),
        help="Output directory for processed dataset"
    )
    
    args = parser.parse_args()
    
    if not args.raw_dir.exists():
        logger.error(f"Raw data directory not found: {args.raw_dir}")
        logger.info("Please run: python scripts/download_datasets.py --dataset oulad")
        return False
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = OULADPreprocessor(args.raw_dir)
    
    # Load and process data
    if not preprocessor.load_raw_data():
        logger.error("Failed to load raw data")
        return False
    
    # Create ML dataset
    ml_data = preprocessor.create_ml_dataset()
    
    # Save processed dataset
    output_path = args.output_dir / "oulad_ml.csv"
    ml_data.to_csv(output_path, index=False)
    logger.info(f"Saved ML dataset to {output_path}")
    
    # Create and save summary statistics
    stats = preprocessor.create_summary_stats(ml_data)
    
    stats_path = args.output_dir / "dataset_summary.json"
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Saved summary statistics to {stats_path}")
    
    # Print summary
    logger.info("Dataset Summary:")
    logger.info(f"  Students: {stats['total_students']:,}")
    logger.info(f"  Pass rate: {stats['pass_rate']:.1%}")
    logger.info(f"  Modules: {stats['modules']}")
    logger.info(f"  Presentations: {stats['presentations']}")
    
    return True


if __name__ == "__main__":
    main()