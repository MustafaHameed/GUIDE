#!/usr/bin/env python3
"""
OULAD Format Validator

Validates that uploaded OULAD files have the correct structure
and applies any needed format fixes.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_and_fix_oulad_format(raw_dir: Path) -> bool:
    """Validate and fix OULAD dataset format if needed."""
    
    # Expected columns for each file
    expected_columns = {
        'studentInfo.csv': ['code_module', 'code_presentation', 'id_student', 'gender', 'region', 
                           'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts', 
                           'studied_credits', 'disability'],
        'studentRegistration.csv': ['code_module', 'code_presentation', 'id_student', 'date_registration', 
                                   'date_unregistration', 'final_result'],
        'studentAssessment.csv': ['id_assessment', 'id_student', 'date', 'is_banked', 'score'],
        'studentVle.csv': ['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click'],
        'vle.csv': ['id_site', 'code_module', 'code_presentation', 'activity_type', 'week_from', 'week_to'],
        'assessments.csv': ['code_module', 'code_presentation', 'id_assessment', 'assessment_type', 'date', 'weight'],
        'courses.csv': ['code_module', 'code_presentation', 'module_presentation_length']
    }
    
    all_valid = True
    
    for filename, expected_cols in expected_columns.items():
        file_path = raw_dir / filename
        if not file_path.exists():
            logger.error(f"Missing file: {filename}")
            all_valid = False
            continue
            
        try:
            df = pd.read_csv(file_path)
            actual_cols = list(df.columns)
            
            missing_cols = set(expected_cols) - set(actual_cols)
            extra_cols = set(actual_cols) - set(expected_cols)
            
            if missing_cols:
                logger.error(f"{filename}: Missing columns: {missing_cols}")
                all_valid = False
            
            if extra_cols:
                logger.warning(f"{filename}: Extra columns (will be ignored): {extra_cols}")
            
            # Check for common column name issues and suggest fixes
            if filename == 'studentAssessment.csv' and 'date_submitted' in actual_cols and 'date' not in actual_cols:
                logger.info(f"Fixing {filename}: Renaming 'date_submitted' to 'date'")
                df = df.rename(columns={'date_submitted': 'date'})
                df[expected_cols].to_csv(file_path, index=False)
                logger.info(f"Fixed and saved {filename}")
                
            if not missing_cols:
                logger.info(f"✓ {filename}: Format validated")
                
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            all_valid = False
    
    return all_valid

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate OULAD format")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/oulad/raw"), 
                       help="Directory containing OULAD files")
    args = parser.parse_args()
    
    success = validate_and_fix_oulad_format(args.raw_dir)
    if success:
        print("✓ All OULAD files validated successfully")
    else:
        print("✗ Some files have format issues")