#!/usr/bin/env python3
"""
OULAD Dataset Upload Helper Script

This script helps validate and upload OULAD dataset files to the repository.
It checks for the required files and validates their basic structure.

Usage:
    python scripts/upload_oulad_dataset.py --source-dir "C:\\Users\\MyName\\Documents\\Github\\10-Aug-25\\data\\oulad\\raw"
    python scripts/upload_oulad_dataset.py --source-dir "/path/to/your/oulad/files"
"""

import argparse
import logging
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required OULAD files
REQUIRED_FILES = [
    "studentInfo.csv",
    "studentRegistration.csv", 
    "studentAssessment.csv",
    "studentVle.csv",
    "vle.csv",
    "assessments.csv",
    "courses.csv"
]

def validate_oulad_files(source_dir: Path) -> bool:
    """Validate that all required OULAD files exist and have basic structure."""
    logger.info(f"Validating OULAD files in {source_dir}")
    
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return False
    
    missing_files = []
    for filename in REQUIRED_FILES:
        file_path = source_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            # Basic validation - check if file can be read as CSV
            try:
                df = pd.read_csv(file_path, nrows=5)  # Read just first 5 rows for validation
                logger.info(f"✓ {filename}: {len(df.columns)} columns, readable")
            except Exception as e:
                logger.error(f"✗ {filename}: Error reading file - {e}")
                return False
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False
    
    logger.info("✓ All required OULAD files found and validated")
    return True


def copy_oulad_files(source_dir: Path, target_dir: Path) -> bool:
    """Copy OULAD files from source to target directory."""
    logger.info(f"Copying OULAD files from {source_dir} to {target_dir}")
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    for filename in REQUIRED_FILES:
        source_file = source_dir / filename
        target_file = target_dir / filename
        
        try:
            shutil.copy2(source_file, target_file)
            copied_files.append(filename)
            logger.info(f"✓ Copied {filename}")
        except Exception as e:
            logger.error(f"✗ Error copying {filename}: {e}")
            return False
    
    logger.info(f"Successfully copied {len(copied_files)} files")
    return True


def generate_dataset_summary(target_dir: Path) -> None:
    """Generate a summary of the uploaded dataset."""
    logger.info("Generating dataset summary...")
    
    summary = {}
    for filename in REQUIRED_FILES:
        file_path = target_dir / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                summary[filename] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': round(file_path.stat().st_size / 1024 / 1024, 2),
                    'column_names': list(df.columns)
                }
            except Exception as e:
                summary[filename] = {'error': str(e)}
    
    # Save summary
    summary_file = target_dir / "dataset_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("OULAD Dataset Upload Summary\n")
        f.write("=" * 40 + "\n\n")
        
        total_size = 0
        total_rows = 0
        
        for filename, info in summary.items():
            f.write(f"{filename}:\n")
            if 'error' in info:
                f.write(f"  Error: {info['error']}\n")
            else:
                f.write(f"  Rows: {info['rows']:,}\n")
                f.write(f"  Columns: {info['columns']}\n")
                f.write(f"  Size: {info['size_mb']} MB\n")
                f.write(f"  Columns: {', '.join(info['column_names'][:5])}{'...' if len(info['column_names']) > 5 else ''}\n")
                total_size += info['size_mb']
                total_rows += info['rows']
            f.write("\n")
        
        f.write(f"Total dataset size: {total_size:.2f} MB\n")
        f.write(f"Total rows across all files: {total_rows:,}\n")
    
    logger.info(f"Dataset summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Upload OULAD dataset files to repository")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Source directory containing OULAD CSV files"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/oulad/raw"),
        help="Target directory in repository (default: data/oulad/raw)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files without copying"
    )
    
    args = parser.parse_args()
    
    logger.info("OULAD Dataset Upload Helper")
    logger.info("=" * 40)
    
    # Validate source files
    if not validate_oulad_files(args.source_dir):
        logger.error("Validation failed. Please check your source files.")
        return False
    
    if args.dry_run:
        logger.info("Dry run completed. Files are valid and ready for upload.")
        return True
    
    # Copy files
    if not copy_oulad_files(args.source_dir, args.target_dir):
        logger.error("Failed to copy files.")
        return False
    
    # Generate summary
    generate_dataset_summary(args.target_dir)
    
    logger.info("=" * 40)
    logger.info("✓ OULAD dataset upload completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Review the dataset summary in data/oulad/raw/dataset_summary.txt")
    logger.info("2. Run preprocessing: python scripts/preprocess_oulad.py")
    logger.info("3. Commit the files to git:")
    logger.info("   git add data/oulad/raw/*.csv")
    logger.info("   git commit -m 'Add OULAD dataset CSV files'")
    logger.info("   git push")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)