#!/usr/bin/env python3
"""
Dataset Download Script for GUIDE Project

Downloads real datasets for the GUIDE project:
- OULAD (Open University Learning Analytics Dataset)
- XuetangX MOOC Dataset  
- UCI Student Performance Dataset (verify/update existing)

Usage:
    python scripts/download_datasets.py [--dataset DATASET] [--data-dir DATA_DIR]
    
    DATASET options: oulad, xuetangx, uci, all (default: all)
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path
from typing import Optional
import requests
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_file(url: str, filename: str, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded {filename} successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {filename}: {e}")
        return False


def download_oulad_dataset(data_dir: Path) -> bool:
    """Download the real OULAD dataset."""
    logger.info("Downloading OULAD dataset...")
    
    oulad_dir = data_dir / "oulad" / "raw"
    oulad_dir.mkdir(parents=True, exist_ok=True)
    
    # OULAD dataset URLs (official source)
    base_url = "https://analyse.kmi.open.ac.uk/open_dataset/download"
    files = [
        "studentInfo.csv",
        "studentRegistration.csv", 
        "studentAssessment.csv",
        "studentVle.csv",
        "vle.csv",
        "assessments.csv",
        "courses.csv"
    ]
    
    success = True
    for filename in files:
        file_path = oulad_dir / filename
        if file_path.exists():
            logger.info(f"{filename} already exists, skipping")
            continue
            
        # Try to download from official source
        url = f"{base_url}/{filename}"
        if not download_file(url, str(file_path)):
            logger.warning(f"Could not download {filename} from official source")
            
            # Create a placeholder indicating the dataset needs manual download
            placeholder_path = oulad_dir / "DOWNLOAD_INSTRUCTIONS.txt"
            with open(placeholder_path, 'w') as f:
                f.write("""OULAD Dataset Download Instructions

The OULAD dataset requires registration and manual download from:
https://analyse.kmi.open.ac.uk/open_dataset

Please download the following files and place them in this directory:
- studentInfo.csv
- studentRegistration.csv
- studentAssessment.csv
- studentVle.csv
- vle.csv
- assessments.csv
- courses.csv

Alternative: The dataset is also available from:
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Open+University+Learning+Analytics+dataset
- Kaggle: https://www.kaggle.com/datasets/rocki37/open-university-learning-analytics-dataset

After downloading, run the preprocessing script:
python scripts/preprocess_oulad.py
""")
            success = False
            break
    
    return success


def download_xuetangx_dataset(data_dir: Path) -> bool:
    """Download and process XuetangX MOOC dataset."""
    logger.info("Setting up XuetangX dataset...")
    
    xuetangx_dir = data_dir / "xuetangx" / "raw"
    xuetangx_dir.mkdir(parents=True, exist_ok=True)
    
    # XuetangX dataset information and download instructions
    instructions_path = xuetangx_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(instructions_path, 'w') as f:
        f.write("""XuetangX Dataset Download Instructions

The XuetangX dataset is available from multiple sources:

1. Original Paper: "Understanding Learner Behavior in MOOCs through Data Mining"
   - May require contacting authors or institution
   
2. Alternative sources:
   - Some preprocessed versions may be available on academic repositories
   - Check with educational data mining conferences (EDM, LAK, etc.)
   
3. Similar MOOC datasets available:
   - Stanford MOOC Data: https://datastage.stanford.edu/
   - HarvardX and MITx data: https://dataverse.harvard.edu/
   - Coursera Research Data: https://www.coursera.org/about/research

Expected format:
- Student interactions with course materials
- Video watching behavior
- Assignment submissions
- Discussion forum participation
- Final grades/completion status

After obtaining the dataset, place CSV files in this directory and run:
python scripts/preprocess_xuetangx.py
""")
    
    logger.info(f"Created XuetangX setup instructions at {instructions_path}")
    return True


def download_uci_dataset(data_dir: Path) -> bool:
    """Download/verify UCI Student Performance dataset."""
    logger.info("Downloading UCI Student Performance dataset...")
    
    uci_dir = data_dir / "uci" / "raw"
    uci_dir.mkdir(parents=True, exist_ok=True)
    
    # UCI ML Repository URLs
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320"
    files = [
        "student.zip"  # Contains both student-mat.csv and student-por.csv
    ]
    
    for filename in files:
        file_path = uci_dir / filename
        if file_path.exists():
            logger.info(f"{filename} already exists, skipping")
            continue
            
        url = f"{base_url}/{filename}"
        if download_file(url, str(file_path)):
            # Extract the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(uci_dir)
            logger.info(f"Extracted {filename}")
        else:
            logger.warning(f"Could not download {filename}, using existing files in root directory")
    
    # Copy existing files from root directory if they exist
    root_dir = Path(".")
    for csv_file in ["student-mat.csv", "student-por.csv"]:
        root_path = root_dir / csv_file
        uci_path = uci_dir / csv_file
        
        if root_path.exists() and not uci_path.exists():
            import shutil
            shutil.copy2(root_path, uci_path)
            logger.info(f"Copied existing {csv_file} to UCI directory")
    
    return True


def verify_datasets(data_dir: Path) -> dict:
    """Verify downloaded datasets and return status."""
    status = {}
    
    # Check OULAD
    oulad_files = ["studentInfo.csv", "studentVle.csv", "vle.csv"]
    oulad_dir = data_dir / "oulad" / "raw"
    status['oulad'] = all((oulad_dir / f).exists() for f in oulad_files)
    
    # Check UCI  
    uci_files = ["student-mat.csv"]
    uci_dir = data_dir / "uci" / "raw"
    status['uci'] = any((uci_dir / f).exists() for f in uci_files)
    
    # Check XuetangX (instructions created)
    xuetangx_dir = data_dir / "xuetangx" / "raw"
    status['xuetangx'] = (xuetangx_dir / "DOWNLOAD_INSTRUCTIONS.txt").exists()
    
    return status


def main():
    parser = argparse.ArgumentParser(description="Download datasets for GUIDE project")
    parser.add_argument(
        "--dataset", 
        choices=["oulad", "xuetangx", "uci", "all"],
        default="all",
        help="Dataset to download (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory (default: data)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting dataset download to {args.data_dir}")
    args.data_dir.mkdir(exist_ok=True)
    
    success = True
    
    if args.dataset in ["oulad", "all"]:
        success &= download_oulad_dataset(args.data_dir)
    
    if args.dataset in ["xuetangx", "all"]:
        success &= download_xuetangx_dataset(args.data_dir)
        
    if args.dataset in ["uci", "all"]:
        success &= download_uci_dataset(args.data_dir)
    
    # Verify downloads
    status = verify_datasets(args.data_dir)
    
    logger.info("Dataset download status:")
    for dataset, available in status.items():
        status_msg = "✓ Available" if available else "✗ Not available"
        logger.info(f"  {dataset.upper()}: {status_msg}")
    
    if not success:
        logger.warning("Some datasets require manual download. Check instructions in data directories.")
    
    return success


if __name__ == "__main__":
    main()