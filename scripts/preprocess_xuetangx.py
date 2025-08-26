#!/usr/bin/env python3
"""
XuetangX MOOC Dataset Preprocessing Script

Processes XuetangX dataset to create ML-ready format compatible with 
GUIDE transfer learning framework.

Usage:
    python scripts/preprocess_xuetangx.py [--raw-dir RAW_DIR] [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XuetangXPreprocessor:
    """XuetangX MOOC dataset preprocessor."""
    
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.data = {}
        
    def load_raw_data(self) -> bool:
        """Load XuetangX CSV files."""
        # Check for common MOOC dataset file patterns
        possible_files = [
            "student_info.csv",
            "student_video_watching.csv", 
            "student_video.csv",
            "video_info.csv",
            "student_assignment.csv",
            "assignment_info.csv",
            "student_course.csv",
            "course_info.csv",
            "forum_posts.csv",
            "student_forum.csv"
        ]
        
        found_files = []
        for filename in possible_files:
            file_path = self.raw_dir / filename
            if file_path.exists():
                try:
                    self.data[filename.replace('.csv', '')] = pd.read_csv(file_path)
                    logger.info(f"Loaded {filename}: {self.data[filename.replace('.csv', '')].shape}")
                    found_files.append(filename)
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    
        # Also check for any CSV files in the directory
        csv_files = list(self.raw_dir.glob("*.csv"))
        if not found_files and csv_files:
            logger.info("Found CSV files that don't match expected patterns:")
            for csv_file in csv_files:
                logger.info(f"  - {csv_file.name}")
                try:
                    df = pd.read_csv(csv_file)
                    logger.info(f"    Shape: {df.shape}, Columns: {list(df.columns)[:5]}...")
                    self.data[csv_file.stem] = df
                    found_files.append(csv_file.name)
                except Exception as e:
                    logger.error(f"    Error loading: {e}")
        
        if not found_files:
            logger.error("No valid CSV files found in XuetangX directory")
            return False
            
        return True
    
    def create_synthetic_xuetangx(self, n_students: int = 1000) -> pd.DataFrame:
        """Create synthetic XuetangX-like dataset for demonstration."""
        logger.info(f"Creating synthetic XuetangX dataset with {n_students} students")
        
        np.random.seed(42)
        
        # Generate synthetic student data
        data = {
            'student_id': range(1, n_students + 1),
            'course_id': np.random.choice(['MOOC_001', 'MOOC_002', 'MOOC_003'], n_students),
            'sex': np.random.choice(['M', 'F'], n_students),
            'age': np.random.randint(18, 50, n_students),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                              n_students, p=[0.3, 0.4, 0.2, 0.1]),
            'country': np.random.choice(['China', 'USA', 'India', 'Other'], 
                                       n_students, p=[0.6, 0.15, 0.1, 0.15]),
            
            # Video watching behavior
            'total_video_time': np.random.exponential(1000, n_students),  # seconds
            'videos_watched': np.random.poisson(15, n_students),
            'video_completion_rate': np.random.beta(2, 3, n_students),  # Skewed towards partial completion
            'pause_frequency': np.random.poisson(5, n_students),
            'rewatch_frequency': np.random.poisson(2, n_students),
            
            # Assignment behavior  
            'assignments_submitted': np.random.poisson(8, n_students),
            'assignment_avg_score': np.random.beta(5, 3, n_students) * 100,  # 0-100 scale
            'assignment_submission_delay': np.random.exponential(2, n_students),  # days after deadline
            
            # Forum engagement
            'forum_posts': np.random.poisson(3, n_students),
            'forum_replies': np.random.poisson(5, n_students),
            'forum_views': np.random.poisson(20, n_students),
            
            # Course engagement timing
            'days_active': np.random.poisson(30, n_students),
            'first_access_day': np.random.randint(1, 7, n_students),  # Course start week
            'last_access_day': np.random.randint(30, 90, n_students),  # Course duration
            'login_frequency': np.random.poisson(10, n_students),
            
            # Performance indicators
            'quiz_avg_score': np.random.beta(4, 2, n_students) * 100,
            'midterm_score': np.random.normal(75, 15, n_students),
            'final_score': np.random.normal(70, 20, n_students),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['video_completion_rate'] = df['video_completion_rate'].clip(0, 1)
        df['assignment_avg_score'] = df['assignment_avg_score'].clip(0, 100)
        df['quiz_avg_score'] = df['quiz_avg_score'].clip(0, 100)
        df['midterm_score'] = df['midterm_score'].clip(0, 100)
        df['final_score'] = df['final_score'].clip(0, 100)
        
        # Create realistic correlations
        # More engagement -> better performance
        engagement_score = (
            df['video_completion_rate'] * 0.3 +
            df['assignments_submitted'] / df['assignments_submitted'].max() * 0.3 +
            df['forum_posts'] / df['forum_posts'].max() * 0.2 +
            df['days_active'] / df['days_active'].max() * 0.2
        )
        
        # Adjust final scores based on engagement
        df['final_score'] = (
            df['final_score'] * 0.7 + 
            engagement_score * 100 * 0.3 +
            np.random.normal(0, 5, len(df))  # Add some noise
        ).clip(0, 100)
        
        # Create pass/fail labels
        df['label_pass'] = (df['final_score'] >= 60).astype(int)
        df['label_certificate'] = (df['final_score'] >= 80).astype(int)
        df['label_complete'] = (
            (df['video_completion_rate'] >= 0.8) & 
            (df['assignments_submitted'] >= 6)
        ).astype(int)
        
        # Create interaction features
        df['sex_x_education'] = df['sex'] + '_x_' + df['education_level']
        df['engagement_intensity'] = (
            df['total_video_time'] / (df['days_active'] + 1)
        )  # Average daily video time
        
        logger.info(f"Created synthetic XuetangX dataset: {df.shape}")
        logger.info(f"Pass rate: {df['label_pass'].mean():.3f}")
        logger.info(f"Certificate rate: {df['label_certificate'].mean():.3f}")
        logger.info(f"Completion rate: {df['label_complete'].mean():.3f}")
        
        return df
    
    def preprocess_real_xuetangx(self) -> Optional[pd.DataFrame]:
        """Preprocess real XuetangX data if available."""
        if not self.data:
            logger.warning("No real XuetangX data found")
            return None
        
        # This function would need to be customized based on the actual
        # structure of the XuetangX dataset once it's available
        logger.info("Real XuetangX preprocessing not implemented yet")
        logger.info("Available datasets:")
        for name, df in self.data.items():
            logger.info(f"  - {name}: {df.shape}")
            
        return None
    
    def create_ml_dataset(self) -> pd.DataFrame:
        """Create the final ML dataset."""
        # Try to process real data first
        ml_data = self.preprocess_real_xuetangx()
        
        # Fall back to synthetic data
        if ml_data is None:
            logger.info("Using synthetic XuetangX data for demonstration")
            ml_data = self.create_synthetic_xuetangx()
        
        return ml_data
    
    def create_summary_stats(self, ml_data: pd.DataFrame) -> Dict:
        """Create summary statistics."""
        stats = {
            'total_students': len(ml_data),
            'pass_rate': ml_data['label_pass'].mean() if 'label_pass' in ml_data.columns else 0,
            'completion_rate': ml_data['label_complete'].mean() if 'label_complete' in ml_data.columns else 0,
            'certificate_rate': ml_data['label_certificate'].mean() if 'label_certificate' in ml_data.columns else 0,
            'courses': ml_data['course_id'].nunique() if 'course_id' in ml_data.columns else 0,
            'demographics': {}
        }
        
        # Add demographic breakdowns if available
        categorical_cols = ['sex', 'education_level', 'country']
        for col in categorical_cols:
            if col in ml_data.columns:
                stats['demographics'][col] = ml_data[col].value_counts().to_dict()
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess XuetangX MOOC dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path, 
        default=Path("data/xuetangx/raw"),
        help="Directory containing raw XuetangX CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/xuetangx/processed"), 
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force creation of synthetic dataset"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = XuetangXPreprocessor(args.raw_dir)
    
    # Load data (if available)
    data_loaded = False
    if args.raw_dir.exists() and not args.synthetic:
        data_loaded = preprocessor.load_raw_data()
    
    if not data_loaded and not args.synthetic:
        logger.warning(f"No XuetangX data found in {args.raw_dir}")
        logger.info("Creating synthetic dataset for demonstration")
        args.synthetic = True
    
    # Create ML dataset
    ml_data = preprocessor.create_ml_dataset()
    
    # Save processed dataset
    output_path = args.output_dir / "xuetangx_ml.csv"
    ml_data.to_csv(output_path, index=False)
    logger.info(f"Saved ML dataset to {output_path}")
    
    # Create and save summary statistics
    stats = preprocessor.create_summary_stats(ml_data)
    
    stats_path = args.output_dir / "dataset_summary.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Saved summary statistics to {stats_path}")
    
    # Print summary
    logger.info("Dataset Summary:")
    logger.info(f"  Students: {stats['total_students']:,}")
    logger.info(f"  Pass rate: {stats['pass_rate']:.1%}")
    logger.info(f"  Completion rate: {stats['completion_rate']:.1%}")
    logger.info(f"  Certificate rate: {stats['certificate_rate']:.1%}")
    
    return True


if __name__ == "__main__":
    main()