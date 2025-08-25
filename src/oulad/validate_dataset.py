#!/usr/bin/env python3
"""
OULAD Dataset Validation Tool

Comprehensive validation and quality assessment for processed OULAD datasets.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
import sys

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from src.logging_config import setup_logging
except ImportError:
    # Fallback if logging_config not available
    def setup_logging():
        logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class OULADValidator:
    """Validator for OULAD processed datasets."""
    
    def __init__(self, dataset_path: Path, config_path: Path = None):
        """Initialize validator.
        
        Args:
            dataset_path: Path to processed OULAD dataset
            config_path: Optional path to preprocessing configuration
        """
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.data = None
        self.config = None
        self.validation_results = {}
        
    def load_data(self) -> None:
        """Load the dataset and configuration."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        # Load dataset
        if self.dataset_path.suffix == '.parquet':
            self.data = pd.read_parquet(self.dataset_path)
        elif self.dataset_path.suffix == '.csv':
            self.data = pd.read_csv(self.dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {self.dataset_path.suffix}")
        
        logger.info(f"Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Load configuration if provided
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
    
    def validate_basic_structure(self) -> Dict[str, Any]:
        """Validate basic dataset structure."""
        logger.info("Validating basic structure...")
        
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check if dataset is empty
        if len(self.data) == 0:
            results["errors"].append("Dataset is empty")
            results["passed"] = False
            
        # Check for required columns
        required_cols = ["id_student", "code_module", "code_presentation", "label_pass"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            results["errors"].append(f"Missing required columns: {missing_cols}")
            results["passed"] = False
        
        # Record basic info
        results["info"] = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict()
        }
        
        return results
    
    def validate_identifiers(self) -> Dict[str, Any]:
        """Validate student and course identifiers."""
        logger.info("Validating identifiers...")
        
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        try:
            # Check for missing student IDs
            missing_students = self.data['id_student'].isna().sum()
            if missing_students > 0:
                results["errors"].append(f"Missing student IDs: {missing_students}")
                results["passed"] = False
            
            # Check for missing module codes
            missing_modules = self.data['code_module'].isna().sum()
            if missing_modules > 0:
                results["errors"].append(f"Missing module codes: {missing_modules}")
                results["passed"] = False
            
            # Check for missing presentations
            missing_presentations = self.data['code_presentation'].isna().sum()
            if missing_presentations > 0:
                results["errors"].append(f"Missing presentations: {missing_presentations}")
                results["passed"] = False
            
            # Check for duplicates
            key_cols = ['id_student', 'code_module', 'code_presentation']
            duplicates = self.data.duplicated(subset=key_cols).sum()
            if duplicates > 0:
                results["warnings"].append(f"Duplicate records: {duplicates}")
            
            # Record statistics
            results["info"] = {
                "unique_students": self.data['id_student'].nunique(),
                "unique_modules": self.data['code_module'].nunique(),
                "unique_presentations": self.data['code_presentation'].nunique(),
                "total_records": len(self.data)
            }
            
        except KeyError as e:
            results["errors"].append(f"Missing column for identifier validation: {e}")
            results["passed"] = False
        
        return results
    
    def validate_labels(self) -> Dict[str, Any]:
        """Validate target labels."""
        logger.info("Validating labels...")
        
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        try:
            # Check primary label
            if 'label_pass' in self.data.columns:
                label_col = self.data['label_pass']
                
                # Check for missing labels
                missing_labels = label_col.isna().sum()
                if missing_labels > 0:
                    results["errors"].append(f"Missing labels: {missing_labels}")
                    results["passed"] = False
                
                # Check label values
                unique_labels = set(label_col.dropna().unique())
                expected_labels = {0, 1}
                
                if not unique_labels.issubset(expected_labels):
                    unexpected = unique_labels - expected_labels
                    results["errors"].append(f"Unexpected label values: {unexpected}")
                    results["passed"] = False
                
                # Check label distribution
                label_dist = label_col.value_counts()
                pass_rate = label_dist.get(1, 0) / len(label_col) if len(label_col) > 0 else 0
                
                if pass_rate < 0.1 or pass_rate > 0.9:
                    results["warnings"].append(f"Extreme label distribution: {pass_rate:.3f} pass rate")
                
                results["info"]["label_distribution"] = label_dist.to_dict()
                results["info"]["pass_rate"] = pass_rate
                
            else:
                results["errors"].append("Primary label 'label_pass' not found")
                results["passed"] = False
                
        except Exception as e:
            results["errors"].append(f"Label validation error: {e}")
            results["passed"] = False
        
        return results
    
    def validate_features(self) -> Dict[str, Any]:
        """Validate feature columns."""
        logger.info("Validating features...")
        
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Expected feature patterns
        vle_features = [col for col in self.data.columns if col.startswith('vle_')]
        assessment_features = [col for col in self.data.columns if col.startswith('assessment_')]
        
        # Check for infinite values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        infinite_cols = []
        
        for col in numeric_cols:
            if np.isinf(self.data[col]).any():
                infinite_cols.append(col)
        
        if infinite_cols:
            results["warnings"].append(f"Columns with infinite values: {infinite_cols}")
        
        # Check for excessive missing data
        high_missing_cols = []
        missing_threshold = 0.8
        
        for col in self.data.columns:
            missing_rate = self.data[col].isna().sum() / len(self.data)
            if missing_rate > missing_threshold:
                high_missing_cols.append((col, missing_rate))
        
        if high_missing_cols:
            results["warnings"].append(f"High missing data: {high_missing_cols}")
        
        # Feature statistics
        results["info"] = {
            "vle_features_count": len(vle_features),
            "assessment_features_count": len(assessment_features),
            "total_features": len(self.data.columns),
            "numeric_features": len(numeric_cols),
            "infinite_values_found": len(infinite_cols) > 0,
            "high_missing_columns": len(high_missing_cols)
        }
        
        return results
    
    def validate_sensitive_attributes(self) -> Dict[str, Any]:
        """Validate sensitive attributes for fairness analysis."""
        logger.info("Validating sensitive attributes...")
        
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        expected_sensitive_attrs = ['sex', 'age_band', 'highest_education', 'imd_band']
        found_attrs = [attr for attr in expected_sensitive_attrs if attr in self.data.columns]
        
        if not found_attrs:
            results["warnings"].append("No sensitive attributes found")
            return results
        
        # Analyze each sensitive attribute
        attr_stats = {}
        
        for attr in found_attrs:
            col_data = self.data[attr]
            
            # Check completeness
            missing_rate = col_data.isna().sum() / len(col_data)
            
            # Check group sizes
            value_counts = col_data.value_counts()
            min_group_size = value_counts.min() if len(value_counts) > 0 else 0
            
            # Warn about small groups
            if min_group_size < 30:
                results["warnings"].append(f"Small groups in {attr}: min size {min_group_size}")
            
            # Warn about high missing rate
            if missing_rate > 0.3:
                results["warnings"].append(f"High missing rate in {attr}: {missing_rate:.3f}")
            
            attr_stats[attr] = {
                "missing_rate": missing_rate,
                "unique_values": len(value_counts),
                "min_group_size": min_group_size,
                "max_group_size": value_counts.max() if len(value_counts) > 0 else 0,
                "value_distribution": value_counts.to_dict()
            }
        
        results["info"] = {
            "found_attributes": found_attrs,
            "missing_attributes": [attr for attr in expected_sensitive_attrs if attr not in found_attrs],
            "attribute_statistics": attr_stats
        }
        
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("Running full validation suite...")
        
        self.load_data()
        
        validation_suite = {
            "basic_structure": self.validate_basic_structure(),
            "identifiers": self.validate_identifiers(),
            "labels": self.validate_labels(),
            "features": self.validate_features(),
            "sensitive_attributes": self.validate_sensitive_attributes()
        }
        
        # Overall assessment
        all_passed = all(result["passed"] for result in validation_suite.values())
        total_errors = sum(len(result["errors"]) for result in validation_suite.values())
        total_warnings = sum(len(result["warnings"]) for result in validation_suite.values())
        
        validation_suite["summary"] = {
            "overall_passed": all_passed,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "dataset_path": str(self.dataset_path),
            "validation_timestamp": pd.Timestamp.now().isoformat()
        }
        
        return validation_suite
    
    def generate_report(self, output_path: Path) -> None:
        """Generate a validation report."""
        results = self.run_full_validation()
        
        # Save detailed results as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("OULAD DATASET VALIDATION SUMMARY")
        print("="*60)
        
        summary = results["summary"]
        print(f"Dataset: {summary['dataset_path']}")
        print(f"Overall Status: {'✓ PASSED' if summary['overall_passed'] else '✗ FAILED'}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Warnings: {summary['total_warnings']}")
        
        # Print section summaries
        for section, result in results.items():
            if section == "summary":
                continue
                
            status = "✓" if result["passed"] else "✗"
            print(f"\n{section.replace('_', ' ').title()}: {status}")
            
            if result["errors"]:
                for error in result["errors"]:
                    print(f"  ERROR: {error}")
            
            if result["warnings"]:
                for warning in result["warnings"][:3]:  # Limit warnings shown
                    print(f"  WARNING: {warning}")
                if len(result["warnings"]) > 3:
                    print(f"  ... and {len(result['warnings']) - 3} more warnings")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="OULAD Dataset Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate processed dataset
  python src/oulad/validate_dataset.py --input data/oulad/processed/oulad_ml.parquet
  
  # Validate with configuration
  python src/oulad/validate_dataset.py --input data/oulad/processed/oulad_ml.parquet --config configs/oulad_preprocessing.json
  
  # Generate detailed report
  python src/oulad/validate_dataset.py --input data/oulad/processed/oulad_ml.parquet --report data/oulad/reports/validation.json
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to processed OULAD dataset'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to preprocessing configuration file'
    )
    
    parser.add_argument(
        '--report',
        type=Path,
        help='Path to save detailed validation report (JSON)'
    )
    
    args = parser.parse_args()
    
    try:
        validator = OULADValidator(args.input, args.config)
        
        if args.report:
            # Generate detailed report
            args.report.parent.mkdir(parents=True, exist_ok=True)
            validator.generate_report(args.report)
        else:
            # Run validation and print summary
            results = validator.run_full_validation()
            
            summary = results["summary"]
            if summary["overall_passed"]:
                print("✓ Validation passed")
            else:
                print(f"✗ Validation failed with {summary['total_errors']} errors")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()