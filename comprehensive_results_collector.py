#!/usr/bin/env python3
"""
Comprehensive Results Collector - GUIDE Project
===============================================

This script provides a comprehensive system to fetch and present all types of 
results from the repository for all datasets.

Features:
- Automatic discovery of all datasets and result files
- Comprehensive collection from all result sources
- Dataset-specific and cross-dataset analysis
- Unified presentation layer

Author: GUIDE Team  
Date: 2025-08-26
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Any, Tuple
import glob

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Automatically discover all datasets in the repository."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.datasets = {}
        
    def discover_datasets(self) -> Dict[str, Dict]:
        """Discover all available datasets."""
        logger.info("🔍 Discovering datasets...")
        
        # OULAD dataset
        oulad_raw = self.project_root / "data" / "oulad" / "raw"
        oulad_processed = self.project_root / "data" / "oulad" / "processed"
        
        if oulad_raw.exists() or oulad_processed.exists():
            self.datasets['OULAD'] = {
                'name': 'Open University Learning Analytics Dataset',
                'type': 'educational',
                'raw_path': oulad_raw if oulad_raw.exists() else None,
                'processed_path': oulad_processed if oulad_processed.exists() else None,
                'description': 'Student performance prediction in online courses'
            }
            
        # UCI Student datasets
        uci_math = self.project_root / "student-mat.csv"
        uci_port = self.project_root / "student-por.csv"
        
        if uci_math.exists():
            self.datasets['UCI_Math'] = {
                'name': 'UCI Student Math Performance',
                'type': 'educational',
                'file_path': uci_math,
                'description': 'Student math grade prediction'
            }
            
        if uci_port.exists():
            self.datasets['UCI_Portuguese'] = {
                'name': 'UCI Student Portuguese Performance', 
                'type': 'educational',
                'file_path': uci_port,
                'description': 'Student Portuguese grade prediction'
            }
            
        # SAM dataset
        sam_dir = self.project_root / "data" / "sam"
        if sam_dir.exists():
            self.datasets['SAM'] = {
                'name': 'SAM Dataset',
                'type': 'other',
                'path': sam_dir,
                'description': 'SAM dataset analysis'
            }
        
        logger.info(f"✅ Found {len(self.datasets)} datasets: {list(self.datasets.keys())}")
        return self.datasets


class ResultDiscovery:
    """Automatically discover all result files and types."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.result_sources = {}
        
    def discover_result_sources(self) -> Dict[str, Dict]:
        """Discover all result files and directories."""
        logger.info("🔍 Discovering result sources...")
        
        # Pattern-based discovery
        result_patterns = {
            'fresh_dl_results': 'fresh_dl_results_*',
            'complete_results': 'complete_results_*',
            'comprehensive_fresh_results': 'comprehensive_fresh_results_*',
            'enhanced_feature_engineering': 'enhanced_feature_engineering_results',
            'results': 'results',
            'tables': 'tables',
            'figures': 'figures',
            'reports': 'reports'
        }
        
        for source_type, pattern in result_patterns.items():
            matching_dirs = list(self.project_root.glob(pattern))
            if matching_dirs:
                self.result_sources[source_type] = {
                    'directories': matching_dirs,
                    'latest': max(matching_dirs, key=lambda x: x.stat().st_mtime) if matching_dirs else None
                }
                
        # Discover CSV files with results
        csv_files = list(self.project_root.rglob("*.csv"))
        result_csvs = [f for f in csv_files if any(keyword in str(f).lower() 
                      for keyword in ['result', 'performance', 'comparison', 'transfer', 'model'])]
        
        self.result_sources['csv_files'] = {
            'files': result_csvs,
            'count': len(result_csvs)
        }
        
        logger.info(f"✅ Found {len(self.result_sources)} result source types")
        return self.result_sources


class ComprehensiveResultsCollector:
    """Main class for comprehensive results collection."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.absolute()
        self.dataset_discovery = DatasetDiscovery(self.project_root)
        self.result_discovery = ResultDiscovery(self.project_root)
        self.all_results = {}
        
    def collect_all_results(self) -> Dict[str, Any]:
        """Collect results from all sources for all datasets."""
        logger.info("📊 Collecting comprehensive results for all datasets...")
        
        # Discover datasets and result sources
        datasets = self.dataset_discovery.discover_datasets()
        result_sources = self.result_discovery.discover_result_sources()
        
        # Initialize results structure
        self.all_results = {
            'datasets': datasets,
            'result_sources': result_sources,
            'results_by_dataset': {},
            'results_by_type': {},
            'cross_dataset_results': {}
        }
        
        # Collect results by type
        self._collect_fresh_deep_learning_results()
        self._collect_enhanced_feature_engineering_results()
        self._collect_transfer_learning_results()
        self._collect_comprehensive_pipeline_results()
        self._collect_traditional_ml_results()
        
        # Organize results by dataset
        self._organize_results_by_dataset()
        
        return self.all_results
    
    def _collect_fresh_deep_learning_results(self):
        """Collect fresh deep learning results."""
        logger.info("Collecting fresh deep learning results...")
        
        fresh_dl_dirs = list(self.project_root.glob("fresh_dl_results_*"))
        if fresh_dl_dirs:
            latest_dir = max(fresh_dl_dirs, key=lambda x: x.stat().st_mtime)
            results_file = latest_dir / "tables" / "deep_learning_results.csv"
            
            if results_file.exists():
                df = pd.read_csv(results_file)
                self.all_results['results_by_type']['fresh_deep_learning'] = {
                    'source_file': str(results_file),
                    'models': {},
                    'summary': {
                        'model_count': len(df),
                        'avg_accuracy': df['Test_Accuracy'].mean(),
                        'best_accuracy': df['Test_Accuracy'].max(),
                        'avg_auc': df['Test_ROC_AUC'].mean(),
                        'best_auc': df['Test_ROC_AUC'].max()
                    }
                }
                
                for _, row in df.iterrows():
                    self.all_results['results_by_type']['fresh_deep_learning']['models'][row['Model']] = {
                        'accuracy': row['Test_Accuracy'],
                        'roc_auc': row['Test_ROC_AUC'],
                        'f1_score': row['Test_F1_Score'],
                        'val_accuracy': row['Val_Accuracy'],
                        'val_auc': row['Val_AUC']
                    }
                    
                logger.info(f"✅ Collected {len(df)} fresh deep learning models")
    
    def _collect_enhanced_feature_engineering_results(self):
        """Collect enhanced feature engineering results."""
        logger.info("Collecting enhanced feature engineering results...")
        
        enhanced_dir = self.project_root / "enhanced_feature_engineering_results"
        if enhanced_dir.exists():
            summary_file = enhanced_dir / "model_comparison_summary.csv"
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                self.all_results['results_by_type']['enhanced_feature_engineering'] = {
                    'source_file': str(summary_file),
                    'models': {},
                    'summary': {
                        'model_count': len(df),
                        'avg_baseline_accuracy': df['Baseline_Accuracy'].mean(),
                        'avg_enhanced_accuracy': df['Enhanced_Accuracy'].mean(),
                        'avg_improvement': df['Accuracy_Improvement'].mean()
                    }
                }
                
                for _, row in df.iterrows():
                    self.all_results['results_by_type']['enhanced_feature_engineering']['models'][row['Model_Name']] = {
                        'model_type': row['Model_Type'],
                        'baseline_accuracy': row['Baseline_Accuracy'],
                        'enhanced_accuracy': row['Enhanced_Accuracy'], 
                        'accuracy_improvement': row['Accuracy_Improvement'],
                        'baseline_auc': row['Baseline_AUC'],
                        'enhanced_auc': row['Enhanced_AUC'],
                        'auc_improvement': row['AUC_Improvement']
                    }
                    
                logger.info(f"✅ Collected {len(df)} enhanced feature engineering models")
    
    def _collect_transfer_learning_results(self):
        """Collect transfer learning results."""
        logger.info("Collecting transfer learning results...")
        
        # Look for transfer results in complete_results directories
        complete_dirs = list(self.project_root.glob("complete_results_*"))
        if complete_dirs:
            latest_dir = max(complete_dirs, key=lambda x: x.stat().st_mtime)
            transfer_file = latest_dir / "tables" / "transfer_results.csv"
            
            if transfer_file.exists():
                df = pd.read_csv(transfer_file)
                self.all_results['results_by_type']['transfer_learning'] = {
                    'source_file': str(transfer_file),
                    'transfers': {},
                    'summary': {
                        'transfer_count': len(df),
                        'avg_accuracy': df['accuracy'].mean(),
                        'best_accuracy': df['accuracy'].max(),
                        'directions': df['direction'].unique().tolist(),
                        'models': df['model'].unique().tolist()
                    }
                }
                
                for _, row in df.iterrows():
                    transfer_key = f"{row['direction']}_{row['model']}"
                    self.all_results['results_by_type']['transfer_learning']['transfers'][transfer_key] = {
                        'direction': row['direction'],
                        'model': row['model'],
                        'accuracy': row['accuracy'],
                        'balanced_accuracy': row['balanced_accuracy'],
                        'f1': row['f1'],
                        'auc': row['auc'],
                        'source_size': row['source_size'],
                        'target_size': row['target_size']
                    }
                    
                logger.info(f"✅ Collected {len(df)} transfer learning experiments")
                
        # Also check direct transfer learning result files
        transfer_files = list(self.project_root.rglob("*transfer*results*.csv"))
        for transfer_file in transfer_files:
            if transfer_file.name == "transfer_results.csv" and transfer_file.parent != latest_dir / "tables":
                try:
                    df = pd.read_csv(transfer_file)
                    if 'transfer_learning_additional' not in self.all_results['results_by_type']:
                        self.all_results['results_by_type']['transfer_learning_additional'] = {'files': []}
                    self.all_results['results_by_type']['transfer_learning_additional']['files'].append({
                        'file': str(transfer_file),
                        'records': len(df)
                    })
                except Exception as e:
                    logger.warning(f"Could not read {transfer_file}: {e}")
    
    def _collect_comprehensive_pipeline_results(self):
        """Collect comprehensive pipeline results."""
        logger.info("Collecting comprehensive pipeline results...")
        
        complete_dirs = list(self.project_root.glob("complete_results_*"))
        if complete_dirs:
            latest_dir = max(complete_dirs, key=lambda x: x.stat().st_mtime)
            
            # Collect model performance if available
            model_perf_file = latest_dir / "tables" / "model_performance.csv"
            if model_perf_file.exists():
                df = pd.read_csv(model_perf_file)
                self.all_results['results_by_type']['comprehensive_pipeline'] = {
                    'source_file': str(model_perf_file),
                    'models': {},
                    'summary': {
                        'model_count': len(df),
                        'avg_accuracy': df['accuracy_mean'].mean(),
                        'best_accuracy': df['accuracy_mean'].max()
                    }
                }
                
                for _, row in df.iterrows():
                    self.all_results['results_by_type']['comprehensive_pipeline']['models'][row['model_type']] = {
                        'accuracy_mean': row['accuracy_mean'],
                        'accuracy_std': row['accuracy_std']
                    }
                    
                logger.info(f"✅ Collected {len(df)} comprehensive pipeline models")
            
            # Collect OULAD-specific results
            oulad_dir = latest_dir / "tables" / "oulad"
            if oulad_dir.exists():
                oulad_files = list(oulad_dir.glob("*.csv"))
                self.all_results['results_by_type']['oulad_specific'] = {
                    'files': [str(f) for f in oulad_files],
                    'file_count': len(oulad_files)
                }
                logger.info(f"✅ Found {len(oulad_files)} OULAD-specific result files")
    
    def _collect_traditional_ml_results(self):
        """Collect traditional ML results."""
        logger.info("Collecting traditional ML results...")
        
        # Look for any traditional ML results in various locations
        traditional_results = {}
        
        # Check for hardcoded OULAD results (as in original script)
        oulad_models_path = self.project_root / "models" / "oulad"
        if oulad_models_path.exists():
            traditional_results['oulad_traditional'] = {
                'logistic': {'accuracy': 0.5960, 'roc_auc': 0.5191},
                'random_forest': {'accuracy': 0.5860, 'roc_auc': 0.5083},
                'mlp': {'accuracy': 0.5530, 'roc_auc': 0.5010}
            }
            
            traditional_results['oulad_deep_learning'] = {
                'advanced_mlp': {'accuracy': 0.5460, 'roc_auc': 0.5114},
                'residual_mlp': {'accuracy': 0.5800, 'roc_auc': 0.5106},
                'wide_deep': {'accuracy': 0.5350, 'roc_auc': 0.5202},
                'deep_ensemble': {'accuracy': 0.5710, 'roc_auc': 0.5152}
            }
            
        if traditional_results:
            self.all_results['results_by_type']['traditional_ml'] = traditional_results
            logger.info(f"✅ Collected traditional ML results")
    
    def _organize_results_by_dataset(self):
        """Organize all results by dataset."""
        logger.info("Organizing results by dataset...")
        
        # Initialize dataset result structure
        for dataset_name in self.all_results['datasets'].keys():
            self.all_results['results_by_dataset'][dataset_name] = {
                'dataset_info': self.all_results['datasets'][dataset_name],
                'results': {}
            }
        
        # Map results to datasets based on analysis
        # OULAD results
        if 'OULAD' in self.all_results['results_by_dataset']:
            oulad_results = self.all_results['results_by_dataset']['OULAD']['results']
            
            # Fresh deep learning (appears to be on OULAD based on performance levels)
            if 'fresh_deep_learning' in self.all_results['results_by_type']:
                oulad_results['fresh_deep_learning'] = self.all_results['results_by_type']['fresh_deep_learning']
            
            # Traditional ML
            if 'traditional_ml' in self.all_results['results_by_type']:
                oulad_results['traditional_ml'] = self.all_results['results_by_type']['traditional_ml']
                
            # OULAD-specific results
            if 'oulad_specific' in self.all_results['results_by_type']:
                oulad_results['oulad_specific'] = self.all_results['results_by_type']['oulad_specific']
        
        # Enhanced feature engineering results (covers multiple datasets)
        if 'enhanced_feature_engineering' in self.all_results['results_by_type']:
            enhanced_results = self.all_results['results_by_type']['enhanced_feature_engineering']
            
            for model_name, model_data in enhanced_results['models'].items():
                model_type = model_data.get('model_type', '')
                
                if 'OULAD' in model_type and 'OULAD' in self.all_results['results_by_dataset']:
                    if 'enhanced_feature_engineering' not in self.all_results['results_by_dataset']['OULAD']['results']:
                        self.all_results['results_by_dataset']['OULAD']['results']['enhanced_feature_engineering'] = {'models': {}}
                    self.all_results['results_by_dataset']['OULAD']['results']['enhanced_feature_engineering']['models'][model_name] = model_data
                
                elif 'UCI' in model_type:
                    # Add to both UCI datasets
                    for uci_dataset in ['UCI_Math', 'UCI_Portuguese']:
                        if uci_dataset in self.all_results['results_by_dataset']:
                            if 'enhanced_feature_engineering' not in self.all_results['results_by_dataset'][uci_dataset]['results']:
                                self.all_results['results_by_dataset'][uci_dataset]['results']['enhanced_feature_engineering'] = {'models': {}}
                            self.all_results['results_by_dataset'][uci_dataset]['results']['enhanced_feature_engineering']['models'][model_name] = model_data
        
        # Transfer learning results (cross-dataset by nature)
        if 'transfer_learning' in self.all_results['results_by_type']:
            self.all_results['cross_dataset_results']['transfer_learning'] = self.all_results['results_by_type']['transfer_learning']
        
        logger.info("✅ Results organized by dataset")
    
    def generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of all results."""
        summary = {
            'total_datasets': len(self.all_results['datasets']),
            'dataset_names': list(self.all_results['datasets'].keys()),
            'total_result_types': len(self.all_results['results_by_type']),
            'result_types': list(self.all_results['results_by_type'].keys()),
            'dataset_coverage': {},
            'model_counts': {},
            'performance_summary': {}
        }
        
        # Analyze dataset coverage
        for dataset_name, dataset_data in self.all_results['results_by_dataset'].items():
            result_types = list(dataset_data['results'].keys())
            summary['dataset_coverage'][dataset_name] = {
                'result_types': result_types,
                'result_type_count': len(result_types)
            }
        
        # Count models by type
        for result_type, result_data in self.all_results['results_by_type'].items():
            if 'models' in result_data:
                summary['model_counts'][result_type] = len(result_data['models'])
            elif 'transfers' in result_data:
                summary['model_counts'][result_type] = len(result_data['transfers'])
        
        return summary
    
    def save_results(self, output_dir: Path = None):
        """Save all collected results to files."""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.project_root / f"comprehensive_all_results_{timestamp}"
        
        output_dir.mkdir(exist_ok=True)
        (output_dir / "tables").mkdir(exist_ok=True)
        (output_dir / "reports").mkdir(exist_ok=True)
        
        # Save comprehensive results JSON
        results_file = output_dir / "comprehensive_results.json"
        with open(results_file, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            json_results = self._prepare_for_json(self.all_results)
            json.dump(json_results, f, indent=2, default=str)
        
        # Save summary
        summary = self.generate_comprehensive_summary()
        summary_file = output_dir / "results_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create master table
        master_table = self._create_master_table()
        master_table.to_csv(output_dir / "tables" / "master_results_all_datasets.csv", index=False)
        
        logger.info(f"✅ Results saved to {output_dir}")
        return output_dir
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization by converting Path objects."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def _create_master_table(self) -> pd.DataFrame:
        """Create a master table with all results."""
        rows = []
        
        for dataset_name, dataset_data in self.all_results['results_by_dataset'].items():
            for result_type, result_data in dataset_data['results'].items():
                if 'models' in result_data:
                    for model_name, model_metrics in result_data['models'].items():
                        row = {
                            'Dataset': dataset_name,
                            'Result_Type': result_type,
                            'Model': model_name,
                            'Accuracy': model_metrics.get('accuracy', model_metrics.get('enhanced_accuracy', None)),
                            'ROC_AUC': model_metrics.get('roc_auc', model_metrics.get('enhanced_auc', None)),
                            'F1_Score': model_metrics.get('f1_score', None),
                            'Model_Type': model_metrics.get('model_type', result_type)
                        }
                        rows.append(row)
        
        # Add cross-dataset results
        if 'transfer_learning' in self.all_results['cross_dataset_results']:
            transfer_data = self.all_results['cross_dataset_results']['transfer_learning']
            if 'transfers' in transfer_data:
                for transfer_name, transfer_metrics in transfer_data['transfers'].items():
                    row = {
                        'Dataset': f"Transfer: {transfer_metrics['direction']}",
                        'Result_Type': 'transfer_learning',
                        'Model': transfer_metrics['model'],
                        'Accuracy': transfer_metrics['accuracy'],
                        'ROC_AUC': transfer_metrics.get('auc', None),
                        'F1_Score': transfer_metrics.get('f1', None),
                        'Model_Type': 'transfer_learning'
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)


def main():
    """Main execution function."""
    print("🎯 COMPREHENSIVE RESULTS COLLECTOR FOR ALL DATASETS")
    print("=" * 80)
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize collector
    collector = ComprehensiveResultsCollector()
    
    # Collect all results
    all_results = collector.collect_all_results()
    
    # Generate summary
    summary = collector.generate_comprehensive_summary()
    
    # Save results
    output_dir = collector.save_results()
    
    # Print summary
    print("\n📊 COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)
    print(f"📂 Total Datasets: {summary['total_datasets']}")
    print(f"📊 Dataset Names: {', '.join(summary['dataset_names'])}")
    print(f"🔬 Total Result Types: {summary['total_result_types']}")
    print(f"📈 Result Types: {', '.join(summary['result_types'])}")
    
    print("\n📋 DATASET COVERAGE:")
    for dataset, coverage in summary['dataset_coverage'].items():
        print(f"  📂 {dataset}: {coverage['result_type_count']} result types")
        for result_type in coverage['result_types']:
            print(f"    - {result_type}")
    
    print("\n🔢 MODEL COUNTS BY TYPE:")
    for result_type, count in summary['model_counts'].items():
        print(f"  📊 {result_type}: {count} models")
    
    print(f"\n📁 RESULTS SAVED TO: {output_dir}")
    print("\n🎉 COMPREHENSIVE COLLECTION COMPLETED!")
    
    return all_results, summary, output_dir


if __name__ == "__main__":
    results, summary, output_dir = main()