#!/usr/bin/env python3
"""
Multi-Dataset Transfer Learning Demo

Demonstrates transfer learning between all three datasets:
- OULAD (Open University Learning Analytics)
- XuetangX (MOOC dataset)  
- UCI (Student Performance)

Usage:
    python demo_multi_dataset_transfer.py [--output-dir OUTPUT_DIR]
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Import enhanced data loader
try:
    from scripts.enhanced_data_loader import load_dataset, load_all_datasets
    ENHANCED_LOADER_AVAILABLE = True
except ImportError:
    from enhanced_transfer_learning_quickwins import load_oulad_data, load_uci_data, load_xuetangx_data
    ENHANCED_LOADER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiDatasetTransferLearning:
    """Transfer learning framework for multiple educational datasets."""
    
    def __init__(self):
        self.datasets = {}
        self.preprocessors = {}
        self.results = {}
        
    def load_all_datasets(self) -> bool:
        """Load all available datasets."""
        logger.info("Loading all datasets...")
        
        if ENHANCED_LOADER_AVAILABLE:
            try:
                self.datasets = load_all_datasets()
                return len(self.datasets) > 0
            except Exception as e:
                logger.warning(f"Enhanced loader failed: {e}")
        
        # Fall back to individual loading
        try:
            self.datasets['oulad'] = load_oulad_data()
        except Exception as e:
            logger.warning(f"Could not load OULAD: {e}")
            
        try:
            self.datasets['uci'] = load_uci_data()
        except Exception as e:
            logger.warning(f"Could not load UCI: {e}")
            
        try:
            self.datasets['xuetangx'] = load_xuetangx_data()
        except Exception as e:
            logger.warning(f"Could not load XuetangX: {e}")
        
        logger.info(f"Loaded {len(self.datasets)} datasets: {list(self.datasets.keys())}")
        return len(self.datasets) > 0
    
    def create_common_features(self, dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Create common features across datasets."""
        processed_df = df.copy()
        
        # Ensure we have a binary target
        if 'label_pass' not in processed_df.columns:
            logger.warning(f"No label_pass in {dataset_name}, creating from available targets")
            if 'G3' in processed_df.columns:
                processed_df['label_pass'] = (processed_df['G3'] >= 10).astype(int)
            elif 'final_score' in processed_df.columns:
                processed_df['label_pass'] = (processed_df['final_score'] >= 60).astype(int)
            elif 'final_result' in processed_df.columns:
                processed_df['label_pass'] = (processed_df['final_result'] == 'Pass').astype(int)
            else:
                logger.error(f"Cannot create target for {dataset_name}")
                return None
        
        # Create common features
        common_features = []
        
        # Gender (standardize naming and values)
        if 'sex' in processed_df.columns:
            # Standardize gender values to M/F
            gender_mapping = {'F': 'F', 'M': 'M', 'Female': 'F', 'Male': 'M'}
            processed_df['gender'] = processed_df['sex'].map(gender_mapping).fillna('F')
            common_features.append('gender')
        
        # Age (normalize different age representations)
        if 'age' in processed_df.columns:
            processed_df['age_normalized'] = processed_df['age']
            common_features.append('age_normalized')
        elif 'age_band' in processed_df.columns:
            # Convert age bands to numeric
            age_mapping = {'0-35': 25, '35-55': 45, '55<=': 60}
            processed_df['age_normalized'] = processed_df['age_band'].map(age_mapping).fillna(25)
            common_features.append('age_normalized')
        
        # Educational background - standardize to numeric scale
        if 'highest_education' in processed_df.columns:
            # Map OULAD education strings to numeric scale
            education_mapping = {
                'No Formal quals': 0,
                'Lower Than A Level': 1, 
                'A Level or Equivalent': 2,
                'HE Qualification': 3,
                'Post Graduate Qualification': 4
            }
            processed_df['education'] = processed_df['highest_education'].map(education_mapping).fillna(1)
            common_features.append('education')
        elif 'education_level' in processed_df.columns:
            # Map synthetic education levels to numeric
            education_mapping = {
                'Primary': 0,
                'Secondary': 1,
                'High School': 2,
                'Bachelor': 3,
                'Master': 4,
                'PhD': 5
            }
            processed_df['education'] = processed_df['education_level'].map(education_mapping).fillna(2)
            common_features.append('education')
        elif 'Medu' in processed_df.columns or 'Fedu' in processed_df.columns:
            # Use parent education as proxy (already numeric)
            medu = processed_df.get('Medu', 0)
            fedu = processed_df.get('Fedu', 0)
            processed_df['education'] = np.maximum(medu, fedu)
            common_features.append('education')
        
        # Engagement/activity metrics
        if 'vle_total_clicks' in processed_df.columns:
            processed_df['engagement_score'] = processed_df['vle_total_clicks']
            common_features.append('engagement_score')
        elif 'total_video_time' in processed_df.columns:
            processed_df['engagement_score'] = processed_df['total_video_time']
            common_features.append('engagement_score')
        elif 'absences' in processed_df.columns:
            # Invert absences to make it an engagement score
            processed_df['engagement_score'] = processed_df['absences'].max() - processed_df['absences']
            common_features.append('engagement_score')
        
        # Academic performance indicators
        if 'G1' in processed_df.columns:
            processed_df['prior_performance'] = processed_df['G1']
            common_features.append('prior_performance')
        elif 'assignment_mean_score' in processed_df.columns:
            processed_df['prior_performance'] = processed_df['assignment_mean_score']
            common_features.append('prior_performance')
        elif 'assignment_avg_score' in processed_df.columns:
            processed_df['prior_performance'] = processed_df['assignment_avg_score']
            common_features.append('prior_performance')
        
        # Create interaction features
        if 'gender' in common_features and 'age_normalized' in common_features:
            processed_df['gender_x_age'] = processed_df['gender'] + '_x_' + pd.cut(
                processed_df['age_normalized'], bins=[0, 25, 35, 100], labels=['young', 'middle', 'older']
            ).astype(str)
            common_features.append('gender_x_age')
        
        # Select final features
        final_features = ['label_pass'] + [f for f in common_features if f in processed_df.columns]
        processed_df = processed_df[final_features]
        
        logger.info(f"{dataset_name} processed features: {len(final_features)-1} features, {len(processed_df)} samples")
        return processed_df
    
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create a preprocessing pipeline for features."""
        # Handle missing values first
        X_clean = X.copy()
        
        # Identify truly numeric vs categorical columns more carefully
        numeric_features = []
        categorical_features = []
        
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object':
                # Definitely categorical
                categorical_features.append(col)
            else:
                # Check if numeric column contains only integers that might be categorical
                # (e.g., education levels coded as 1,2,3,4)
                unique_vals = X_clean[col].dropna().unique()
                if len(unique_vals) <= 10 and all(isinstance(x, (int, np.integer)) or x.is_integer() for x in unique_vals if pd.notna(x)):
                    # Likely categorical (few integer values)
                    categorical_features.append(col)
                else:
                    # Truly numeric
                    numeric_features.append(col)
        
        # Fill missing values appropriately
        for col in X_clean.columns:
            if col in categorical_features:
                X_clean[col] = X_clean[col].fillna('Unknown')
                # Convert to string to ensure consistent handling
                X_clean[col] = X_clean[col].astype(str)
            else:
                # Only use median for truly numeric columns
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0  # fallback
                X_clean[col] = X_clean[col].fillna(median_val)
        
        from sklearn.impute import SimpleImputer
        
        transformers = []
        if numeric_features:
            transformers.append(('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features))
            
        if categorical_features:
            transformers.append(('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features))
        
        if not transformers:
            # Fallback if no features detected
            transformers = [('passthrough', 'passthrough', list(X_clean.columns))]
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        return preprocessor
    
    def run_transfer_experiment(self, source_name: str, target_name: str, 
                               model_type: str = 'logistic') -> Dict:
        """Run transfer learning experiment between two datasets."""
        logger.info(f"Running transfer: {source_name} → {target_name}")
        
        # Get processed datasets
        source_df = self.create_common_features(source_name, self.datasets[source_name])
        target_df = self.create_common_features(target_name, self.datasets[target_name])
        
        if source_df is None or target_df is None:
            logger.error(f"Failed to process datasets for {source_name} → {target_name}")
            return {}
        
        # Align features
        common_cols = set(source_df.columns) & set(target_df.columns)
        common_cols.discard('label_pass')
        common_features = list(common_cols)
        
        if not common_features:
            logger.warning(f"No common features for {source_name} → {target_name}")
            return {'error': 'No common features'}
        
        logger.info(f"Using {len(common_features)} common features: {common_features}")
        
        # Prepare data with missing value handling
        X_source = source_df[common_features].copy()
        y_source = source_df['label_pass']
        X_target = target_df[common_features].copy()
        y_target = target_df['label_pass']
        
        # Fill missing values
        for col in common_features:
            if X_source[col].dtype == 'object' or X_target[col].dtype == 'object':
                # Handle categorical columns
                X_source[col] = X_source[col].fillna('Unknown')
                X_target[col] = X_target[col].fillna('Unknown')
            else:
                # Handle numeric columns - use source median for both datasets to maintain consistency
                fill_value = X_source[col].median()
                if pd.isna(fill_value):
                    fill_value = 0  # fallback if median is NaN
                X_source[col] = X_source[col].fillna(fill_value)
                X_target[col] = X_target[col].fillna(fill_value)
        
        # Split target data
        X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
            X_target, y_target, test_size=0.3, random_state=42, stratify=y_target
        )
        
        # Create and fit model on source data
        preprocessor = self.create_preprocessor(X_source)
        
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train on source
        pipeline.fit(X_source, y_source)
        
        # Evaluate on target
        y_pred = pipeline.predict(X_target_test)
        y_pred_proba = pipeline.predict_proba(X_target_test)[:, 1]
        
        # Calculate metrics
        results = {
            'source_dataset': source_name,
            'target_dataset': target_name,
            'model_type': model_type,
            'common_features': common_features,
            'n_common_features': len(common_features),
            'source_samples': len(X_source),
            'target_train_samples': len(X_target_train),
            'target_test_samples': len(X_target_test),
            'source_pass_rate': y_source.mean(),
            'target_pass_rate': y_target.mean(),
            'accuracy': accuracy_score(y_target_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_target_test, y_pred),
            'f1_score': f1_score(y_target_test, y_pred),
            'roc_auc': roc_auc_score(y_target_test, y_pred_proba)
        }
        
        logger.info(f"Results: Acc={results['accuracy']:.3f}, F1={results['f1_score']:.3f}, AUC={results['roc_auc']:.3f}")
        return results
    
    def run_all_transfer_experiments(self) -> Dict:
        """Run all possible transfer learning experiments."""
        logger.info("Running all transfer learning experiments...")
        
        all_results = {}
        datasets = list(self.datasets.keys())
        
        for source in datasets:
            for target in datasets:
                if source != target:
                    exp_name = f"{source}_to_{target}"
                    
                    # Try both model types
                    for model_type in ['logistic', 'random_forest']:
                        model_exp_name = f"{exp_name}_{model_type}"
                        
                        try:
                            result = self.run_transfer_experiment(source, target, model_type)
                            if result:
                                all_results[model_exp_name] = result
                        except Exception as e:
                            logger.error(f"Failed {model_exp_name}: {e}")
                            all_results[model_exp_name] = {'error': str(e)}
        
        return all_results
    
    def generate_summary_report(self, results: Dict) -> Dict:
        """Generate summary report of transfer learning results."""
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful experiments'}
        
        # Calculate summary statistics
        accuracies = [r['accuracy'] for r in successful_results.values()]
        f1_scores = [r['f1_score'] for r in successful_results.values()]
        aucs = [r['roc_auc'] for r in successful_results.values()]
        
        summary = {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(results) - len(successful_results),
            'average_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'average_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'average_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'best_transfer': max(successful_results.items(), key=lambda x: x[1]['accuracy']),
            'dataset_stats': {}
        }
        
        # Dataset-specific statistics
        for dataset_name, df in self.datasets.items():
            summary['dataset_stats'][dataset_name] = {
                'samples': len(df),
                'features': len(df.columns),
                'pass_rate': df.get('label_pass', pd.Series([0])).mean()
            }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset transfer learning demo")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/multi_dataset_transfer"),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize transfer learning framework
    framework = MultiDatasetTransferLearning()
    
    # Load datasets
    if not framework.load_all_datasets():
        logger.error("No datasets could be loaded. Run setup scripts first.")
        return False
    
    # Run all transfer experiments
    start_time = time.time()
    results = framework.run_all_transfer_experiments()
    duration = time.time() - start_time
    
    # Generate summary
    summary = framework.generate_summary_report(results)
    summary['execution_time'] = duration
    
    # Save results
    results_path = args.output_dir / "transfer_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    summary_path = args.output_dir / "summary_report.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI-DATASET TRANSFER LEARNING RESULTS")
    print("="*60)
    
    print(f"\nDatasets loaded: {len(framework.datasets)}")
    for name, df in framework.datasets.items():
        print(f"  {name.upper()}: {len(df)} samples, {len(df.columns)} features")
    
    print(f"\nTransfer experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_experiments']}")
    print(f"Failed: {summary['failed_experiments']}")
    
    if summary['successful_experiments'] > 0:
        print(f"\nAverage Performance:")
        print(f"  Accuracy: {summary['average_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
        print(f"  F1-Score: {summary['average_f1']:.3f} ± {summary['std_f1']:.3f}")
        print(f"  ROC-AUC:  {summary['average_auc']:.3f} ± {summary['std_auc']:.3f}")
        
        best_exp, best_result = summary['best_transfer']
        print(f"\nBest Transfer: {best_exp}")
        print(f"  {best_result['source_dataset'].upper()} → {best_result['target_dataset'].upper()}")
        print(f"  Accuracy: {best_result['accuracy']:.3f}")
        print(f"  F1-Score: {best_result['f1_score']:.3f}")
        print(f"  ROC-AUC:  {best_result['roc_auc']:.3f}")
    
    print(f"\nExecution time: {duration:.1f} seconds")
    print(f"Results saved to: {args.output_dir}")
    
    return True


if __name__ == "__main__":
    main()