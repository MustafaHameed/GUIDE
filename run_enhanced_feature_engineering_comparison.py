#!/usr/bin/env python3
"""
Enhanced Feature Engineering ML/DL Results Comparison
=====================================================

This script re-runs ML/DL models with enhanced feature engineering and calculates
comprehensive results comparing baseline vs enhanced approaches.

Author: GUIDE Team
Date: 2025-08-25
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Import enhanced feature engineering modules
from src.enhanced_feature_engineering import EnhancedFeatureEngineer
from src.enhanced_oulad_integration import compare_feature_engineering_impact
from src.feature_engineering_integration import (
    compare_feature_engineering_approaches, 
    create_feature_engineering_report,
    run_enhanced_feature_engineering_demo
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load sample data for testing enhanced feature engineering.
    
    Returns:
        Feature matrix and target series
    """
    try:
        # Try to load real student data first
        if Path("student-mat.csv").exists():
            logger.info("Loading student-mat.csv data...")
            df = pd.read_csv("student-mat.csv")
            
            # Prepare features and target
            target_col = 'G3'  # Final grade
            if target_col in df.columns:
                y = (df[target_col] >= df[target_col].median()).astype(int)  # Binary classification
                X = df.drop(columns=[target_col, 'G1', 'G2'] if 'G1' in df.columns else [target_col])
                
                # Encode categorical variables
                for col in X.select_dtypes(include=['object']).columns:
                    X[col] = pd.Categorical(X[col]).codes
                
                logger.info(f"Loaded student data: {X.shape} features, {len(y)} samples")
                return X, y
    except Exception as e:
        logger.warning(f"Could not load student data: {e}")
    
    # Fallback to synthetic data
    logger.info("Generating synthetic data for testing...")
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=800,
        n_features=25,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(25)])
    y_series = pd.Series(y)
    
    # Add some categorical features to make it more realistic
    X_df['category_A'] = np.random.choice(['high', 'medium', 'low'], size=len(X_df))
    X_df['category_B'] = np.random.choice(['yes', 'no'], size=len(X_df))
    X_df['category_C'] = np.random.choice(['type1', 'type2', 'type3'], size=len(X_df))
    
    logger.info(f"Generated synthetic data: {X_df.shape} features, {len(y_series)} samples")
    return X_df, y_series


def run_oulad_deep_learning_comparison(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Run OULAD deep learning comparison with enhanced feature engineering.
    
    Args:
        X: Feature matrix
        y: Target series
        
    Returns:
        Comparison results
    """
    logger.info("Running OULAD deep learning comparison...")
    
    try:
        # Use the existing OULAD integration function
        results = compare_feature_engineering_impact(X, y, test_size=0.2, random_state=42)
        
        logger.info("OULAD deep learning comparison completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"OULAD deep learning comparison failed: {e}")
        return {"error": str(e)}


def run_uci_ml_comparison(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
    """
    Run UCI ML model comparison with enhanced feature engineering.
    
    Args:
        X: Feature matrix
        y: Target series
        
    Returns:
        Comparison results for multiple models
    """
    logger.info("Running UCI ML model comparison...")
    
    try:
        # Use the existing UCI integration function
        results = compare_feature_engineering_approaches(X, y, test_size=0.3, random_state=42)
        
        logger.info("UCI ML model comparison completed successfully")
        return results
    
    except Exception as e:
        logger.error(f"UCI ML model comparison failed: {e}")
        return {"error": str(e)}


def calculate_comprehensive_metrics(
    oulad_results: Dict[str, Any], 
    uci_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics from all results.
    
    Args:
        oulad_results: Results from OULAD deep learning
        uci_results: Results from UCI ML models
        
    Returns:
        Comprehensive metrics summary
    """
    logger.info("Calculating comprehensive metrics...")
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "summary": {},
        "oulad_deep_learning": {},
        "uci_ml_models": {},
        "overall_improvements": {}
    }
    
    # OULAD Deep Learning Metrics
    if "error" not in oulad_results:
        try:
            baseline = oulad_results.get("baseline", {})
            enhanced = oulad_results.get("enhanced", {})
            improvement = oulad_results.get("improvement", {})
            
            metrics["oulad_deep_learning"] = {
                "baseline_accuracy": baseline.get("test_accuracy", 0),
                "baseline_auc": baseline.get("test_auc", 0),
                "baseline_f1": baseline.get("test_f1", 0),
                "enhanced_accuracy": enhanced.get("test_accuracy", 0),
                "enhanced_auc": enhanced.get("test_auc", 0),
                "enhanced_f1": enhanced.get("test_f1", 0),
                "accuracy_improvement": improvement.get("accuracy", 0),
                "auc_improvement": improvement.get("auc", 0),
                "f1_improvement": improvement.get("f1", 0),
                "feature_ratio": improvement.get("feature_ratio", 1),
                "baseline_features": baseline.get("n_features", 0),
                "enhanced_features": enhanced.get("n_features", 0)
            }
        except Exception as e:
            logger.warning(f"Could not process OULAD results: {e}")
    
    # UCI ML Models Metrics
    if "error" not in uci_results:
        try:
            model_improvements = {}
            for model_name, model_data in uci_results.items():
                if isinstance(model_data, dict) and "improvement" in model_data:
                    baseline = model_data["baseline"]
                    enhanced = model_data["enhanced"]
                    improvement = model_data["improvement"]
                    
                    model_improvements[model_name] = {
                        "baseline_accuracy": baseline.get("accuracy", 0),
                        "baseline_auc": baseline.get("roc_auc", 0),
                        "baseline_f1": baseline.get("f1_score", 0),
                        "enhanced_accuracy": enhanced.get("accuracy", 0),
                        "enhanced_auc": enhanced.get("roc_auc", 0),
                        "enhanced_f1": enhanced.get("f1_score", 0),
                        "accuracy_improvement": improvement.get("accuracy", 0),
                        "auc_improvement": improvement.get("roc_auc", 0),
                        "f1_improvement": improvement.get("f1_score", 0)
                    }
            
            metrics["uci_ml_models"] = model_improvements
        except Exception as e:
            logger.warning(f"Could not process UCI results: {e}")
    
    # Overall Improvements Summary
    try:
        all_accuracy_improvements = []
        all_auc_improvements = []
        all_f1_improvements = []
        
        # Collect OULAD improvements
        if "oulad_deep_learning" in metrics and metrics["oulad_deep_learning"]:
            oulad_dl = metrics["oulad_deep_learning"]
            all_accuracy_improvements.append(oulad_dl.get("accuracy_improvement", 0))
            all_auc_improvements.append(oulad_dl.get("auc_improvement", 0))
            all_f1_improvements.append(oulad_dl.get("f1_improvement", 0))
        
        # Collect UCI improvements
        for model_name, model_metrics in metrics.get("uci_ml_models", {}).items():
            all_accuracy_improvements.append(model_metrics.get("accuracy_improvement", 0))
            all_auc_improvements.append(model_metrics.get("auc_improvement", 0))
            all_f1_improvements.append(model_metrics.get("f1_improvement", 0))
        
        if all_accuracy_improvements:
            metrics["overall_improvements"] = {
                "mean_accuracy_improvement": np.mean(all_accuracy_improvements),
                "mean_auc_improvement": np.mean(all_auc_improvements),
                "mean_f1_improvement": np.mean(all_f1_improvements),
                "median_accuracy_improvement": np.median(all_accuracy_improvements),
                "median_auc_improvement": np.median(all_auc_improvements),
                "median_f1_improvement": np.median(all_f1_improvements),
                "models_with_accuracy_improvement": sum(1 for x in all_accuracy_improvements if x > 0),
                "models_with_auc_improvement": sum(1 for x in all_auc_improvements if x > 0),
                "models_with_f1_improvement": sum(1 for x in all_f1_improvements if x > 0),
                "total_models_tested": len(all_accuracy_improvements)
            }
    except Exception as e:
        logger.warning(f"Could not calculate overall improvements: {e}")
    
    # Summary
    try:
        metrics["summary"] = {
            "enhanced_feature_engineering_applied": True,
            "oulad_deep_learning_tested": "error" not in oulad_results,
            "uci_ml_models_tested": "error" not in uci_results,
            "total_models_compared": len(metrics.get("uci_ml_models", {})) + (1 if "error" not in oulad_results else 0),
            "overall_improvement_positive": metrics.get("overall_improvements", {}).get("mean_accuracy_improvement", 0) > 0
        }
    except Exception as e:
        logger.warning(f"Could not create summary: {e}")
    
    logger.info("Comprehensive metrics calculation completed")
    return metrics


def save_results(results: Dict[str, Any], output_dir: Path = None) -> Path:
    """
    Save results to files.
    
    Args:
        results: Comprehensive results dictionary
        output_dir: Output directory (defaults to current directory)
        
    Returns:
        Path to saved results directory
    """
    if output_dir is None:
        output_dir = Path("enhanced_feature_engineering_results")
    
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    json_path = output_dir / "comprehensive_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary CSV
    summary_data = []
    
    # OULAD DL summary
    if "oulad_deep_learning" in results and results["oulad_deep_learning"]:
        oulad_dl = results["oulad_deep_learning"]
        summary_data.append({
            "Model_Type": "OULAD_Deep_Learning",
            "Model_Name": "Deep_Neural_Network", 
            "Baseline_Accuracy": oulad_dl.get("baseline_accuracy", 0),
            "Enhanced_Accuracy": oulad_dl.get("enhanced_accuracy", 0),
            "Accuracy_Improvement": oulad_dl.get("accuracy_improvement", 0),
            "Baseline_AUC": oulad_dl.get("baseline_auc", 0),
            "Enhanced_AUC": oulad_dl.get("enhanced_auc", 0),
            "AUC_Improvement": oulad_dl.get("auc_improvement", 0),
            "Baseline_F1": oulad_dl.get("baseline_f1", 0),
            "Enhanced_F1": oulad_dl.get("enhanced_f1", 0),
            "F1_Improvement": oulad_dl.get("f1_improvement", 0),
            "Feature_Ratio": oulad_dl.get("feature_ratio", 1)
        })
    
    # UCI ML models summary
    for model_name, model_metrics in results.get("uci_ml_models", {}).items():
        summary_data.append({
            "Model_Type": "UCI_ML",
            "Model_Name": model_name,
            "Baseline_Accuracy": model_metrics.get("baseline_accuracy", 0),
            "Enhanced_Accuracy": model_metrics.get("enhanced_accuracy", 0),
            "Accuracy_Improvement": model_metrics.get("accuracy_improvement", 0),
            "Baseline_AUC": model_metrics.get("baseline_auc", 0),
            "Enhanced_AUC": model_metrics.get("enhanced_auc", 0),
            "AUC_Improvement": model_metrics.get("auc_improvement", 0),
            "Baseline_F1": model_metrics.get("baseline_f1", 0),
            "Enhanced_F1": model_metrics.get("enhanced_f1", 0),
            "F1_Improvement": model_metrics.get("f1_improvement", 0),
            "Feature_Ratio": "N/A"
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        csv_path = output_dir / "model_comparison_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Summary saved to: {csv_path}")
    
    # Save overall improvements
    if "overall_improvements" in results and results["overall_improvements"]:
        improvements_df = pd.DataFrame([results["overall_improvements"]])
        improvements_path = output_dir / "overall_improvements.csv"
        improvements_df.to_csv(improvements_path, index=False)
        logger.info(f"Overall improvements saved to: {improvements_path}")
    
    logger.info(f"Results saved to: {output_dir}")
    return output_dir


def print_results_summary(results: Dict[str, Any]):
    """
    Print a formatted summary of results.
    
    Args:
        results: Comprehensive results dictionary
    """
    print("\n" + "="*80)
    print("ENHANCED FEATURE ENGINEERING ML/DL RESULTS SUMMARY")
    print("="*80)
    
    # Overall Summary
    summary = results.get("summary", {})
    print(f"\n📊 Overall Summary:")
    print(f"   ✅ Enhanced feature engineering applied: {summary.get('enhanced_feature_engineering_applied', False)}")
    print(f"   🧠 OULAD deep learning tested: {summary.get('oulad_deep_learning_tested', False)}")
    print(f"   🔬 UCI ML models tested: {summary.get('uci_ml_models_tested', False)}")
    print(f"   📈 Total models compared: {summary.get('total_models_compared', 0)}")
    print(f"   📊 Overall improvement positive: {summary.get('overall_improvement_positive', False)}")
    
    # OULAD Deep Learning Results
    oulad_dl = results.get("oulad_deep_learning", {})
    if oulad_dl:
        print(f"\n🧠 OULAD Deep Learning Results:")
        print(f"   Baseline:  Acc={oulad_dl.get('baseline_accuracy', 0):.3f}, AUC={oulad_dl.get('baseline_auc', 0):.3f}, F1={oulad_dl.get('baseline_f1', 0):.3f}")
        print(f"   Enhanced:  Acc={oulad_dl.get('enhanced_accuracy', 0):.3f}, AUC={oulad_dl.get('enhanced_auc', 0):.3f}, F1={oulad_dl.get('enhanced_f1', 0):.3f}")
        print(f"   Improvement: Acc={oulad_dl.get('accuracy_improvement', 0):+.3f}, AUC={oulad_dl.get('auc_improvement', 0):+.3f}, F1={oulad_dl.get('f1_improvement', 0):+.3f}")
        print(f"   Features: {oulad_dl.get('baseline_features', 0)} → {oulad_dl.get('enhanced_features', 0)} (ratio: {oulad_dl.get('feature_ratio', 1):.2f}x)")
    
    # UCI ML Models Results
    uci_models = results.get("uci_ml_models", {})
    if uci_models:
        print(f"\n🔬 UCI ML Models Results:")
        for model_name, metrics in uci_models.items():
            print(f"\n   {model_name}:")
            print(f"     Baseline:  Acc={metrics.get('baseline_accuracy', 0):.3f}, AUC={metrics.get('baseline_auc', 0):.3f}, F1={metrics.get('baseline_f1', 0):.3f}")
            print(f"     Enhanced:  Acc={metrics.get('enhanced_accuracy', 0):.3f}, AUC={metrics.get('enhanced_auc', 0):.3f}, F1={metrics.get('enhanced_f1', 0):.3f}")
            print(f"     Improvement: Acc={metrics.get('accuracy_improvement', 0):+.3f}, AUC={metrics.get('auc_improvement', 0):+.3f}, F1={metrics.get('f1_improvement', 0):+.3f}")
    
    # Overall Improvements
    overall = results.get("overall_improvements", {})
    if overall:
        print(f"\n📈 Overall Improvements Across All Models:")
        print(f"   Mean Accuracy Improvement: {overall.get('mean_accuracy_improvement', 0):+.3f}")
        print(f"   Mean AUC Improvement: {overall.get('mean_auc_improvement', 0):+.3f}")
        print(f"   Mean F1 Improvement: {overall.get('mean_f1_improvement', 0):+.3f}")
        print(f"   Models with Accuracy Improvement: {overall.get('models_with_accuracy_improvement', 0)}/{overall.get('total_models_tested', 0)}")
        print(f"   Models with AUC Improvement: {overall.get('models_with_auc_improvement', 0)}/{overall.get('total_models_tested', 0)}")
        print(f"   Models with F1 Improvement: {overall.get('models_with_f1_improvement', 0)}/{overall.get('total_models_tested', 0)}")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("🚀 ENHANCED FEATURE ENGINEERING ML/DL COMPARISON")
    print("="*60)
    
    # Load data
    logger.info("Loading data...")
    X, y = load_sample_data()
    print(f"📊 Data loaded: {X.shape} features, {len(y)} samples")
    
    # Run OULAD deep learning comparison
    print("\n🧠 Running OULAD deep learning comparison...")
    oulad_results = run_oulad_deep_learning_comparison(X, y)
    
    # Run UCI ML model comparison  
    print("\n🔬 Running UCI ML model comparison...")
    uci_results = run_uci_ml_comparison(X, y)
    
    # Calculate comprehensive metrics
    print("\n📊 Calculating comprehensive metrics...")
    comprehensive_results = calculate_comprehensive_metrics(oulad_results, uci_results)
    
    # Save results
    print("\n💾 Saving results...")
    results_dir = save_results(comprehensive_results)
    
    # Print summary
    print_results_summary(comprehensive_results)
    
    print(f"\n✅ Analysis complete! Results saved to: {results_dir}")
    print(f"📁 View comprehensive_results.json for detailed results")
    print(f"📊 View model_comparison_summary.csv for tabular comparison")
    
    return comprehensive_results, results_dir


if __name__ == "__main__":
    try:
        results, output_dir = main()
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise