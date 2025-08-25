"""
Integration module for enhanced feature engineering with existing OULAD and transfer learning.

This module demonstrates how to integrate the enhanced feature engineering
capabilities with the existing OULAD and UCI transfer learning workflows.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.enhanced_feature_engineering import EnhancedFeatureEngineer, create_domain_adaptive_features

try:
    from src.oulad.advanced_deep_learning import advanced_feature_engineering as oulad_advanced_fe
    from src.transfer.improved_transfer import improved_feature_alignment
    from src.transfer.advanced_transfer import advanced_feature_engineering as transfer_advanced_fe
except ImportError as e:
    logging.warning(f"Could not import existing modules: {e}")

logger = logging.getLogger(__name__)


def enhance_oulad_pipeline(
    X: pd.DataFrame, 
    y: pd.Series, 
    use_existing: bool = True,
    combine_approaches: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Enhance OULAD feature engineering by combining existing and new approaches.
    
    Args:
        X: OULAD feature matrix
        y: Target labels
        use_existing: Whether to include existing OULAD feature engineering
        combine_approaches: Whether to combine multiple feature engineering approaches
        
    Returns:
        Enhanced feature matrix and feature names
    """
    logger.info("Enhancing OULAD pipeline with comprehensive feature engineering...")
    
    features_list = []
    feature_names = []
    
    # Enhanced feature engineering (new approach)
    engineer = EnhancedFeatureEngineer(dataset_type="oulad")
    X_enhanced = engineer.fit_transform(X, y)
    features_list.append(X_enhanced)
    feature_names.extend(engineer.get_feature_names())
    
    if use_existing and combine_approaches:
        try:
            # Existing OULAD advanced feature engineering
            X_existing, existing_names = oulad_advanced_fe(X, y)
            features_list.append(X_existing)
            feature_names.extend([f"existing_{name}" for name in existing_names])
        except Exception as e:
            logger.warning(f"Could not apply existing OULAD feature engineering: {e}")
    
    # Combine all features
    if len(features_list) > 1:
        X_final = np.hstack(features_list)
    else:
        X_final = features_list[0]
    
    logger.info(f"Enhanced OULAD features: {X_final.shape[1]} (original: {X.shape[1]})")
    
    return X_final, feature_names


def enhance_transfer_learning(
    source_X: pd.DataFrame,
    target_X: pd.DataFrame,
    source_y: pd.Series = None,
    target_y: pd.Series = None,
    use_domain_adaptation: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Enhance transfer learning with comprehensive feature engineering.
    
    Args:
        source_X: Source domain features (e.g., OULAD)
        target_X: Target domain features (e.g., UCI)
        source_y: Source domain labels
        target_y: Target domain labels
        use_domain_adaptation: Whether to use domain adaptation techniques
        
    Returns:
        Enhanced source features, enhanced target features, feature names
    """
    logger.info("Enhancing transfer learning with domain-adaptive features...")
    
    if use_domain_adaptation:
        # Use domain-adaptive feature engineering
        source_enhanced, target_enhanced = create_domain_adaptive_features(
            source_X, target_X, source_y, target_y
        )
        
        # Get feature names from a sample engineer
        engineer = EnhancedFeatureEngineer(dataset_type="generic")
        engineer.fit_transform(source_X.iloc[:min(50, len(source_X))], 
                              source_y[:min(50, len(source_X))] if source_y is not None else None)
        feature_names = engineer.get_feature_names()
        
    else:
        # Apply enhanced feature engineering separately
        source_engineer = EnhancedFeatureEngineer(dataset_type="auto")
        target_engineer = EnhancedFeatureEngineer(dataset_type="auto")
        
        source_enhanced = source_engineer.fit_transform(source_X, source_y)
        target_enhanced = target_engineer.fit_transform(target_X, target_y)
        
        feature_names = source_engineer.get_feature_names()
    
    logger.info(f"Transfer learning enhancement: Source {source_enhanced.shape}, Target {target_enhanced.shape}")
    
    return source_enhanced, target_enhanced, feature_names


def compare_feature_engineering_approaches(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Compare different feature engineering approaches on the same data.
    
    Args:
        X: Input features
        y: Target labels
        test_size: Fraction of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    logger.info("Comparing feature engineering approaches...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    results = {}
    
    # Baseline: No feature engineering
    for model_name, model in [
        ("Random Forest", RandomForestClassifier(n_estimators=50, random_state=random_state)),
        ("Logistic Regression", LogisticRegression(random_state=random_state, max_iter=1000))
    ]:
        
        # Baseline
        model_baseline = model.__class__(**model.get_params())
        
        # Handle categorical variables for baseline
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        for col in X_train_processed.select_dtypes(include=['object', 'category']).columns:
            X_train_processed[col] = pd.Categorical(X_train_processed[col]).codes
            X_test_processed[col] = pd.Categorical(X_test_processed[col]).codes
            
        X_train_processed = X_train_processed.fillna(0)
        X_test_processed = X_test_processed.fillna(0)
        
        model_baseline.fit(X_train_processed, y_train)
        baseline_pred = model_baseline.predict(X_test_processed)
        baseline_prob = model_baseline.predict_proba(X_test_processed)[:, 1] if hasattr(model_baseline, 'predict_proba') else baseline_pred
        
        # Enhanced feature engineering
        engineer = EnhancedFeatureEngineer(dataset_type="auto")
        X_train_enhanced = engineer.fit_transform(X_train, y_train)
        X_test_enhanced = engineer.transform(X_test)
        
        model_enhanced = model.__class__(**model.get_params())
        model_enhanced.fit(X_train_enhanced, y_train)
        enhanced_pred = model_enhanced.predict(X_test_enhanced)
        enhanced_prob = model_enhanced.predict_proba(X_test_enhanced)[:, 1] if hasattr(model_enhanced, 'predict_proba') else enhanced_pred
        
        # Store results
        results[model_name] = {
            'baseline': {
                'accuracy': accuracy_score(y_test, baseline_pred),
                'roc_auc': roc_auc_score(y_test, baseline_prob),
                'f1_score': f1_score(y_test, baseline_pred),
                'n_features': X_train_processed.shape[1]
            },
            'enhanced': {
                'accuracy': accuracy_score(y_test, enhanced_pred),
                'roc_auc': roc_auc_score(y_test, enhanced_prob),
                'f1_score': f1_score(y_test, enhanced_pred),
                'n_features': X_train_enhanced.shape[1]
            }
        }
        
        # Calculate improvement
        results[model_name]['improvement'] = {
            'accuracy': results[model_name]['enhanced']['accuracy'] - results[model_name]['baseline']['accuracy'],
            'roc_auc': results[model_name]['enhanced']['roc_auc'] - results[model_name]['baseline']['roc_auc'],
            'f1_score': results[model_name]['enhanced']['f1_score'] - results[model_name]['baseline']['f1_score']
        }
    
    return results


def create_feature_engineering_report(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create a comprehensive feature engineering report.
    
    Args:
        X: Input features
        y: Target labels
        output_path: Path to save the report
        
    Returns:
        Report dataframe
    """
    logger.info("Creating feature engineering report...")
    
    # Basic dataset information
    report_data = []
    
    # Original dataset info
    report_data.append({
        'metric': 'Original Features',
        'value': X.shape[1],
        'description': 'Number of original features'
    })
    
    report_data.append({
        'metric': 'Samples',
        'value': X.shape[0],
        'description': 'Number of samples'
    })
    
    # Apply enhanced feature engineering
    engineer = EnhancedFeatureEngineer(dataset_type="auto")
    X_enhanced = engineer.fit_transform(X, y)
    
    report_data.append({
        'metric': 'Enhanced Features',
        'value': X_enhanced.shape[1],
        'description': 'Number of features after enhancement'
    })
    
    report_data.append({
        'metric': 'Feature Increase',
        'value': f"{((X_enhanced.shape[1] / X.shape[1]) - 1) * 100:.1f}%",
        'description': 'Percentage increase in features'
    })
    
    report_data.append({
        'metric': 'Dataset Type Detected',
        'value': engineer.dataset_type,
        'description': 'Automatically detected dataset type'
    })
    
    # Feature importance analysis
    try:
        importance_df = engineer.get_feature_importance(X_enhanced, y)
        top_features = importance_df.head(5)['feature'].tolist()
        
        report_data.append({
            'metric': 'Top Features',
            'value': ', '.join(top_features[:3]) + '...',
            'description': 'Most important engineered features'
        })
    except Exception as e:
        logger.warning(f"Could not compute feature importance: {e}")
    
    # Performance comparison
    try:
        comparison_results = compare_feature_engineering_approaches(X, y)
        
        for model_name, results in comparison_results.items():
            improvement = results['improvement']
            
            report_data.append({
                'metric': f'{model_name} Accuracy Improvement',
                'value': f"{improvement['accuracy']:.3f}",
                'description': f'Accuracy improvement with enhanced features'
            })
            
            report_data.append({
                'metric': f'{model_name} ROC-AUC Improvement',
                'value': f"{improvement['roc_auc']:.3f}",
                'description': f'ROC-AUC improvement with enhanced features'
            })
            
    except Exception as e:
        logger.warning(f"Could not perform performance comparison: {e}")
    
    # Create dataframe
    report_df = pd.DataFrame(report_data)
    
    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)
        logger.info(f"Feature engineering report saved to {output_path}")
    
    return report_df


def run_enhanced_feature_engineering_demo():
    """
    Run a demonstration of the enhanced feature engineering capabilities.
    """
    from sklearn.datasets import make_classification
    
    print("="*80)
    print("ENHANCED FEATURE ENGINEERING DEMONSTRATION")
    print("="*80)
    
    # Create sample data
    print("\n1. Creating sample data...")
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=15, 
        n_redundant=3, n_clusters_per_class=2, random_state=42
    )
    
    # Convert to DataFrame with mixed feature types
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # Add some categorical features
    X_df['category_1'] = np.random.choice(['A', 'B', 'C'], size=len(X_df))
    X_df['category_2'] = np.random.choice(['high', 'medium', 'low'], size=len(X_df))
    
    print(f"   Original data shape: {X_df.shape}")
    print(f"   Target distribution: {y_series.value_counts().to_dict()}")
    
    # Apply enhanced feature engineering
    print("\n2. Applying enhanced feature engineering...")
    engineer = EnhancedFeatureEngineer(dataset_type="auto")
    X_enhanced = engineer.fit_transform(X_df, y_series)
    
    print(f"   Enhanced data shape: {X_enhanced.shape}")
    print(f"   Feature increase: {((X_enhanced.shape[1] / X_df.shape[1]) - 1) * 100:.1f}%")
    print(f"   Detected dataset type: {engineer.dataset_type}")
    
    # Feature importance
    print("\n3. Analyzing feature importance...")
    try:
        importance_df = engineer.get_feature_importance(X_enhanced, y_series)
        print("   Top 5 most important features:")
        for i, row in importance_df.head(5).iterrows():
            print(f"     {row['feature']}: MI={row['mutual_info']:.3f}, RF={row['rf_importance']:.3f}")
    except Exception as e:
        print(f"   Could not compute feature importance: {e}")
    
    # Performance comparison
    print("\n4. Comparing model performance...")
    try:
        results = compare_feature_engineering_approaches(X_df, y_series)
        
        for model_name, model_results in results.items():
            baseline = model_results['baseline']
            enhanced = model_results['enhanced']
            improvement = model_results['improvement']
            
            print(f"\n   {model_name}:")
            print(f"     Baseline:  Acc={baseline['accuracy']:.3f}, AUC={baseline['roc_auc']:.3f}, F1={baseline['f1_score']:.3f}")
            print(f"     Enhanced:  Acc={enhanced['accuracy']:.3f}, AUC={enhanced['roc_auc']:.3f}, F1={enhanced['f1_score']:.3f}")
            print(f"     Improvement: Acc={improvement['accuracy']:+.3f}, AUC={improvement['roc_auc']:+.3f}, F1={improvement['f1_score']:+.3f}")
            
    except Exception as e:
        print(f"   Could not perform comparison: {e}")
    
    # Create report
    print("\n5. Generating comprehensive report...")
    try:
        report_df = create_feature_engineering_report(X_df, y_series)
        print("\n   Feature Engineering Report:")
        for _, row in report_df.iterrows():
            print(f"     {row['metric']}: {row['value']} - {row['description']}")
    except Exception as e:
        print(f"   Could not generate report: {e}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    # Run demonstration
    run_enhanced_feature_engineering_demo()