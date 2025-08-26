"""
Advanced Transfer Learning R&D Demo

This script demonstrates all the advanced transfer learning techniques developed
for the GUIDE project, fixing data preprocessing issues and showcasing the
improvements over baseline methods.
"""

import logging
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import our advanced modules
from src.transfer.advanced_neural_transfer import (
    NeuralTransferLearningClassifier,
    evaluate_neural_transfer_methods
)
from src.transfer.advanced_ensemble import (
    AdvancedEnsembleTransfer,
    MixtureOfExpertsTransfer
)
from src.transfer.advanced_augmentation import (
    TransferAwareSMOTE,
    DomainAdaptationMixup,
    comprehensive_augmentation_evaluation
)
from src.transfer.theoretical_improvements import (
    TheoreticalTransferEnsemble,
    HDivergenceMinimizer,
    WassersteinDomainAlignment
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Robust data preprocessor that handles both OULAD and UCI datasets
    with proper categorical encoding and missing value handling.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_mapping = {}
        self.fitted = False
    
    def fit_transform_oulad(self, df):
        """Process OULAD dataset with proper encoding."""
        df = df.copy()
        
        # Handle categorical columns
        categorical_columns = ['code_module', 'code_presentation', 'sex', 'age_band', 
                              'highest_education', 'imd_band', 'sex_x_age']
        
        for col in categorical_columns:
            if col in df.columns:
                # Create label encoder for this column
                self.encoders[f'oulad_{col}'] = LabelEncoder()
                
                # Fill NaN with 'Unknown' and encode
                df[col] = df[col].fillna('Unknown').astype(str)
                df[col] = self.encoders[f'oulad_{col}'].fit_transform(df[col])
        
        # Handle numeric columns
        numeric_columns = [col for col in df.columns if col not in categorical_columns 
                          and col not in ['id_student', 'label_pass', 'label_fail_or_withdraw']]
        
        for col in numeric_columns:
            if col in df.columns:
                # Fill NaN with median
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Create target variable - ensure binary classification
        if 'label_pass' in df.columns:
            y = df['label_pass'].fillna(0).astype(int)
        else:
            y = np.zeros(len(df))
        
        # Select feature columns
        feature_columns = [col for col in df.columns 
                          if col not in ['id_student', 'label_pass', 'label_fail_or_withdraw']]
        
        X = df[feature_columns]
        
        # Scale features
        self.scalers['oulad'] = StandardScaler()
        X_scaled = self.scalers['oulad'].fit_transform(X)
        
        logger.info(f"OULAD processed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"OULAD class distribution: {np.bincount(y)}")
        
        return X_scaled, y
    
    def fit_transform_uci(self, df):
        """Process UCI dataset with proper encoding."""
        df = df.copy()
        
        # Handle categorical columns for UCI
        categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                              'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                              'famsup', 'paid', 'activities', 'nursery', 'higher', 
                              'internet', 'romantic']
        
        for col in categorical_columns:
            if col in df.columns:
                self.encoders[f'uci_{col}'] = LabelEncoder()
                df[col] = df[col].fillna('Unknown').astype(str)
                df[col] = self.encoders[f'uci_{col}'].fit_transform(df[col])
        
        # Handle numeric columns
        numeric_columns = [col for col in df.columns if col not in categorical_columns]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        # Create target variable from G3 (final grade)
        if 'G3' in df.columns:
            # Binary classification: pass (>=10) vs fail (<10)
            y = (df['G3'] >= 10).astype(int)
        else:
            y = np.zeros(len(df))
        
        # Remove target-related columns
        feature_columns = [col for col in df.columns if col not in ['G1', 'G2', 'G3']]
        X = df[feature_columns]
        
        # Scale features
        self.scalers['uci'] = StandardScaler()
        X_scaled = self.scalers['uci'].fit_transform(X)
        
        logger.info(f"UCI processed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"UCI class distribution: {np.bincount(y)}")
        
        return X_scaled, y
    
    def align_features(self, X_source, X_target):
        """Align feature dimensions between source and target domains."""
        # Ensure both have same number of features by padding or truncating
        n_features_source = X_source.shape[1]
        n_features_target = X_target.shape[1]
        
        if n_features_source == n_features_target:
            return X_source, X_target
        
        min_features = min(n_features_source, n_features_target)
        
        logger.info(f"Aligning features: source {n_features_source} -> {min_features}, "
                   f"target {n_features_target} -> {min_features}")
        
        return X_source[:, :min_features], X_target[:, :min_features]


def load_and_preprocess_data():
    """Load and preprocess OULAD and UCI datasets."""
    logger.info("Loading and preprocessing datasets...")
    
    preprocessor = DataPreprocessor()
    
    # Load OULAD
    try:
        oulad_df = pd.read_csv('data/oulad/processed/oulad_ml.csv')
        X_source, y_source = preprocessor.fit_transform_oulad(oulad_df)
    except FileNotFoundError:
        logger.warning("OULAD dataset not found, creating synthetic source data")
        n_source = 2000
        X_source = np.random.randn(n_source, 15)
        y_source = (X_source.sum(axis=1) + np.random.randn(n_source) * 0.3 > 0).astype(int)
        logger.info(f"Created synthetic source data: {X_source.shape}")
    
    # Load UCI
    try:
        uci_df = pd.read_csv('student-mat-fixed.csv')
        X_target, y_target = preprocessor.fit_transform_uci(uci_df)
    except FileNotFoundError:
        logger.warning("UCI dataset not found, creating synthetic target data")
        n_target = 400
        X_target = np.random.randn(n_target, 15) + 0.5  # Domain shift
        y_target = (X_target.sum(axis=1) + np.random.randn(n_target) * 0.3 > 0.2).astype(int)
        logger.info(f"Created synthetic target data: {X_target.shape}")
    
    # Align features
    X_source, X_target = preprocessor.align_features(X_source, X_target)
    
    return X_source, y_source, X_target, y_target


def evaluate_baseline_methods(X_source, y_source, X_target, y_target):
    """Evaluate baseline transfer learning methods."""
    logger.info("Evaluating baseline methods...")
    
    # Split target data
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target, test_size=0.3, random_state=42, stratify=y_target
    )
    
    baseline_results = {}
    
    # Direct transfer (train on source, test on target)
    models = {
        'direct_transfer_lr': LogisticRegression(random_state=42, max_iter=1000),
        'direct_transfer_rf': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        try:
            model.fit(X_source, y_source)
            y_pred = model.predict(X_target_test)
            
            baseline_results[name] = {
                'accuracy': accuracy_score(y_target_test, y_pred),
                'f1': f1_score(y_target_test, y_pred, zero_division=0)
            }
            
            logger.info(f"{name}: Accuracy={baseline_results[name]['accuracy']:.3f}, "
                       f"F1={baseline_results[name]['f1']:.3f}")
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            baseline_results[name] = {'accuracy': 0.0, 'f1': 0.0}
    
    # Target-only baseline
    try:
        target_model = RandomForestClassifier(n_estimators=100, random_state=42)
        target_model.fit(X_target_train, y_target_train)
        y_pred_target = target_model.predict(X_target_test)
        
        baseline_results['target_only'] = {
            'accuracy': accuracy_score(y_target_test, y_pred_target),
            'f1': f1_score(y_target_test, y_pred_target, zero_division=0)
        }
        
        logger.info(f"target_only: Accuracy={baseline_results['target_only']['accuracy']:.3f}, "
                   f"F1={baseline_results['target_only']['f1']:.3f}")
    except Exception as e:
        logger.error(f"Error in target_only: {e}")
        baseline_results['target_only'] = {'accuracy': 0.0, 'f1': 0.0}
    
    return baseline_results, X_target_train, X_target_test, y_target_train, y_target_test


def evaluate_advanced_methods(X_source, y_source, X_target_train, y_target_train, 
                             X_target_test, y_target_test):
    """Evaluate all advanced transfer learning methods."""
    results = {}
    
    # 1. Neural Transfer Learning
    logger.info("Evaluating neural transfer learning methods...")
    try:
        neural_methods = ['transformer', 'contrastive']  # Simplified for demo
        
        for method in neural_methods:
            try:
                model = NeuralTransferLearningClassifier(
                    method=method,
                    hidden_dim=64,  # Reduced for faster training
                    num_epochs=20,  # Reduced for demo
                    batch_size=32,
                    random_state=42
                )
                
                model.fit(X_source, y_source, X_target_train, y_target_train)
                y_pred = model.predict(X_target_test)
                
                results[f'neural_{method}'] = {
                    'accuracy': accuracy_score(y_target_test, y_pred),
                    'f1': f1_score(y_target_test, y_pred, zero_division=0)
                }
                
                logger.info(f"Neural {method}: Accuracy={results[f'neural_{method}']['accuracy']:.3f}, "
                           f"F1={results[f'neural_{method}']['f1']:.3f}")
            except Exception as e:
                logger.error(f"Error in neural {method}: {e}")
                results[f'neural_{method}'] = {'accuracy': 0.0, 'f1': 0.0}
    except Exception as e:
        logger.error(f"Error in neural methods: {e}")
    
    # 2. Advanced Ensemble Methods
    logger.info("Evaluating advanced ensemble methods...")
    try:
        ensemble_configs = {
            'mixture_experts': {'use_mixture_experts': True, 'use_nas': False, 'use_bayesian': False},
            'bayesian': {'use_mixture_experts': False, 'use_nas': False, 'use_bayesian': True}
        }
        
        for name, config in ensemble_configs.items():
            try:
                model = AdvancedEnsembleTransfer(**config, random_state=42)
                model.fit(X_source, y_source, X_target_train, y_target_train)
                y_pred = model.predict(X_target_test)
                
                results[f'ensemble_{name}'] = {
                    'accuracy': accuracy_score(y_target_test, y_pred),
                    'f1': f1_score(y_target_test, y_pred, zero_division=0)
                }
                
                logger.info(f"Ensemble {name}: Accuracy={results[f'ensemble_{name}']['accuracy']:.3f}, "
                           f"F1={results[f'ensemble_{name}']['f1']:.3f}")
            except Exception as e:
                logger.error(f"Error in ensemble {name}: {e}")
                results[f'ensemble_{name}'] = {'accuracy': 0.0, 'f1': 0.0}
    except Exception as e:
        logger.error(f"Error in ensemble methods: {e}")
    
    # 3. Data Augmentation
    logger.info("Evaluating data augmentation methods...")
    try:
        # Transfer-aware SMOTE
        try:
            smote = TransferAwareSMOTE(random_state=42)
            aug_X, aug_y = smote.fit_resample(X_source, y_source, X_target_train)
            
            aug_model = RandomForestClassifier(n_estimators=100, random_state=42)
            aug_model.fit(aug_X, aug_y)
            y_pred_aug = aug_model.predict(X_target_test)
            
            results['augmentation_smote'] = {
                'accuracy': accuracy_score(y_target_test, y_pred_aug),
                'f1': f1_score(y_target_test, y_pred_aug, zero_division=0)
            }
            
            logger.info(f"Augmentation SMOTE: Accuracy={results['augmentation_smote']['accuracy']:.3f}, "
                       f"F1={results['augmentation_smote']['f1']:.3f}")
        except Exception as e:
            logger.error(f"Error in SMOTE augmentation: {e}")
            results['augmentation_smote'] = {'accuracy': 0.0, 'f1': 0.0}
        
        # Domain adaptation mixup
        try:
            mixup = DomainAdaptationMixup(random_state=42)
            aug_X_mixup, aug_y_mixup = mixup.fit_transform(
                X_source, y_source, X_target_train, y_target_train
            )
            
            mixup_model = RandomForestClassifier(n_estimators=100, random_state=42)
            mixup_model.fit(aug_X_mixup, aug_y_mixup)
            y_pred_mixup = mixup_model.predict(X_target_test)
            
            results['augmentation_mixup'] = {
                'accuracy': accuracy_score(y_target_test, y_pred_mixup),
                'f1': f1_score(y_target_test, y_pred_mixup, zero_division=0)
            }
            
            logger.info(f"Augmentation Mixup: Accuracy={results['augmentation_mixup']['accuracy']:.3f}, "
                       f"F1={results['augmentation_mixup']['f1']:.3f}")
        except Exception as e:
            logger.error(f"Error in mixup augmentation: {e}")
            results['augmentation_mixup'] = {'accuracy': 0.0, 'f1': 0.0}
    except Exception as e:
        logger.error(f"Error in augmentation methods: {e}")
    
    # 4. Theoretical Methods
    logger.info("Evaluating theoretical methods...")
    try:
        theoretical_configs = {
            'h_divergence': {'use_h_divergence': True, 'use_wasserstein': False, 
                           'use_information_theory': False, 'use_causal': False},
            'causal': {'use_h_divergence': False, 'use_wasserstein': False,
                      'use_information_theory': False, 'use_causal': True}
        }
        
        for name, config in theoretical_configs.items():
            try:
                model = TheoreticalTransferEnsemble(**config)
                
                # Combine source and target training data
                X_combined = np.vstack([X_source, X_target_train])
                y_combined = np.hstack([y_source, y_target_train])
                
                model.fit(X_combined, y_combined, X_target_train)
                y_pred = model.predict(X_target_test)
                
                results[f'theoretical_{name}'] = {
                    'accuracy': accuracy_score(y_target_test, y_pred),
                    'f1': f1_score(y_target_test, y_pred, zero_division=0)
                }
                
                logger.info(f"Theoretical {name}: Accuracy={results[f'theoretical_{name}']['accuracy']:.3f}, "
                           f"F1={results[f'theoretical_{name}']['f1']:.3f}")
            except Exception as e:
                logger.error(f"Error in theoretical {name}: {e}")
                results[f'theoretical_{name}'] = {'accuracy': 0.0, 'f1': 0.0}
    except Exception as e:
        logger.error(f"Error in theoretical methods: {e}")
    
    return results


def generate_improvement_report(baseline_results, advanced_results):
    """Generate comprehensive improvement report."""
    logger.info("Generating improvement report...")
    
    # Find best baseline
    best_baseline_f1 = 0
    best_baseline_method = None
    for method, metrics in baseline_results.items():
        if metrics['f1'] > best_baseline_f1:
            best_baseline_f1 = metrics['f1']
            best_baseline_method = method
    
    print(f"\n{'='*80}")
    print("TRANSFER LEARNING R&D RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 BASELINE PERFORMANCE:")
    print(f"{'Method':<25} {'Accuracy':<12} {'F1 Score':<12}")
    print(f"{'-'*50}")
    for method, metrics in baseline_results.items():
        print(f"{method:<25} {metrics['accuracy']:<12.3f} {metrics['f1']:<12.3f}")
    
    print(f"\n🚀 ADVANCED METHODS PERFORMANCE:")
    print(f"{'Method':<25} {'Accuracy':<12} {'F1 Score':<12} {'Improvement':<12}")
    print(f"{'-'*65}")
    
    improvements = []
    for method, metrics in advanced_results.items():
        improvement = metrics['f1'] - best_baseline_f1
        improvements.append((method, improvement, metrics))
        status = "✅" if improvement > 0 else "❌" if improvement < -0.01 else "➖"
        print(f"{method:<25} {metrics['accuracy']:<12.3f} {metrics['f1']:<12.3f} "
              f"{improvement:+.3f} {status}")
    
    # Find best performing method
    best_advanced = max(improvements, key=lambda x: x[1])
    best_method, best_improvement, best_metrics = best_advanced
    
    print(f"\n🏆 BEST PERFORMING METHOD:")
    print(f"Method: {best_method}")
    print(f"F1 Score: {best_metrics['f1']:.3f}")
    print(f"Accuracy: {best_metrics['accuracy']:.3f}")
    print(f"Improvement over baseline: {best_improvement:+.3f}")
    
    # Count improvements
    significant_improvements = sum(1 for _, imp, _ in improvements if imp > 0.01)
    total_methods = len(improvements)
    
    print(f"\n📈 IMPROVEMENT STATISTICS:")
    print(f"Best baseline F1: {best_baseline_f1:.3f} ({best_baseline_method})")
    print(f"Methods with significant improvement (>0.01): {significant_improvements}/{total_methods}")
    print(f"Maximum improvement: {best_improvement:+.3f}")
    
    # Research insights
    print(f"\n🔬 KEY R&D INSIGHTS:")
    
    # Neural methods analysis
    neural_methods = {k: v for k, v in advanced_results.items() if k.startswith('neural_')}
    if neural_methods:
        best_neural = max(neural_methods.items(), key=lambda x: x[1]['f1'])
        print(f"• Best neural method: {best_neural[0]} (F1: {best_neural[1]['f1']:.3f})")
    
    # Ensemble methods analysis
    ensemble_methods = {k: v for k, v in advanced_results.items() if k.startswith('ensemble_')}
    if ensemble_methods:
        best_ensemble = max(ensemble_methods.items(), key=lambda x: x[1]['f1'])
        print(f"• Best ensemble method: {best_ensemble[0]} (F1: {best_ensemble[1]['f1']:.3f})")
    
    # Augmentation methods analysis
    aug_methods = {k: v for k, v in advanced_results.items() if k.startswith('augmentation_')}
    if aug_methods:
        best_aug = max(aug_methods.items(), key=lambda x: x[1]['f1'])
        print(f"• Best augmentation method: {best_aug[0]} (F1: {best_aug[1]['f1']:.3f})")
    
    # Theoretical methods analysis
    theoretical_methods = {k: v for k, v in advanced_results.items() if k.startswith('theoretical_')}
    if theoretical_methods:
        best_theoretical = max(theoretical_methods.items(), key=lambda x: x[1]['f1'])
        print(f"• Best theoretical method: {best_theoretical[0]} (F1: {best_theoretical[1]['f1']:.3f})")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if best_improvement > 0.05:
        print(f"• Deploy {best_method} for significant performance gains")
    elif best_improvement > 0.01:
        print(f"• Consider {best_method} for moderate improvements")
    else:
        print("• Further R&D needed - current methods show limited improvement")
    
    if significant_improvements > total_methods // 2:
        print("• Multiple advanced methods show promise - consider ensemble approach")
    
    print("• Continue research into domain adaptation and meta-learning")
    print("• Investigate dataset-specific optimizations")
    
    print(f"\n{'='*80}")
    
    return {
        'best_method': best_method,
        'best_improvement': best_improvement,
        'best_metrics': best_metrics,
        'significant_improvements': significant_improvements,
        'total_methods': total_methods
    }


def main():
    """Main demo function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Advanced Transfer Learning R&D Demo for GUIDE Project")
    print("="*80)
    
    # Load and preprocess data
    X_source, y_source, X_target, y_target = load_and_preprocess_data()
    
    # Evaluate baseline methods
    baseline_results, X_target_train, X_target_test, y_target_train, y_target_test = (
        evaluate_baseline_methods(X_source, y_source, X_target, y_target)
    )
    
    # Evaluate advanced methods
    advanced_results = evaluate_advanced_methods(
        X_source, y_source, X_target_train, y_target_train, 
        X_target_test, y_target_test
    )
    
    # Generate comprehensive report
    summary = generate_improvement_report(baseline_results, advanced_results)
    
    # Save results
    results_summary = {
        'baseline_results': baseline_results,
        'advanced_results': advanced_results,
        'summary': summary,
        'data_info': {
            'source_samples': len(X_source),
            'target_samples': len(X_target),
            'features': X_source.shape[1],
            'source_class_dist': np.bincount(y_source).tolist(),
            'target_class_dist': np.bincount(y_target).tolist()
        }
    }
    
    # Create output directory
    output_dir = Path('results/transfer_rd_demo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    import json
    with open(output_dir / 'rd_demo_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}/rd_demo_results.json")
    
    return results_summary


if __name__ == "__main__":
    results = main()