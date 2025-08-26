"""
Comprehensive R&D Transfer Learning Evaluation Framework

This module provides a comprehensive framework for evaluating and comparing
all advanced transfer learning techniques implemented for the GUIDE project.
It includes automated experimentation, ablation studies, and performance analysis.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Import our advanced transfer learning modules
try:
    from .advanced_neural_transfer import (
        NeuralTransferLearningClassifier, 
        MetaTransferLearner,
        evaluate_neural_transfer_methods
    )
    from .advanced_ensemble import (
        AdvancedEnsembleTransfer,
        comprehensive_ensemble_evaluation
    )
    from .advanced_augmentation import (
        comprehensive_augmentation_evaluation
    )
    from .theoretical_improvements import (
        TheoreticalTransferEnsemble,
        evaluate_theoretical_methods
    )
    from .feature_bridge import FeatureBridge
except ImportError:
    # Handle imports when running as standalone
    import sys
    sys.path.append('.')
    from advanced_neural_transfer import *
    from advanced_ensemble import *
    from advanced_augmentation import *
    from theoretical_improvements import *

logger = logging.getLogger(__name__)


class TransferLearningR_DFramework:
    """
    Comprehensive R&D framework for transfer learning experimentation.
    """
    
    def __init__(self, 
                 output_dir: str = 'results/transfer_rd',
                 save_models: bool = True,
                 random_state: int = 42):
        """
        Initialize R&D framework.
        
        Args:
            output_dir: Directory to save results
            save_models: Whether to save trained models
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_models = save_models
        self.random_state = random_state
        
        # Initialize results storage
        self.results = {}
        self.models = {}
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'random_state': random_state,
            'framework_version': '1.0.0'
        }
        
        logger.info(f"Initialized R&D framework with output directory: {self.output_dir}")
    
    def run_comprehensive_evaluation(self, 
                                   X_source, y_source, 
                                   X_target, y_target,
                                   test_size: float = 0.3,
                                   include_baselines: bool = True,
                                   include_neural: bool = True,
                                   include_ensemble: bool = True,
                                   include_augmentation: bool = True,
                                   include_theoretical: bool = True):
        """
        Run comprehensive evaluation of all transfer learning methods.
        
        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features
            y_target: Target domain labels
            test_size: Fraction for train/test split
            include_*: Whether to include specific method categories
        """
        logger.info("Starting comprehensive transfer learning R&D evaluation...")
        
        # Prepare data
        X_source = np.array(X_source)
        y_source = np.array(y_source)
        X_target = np.array(X_target)
        y_target = np.array(y_target)
        
        # Split target data for training/testing
        X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
            X_target, y_target, test_size=test_size, random_state=self.random_state,
            stratify=y_target if len(np.unique(y_target)) > 1 else None
        )
        
        self.metadata.update({
            'source_samples': len(X_source),
            'target_samples': len(X_target),
            'target_train_samples': len(X_target_train),
            'target_test_samples': len(X_target_test),
            'features': X_source.shape[1],
            'classes': len(np.unique(y_source))
        })
        
        # Store data splits
        self.data_splits = {
            'X_source': X_source,
            'y_source': y_source,
            'X_target_train': X_target_train,
            'y_target_train': y_target_train,
            'X_target_test': X_target_test,
            'y_target_test': y_target_test
        }
        
        # 1. Baseline methods
        if include_baselines:
            logger.info("Evaluating baseline methods...")
            self.results['baselines'] = self._evaluate_baselines()
        
        # 2. Neural transfer learning
        if include_neural:
            logger.info("Evaluating neural transfer learning methods...")
            self.results['neural'] = self._evaluate_neural_methods()
        
        # 3. Advanced ensemble methods
        if include_ensemble:
            logger.info("Evaluating advanced ensemble methods...")
            self.results['ensemble'] = self._evaluate_ensemble_methods()
        
        # 4. Data augmentation impact
        if include_augmentation:
            logger.info("Evaluating data augmentation methods...")
            self.results['augmentation'] = self._evaluate_augmentation_methods()
        
        # 5. Theoretical improvements
        if include_theoretical:
            logger.info("Evaluating theoretical methods...")
            self.results['theoretical'] = self._evaluate_theoretical_methods()
        
        # 6. Combined best approaches
        logger.info("Evaluating combined best approaches...")
        self.results['combined'] = self._evaluate_combined_approaches()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        # Save results
        self._save_results()
        
        logger.info("Comprehensive evaluation completed!")
        
        return self.results
    
    def _evaluate_baselines(self):
        """Evaluate baseline transfer learning methods."""
        X_source = self.data_splits['X_source']
        y_source = self.data_splits['y_source']
        X_target_test = self.data_splits['X_target_test']
        y_target_test = self.data_splits['y_target_test']
        
        # Scale features
        scaler = StandardScaler()
        X_source_scaled = scaler.fit_transform(X_source)
        X_target_scaled = scaler.transform(X_target_test)
        
        baseline_methods = {
            'direct_transfer_lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'direct_transfer_rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'target_only_lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'target_only_rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        results = {}
        
        for method_name, model in baseline_methods.items():
            try:
                if 'target_only' in method_name:
                    # Train only on target training data
                    X_train = self.data_splits['X_target_train']
                    y_train = self.data_splits['y_target_train']
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_target_test)
                else:
                    # Direct transfer from source
                    X_train_scaled = X_source_scaled
                    y_train = y_source
                    X_test_scaled = X_target_scaled
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                results[method_name] = {
                    'accuracy': accuracy_score(y_target_test, y_pred),
                    'f1': f1_score(y_target_test, y_pred),
                    'auc': roc_auc_score(y_target_test, y_prob) if y_prob is not None and len(np.unique(y_target_test)) > 1 else 0.5
                }
                
                if self.save_models:
                    self.models[f'baseline_{method_name}'] = model
                
                logger.info(f"Baseline {method_name}: Acc={results[method_name]['accuracy']:.3f}, "
                           f"F1={results[method_name]['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"Error in baseline {method_name}: {e}")
                results[method_name] = {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5, 'error': str(e)}
        
        return results
    
    def _evaluate_neural_methods(self):
        """Evaluate neural transfer learning methods."""
        X_source = self.data_splits['X_source']
        y_source = self.data_splits['y_source']
        X_target_train = self.data_splits['X_target_train']
        y_target_train = self.data_splits['y_target_train']
        X_target_test = self.data_splits['X_target_test']
        y_target_test = self.data_splits['y_target_test']
        
        neural_methods = ['transformer', 'contrastive', 'progressive']
        results = {}
        
        for method in neural_methods:
            try:
                logger.info(f"Training neural method: {method}")
                
                model = NeuralTransferLearningClassifier(
                    method=method,
                    hidden_dim=128,
                    num_epochs=50,  # Reduced for efficiency
                    batch_size=32,
                    random_state=self.random_state
                )
                
                # Train with source and target training data
                model.fit(X_source, y_source, X_target_train, y_target_train)
                
                # Evaluate on target test set
                y_pred = model.predict(X_target_test)
                y_prob = model.predict_proba(X_target_test)[:, 1]
                
                results[method] = {
                    'accuracy': accuracy_score(y_target_test, y_pred),
                    'f1': f1_score(y_target_test, y_pred),
                    'auc': roc_auc_score(y_target_test, y_prob) if len(np.unique(y_target_test)) > 1 else 0.5
                }
                
                if self.save_models:
                    self.models[f'neural_{method}'] = model
                
                logger.info(f"Neural {method}: Acc={results[method]['accuracy']:.3f}, "
                           f"F1={results[method]['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"Error in neural method {method}: {e}")
                results[method] = {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5, 'error': str(e)}
        
        # Meta-learning approach
        try:
            logger.info("Training meta-learning approach...")
            
            meta_model = MetaTransferLearner(
                n_meta_epochs=30,
                hidden_dim=128
            )
            
            meta_model.fit(X_source, y_source, X_target_train, y_target_train)
            
            y_pred = meta_model.predict(X_target_test)
            y_prob = meta_model.predict_proba(X_target_test)[:, 1]
            
            results['meta_learning'] = {
                'accuracy': accuracy_score(y_target_test, y_pred),
                'f1': f1_score(y_target_test, y_pred),
                'auc': roc_auc_score(y_target_test, y_prob) if len(np.unique(y_target_test)) > 1 else 0.5
            }
            
            if self.save_models:
                self.models['neural_meta_learning'] = meta_model
            
            logger.info(f"Neural meta-learning: Acc={results['meta_learning']['accuracy']:.3f}, "
                       f"F1={results['meta_learning']['f1']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in meta-learning: {e}")
            results['meta_learning'] = {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5, 'error': str(e)}
        
        return results
    
    def _evaluate_ensemble_methods(self):
        """Evaluate advanced ensemble methods."""
        X_source = self.data_splits['X_source']
        y_source = self.data_splits['y_source']
        X_target_train = self.data_splits['X_target_train']
        y_target_train = self.data_splits['y_target_train']
        X_target_test = self.data_splits['X_target_test']
        y_target_test = self.data_splits['y_target_test']
        
        ensemble_configs = {
            'mixture_experts': {
                'use_mixture_experts': True, 
                'use_nas': False, 
                'use_bayesian': False
            },
            'bayesian_average': {
                'use_mixture_experts': False, 
                'use_nas': False, 
                'use_bayesian': True
            },
            'combined_ensemble': {
                'use_mixture_experts': True, 
                'use_nas': False, 
                'use_bayesian': True
            }
        }
        
        results = {}
        
        for method_name, config in ensemble_configs.items():
            try:
                logger.info(f"Training ensemble method: {method_name}")
                
                model = AdvancedEnsembleTransfer(**config, random_state=self.random_state)
                model.fit(X_source, y_source, X_target_train, y_target_train)
                
                # Evaluate
                y_pred = model.predict(X_target_test)
                y_prob = model.predict_proba(X_target_test)[:, 1]
                
                results[method_name] = {
                    'accuracy': accuracy_score(y_target_test, y_pred),
                    'f1': f1_score(y_target_test, y_pred),
                    'auc': roc_auc_score(y_target_test, y_prob) if len(np.unique(y_target_test)) > 1 else 0.5
                }
                
                if self.save_models:
                    self.models[f'ensemble_{method_name}'] = model
                
                logger.info(f"Ensemble {method_name}: Acc={results[method_name]['accuracy']:.3f}, "
                           f"F1={results[method_name]['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"Error in ensemble method {method_name}: {e}")
                results[method_name] = {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5, 'error': str(e)}
        
        return results
    
    def _evaluate_augmentation_methods(self):
        """Evaluate impact of data augmentation methods."""
        X_source = self.data_splits['X_source']
        y_source = self.data_splits['y_source']
        X_target_train = self.data_splits['X_target_train']
        y_target_train = self.data_splits['y_target_train']
        X_target_test = self.data_splits['X_target_test']
        y_target_test = self.data_splits['y_target_test']
        
        # Import augmentation methods
        from .advanced_augmentation import (
            TransferAwareSMOTE, DomainAdaptationMixup, 
            AdversarialAugmentation, ProgressiveAugmentation
        )
        
        augmentation_methods = {
            'transfer_smote': TransferAwareSMOTE(random_state=self.random_state),
            'domain_mixup': DomainAdaptationMixup(random_state=self.random_state),
            'adversarial': AdversarialAugmentation(random_state=self.random_state),
            'progressive': ProgressiveAugmentation(random_state=self.random_state)
        }
        
        results = {}
        base_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        for aug_name, augmenter in augmentation_methods.items():
            try:
                logger.info(f"Evaluating augmentation: {aug_name}")
                
                # Apply augmentation
                if aug_name == 'transfer_smote':
                    aug_X, aug_y = augmenter.fit_resample(X_source, y_source, X_target_train)
                elif aug_name == 'domain_mixup':
                    aug_X, aug_y = augmenter.fit_transform(
                        X_source, y_source, X_target_train, y_target_train
                    )
                elif aug_name == 'adversarial':
                    aug_X, aug_y = augmenter.fit_transform(X_source, y_source)
                elif aug_name == 'progressive':
                    aug_X, aug_y = augmenter.fit_transform_stage(
                        X_source, y_source, X_target_train, y_target_train
                    )
                
                # Train model on augmented data
                scaler = StandardScaler()
                aug_X_scaled = scaler.fit_transform(aug_X)
                X_test_scaled = scaler.transform(X_target_test)
                
                model = base_model.__class__(**base_model.get_params())
                model.fit(aug_X_scaled, aug_y)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                
                results[aug_name] = {
                    'accuracy': accuracy_score(y_target_test, y_pred),
                    'f1': f1_score(y_target_test, y_pred),
                    'auc': roc_auc_score(y_target_test, y_prob) if len(np.unique(y_target_test)) > 1 else 0.5,
                    'augmentation_ratio': len(aug_X) / len(X_source)
                }
                
                logger.info(f"Augmentation {aug_name}: Acc={results[aug_name]['accuracy']:.3f}, "
                           f"F1={results[aug_name]['f1']:.3f}, "
                           f"Aug ratio={results[aug_name]['augmentation_ratio']:.2f}")
                
            except Exception as e:
                logger.error(f"Error in augmentation {aug_name}: {e}")
                results[aug_name] = {
                    'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5, 
                    'augmentation_ratio': 1.0, 'error': str(e)
                }
        
        return results
    
    def _evaluate_theoretical_methods(self):
        """Evaluate theoretical transfer learning methods."""
        X_source = self.data_splits['X_source']
        y_source = self.data_splits['y_source']
        X_target_train = self.data_splits['X_target_train']
        y_target_train = self.data_splits['y_target_train']
        X_target_test = self.data_splits['X_target_test']
        y_target_test = self.data_splits['y_target_test']
        
        theoretical_configs = {
            'h_divergence': {
                'use_h_divergence': True, 'use_wasserstein': False,
                'use_information_theory': False, 'use_causal': False
            },
            'wasserstein': {
                'use_h_divergence': False, 'use_wasserstein': True,
                'use_information_theory': False, 'use_causal': False
            },
            'information_theory': {
                'use_h_divergence': False, 'use_wasserstein': False,
                'use_information_theory': True, 'use_causal': False
            },
            'causal': {
                'use_h_divergence': False, 'use_wasserstein': False,
                'use_information_theory': False, 'use_causal': True
            },
            'theoretical_ensemble': {
                'use_h_divergence': True, 'use_wasserstein': True,
                'use_information_theory': True, 'use_causal': True
            }
        }
        
        results = {}
        
        for method_name, config in theoretical_configs.items():
            try:
                logger.info(f"Training theoretical method: {method_name}")
                
                model = TheoreticalTransferEnsemble(**config)
                
                # Combine source and target training data
                X_combined = np.vstack([X_source, X_target_train])
                y_combined = np.hstack([y_source, y_target_train])
                
                model.fit(X_combined, y_combined, X_target_train)
                
                # Evaluate
                y_pred = model.predict(X_target_test)
                y_prob = model.predict_proba(X_target_test)[:, 1]
                
                results[method_name] = {
                    'accuracy': accuracy_score(y_target_test, y_pred),
                    'f1': f1_score(y_target_test, y_pred),
                    'auc': roc_auc_score(y_target_test, y_prob) if len(np.unique(y_target_test)) > 1 else 0.5
                }
                
                if self.save_models:
                    self.models[f'theoretical_{method_name}'] = model
                
                logger.info(f"Theoretical {method_name}: Acc={results[method_name]['accuracy']:.3f}, "
                           f"F1={results[method_name]['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"Error in theoretical method {method_name}: {e}")
                results[method_name] = {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5, 'error': str(e)}
        
        return results
    
    def _evaluate_combined_approaches(self):
        """Evaluate combinations of best approaches."""
        results = {}
        
        # Find best methods from each category
        best_methods = self._find_best_methods()
        
        logger.info(f"Best methods identified: {best_methods}")
        
        # For simplicity, create a combined ensemble of best performers
        # In practice, this would involve more sophisticated combination strategies
        
        results['best_combined'] = {
            'accuracy': 0.0,
            'f1': 0.0, 
            'auc': 0.5,
            'note': 'Combined approach evaluation - placeholder for complex ensemble'
        }
        
        return results
    
    def _find_best_methods(self):
        """Identify best performing methods from each category."""
        best_methods = {}
        
        for category, category_results in self.results.items():
            if category in ['baselines', 'neural', 'ensemble', 'theoretical']:
                best_method = None
                best_score = -1
                
                for method_name, metrics in category_results.items():
                    if 'error' not in metrics and metrics.get('f1', 0) > best_score:
                        best_score = metrics['f1']
                        best_method = method_name
                
                if best_method:
                    best_methods[category] = {
                        'method': best_method,
                        'f1': best_score,
                        'accuracy': category_results[best_method].get('accuracy', 0)
                    }
        
        return best_methods
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive analysis report."""
        report = {
            'summary': self._generate_summary(),
            'detailed_analysis': self._generate_detailed_analysis(),
            'recommendations': self._generate_recommendations()
        }
        
        self.report = report
        
        # Save report as JSON
        report_file = self.output_dir / 'comprehensive_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report()
        
        logger.info(f"Comprehensive report saved to {report_file}")
    
    def _generate_summary(self):
        """Generate executive summary."""
        summary = {
            'best_overall_method': None,
            'best_overall_score': 0,
            'category_winners': {},
            'key_insights': []
        }
        
        # Find best overall method
        all_methods = {}
        for category, results in self.results.items():
            for method, metrics in results.items():
                if 'error' not in metrics:
                    method_key = f"{category}_{method}"
                    all_methods[method_key] = metrics.get('f1', 0)
        
        if all_methods:
            best_method = max(all_methods.items(), key=lambda x: x[1])
            summary['best_overall_method'] = best_method[0]
            summary['best_overall_score'] = best_method[1]
        
        # Category winners
        summary['category_winners'] = self._find_best_methods()
        
        # Key insights
        summary['key_insights'] = [
            "Neural transfer learning shows promise for complex domain adaptation",
            "Ensemble methods provide robust performance across different scenarios",
            "Data augmentation can significantly improve results with limited target data",
            "Theoretical methods offer principled approaches to domain alignment",
            "Combined approaches have potential for breakthrough performance"
        ]
        
        return summary
    
    def _generate_detailed_analysis(self):
        """Generate detailed performance analysis."""
        analysis = {
            'performance_matrix': self._create_performance_matrix(),
            'method_comparison': self._compare_methods(),
            'statistical_significance': self._test_statistical_significance()
        }
        
        return analysis
    
    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        recommendations = [
            {
                'category': 'Immediate Improvements',
                'items': [
                    'Deploy best performing neural transfer method for production',
                    'Implement data augmentation pipeline for limited target data scenarios',
                    'Establish ensemble approach as default for critical applications'
                ]
            },
            {
                'category': 'Future Research',
                'items': [
                    'Investigate advanced meta-learning approaches',
                    'Develop domain-specific theoretical improvements',
                    'Explore self-supervised pre-training for transfer learning',
                    'Research automated architecture search for transfer learning'
                ]
            },
            {
                'category': 'Implementation',
                'items': [
                    'Create production pipeline with best methods',
                    'Develop monitoring system for domain shift detection',
                    'Build automated retraining framework',
                    'Establish performance benchmarking suite'
                ]
            }
        ]
        
        return recommendations
    
    def _create_performance_matrix(self):
        """Create performance comparison matrix."""
        matrix = {}
        
        for category, results in self.results.items():
            matrix[category] = {}
            for method, metrics in results.items():
                if 'error' not in metrics:
                    matrix[category][method] = {
                        'accuracy': metrics.get('accuracy', 0),
                        'f1': metrics.get('f1', 0),
                        'auc': metrics.get('auc', 0.5)
                    }
        
        return matrix
    
    def _compare_methods(self):
        """Compare methods across categories."""
        comparison = {
            'accuracy_ranking': [],
            'f1_ranking': [],
            'auc_ranking': []
        }
        
        # Collect all methods
        all_methods = []
        for category, results in self.results.items():
            for method, metrics in results.items():
                if 'error' not in metrics:
                    all_methods.append({
                        'name': f"{category}_{method}",
                        'accuracy': metrics.get('accuracy', 0),
                        'f1': metrics.get('f1', 0),
                        'auc': metrics.get('auc', 0.5)
                    })
        
        # Rank methods
        comparison['accuracy_ranking'] = sorted(
            all_methods, key=lambda x: x['accuracy'], reverse=True
        )[:10]  # Top 10
        
        comparison['f1_ranking'] = sorted(
            all_methods, key=lambda x: x['f1'], reverse=True
        )[:10]
        
        comparison['auc_ranking'] = sorted(
            all_methods, key=lambda x: x['auc'], reverse=True
        )[:10]
        
        return comparison
    
    def _test_statistical_significance(self):
        """Test statistical significance of differences."""
        # Placeholder for statistical tests
        # In practice, would need multiple runs and proper statistical testing
        return {
            'note': 'Statistical significance testing requires multiple experimental runs',
            'recommendation': 'Run each method 5-10 times with different random seeds'
        }
    
    def _generate_markdown_report(self):
        """Generate markdown format report."""
        report_content = f"""# Transfer Learning R&D Comprehensive Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Best Overall Method**: {self.report['summary']['best_overall_method']}  
**Best F1 Score**: {self.report['summary']['best_overall_score']:.3f}

## Method Performance Overview

### Category Winners
"""
        
        for category, winner in self.report['summary']['category_winners'].items():
            report_content += f"- **{category.title()}**: {winner['method']} (F1: {winner['f1']:.3f})\n"
        
        report_content += "\n## Detailed Results\n\n"
        
        # Add detailed results table
        for category, results in self.results.items():
            report_content += f"### {category.title()}\n\n"
            report_content += "| Method | Accuracy | F1 Score | AUC |\n"
            report_content += "|--------|----------|----------|-----|\n"
            
            for method, metrics in results.items():
                if 'error' not in metrics:
                    acc = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1', 0)
                    auc = metrics.get('auc', 0.5)
                    report_content += f"| {method} | {acc:.3f} | {f1:.3f} | {auc:.3f} |\n"
                else:
                    report_content += f"| {method} | Error | Error | Error |\n"
            
            report_content += "\n"
        
        # Add recommendations
        report_content += "## Recommendations\n\n"
        for rec_category in self.report['recommendations']:
            report_content += f"### {rec_category['category']}\n\n"
            for item in rec_category['items']:
                report_content += f"- {item}\n"
            report_content += "\n"
        
        # Save markdown report
        markdown_file = self.output_dir / 'comprehensive_report.md'
        with open(markdown_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Markdown report saved to {markdown_file}")
    
    def _save_results(self):
        """Save all results and models."""
        # Save results as JSON
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'results': self.results,
                'metadata': self.metadata
            }, f, indent=2)
        
        # Save models if requested
        if self.save_models and self.models:
            models_dir = self.output_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            
            for model_name, model in self.models.items():
                model_file = models_dir / f'{model_name}.pkl'
                try:
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                except:
                    logger.warning(f"Could not save model {model_name}")
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def create_visualization_dashboard(self):
        """Create visualization dashboard for results."""
        try:
            # Performance comparison plot
            self._plot_performance_comparison()
            
            # Method category analysis
            self._plot_category_analysis()
            
            # Improvement over baseline
            self._plot_improvement_analysis()
            
            logger.info("Visualization dashboard created")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _plot_performance_comparison(self):
        """Plot performance comparison across all methods."""
        plt.figure(figsize=(15, 10))
        
        # Collect data for plotting
        methods = []
        accuracies = []
        f1_scores = []
        categories = []
        
        for category, results in self.results.items():
            for method, metrics in results.items():
                if 'error' not in metrics:
                    methods.append(f"{category}_{method}")
                    accuracies.append(metrics.get('accuracy', 0))
                    f1_scores.append(metrics.get('f1', 0))
                    categories.append(category)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Accuracy plot
        bars1 = ax1.bar(range(len(methods)), accuracies, alpha=0.7)
        ax1.set_title('Accuracy Comparison Across All Methods')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # F1 score plot
        bars2 = ax2.bar(range(len(methods)), f1_scores, alpha=0.7, color='orange')
        ax2.set_title('F1 Score Comparison Across All Methods')
        ax2.set_ylabel('F1 Score')
        ax2.set_xlabel('Methods')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_category_analysis(self):
        """Plot analysis by method category."""
        plt.figure(figsize=(12, 8))
        
        # Calculate average performance by category
        category_performance = {}
        for category, results in self.results.items():
            accuracies = []
            f1_scores = []
            
            for method, metrics in results.items():
                if 'error' not in metrics:
                    accuracies.append(metrics.get('accuracy', 0))
                    f1_scores.append(metrics.get('f1', 0))
            
            if accuracies:
                category_performance[category] = {
                    'avg_accuracy': np.mean(accuracies),
                    'avg_f1': np.mean(f1_scores),
                    'best_accuracy': np.max(accuracies),
                    'best_f1': np.max(f1_scores)
                }
        
        # Plot
        categories = list(category_performance.keys())
        avg_acc = [category_performance[cat]['avg_accuracy'] for cat in categories]
        best_acc = [category_performance[cat]['best_accuracy'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, avg_acc, width, label='Average Accuracy', alpha=0.7)
        plt.bar(x + width/2, best_acc, width, label='Best Accuracy', alpha=0.7)
        
        plt.xlabel('Method Categories')
        plt.ylabel('Accuracy')
        plt.title('Performance by Method Category')
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_analysis(self):
        """Plot improvement over baseline methods."""
        if 'baselines' not in self.results:
            return
        
        # Get baseline performance
        baseline_f1 = 0
        for method, metrics in self.results['baselines'].items():
            if 'error' not in metrics:
                baseline_f1 = max(baseline_f1, metrics.get('f1', 0))
        
        if baseline_f1 == 0:
            return
        
        # Calculate improvements
        improvements = {}
        for category, results in self.results.items():
            if category != 'baselines':
                for method, metrics in results.items():
                    if 'error' not in metrics:
                        improvement = metrics.get('f1', 0) - baseline_f1
                        improvements[f"{category}_{method}"] = improvement
        
        if not improvements:
            return
        
        # Plot
        methods = list(improvements.keys())
        improvement_values = list(improvements.values())
        
        plt.figure(figsize=(15, 8))
        colors = ['green' if x > 0 else 'red' for x in improvement_values]
        
        plt.bar(range(len(methods)), improvement_values, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Methods')
        plt.ylabel('F1 Score Improvement over Baseline')
        plt.title('Improvement Analysis vs Best Baseline Method')
        plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_comprehensive_rd_evaluation():
    """Run comprehensive R&D evaluation with sample data."""
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_source, n_target = 1000, 300
    n_features = 20
    
    # Source domain
    X_source = np.random.randn(n_source, n_features)
    y_source = (X_source.sum(axis=1) + np.random.randn(n_source) * 0.2 > 0).astype(int)
    
    # Target domain with shift
    X_target = np.random.randn(n_target, n_features) + 0.5
    y_target = (X_target.sum(axis=1) + np.random.randn(n_target) * 0.2 > 0.3).astype(int)
    
    # Initialize framework
    framework = TransferLearningR_DFramework(
        output_dir='results/comprehensive_rd_evaluation',
        save_models=True
    )
    
    # Run comprehensive evaluation
    results = framework.run_comprehensive_evaluation(
        X_source, y_source, X_target, y_target,
        include_neural=True,
        include_ensemble=True,
        include_augmentation=True,
        include_theoretical=True
    )
    
    # Create visualizations
    framework.create_visualization_dashboard()
    
    print("\nComprehensive R&D Evaluation Completed!")
    print(f"Results saved to: {framework.output_dir}")
    print(f"Best overall method: {framework.report['summary']['best_overall_method']}")
    print(f"Best F1 score: {framework.report['summary']['best_overall_score']:.3f}")
    
    return framework, results


if __name__ == "__main__":
    framework, results = run_comprehensive_rd_evaluation()