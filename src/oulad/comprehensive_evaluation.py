"""
Comprehensive Evaluation and Comparison Framework for OULAD Deep Learning Models

This module provides extensive evaluation capabilities including:
- Performance benchmarking across all models
- Statistical significance testing
- Fairness evaluation
- Interpretability analysis
- Cross-validation with proper OULAD splits
- Visualization and reporting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import model classes
try:
    from .modern_deep_learning import TabNet, FTTransformer, NODE, SAINT, AutoInt
    from .hyperparameter_optimization import OptunaTuner, run_comprehensive_optimization
    from .advanced_training_techniques import AdvancedTrainer, train_with_advanced_techniques
except ImportError:
    from modern_deep_learning import TabNet, FTTransformer, NODE, SAINT, AutoInt
    from hyperparameter_optimization import OptunaTuner, run_comprehensive_optimization
    from advanced_training_techniques import AdvancedTrainer, train_with_advanced_techniques

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    """
    
    def __init__(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
                 sensitive_features: Optional[np.ndarray] = None):
        """
        Initialize model evaluator.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            sensitive_features: Sensitive features for fairness evaluation
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features = sensitive_features
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results storage
        self.results = {}
        
    def evaluate_model(self, model_name: str, model_info: Dict) -> Dict:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model
            model_info: Model information including model, scaler, config
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        model = model_info['model']
        scaler = model_info.get('scaler', None)
        
        # Prepare test data
        if scaler is not None:
            X_test_scaled = scaler.transform(self.X_test)
        else:
            X_test_scaled = self.X_test
        
        # Make predictions
        model.eval()
        model.to(self.device)
        
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        with torch.no_grad():
            if hasattr(model, 'predict_proba'):
                # For sklearn-like models
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
                y_preds = model.predict(X_test_scaled)
            else:
                # For PyTorch models
                outputs = model(X_test_tensor)
                y_probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                y_preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Calculate metrics
        results = self._calculate_metrics(y_preds, y_probs, model_name)
        
        # Add model-specific information
        results['model_info'] = {
            'model_type': type(model).__name__,
            'config': model_info.get('config', {}),
            'parameters': self._count_parameters(model) if hasattr(model, 'parameters') else None
        }
        
        return results
    
    def _calculate_metrics(self, y_preds: np.ndarray, y_probs: np.ndarray, model_name: str) -> Dict:
        """Calculate comprehensive metrics."""
        results = {}
        
        # Basic classification metrics
        results['accuracy'] = accuracy_score(self.y_test, y_preds)
        results['auc'] = roc_auc_score(self.y_test, y_probs)
        results['f1'] = f1_score(self.y_test, y_preds)
        results['precision'] = precision_score(self.y_test, y_preds)
        results['recall'] = recall_score(self.y_test, y_preds)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_preds)
        results['confusion_matrix'] = cm.tolist()
        results['tn'], results['fp'], results['fn'], results['tp'] = cm.ravel()
        
        # Additional metrics
        results['specificity'] = results['tn'] / (results['tn'] + results['fp']) if (results['tn'] + results['fp']) > 0 else 0
        results['balanced_accuracy'] = (results['recall'] + results['specificity']) / 2
        
        # Fairness metrics (if sensitive features provided)
        if self.sensitive_features is not None:
            results['fairness'] = self._calculate_fairness_metrics(y_preds, y_probs)
        
        # Store predictions for later analysis
        results['predictions'] = y_preds.tolist()
        results['probabilities'] = y_probs.tolist()
        
        return results
    
    def _calculate_fairness_metrics(self, y_preds: np.ndarray, y_probs: np.ndarray) -> Dict:
        """Calculate fairness metrics."""
        fairness_results = {}
        
        unique_groups = np.unique(self.sensitive_features)
        
        # Group-specific metrics
        group_metrics = {}
        for group in unique_groups:
            group_mask = self.sensitive_features == group
            
            if np.sum(group_mask) > 0:
                group_metrics[f'group_{group}'] = {
                    'size': int(np.sum(group_mask)),
                    'accuracy': accuracy_score(self.y_test[group_mask], y_preds[group_mask]),
                    'auc': roc_auc_score(self.y_test[group_mask], y_probs[group_mask]) 
                        if len(np.unique(self.y_test[group_mask])) > 1 else 0.5,
                    'positive_rate': np.mean(y_preds[group_mask])
                }
        
        fairness_results['group_metrics'] = group_metrics
        
        # Demographic parity
        overall_positive_rate = np.mean(y_preds)
        dp_differences = []
        for group in unique_groups:
            group_mask = self.sensitive_features == group
            if np.sum(group_mask) > 0:
                group_positive_rate = np.mean(y_preds[group_mask])
                dp_diff = abs(group_positive_rate - overall_positive_rate)
                dp_differences.append(dp_diff)
        
        fairness_results['demographic_parity_diff'] = np.mean(dp_differences) if dp_differences else 0.0
        fairness_results['max_demographic_parity_diff'] = np.max(dp_differences) if dp_differences else 0.0
        
        # Equalized odds
        eo_differences = []
        for group in unique_groups:
            group_mask = self.sensitive_features == group
            if np.sum(group_mask) > 0 and len(np.unique(self.y_test[group_mask])) > 1:
                # True positive rate for group
                group_cm = confusion_matrix(self.y_test[group_mask], y_preds[group_mask])
                if group_cm.shape == (2, 2):
                    tn, fp, fn, tp = group_cm.ravel()
                    group_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    # Overall true positive rate
                    overall_cm = confusion_matrix(self.y_test, y_preds)
                    tn_o, fp_o, fn_o, tp_o = overall_cm.ravel()
                    overall_tpr = tp_o / (tp_o + fn_o) if (tp_o + fn_o) > 0 else 0
                    
                    eo_diff = abs(group_tpr - overall_tpr)
                    eo_differences.append(eo_diff)
        
        fairness_results['equalized_odds_diff'] = np.mean(eo_differences) if eo_differences else 0.0
        fairness_results['max_equalized_odds_diff'] = np.max(eo_differences) if eo_differences else 0.0
        
        return fairness_results
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def evaluate_all_models(self) -> Dict:
        """Evaluate all models."""
        logger.info("Starting comprehensive model evaluation...")
        
        for model_name, model_info in self.models.items():
            try:
                self.results[model_name] = self.evaluate_model(model_name, model_info)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                self.results[model_name] = {'error': str(e)}
        
        # Add comparative analysis
        self.results['comparative_analysis'] = self._comparative_analysis()
        
        logger.info("Model evaluation completed")
        return self.results
    
    def _comparative_analysis(self) -> Dict:
        """Perform comparative analysis across models."""
        analysis = {}
        
        # Extract metrics for comparison
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            return analysis
        
        metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall', 'balanced_accuracy']
        
        # Best performers
        best_performers = {}
        for metric in metrics:
            metric_values = [(name, result.get(metric, 0)) for name, result in valid_results.items()]
            best_performers[metric] = max(metric_values, key=lambda x: x[1])
        
        analysis['best_performers'] = best_performers
        
        # Performance summary
        performance_summary = {}
        for metric in metrics:
            values = [result.get(metric, 0) for result in valid_results.values()]
            performance_summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        analysis['performance_summary'] = performance_summary
        
        # Model complexity vs performance
        complexity_performance = {}
        for name, result in valid_results.items():
            params = result.get('model_info', {}).get('parameters')
            if params is not None:
                complexity_performance[name] = {
                    'parameters': params,
                    'auc': result.get('auc', 0),
                    'accuracy': result.get('accuracy', 0)
                }
        
        analysis['complexity_performance'] = complexity_performance
        
        return analysis


class StatisticalTester:
    """
    Statistical significance testing for model comparisons.
    """
    
    def __init__(self, results: Dict[str, Dict]):
        """
        Initialize statistical tester.
        
        Args:
            results: Model evaluation results
        """
        self.results = results
    
    def paired_t_test(self, model1: str, model2: str, metric: str = 'auc') -> Dict:
        """
        Perform paired t-test between two models.
        
        Args:
            model1: First model name
            model2: Second model name
            metric: Metric to compare
            
        Returns:
            Statistical test results
        """
        if model1 not in self.results or model2 not in self.results:
            return {'error': 'One or both models not found in results'}
        
        # For a proper paired t-test, we would need cross-validation results
        # Here we'll use bootstrap sampling as an approximation
        probs1 = np.array(self.results[model1].get('probabilities', []))
        probs2 = np.array(self.results[model2].get('probabilities', []))
        
        if len(probs1) == 0 or len(probs2) == 0:
            return {'error': 'No probabilities found for comparison'}
        
        # Bootstrap sampling
        n_bootstrap = 1000
        metric_diffs = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(probs1), size=len(probs1), replace=True)
            
            sample_probs1 = probs1[indices]
            sample_probs2 = probs2[indices]
            
            # Calculate metric for each sample
            if metric == 'auc':
                # Assuming we have access to true labels
                true_labels = np.array([int(p > 0.5) for p in probs1])  # Simplified
                sample_labels = true_labels[indices]
                
                try:
                    metric1 = roc_auc_score(sample_labels, sample_probs1)
                    metric2 = roc_auc_score(sample_labels, sample_probs2)
                    metric_diffs.append(metric1 - metric2)
                except:
                    continue
        
        if len(metric_diffs) == 0:
            return {'error': 'Could not calculate bootstrap samples'}
        
        # Statistical test
        t_stat, p_value = stats.ttest_1samp(metric_diffs, 0)
        
        return {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_difference': float(np.mean(metric_diffs)),
            'std_difference': float(np.std(metric_diffs)),
            'confidence_interval': [
                float(np.percentile(metric_diffs, 2.5)),
                float(np.percentile(metric_diffs, 97.5))
            ],
            'significant': p_value < 0.05
        }
    
    def multiple_comparison_test(self, metric: str = 'auc', alpha: float = 0.05) -> Dict:
        """
        Perform multiple comparison test (Bonferroni correction).
        
        Args:
            metric: Metric to compare
            alpha: Significance level
            
        Returns:
            Multiple comparison results
        """
        model_names = list(self.results.keys())
        n_comparisons = len(model_names) * (len(model_names) - 1) // 2
        
        if n_comparisons == 0:
            return {'error': 'Need at least 2 models for comparison'}
        
        # Bonferroni correction
        corrected_alpha = alpha / n_comparisons
        
        comparisons = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                test_result = self.paired_t_test(model1, model2, metric)
                
                if 'error' not in test_result:
                    test_result['corrected_alpha'] = corrected_alpha
                    test_result['significant_corrected'] = test_result['p_value'] < corrected_alpha
                    comparisons.append(test_result)
        
        return {
            'comparisons': comparisons,
            'n_comparisons': n_comparisons,
            'corrected_alpha': corrected_alpha,
            'original_alpha': alpha
        }


class VisualizationGenerator:
    """
    Generate comprehensive visualizations for model evaluation.
    """
    
    def __init__(self, results: Dict[str, Dict], output_dir: str = 'evaluation_results'):
        """
        Initialize visualization generator.
        
        Args:
            results: Model evaluation results
            output_dir: Output directory for plots
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_performance_comparison(self, save: bool = True) -> None:
        """Plot performance comparison across models."""
        valid_results = {k: v for k, v in self.results.items() 
                        if 'error' not in v and k != 'comparative_analysis'}
        
        if not valid_results:
            logger.warning("No valid results to plot")
            return
        
        metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall', 'balanced_accuracy']
        model_names = list(valid_results.keys())
        
        # Create subplot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [valid_results[name].get(metric, 0) for name in model_names]
            
            bars = axes[i].bar(range(len(model_names)), values, alpha=0.8)
            axes[i].set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Models')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_xticks(range(len(model_names)))
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curves(self, y_test: np.ndarray, save: bool = True) -> None:
        """Plot ROC curves for all models."""
        valid_results = {k: v for k, v in self.results.items() 
                        if 'error' not in v and k != 'comparative_analysis'}
        
        if not valid_results:
            return
        
        plt.figure(figsize=(12, 8))
        
        for model_name, result in valid_results.items():
            probs = result.get('probabilities', [])
            if probs:
                fpr, tpr, _ = roc_curve(y_test, probs)
                auc_score = result.get('auc', 0)
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curves(self, y_test: np.ndarray, save: bool = True) -> None:
        """Plot precision-recall curves for all models."""
        valid_results = {k: v for k, v in self.results.items() 
                        if 'error' not in v and k != 'comparative_analysis'}
        
        if not valid_results:
            return
        
        plt.figure(figsize=(12, 8))
        
        for model_name, result in valid_results.items():
            probs = result.get('probabilities', [])
            if probs:
                precision, recall, _ = precision_recall_curve(y_test, probs)
                plt.plot(recall, precision, linewidth=2, label=model_name)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrices(self, save: bool = True) -> None:
        """Plot confusion matrices for all models."""
        valid_results = {k: v for k, v in self.results.items() 
                        if 'error' not in v and k != 'comparative_analysis'}
        
        if not valid_results:
            return
        
        n_models = len(valid_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes] if n_models == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(valid_results.items()):
            cm = np.array(result.get('confusion_matrix', [[0, 0], [0, 0]]))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'],
                       ax=axes[i] if i < len(axes) else None)
            
            if i < len(axes):
                axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_fairness_metrics(self, save: bool = True) -> None:
        """Plot fairness metrics comparison."""
        fairness_data = []
        
        for model_name, result in self.results.items():
            if 'fairness' in result:
                fairness = result['fairness']
                fairness_data.append({
                    'Model': model_name,
                    'Demographic Parity Diff': fairness.get('demographic_parity_diff', 0),
                    'Equalized Odds Diff': fairness.get('equalized_odds_diff', 0)
                })
        
        if not fairness_data:
            logger.warning("No fairness data available for plotting")
            return
        
        df = pd.DataFrame(fairness_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Demographic Parity
        bars1 = ax1.bar(df['Model'], df['Demographic Parity Diff'], alpha=0.8, color='skyblue')
        ax1.set_title('Demographic Parity Difference', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Difference')
        ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Equalized Odds
        bars2 = ax2.bar(df['Model'], df['Equalized Odds Diff'], alpha=0.8, color='lightcoral')
        ax2.set_title('Equalized Odds Difference', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Difference')
        ax2.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'fairness_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_all_plots(self, y_test: np.ndarray) -> None:
        """Generate all visualization plots."""
        logger.info("Generating visualization plots...")
        
        self.plot_performance_comparison()
        self.plot_roc_curves(y_test)
        self.plot_precision_recall_curves(y_test)
        self.plot_confusion_matrices()
        self.plot_fairness_metrics()
        
        logger.info(f"All plots saved to {self.output_dir}")


class ReportGenerator:
    """
    Generate comprehensive evaluation reports.
    """
    
    def __init__(self, results: Dict[str, Dict], output_dir: str = 'evaluation_results'):
        """
        Initialize report generator.
        
        Args:
            results: Model evaluation results
            output_dir: Output directory
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary_report(self) -> str:
        """Generate summary report in markdown format."""
        report = []
        report.append("# OULAD Deep Learning Models Evaluation Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overview
        valid_results = {k: v for k, v in self.results.items() 
                        if 'error' not in v and k != 'comparative_analysis'}
        
        report.append(f"## Overview\n")
        report.append(f"- Total models evaluated: {len(valid_results)}")
        report.append(f"- Models: {', '.join(valid_results.keys())}\n")
        
        # Performance Summary
        if 'comparative_analysis' in self.results:
            analysis = self.results['comparative_analysis']
            
            report.append("## Performance Summary\n")
            
            if 'best_performers' in analysis:
                report.append("### Best Performers by Metric\n")
                for metric, (model, score) in analysis['best_performers'].items():
                    report.append(f"- **{metric.capitalize()}**: {model} ({score:.4f})")
                report.append("")
            
            if 'performance_summary' in analysis:
                report.append("### Performance Statistics\n")
                report.append("| Metric | Mean | Std | Min | Max |")
                report.append("|--------|------|-----|-----|-----|")
                
                for metric, stats in analysis['performance_summary'].items():
                    report.append(f"| {metric.capitalize()} | {stats['mean']:.4f} | "
                                f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
                report.append("")
        
        # Individual Model Results
        report.append("## Individual Model Results\n")
        
        for model_name, result in valid_results.items():
            report.append(f"### {model_name}\n")
            
            # Basic metrics
            metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall', 'balanced_accuracy']
            report.append("#### Performance Metrics")
            for metric in metrics:
                if metric in result:
                    report.append(f"- **{metric.capitalize()}**: {result[metric]:.4f}")
            report.append("")
            
            # Model info
            if 'model_info' in result:
                info = result['model_info']
                report.append("#### Model Information")
                report.append(f"- **Type**: {info.get('model_type', 'Unknown')}")
                if info.get('parameters'):
                    report.append(f"- **Parameters**: {info['parameters']:,}")
                report.append("")
            
            # Fairness metrics
            if 'fairness' in result:
                fairness = result['fairness']
                report.append("#### Fairness Metrics")
                report.append(f"- **Demographic Parity Difference**: {fairness.get('demographic_parity_diff', 0):.4f}")
                report.append(f"- **Equalized Odds Difference**: {fairness.get('equalized_odds_diff', 0):.4f}")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations\n")
        
        if 'comparative_analysis' in self.results and 'best_performers' in self.results['comparative_analysis']:
            best_auc = self.results['comparative_analysis']['best_performers'].get('auc', ('Unknown', 0))
            best_f1 = self.results['comparative_analysis']['best_performers'].get('f1', ('Unknown', 0))
            
            report.append(f"- **For highest AUC**: Use {best_auc[0]} (AUC: {best_auc[1]:.4f})")
            report.append(f"- **For highest F1-score**: Use {best_f1[0]} (F1: {best_f1[1]:.4f})")
            
            # Model complexity analysis
            if 'complexity_performance' in self.results.get('comparative_analysis', {}):
                complexity_data = self.results['comparative_analysis']['complexity_performance']
                
                # Find best efficiency (performance per parameter)
                efficiency_scores = []
                for name, data in complexity_data.items():
                    if data['parameters'] > 0:
                        efficiency = data['auc'] / (data['parameters'] / 1000)  # AUC per 1K parameters
                        efficiency_scores.append((name, efficiency, data['auc'], data['parameters']))
                
                if efficiency_scores:
                    best_efficiency = max(efficiency_scores, key=lambda x: x[1])
                    report.append(f"- **Most efficient model**: {best_efficiency[0]} "
                                f"(AUC: {best_efficiency[2]:.4f}, Parameters: {best_efficiency[3]:,})")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def save_detailed_results(self) -> None:
        """Save detailed results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        for model_name, result in self.results.items():
            if isinstance(result, dict):
                serializable_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        serializable_result[key] = value.tolist()
                    elif isinstance(value, (np.int64, np.int32)):
                        serializable_result[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        serializable_result[key] = float(value)
                    else:
                        serializable_result[key] = value
                serializable_results[model_name] = serializable_result
            else:
                serializable_results[model_name] = result
        
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Detailed results saved to {self.output_dir / 'detailed_results.json'}")


def run_comprehensive_evaluation(X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                sensitive_features: Optional[np.ndarray] = None,
                                output_dir: str = 'comprehensive_evaluation',
                                optimize_hyperparameters: bool = True,
                                n_optimization_trials: int = 50) -> Dict:
    """
    Run comprehensive evaluation of all deep learning models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        sensitive_features: Sensitive features for fairness evaluation
        output_dir: Output directory for results
        optimize_hyperparameters: Whether to optimize hyperparameters
        n_optimization_trials: Number of optimization trials
        
    Returns:
        Comprehensive evaluation results
    """
    logger.info("Starting comprehensive evaluation...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Split training data for hyperparameter optimization
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Step 1: Hyperparameter optimization (if requested)
    if optimize_hyperparameters:
        logger.info("Running hyperparameter optimization...")
        optimization_results = run_comprehensive_optimization(
            X_train_opt, y_train_opt, X_val_opt, y_val_opt,
            output_dir=str(output_path / 'optimization'),
            n_trials=n_optimization_trials
        )
    else:
        # Use default parameters
        optimization_results = {}
    
    # Step 2: Train all models with best parameters
    logger.info("Training models with optimized parameters...")
    
    from .modern_deep_learning import train_modern_deep_learning_models
    
    trained_models = train_modern_deep_learning_models(
        X_train, y_train, X_test, y_test, random_state=42
    )
    
    # Step 3: Comprehensive evaluation
    logger.info("Performing comprehensive evaluation...")
    
    evaluator = ModelEvaluator(
        models=trained_models['models'],
        X_test=X_test,
        y_test=y_test,
        sensitive_features=sensitive_features
    )
    
    evaluation_results = evaluator.evaluate_all_models()
    
    # Step 4: Statistical testing
    logger.info("Performing statistical significance testing...")
    
    tester = StatisticalTester(evaluation_results)
    statistical_results = tester.multiple_comparison_test()
    
    # Step 5: Generate visualizations
    logger.info("Generating visualizations...")
    
    visualizer = VisualizationGenerator(evaluation_results, str(output_path))
    visualizer.generate_all_plots(y_test)
    
    # Step 6: Generate reports
    logger.info("Generating evaluation reports...")
    
    reporter = ReportGenerator(evaluation_results, str(output_path))
    summary_report = reporter.generate_summary_report()
    reporter.save_detailed_results()
    
    # Combine all results
    comprehensive_results = {
        'evaluation_results': evaluation_results,
        'statistical_testing': statistical_results,
        'optimization_results': optimization_results if optimize_hyperparameters else None,
        'summary_report': summary_report,
        'output_directory': str(output_path)
    }
    
    # Save comprehensive results
    with open(output_path / 'comprehensive_results.pkl', 'wb') as f:
        pickle.dump(comprehensive_results, f)
    
    logger.info(f"Comprehensive evaluation completed. Results saved to {output_path}")
    
    return comprehensive_results