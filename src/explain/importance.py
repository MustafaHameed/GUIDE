"""
Model Explainability and Importance Analysis

Provides SHAP global and local explanations for tree models, with fallback to
permutation importance for other models. Includes faithfulness checks and
stability analysis by demographic groups.

References:
- SHAP documentation: https://shap.readthedocs.io/
- SHAP GitHub: https://github.com/shap/shap
- LIME documentation: https://lime-ml.readthedocs.io/
- LIME readthedocs: https://lime.readthedocs.io/en/latest/
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from logging_config import setup_logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# SHAP imports
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None

# LIME imports  
try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    lime = None

# Optional PD/ICE utilities
try:
    from sklearn.inspection import PartialDependenceDisplay
    HAS_PD = True
except ImportError:
    HAS_PD = False

# Configure logging
logger = logging.getLogger(__name__)


class ExplainabilityAnalyzer:
    """Comprehensive explainability analysis for machine learning models."""
    
    def __init__(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: pd.Series, y_test: pd.Series,
                 sensitive_features: Optional[pd.Series] = None,
                 feature_names: Optional[List[str]] = None):
        """Initialize explainability analyzer.
        
        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features  
            y_train: Training labels
            y_test: Test labels
            sensitive_features: Sensitive attribute values for stability analysis
            feature_names: Feature names for display
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sensitive_features = sensitive_features
        self.feature_names = feature_names or list(X_train.columns)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None
        
        logger.info(f"Initialized explainability analyzer for {type(model).__name__}")
        
    def _is_tree_model(self) -> bool:
        """Check if model is tree-based for SHAP TreeExplainer."""
        tree_models = (
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'DecisionTreeClassifier', 'DecisionTreeRegressor'
        )
        return type(self.model).__name__ in tree_models
    
    def setup_shap_explainer(self, max_samples: int = 100) -> None:
        """Setup SHAP explainer based on model type.
        
        Args:
            max_samples: Maximum samples for background dataset
        """
        if not HAS_SHAP:
            logger.warning("SHAP not available, skipping SHAP analysis")
            return
            
        logger.info("Setting up SHAP explainer...")
        
        # Sample background data for efficiency
        if len(self.X_train) > max_samples:
            background_data = self.X_train.sample(n=max_samples, random_state=42)
        else:
            background_data = self.X_train
        
        try:
            if self._is_tree_model():
                # Use TreeExplainer for tree models
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.info("Using SHAP TreeExplainer")
            else:
                # Use KernelExplainer for other models
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    background_data
                )
                logger.info("Using SHAP KernelExplainer")
                
        except Exception as e:
            logger.warning(f"Failed to setup SHAP explainer: {e}")
            self.shap_explainer = None
    
    def setup_lime_explainer(self) -> None:
        """Setup LIME explainer for tabular data."""
        if not HAS_LIME:
            logger.warning("LIME not available, skipping LIME analysis")
            return
            
        logger.info("Setting up LIME explainer...")
        
        try:
            # Determine categorical features (non-numeric)
            categorical_features = []
            for i, col in enumerate(self.X_train.columns):
                if self.X_train[col].dtype == 'object':
                    categorical_features.append(i)
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train.values,
                feature_names=self.feature_names,
                categorical_features=categorical_features,
                class_names=['Fail', 'Pass'],
                mode='classification'
            )
            logger.info("LIME explainer setup complete")
            
        except Exception as e:
            logger.warning(f"Failed to setup LIME explainer: {e}")
            self.lime_explainer = None
    
    def compute_shap_values(self, max_samples: int = 100) -> Optional[np.ndarray]:
        """Compute SHAP values for test set.
        
        Args:
            max_samples: Maximum test samples to explain
            
        Returns:
            SHAP values array or None if failed
        """
        if self.shap_explainer is None:
            return None
            
        logger.info("Computing SHAP values...")
        
        # Sample test data for efficiency
        if len(self.X_test) > max_samples:
            test_sample = self.X_test.sample(n=max_samples, random_state=42)
        else:
            test_sample = self.X_test
            
        try:
            if self._is_tree_model():
                # TreeExplainer returns values for all classes
                shap_values = self.shap_explainer.shap_values(test_sample)
                if isinstance(shap_values, list):
                    # For binary classification, take positive class
                    self.shap_values = shap_values[1] 
                else:
                    self.shap_values = shap_values
            else:
                # KernelExplainer
                self.shap_values = self.shap_explainer.shap_values(test_sample)
                
            logger.info(f"Computed SHAP values shape: {self.shap_values.shape}")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Failed to compute SHAP values: {e}")
            return None
    
    def global_feature_importance(self, method: str = 'shap') -> pd.DataFrame:
        """Compute global feature importance ranking.
        
        Args:
            method: Method to use ('shap' or 'permutation')
            
        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Computing global feature importance using {method}")
        
        if method == 'shap' and self.shap_values is not None:
            # SHAP-based importance (mean absolute SHAP values)
            importance_scores = np.abs(self.shap_values).mean(axis=0)
            
        elif method == 'permutation' or self.shap_values is None:
            # Fallback to permutation importance
            logger.info("Using permutation importance")
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test, 
                n_repeats=10, random_state=42
            )
            importance_scores = perm_importance.importances_mean
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def local_explanations(self, instance_indices: List[int], 
                          method: str = 'shap', top_k: int = 10) -> Dict[int, Dict]:
        """Generate local explanations for specific instances.
        
        Args:
            instance_indices: Indices of instances to explain
            method: Explanation method ('shap' or 'lime')
            top_k: Number of top features to include
            
        Returns:
            Dictionary mapping instance index to explanation
        """
        logger.info(f"Generating local explanations for {len(instance_indices)} instances using {method}")
        
        explanations = {}
        
        for idx in instance_indices:
            try:
                if method == 'shap' and self.shap_values is not None:
                    # SHAP explanation
                    if idx < len(self.shap_values):
                        feature_contributions = self.shap_values[idx]
                        
                        # Get top-k features by absolute contribution
                        top_indices = np.argsort(np.abs(feature_contributions))[-top_k:][::-1]
                        
                        explanations[idx] = {
                            'features': [self.feature_names[i] for i in top_indices],
                            'contributions': feature_contributions[top_indices].tolist(),
                            'values': self.X_test.iloc[idx][top_indices].tolist(),
                            'method': 'shap'
                        }
                
                elif method == 'lime' and self.lime_explainer is not None:
                    # LIME explanation
                    instance = self.X_test.iloc[idx].values
                    lime_exp = self.lime_explainer.explain_instance(
                        instance, 
                        self.model.predict_proba,
                        num_features=top_k
                    )
                    
                    explanations[idx] = {
                        'features': [self.feature_names[f] for f, _ in lime_exp.as_list()],
                        'contributions': [w for _, w in lime_exp.as_list()],
                        'values': [instance[f] for f, _ in lime_exp.as_list()],
                        'method': 'lime'
                    }
                
            except Exception as e:
                logger.warning(f"Failed to explain instance {idx}: {e}")
                
        return explanations
    
    def faithfulness_check(self, top_k: int = 5, method: str = 'remove') -> Dict[str, float]:
        """Check explanation faithfulness by perturbing top features.
        
        Args:
            top_k: Number of top features to perturb
            method: Perturbation method ('remove' or 'randomize')
            
        Returns:
            Dictionary with performance drops
        """
        logger.info(f"Performing faithfulness check with top-{top_k} features")
        
        # Get global feature importance
        importance_df = self.global_feature_importance()
        top_features = importance_df['feature'].head(top_k).tolist()
        
        # Get baseline performance
        baseline_preds = self.model.predict_proba(self.X_test)[:, 1]
        baseline_auc = roc_auc_score(self.y_test, baseline_preds)
        baseline_acc = accuracy_score(self.y_test, (baseline_preds > 0.5).astype(int))
        
        # Create perturbed dataset
        X_perturbed = self.X_test.copy()
        
        for feature in top_features:
            if feature in X_perturbed.columns:
                if method == 'remove':
                    # Set to zero/mean
                    if X_perturbed[feature].dtype in ['int64', 'float64']:
                        X_perturbed[feature] = X_perturbed[feature].mean()
                    else:
                        X_perturbed[feature] = X_perturbed[feature].mode()[0]
                elif method == 'randomize':
                    # Shuffle values
                    X_perturbed[feature] = np.random.permutation(X_perturbed[feature].values)
        
        # Get perturbed performance
        perturbed_preds = self.model.predict_proba(X_perturbed)[:, 1]
        perturbed_auc = roc_auc_score(self.y_test, perturbed_preds)
        perturbed_acc = accuracy_score(self.y_test, (perturbed_preds > 0.5).astype(int))
        
        results = {
            'auc_drop': baseline_auc - perturbed_auc,
            'accuracy_drop': baseline_acc - perturbed_acc,
            'baseline_auc': baseline_auc,
            'perturbed_auc': perturbed_auc,
            'top_features': top_features
        }
        
        logger.info(f"Faithfulness check: AUC drop = {results['auc_drop']:.3f}, "
                   f"Accuracy drop = {results['accuracy_drop']:.3f}")
        
        return results
    
    def stability_by_group(self, group_column: str = 'sex') -> Dict[str, pd.DataFrame]:
        """Analyze explanation stability across demographic groups.
        
        Args:
            group_column: Column name for grouping
            
        Returns:
            Dictionary mapping groups to feature importance distributions
        """
        if self.sensitive_features is None:
            logger.warning("No sensitive features provided for stability analysis")
            return {}
            
        logger.info(f"Analyzing stability by {group_column}")
        
        if self.shap_values is None:
            logger.warning("No SHAP values available for stability analysis")
            return {}
        
        # Align SHAP values with sensitive features
        test_indices = self.X_test.index
        aligned_sensitive = self.sensitive_features.loc[test_indices]
        
        stability_results = {}
        
        for group_value in aligned_sensitive.unique():
            if pd.isna(group_value):
                continue
                
            group_mask = (aligned_sensitive == group_value)
            group_shap = self.shap_values[group_mask.values]
            
            if len(group_shap) > 0:
                # Calculate feature importance distribution for this group
                group_importance = np.abs(group_shap).mean(axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance_mean': group_importance,
                    'importance_std': np.abs(group_shap).std(axis=0),
                    'group': group_value,
                    'n_samples': len(group_shap)
                })
                
                stability_results[group_value] = importance_df.sort_values('importance_mean', ascending=False)
        
        return stability_results
    
    def save_shap_plots(self, output_dir: Path, max_features: int = 20) -> None:
        """Save SHAP summary and dependence plots.
        
        Args:
            output_dir: Directory to save plots
            max_features: Maximum features to show in plots
        """
        if not HAS_SHAP or self.shap_values is None:
            logger.warning("SHAP not available or values not computed")
            return
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values, 
                self.X_test.iloc[:len(self.shap_values)], 
                feature_names=self.feature_names,
                max_display=max_features,
                show=False
            )
            plt.savefig(output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # SHAP dependence plots for top features
            importance_df = self.global_feature_importance(method='shap')
            top_features = importance_df['feature'].head(5).tolist()
            
            for feature in top_features:
                if feature in self.feature_names:
                    feature_idx = self.feature_names.index(feature)
                    
                    plt.figure(figsize=(8, 6))
                    shap.dependence_plot(
                        feature_idx,
                        self.shap_values,
                        self.X_test.iloc[:len(self.shap_values)],
                        feature_names=self.feature_names,
                        show=False
                    )
                    plt.savefig(output_dir / f'shap_dependence_{feature}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info(f"SHAP plots saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save SHAP plots: {e}")
    
    def save_pdp_ice_plots(self, output_dir: Path,
                            features: Optional[List[str]] = None) -> None:
        """Save partial dependence and ICE plots for selected features.

        Args:
            output_dir: Directory to save plots
            features: Optional list of feature names. If ``None``,
                the top three features by global importance are used.
        """
        if not HAS_PD:
            logger.warning("Partial dependence utilities not available")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        if features is None:
            importance_df = self.global_feature_importance()
            features = importance_df['feature'].head(3).tolist()

        for feature in features:
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                PartialDependenceDisplay.from_estimator(
                    self.model,
                    self.X_test,
                    [feature],
                    kind="both",
                    ax=ax
                )
                fig.savefig(output_dir / f'pdp_ice_{feature}.png',
                            dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to save PDP/ICE plot for {feature}: {e}")

    def save_lime_explanations(self, figures_dir: Path, reports_dir: Path,
                               instance_indices: List[int]) -> None:
        """Save LIME explanations to figures and reports directories.

        Args:
            figures_dir: Directory to save PNG figures
            reports_dir: Directory to save HTML reports
            instance_indices: Indices of instances to explain
        """
        if not HAS_LIME or self.lime_explainer is None:
            logger.warning("LIME not available")
            return

        figures_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        for idx in instance_indices:
            try:
                instance = self.X_test.iloc[idx].values
                lime_exp = self.lime_explainer.explain_instance(
                    instance,
                    self.model.predict_proba,
                    num_features=10
                )

                # Save HTML report (for dashboard display)
                lime_exp.save_to_file(reports_dir / f'lime_{idx}.html')

                # Save PNG figure for quick preview if desired
                fig = lime_exp.as_pyplot_figure()
                fig.savefig(figures_dir / f'lime_{idx}.png',
                            dpi=300, bbox_inches='tight')
                plt.close(fig)

            except Exception as e:
                logger.warning(f"Failed to save LIME explanation for instance {idx}: {e}")

        logger.info(
            f"LIME explanations saved to {figures_dir} (figures) and {reports_dir} (reports)")

    def generate_report(self, report_dir: Path, usage: str) -> None:
        """Generate a markdown report summarizing outputs with references.

        Args:
            report_dir: Directory to save the report
            usage: Example CLI usage string to include
        """
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / 'explainability_report.md'
        with open(report_path, 'w') as f:
            f.write('# Explainability Report\n\n')
            f.write('Global feature importance: `reports/global_importance.csv`.\n\n')
            f.write('SHAP summary, PDP/ICE plots, and LIME figures are stored in `figures/`.\n')
            f.write('LIME HTML reports are stored in `reports/`.\n\n')
            f.write('## Usage\n')
            f.write('```bash\n')
            f.write(f'{usage}\n')
            f.write('```\n\n')
            f.write('## References\n')
            f.write('- SHAP documentation: https://shap.readthedocs.io/\n')
            f.write('- LIME documentation: https://lime.readthedocs.io/en/latest/\n')
        logger.info(f"Explainability report written to {report_path}")


def main():
    """CLI interface for explainability analysis."""
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Model explainability analysis')
    parser.add_argument('--model-path', type=Path, required=True, help='Path to trained model')
    parser.add_argument('--data-path', type=Path, required=True, help='Path to dataset')
    parser.add_argument('--figures-dir', type=Path, default=Path('figures'), help='Directory for figure outputs')
    parser.add_argument('--reports-dir', type=Path, default=Path('reports'), help='Directory for reports')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples for explanation')
    parser.add_argument('--sensitive-attr', default='sex', help='Sensitive attribute for stability')
    
    args = parser.parse_args()
    
    # Load model and data
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    
    df = pd.read_parquet(args.data_path)
    
    # Split data (using simple split for demo)
    X = df.drop(columns=['id_student', 'label_pass', 'label_fail_or_withdraw', args.sensitive_attr])
    y = df['label_pass']
    sensitive = df[args.sensitive_attr]
    
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )
    
    # Initialize analyzer
    analyzer = ExplainabilityAnalyzer(
        model, X_train, X_test, y_train, y_test, 
        sensitive_features=sens_test
    )
    
    # Setup explainers
    analyzer.setup_shap_explainer(max_samples=args.max_samples)
    analyzer.setup_lime_explainer()
    
    # Compute explanations
    analyzer.compute_shap_values(max_samples=args.max_samples)
    
    # Global importance
    importance_df = analyzer.global_feature_importance()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(args.reports_dir / 'global_importance.csv', index=False)
    
    # Local explanations for a few instances
    local_explanations = analyzer.local_explanations([0, 1, 2])
    
    # Faithfulness check
    faithfulness = analyzer.faithfulness_check()
    
    # Stability analysis
    stability = analyzer.stability_by_group(args.sensitive_attr)
    
    # Save plots
    analyzer.save_shap_plots(args.figures_dir)
    analyzer.save_pdp_ice_plots(args.figures_dir)
    analyzer.save_lime_explanations(args.figures_dir, args.reports_dir, [0, 1, 2])

    # Generate summary report
    usage = f"python src/explain/importance.py --model-path {args.model_path} --data-path {args.data_path}"
    analyzer.generate_report(args.reports_dir, usage)

    logger.info("Explainability analysis completed!")


if __name__ == '__main__':
    setup_logging()
    main()
