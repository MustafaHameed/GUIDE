"""
Training and Fairness Evaluation Pipeline

Implements training pipeline with multiple models, calibration, and fairness
mitigation techniques including preprocessing reweighing and postprocessing
threshold optimization.

References:
- Fairlearn postprocessing: https://fairlearn.org/v0.10/user_guide/mitigation/postprocessing.html
- Fairlearn equalized odds: https://fairlearn.org/v0.10/user_guide/fairness_in_machine_learning.html
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from logging_config import setup_logging

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, brier_score_loss,
    confusion_matrix, classification_report, RocCurveDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# Explainability
from explain.importance import ExplainabilityAnalyzer

# Fairness imports
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
    selection_rate
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.preprocessing import CorrelationRemover
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

# Deep learning imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Optional imports for additional models
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Configure logging
logger = logging.getLogger(__name__)


def load_oulad_data(dataset_path: Path, split_path: Path) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Load OULAD dataset and split definitions.
    
    Args:
        dataset_path: Path to OULAD parquet file
        split_path: Path to split JSON file
        
    Returns:
        Tuple of (dataset, split_dict)
    """
    logger.info(f"Loading dataset from {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    logger.info(f"Loading splits from {split_path}")
    with open(split_path, 'r') as f:
        splits = json.load(f)

    return df, splits


class DnnClassifier:
    """Simple feed-forward neural network for tabular data.

    Handles categorical features via embeddings and normalizes numeric
    features. Implements a scikit-learn like interface with `fit`,
    `predict`, and `predict_proba` methods.
    """

    def __init__(
        self,
        num_layers: int = 2,
        hidden_units: int = 64,
        dropout: float = 0.0,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.cat_mapping: Dict[str, Dict[Any, int]] = {}
        self.num_means: pd.Series = pd.Series(dtype=float)
        self.num_stds: pd.Series = pd.Series(dtype=float)
        self.model: Optional[nn.Module] = None

    def _prepare_tensors(self, X: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform DataFrame into tensors using fitted statistics."""
        if self.cat_cols:
            cat_arrays = []
            for col in self.cat_cols:
                mapping = self.cat_mapping[col]
                cat_arrays.append(
                    X[col].map(mapping).fillna(0).astype("int64").values
                )
            x_cat = torch.tensor(np.vstack(cat_arrays).T, dtype=torch.long)
        else:
            x_cat = torch.zeros(len(X), 0, dtype=torch.long)

        if self.num_cols:
            nums = (X[self.num_cols] - self.num_means) / self.num_stds
            x_num = torch.tensor(nums.values, dtype=torch.float32)
        else:
            x_num = torch.zeros(len(X), 0, dtype=torch.float32)

        return x_cat, x_num

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "DnnClassifier":
        self.cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        emb_dims = []
        cat_data = []
        for col in self.cat_cols:
            codes, uniques = pd.factorize(X[col])
            mapping = {cat: i + 1 for i, cat in enumerate(uniques)}
            self.cat_mapping[col] = mapping
            cat_data.append(codes + 1)
            n_categories = len(uniques) + 1
            emb_dim = min(50, (n_categories + 1) // 2)
            emb_dims.append((n_categories, emb_dim))
        if cat_data:
            x_cat = torch.tensor(np.vstack(cat_data).T, dtype=torch.long)
        else:
            x_cat = torch.zeros(len(X), 0, dtype=torch.long)

        if self.num_cols:
            self.num_means = X[self.num_cols].mean()
            self.num_stds = X[self.num_cols].std().replace(0, 1)
            nums = (X[self.num_cols] - self.num_means) / self.num_stds
            x_num = torch.tensor(nums.values, dtype=torch.float32)
        else:
            x_num = torch.zeros(len(X), 0, dtype=torch.float32)

        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        if sample_weight is not None:
            w_tensor = torch.tensor(sample_weight, dtype=torch.float32)
            dataset = TensorDataset(x_cat, x_num, y_tensor, w_tensor)
        else:
            dataset = TensorDataset(x_cat, x_num, y_tensor)

        class TabularNN(nn.Module):
            def __init__(self, emb_dims, n_numeric, hidden_units, dropout, num_layers):
                super().__init__()
                self.embeddings = nn.ModuleList(
                    [nn.Embedding(n, d) for n, d in emb_dims]
                )
                input_dim = n_numeric + sum(d for _, d in emb_dims)
                layers = []
                for _ in range(num_layers):
                    layers.append(nn.Linear(input_dim, hidden_units))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    input_dim = hidden_units
                self.layers = nn.Sequential(*layers)
                self.output = nn.Linear(input_dim, 1)

            def forward(self, x_cat, x_num):
                if self.embeddings:
                    embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
                    x = torch.cat(embeds, dim=1)
                    if x_num.shape[1] > 0:
                        x = torch.cat([x, x_num], dim=1)
                else:
                    x = x_num
                x = self.layers(x)
                return self.output(x).squeeze(-1)

        self.model = TabularNN(emb_dims, x_num.shape[1], self.hidden_units, self.dropout, self.num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                optimizer.zero_grad()
                if sample_weight is not None:
                    xb_cat, xb_num, yb, wb = batch
                    logits = self.model(xb_cat, xb_num)
                    loss = (criterion(logits, yb) * wb).mean()
                else:
                    xb_cat, xb_num, yb = batch
                    logits = self.model(xb_cat, xb_num)
                    loss = criterion(logits, yb).mean()
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_cat, x_num = self._prepare_tensors(X)
            logits = self.model(x_cat, x_num)
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1 - probs, probs]).T

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def create_model(model_type: str, **kwargs) -> Any:
    """Create model instance.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
    """
    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000, **kwargs),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100, **kwargs),
        'dnn': DnnClassifier(**kwargs),
    }
    
    if HAS_XGB:
        models['xgboost'] = xgb.XGBClassifier(random_state=42, **kwargs)
    
    if HAS_LGB:
        models['lightgbm'] = lgb.LGBMClassifier(random_state=42, **kwargs)
    
    if model_type not in models:
        available = list(models.keys())
        raise ValueError(f"Model {model_type} not available. Choose from: {available}")
    
    return models[model_type]


def preprocess_reweighing(X_train: pd.DataFrame, y_train: pd.Series, 
                         sensitive_features: pd.Series) -> np.ndarray:
    """Apply AIF360 reweighing preprocessing.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        sensitive_features: Sensitive attribute values
        
    Returns:
        Sample weights for training
    """
    logger.info("Applying preprocessing reweighing...")
    
    # Prepare data for AIF360
    df_train = X_train.copy()
    df_train['y'] = y_train
    df_train['sensitive'] = sensitive_features
    
    # Create AIF360 dataset
    dataset = BinaryLabelDataset(
        df=df_train,
        label_names=['y'],
        protected_attribute_names=['sensitive']
    )
    
    # Apply reweighing
    reweighing = Reweighing(
        unprivileged_groups=[{'sensitive': 0}],
        privileged_groups=[{'sensitive': 1}]
    )
    
    reweighed_dataset = reweighing.fit_transform(dataset)
    
    return reweighed_dataset.instance_weights


def apply_postprocessing(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, sensitive_features_train: pd.Series,
                        sensitive_features_test: pd.Series, constraint: str = 'equalized_odds') -> Tuple[np.ndarray, np.ndarray]:
    """Apply Fairlearn postprocessing threshold optimization.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features  
        sensitive_features_train: Training sensitive features
        sensitive_features_test: Test sensitive features
        constraint: Fairness constraint ('equalized_odds' or 'demographic_parity')
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    logger.info(f"Applying postprocessing with {constraint} constraint...")
    
    # Create threshold optimizer
    postprocessor = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        prefit=True
    )
    
    # Fit postprocessor
    postprocessor.fit(X_train, y_train, sensitive_features=sensitive_features_train)
    
    # Make fair predictions
    y_pred = postprocessor.predict(X_test, sensitive_features=sensitive_features_test)
    
    # Get probabilities if available
    try:
        y_prob = postprocessor.predict_proba(X_test, sensitive_features=sensitive_features_test)[:, 1]
    except:
        y_prob = None
    
    return y_pred, y_prob


def calculate_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Calculate comprehensive fairness metrics and group report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Sensitive attribute values

    Returns:
        Tuple of (metrics dictionary, per-group report DataFrame)
    """
    metrics = {}

    # Create MetricFrame for group-wise metrics
    mf = MetricFrame(
        metrics={
            'accuracy': accuracy_score,
            'tpr': true_positive_rate,
            'fpr': false_positive_rate,
            'selection_rate': selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    # Demographic parity difference
    metrics['demographic_parity_diff'] = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    # Equalized odds difference
    metrics['equalized_odds_diff'] = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_features
    )

    # TPR and FPR gaps
    tpr_by_group = mf.by_group['tpr']
    fpr_by_group = mf.by_group['fpr']

    metrics['tpr_gap'] = tpr_by_group.max() - tpr_by_group.min()
    metrics['fpr_gap'] = fpr_by_group.max() - fpr_by_group.min()

    # Worst-group error (1 - accuracy)
    accuracy_by_group = mf.by_group['accuracy']
    metrics['worst_group_error'] = 1 - accuracy_by_group.min()

    return metrics, mf.by_group


def compute_fairness_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    attr_name: str,
) -> pd.DataFrame:
    """Compute per-group fairness metrics expected by the dashboard."""
    overall_tpr = true_positive_rate(y_true, y_pred)
    overall_fpr = false_positive_rate(y_true, y_pred)

    mf = MetricFrame(
        metrics={
            "demographic_parity": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    records: List[Dict[str, Any]] = []
    for group_val, row in mf.by_group.iterrows():
        eo = max(
            abs(row["true_positive_rate"] - overall_tpr),
            abs(row["false_positive_rate"] - overall_fpr),
        )
        records.append(
            {
                attr_name: group_val,
                "demographic_parity": row["demographic_parity"],
                "equalized_odds": eo,
            }
        )

    return pd.DataFrame(records)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray],
                     sensitive_features: pd.Series, student_ids: pd.Series, 
                     n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
    """Calculate bootstrap confidence intervals for metrics.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        y_prob: Prediction probabilities  
        sensitive_features: Sensitive attributes
        student_ids: Student identifiers for proper resampling
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with mean and CI for each metric
    """
    logger.info(f"Computing bootstrap CIs with {n_bootstrap} resamples...")
    
    metrics_list = []
    unique_students = student_ids.unique()
    
    for i in range(n_bootstrap):
        # Resample at student level
        bootstrap_students = np.random.choice(unique_students, size=len(unique_students), replace=True)
        bootstrap_mask = student_ids.isin(bootstrap_students)
        
        # Calculate metrics for this bootstrap sample
        sample_metrics = {
            'accuracy': accuracy_score(y_true[bootstrap_mask], y_pred[bootstrap_mask]),
            'auc': roc_auc_score(y_true[bootstrap_mask], y_prob[bootstrap_mask]) if y_prob is not None else np.nan,
            'f1': f1_score(y_true[bootstrap_mask], y_pred[bootstrap_mask]),
        }
        
        if y_prob is not None:
            sample_metrics['brier'] = brier_score_loss(y_true[bootstrap_mask], y_prob[bootstrap_mask])
            sample_metrics['ece'] = expected_calibration_error(y_true[bootstrap_mask], y_prob[bootstrap_mask])
        
        # Add fairness metrics
        fairness_metrics, _ = calculate_fairness_metrics(
            y_true[bootstrap_mask],
            y_pred[bootstrap_mask],
            sensitive_features[bootstrap_mask],
        )
        sample_metrics.update(fairness_metrics)
        
        metrics_list.append(sample_metrics)
    
    # Calculate confidence intervals
    results = {}
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list if not np.isnan(m[metric])]
        if values:
            results[metric] = {
                'mean': np.mean(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'std': np.std(values)
            }
    
    return results


def train_and_evaluate_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_features_train: pd.Series,
    sensitive_features_test: pd.Series,
    student_ids_test: pd.Series,
    use_reweighing: bool = False,
    postprocess: Optional[str] = None,
    figures_dir: Path = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train and evaluate a model with optional fairness mitigation.

    Args:
        model_type: Type of model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        sensitive_features_train/test: Sensitive attributes
        student_ids_test: Student IDs for bootstrap resampling
        use_reweighing: Whether to apply preprocessing reweighing
        postprocess: Fairness constraint for postprocessing ('equalized_odds',
            'demographic_parity', or None)
        figures_dir: Directory to save figures
        model_params: Additional parameters for model creation

    Returns:
        Dictionary with evaluation results and fairness report
    """
    logger.info(
        f"Training {model_type} with reweighing={use_reweighing} and postprocess={postprocess}"
    )

    results: Dict[str, Any] = {'model_type': model_type}

    # Create and train model
    model = create_model(model_type, **(model_params or {}))

    # Apply preprocessing reweighing if requested
    sample_weight = None
    if use_reweighing:
        # Convert sensitive features to binary for AIF360
        sensitive_binary = (
            sensitive_features_train == sensitive_features_train.mode()[0]
        ).astype(int)
        sample_weight = preprocess_reweighing(X_train, y_train, sensitive_binary)

    # Train model
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    # Make initial predictions (pre-mitigation)
    y_pred_pre = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, 'predict_proba')
        else None
    )

    attr_name = sensitive_features_test.name
    fairness_pre_table = compute_fairness_table(
        y_test, y_pred_pre, sensitive_features_test, attr_name
    )

    # Start with pre-mitigation predictions
    y_pred = y_pred_pre

    # Apply postprocessing if requested
    if postprocess and postprocess != 'none':
        y_pred_post, y_prob_post = apply_postprocessing(
            model,
            X_train,
            y_train,
            X_test,
            sensitive_features_train,
            sensitive_features_test,
            constraint=postprocess,
        )
        y_pred = y_pred_post
        if y_prob_post is not None:
            y_prob = y_prob_post

    fairness_post_table = compute_fairness_table(
        y_test, y_pred, sensitive_features_test, attr_name
    )

    # Calculate basic metrics
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['f1'] = f1_score(y_test, y_pred)

    if y_prob is not None:
        results['auc'] = roc_auc_score(y_test, y_prob)
        results['brier'] = brier_score_loss(y_test, y_prob)
        results['ece'] = expected_calibration_error(y_test, y_prob)

    # Calculate fairness metrics and report for post-mitigation predictions
    fairness_metrics, fairness_by_group = calculate_fairness_metrics(
        y_test, y_pred, sensitive_features_test
    )
    results.update(fairness_metrics)
    results['fairness_by_group'] = fairness_by_group
    results['fairness_pre_table'] = fairness_pre_table
    results['fairness_post_table'] = fairness_post_table

    # Bootstrap confidence intervals
    bootstrap_results = bootstrap_metrics(
        y_test, y_pred, y_prob, sensitive_features_test, student_ids_test
    )
    results['bootstrap'] = bootstrap_results

    # Save performance figures if requested
    if figures_dir is not None:
        prefix = f"oulad_{model_type}"
        if y_prob is not None:
            save_reliability_plot(y_test, y_prob, figures_dir, prefix)
            save_roc_curve(y_test, y_prob, figures_dir, prefix)
        save_confusion_matrix(y_test, y_pred, figures_dir, prefix)

    # Include trained model for downstream analysis
    results['model'] = model

    return results


def save_reliability_plot(y_true: np.ndarray, y_prob: np.ndarray,
                         output_dir: Path, filename_prefix: str) -> None:
    """Save reliability diagram for probability calibration.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_dir: Output directory
        filename_prefix: Prefix for filename
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate bin statistics
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            confidence_in_bin = y_prob[in_bin].mean()
            
            bin_centers.append(confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(prop_in_bin)
    
    # Plot reliability diagram
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.scatter(bin_centers, bin_accuracies, s=bin_counts, alpha=0.7, label='Observed')
    
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Reliability Diagram - {filename_prefix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{filename_prefix}_reliability.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   output_dir: Path, filename_prefix: str) -> None:
    """Save ROC curve plot."""
    roc_disp = RocCurveDisplay.from_predictions(y_true, y_prob)
    roc_disp.ax_.set_title(f"ROC Curve - {filename_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{filename_prefix}_roc.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          output_dir: Path, filename_prefix: str) -> None:
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {filename_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{filename_prefix}_confusion.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """CLI interface for training and fairness evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Train model with fairness evaluation")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset parquet file")
    parser.add_argument("--split", type=Path, required=True, help="Path to split JSON file")
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest", "xgboost", "lightgbm", "dnn"],
        default="logistic",
        help="Model type",
    )
    parser.add_argument(
        "--dnn-layers", type=int, default=2, help="Number of hidden layers for DNN"
    )
    parser.add_argument(
        "--dnn-hidden-units", type=int, default=64, help="Units per hidden layer"
    )
    parser.add_argument(
        "--dnn-dropout", type=float, default=0.0, help="Dropout rate for DNN"
    )
    parser.add_argument(
        "--sensitive-attr", default="sex", help="Sensitive attribute column"
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("reports"), help="Directory for reports"
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("figures"), help="Directory for figures"
    )
    parser.add_argument(
        "--reweighing",
        action="store_true",
        help="Apply preprocessing reweighing",
    )
    parser.add_argument(
        "--postprocess",
        choices=["none", "equalized_odds", "demographic_parity"],
        default="none",
        help="Postprocessing threshold optimization",
    )

    args = parser.parse_args()

    # Load data
    df, splits = load_oulad_data(args.dataset, args.split)

    # Prepare train/test splits
    train_mask = df['id_student'].isin(splits['train'])
    test_mask = df['id_student'].isin(splits['test'])

    X_train = df[train_mask].drop(
        columns=['id_student', 'label_pass', 'label_fail_or_withdraw', args.sensitive_attr]
    )
    y_train = df[train_mask]['label_pass']
    sensitive_train = df[train_mask][args.sensitive_attr]

    X_test = df[test_mask].drop(
        columns=['id_student', 'label_pass', 'label_fail_or_withdraw', args.sensitive_attr]
    )
    y_test = df[test_mask]['label_pass']
    sensitive_test = df[test_mask][args.sensitive_attr]
    student_ids_test = df[test_mask]['id_student']

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train model
    model_params = None
    if args.model == "dnn":
        model_params = {
            "num_layers": args.dnn_layers,
            "hidden_units": args.dnn_hidden_units,
            "dropout": args.dnn_dropout,
        }

    results = train_and_evaluate_model(
        args.model,
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_train,
        sensitive_test,
        student_ids_test,
        use_reweighing=args.reweighing,
        postprocess=args.postprocess,
        figures_dir=args.figures_dir,
        model_params=model_params,
    )

    # After training, create explainability artifacts
    try:
        analyzer = ExplainabilityAnalyzer(
            results['model'],
            X_train,
            X_test,
            y_train,
            y_test,
            sensitive_features=sensitive_test,
        )
        analyzer.setup_shap_explainer()
        analyzer.setup_lime_explainer()
        analyzer.compute_shap_values()
        analyzer.save_shap_plots(args.figures_dir)
        analyzer.save_lime_explanations(args.figures_dir, args.reports_dir, [0])
    except Exception as e:
        logger.warning(f"Explainability analysis failed: {e}")

    # Save reports
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_save = {
        k: (float(v) if isinstance(v, (np.floating, np.float64, np.float32)) else v)
        for k, v in results.items()
        if k not in ['bootstrap', 'fairness_by_group', 'model']
    }
    with open(args.reports_dir / f"{args.model}_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    # Export metrics to tables for dashboard
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([metrics_to_save])
    metrics_df.to_csv(tables_dir / f"oulad_{args.model}_metrics.csv", index=False)

    # Fairness reports
    fairness_df: pd.DataFrame = results['fairness_by_group']
    fairness_df.to_csv(args.reports_dir / f"{args.model}_fairness.csv")

    fairness_pre_df: pd.DataFrame = results['fairness_pre_table']
    fairness_post_df: pd.DataFrame = results['fairness_post_table']
    fairness_pre_df.to_csv(
        args.reports_dir / f"fairness_{args.sensitive_attr}_pre.csv", index=False
    )
    fairness_post_df.to_csv(
        args.reports_dir / f"fairness_{args.sensitive_attr}_post.csv", index=False
    )

    # Bootstrap metrics
    with open(args.reports_dir / f"{args.model}_bootstrap.json", "w") as f:
        json.dump(results['bootstrap'], f, indent=2)

    logger.info(f"Reports saved to {args.reports_dir}")
    logger.info("Training and evaluation completed!")


if __name__ == '__main__':
    setup_logging()
    main()
