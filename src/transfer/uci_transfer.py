"""
Cross-Dataset Transfer Learning: OULAD to UCI

Implements transfer learning between OULAD and UCI student performance datasets
with minimal shared feature mapping for external validity assessment.

Uses existing UCI loader and creates bidirectional transfer experiments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from logging_config import setup_logging

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

# Import existing UCI data loader
from data import load_data

# Configure logging
logger = logging.getLogger(__name__)

# Epoch defaults for the optional MLP model. Tests may monkeypatch these to
# reduce training time.
MLP_PRETRAIN_EPOCHS = 20
MLP_FINETUNE_EPOCHS = 10


def create_shared_feature_mapping() -> Dict[str, Dict]:
    """Define mapping between OULAD and UCI features for transfer learning.

    Returns:
        Dictionary with feature mappings and transformations
    """
    mapping = {
        "shared_features": {
            # Demographics
            "sex": {
                "oulad_col": "sex",
                "uci_col": "sex",
                "mapping": {"Female": "F", "Male": "M"},
            },
            # Age (convert to bands for compatibility)
            "age_band": {
                "oulad_col": "age_band",
                "uci_col": "age",
                "transform": "age_to_band",
            },
            # Socioeconomic proxy
            "ses_proxy": {
                "oulad_col": "imd_band",  # Index of Multiple Deprivation
                "uci_col": "Medu",  # Mother's education as SES proxy
                "transform": "ses_standardize",
            },
            # Attendance/engagement proxy
            "attendance_proxy": {
                "oulad_col": "vle_total_clicks",  # VLE engagement
                "uci_col": "absences",  # School absences
                "transform": "attendance_normalize",
            },
            # Study time - directly shared numeric category
            "studytime": {"oulad_col": "studytime", "uci_col": "studytime"},
            # Higher education intention
            "higher": {
                "oulad_col": "higher",
                "uci_col": "higher",
                "mapping": {"Yes": "yes", "No": "no", "Y": "yes", "N": "no"},
            },
            # Family support
            "famsup": {
                "oulad_col": "famsup",
                "uci_col": "famsup",
                "mapping": {"Yes": "yes", "No": "no", "Y": "yes", "N": "no"},
            },
            # Social outings frequency
            "goout": {"oulad_col": "goout", "uci_col": "goout"},
            # Internet access
            "internet": {
                "oulad_col": "vle_total_clicks",
                "uci_col": "internet",
                "transform": "clicks_to_binary",
                "mapping": {"yes": "yes", "no": "no"},
            },
            # Participation in activities
            "activities": {
                "oulad_col": "goout",
                "uci_col": "activities",
                "transform": "goout_to_binary",
                "mapping": {"yes": "yes", "no": "no"},
            },
            # Free time (inverse of study time)
            "freetime": {
                "oulad_col": "studytime",
                "uci_col": "freetime",
                "transform": "studytime_to_freetime",
            },
            # Family relationship quality
            "famrel": {
                "oulad_col": "famsup",
                "uci_col": "famrel",
                "transform": "famsup_to_famrel",
            },
        },
        "label_mapping": {
            "oulad_label": "label_pass",
            "uci_label": "pass",  # From UCI data loader
        },
    }

    return mapping


def age_to_band_transform(age_values: pd.Series) -> pd.Series:
    """Transform UCI age values to OULAD-style age bands.

    Args:
        age_values: Series with numeric ages

    Returns:
        Series with age band categories
    """

    def age_to_band(age):
        if age <= 18:
            return "0-35"  # Young students
        elif age <= 22:
            return "0-35"  # Still young adult
        else:
            return "35-55"  # Older students

    return age_values.apply(age_to_band)


def ses_standardize_transform(
    oulad_imd: Optional[pd.Series], uci_medu: Optional[pd.Series]
) -> Tuple[pd.Series, pd.Series]:
    """Standardize socioeconomic indicators between datasets.

    Args:
        oulad_imd: OULAD IMD band values
        uci_medu: UCI mother's education values

    Returns:
        Tuple of standardized SES proxies
    """
    # Convert both to 0-1 scale where higher = higher SES
    oulad_ses = None
    uci_ses = None

    if oulad_imd is not None:
        # IMD bands: higher band = more deprived, so invert
        imd_map = {
            "0-10%": 0.9,
            "10-20%": 0.7,
            "20-30%": 0.5,
            "30-40%": 0.3,
            "40-50%": 0.1,
        }
        oulad_ses = oulad_imd.map(imd_map).fillna(0.5)

    if uci_medu is not None:
        # Education: 0-4 scale, higher = better education
        uci_ses = uci_medu / 4.0

    return oulad_ses, uci_ses


def attendance_normalize_transform(
    oulad_vle: Optional[pd.Series], uci_absences: Optional[pd.Series]
) -> Tuple[pd.Series, pd.Series]:
    """Normalize attendance/engagement proxies between datasets.

    Args:
        oulad_vle: OULAD VLE total clicks
        uci_absences: UCI absences count

    Returns:
        Tuple of normalized attendance proxies (higher = better attendance)
    """
    oulad_attend = None
    uci_attend = None

    if oulad_vle is not None:
        # Normalize VLE clicks to 0-1 scale
        oulad_attend = (oulad_vle - oulad_vle.min()) / (
            oulad_vle.max() - oulad_vle.min() + 1e-8
        )

    if uci_absences is not None:
        # Convert absences to attendance (invert and normalize)
        max_absences = uci_absences.max()
        uci_attend = 1 - (uci_absences / (max_absences + 1))

    return oulad_attend, uci_attend


def prepare_oulad_features(
    oulad_df: pd.DataFrame, feature_mapping: Dict
) -> pd.DataFrame:
    """Extract and transform OULAD features for transfer learning.

    Args:
        oulad_df: OULAD dataset
        feature_mapping: Feature mapping configuration

    Returns:
        DataFrame with shared features
    """
    logger.info("Preparing OULAD features for transfer...")

    shared_data = pd.DataFrame()

    # Extract shared features
    for feature_name, config in feature_mapping["shared_features"].items():
        oulad_col = config["oulad_col"]

        if oulad_col in oulad_df.columns:
            if feature_name == "sex":
                # Direct mapping for sex
                shared_data[feature_name] = oulad_df[oulad_col]

            elif feature_name == "age_band":
                # Use existing age_band
                shared_data[feature_name] = oulad_df[oulad_col]

            elif feature_name == "ses_proxy":
                # Transform IMD to SES proxy
                ses_values, _ = ses_standardize_transform(oulad_df[oulad_col], None)
                shared_data[feature_name] = ses_values

            elif feature_name == "attendance_proxy":
                # Transform VLE clicks to attendance proxy
                attend_values, _ = attendance_normalize_transform(
                    oulad_df[oulad_col], None
                )
                shared_data[feature_name] = attend_values

            elif feature_name == "internet":
                # Derive internet access from VLE activity
                shared_data[feature_name] = np.where(
                    oulad_df[oulad_col] > 0, "yes", "no"
                )

            elif feature_name == "activities":
                # Use social outings as proxy for extracurricular activities
                shared_data[feature_name] = np.where(
                    oulad_df[oulad_col] > 3, "yes", "no"
                )

            elif feature_name == "freetime":
                # Approximate free time as inverse of study time
                shared_data[feature_name] = 5 - oulad_df[oulad_col].astype(float)

            elif feature_name == "famrel":
                # Map family support to a coarse relationship quality score
                shared_data[feature_name] = (
                    oulad_df[oulad_col]
                    .map({"Yes": 4, "No": 2, "Y": 4, "N": 2})
                    .fillna(3)
                )

            elif "mapping" in config:
                # Apply simple value mapping if provided
                shared_data[feature_name] = (
                    oulad_df[oulad_col]
                    .map(config["mapping"])
                    .fillna(oulad_df[oulad_col])
                )

            else:
                # Direct copy for shared numeric/categorical features
                shared_data[feature_name] = oulad_df[oulad_col]

    # Add label
    if feature_mapping["label_mapping"]["oulad_label"] in oulad_df.columns:
        shared_data["label"] = oulad_df[feature_mapping["label_mapping"]["oulad_label"]]

    logger.info(f"OULAD shared features shape: {shared_data.shape}")
    return shared_data


def prepare_uci_features(uci_csv_path: str, feature_mapping: Dict) -> pd.DataFrame:
    """Extract and transform UCI features for transfer learning.

    Args:
        uci_csv_path: Path to UCI dataset
        feature_mapping: Feature mapping configuration

    Returns:
        DataFrame with shared features
    """
    logger.info("Preparing UCI features for transfer...")

    # Load UCI data using existing loader
    X_uci, y_uci = load_data(uci_csv_path, task="classification")

    # Combine features and labels
    uci_df = X_uci.copy()
    uci_df["pass"] = y_uci

    shared_data = pd.DataFrame()

    # Extract shared features
    for feature_name, config in feature_mapping["shared_features"].items():
        uci_col = config["uci_col"]

        if uci_col in uci_df.columns:
            if feature_name == "sex":
                # Direct mapping for sex (already F/M in UCI)
                shared_data[feature_name] = uci_df[uci_col]

            elif feature_name == "age_band":
                # Transform age to bands
                shared_data[feature_name] = age_to_band_transform(uci_df[uci_col])

            elif feature_name == "ses_proxy":
                # Transform mother's education to SES proxy
                _, ses_values = ses_standardize_transform(None, uci_df[uci_col])
                shared_data[feature_name] = ses_values

            elif feature_name == "attendance_proxy":
                # Transform absences to attendance proxy
                _, attend_values = attendance_normalize_transform(None, uci_df[uci_col])
                shared_data[feature_name] = attend_values

            elif "mapping" in config:
                # Apply value mapping if provided
                shared_data[feature_name] = (
                    uci_df[uci_col].map(config["mapping"]).fillna(uci_df[uci_col])
                )

            else:
                # Direct copy for shared numeric/categorical features
                shared_data[feature_name] = uci_df[uci_col]

    # Add label
    shared_data["label"] = uci_df["pass"]

    logger.info(f"UCI shared features shape: {shared_data.shape}")
    return shared_data


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Encode categorical features consistently across datasets.

    Args:
        df: DataFrame with categorical features

    Returns:
        Tuple of (encoded_df, encoders_dict)
    """
    encoders = {}
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if col != "label" and df_encoded[col].dtype == "object":
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].fillna("unknown"))
            encoders[col] = encoder

    return df_encoded, encoders


def transfer_experiment(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    model_type: str = "logistic",
    use_cv: bool = False,
    mlp_pretrain_epochs: Optional[int] = None,
    mlp_finetune_epochs: Optional[int] = None,
    mlp_lr: float = 0.01,
    hidden_sizes: Optional[List[int]] = None,
    dropout: float = 0.0,
    reweight_sex: bool = False,
) -> Dict[str, float]:
    """Run transfer learning experiment from source to target dataset.

    Args:
        source_data: Source dataset with shared features and labels
        target_data: Target dataset with shared features and labels
        model_type: Type of model to use
        use_cv: Whether to perform hyperparameter search with cross-validation
        mlp_pretrain_epochs: Epochs for pretraining when using MLP
        mlp_finetune_epochs: Epochs for fine-tuning when using MLP
        mlp_lr: Learning rate for the MLP optimizer
        hidden_sizes: Hidden layer sizes for the MLP
        dropout: Dropout rate applied after activations in the MLP
        reweight_sex: If True, weight training samples inversely
            proportional to class frequency within each sex group

    Returns:
        Dictionary with performance metrics including accuracy,
        balanced accuracy, F1 score, and optionally AUC and group
        accuracies
    """
    logger.info(f"Running transfer experiment: {model_type}")

    if mlp_pretrain_epochs is None:
        mlp_pretrain_epochs = MLP_PRETRAIN_EPOCHS
    if mlp_finetune_epochs is None:
        mlp_finetune_epochs = MLP_FINETUNE_EPOCHS

    # Prepare features and labels
    feature_cols = [col for col in source_data.columns if col != "label"]

    X_source = source_data[feature_cols]
    y_source = source_data["label"]
    X_target = target_data[feature_cols]
    y_target = target_data["label"]

    # Handle missing values
    X_source = X_source.fillna(X_source.mean())
    X_target = X_target.fillna(X_target.mean())

    # Standardize features
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)

    if model_type in {"logistic", "random_forest"}:
        # Classical sklearn models with optional hyperparameter search

        sample_weight = None
        if reweight_sex and "sex" in source_data.columns:
            group_counts = source_data.groupby(["sex", "label"]).size()
            weights = source_data[["sex", "label"]].apply(
                lambda r: 1.0 / group_counts.loc[r["sex"], r["label"]], axis=1
            )
            # Normalize weights to maintain comparable scale
            weights = weights * len(weights) / weights.sum()
            sample_weight = weights.values

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if model_type == "logistic":
            base_model = LogisticRegression(random_state=42, max_iter=1000)
            if use_cv:
                param_grid = {"C": [0.01, 0.1, 1, 10]}
                search = GridSearchCV(base_model, param_grid, cv=3)
                search.fit(
                    X_source_scaled,
                    y_source,
                    **fit_kwargs,
                )
                model = search.best_estimator_
            else:
                base_model.fit(X_source_scaled, y_source, **fit_kwargs)
                model = base_model
        else:
            base_model = RandomForestClassifier(random_state=42)
            if use_cv:
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10],
                }
                search = GridSearchCV(base_model, param_grid, cv=3)
                search.fit(
                    X_source_scaled,
                    y_source,
                    **fit_kwargs,
                )
                model = search.best_estimator_
            else:
                base_model.set_params(n_estimators=100)
                base_model.fit(X_source_scaled, y_source, **fit_kwargs)
                model = base_model

        # Evaluate on target dataset
        y_pred = model.predict(X_target_scaled)
        y_prob = (
            model.predict_proba(X_target_scaled)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
    elif model_type == "mlp":
        # Delayed import to keep torch optional
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset, random_split

        input_dim = X_source_scaled.shape[1]

        class SimpleMLP(nn.Module):
            def __init__(
                self,
                in_dim: int,
                hidden_sizes: Optional[List[int]] = None,
                dropout: float = 0.0,
            ):
                super().__init__()
                if hidden_sizes is None:
                    hidden_sizes = [32, 16]
                layers = []
                last = in_dim
                for h in hidden_sizes:
                    layers.append(nn.Linear(last, h))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                    last = h
                self.feature = nn.Sequential(*layers)
                self.classifier = nn.Linear(last, 1)

            def forward(self, x):
                x = self.feature(x)
                return self.classifier(x)

        def _train(model, dataset, epochs, lr=0.01, patience=5):
            crit = nn.BCEWithLogitsLoss()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            val_size = max(1, int(0.1 * len(dataset)))
            train_size = len(dataset) - val_size
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
            train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=32)
            best_val = float("inf")
            epochs_no_improve = 0
            for _ in range(epochs):
                model.train()
                for xb, yb in train_dl:
                    opt.zero_grad()
                    out = model(xb).squeeze(-1)
                    loss = crit(out, yb.float())
                    loss.backward()
                    opt.step()
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_dl:
                        out = model(xb).squeeze(-1)
                        val_loss += crit(out, yb.float()).item() * len(xb)
                val_loss /= len(val_dl.dataset)
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

        # Pretrain on source
        model = SimpleMLP(input_dim, hidden_sizes=hidden_sizes, dropout=dropout)
        source_ds = TensorDataset(
            torch.tensor(X_source_scaled, dtype=torch.float32),
            torch.tensor(y_source.values, dtype=torch.float32),
        )
        _train(model, source_ds, mlp_pretrain_epochs, lr=mlp_lr)

        # Replace final layer and fine-tune on target data
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        target_ds = TensorDataset(
            torch.tensor(X_target_scaled, dtype=torch.float32),
            torch.tensor(y_target.values, dtype=torch.float32),
        )
        _train(model, target_ds, mlp_finetune_epochs, lr=mlp_lr)

        # Predictions on target
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_target_scaled, dtype=torch.float32)).squeeze(
                -1
            )
            y_prob = torch.sigmoid(logits).cpu().numpy()
            y_pred = (y_prob > 0.5).astype(int)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Calculate metrics
    results = {
        "accuracy": accuracy_score(y_target, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_target, y_pred),
        "f1": f1_score(y_target, y_pred),
        "source_size": len(X_source),
        "target_size": len(X_target),
        "n_features": len(feature_cols),
    }

    if y_prob is not None:
        results["auc"] = roc_auc_score(y_target, y_prob)

    # Calculate fairness metrics by sex if available
    if "sex" in target_data.columns:
        target_sex = target_data["sex"]
        for sex_value in target_sex.unique():
            if pd.notna(sex_value):
                sex_mask = target_sex == sex_value
                sex_acc = accuracy_score(y_target[sex_mask], y_pred[sex_mask])
                results[f"accuracy_{sex_value}"] = sex_acc

        # Worst-group accuracy
        sex_accuracies = [
            results[k] for k in results.keys() if k.startswith("accuracy_")
        ]
        if sex_accuracies:
            results["worst_group_accuracy"] = min(sex_accuracies)

    return results


def run_bidirectional_transfer(
    oulad_data_path: str,
    uci_data_path: str,
    output_dir: Path,
    table_path: Path = Path("tables/transfer_results.csv"),
    figure_path: Path = Path("figures/transfer_performance.png"),
    models: Optional[List[str]] = None,
    use_cv: bool = False,
    mlp_pretrain_epochs: Optional[int] = None,
    mlp_finetune_epochs: Optional[int] = None,
    mlp_lr: float = 0.01,
    hidden_sizes: Optional[List[int]] = None,
    dropout: float = 0.0,
    reweight_sex: bool = False,
    plot_metrics: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Run bidirectional transfer learning experiments.

    Args:
        oulad_data_path: Path to OULAD parquet file
        uci_data_path: Path to UCI CSV file
        output_dir: Directory to save intermediate results
        table_path: Location to save combined performance and fairness metrics
        figure_path: Location to save transfer performance visualization
        models: Which model types to evaluate. Defaults to logistic,
            random forest, and a small MLP.
        use_cv: Whether to perform hyperparameter search for classical models.
        mlp_pretrain_epochs: Epochs for MLP pretraining
        mlp_finetune_epochs: Epochs for MLP fine-tuning
        mlp_lr: Learning rate for MLP optimizer
        hidden_sizes: Hidden layer sizes for the MLP
        dropout: Dropout rate for the MLP
        reweight_sex: If True, weight training data by inverse class
            frequency within each sex during classical model training
        plot_metrics: Metrics to visualize in output figures. Defaults to
            ['accuracy'].

    Returns:
        Dictionary with transfer results
    """
    logger.info("Starting bidirectional transfer learning experiments...")

    # Create feature mapping
    feature_mapping = create_shared_feature_mapping()

    # Load and prepare datasets
    oulad_df = pd.read_parquet(oulad_data_path)
    oulad_shared = prepare_oulad_features(oulad_df, feature_mapping)

    uci_shared = prepare_uci_features(uci_data_path, feature_mapping)

    # Encode categorical features consistently
    # Find common categories across datasets
    common_features = set(oulad_shared.columns) & set(uci_shared.columns) - {"label"}

    for col in common_features:
        if oulad_shared[col].dtype == "object" or uci_shared[col].dtype == "object":
            # Get union of all categories
            all_categories = set(oulad_shared[col].dropna().unique()) | set(
                uci_shared[col].dropna().unique()
            )
            all_categories = sorted(list(all_categories))

            # Create consistent encoding
            encoder = LabelEncoder()
            encoder.fit(all_categories)

            oulad_shared[col] = encoder.transform(oulad_shared[col].fillna("unknown"))
            uci_shared[col] = encoder.transform(uci_shared[col].fillna("unknown"))

    # Filter to common features
    feature_cols = list(common_features) + ["label"]
    oulad_final = oulad_shared[feature_cols].dropna()
    uci_final = uci_shared[feature_cols].dropna()

    logger.info(f"Final OULAD shape: {oulad_final.shape}")
    logger.info(f"Final UCI shape: {uci_final.shape}")

    # Run transfer experiments
    results = {}

    if models is None:
        models = ["logistic", "random_forest", "mlp"]

    for model_type in models:
        # OULAD -> UCI transfer
        oulad_to_uci = transfer_experiment(
            oulad_final,
            uci_final,
            model_type,
            use_cv=use_cv,
            mlp_pretrain_epochs=mlp_pretrain_epochs,
            mlp_finetune_epochs=mlp_finetune_epochs,
            mlp_lr=mlp_lr,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            reweight_sex=reweight_sex,
        )
        oulad_to_uci["direction"] = "OULAD_to_UCI"
        oulad_to_uci["model"] = model_type

        # UCI -> OULAD transfer
        uci_to_oulad = transfer_experiment(
            uci_final,
            oulad_final,
            model_type,
            use_cv=use_cv,
            mlp_pretrain_epochs=mlp_pretrain_epochs,
            mlp_finetune_epochs=mlp_finetune_epochs,
            mlp_lr=mlp_lr,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            reweight_sex=reweight_sex,
        )
        uci_to_oulad["direction"] = "UCI_to_OULAD"
        uci_to_oulad["model"] = model_type

        results[f"{model_type}_oulad_to_uci"] = oulad_to_uci
        results[f"{model_type}_uci_to_oulad"] = uci_to_oulad

        logger.info(
            f"{model_type} OULAD->UCI: Accuracy = {oulad_to_uci['accuracy']:.3f}"
        )
        logger.info(
            f"{model_type} UCI->OULAD: Accuracy = {uci_to_oulad['accuracy']:.3f}"
        )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV in results directory
    results_df = pd.DataFrame([results[k] for k in results.keys()])
    results_df.to_csv(output_dir / "transfer_summary.csv", index=False)

    # Save feature mapping
    mapping_df = pd.DataFrame(
        [
            {
                "feature": fname,
                "oulad_col": config["oulad_col"],
                "uci_col": config["uci_col"],
            }
            for fname, config in feature_mapping["shared_features"].items()
        ]
    )
    mapping_df.to_csv(output_dir / "feature_mapping.csv", index=False)

    # Persist combined metrics table
    table_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_path, index=False)

    # Visualize transfer performance
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if plot_metrics is None:
            plot_metrics = ["accuracy"]
        for metric in plot_metrics:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=results_df, x="direction", y=metric, hue="model")
            plt.title(
                f"Transfer {metric.replace('_', ' ').title()} by Direction and Model"
            )
            plt.tight_layout()
            metric_path = (
                figure_path
                if len(plot_metrics) == 1
                else figure_path.with_name(
                    f"{figure_path.stem}_{metric}{figure_path.suffix}"
                )
            )
            plt.savefig(metric_path)
            plt.close()
    except Exception as exc:  # pragma: no cover - visualization is best effort
        logger.warning(f"Visualization failed: {exc}")

    logger.info(f"Transfer learning results saved to {output_dir}")

    return results


def main():
    """CLI interface for transfer learning experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-dataset transfer learning")
    parser.add_argument(
        "--oulad-data",
        type=str,
        default="data/oulad/processed/oulad_ml.parquet",
        help="Path to OULAD parquet file",
    )
    parser.add_argument(
        "--uci-data", type=str, default="student-mat.csv", help="Path to UCI CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results/transfer",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "random_forest", "mlp"],
        help="Models to evaluate: logistic, random_forest, mlp",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Enable hyperparameter search with cross-validation",
    )
    parser.add_argument(
        "--mlp-pretrain-epochs",
        type=int,
        default=MLP_PRETRAIN_EPOCHS,
        help="Epochs for MLP pretraining",
    )
    parser.add_argument(
        "--mlp-finetune-epochs",
        type=int,
        default=MLP_FINETUNE_EPOCHS,
        help="Epochs for MLP fine-tuning",
    )
    parser.add_argument(
        "--mlp-lr", type=float, default=0.01, help="Learning rate for MLP"
    )
    parser.add_argument(
        "--reweight-sex",
        action="store_true",
        help="Reweight classes inversely within each sex for classical models",
    )

    args = parser.parse_args()

    try:
        results = run_bidirectional_transfer(
            args.oulad_data,
            args.uci_data,
            args.output_dir,
            models=args.models,
            use_cv=args.cv,
            mlp_pretrain_epochs=args.mlp_pretrain_epochs,
            mlp_finetune_epochs=args.mlp_finetune_epochs,
            mlp_lr=args.mlp_lr,
            reweight_sex=args.reweight_sex,
        )

        # Log summary
        logger.info("\nTransfer Learning Results Summary:")
        logger.info("=" * 50)
        for exp_name, result in results.items():
            logger.info("%s:", exp_name)
            logger.info("  Accuracy: %.3f", result["accuracy"])
            if "auc" in result:
                logger.info("  AUC: %.3f", result["auc"])
            if "worst_group_accuracy" in result:
                logger.info(
                    "  Worst Group Accuracy: %.3f",
                    result["worst_group_accuracy"],
                )
            logger.info("")

        logger.info("Transfer learning experiments completed successfully!")

    except Exception as e:
        logger.error(f"Transfer learning failed: {e}")
        raise


if __name__ == "__main__":
    setup_logging()
    main()
