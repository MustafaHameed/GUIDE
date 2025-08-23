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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

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
        'shared_features': {
            # Demographics
            'sex': {
                'oulad_col': 'sex',
                'uci_col': 'sex', 
                'mapping': {'Female': 'F', 'Male': 'M'}
            },
            
            # Age (convert to bands for compatibility)
            'age_band': {
                'oulad_col': 'age_band',
                'uci_col': 'age',
                'transform': 'age_to_band'
            },
            
            # Socioeconomic proxy
            'ses_proxy': {
                'oulad_col': 'imd_band',  # Index of Multiple Deprivation
                'uci_col': 'Medu',        # Mother's education as SES proxy
                'transform': 'ses_standardize'
            },
            
            # Attendance/engagement proxy  
            'attendance_proxy': {
                'oulad_col': 'vle_total_clicks',  # VLE engagement
                'uci_col': 'absences',            # School absences
                'transform': 'attendance_normalize'
            }
        },
        
        'label_mapping': {
            'oulad_label': 'label_pass',
            'uci_label': 'pass'  # From UCI data loader
        }
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
            return '0-35'  # Young students
        elif age <= 22:
            return '0-35'  # Still young adult
        else:
            return '35-55'  # Older students
    
    return age_values.apply(age_to_band)


def ses_standardize_transform(oulad_imd: Optional[pd.Series], uci_medu: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
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
        imd_map = {'0-10%': 0.9, '10-20%': 0.7, '20-30%': 0.5, '30-40%': 0.3, '40-50%': 0.1}
        oulad_ses = oulad_imd.map(imd_map).fillna(0.5)
    
    if uci_medu is not None:
        # Education: 0-4 scale, higher = better education
        uci_ses = uci_medu / 4.0
    
    return oulad_ses, uci_ses


def attendance_normalize_transform(oulad_vle: Optional[pd.Series], uci_absences: Optional[pd.Series]) -> Tuple[pd.Series, pd.Series]:
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
        oulad_attend = (oulad_vle - oulad_vle.min()) / (oulad_vle.max() - oulad_vle.min() + 1e-8)
    
    if uci_absences is not None:
        # Convert absences to attendance (invert and normalize)
        max_absences = uci_absences.max()
        uci_attend = 1 - (uci_absences / (max_absences + 1))
    
    return oulad_attend, uci_attend


def prepare_oulad_features(oulad_df: pd.DataFrame, feature_mapping: Dict) -> pd.DataFrame:
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
    for feature_name, config in feature_mapping['shared_features'].items():
        oulad_col = config['oulad_col']
        
        if oulad_col in oulad_df.columns:
            if feature_name == 'sex':
                # Direct mapping for sex
                shared_data[feature_name] = oulad_df[oulad_col]
                
            elif feature_name == 'age_band':
                # Use existing age_band
                shared_data[feature_name] = oulad_df[oulad_col]
                
            elif feature_name == 'ses_proxy':
                # Transform IMD to SES proxy
                ses_values, _ = ses_standardize_transform(oulad_df[oulad_col], None)
                shared_data[feature_name] = ses_values
                
            elif feature_name == 'attendance_proxy':
                # Transform VLE clicks to attendance proxy
                attend_values, _ = attendance_normalize_transform(oulad_df[oulad_col], None)
                shared_data[feature_name] = attend_values
    
    # Add label
    if feature_mapping['label_mapping']['oulad_label'] in oulad_df.columns:
        shared_data['label'] = oulad_df[feature_mapping['label_mapping']['oulad_label']]
    
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
    X_uci, y_uci = load_data(uci_csv_path, task='classification')
    
    # Combine features and labels
    uci_df = X_uci.copy()
    uci_df['pass'] = y_uci
    
    shared_data = pd.DataFrame()
    
    # Extract shared features
    for feature_name, config in feature_mapping['shared_features'].items():
        uci_col = config['uci_col']
        
        if uci_col in uci_df.columns:
            if feature_name == 'sex':
                # Direct mapping for sex (already F/M in UCI)
                shared_data[feature_name] = uci_df[uci_col]
                
            elif feature_name == 'age_band':
                # Transform age to bands
                shared_data[feature_name] = age_to_band_transform(uci_df[uci_col])
                
            elif feature_name == 'ses_proxy':
                # Transform mother's education to SES proxy
                _, ses_values = ses_standardize_transform(None, uci_df[uci_col])
                shared_data[feature_name] = ses_values
                
            elif feature_name == 'attendance_proxy':
                # Transform absences to attendance proxy
                _, attend_values = attendance_normalize_transform(None, uci_df[uci_col])
                shared_data[feature_name] = attend_values
    
    # Add label
    shared_data['label'] = uci_df['pass']
    
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
        if col != 'label' and df_encoded[col].dtype == 'object':
            encoder = LabelEncoder()
            df_encoded[col] = encoder.fit_transform(df_encoded[col].fillna('unknown'))
            encoders[col] = encoder
    
    return df_encoded, encoders


def transfer_experiment(source_data: pd.DataFrame, target_data: pd.DataFrame, 
                       model_type: str = 'logistic') -> Dict[str, float]:
    """Run transfer learning experiment from source to target dataset.
    
    Args:
        source_data: Source dataset with shared features and labels
        target_data: Target dataset with shared features and labels
        model_type: Type of model to use
        
    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Running transfer experiment: {model_type}")
    
    # Prepare features and labels
    feature_cols = [col for col in source_data.columns if col != 'label']
    
    X_source = source_data[feature_cols]
    y_source = source_data['label']
    X_target = target_data[feature_cols]
    y_target = target_data['label']
    
    # Handle missing values
    X_source = X_source.fillna(X_source.mean())
    X_target = X_target.fillna(X_target.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    if model_type in {'logistic', 'random_forest'}:
        # Classical sklearn models
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            model = RandomForestClassifier(random_state=42, n_estimators=100)

        # Train on source dataset
        model.fit(X_source_scaled, y_source)

        # Evaluate on target dataset
        y_pred = model.predict(X_target_scaled)
        y_prob = (
            model.predict_proba(X_target_scaled)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
    elif model_type == 'mlp':
        # Delayed import to keep torch optional
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        input_dim = X_source_scaled.shape[1]

        class SimpleMLP(nn.Module):
            def __init__(self, in_dim: int):
                super().__init__()
                self.feature = nn.Sequential(
                    nn.Linear(in_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )
                self.classifier = nn.Linear(16, 1)

            def forward(self, x):
                x = self.feature(x)
                return self.classifier(x)

        def _train(model, dl, epochs):
            crit = nn.BCEWithLogitsLoss()
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    opt.zero_grad()
                    out = model(xb).squeeze()
                    loss = crit(out, yb.float())
                    loss.backward()
                    opt.step()

        # Pretrain on source
        model = SimpleMLP(input_dim)
        source_ds = TensorDataset(
            torch.tensor(X_source_scaled, dtype=torch.float32),
            torch.tensor(y_source.values, dtype=torch.float32),
        )
        source_dl = DataLoader(source_ds, batch_size=32, shuffle=True)
        _train(model, source_dl, MLP_PRETRAIN_EPOCHS)

        # Replace final layer and fine-tune on target data
        model.classifier = nn.Linear(16, 1)
        target_ds = TensorDataset(
            torch.tensor(X_target_scaled, dtype=torch.float32),
            torch.tensor(y_target.values, dtype=torch.float32),
        )
        target_dl = DataLoader(target_ds, batch_size=32, shuffle=True)
        _train(model, target_dl, MLP_FINETUNE_EPOCHS)

        # Predictions on target
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(X_target_scaled, dtype=torch.float32)
            ).squeeze()
            y_prob = torch.sigmoid(logits).cpu().numpy()
            y_pred = (y_prob > 0.5).astype(int)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(y_target, y_pred),
        'source_size': len(X_source),
        'target_size': len(X_target),
        'n_features': len(feature_cols)
    }
    
    if y_prob is not None:
        results['auc'] = roc_auc_score(y_target, y_prob)
    
    # Calculate fairness metrics by sex if available
    if 'sex' in target_data.columns:
        target_sex = target_data['sex']
        for sex_value in target_sex.unique():
            if pd.notna(sex_value):
                sex_mask = (target_sex == sex_value)
                sex_acc = accuracy_score(y_target[sex_mask], y_pred[sex_mask])
                results[f'accuracy_{sex_value}'] = sex_acc
        
        # Worst-group accuracy
        sex_accuracies = [results[k] for k in results.keys() if k.startswith('accuracy_')]
        if sex_accuracies:
            results['worst_group_accuracy'] = min(sex_accuracies)
    
    return results


def run_bidirectional_transfer(
    oulad_data_path: str,
    uci_data_path: str,
    output_dir: Path,
    table_path: Path = Path("tables/transfer_results.csv"),
    figure_path: Path = Path("figures/transfer_performance.png"),
    models: Optional[List[str]] = None,
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
    common_features = set(oulad_shared.columns) & set(uci_shared.columns) - {'label'}
    
    for col in common_features:
        if oulad_shared[col].dtype == 'object' or uci_shared[col].dtype == 'object':
            # Get union of all categories
            all_categories = set(oulad_shared[col].dropna().unique()) | set(uci_shared[col].dropna().unique())
            all_categories = sorted(list(all_categories))
            
            # Create consistent encoding
            encoder = LabelEncoder()
            encoder.fit(all_categories)
            
            oulad_shared[col] = encoder.transform(oulad_shared[col].fillna('unknown'))
            uci_shared[col] = encoder.transform(uci_shared[col].fillna('unknown'))
    
    # Filter to common features
    feature_cols = list(common_features) + ['label']
    oulad_final = oulad_shared[feature_cols].dropna()
    uci_final = uci_shared[feature_cols].dropna()
    
    logger.info(f"Final OULAD shape: {oulad_final.shape}")
    logger.info(f"Final UCI shape: {uci_final.shape}")
    
    # Run transfer experiments
    results = {}

    if models is None:
        models = ['logistic', 'random_forest', 'mlp']
    
    for model_type in models:
        # OULAD -> UCI transfer
        oulad_to_uci = transfer_experiment(oulad_final, uci_final, model_type)
        oulad_to_uci['direction'] = 'OULAD_to_UCI'
        oulad_to_uci['model'] = model_type
        
        # UCI -> OULAD transfer  
        uci_to_oulad = transfer_experiment(uci_final, oulad_final, model_type)
        uci_to_oulad['direction'] = 'UCI_to_OULAD'
        uci_to_oulad['model'] = model_type
        
        results[f'{model_type}_oulad_to_uci'] = oulad_to_uci
        results[f'{model_type}_uci_to_oulad'] = uci_to_oulad
        
        logger.info(f"{model_type} OULAD->UCI: Accuracy = {oulad_to_uci['accuracy']:.3f}")
        logger.info(f"{model_type} UCI->OULAD: Accuracy = {uci_to_oulad['accuracy']:.3f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV in results directory
    results_df = pd.DataFrame([results[k] for k in results.keys()])
    results_df.to_csv(output_dir / "transfer_summary.csv", index=False)

    # Save feature mapping
    mapping_df = pd.DataFrame([
        {"feature": fname, "oulad_col": config["oulad_col"], "uci_col": config["uci_col"]}
        for fname, config in feature_mapping["shared_features"].items()
    ])
    mapping_df.to_csv(output_dir / "feature_mapping.csv", index=False)

    # Persist combined metrics table
    table_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(table_path, index=False)

    # Visualize transfer performance
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 4))
        sns.barplot(data=results_df, x="direction", y="accuracy", hue="model")
        plt.title("Transfer Accuracy by Direction and Model")
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()
    except Exception as exc:  # pragma: no cover - visualization is best effort
        logger.warning(f"Visualization failed: {exc}")

    logger.info(f"Transfer learning results saved to {output_dir}")
    
    return results


def main():
    """CLI interface for transfer learning experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-dataset transfer learning')
    parser.add_argument(
        '--oulad-data', 
        type=str, 
        default='data/oulad/processed/oulad_ml.parquet',
        help='Path to OULAD parquet file'
    )
    parser.add_argument(
        '--uci-data',
        type=str,
        default='student-mat.csv', 
        help='Path to UCI CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='results/transfer',
        help='Output directory for results'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['logistic', 'random_forest', 'mlp'],
        help='Models to evaluate: logistic, random_forest, mlp'
    )

    args = parser.parse_args()

    try:
        results = run_bidirectional_transfer(
            args.oulad_data,
            args.uci_data,
            args.output_dir,
            models=args.models
        )
        
        # Log summary
        logger.info("\nTransfer Learning Results Summary:")
        logger.info("=" * 50)
        for exp_name, result in results.items():
            logger.info("%s:", exp_name)
            logger.info("  Accuracy: %.3f", result['accuracy'])
            if 'auc' in result:
                logger.info("  AUC: %.3f", result['auc'])
            if 'worst_group_accuracy' in result:
                logger.info(
                    "  Worst Group Accuracy: %.3f",
                    result['worst_group_accuracy'],
                )
            logger.info("")

        logger.info("Transfer learning experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"Transfer learning failed: {e}")
        raise


if __name__ == '__main__':
    setup_logging()
    main()
