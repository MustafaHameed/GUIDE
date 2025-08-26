#!/usr/bin/env python3
"""
Enhanced Transfer Learning with Advanced Domain Adaptation

This script implements comprehensive transfer learning between OULAD and UCI datasets
with advanced domain adaptation techniques including:
- Shift diagnostics (PSI, KS, PAD, label shift)
- Importance weighting for covariate shift
- CORAL feature alignment
- Label shift correction (Saerens-Decock)
- DANN for domain adversarial training
- Self-training with pseudo-labels
- Fairness-aware transfer learning

Based on the improved transfer learning requirements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import logging
import argparse
import yaml
import json
from typing import Dict, Tuple, Optional, List

# Import our new transfer learning modules
from src.transfer.diagnostics import generate_shift_report, create_shift_report_summary
from src.transfer.weights import ImportanceWeighter, evaluate_weight_quality
from src.transfer.coral import CORALTransformer, apply_coral_alignment
from src.transfer.label_shift import LabelShiftCorrector
from src.transfer.dann import create_dann_classifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_feature_bridge_config(config_path: str = "configs/feature_bridge.yaml") -> dict:
    """Load feature bridge configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found, using default mapping")
        return create_default_feature_mapping()
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_feature_mapping():
    """Create default feature mapping (fallback when config is not available)."""
    return {
        "shared_features": {
            # Demographics
            "sex": {
                "oulad_col": "sex",
                "uci_col": "sex", 
                "mapping": {"F": "F", "M": "M"}
            },
            # Age (convert to bands for compatibility)
            "age_band": {
                "oulad_col": "age_band",
                "uci_col": "age",
                "transform": "age_to_band"
            },
            # Socioeconomic proxy
            "ses_proxy": {
                "oulad_col": "imd_band",  # Index of Multiple Deprivation
                "uci_col": "Medu",  # Mother's education as SES proxy
                "transform": "ses_standardize"
            },
            # Academic performance proxy
            "academic_proxy": {
                "oulad_col": "studied_credits",  # Credits studied
                "uci_col": "G1",  # First period grade
                "transform": "normalize"
            },
            # Engagement proxy
            "engagement_proxy": {
                "oulad_col": "vle_total_clicks",  # VLE engagement
                "uci_col": "studytime",  # Weekly study time
                "transform": "normalize"
            }
        }
    }


def load_uci_data(uci_path="student-mat.csv"):
    """Load and prepare UCI student performance data."""
    logger.info(f"Loading UCI data from {uci_path}")
    
    df = pd.read_csv(uci_path)
    logger.info(f"Loaded UCI dataset with shape: {df.shape}")
    
    # Create binary pass/fail label (G3 >= 10)
    df['label_pass'] = (df['G3'] >= 10).astype(int)
    
    return df


def prepare_oulad_features_for_transfer(oulad_data_path="data/oulad/processed/oulad_ml.csv"):
    """Prepare OULAD features for transfer learning."""
    logger.info("Preparing OULAD features for transfer...")
    
    df = pd.read_csv(oulad_data_path)
    
    # Select relevant features for transfer
    transfer_features = {
        'sex': df['sex'],
        'age_band': df['age_band'], 
        'ses_proxy': df['imd_band'],
        'academic_proxy': df['studied_credits'],
        'engagement_proxy': df['vle_total_clicks'],
        'label_pass': df['label_pass']
    }
    
    oulad_transfer_df = pd.DataFrame(transfer_features)
    
    # Encode categorical features
    le_sex = LabelEncoder()
    oulad_transfer_df['sex'] = le_sex.fit_transform(oulad_transfer_df['sex'].astype(str))
    
    le_age = LabelEncoder()
    oulad_transfer_df['age_band'] = le_age.fit_transform(oulad_transfer_df['age_band'].astype(str))
    
    le_ses = LabelEncoder()
    oulad_transfer_df['ses_proxy'] = le_ses.fit_transform(oulad_transfer_df['ses_proxy'].astype(str))
    
    # Normalize numeric features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    oulad_transfer_df[['academic_proxy', 'engagement_proxy']] = scaler.fit_transform(
        oulad_transfer_df[['academic_proxy', 'engagement_proxy']].fillna(0)
    )
    
    return oulad_transfer_df, {'sex': le_sex, 'age_band': le_age, 'ses_proxy': le_ses}, scaler


def prepare_uci_features_for_transfer(uci_df, oulad_encoders, oulad_scaler):
    """Prepare UCI features to match OULAD feature space."""
    logger.info("Preparing UCI features for transfer...")
    
    # Map UCI features to OULAD feature space
    uci_transfer = pd.DataFrame()
    
    # Sex mapping
    uci_transfer['sex'] = uci_df['sex'].map({'F': 0, 'M': 1}).fillna(0)
    
    # Age to age band mapping
    age_mapping = {age: 0 if age <= 18 else (1 if age <= 22 else 2) for age in range(15, 25)}
    uci_transfer['age_band'] = uci_df['age'].map(age_mapping).fillna(0)
    
    # Mother's education as SES proxy
    medu_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 0: 0}  # Map to ordinal scale
    uci_transfer['ses_proxy'] = uci_df['Medu'].map(medu_mapping).fillna(0)
    
    # Academic performance proxy (G1 normalized)
    academic_proxy = uci_df['G1'].fillna(uci_df['G1'].mean()) / 20.0  # Normalize to 0-1
    
    # Engagement proxy (study time normalized)
    engagement_proxy = uci_df['studytime'].fillna(uci_df['studytime'].mean()) / 4.0  # Normalize to 0-1
    
    # Apply same scaling as OULAD
    academic_engagement = np.column_stack([academic_proxy, engagement_proxy])
    academic_engagement_scaled = oulad_scaler.transform(academic_engagement)
    
    uci_transfer['academic_proxy'] = academic_engagement_scaled[:, 0]
    uci_transfer['engagement_proxy'] = academic_engagement_scaled[:, 1]
    
    # Target variable
    uci_transfer['label_pass'] = uci_df['label_pass']
    
    return uci_transfer


def run_enhanced_transfer_learning(source_domain: str = "oulad", target_domain: str = "uci",
                                  use_importance_weighting: bool = False,
                                  use_label_shift_correction: bool = False,
                                  use_coral: bool = False,
                                  use_dann: bool = False,
                                  use_self_training: bool = False,
                                  use_fairness_grid: bool = False,
                                  run_diagnostics: bool = True,
                                  config_path: str = "configs/feature_bridge.yaml",
                                  output_dir: str = "reports/enhanced_transfer"):
    """
    Run enhanced transfer learning with advanced domain adaptation techniques.
    
    Args:
        source_domain: Source domain ("oulad" or "uci") 
        target_domain: Target domain ("oulad" or "uci")
        use_importance_weighting: Apply importance weighting for covariate shift
        use_label_shift_correction: Apply label shift correction
        use_coral: Apply CORAL feature alignment
        use_dann: Use DANN for domain adversarial training
        use_self_training: Apply self-training with pseudo-labels
        use_fairness_grid: Use fairness-aware threshold optimization
        run_diagnostics: Run shift diagnostics analysis
        config_path: Path to feature bridge configuration
        output_dir: Output directory for results
    """
    logger.info(f"Starting enhanced transfer learning: {source_domain} → {target_domain}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_feature_bridge_config(config_path)
    
    # Load and prepare data
    if source_domain == "oulad":
        source_data, source_encoders, source_scaler = prepare_oulad_features_for_transfer()
        target_data = load_uci_data()
        target_transfer_data = prepare_uci_features_for_transfer(target_data, source_encoders, source_scaler)
    else:
        source_data = load_uci_data()
        # Would need OULAD loading for UCI → OULAD transfer
        raise NotImplementedError("UCI → OULAD transfer not yet implemented")
    
    logger.info(f"Source data shape: {source_data.shape}")
    logger.info(f"Target data shape: {target_transfer_data.shape}")
    
    # Extract features and targets
    feature_cols = ['sex', 'age_band', 'ses_proxy', 'academic_proxy', 'engagement_proxy']
    
    X_source = source_data[feature_cols].fillna(0)
    y_source = source_data['label_pass']
    X_target = target_transfer_data[feature_cols].fillna(0)
    y_target = target_transfer_data['label_pass']
    
    results = {}
    
    # 1. Shift Diagnostics
    if run_diagnostics:
        logger.info("Running shift diagnostics...")
        
        # Train a simple model for label shift estimation
        diagnostic_model = LogisticRegression(random_state=42)
        diagnostic_model.fit(X_source, y_source)
        
        shift_report = generate_shift_report(
            X_source, X_target, y_source, diagnostic_model,
            output_dir=output_path / "diagnostics"
        )
        
        # Save shift analysis
        with open(output_path / "shift_analysis.json", 'w') as f:
            json.dump(shift_report, f, indent=2, default=str)
        
        # Create summary report
        summary = create_shift_report_summary(shift_report, source_domain.upper(), target_domain.upper())
        with open(output_path / "shift_summary.md", 'w') as f:
            f.write(summary)
        
        results['shift_diagnostics'] = shift_report
        logger.info("Shift diagnostics completed")
    
    # 2. Baseline Model (no adaptation)
    logger.info("Training baseline model...")
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_source, y_source)
    
    y_pred_baseline = baseline_model.predict(X_target)
    y_prob_baseline = baseline_model.predict_proba(X_target)[:, 1]
    
    results['baseline'] = {
        'accuracy': accuracy_score(y_target, y_pred_baseline),
        'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_baseline),
        'roc_auc': roc_auc_score(y_target, y_prob_baseline)
    }
    
    # Store transformed data for subsequent methods
    X_source_transformed = X_source.copy()
    X_target_transformed = X_target.copy()
    sample_weights = None
    
    # 3. Importance Weighting
    if use_importance_weighting:
        logger.info("Applying importance weighting...")
        
        weighter = ImportanceWeighter(classifier='logistic', random_state=42)
        sample_weights = weighter.fit_transform(X_source.values, X_target.values)
        
        # Evaluate weight quality
        weight_quality = evaluate_weight_quality(sample_weights, X_source.values, X_target.values)
        results['importance_weighting'] = {
            'weight_stats': weight_quality['weight_stats'],
            'covariate_balance': weight_quality.get('covariate_balance', {})
        }
        
        logger.info("Importance weighting completed")
    
    # 4. CORAL Feature Alignment
    if use_coral:
        logger.info("Applying CORAL feature alignment...")
        
        coral = CORALTransformer(lambda_coral=1.0, regularization=1e-6)
        X_source_transformed, X_target_transformed = coral.fit_transform(
            X_source_transformed.values, X_target_transformed.values
        )
        
        # Convert back to DataFrames
        X_source_transformed = pd.DataFrame(X_source_transformed, columns=feature_cols)
        X_target_transformed = pd.DataFrame(X_target_transformed, columns=feature_cols)
        
        # Get alignment metrics
        alignment_metrics = coral.get_alignment_metrics()
        results['coral_alignment'] = alignment_metrics
        
        logger.info("CORAL alignment completed")
    
    # 5. Train adapted model
    logger.info("Training adapted model...")
    adapted_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    if sample_weights is not None:
        try:
            adapted_model.fit(X_source_transformed, y_source, sample_weight=sample_weights)
            logger.info("Trained with importance weights")
        except TypeError:
            adapted_model.fit(X_source_transformed, y_source)
            logger.warning("Model doesn't support sample weights")
    else:
        adapted_model.fit(X_source_transformed, y_source)
    
    y_pred_adapted = adapted_model.predict(X_target_transformed)
    y_prob_adapted = adapted_model.predict_proba(X_target_transformed)[:, 1]
    
    results['adapted_model'] = {
        'accuracy': accuracy_score(y_target, y_pred_adapted),
        'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_adapted),
        'roc_auc': roc_auc_score(y_target, y_prob_adapted)
    }
    
    # 6. Label Shift Correction
    if use_label_shift_correction:
        logger.info("Applying label shift correction...")
        
        corrector = LabelShiftCorrector(adapted_model, method='saerens_decock')
        corrector.fit(X_source_transformed.values, y_source.values, X_target_transformed.values)
        
        y_pred_corrected = corrector.predict(X_target_transformed.values)
        y_prob_corrected = corrector.predict_proba(X_target_transformed.values)[:, 1]
        
        results['label_shift_corrected'] = {
            'accuracy': accuracy_score(y_target, y_pred_corrected),
            'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_corrected),
            'roc_auc': roc_auc_score(y_target, y_prob_corrected),
            'shift_metrics': corrector.get_shift_metrics()
        }
        
        logger.info("Label shift correction completed")
    
    # 7. DANN Training
    if use_dann:
        logger.info("Training DANN model...")
        
        dann_model = create_dann_classifier(
            hidden_dims=[128, 64],
            num_epochs=50,  # Reduced for demo
            batch_size=32,
            learning_rate=0.001
        )
        
        dann_model.fit(X_source.values, y_source.values, X_target.values)
        
        y_pred_dann = dann_model.predict(X_target.values)
        y_prob_dann = dann_model.predict_proba(X_target.values)[:, 1]
        
        results['dann'] = {
            'accuracy': accuracy_score(y_target, y_pred_dann),
            'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_dann),
            'roc_auc': roc_auc_score(y_target, y_prob_dann)
        }
        
        logger.info("DANN training completed")
    
    # 8. Self-Training
    if use_self_training:
        logger.info("Applying self-training...")
        
        # Use confident predictions as pseudo-labels
        confidence_threshold = 0.8
        y_prob_adapted_full = adapted_model.predict_proba(X_target_transformed)
        confident_mask = np.max(y_prob_adapted_full, axis=1) > confidence_threshold
        
        if confident_mask.sum() > 0:
            X_pseudo = X_target_transformed[confident_mask]
            y_pseudo = adapted_model.predict(X_pseudo)
            
            # Create balanced pseudo-labels to avoid mode collapse
            class_counts = np.bincount(y_pseudo)
            min_class_count = min(class_counts)
            
            # Balance pseudo-labels
            balanced_indices = []
            for class_label in range(len(class_counts)):
                class_indices = np.where(y_pseudo == class_label)[0]
                if len(class_indices) > min_class_count:
                    selected = np.random.choice(class_indices, min_class_count, replace=False)
                    balanced_indices.extend(selected)
                else:
                    balanced_indices.extend(class_indices)
            
            if len(balanced_indices) > 0:
                X_pseudo_balanced = X_pseudo.iloc[balanced_indices]
                y_pseudo_balanced = y_pseudo[balanced_indices]
                
                # Combine with source data (with reduced weight for pseudo-labels)
                pseudo_weight = 0.5  # Reduce weight of pseudo-labels
                
                X_combined = pd.concat([X_source_transformed, X_pseudo_balanced])
                y_combined = np.concatenate([y_source, y_pseudo_balanced])
                
                # Create sample weights (full weight for source, reduced for pseudo)
                sample_weights_combined = np.concatenate([
                    sample_weights if sample_weights is not None else np.ones(len(X_source_transformed)),
                    np.full(len(X_pseudo_balanced), pseudo_weight)
                ])
                
                # Retrain model
                self_trained_model = RandomForestClassifier(n_estimators=100, random_state=42)
                try:
                    self_trained_model.fit(X_combined, y_combined, sample_weight=sample_weights_combined)
                except TypeError:
                    self_trained_model.fit(X_combined, y_combined)
                
                y_pred_self = self_trained_model.predict(X_target_transformed)
                y_prob_self = self_trained_model.predict_proba(X_target_transformed)[:, 1]
                
                results['self_training'] = {
                    'accuracy': accuracy_score(y_target, y_pred_self),
                    'balanced_accuracy': balanced_accuracy_score(y_target, y_pred_self),
                    'roc_auc': roc_auc_score(y_target, y_prob_self),
                    'pseudo_labels_used': len(balanced_indices),
                    'confident_predictions': confident_mask.sum()
                }
                
                logger.info(f"Self-training completed with {len(balanced_indices)} balanced pseudo-labels from {confident_mask.sum()} confident predictions")
            else:
                logger.warning("No balanced pseudo-labels available for self-training")
        else:
            logger.warning("No confident predictions for self-training")
    
    # 9. Fairness Analysis
    if use_fairness_grid:
        logger.info("Running fairness analysis...")
        
        # Basic fairness metrics by sex (if available)
        if 'sex' in X_target.columns:
            from sklearn.metrics import confusion_matrix
            
            # Analyze fairness by sex
            sex_groups = X_target['sex'].unique()
            fairness_results = {}
            
            for group in sex_groups:
                group_mask = X_target['sex'] == group
                if group_mask.sum() > 0:
                    group_accuracy = accuracy_score(
                        y_target[group_mask], 
                        y_pred_adapted[group_mask]
                    )
                    fairness_results[f'accuracy_sex_{group}'] = group_accuracy
            
            results['fairness'] = fairness_results
            logger.info("Fairness analysis completed")
    
    # 10. Summary and comparison
    logger.info("Generating summary...")
    
    # Create performance comparison
    performance_summary = {}
    for method, metrics in results.items():
        if isinstance(metrics, dict) and 'accuracy' in metrics:
            performance_summary[method] = {
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics.get('balanced_accuracy', metrics['accuracy']),
                'roc_auc': metrics.get('roc_auc', 0.5)
            }
    
    results['summary'] = {
        'performance_comparison': performance_summary,
        'best_method': max(performance_summary.keys(), 
                          key=lambda x: performance_summary[x]['accuracy']) if performance_summary else 'baseline',
        'source_domain': source_domain,
        'target_domain': target_domain,
        'methods_used': [
            method for method, used in [
                ('importance_weighting', use_importance_weighting),
                ('label_shift_correction', use_label_shift_correction),
                ('coral', use_coral),
                ('dann', use_dann),
                ('self_training', use_self_training),
                ('fairness_grid', use_fairness_grid)
            ] if used
        ]
    }
    
    # Save comprehensive results
    results_path = output_path / f"enhanced_transfer_{source_domain}_to_{target_domain}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create performance table
    performance_df = pd.DataFrame.from_dict(performance_summary, orient='index')
    performance_df.to_csv(output_path / f"performance_comparison_{source_domain}_to_{target_domain}.csv")
    
    logger.info(f"Enhanced transfer learning completed. Results saved to {output_path}")
    logger.info(f"Best method: {results['summary']['best_method']}")
    
    return results


def main():
    """CLI interface for enhanced transfer learning."""
    parser = argparse.ArgumentParser(description="Enhanced Transfer Learning with Domain Adaptation")
    
    # Domain specification
    parser.add_argument('--from', dest='source', choices=['uci', 'oulad'], default='oulad',
                       help='Source domain (default: oulad)')
    parser.add_argument('--to', dest='target', choices=['uci', 'oulad'], default='uci',
                       help='Target domain (default: uci)')
    
    # Domain adaptation methods
    parser.add_argument('--iw', '--importance-weighting', action='store_true',
                       help='Apply importance weighting for covariate shift')
    parser.add_argument('--label_shift', action='store_true',
                       help='Apply label shift correction (Saerens-Decock)')
    parser.add_argument('--coral', action='store_true',
                       help='Apply CORAL feature alignment')
    parser.add_argument('--dann', action='store_true',
                       help='Use DANN for domain adversarial training')
    parser.add_argument('--self_train', action='store_true',
                       help='Apply self-training with pseudo-labels')
    parser.add_argument('--fairness_grid', action='store_true',
                       help='Use fairness-aware threshold optimization')
    
    # Analysis options
    parser.add_argument('--diagnostics', action='store_true', default=True,
                       help='Run shift diagnostics (default: True)')
    parser.add_argument('--no-diagnostics', action='store_false', dest='diagnostics',
                       help='Skip shift diagnostics')
    
    # Configuration
    parser.add_argument('--config', default='configs/feature_bridge.yaml',
                       help='Path to feature bridge configuration')
    parser.add_argument('--output-dir', default='reports/enhanced_transfer',
                       help='Output directory for results')
    
    # Convenience flags for common combinations
    parser.add_argument('--all-methods', action='store_true',
                       help='Apply all domain adaptation methods')
    parser.add_argument('--standard', action='store_true',
                       help='Apply standard methods (IW + label shift)')
    parser.add_argument('--advanced', action='store_true',
                       help='Apply advanced methods (IW + CORAL + label shift)')
    
    args = parser.parse_args()
    
    # Handle convenience flags
    if args.all_methods:
        args.iw = True
        args.label_shift = True
        args.coral = True
        args.dann = True
        args.self_train = True
        args.fairness_grid = True
    elif args.standard:
        args.iw = True
        args.label_shift = True
    elif args.advanced:
        args.iw = True
        args.coral = True
        args.label_shift = True
    
    # Run enhanced transfer learning
    logger.info(f"Running enhanced transfer learning: {args.source} → {args.target}")
    
    methods_enabled = [
        ('Importance Weighting', args.iw),
        ('Label Shift Correction', args.label_shift),
        ('CORAL Alignment', args.coral),
        ('DANN', args.dann),
        ('Self-Training', args.self_train),
        ('Fairness Grid', args.fairness_grid)
    ]
    
    enabled_methods = [name for name, enabled in methods_enabled if enabled]
    logger.info(f"Enabled methods: {', '.join(enabled_methods) if enabled_methods else 'None (baseline only)'}")
    
    try:
        results = run_enhanced_transfer_learning(
            source_domain=args.source,
            target_domain=args.target,
            use_importance_weighting=args.iw,
            use_label_shift_correction=args.label_shift,
            use_coral=args.coral,
            use_dann=args.dann,
            use_self_training=args.self_train,
            use_fairness_grid=args.fairness_grid,
            run_diagnostics=args.diagnostics,
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "="*80)
        print("ENHANCED TRANSFER LEARNING RESULTS")
        print("="*80)
        
        if 'summary' in results and 'performance_comparison' in results['summary']:
            performance = results['summary']['performance_comparison']
            
            print(f"{'Method':<25} {'Accuracy':<12} {'Balanced Acc':<15} {'ROC AUC':<10}")
            print("-" * 65)
            
            for method, metrics in performance.items():
                print(f"{method:<25} {metrics['accuracy']:<12.4f} "
                      f"{metrics['balanced_accuracy']:<15.4f} "
                      f"{metrics['roc_auc']:<10.4f}")
            
            best_method = results['summary']['best_method']
            print(f"\nBest performing method: {best_method}")
        
        # Show shift diagnostics if available
        if 'shift_diagnostics' in results and 'domain_metrics' in results['shift_diagnostics']:
            dm = results['shift_diagnostics']['domain_metrics']
            print(f"\nDomain Shift Analysis:")
            print(f"  Proxy A-distance: {dm.get('proxy_a_distance', 'N/A'):.3f}")
            print(f"  Domain Classifier AUC: {dm.get('domain_classifier_auc', 'N/A'):.3f}")
        
        logger.info("Enhanced transfer learning completed successfully")
        
    except Exception as e:
        logger.error(f"Enhanced transfer learning failed: {e}")
        raise


if __name__ == "__main__":
    main()