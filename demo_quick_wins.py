"""
Quick Wins Transfer Learning Demo

Demonstrates the key components of the enhanced transfer learning pipeline
with simple synthetic data.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import our enhanced transfer learning modules
from src.transfer import (
    FeatureBridge, CORALTransformer, ImportanceWeighter,
    LabelShiftCorrector, CalibratedTransferClassifier,
    expected_calibration_error
)

def create_demo_datasets():
    """Create demo OULAD and UCI-like datasets."""
    # Create synthetic OULAD-like data
    oulad_data = pd.DataFrame({
        'sex': np.random.choice(['F', 'M'], 300),
        'age_band': np.random.choice(['0-35', '35-55', '55<='], 300),
        'imd_band': np.random.uniform(0, 100, 300),
        'prev_attempts': np.random.choice([0, 1, 2], 300, p=[0.7, 0.2, 0.1]),
        'studied_credits': np.random.uniform(30, 240, 300),
        'vle_total_clicks': np.random.exponential(1000, 300),
        'final_result': np.random.choice(['Pass', 'Fail', 'Distinction'], 300, p=[0.5, 0.3, 0.2])
    })
    
    # Create synthetic UCI-like data
    uci_data = pd.DataFrame({
        'sex': np.random.choice(['F', 'M'], 200),
        'age': np.random.randint(15, 25, 200),
        'Medu': np.random.choice([1, 2, 3, 4], 200),
        'G1': np.random.uniform(0, 20, 200),
        'studytime': np.random.choice([1, 2, 3, 4], 200),
        'famrel': np.random.choice([1, 2, 3, 4, 5], 200),
        'G3': np.random.uniform(0, 20, 200)
    })
    uci_data['label_pass'] = (uci_data['G3'] >= 10).astype(int)
    
    return oulad_data, uci_data

def demo_quick_wins_pipeline():
    """Demonstrate the quick wins transfer learning pipeline."""
    print("🚀 Enhanced Transfer Learning Quick Wins Demo")
    print("=" * 50)
    
    # Create demo datasets
    oulad_data, uci_data = create_demo_datasets()
    print(f"Source (OULAD): {oulad_data.shape[0]} samples")
    print(f"Target (UCI): {uci_data.shape[0]} samples")
    
    # Step 1: FeatureBridge - Unified preprocessing
    print("\n📊 Step 1: FeatureBridge Preprocessing")
    bridge = FeatureBridge(enforce_positive_class=True)
    
    # Fit on source and transform both
    bridge.fit(oulad_data, source_type='oulad')
    X_source = bridge.transform(oulad_data, source_type='oulad')
    X_target = bridge.transform(uci_data, source_type='uci')
    
    # Get targets
    y_source = bridge.get_target(oulad_data, source_type='oulad')
    y_target = bridge.get_target(uci_data, source_type='uci')
    
    print(f"✓ Unified features: {X_source.shape[1]} dimensions")
    print(f"✓ Source positive rate: {y_source.mean():.2f}")
    print(f"✓ Target positive rate: {y_target.mean():.2f}")
    
    # Step 2: Baseline model
    print("\n📈 Step 2: Baseline Transfer")
    baseline_model = LogisticRegression(random_state=42)
    baseline_model.fit(X_source, y_source)
    
    baseline_pred = baseline_model.predict(X_target)
    baseline_acc = accuracy_score(y_target, baseline_pred)
    print(f"Baseline accuracy: {baseline_acc:.3f}")
    
    # Step 3: CORAL alignment
    print("\n🎯 Step 3: CORAL Domain Adaptation")
    coral = CORALTransformer(lambda_coral=0.5)
    X_source_coral, X_target_coral = coral.fit_transform(X_source, X_target)
    
    # Retrain with CORAL-aligned features
    coral_model = LogisticRegression(random_state=42)
    coral_model.fit(X_source_coral, y_source)
    
    coral_pred = coral_model.predict(X_target_coral)
    coral_acc = accuracy_score(y_target, coral_pred)
    
    alignment_metrics = coral.get_alignment_metrics()
    print(f"✓ CORAL alignment improvement: {alignment_metrics['relative_improvement']:.1%}")
    print(f"CORAL accuracy: {coral_acc:.3f} ({coral_acc - baseline_acc:+.3f})")
    
    # Step 4: Importance weighting
    print("\n⚖️ Step 4: Importance Weighting")
    weighter = ImportanceWeighter(classifier='logistic')
    sample_weights = weighter.fit_transform(X_source_coral, X_target_coral)
    
    # Retrain with importance weights
    weighted_model = LogisticRegression(random_state=42)
    weighted_model.fit(X_source_coral, y_source, sample_weight=sample_weights)
    
    weighted_pred = weighted_model.predict(X_target_coral)
    weighted_acc = accuracy_score(y_target, weighted_pred)
    
    print(f"✓ Weight statistics: {np.mean(sample_weights):.2f} ± {np.std(sample_weights):.2f}")
    print(f"Weighted accuracy: {weighted_acc:.3f} ({weighted_acc - baseline_acc:+.3f})")
    
    # Step 5: Label shift correction
    print("\n🔄 Step 5: Label Shift Correction")
    corrector = LabelShiftCorrector(weighted_model, method='saerens_decock')
    corrector.fit(X_source_coral, y_source, X_target_coral)
    
    shift_corrected_pred = corrector.predict(X_target_coral)
    shift_corrected_acc = accuracy_score(y_target, shift_corrected_pred)
    
    shift_metrics = corrector.get_shift_metrics()
    print(f"✓ Label shift detected: {shift_metrics['shift_detected']}")
    print(f"Label shift corrected accuracy: {shift_corrected_acc:.3f} ({shift_corrected_acc - baseline_acc:+.3f})")
    
    # Step 6: Calibration and threshold tuning
    print("\n🎛️ Step 6: Calibration + Threshold Tuning")
    calib_model = CalibratedTransferClassifier(
        LogisticRegression(random_state=42),
        calibration_method='platt',
        threshold_metric='f1'
    )
    calib_model.fit(X_source_coral, y_source)
    
    calib_pred = calib_model.predict(X_target_coral)
    calib_prob = calib_model.predict_proba(X_target_coral)[:, 1]
    calib_acc = accuracy_score(y_target, calib_pred)
    
    baseline_prob = baseline_model.predict_proba(X_target)[:, 1] 
    baseline_ece = expected_calibration_error(y_target, baseline_prob)
    calib_ece = expected_calibration_error(y_target, calib_prob)
    
    calib_summary = calib_model.get_calibration_summary()
    print(f"✓ Optimal threshold: {calib_summary['threshold_analysis']['optimal_threshold']:.3f}")
    print(f"Calibrated accuracy: {calib_acc:.3f} ({calib_acc - baseline_acc:+.3f})")
    print(f"ECE improvement: {baseline_ece:.3f} → {calib_ece:.3f} ({calib_ece - baseline_ece:+.3f})")
    
    # Final summary
    print("\n📊 FINAL SUMMARY")
    print("=" * 30)
    
    improvements = {
        'Baseline': baseline_acc,
        'CORAL': coral_acc,
        'Importance Weighting': weighted_acc,  
        'Label Shift': shift_corrected_acc,
        'Calibration': calib_acc
    }
    
    best_method = max(improvements, key=improvements.get)
    best_score = improvements[best_method]
    
    for method, score in improvements.items():
        improvement = score - baseline_acc
        marker = "🏆" if method == best_method else "✓"
        print(f"{marker} {method}: {score:.3f} ({improvement:+.3f})")
    
    print(f"\n🎉 Best method: {best_method} with {best_score:.3f} accuracy")
    print(f"🚀 Total improvement: {best_score - baseline_acc:+.3f}")

if __name__ == "__main__":
    demo_quick_wins_pipeline()