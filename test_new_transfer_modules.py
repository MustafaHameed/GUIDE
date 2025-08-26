"""
Test suite for the new transfer learning modules.

Tests the FeatureBridge, MMD, TENT, calibration, and ablation runner modules
to ensure they work correctly with both synthetic and real-like data.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.transfer.feature_bridge import FeatureBridge, create_feature_bridge
from src.transfer.mmd import compute_mmd, MMDTransformer, apply_mmd_alignment
from src.transfer.tent import TENTAdapter, apply_tent_adaptation
from src.transfer.calibration import (
    expected_calibration_error, CalibratedTransferClassifier,
    OptimalThresholdFinder, evaluate_calibration_transfer
)
from src.transfer.ablation_runner import TransferLearningAblation


class TestFeatureBridge:
    """Test the FeatureBridge unified preprocessing."""
    
    def test_feature_bridge_initialization(self):
        """Test FeatureBridge can be initialized."""
        bridge = FeatureBridge()
        assert bridge.config is not None
        assert bridge.enforce_positive_class == True
        
    def test_feature_bridge_uci_mapping(self):
        """Test UCI to canonical mapping."""
        # Create synthetic UCI-like data
        uci_data = pd.DataFrame({
            'sex': ['F', 'M', 'F'],
            'age': [18, 19, 20],
            'Medu': [2, 3, 1],
            'G1': [12, 8, 15],
            'studytime': [2, 1, 3],
            'G3': [14, 6, 16]
        })
        
        bridge = FeatureBridge()
        canonical_df = bridge.map_dataset_to_canonical(uci_data, 'uci')
        
        # Check that canonical features exist
        assert 'sex' in canonical_df.columns
        assert 'age_group' in canonical_df.columns
        assert 'ses_index' in canonical_df.columns
        assert 'label_pass' in canonical_df.columns
        
        # Check positive class convention
        assert canonical_df['label_pass'].dtype == int
        assert set(canonical_df['label_pass'].unique()).issubset({0, 1})
        
    def test_feature_bridge_oulad_mapping(self):
        """Test OULAD to canonical mapping."""
        # Create synthetic OULAD-like data
        oulad_data = pd.DataFrame({
            'sex': ['F', 'M', 'F'],
            'age_band': ['0-35', '35-55', '0-35'],
            'imd_band': [20, 40, 60],
            'prev_attempts': [0, 1, 0],
            'studied_credits': [60, 120, 90],
            'vle_total_clicks': [500, 1200, 300],
            'final_result': ['Pass', 'Fail', 'Distinction']
        })
        
        bridge = FeatureBridge()
        canonical_df = bridge.map_dataset_to_canonical(oulad_data, 'oulad')
        
        # Check canonical features
        assert 'sex' in canonical_df.columns
        assert 'age_group' in canonical_df.columns
        assert 'ses_index' in canonical_df.columns
        assert 'label_pass' in canonical_df.columns
        
        # Check label mapping
        assert canonical_df.loc[0, 'label_pass'] == 1  # Pass
        assert canonical_df.loc[1, 'label_pass'] == 0  # Fail
        assert canonical_df.loc[2, 'label_pass'] == 1  # Distinction
        
    def test_feature_bridge_fit_transform(self):
        """Test FeatureBridge fit and transform."""
        # Create synthetic data
        data = pd.DataFrame({
            'sex': ['F', 'M', 'F', 'M'],
            'age': [18, 19, 20, 21],
            'Medu': [2, 3, 1, 4],
            'G1': [12, 8, 15, 18],
            'studytime': [2, 1, 3, 4],
            'G3': [14, 6, 16, 19]
        })
        
        bridge = FeatureBridge()
        bridge.fit(data, source_type='uci')
        
        # Check fitted attributes
        assert bridge.is_fitted_
        assert bridge.canonical_features_ is not None
        assert bridge.preprocessor_ is not None
        
        # Transform data
        X_transformed = bridge.transform(data, source_type='uci')
        assert X_transformed.shape[0] == len(data)
        assert X_transformed.shape[1] > 0
        
        # Get target
        y = bridge.get_target(data, source_type='uci')
        assert len(y) == len(data)
        assert y.dtype == int


class TestMMD:
    """Test MMD domain adaptation."""
    
    def test_mmd_computation(self):
        """Test basic MMD computation."""
        # Generate data with known distribution difference
        X_source = np.random.normal(0, 1, (100, 3))
        X_target = np.random.normal(1, 1, (80, 3))
        
        # Linear MMD should be positive (different means)
        mmd_linear = compute_mmd(X_source, X_target, kernel='linear')
        assert mmd_linear > 0
        
        # RBF MMD should be positive
        mmd_rbf = compute_mmd(X_source, X_target, kernel='rbf', gamma=1.0)
        assert mmd_rbf > 0
        
        # Identical distributions should have MMD ≈ 0
        mmd_identical = compute_mmd(X_source, X_source, kernel='linear')
        assert mmd_identical < 0.1  # Should be close to 0
        
    def test_mmd_transformer(self):
        """Test MMD transformer."""
        X_source = np.random.normal(0, 1, (100, 3))
        X_target = np.random.normal(0.5, 1.2, (80, 3))
        
        mmd_transformer = MMDTransformer(kernel='rbf', max_iterations=10)
        mmd_transformer.fit(X_source, X_target)
        
        assert mmd_transformer.is_fitted_
        
        # Transform data
        X_source_transformed = mmd_transformer.transform(X_source, domain='source')
        X_target_transformed = mmd_transformer.transform(X_target, domain='target')
        
        assert X_source_transformed.shape == X_source.shape
        assert X_target_transformed.shape == X_target.shape
        
        # Check MMD reduction
        metrics = mmd_transformer.get_mmd_reduction(X_source, X_target)
        assert 'mmd_before' in metrics
        assert 'mmd_after' in metrics
        assert 'mmd_reduction' in metrics


class TestTENT:
    """Test TENT adaptation."""
    
    def test_tent_adapter_initialization(self):
        """Test TENT adapter initialization."""
        base_clf = LogisticRegression()
        tent = TENTAdapter(base_clf)
        
        assert tent.base_classifier is base_clf
        assert tent.adaptation_strategy == 'entropy'
        assert not tent.is_fitted_
        
    def test_tent_adaptation(self):
        """Test TENT adaptation process."""
        # Generate classification data
        X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
        X_source, X_target, y_source, y_target = train_test_split(X, y, test_size=0.5, random_state=42)
        
        # Add domain shift
        X_target += np.random.normal(0.3, 0.2, X_target.shape)
        
        # Train base classifier
        base_clf = LogisticRegression(random_state=42)
        base_clf.fit(X_source, y_source)
        
        # Apply TENT
        tent = TENTAdapter(base_clf, max_iterations=5)
        tent.adapt(X_target)
        
        assert tent.is_fitted_
        assert tent.adapted_classifier_ is not None
        
        # Test predictions
        y_pred = tent.predict(X_target)
        assert len(y_pred) == len(X_target)
        assert set(y_pred).issubset({0, 1})
        
        # Test probabilities
        y_prob = tent.predict_proba(X_target)
        assert y_prob.shape == (len(X_target), 2)


class TestCalibration:
    """Test calibration and threshold tuning."""
    
    def test_ece_computation(self):
        """Test Expected Calibration Error computation."""
        # Perfect calibration case
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob_perfect = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        
        ece_perfect = expected_calibration_error(y_true, y_prob_perfect, n_bins=5)
        assert ece_perfect < 0.3  # Should be low for well-calibrated predictions
        
        # Poor calibration case
        y_prob_poor = np.array([0.9, 0.8, 0.2, 0.1, 0.3])
        ece_poor = expected_calibration_error(y_true, y_prob_poor, n_bins=5)
        assert ece_poor > ece_perfect  # Should be higher for poorly calibrated
        
    def test_threshold_finder(self):
        """Test optimal threshold finding."""
        # Generate imbalanced data
        y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.4, 0.8, 0.15, 0.9, 0.85])
        
        # Find optimal threshold for F1
        finder = OptimalThresholdFinder(metric='f1')
        finder.fit(y_true, y_prob)
        
        assert finder.optimal_threshold_ is not None
        assert 0 <= finder.optimal_threshold_ <= 1
        
        # Make predictions
        y_pred = finder.predict(y_prob)
        assert len(y_pred) == len(y_prob)
        assert set(y_pred).issubset({0, 1})
        
    def test_calibrated_transfer_classifier(self):
        """Test calibrated transfer classifier."""
        # Generate data
        X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create calibrated classifier
        base_clf = RandomForestClassifier(n_estimators=10, random_state=42)
        calib_clf = CalibratedTransferClassifier(
            base_clf, 
            calibration_method='platt',
            threshold_metric='f1'
        )
        
        # Fit and test
        calib_clf.fit(X_train, y_train)
        assert calib_clf.is_fitted_
        
        y_pred = calib_clf.predict(X_test)
        y_prob = calib_clf.predict_proba(X_test)
        
        assert len(y_pred) == len(X_test)
        assert y_prob.shape == (len(X_test), 2)
        
        # Get calibration summary
        summary = calib_clf.get_calibration_summary()
        assert 'calibration_method' in summary
        assert 'calibration_metrics' in summary


class TestAblationRunner:
    """Test ablation study runner."""
    
    def test_ablation_initialization(self):
        """Test ablation runner initialization."""
        base_clf = LogisticRegression()
        ablation = TransferLearningAblation(base_clf, output_dir="test_ablation")
        
        assert ablation.base_classifier is base_clf
        assert ablation.output_dir.name == "test_ablation"
        assert ablation.results_ == []
        
    def test_single_experiment(self):
        """Test single ablation experiment."""
        # Generate simple data
        X_source = np.random.normal(0, 1, (50, 4))
        y_source = np.random.binomial(1, 0.5, 50)
        X_target = np.random.normal(0.2, 1, (30, 4))
        y_target = np.random.binomial(1, 0.5, 30)
        
        base_clf = LogisticRegression()
        ablation = TransferLearningAblation(base_clf, output_dir="test_ablation")
        
        # Run single experiment
        result = ablation._run_single_experiment(
            X_source, y_source, X_target, y_target,
            'uci', 'uci', 'test_experiment', {}
        )
        
        assert 'experiment_name' in result
        assert 'status' in result
        assert result['experiment_name'] == 'test_experiment'


def test_integration_workflow():
    """Test integration of multiple components."""
    # Generate transfer learning scenario
    X, y = make_classification(n_samples=300, n_features=8, n_classes=2, random_state=42)
    X_source, X_target, y_source, y_target = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Add domain shift
    X_target += np.random.normal(0.3, 0.2, X_target.shape)
    
    # Convert to DataFrame for FeatureBridge compatibility
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_source_df = pd.DataFrame(X_source, columns=feature_names)
    X_target_df = pd.DataFrame(X_target, columns=feature_names)
    
    # Step 1: Feature preprocessing (simplified for synthetic data)
    # Skip FeatureBridge for synthetic data
    
    # Step 2: Train base classifier
    base_clf = LogisticRegression(random_state=42)
    base_clf.fit(X_source, y_source)
    
    # Step 3: Apply domain adaptation (CORAL)
    from src.transfer.coral import CORALTransformer
    coral = CORALTransformer(lambda_coral=0.5)
    X_source_coral, X_target_coral = coral.fit_transform(X_source, X_target)
    
    # Step 4: Retrain with adapted features
    adapted_clf = LogisticRegression(random_state=42)
    adapted_clf.fit(X_source_coral, y_source)
    
    # Step 5: Apply calibration
    calib_clf = CalibratedTransferClassifier(
        adapted_clf,
        calibration_method='platt',
        threshold_metric='f1'
    )
    calib_clf.fit(X_source_coral, y_source)
    
    # Step 6: Evaluate
    y_pred = calib_clf.predict(X_target_coral)
    
    # Basic checks
    assert len(y_pred) == len(y_target)
    assert set(y_pred).issubset({0, 1})
    
    # Compute accuracy (should be reasonable)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_target, y_pred)
    assert 0.3 <= accuracy <= 1.0  # Reasonable range


if __name__ == "__main__":
    # Run tests
    print("Running transfer learning module tests...")
    
    # Test individual components
    test_feature_bridge = TestFeatureBridge()
    test_feature_bridge.test_feature_bridge_initialization()
    test_feature_bridge.test_feature_bridge_uci_mapping()
    test_feature_bridge.test_feature_bridge_oulad_mapping()
    test_feature_bridge.test_feature_bridge_fit_transform()
    print("✓ FeatureBridge tests passed")
    
    test_mmd = TestMMD()
    test_mmd.test_mmd_computation()
    test_mmd.test_mmd_transformer()
    print("✓ MMD tests passed")
    
    test_tent = TestTENT()
    test_tent.test_tent_adapter_initialization()
    test_tent.test_tent_adaptation()
    print("✓ TENT tests passed")
    
    test_calibration = TestCalibration()
    test_calibration.test_ece_computation()
    test_calibration.test_threshold_finder()
    test_calibration.test_calibrated_transfer_classifier()
    print("✓ Calibration tests passed")
    
    test_ablation = TestAblationRunner()
    test_ablation.test_ablation_initialization()
    test_ablation.test_single_experiment()
    print("✓ Ablation runner tests passed")
    
    # Test integration
    test_integration_workflow()
    print("✓ Integration workflow test passed")
    
    print("\nAll tests completed successfully! ✓")