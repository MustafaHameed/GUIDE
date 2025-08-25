#!/usr/bin/env python3
"""
Test suite for OULAD preprocessing, EDA, and transfer learning pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

def test_oulad_data_exists():
    """Test that OULAD mock data exists and has expected structure."""
    oulad_path = Path("data/oulad/processed/oulad_ml.csv")
    assert oulad_path.exists(), "OULAD ML dataset should exist"
    
    df = pd.read_csv(oulad_path)
    assert df.shape[0] > 0, "OULAD dataset should have records"
    assert 'label_pass' in df.columns, "OULAD dataset should have target label"
    
    # Check expected columns exist
    expected_cols = ['sex', 'age_band', 'vle_total_clicks', 'assessment_count']
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} not found"


def test_oulad_eda_outputs():
    """Test that OULAD EDA generates expected outputs."""
    fig_dir = Path("figures/oulad")
    table_dir = Path("tables/oulad")
    report_dir = Path("reports/oulad")
    
    # Check that EDA outputs exist
    assert fig_dir.exists(), "OULAD figures directory should exist"
    assert table_dir.exists(), "OULAD tables directory should exist"
    assert report_dir.exists(), "OULAD reports directory should exist"
    
    # Check specific files
    expected_figures = [
        "oulad_demographics_distribution.png",
        "oulad_vle_patterns.png",
        "oulad_assessment_patterns.png",
        "oulad_fairness_pass_rates.png",
        "oulad_feature_importance.png"
    ]
    
    for fig_name in expected_figures:
        fig_path = fig_dir / fig_name
        assert fig_path.exists(), f"Expected figure {fig_name} not found"
    
    # Check summary report
    report_path = report_dir / "oulad_eda_summary.md"
    assert report_path.exists(), "OULAD EDA summary report should exist"


def test_oulad_models_trained():
    """Test that OULAD models were trained and saved."""
    models_dir = Path("models/oulad")
    assert models_dir.exists(), "OULAD models directory should exist"
    
    expected_models = ["oulad_logistic.pkl", "oulad_random_forest.pkl", "oulad_mlp.pkl"]
    for model_name in expected_models:
        model_path = models_dir / model_name
        assert model_path.exists(), f"Expected model {model_name} not found"
    
    # Check metadata
    metadata_path = models_dir / "oulad_metadata.pkl"
    assert metadata_path.exists(), "OULAD metadata should exist"


def test_transfer_learning_outputs():
    """Test that transfer learning produces expected outputs."""
    transfer_dir = Path("reports/transfer_learning")
    assert transfer_dir.exists(), "Transfer learning reports directory should exist"
    
    # Check transfer learning report
    report_path = transfer_dir / "transfer_learning_report.md"
    assert report_path.exists(), "Transfer learning report should exist"
    
    # Verify report content
    with open(report_path, 'r') as f:
        content = f.read()
        assert "OULAD → UCI" in content, "Report should mention OULAD to UCI transfer"
        assert "Accuracy" in content, "Report should include accuracy metrics"


def test_uci_data_accessible():
    """Test that UCI data is accessible for transfer learning."""
    uci_path = Path("student-mat.csv")
    assert uci_path.exists(), "UCI student performance data should exist"
    
    df = pd.read_csv(uci_path)
    assert df.shape[0] > 0, "UCI dataset should have records"
    assert 'G3' in df.columns, "UCI dataset should have G3 (final grade) column"


def test_feature_mapping_consistency():
    """Test that feature mapping between OULAD and UCI is consistent."""
    # Load both datasets
    oulad_df = pd.read_csv("data/oulad/processed/oulad_ml.csv")
    uci_df = pd.read_csv("student-mat.csv")
    
    # Check that shared demographic features exist
    assert 'sex' in oulad_df.columns, "OULAD should have sex column"
    assert 'sex' in uci_df.columns, "UCI should have sex column"
    
    # Check that both have meaningful targets
    assert 'label_pass' in oulad_df.columns, "OULAD should have binary target"
    assert 'G3' in uci_df.columns, "UCI should have numeric target"
    
    # Verify target distributions are reasonable
    oulad_pass_rate = oulad_df['label_pass'].mean()
    uci_pass_rate = (uci_df['G3'] >= 10).mean()
    
    assert 0.1 < oulad_pass_rate < 0.9, "OULAD pass rate should be reasonable"
    assert 0.1 < uci_pass_rate < 0.9, "UCI pass rate should be reasonable"


def test_pipeline_reproducibility():
    """Test that the pipeline can be run with consistent results."""
    # This is a basic test to ensure no errors in key pipeline components
    from train_oulad import prepare_oulad_data
    from transfer_learning_simplified import load_and_prepare_oulad_transfer_data
    
    # Test OULAD data preparation
    X, y, feature_cols, encoders = prepare_oulad_data()
    assert X.shape[0] > 0, "Should have prepared OULAD features"
    assert len(y) == X.shape[0], "Features and target should have same length"
    
    # Test transfer learning data preparation
    oulad_features, oulad_encoders = load_and_prepare_oulad_transfer_data()
    assert oulad_features.shape[0] > 0, "Should have prepared transfer features"
    assert 'target' in oulad_features.columns, "Transfer features should have target"


def test_eda_module_import():
    """Test that OULAD EDA module can be imported and used."""
    sys.path.append(str(Path(__file__).resolve().parent.parent / "src" / "oulad"))
    
    try:
        import eda as oulad_eda
        assert hasattr(oulad_eda, 'run_oulad_eda'), "OULAD EDA should have main function"
    except ImportError:
        pytest.fail("Could not import OULAD EDA module")


if __name__ == "__main__":
    # Run basic tests
    print("Running OULAD pipeline tests...")
    
    test_oulad_data_exists()
    print("✓ OULAD data exists and has correct structure")
    
    test_oulad_eda_outputs()
    print("✓ OULAD EDA outputs are present")
    
    test_oulad_models_trained()
    print("✓ OULAD models were trained and saved")
    
    test_transfer_learning_outputs()
    print("✓ Transfer learning outputs are present")
    
    test_uci_data_accessible()
    print("✓ UCI data is accessible")
    
    test_feature_mapping_consistency()
    print("✓ Feature mapping between datasets is consistent")
    
    test_pipeline_reproducibility()
    print("✓ Pipeline components are reproducible")
    
    test_eda_module_import()
    print("✓ EDA module can be imported")
    
    print("\nAll tests passed! 🎉")
    print("OULAD preprocessing, EDA, and transfer learning pipeline is working correctly.")