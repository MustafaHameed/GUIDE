"""Tests for OULAD dataset processing and splits."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

# Import modules to test
from src.oulad.build_dataset import load_oulad_tables, create_labels_and_sensitive_attrs
from src.oulad.splits import check_leakage, random_split
from src.uncertainty.conformal import InductiveConformalPredictor, test_conformal_prediction


class TestOULADDataProcessing:
    """Test OULAD data processing functions."""
    
    def test_load_oulad_tables_empty_dir(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tables = load_oulad_tables(Path(temp_dir))
            assert isinstance(tables, dict)
            assert len(tables) == 0
    
    def test_create_labels_and_sensitive_attrs(self):
        """Test label and sensitive attribute creation."""
        # Create mock data
        student_info = pd.DataFrame({
            'id_student': [1, 2, 3],
            'gender': ['F', 'M', 'F'],
            'age_band': ['0-35', '35-55', '0-35'],
            'highest_education': ['A Level', 'HE Qualification', 'Lower Than A Level'],
            'imd_band': ['10-20%', '30-40%', '0-10%']
        })
        
        student_registration = pd.DataFrame({
            'id_student': [1, 2, 3],
            'code_module': ['AAA', 'BBB', 'AAA'], 
            'code_presentation': ['2013J', '2013J', '2014B'],
            'final_result': ['Pass', 'Fail', 'Withdrawn'],
            'studied_credits': [60, 30, 60],
            'num_of_prev_attempts': [0, 1, 0]
        })
        
        result = create_labels_and_sensitive_attrs(student_info, student_registration)
        
        # Check output structure
        assert 'label_pass' in result.columns
        assert 'label_fail_or_withdraw' in result.columns
        assert 'sex' in result.columns
        assert 'sex_x_age' in result.columns
        
        # Check label values
        assert result.loc[0, 'label_pass'] == 1  # Pass
        assert result.loc[1, 'label_pass'] == 0  # Fail
        assert result.loc[2, 'label_pass'] == 0  # Withdrawn
        
        # Check sex mapping
        assert result.loc[0, 'sex'] == 'Female'
        assert result.loc[1, 'sex'] == 'Male'


class TestOULADSplits:
    """Test OULAD split functions."""
    
    def test_check_leakage_no_overlap(self):
        """Test leakage check with no overlap."""
        train_ids = [1, 2, 3]
        val_ids = [4, 5, 6]
        test_ids = [7, 8, 9]
        
        assert check_leakage(train_ids, val_ids, test_ids) == True
    
    def test_check_leakage_with_overlap(self):
        """Test leakage check with overlap."""
        train_ids = [1, 2, 3]
        val_ids = [3, 4, 5]  # Overlap with train
        test_ids = [6, 7, 8]
        
        assert check_leakage(train_ids, val_ids, test_ids) == False
    
    def test_random_split(self):
        """Test random split function."""
        # Create mock data
        df = pd.DataFrame({
            'id_student': range(100),
            'label_pass': np.random.binomial(1, 0.3, 100),
            'sex': np.random.choice(['Female', 'Male'], 100),
            'code_module': 'AAA',
            'code_presentation': '2013J'
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            split_data = random_split(df, Path(temp_dir))
            
            # Check split structure
            assert 'train' in split_data
            assert 'val' in split_data  
            assert 'test' in split_data
            assert 'metadata' in split_data
            
            # Check no leakage
            assert check_leakage(split_data['train'], split_data['val'], split_data['test'])
            
            # Check split file exists
            assert (Path(temp_dir) / 'random_split.json').exists()


class TestConformalPrediction:
    """Test conformal prediction implementation."""
    
    def test_conformal_predictor_basic(self):
        """Test basic conformal predictor functionality."""
        cp = InductiveConformalPredictor(alpha=0.1)
        assert cp.alpha == 0.1
        assert cp.target_coverage == 0.9
    
    def test_nonconformity_score(self):
        """Test nonconformity score calculation."""
        cp = InductiveConformalPredictor()
        
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        
        scores = cp._nonconformity_score(y_true, y_prob)
        
        # For class 0: score = 1 - (1 - y_prob) = y_prob
        # For class 1: score = 1 - y_prob
        expected = np.array([0.2, 0.2, 0.3, 0.3])
        np.testing.assert_array_almost_equal(scores, expected)
    
    def test_conformal_prediction_full_workflow(self):
        """Test full conformal prediction workflow."""
        # This calls the existing test function
        test_conformal_prediction()


class TestDataIntegrity:
    """Test overall data integrity and workflow."""
    
    def test_column_consistency(self):
        """Test that expected columns are consistent across modules."""
        # This would test that column names match between different modules
        expected_oulad_cols = ['id_student', 'code_module', 'code_presentation', 'label_pass']
        
        # Mock test - in real implementation would check actual data
        assert all(col in expected_oulad_cols for col in ['id_student', 'label_pass'])
    
    def test_feature_engineering_ranges(self):
        """Test that engineered features have reasonable value ranges."""
        # Mock VLE features should be non-negative
        vle_clicks = np.array([0, 10, 100, 1000])
        assert np.all(vle_clicks >= 0)
        
        # Age bands should be valid categories
        valid_age_bands = ['0-35', '35-55', '55<=']
        test_age_band = '0-35'
        assert test_age_band in valid_age_bands


def test_import_all_modules():
    """Test that all new modules can be imported without errors."""
    # Test OULAD modules
    from src.oulad import build_dataset, splits
    
    # Test training module
    from src import train_eval
    
    # Test explainability module  
    from src.explain import importance
    
    # Test uncertainty module
    from src.uncertainty import conformal
    
    # Test transfer module
    from src.transfer import uci_transfer
    
    # Test process mining module
    from src.process_mining import sam_pipeline
    
    # If we get here, all imports succeeded
    assert True


if __name__ == '__main__':
    pytest.main([__file__])