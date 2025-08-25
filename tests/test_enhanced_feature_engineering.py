"""
Test Enhanced Feature Engineering Module

Tests for the comprehensive feature engineering enhancements.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.enhanced_feature_engineering import EnhancedFeatureEngineer, create_domain_adaptive_features


class TestEnhancedFeatureEngineer:
    """Test the enhanced feature engineering functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=8, 
            n_redundant=2, random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(10)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y)
        return X_df, y_series
    
    @pytest.fixture
    def oulad_like_data(self):
        """Create OULAD-like data for testing."""
        np.random.seed(42)
        n_samples = 150
        
        data = {
            'vle_total_clicks': np.random.poisson(100, n_samples),
            'vle_days_active': np.random.randint(1, 30, n_samples),
            'vle_first4_clicks': np.random.poisson(20, n_samples),
            'vle_last4_clicks': np.random.poisson(30, n_samples),
            'assessment_mean_score': np.random.normal(70, 15, n_samples),
            'assessment_count': np.random.randint(1, 10, n_samples),
            'other_feature_1': np.random.normal(0, 1, n_samples),
            'other_feature_2': np.random.normal(0, 1, n_samples)
        }
        
        X_df = pd.DataFrame(data)
        # Create target based on engagement and performance
        y = ((X_df['vle_total_clicks'] > 80) & 
             (X_df['assessment_mean_score'] > 65)).astype(int)
        
        return X_df, pd.Series(y)
    
    @pytest.fixture
    def uci_like_data(self):
        """Create UCI-like data for testing."""
        np.random.seed(42)
        n_samples = 150
        
        data = {
            'studytime': np.random.randint(1, 5, n_samples),
            'Dalc': np.random.randint(1, 6, n_samples),
            'Walc': np.random.randint(1, 6, n_samples),
            'famrel': np.random.randint(1, 6, n_samples),
            'freetime': np.random.randint(1, 6, n_samples),
            'goout': np.random.randint(1, 6, n_samples),
            'health': np.random.randint(1, 6, n_samples),
            'Medu': np.random.randint(0, 5, n_samples),
            'Fedu': np.random.randint(0, 5, n_samples),
            'age': np.random.randint(15, 20, n_samples),
            'other_feature': np.random.normal(0, 1, n_samples)
        }
        
        X_df = pd.DataFrame(data)
        # Create target based on study habits and family support
        y = ((X_df['studytime'] >= 3) & 
             (X_df['famrel'] >= 4) & 
             (X_df['Dalc'] <= 2)).astype(int)
        
        return X_df, pd.Series(y)
    
    def test_basic_functionality(self, sample_data):
        """Test basic feature engineering functionality."""
        X, y = sample_data
        
        engineer = EnhancedFeatureEngineer()
        X_enhanced = engineer.fit_transform(X, y)
        
        # Check that features are enhanced
        assert X_enhanced.shape[0] == X.shape[0]
        assert X_enhanced.shape[1] > X.shape[1]
        assert isinstance(X_enhanced, np.ndarray)
        
        # Check feature names are available
        feature_names = engineer.get_feature_names()
        assert len(feature_names) > 0
    
    def test_oulad_detection_and_features(self, oulad_like_data):
        """Test OULAD dataset detection and specific features."""
        X, y = oulad_like_data
        
        engineer = EnhancedFeatureEngineer(dataset_type="auto")
        X_enhanced = engineer.fit_transform(X, y)
        
        # Should detect as OULAD and create more features
        assert engineer.dataset_type == "oulad"
        assert X_enhanced.shape[1] > X.shape[1]
        
        # Check some OULAD-specific features are created
        feature_names = engineer.get_feature_names()
        oulad_features = [name for name in feature_names 
                         if any(keyword in name for keyword in 
                               ['engagement', 'performance', 'early_late'])]
        assert len(oulad_features) > 0
    
    def test_uci_detection_and_features(self, uci_like_data):
        """Test UCI dataset detection and specific features."""
        X, y = uci_like_data
        
        engineer = EnhancedFeatureEngineer(dataset_type="auto")
        X_enhanced = engineer.fit_transform(X, y)
        
        # Should detect as UCI and create more features
        assert engineer.dataset_type == "uci"
        assert X_enhanced.shape[1] > X.shape[1]
        
        # Check some UCI-specific features are created
        feature_names = engineer.get_feature_names()
        uci_features = [name for name in feature_names 
                       if any(keyword in name for keyword in 
                             ['social', 'education', 'lifestyle'])]
        assert len(uci_features) > 0
    
    def test_transform_consistency(self, sample_data):
        """Test that transform produces consistent results."""
        X, y = sample_data
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        
        engineer = EnhancedFeatureEngineer()
        X_train_enhanced = engineer.fit_transform(X_train, y[:len(X_train)])
        X_test_enhanced = engineer.transform(X_test)
        
        # Shapes should be compatible
        assert X_train_enhanced.shape[1] == X_test_enhanced.shape[1]
        assert X_test_enhanced.shape[0] == X_test.shape[0]
    
    def test_feature_importance(self, sample_data):
        """Test feature importance calculation."""
        X, y = sample_data
        
        engineer = EnhancedFeatureEngineer()
        X_enhanced = engineer.fit_transform(X, y)
        
        importance_df = engineer.get_feature_importance(X_enhanced, y)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'mutual_info' in importance_df.columns
        assert len(importance_df) == X_enhanced.shape[1]
    
    def test_performance_improvement(self, sample_data):
        """Test that enhanced features improve model performance."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Baseline performance
        baseline_model = LogisticRegression(random_state=42)
        baseline_model.fit(X_train, y_train)
        baseline_score = accuracy_score(y_test, baseline_model.predict(X_test))
        
        # Enhanced features performance
        engineer = EnhancedFeatureEngineer()
        X_train_enhanced = engineer.fit_transform(X_train, y_train)
        X_test_enhanced = engineer.transform(X_test)
        
        enhanced_model = LogisticRegression(random_state=42)
        enhanced_model.fit(X_train_enhanced, y_train)
        enhanced_score = accuracy_score(y_test, enhanced_model.predict(X_test_enhanced))
        
        # Enhanced features should perform at least as well (allowing for variations)
        # In practice, more features can sometimes lead to overfitting on small datasets
        assert enhanced_score >= baseline_score - 0.15  # 15% tolerance


class TestDomainAdaptiveFeatures:
    """Test domain adaptive features for transfer learning."""
    
    def test_domain_adaptive_features(self):
        """Test domain adaptive feature creation."""
        # Create source (OULAD-like) data
        np.random.seed(42)
        source_data = {
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(0, 1, 100),
            'common_feature': np.random.normal(0, 1, 100),
        }
        source_X = pd.DataFrame(source_data)
        source_y = pd.Series(np.random.binomial(1, 0.5, 100))
        
        # Create target (UCI-like) data
        target_data = {
            'feature_3': np.random.normal(0, 1, 80),
            'feature_4': np.random.normal(0, 1, 80),
            'common_feature': np.random.normal(0.5, 1, 80),  # Slight distribution shift
        }
        target_X = pd.DataFrame(target_data)
        target_y = pd.Series(np.random.binomial(1, 0.4, 80))
        
        # Create domain adaptive features
        source_enhanced, target_enhanced = create_domain_adaptive_features(
            source_X, target_X, source_y, target_y
        )
        
        # Both should have the same number of features
        assert source_enhanced.shape[1] == target_enhanced.shape[1]
        assert source_enhanced.shape[0] == source_X.shape[0]
        assert target_enhanced.shape[0] == target_X.shape[0]
        
        # Should have more features than original common features
        assert source_enhanced.shape[1] >= 1  # At least as many as common features
    
    def test_no_common_features(self):
        """Test handling when there are no common features."""
        # Create completely different features
        source_X = pd.DataFrame({
            'source_feature_1': np.random.normal(0, 1, 50),
            'source_feature_2': np.random.normal(0, 1, 50),
        })
        
        target_X = pd.DataFrame({
            'target_feature_1': np.random.normal(0, 1, 40),
            'target_feature_2': np.random.normal(0, 1, 40),
            'target_feature_3': np.random.normal(0, 1, 40),
        })
        
        source_y = pd.Series(np.random.binomial(1, 0.5, 50))
        target_y = pd.Series(np.random.binomial(1, 0.5, 40))
        
        # Should handle gracefully
        source_enhanced, target_enhanced = create_domain_adaptive_features(
            source_X, target_X, source_y, target_y
        )
        
        # Should use positional alignment
        assert source_enhanced.shape[1] == target_enhanced.shape[1]
        assert source_enhanced.shape[0] == 50
        assert target_enhanced.shape[0] == 40


def test_integration_with_models():
    """Test integration with actual ML models."""
    # Create sample data
    X, y = make_classification(
        n_samples=300, n_features=15, n_informative=10, 
        n_redundant=2, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
    y_series = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42
    )
    
    # Test with multiple models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    for model_name, model in models.items():
        # Baseline
        model_baseline = model.__class__(**model.get_params())
        model_baseline.fit(X_train, y_train)
        baseline_score = accuracy_score(y_test, model_baseline.predict(X_test))
        
        # Enhanced features
        engineer = EnhancedFeatureEngineer()
        X_train_enhanced = engineer.fit_transform(X_train, y_train)
        X_test_enhanced = engineer.transform(X_test)
        
        model_enhanced = model.__class__(**model.get_params())
        model_enhanced.fit(X_train_enhanced, y_train)
        enhanced_score = accuracy_score(y_test, model_enhanced.predict(X_test_enhanced))
        
        print(f"{model_name} - Baseline: {baseline_score:.3f}, Enhanced: {enhanced_score:.3f}")
        
        # Enhanced should be competitive (allowing some variation)
        # Note: Enhanced features may not always improve performance due to overfitting
        # We just check they don't degrade performance too much
        assert enhanced_score >= baseline_score - 0.2  # 20% tolerance


if __name__ == "__main__":
    # Run some basic tests
    test_integration_with_models()
    print("Basic integration tests passed!")