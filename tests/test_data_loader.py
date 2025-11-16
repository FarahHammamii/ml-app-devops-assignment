"""
Unit tests for data_loader.py
Tests data loading and preprocessing functionality
"""

import pytest
import numpy as np
import sys
import os

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_iris_data, get_feature_names, get_target_names


class TestDataLoader:
    """Test suite for data loading functionality"""

    def test_load_iris_data_returns_correct_shapes(self):
        """Test that load_iris_data returns correct data shapes"""
        X_train, X_test, y_train, y_test = load_iris_data()
        
        # Check that data is returned
        assert X_train is not None, "X_train should not be None"
        assert X_test is not None, "X_test should not be None"
        assert y_train is not None, "y_train should not be None"
        assert y_test is not None, "y_test should not be None"
        
        # Check shapes
        assert X_train.shape[1] == 4, "Should have 4 features"
        assert X_test.shape[1] == 4, "Should have 4 features"
        
        # Check that train and test sets have matching samples
        assert X_train.shape[0] == y_train.shape[0], "X_train and y_train should have same number of samples"
        assert X_test.shape[0] == y_test.shape[0], "X_test and y_test should have same number of samples"
        
        # Check total samples (Iris dataset has 150 samples)
        total_samples = X_train.shape[0] + X_test.shape[0]
        assert total_samples == 150, "Total samples should be 150"

    def test_load_iris_data_types(self):
        """Test that load_iris_data returns correct data types"""
        X_train, X_test, y_train, y_test = load_iris_data()
        
        # Check types
        assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
        assert isinstance(X_test, np.ndarray), "X_test should be numpy array"
        assert isinstance(y_train, np.ndarray), "y_train should be numpy array"
        assert isinstance(y_test, np.ndarray), "y_test should be numpy array"
        
        # Check data types
        assert X_train.dtype in [np.float32, np.float64], "Features should be float"
        assert y_train.dtype in [np.int32, np.int64], "Labels should be integers"

    def test_get_feature_names(self):
        """Test that get_feature_names returns correct feature names"""
        feature_names = get_feature_names()
        
        # Check that we get a list
        assert isinstance(feature_names, list), "Should return a list"
        
        # Iris dataset has 4 features
        assert len(feature_names) == 4, "Should have 4 feature names"
        
        # Check that feature names are strings
        for name in feature_names:
            assert isinstance(name, str), "Feature names should be strings"
            assert len(name) > 0, "Feature names should not be empty"

    def test_get_target_names(self):
        """Test that get_target_names returns correct class names"""
        target_names = get_target_names()
        
        # Check that we get a list or array
        assert target_names is not None, "Should return target names"
        
        # Iris dataset has 3 classes
        assert len(target_names) == 3, "Should have 3 target names"
        
        # Check expected class names
        expected_names = ['setosa', 'versicolor', 'virginica']
        for expected in expected_names:
            assert expected in target_names, f"Should contain {expected}"

    def test_data_ranges(self):
        """Test that loaded data has reasonable value ranges"""
        X_train, X_test, y_train, y_test = load_iris_data()
        
        # Check feature ranges (Iris features are in cm, typically 0-10)
        assert X_train.min() >= 0, "Features should be non-negative"
        assert X_train.max() <= 10, "Features should be reasonable (< 10cm)"
        
        # Check label ranges (should be 0, 1, or 2)
        assert y_train.min() >= 0, "Labels should be >= 0"
        assert y_train.max() <= 2, "Labels should be <= 2"
        assert len(np.unique(y_train)) == 3, "Should have 3 classes in training data"