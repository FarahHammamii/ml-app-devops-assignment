import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from src.data_loader import load_iris_data
from src.model import IrisClassifier

class TestIrisClassifier:
    def setup_method(self):
        """Setup method that runs before each test"""
        self.X_train, self.X_test, self.y_train, self.y_test = load_iris_data(test_size=0.3, random_state=42)
        self.classifier = IrisClassifier()

    def test_model_initialization(self):
        """Test that model initializes correctly"""
        assert not self.classifier.is_trained
        assert self.classifier.model is not None

    def test_model_training(self):
        """Test model training functionality"""
        self.classifier.train(self.X_train, self.y_train)
        assert self.classifier.is_trained

    def test_model_prediction(self):
        """Test model prediction functionality"""
        self.classifier.train(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test[:5])
        assert len(predictions) == 5
        assert all(isinstance(pred, (np.int32, np.int64, int)) for pred in predictions)

    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        self.classifier.train(self.X_train, self.y_train)
        accuracy, report = self.classifier.evaluate(self.X_test, self.y_test)

        assert 0 <= accuracy <= 1
        assert isinstance(report, str)
        assert "precision" in report.lower()

    def test_model_save_load(self, tmp_path):
        """Test model saving and loading"""
        self.classifier.train(self.X_train, self.y_train)

        # Save model
        save_path = tmp_path / "test_model.pkl"
        self.classifier.save_model(str(save_path))
        assert save_path.exists()

        # Load model
        new_classifier = IrisClassifier()
        new_classifier.load_model(str(save_path))
        assert new_classifier.is_trained

        # Verify predictions match
        original_pred = self.classifier.predict(self.X_test[:5])
        loaded_pred = new_classifier.predict(self.X_test[:5])
        assert np.array_equal(original_pred, loaded_pred)

def test_data_loading():
    """Test data loading functionality"""
    X_train, X_test, y_train, y_test = load_iris_data()

    assert X_train.shape[1] == 4  # 4 features
    assert len(np.unique(y_train)) == 3  # 3 classes
    assert len(X_train) + len(X_test) == 150  # Total samples
