from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class IrisClassifier:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(random_state=random_state, max_iter=200)
        self.is_trained = False

    def train(self, X_train, y_train):
        """Train the logistic regression model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        return accuracy, report

    def save_model(self, filepath='models/iris_classifier.pkl'):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load_model(self, filepath='models/iris_classifier.pkl'):
        """Load trained model from file"""
        self.model = joblib.load(filepath)
        self.is_trained = True
