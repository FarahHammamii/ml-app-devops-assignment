#!/usr/bin/env python3
"""
Training script for Iris classifier
"""

import sys
import os

from data_loader import load_iris_data, get_feature_names
from model import IrisClassifier
from utils import plot_confusion_matrix, plot_feature_importance

def main():
    print("Starting Iris Classifier Training...")

    # Load data
    print("Loading Iris dataset...")
    X_train, X_test, y_train, y_test = load_iris_data()
    feature_names = get_feature_names()

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model
    print("Training Logistic Regression model...")
    classifier = IrisClassifier()
    classifier.train(X_train, y_train)

    # Evaluate model
    print("Evaluating model...")
    accuracy, report = classifier.evaluate(X_test, y_test)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    # Save model
    print("Saving model...")
    classifier.save_model('models/iris_classifier.pkl')

    # Generate plots
    print("Generating evaluation plots...")
    y_pred = classifier.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(classifier.model, feature_names)

    print("Training completed successfully!")
    print("Model saved to: models/iris_classifier.pkl")
    print("Plots saved: confusion_matrix.png, feature_importance.png")

if __name__ == "__main__":
    main()
