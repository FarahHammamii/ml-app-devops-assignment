#!/usr/bin/env python3
"""
Prediction script for Iris classifier
"""

import sys
import os
import numpy as np

from model import IrisClassifier
from data_loader import get_target_names

def main():
    print("Iris Classifier Prediction")

    # Load model
    try:
        classifier = IrisClassifier()
        classifier.load_model('models/iris_classifier.pkl')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model not found. Please run train.py first.")
        return

    # Get target names
    target_names = get_target_names()

    # Example predictions
    print("\n Example Predictions:")
    print("Features: [sepal length, sepal width, petal length, petal width]")

    # Example data for prediction
    examples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.7, 3.0, 5.2, 2.3],  # Virginica
        [5.9, 3.0, 4.2, 1.5],  # Versicolor
    ]

    for i, features in enumerate(examples, 1):
        prediction = classifier.predict([features])[0]
        probability = classifier.model.predict_proba([features])[0]

        print(f"\nExample {i}: {features}")
        print(f"Prediction: {target_names[prediction]}")
        print("Probabilities:")
        for j, prob in enumerate(probability):
            print(f"  {target_names[j]}: {prob:.4f}")

if __name__ == "__main__":
    main()
