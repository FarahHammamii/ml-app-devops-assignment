import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, target_names=None):
    """Plot confusion matrix"""
    if target_names is None:
        iris = load_iris()
        target_names = iris.target_names

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def plot_feature_importance(model, feature_names):
    """Plot feature importance for logistic regression"""
    importance = model.coef_[0]
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(importance)
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_imp['feature'], feature_imp['importance'])
    plt.title('Feature Importance (Logistic Regression)')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
