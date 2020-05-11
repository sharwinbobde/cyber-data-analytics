import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importances(importances: np.array, features: list):
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [features[i] for i in indices]

    # Create plot
    plt.figure()

    # Create plot title
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(len(features)), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(len(features)), names, rotation=90)

    # Show plot
    plt.show()


def plot_roc_auc_curve(fpr, tpr, auc):
    plt.title("Receiver Operating Characteristic")
    for fpr, tpr, auc in zip(fpr, tpr, auc):
        plt.plot(fpr, tpr, label="AUC = %0.2f" % auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()
