import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import REPORTS_FIGURES_DIR


def _save_and_return(fig, filename):
    """Save figure as PNG and return the figure object."""
    filepath = REPORTS_FIGURES_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    return fig


def plot_failure_distribution(df: pd.DataFrame, save=True, filename="failure_distribution.png"):
    """Plot distribution of Machine_Failure (0/1)."""
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Machine_Failure", ax=ax, palette="Set1")
    ax.set_title("Failure Distribution")
    ax.set_xlabel("Machine Failure (0=No, 1=Yes)")
    ax.set_ylabel("Count")
    return _save_and_return(fig, filename)


def plot_product_type_distribution(df: pd.DataFrame, save=True, filename="product_type_distribution.png"):
    """Plot distribution of Product Type."""
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Type", ax=ax, palette="Set2")
    ax.set_title("Product Type Distribution")
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")
    return _save_and_return(fig, filename)


def plot_numerical_distributions(df: pd.DataFrame, save=True, filename="numerical_distributions.png"):
    """Plot histograms for all numerical features."""
    numerical = ["Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear"]
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    axes = axes.ravel()
    for i, col in enumerate(numerical):
        if i < len(axes):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i], color="skyblue")
            axes[i].set_title(col)
    # Remove any empty subplots
    for i in range(len(numerical), len(axes)):
        fig.delaxes(axes[i])
    fig.suptitle("Numerical Feature Distributions", y=1.02)
    return _save_and_return(fig, filename)


def plot_correlation_heatmap(df: pd.DataFrame, save=True, filename="correlation_heatmap.png"):
    """Plot correlation heatmap for numerical features."""
    numerical = ["Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear"]
    corr = df[numerical].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap")
    return _save_and_return(fig, filename)


def plot_feature_vs_failure_boxplots(df: pd.DataFrame, save=True, filename="feature_vs_failure_boxplots.png"):
    """Plot boxplots of each numerical feature vs Machine_Failure."""
    numerical = ["Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear"]
    n = len(numerical)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.ravel()
    for i, col in enumerate(numerical):
        if i < len(axes):
            sns.boxplot(data=df, x="Machine_Failure", y=col, ax=axes[i], palette="Set1")
            axes[i].set_title(f"{col} vs Failure")
    for i in range(len(numerical), len(axes)):
        fig.delaxes(axes[i])
    fig.tight_layout()
    return _save_and_return(fig, filename)


def plot_failure_by_product_type(df: pd.DataFrame, save=True, filename="failure_by_product_type.png"):
    """Plot failure count by Product Type."""
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Type", hue="Machine_Failure", ax=ax, palette="Set1")
    ax.set_title("Failure by Product Type")
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")
    return _save_and_return(fig, filename)


def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save=True, filename="confusion_matrix.png"):
    """Plot confusion matrix."""
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    return _save_and_return(fig, filename)


def plot_roc_curves(models_dict, X_test, y_test, save=True, filename="roc_curves.png"):
    """Plot ROC curves for multiple models."""
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots()
    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison")
    ax.legend(loc="lower right")
    return _save_and_return(fig, filename)


def plot_model_comparison(results_df, save=True, filename="model_comparison.png"):
    """Plot model comparison bar chart for metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(results_df))
    width = 0.2
    for i, metric in enumerate(metrics):
        values = results_df[metric]
        ax.bar(x + i * width, values, width=width, label=metric)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.legend(metrics)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"], rotation=15)
    return _save_and_return(fig, filename)


def plot_feature_importance(feature_importance_df, save=True, filename="feature_importance.png"):
    """Plot feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x="importance", y="feature", ax=ax, palette="viridis")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    return _save_and_return(fig, filename)