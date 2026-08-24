import logging
import pandas as pd
import numpy as np

from src.config import REPORTS_RESULTS_DIR, REPORTS_FIGURES_DIR
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)


def evaluate_models(models_dict, X_test, y_test):
    """Compute Accuracy/Precision/Recall/F1/ROC-AUC per model, return results_df."""
    results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = _precision(y_test, y_pred)
        rec = _recall(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
                "ROC-AUC": roc_auc,
            }
        )
    results_df = pd.DataFrame(results)
    logger.info(f"Evaluation results:\n{results_df}")
    return results_df


def _precision(y_true, y_pred, zero_division=0):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def _recall(y_true, y_pred, zero_division=0):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def generate_classification_reports(models_dict, X_test, y_test):
    """Log full classification report per model."""
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=4)
        logger.info(f"\n{name} Classification Report:\n{report}")


def generate_confusion_matrices(models_dict, X_test, y_test):
    """Call plots.plot_confusion_matrix per model."""
    labels = ["No Failure", "Failure"]
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, labels, title=f"{name} Confusion Matrix")


def generate_roc_comparison(models_dict, X_test, y_test):
    """Call plots.plot_roc_curves."""
    plot_roc_curves(models_dict, X_test, y_test)


def generate_model_comparison_plot(results_df):
    """Call plots.plot_model_comparison."""
    plot_model_comparison(results_df)


def log_recall_limitation_note():
    """Log the explicit note about class imbalance / recall trade-off."""
    note = (
        "\n[INFO] CLASS IMPBALANCE NOTE: The dataset is ~96.6% no-failure / ~3.4% failure. "
        "Recall is the weaker metric due to class imbalance. Threshold tuning, SMOTE, "
        "or cost-sensitive learning are valid future improvements (mentioned only, not implemented)."
    )
    logger.info(note)


def _train_models(models_dict, X_train, y_train):
    """Fit each model and return fitted dict."""
    fitted = {}
    for name, model in models_dict.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        logger.info(f"{name} trained.")
    return fitted


def save_results(results_df):
    """Write reports/results/model_comparison_results.csv."""
    REPORTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_RESULTS_DIR / "model_comparison_results.csv"
    results_df.to_csv(filepath, index=False)
    logger.info(f"Saved results to {filepath}")


if __name__ == "__main__":
    logger.info("Running evaluation module standalone...")
    from src.data.preprocessing import load_data, clean_data, get_features_and_target, split_data
    from src.models.train import build_models

    df = load_data()
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = build_models()
    fitted = _train_models(models, X_train, y_train)
    results = evaluate_models(fitted, X_test, y_test)
    save_results(results)
    log_recall_limitation_note()
    logger.info("Evaluation standalone complete.")


def _train_models(models_dict, X_train, y_train):
    """Fit each model and return fitted dict."""
    fitted = {}
    for name, model in models_dict.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        logger.info(f"{name} trained.")
    return fitted