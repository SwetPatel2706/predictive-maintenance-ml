import pandas as pd

from src.config import REPORTS_RESULTS_DIR
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_model_comparison,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Column names shared by both model_comparison_results.csv (baseline models)
# and final_model_results.csv (tuned model) so the two are directly comparable.
METRIC_COLUMNS = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def evaluate_models(models_dict, X_test, y_test):
    """Compute Accuracy/Precision/Recall/F1/ROC-AUC per model, return results_df."""
    results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
            }
        )
    results_df = pd.DataFrame(results, columns=METRIC_COLUMNS)
    logger.info(f"Evaluation results:\n{results_df}")
    return results_df


def generate_classification_reports(models_dict, X_test, y_test):
    """Log full classification report per model."""
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, digits=4)
        logger.info(f"\n{name} Classification Report:\n{report}")


def generate_confusion_matrices(models_dict, X_test, y_test):
    """Call plots.plot_confusion_matrix per model.

    Each model gets its own filename (derived from its name) so evaluating
    multiple models doesn't overwrite the same confusion_matrix.png.
    """
    labels = ["No Failure", "Failure"]
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            cm,
            labels,
            title=f"{name} Confusion Matrix",
            filename=f"confusion_matrix_{safe_name}.png",
        )


def generate_roc_comparison(models_dict, X_test, y_test):
    """Call plots.plot_roc_curves."""
    plot_roc_curves(models_dict, X_test, y_test)


def generate_model_comparison_plot(results_df):
    """Call plots.plot_model_comparison."""
    plot_model_comparison(results_df)


def log_recall_limitation_note():
    """Log the explicit note about class imbalance / recall trade-off."""
    note = (
        "\n[INFO] CLASS IMBALANCE NOTE: The dataset is ~96.6% no-failure / ~3.4% failure. "
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
    results_df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Saved results to {filepath}")


if __name__ == "__main__":
    logger.info("Running evaluation module standalone...")
    from src.data.preprocessing import (
        load_data,
        clean_data,
        get_features_and_target,
        split_data,
        build_preprocessor,
    )
    from src.models.train import build_models

    df = load_data()
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    models = build_models(preprocessor)
    fitted = _train_models(models, X_train, y_train)
    results = evaluate_models(fitted, X_test, y_test)
    generate_classification_reports(fitted, X_test, y_test)
    generate_confusion_matrices(fitted, X_test, y_test)
    generate_roc_comparison(fitted, X_test, y_test)
    generate_model_comparison_plot(results)
    save_results(results)
    log_recall_limitation_note()
    logger.info("Evaluation standalone complete.")