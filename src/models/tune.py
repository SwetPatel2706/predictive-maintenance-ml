import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from src.config import RANDOM_STATE, REPORTS_RESULTS_DIR, REPORTS_FIGURES_DIR, MODELS_DIR
from src.data.preprocessing import build_preprocessor
from src.models.evaluate import _train_models, log_recall_limitation_note, save_results
from src.visualization.plots import plot_feature_importance

logger = logging.getLogger(__name__)


param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}


def tune_random_forest(rf_pipeline, X_train, y_train):
    """GridSearchCV on rf_pipeline, cv=5, scoring='f1', n_jobs=-1, log best params + best CV F1."""
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    logger.info("Starting GridSearchCV for Random Forest...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best CV F1: {grid_search.best_score_:.4f}")
    return grid_search


def evaluate_tuned_model(best_model, X_test, y_test):
    """Final classification report, confusion matrix, final metrics."""
    from src.models.evaluate import evaluate_models, generate_classification_reports, generate_confusion_matrices

    logger.info("Evaluating tuned model...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = _precision(y_test, y_pred)
    rec = _recall(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"Tuned Model Metrics - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, ROC-AUC: {roc_auc:.4f}")

    report = classification_report(y_test, y_pred, digits=4)
    logger.info(f"Classification Report:\n{report}")

    labels = ["No Failure", "Failure"]
    cm = confusion_matrix(y_test, y_pred)
    generate_confusion_matrices({"Tuned RF": best_model}, X_test, y_test)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0, "roc_auc": roc_auc}


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


def save_best_model(best_model):
    """joblib.dump to models/best_model.joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, filepath)
    logger.info(f"Saved best model to {filepath}")


def save_final_results(final_results_df):
    """Write reports/results/final_model_results.csv."""
    REPORTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_RESULTS_DIR / "final_model_results.csv"
    final_results_df.to_csv(filepath, index=False)
    logger.info(f"Saved final results to {filepath}")


def save_feature_importance(feature_importance_df):
    """Write reports/results/feature_importance.csv."""
    REPORTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_RESULTS_DIR / "feature_importance.csv"
    feature_importance_df.to_csv(filepath, index=False)
    logger.info(f"Saved feature importance to {filepath}")


if __name__ == "__main__":
    logger.info("Running tuning module standalone...")
    from src.data.preprocessing import load_data, clean_data, get_features_and_target, split_data
    from src.models.train import build_models

    df = load_data()
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = build_models()
    rf_pipeline = models["Random Forest"]
    grid_search = tune_random_forest(rf_pipeline, X_train, y_train)
    best_model = grid_search.best_estimator_
    save_best_model(best_model)
    metrics = evaluate_tuned_model(best_model, X_test, y_test)
    save_final_results(pd.DataFrame([metrics]))
    save_feature_importance(pd.DataFrame({"feature": [], "importance": []}))
    log_recall_limitation_note()
    logger.info("Tuning standalone complete.")