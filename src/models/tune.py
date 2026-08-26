import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import GridSearchCV
from src.config import REPORTS_RESULTS_DIR, MODELS_DIR
from src.data.preprocessing import build_preprocessor
from src.models.evaluate import generate_confusion_matrices, log_recall_limitation_note
from src.visualization.plots import plot_feature_importance
from src.logger import get_logger

logger = get_logger(__name__)


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
    )
    logger.info("Starting GridSearchCV for Random Forest...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best CV F1: {grid_search.best_score_:.4f}")
    return grid_search


def evaluate_tuned_model(best_model, X_test, y_test):
    """Final classification report, confusion matrix, and metrics for the tuned model."""
    logger.info("Evaluating tuned model...")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": "Tuned Random Forest",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    logger.info(f"Tuned model metrics: {metrics}")

    report = classification_report(y_test, y_pred, digits=4)
    logger.info(f"Classification Report:\n{report}")

    generate_confusion_matrices({"Tuned Random Forest": best_model}, X_test, y_test)

    return metrics


def get_feature_importance(best_model) -> pd.DataFrame:
    """Read feature importances off the tuned Random Forest, sorted highest first."""
    classifier = best_model.named_steps["classifier"]
    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": classifier.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"Feature importances:\n{importance_df.to_string(index=False)}")
    return importance_df


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
    final_results_df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Saved final results to {filepath}")


def save_feature_importance(feature_importance_df):
    """Write reports/results/feature_importance.csv."""
    REPORTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_RESULTS_DIR / "feature_importance.csv"
    feature_importance_df.to_csv(filepath, index=False, encoding="utf-8")
    logger.info(f"Saved feature importance to {filepath}")


if __name__ == "__main__":
    logger.info("Running tuning module standalone...")
    from src.data.preprocessing import load_data, clean_data, get_features_and_target, split_data
    from src.models.train import build_models

    df = load_data()
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()
    rf_pipeline = build_models(preprocessor)["Random Forest"]

    grid_search = tune_random_forest(rf_pipeline, X_train, y_train)
    best_model = grid_search.best_estimator_

    metrics = evaluate_tuned_model(best_model, X_test, y_test)
    feature_importance_df = get_feature_importance(best_model)
    plot_feature_importance(feature_importance_df)

    save_best_model(best_model)
    save_final_results(pd.DataFrame([metrics]))
    save_feature_importance(feature_importance_df)
    log_recall_limitation_note()
    logger.info("Tuning standalone complete.")
