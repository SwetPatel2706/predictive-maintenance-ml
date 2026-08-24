import logging
import sys
import pandas as pd
from pathlib import Path

from src.config import PROJECT_ROOT
from src.logger import get_logger

logger = get_logger(__name__)


def run_pipeline():
    """Run the full ML pipeline: preprocessing → train → evaluate → tune → save results."""
    logger.info("=" * 60)
    logger.info("STARTING FULL PREDICTIVE MAINTENANCE PIPELINE")
    logger.info("=" * 60)

    # Step 1: Data preprocessing
    logger.info("\n--- Step 1: Data Preprocessing ---")
    from src.data.preprocessing import load_data, clean_data, get_features_and_target, split_data, build_preprocessor

    df = load_data()
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    logger.info("Preprocessing complete.\n")

    # Step 2: Train models
    logger.info("--- Step 2: Training Models ---")
    from src.models.train import build_models, train_models

    models = build_models()
    fitted_models = train_models(models, X_train, y_train)
    logger.info("Training complete.\n")

    # Step 3: Evaluate models
    logger.info("--- Step 3: Model Evaluation ---")
    from src.models.evaluate import evaluate_models, generate_classification_reports, generate_confusion_matrices, generate_roc_comparison, generate_model_comparison_plot, log_recall_limitation_note, save_results

    results_df = evaluate_models(fitted_models, X_test, y_test)
    generate_classification_reports(fitted_models, X_test, y_test)
    generate_confusion_matrices(fitted_models, X_test, y_test)
    generate_roc_comparison(fitted_models, X_test, y_test)
    generate_model_comparison_plot(results_df)
    log_recall_limitation_note()
    save_results(results_df)
    logger.info("Evaluation complete.\n")

    # Step 4: Tune Random Forest
    logger.info("--- Step 4: Random Forest Tuning ---")
    from src.models.tune import tune_random_forest, evaluate_tuned_model, save_best_model, save_final_results, save_feature_importance

    rf_pipeline = fitted_models["Random Forest"]
    grid_search = tune_random_forest(rf_pipeline, X_train, y_train)
    best_model = grid_search.best_estimator_
    save_best_model(best_model)
    metrics = evaluate_tuned_model(best_model, X_test, y_test)
    save_final_results(pd.DataFrame([metrics]))
    # Feature importance from best model
    try:
        importances = best_model.named_steps["classifier"].feature_importances_
        feature_names = list(best_model.named_steps["preprocessor"].get_feature_names_out())
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        save_feature_importance(fi_df)
    except Exception as e:
        logger.warning(f"Could not save feature importance: {e}")
    logger.info("Tuning complete.\n")

    # Step 5: Final summary
    logger.info("--- Step 5: Pipeline Summary ---")
    logger.info("Best baseline model metrics:")
    logger.info(results_df.to_string(index=False))
    logger.info("\nTuned Random Forest metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()