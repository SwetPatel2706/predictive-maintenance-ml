import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def run_pipeline():
    """Run the full ML pipeline: preprocessing -> EDA plots -> train -> evaluate -> tune -> save results."""
    logger.info("=" * 60)
    logger.info("STARTING FULL PREDICTIVE MAINTENANCE PIPELINE")
    logger.info("=" * 60)

    # Step 1: Data preprocessing
    logger.info("--- Step 1: Data Preprocessing ---")
    from src.data.preprocessing import load_data, clean_data, get_features_and_target, split_data, build_preprocessor

    df = load_data()
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    logger.info("Preprocessing complete.")

    # Step 2: EDA plots (on the full cleaned dataset)
    logger.info("--- Step 2: EDA Visualizations ---")
    from src.visualization import plots

    plots.plot_failure_distribution(df)
    plots.plot_product_type_distribution(df)
    plots.plot_numerical_distributions(df)
    plots.plot_correlation_heatmap(df)
    plots.plot_feature_vs_failure_boxplots(df)
    plots.plot_failure_by_product_type(df)
    logger.info("EDA visualizations complete.")

    # Step 3: Train baseline models
    logger.info("--- Step 3: Training Models ---")
    from src.models.train import build_models, train_models

    models = build_models(preprocessor)
    fitted_models = train_models(models, X_train, y_train)
    logger.info("Training complete.")

    # Step 4: Evaluate baseline models
    logger.info("--- Step 4: Model Evaluation ---")
    from src.models import evaluate

    results_df = evaluate.evaluate_models(fitted_models, X_test, y_test)
    evaluate.generate_classification_reports(fitted_models, X_test, y_test)
    evaluate.generate_confusion_matrices(fitted_models, X_test, y_test)
    evaluate.generate_roc_comparison(fitted_models, X_test, y_test)
    evaluate.generate_model_comparison_plot(results_df)
    evaluate.log_recall_limitation_note()
    evaluate.save_results(results_df)
    logger.info("Evaluation complete.")

    # Step 5: Tune Random Forest (GridSearchCV)
    logger.info("--- Step 5: Random Forest Tuning ---")
    from src.models import tune

    grid_search = tune.tune_random_forest(models["Random Forest"], X_train, y_train)
    best_model = grid_search.best_estimator_

    final_metrics = tune.evaluate_tuned_model(best_model, X_test, y_test)
    feature_importance_df = tune.get_feature_importance(best_model)
    plots.plot_feature_importance(feature_importance_df)

    tune.save_best_model(best_model)
    tune.save_final_results(pd.DataFrame([final_metrics]))
    tune.save_feature_importance(feature_importance_df)
    logger.info("Tuning complete.")

    # Step 6: Final summary
    logger.info("--- Step 6: Pipeline Summary ---")
    best_baseline = results_df.loc[results_df["F1"].idxmax()]
    logger.info(f"Best baseline model:\n{best_baseline.to_string()}")
    logger.info(f"Final tuned model:\n{pd.Series(final_metrics).to_string()}")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
