from sklearn.base import clone
from src.config import RANDOM_STATE
from src.data.preprocessing import build_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.logger import get_logger

logger = get_logger(__name__)


def build_models(preprocessor=None):
    """Return a dict of 3 unfitted Pipelines: Logistic Regression, Decision Tree, Random Forest.

    Each pipeline gets its own clone of `preprocessor` (or a freshly built one
    if none is passed) so the three models never share mutable fitted state.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )

    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=1000,
            )),
        ]
    )

    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_depth=5,
            )),
        ]
    )

    models = {
        "Logistic Regression": lr_pipeline,
        "Decision Tree": dt_pipeline,
        "Random Forest": rf_pipeline,
    }
    logger.info(f"Built {len(models)} model pipelines: {list(models.keys())}")
    return models


def train_models(models_dict, X_train, y_train):
    """Fit each model in the dict, log progress, return fitted dict."""
    fitted = {}
    for name, model in models_dict.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        logger.info(f"{name} trained. Training set shape: {X_train.shape}")
    return fitted


if __name__ == "__main__":
    logger.info("Running training module standalone...")
    from src.data.preprocessing import load_data, clean_data, get_features_and_target, split_data

    df = load_data()
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    models = build_models(preprocessor)
    fitted = train_models(models, X_train, y_train)
    logger.info("Training standalone complete.")