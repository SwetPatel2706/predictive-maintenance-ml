import pandas as pd
import numpy as np

from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    COLUMN_RENAME_MAP,
    LEAKAGE_COLUMNS,
    FEATURES,
    TARGET,
    RANDOM_STATE,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.logger import get_logger

logger = get_logger(__name__)


def load_data() -> pd.DataFrame:
    """Read CSV from config.DATA_RAW_DIR, log shape, dtypes, missing values, duplicates."""
    csv_path = DATA_RAW_DIR / "ai4i2020.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Dtypes:\n{df.dtypes}")
    missing = df.isnull().sum().sum()
    logger.info(f"Missing values count: {missing}")
    duplicates = df.duplicated().sum()
    logger.info(f"Duplicate rows: {duplicates}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns per COLUMN_RENAME_MAP, drop leakage columns from modeling use."""
    df = df.rename(columns=COLUMN_RENAME_MAP)
    logger.info(f"After renaming columns: {list(df.columns)}")
    # leakage columns are kept in raw df for EDA reference but excluded from X
    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            logger.info(f"Leakage column '{col}' present - will be excluded from features")
    return df


def get_features_and_target(df: pd.DataFrame):
    """Return X, y using config.FEATURES / config.TARGET."""
    X = df[FEATURES]
    y = df[TARGET]
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"y distribution:\n{y.value_counts()}")
    return X, y


def split_data(X, y, test_size=0.2, random_state=None):
    """Stratified train/test split, log resulting shapes."""
    from sklearn.model_selection import train_test_split

    if random_state is None:
        random_state = RANDOM_STATE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """Return ColumnTransformer: StandardScaler on numerical, OneHotEncoder on categorical."""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    logger.info(f"Built preprocessor with numerical={NUMERICAL_FEATURES}, categorical={CATEGORICAL_FEATURES}")
    return preprocessor


if __name__ == "__main__":
    logger.info("Running preprocessing standalone...")
    df = load_data()
    df = clean_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    logger.info("Preprocessing standalone complete.")