from pathlib import Path

# Project root is the parent of the src/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path constants (all relative to PROJECT_ROOT, using pathlib for cross-OS compatibility)
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
REPORTS_LOGS_DIR = PROJECT_ROOT / "reports" / "logs"

# Reproducibility
RANDOM_STATE = 42

# Column rename map (matches original notebook's renames)
COLUMN_RENAME_MAP = {
    "Type": "Type",
    "Air temperature [K]": "Air_Temperature",
    "Process temperature [K]": "Process_Temperature",
    "Rotational speed [rpm]": "Rotational_Speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool_Wear",
    "Machine failure": "Machine_Failure",
}

# Feature lists
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

FEATURES = ["Type", "Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear"]
TARGET = "Machine_Failure"

CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES = ["Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear"]