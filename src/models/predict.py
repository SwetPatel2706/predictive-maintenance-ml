"""
Predict Machine_Failure for one new machine reading using the tuned
Random Forest saved at models/best_model.joblib.

Standalone usage (from the project root, after running the pipeline once
so best_model.joblib exists):
    python -m src.models.predict
"""

import joblib
import pandas as pd

from src.config import MODELS_DIR, FEATURES

MODEL_PATH = MODELS_DIR / "best_model.joblib"


def load_model():
    """Load the tuned model pipeline saved by src/models/tune.py."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python -m src.run_pipeline` first."
        )
    return joblib.load(MODEL_PATH)


def predict_failure(model, reading: dict) -> dict:
    """Predict Machine_Failure (0/1) for one machine reading.

    `reading` must have the same keys as config.FEATURES:
    Type, Air_Temperature, Process_Temperature, Rotational_Speed, Torque, Tool_Wear
    """
    X = pd.DataFrame([reading], columns=FEATURES)
    prediction = int(model.predict(X)[0])
    failure_probability = float(model.predict_proba(X)[0][1])
    return {"prediction": prediction, "failure_probability": failure_probability}


def _prompt_for_reading() -> dict:
    """Ask the user for one machine reading via the terminal."""
    print("Enter the machine's current readings:")
    return {
        "Type": input("  Product type (L / M / H): ").strip().upper(),
        "Air_Temperature": float(input("  Air temperature in K, e.g. 298.1: ")),
        "Process_Temperature": float(input("  Process temperature in K, e.g. 308.6: ")),
        "Rotational_Speed": float(input("  Rotational speed in rpm, e.g. 1551: ")),
        "Torque": float(input("  Torque in Nm, e.g. 42.8: ")),
        "Tool_Wear": float(input("  Tool wear in minutes, e.g. 0: ")),
    }


if __name__ == "__main__":
    model = load_model()
    reading = _prompt_for_reading()
    result = predict_failure(model, reading)

    print("\n--- Prediction ---")
    label = "FAILURE" if result["prediction"] == 1 else "NO FAILURE"
    print(f"Result: {label}")
    print(f"Failure probability: {result['failure_probability']:.1%}")
