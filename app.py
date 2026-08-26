"""Streamlit Web UI for Predictive Maintenance ML System.

Run standalone with:
    streamlit run app.py
"""

import math
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean, modern aesthetic
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .status-safe {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }
    .status-danger {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants & Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
FEATURES = ["Type", "Air_Temperature", "Process_Temperature", "Rotational_Speed", "Torque", "Tool_Wear"]


@st.cache_resource(show_spinner="Loading machine learning model...")
def load_trained_model():
    """Load the trained machine learning pipeline from disk."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_trained_model()

# Header
st.markdown("<div class=\"main-header\">⚙️ Predictive Maintenance AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class=\"sub-header\">Real-time machine health monitoring and predictive failure risk assessment.</div>",
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        f"⚠️ Model file not found at `{MODEL_PATH}`. Please run `python -m src.run_pipeline` to train and save the model first."
    )
    st.stop()

# Sidebar: Preset quick selection and information
with st.sidebar:
    st.header("⚡ Preset Configurations")
    st.caption("Quickly test known sensor telemetry conditions:")

    preset = st.selectbox(
        "Load Example State",
        options=[
            "Custom Input",
            "Normal Operation (Healthy)",
            "High Tool Wear Risk",
            "High Torque & Overheating Risk",
            "High Speed Strain",
        ],
        index=0,
    )

    st.markdown("---")
    st.header("ℹ️ Model Details")
    st.markdown(
        """
        - **Model**: Tuned Random Forest Classifier
        - **Target**: Machine Failure (0 = Normal, 1 = Failure)
        - **Features**: Type, Air & Process Temps, RPM, Torque, Tool Wear
        """
    )
    st.markdown("---")
    st.caption("Predictive Maintenance ML Pipeline • AI4I 2020 Dataset")

# Define default values based on preset
default_vals = {
    "Type": "L",
    "Air_Temperature": 298.1,
    "Process_Temperature": 308.6,
    "Rotational_Speed": 1500,
    "Torque": 40.0,
    "Tool_Wear": 0.0,
}

if preset == "Normal Operation (Healthy)":
    default_vals = {
        "Type": "M",
        "Air_Temperature": 298.1,
        "Process_Temperature": 308.6,
        "Rotational_Speed": 1551,
        "Torque": 42.8,
        "Tool_Wear": 15.0,
    }
elif preset == "High Tool Wear Risk":
    default_vals = {
        "Type": "H",
        "Air_Temperature": 299.5,
        "Process_Temperature": 309.8,
        "Rotational_Speed": 1400,
        "Torque": 50.0,
        "Tool_Wear": 235.0,
    }
elif preset == "High Torque & Overheating Risk":
    default_vals = {
        "Type": "L",
        "Air_Temperature": 304.5,
        "Process_Temperature": 313.8,
        "Rotational_Speed": 1250,
        "Torque": 72.0,
        "Tool_Wear": 190.0,
    }
elif preset == "High Speed Strain":
    default_vals = {
        "Type": "L",
        "Air_Temperature": 298.2,
        "Process_Temperature": 308.7,
        "Rotational_Speed": 2600,
        "Torque": 16.5,
        "Tool_Wear": 180.0,
    }

# Tabs for Single Prediction and Batch CSV Prediction
tab1, tab2 = st.tabs(["🔍 Single Machine Assessment", "📊 Batch CSV Prediction"])

with tab1:
    col_input, col_pred = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.subheader("Machine Sensor Telemetry")

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            product_type = st.selectbox(
                "Product Type Variant",
                options=["L", "M", "H"],
                index=["L", "M", "H"].index(default_vals["Type"]),
                help="L: Low (50%), M: Medium (30%), H: High (20%) quality variants",
            )
        with row1_col2:
            tool_wear = st.number_input(
                "Tool Wear (minutes)",
                min_value=0.0,
                max_value=300.0,
                value=float(default_vals["Tool_Wear"]),
                step=1.0,
                help="Accumulated time in minutes the current cutting tool has been in service.",
            )

        st.markdown("##### 🌡️ Temperature Sensors")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=290.0,
                max_value=315.0,
                value=float(default_vals["Air_Temperature"]),
                step=0.1,
                format="%.1f",
                help="Ambient air temperature in Kelvin (~295K to 305K)",
            )
        with t_col2:
            process_temp = st.number_input(
                "Process Temperature [K]",
                min_value=300.0,
                max_value=325.0,
                value=float(default_vals["Process_Temperature"]),
                step=0.1,
                format="%.1f",
                help="Internal process operating temperature in Kelvin",
            )

        temp_diff = process_temp - air_temp
        st.caption(f"ℹ️ Temperature Difference ($\Delta T$): **{temp_diff:.1f} K**")

        st.markdown("##### ⚡ Rotational & Mechanical Telemetry")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            rot_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=1000,
                max_value=3000,
                value=int(default_vals["Rotational_Speed"]),
                step=10,
                help="Spindle rotational speed in revolutions per minute",
            )
        with m_col2:
            torque = st.number_input(
                "Torque [Nm]",
                min_value=0.0,
                max_value=100.0,
                value=float(default_vals["Torque"]),
                step=0.5,
                format="%.1f",
                help="Torque applied during machining in Newton-meters",
            )

        power_kw = (rot_speed * torque * 2 * math.pi) / (60 * 1000)
        st.caption(f"ℹ️ Estimated Power Output: **{power_kw:.2f} kW**")

    with col_pred:
        st.subheader("Diagnostic Assessment")

        # Prepare feature dataframe
        input_data = {
            "Type": product_type,
            "Air_Temperature": air_temp,
            "Process_Temperature": process_temp,
            "Rotational_Speed": rot_speed,
            "Torque": torque,
            "Tool_Wear": tool_wear,
        }
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        # Run inference
        prediction = int(model.predict(input_df)[0])
        failure_prob = float(model.predict_proba(input_df)[0][1])

        # Status Display
        if prediction == 1:
            st.markdown(
                f"""
                <div class="status-card status-danger">
                    <h3 style="margin: 0; color: #991B1B;">⚠️ MAINTENANCE REQUIRED</h3>
                    <p style="margin-top: 0.5rem; margin-bottom: 0;">
                        The machine is exhibiting abnormal telemetry patterns indicating a high likelihood of impending component failure.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-card status-safe">
                    <h3 style="margin: 0; color: #166534;">✅ OPERATING NORMALLY</h3>
                    <p style="margin-top: 0.5rem; margin-bottom: 0;">
                        Sensor readings are within safe operating bounds. No immediate maintenance intervention required.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Failure Probability Metric
        st.markdown("##### 📈 Failure Risk Probability")
        st.progress(min(max(failure_prob, 0.0), 1.0))
        
        m1, m2 = st.columns(2)
        m1.metric("Predicted State", "FAILURE" if prediction == 1 else "HEALTHY")
        m2.metric("Failure Risk", f"{failure_prob:.1%}", delta=f"{failure_prob - 0.05:.1%}" if failure_prob > 0.05 else "Normal", delta_color="inverse")

        # Telemetry Summary
        with st.expander("🔍 View Raw Features Sent to Model", expanded=False):
            st.dataframe(input_df, use_container_width=True)

with tab2:
    st.subheader("Bulk Sensor Log Inference")
    st.markdown("Upload a CSV containing machine sensor telemetry to perform bulk predictive maintenance screening.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Must contain Type, Air_Temperature, Process_Temperature, Rotational_Speed, Torque, Tool_Wear (or raw AI4I columns)")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Map column names if original dataset headers are used
            rename_map = {
                "Air temperature [K]": "Air_Temperature",
                "Process temperature [K]": "Process_Temperature",
                "Rotational speed [rpm]": "Rotational_Speed",
                "Torque [Nm]": "Torque",
                "Tool wear [min]": "Tool_Wear",
            }
            clean_df = df.rename(columns=rename_map)

            missing_cols = [col for col in FEATURES if col not in clean_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: `{missing_cols}`. Expected: `{FEATURES}`")
            else:
                X_batch = clean_df[FEATURES]
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)[:, 1]

                results_df = clean_df.copy()
                results_df["Predicted_Failure"] = preds
                results_df["Failure_Probability"] = probs.round(4)

                failure_count = int(preds.sum())
                total_count = len(preds)

                st.success(f"Successfully processed **{total_count}** machine records.")

                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Total Machines Checked", total_count)
                kpi2.metric("Failures Predicted", failure_count, delta=f"{(failure_count/total_count):.1%}", delta_color="inverse")
                kpi3.metric("Healthy Machines", total_count - failure_count)

                st.dataframe(results_df, use_container_width=True)

                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="predictive_maintenance_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
