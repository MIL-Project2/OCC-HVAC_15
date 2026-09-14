"""
streamlit_app.py

Upload a new raw occupancy-sensor CSV -> get a 15-minute-ahead
occupancy forecast -> see it drive an HVAC 1R1C simulation (scheduled
baseline vs thermal-predictive control).

This app does NOT train anything. It loads the model trained by
Occupancy_Training.ipynb and runs inference + simulation only.

Run with:
    streamlit run streamlit_app.py

Expects, in the same folder:
    occupancy_model_15min.pkl
    occupancy_model_15min_config.json
    feature_engineering.py
    Occupancy_Inference.py
    hvac_simulation.py
"""

import pandas as pd
import streamlit as st

from Occupancy_Inference import load_model, predict, DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH
from hvac_simulation import run_simulation, HVACConfig


st.set_page_config(page_title="Occupancy + HVAC Simulator", layout="wide")
st.title("Occupancy Forecast -> HVAC Simulation")
st.caption(
    "Upload raw sensor data (a timestamp column and a headcount column). "
    "This runs the trained 15-minute-ahead occupancy model, then a 1R1C "
    "HVAC simulation comparing a scheduled baseline against thermal-"
    "predictive control."
)


# ----------------------------------------------------------------------
# Sidebar: model status + HVAC parameters
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    try:
        model, model_config = load_model(DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH)
        st.success(f"Loaded {DEFAULT_MODEL_PATH}")
        st.caption(f"Forecast horizon: {model_config.get('forecast_horizon_minutes', '?')} min")
    except FileNotFoundError:
        model = None
        model_config = None
        st.error(
            f"Could not find '{DEFAULT_MODEL_PATH}' / '{DEFAULT_CONFIG_PATH}'. "
            "Run Occupancy_Training.ipynb first, and place the two output "
            "files next to this app."
        )
    except ValueError as e:
        model = None
        model_config = None
        st.error(str(e))

    st.header("HVAC parameters")
    st.caption("Prototype defaults -- calibrate against the real room before trusting results.")

    comfort_min = st.slider("Comfort min (deg C)", 18.0, 26.0, 22.8, 0.1)
    comfort_max = st.slider("Comfort max (deg C)", comfort_min, 30.0, 25.8, 0.1)
    cop = st.slider("HVAC COP", 1.0, 6.0, 3.0, 0.1)
    q_cooling_max_w = st.slider("Max cooling capacity (W)", 500, 20000, 5000, 500)
    r_thermal = st.number_input("R thermal (K/W)", value=0.01, format="%.5f")
    c_thermal = st.number_input("C thermal (J/K)", value=1_000_000.0, step=50_000.0)

    with st.expander("Scheduled baseline hours"):
        schedule_start = st.slider("Start hour", 0.0, 23.0, 8.0, 0.5)
        schedule_end = st.slider("End hour", schedule_start, 24.0, 18.0, 0.5)
        weekdays_only = st.checkbox("Weekdays only", value=True)

    cfg = HVACConfig(
        comfort_min_c=comfort_min,
        comfort_max_c=comfort_max,
        cop=cop,
        q_cooling_max_w=float(q_cooling_max_w),
        r_thermal_k_per_w=r_thermal,
        c_thermal_j_per_k=c_thermal,
        schedule_start_hour=schedule_start,
        schedule_end_hour=schedule_end,
        schedule_weekdays_only=weekdays_only,
    )


# ----------------------------------------------------------------------
# Main: file upload
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a CSV of raw sensor readings", type=["csv"])

with st.expander("Expected columns"):
    st.markdown(
        "- **Timestamp column** (default name `collecteddate`) -- any format "
        "`pandas.to_datetime` can parse.\n"
        "- **Headcount column** (default name `headcount`) -- the raw "
        "occupancy sensor reading.\n"
        "- *(optional)* An outdoor-temperature column named "
        "`outdoor_temperature_C` -- if missing, a constant fallback "
        "temperature is used for the HVAC simulation, which will make "
        "the pre-cooling comparison less meaningful.\n\n"
        "Provide at least 15 minutes of continuous, regularly-spaced "
        "readings before the first point you want a forecast for."
    )

col1, col2 = st.columns(2)
with col1:
    timestamp_col = st.text_input("Timestamp column name", value="collecteddate")
with col2:
    headcount_col = st.text_input("Headcount column name", value="headcount")


if uploaded_file is not None and model is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()

    with st.spinner("Engineering features and running the occupancy forecast..."):
        try:
            pred_df = predict(raw_df, model, timestamp_col=timestamp_col, headcount_col=headcount_col)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.subheader("Occupancy forecast")
    forecast_chart_df = pred_df.set_index('timestamp')[
        ['occupancy_now', 'occupancy_forecast_15min']
    ]
    st.line_chart(forecast_chart_df)

    if 'occupancy_actual_15min_later' in pred_df.columns and pred_df['occupancy_actual_15min_later'].notna().any():
        with st.expander("Forecast accuracy (this upload includes enough history to check)"):
            accuracy_df = pred_df.dropna(subset=['occupancy_actual_15min_later']).set_index('forecast_time')[
                ['occupancy_actual_15min_later', 'occupancy_forecast_15min']
            ]
            st.line_chart(accuracy_df)

    st.subheader("HVAC simulation")
    with st.spinner("Running the 1R1C HVAC simulation..."):
        try:
            sim = run_simulation(pred_df, cfg=cfg)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if sim['outdoor_source'] == 'constant_fallback':
        st.warning(
            f"No outdoor-temperature column found -- using a constant "
            f"{cfg.fallback_outdoor_temp_c:.1f} deg C for this simulation."
        )

    kpi_cols = st.columns(3)
    energy_saving = (
        sim['scheduled_kpi']['Total HVAC Energy (kWh)']
        - sim['predictive_kpi']['Total HVAC Energy (kWh)']
    )
    kpi_cols[0].metric(
        "Predictive HVAC energy",
        f"{sim['predictive_kpi']['Total HVAC Energy (kWh)']:.2f} kWh",
        delta=f"{-energy_saving:+.2f} kWh vs scheduled",
        delta_color="inverse",
    )
    kpi_cols[1].metric(
        "Predictive comfort compliance",
        f"{sim['predictive_kpi']['Occupied Comfort Compliance (%)']:.1f}%",
        delta=(
            f"{sim['predictive_kpi']['Occupied Comfort Compliance (%)'] - sim['scheduled_kpi']['Occupied Comfort Compliance (%)']:+.1f} pts"
        ),
    )
    kpi_cols[2].metric(
        "Predictive cooling runtime",
        f"{sim['predictive_kpi']['Cooling Runtime (h)']:.1f} h",
        delta=f"{sim['predictive_kpi']['Cooling Runtime (h)'] - sim['scheduled_kpi']['Cooling Runtime (h)']:+.1f} h vs scheduled",
        delta_color="inverse",
    )

    st.dataframe(sim['summary_df'].round(3), use_container_width=True)

    st.markdown("**Zone temperature**")
    temp_df = sim['comparison_df'].set_index('timestamp')[
        ['scheduled_room_temperature_C', 'predictive_room_temperature_C', 'outdoor_temperature_C']
    ]
    st.line_chart(temp_df)

    st.markdown("**Cumulative HVAC energy**")
    energy_df = sim['comparison_df'].set_index('timestamp')[
        ['scheduled_energy_cumulative_kWh', 'predictive_energy_cumulative_kWh']
    ]
    st.line_chart(energy_df)

    st.download_button(
        "Download occupancy forecast (CSV)",
        pred_df.to_csv(index=False).encode('utf-8'),
        file_name='occupancy_forecast.csv',
    )
    st.download_button(
        "Download HVAC comparison (CSV)",
        sim['comparison_df'].to_csv(index=False).encode('utf-8'),
        file_name='hvac_comparison.csv',
    )

    st.caption(
        "Reminder: R, C, HVAC capacity/COP and the comfort band above are "
        "prototype defaults. Calibrate them against the real room before "
        "reporting any energy-saving or comfort number as a building result."
    )

elif uploaded_file is not None and model is None:
    st.info("Fix the model-loading error in the sidebar before uploading data.")
else:
    st.info("Upload a CSV to run the forecast and simulation.")
