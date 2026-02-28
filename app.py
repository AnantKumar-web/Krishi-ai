import streamlit as st
import pandas as pd
import numpy as np
import joblib
from price_forecast import run_forecast

# -----------------------------
# Crop Profiles
# -----------------------------
crop_profiles = {
    "Paddy": {"temp_opt": (20, 35), "moisture_opt": (60, 85), "humidity_risk": 85},
    "Wheat": {"temp_opt": (15, 25), "moisture_opt": (40, 60), "humidity_risk": 70},
    "Maize": {"temp_opt": (18, 30), "moisture_opt": (45, 70), "humidity_risk": 75},
    "Rice": {"temp_opt": (22, 34), "moisture_opt": (65, 85), "humidity_risk": 80},
    "Coffee": {"temp_opt": (18, 28), "moisture_opt": (50, 70), "humidity_risk": 75},
}

# Load ML Models
disease_model = joblib.load("disease_rf_model.pkl")
yield_model = joblib.load("yield_rf_model.pkl")

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Krishi AI – Integrated Crop & Market Intelligence System")
st.caption("AI-driven crop health risk assessment + price forecasting")

st.sidebar.header("Input Farm Conditions")

crop_type = st.sidebar.selectbox("Select Crop",
                                 ["Paddy", "Wheat", "Maize", "Rice", "Coffee"])

soil_type = st.sidebar.selectbox("Soil Type", ["Sandy", "Loamy", "Clay"])
crop_stage = st.sidebar.selectbox("Crop Stage", ["Early", "Mid", "Late"])

soil_moisture = st.sidebar.slider("Soil Moisture (%)", 10.0, 60.0, 30.0)
temperature = st.sidebar.slider("Temperature (°C)", 15.0, 45.0, 30.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 200.0, 50.0)
fertilizer = st.sidebar.slider("Fertilizer (kg/acre)", 0.0, 150.0, 50.0)
humidity = st.sidebar.slider("Humidity (%)", 30.0, 90.0, 60.0)

soil_map = {"Clay": 0, "Loamy": 1, "Sandy": 2}
stage_map = {"Early": 0, "Mid": 1, "Late": 2}

input_data = pd.DataFrame([{
    "soil_type": soil_map[soil_type],
    "soil_moisture": soil_moisture,
    "temperature": temperature,
    "rainfall": rainfall,
    "crop_stage": stage_map[crop_stage],
    "fertilizer": fertilizer,
    "humidity": humidity
}])

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if st.button("Predict"):

    # -----------------------------
    # 1️⃣ CROP RISK ENGINE
    # -----------------------------
    disease_prob = disease_model.predict_proba(input_data)[0][1]
    disease_score = disease_prob * 100

    predicted_yield = yield_model.predict(input_data)[0]
    profile = crop_profiles[crop_type]

    # Temperature stress
    temp_min, temp_max = profile["temp_opt"]
    if temperature < temp_min:
        temp_stress = min((temp_min - temperature) * 10, 100)
    elif temperature > temp_max:
        temp_stress = min((temperature - temp_max) * 10, 100)
    else:
        temp_stress = 10

    # Moisture stress
    moist_min, moist_max = profile["moisture_opt"]
    moisture_stress = 80 if soil_moisture < moist_min or soil_moisture > moist_max else 20

    # Humidity boost
    if humidity > profile["humidity_risk"]:
        disease_score = min(disease_score + 20, 100)

    # Yield stress
    yield_stress = min(max(0, 7 - predicted_yield) / 3 * 100, 100)

    # Soil stress
    soil_stress = min(max(0, 60 - soil_moisture) / 50 * 100, 100)

    # Weather stress
    weather_stress = min(max((temperature - rainfall * 0.1) * 2, 0), 100)

    # Crop-specific weight
    disease_weight = 0.4 if crop_type == "Wheat" else 0.6 if crop_type == "Coffee" else 0.5

    # Stage factor
    stage_factor = 1.2 if crop_stage == "Early" else 1.1 if crop_stage == "Late" else 1.0
    disease_score = min(disease_score * stage_factor, 100)

    # Soil type adjustment
    if soil_type == "Sandy":
        soil_stress *= 1.2
    elif soil_type == "Clay":
        soil_stress *= 1.1

    soil_stress = min(soil_stress, 100)

    KRI = (
        disease_weight * disease_score +
        0.20 * yield_stress +
        0.20 * soil_stress +
        0.10 * weather_stress
    )

    risk_level = "Low" if KRI < 40 else "Moderate" if KRI < 70 else "High"
    if disease_score > 85:
        risk_level = "High"

    # -----------------------------
    # Display Crop Risk
    # -----------------------------
    st.markdown("---")
    st.subheader("🔍 Crop Risk Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Disease Probability (%)", f"{disease_score:.2f}")
    col2.metric("Predicted Yield (tons/acre)", f"{predicted_yield:.2f}")
    col3.metric("Krishi Risk Index (KRI)", f"{KRI:.2f}")

    st.progress(min(int(KRI), 100))

    if risk_level == "Low":
        st.success("Low Risk – Maintain current farming practices.")
    elif risk_level == "Moderate":
        st.warning("Moderate Risk – Monitor crop and adjust inputs.")
    else:
        st.error("High Risk – Immediate intervention required.")

    # -----------------------------
    # 2️⃣ MARKET FORECAST ENGINE
    # -----------------------------
    monthly, forecast_mean, conf_int = run_forecast(crop_type)

    current_price = monthly.iloc[-1]
    future_price = forecast_mean.iloc[-1]
    price_change = ((future_price - current_price) / current_price) * 100

    st.markdown("---")
    st.subheader("📈 Market Forecast")

    st.write(f"Current Market Price: ₹{current_price:.2f}")
    st.write(f"Forecasted Price (6 months ahead): ₹{future_price:.2f}")
    st.write(f"Expected Price Change: {price_change:.2f}%")

    # -----------------------------
    # 3️⃣ SMART DECISION ENGINE
    # -----------------------------
    if KRI > 70 and price_change < 0:
        decision = "🚨 SELL IMMEDIATELY (High crop risk & falling prices)"
    elif KRI > 70 and price_change > 5:
        decision = "⚠ Harvest quickly & sell at predicted peak"
    elif KRI < 40 and price_change > 5:
        decision = "✅ HOLD – Prices likely to rise"
    elif KRI < 40 and price_change < 0:
        decision = "⚠ Monitor market – consider early sale"
    else:
        decision = "📊 Monitor crop & market conditions"

    st.markdown("## 🧠 Smart Market Decision")
    st.info(decision)

# -----------------------------
# MODEL INFO
# -----------------------------
st.markdown("---")
st.subheader("📊 Model Performance Summary")

col1, col2 = st.columns(2)
col1.metric("Disease Model Accuracy", "94%")
col2.metric("Yield Model R² Score", "0.99")