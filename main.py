from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from price_forecast import run_forecast

# -----------------------------
# Initialize App
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS Middleware (ADD THIS HERE)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# Load Trained Models
# -----------------------------
disease_model = joblib.load("disease_rf_model.pkl")
yield_model = joblib.load("yield_rf_model.pkl")

# -----------------------------
# Input Schema
# -----------------------------
class FarmInput(BaseModel):
    crop_type: str
    soil_type: int
    soil_moisture: float
    temperature: float
    rainfall: float
    crop_stage: int
    fertilizer: float
    humidity: float

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Krishi AI Backend Running"}

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: FarmInput):

    # Prepare input dataframe
    input_df = pd.DataFrame([{
        "soil_type": data.soil_type,
        "soil_moisture": data.soil_moisture,
        "temperature": data.temperature,
        "rainfall": data.rainfall,
        "crop_stage": data.crop_stage,
        "fertilizer": data.fertilizer,
        "humidity": data.humidity
    }])

    # -----------------------------
    # Disease Prediction
    # -----------------------------
    disease_prob = disease_model.predict_proba(input_df)[0][1]
    disease_score = disease_prob * 100

    # -----------------------------
    # Yield Prediction
    # -----------------------------
    predicted_yield = yield_model.predict(input_df)[0]

    # -----------------------------
    # KRI Calculation
    # -----------------------------
    yield_stress = max(0, 7 - predicted_yield) * 10
    KRI = (0.5 * disease_score) + (0.5 * yield_stress)

    if KRI < 40:
        risk_level = "Low"
    elif KRI < 70:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # -----------------------------
    # Market Forecast
    # -----------------------------
    future_price, price_change = run_forecast(data.crop_type)

    if future_price is None:
        decision = "Forecast not available for this crop"
    else:
        if KRI > 70 and price_change < 0:
            decision = "SELL IMMEDIATELY (High risk & falling prices)"
        elif KRI > 70 and price_change > 5:
            decision = "Harvest quickly & sell at predicted peak"
        elif KRI < 40 and price_change > 5:
            decision = "HOLD – Prices likely to rise"
        elif KRI < 40 and price_change < 0:
            decision = "Monitor market – consider early sale"
        else:
            decision = "Monitor crop & market conditions"

    # -----------------------------
    # Return Response
    # -----------------------------
    return {
        "crop_type": data.crop_type,
        "disease_probability_percent": round(disease_score, 2),
        "predicted_yield_tons_per_acre": round(float(predicted_yield), 2),
        "KRI": round(KRI, 2),
        "risk_level": risk_level,
        "future_price": round(future_price, 2) if future_price else None,
        "price_change_percent": round(price_change, 2) if price_change else None,
        "decision": decision
    }