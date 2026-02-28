# price_forecast.py

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def run_forecast(crop_type):

    # -----------------------------
    # 1️⃣ Select Dataset Based on Crop
    # -----------------------------
    try:
        if crop_type == "Coffee":
            df1 = pd.read_csv("coffee_2023.csv", skiprows=1)
            df2 = pd.read_csv("coffee_2024.csv", skiprows=1)
            df3 = pd.read_csv("coffee_2025.csv", skiprows=1)

        elif crop_type == "Paddy":
            df1 = pd.read_csv("paddy_2023.csv", skiprows=1)
            df2 = pd.read_csv("paddy_2024.csv", skiprows=1)
            df3 = pd.read_csv("paddy_2025.csv", skiprows=1)

        else:
            # No dataset available
            return None, None, None

    except FileNotFoundError:
        return None, None, None

    # -----------------------------
    # 2️⃣ Combine & Clean Data
    # -----------------------------
    df = pd.concat([df1, df2, df3])

    df = df[["Price Date", "Modal Price"]]

    df["Price Date"] = pd.to_datetime(df["Price Date"], dayfirst=True)

    df["Modal Price"] = df["Modal Price"].str.replace(",", "")
    df["Modal Price"] = df["Modal Price"].astype(float)

    df = df.sort_values("Price Date")

    # -----------------------------
    # 3️⃣ Monthly Aggregation
    # -----------------------------
    monthly = df.groupby(
        pd.Grouper(key="Price Date", freq="M")
    )["Modal Price"].mean()

    # -----------------------------
    # 4️⃣ Build SARIMA Model
    # -----------------------------
    model = SARIMAX(
        monthly,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )

    model_fit = model.fit(disp=False)

    forecast_obj = model_fit.get_forecast(steps=6)

    forecast_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    return monthly, forecast_mean, conf_int

    # Build SARIMA
    model = SARIMAX(
        monthly,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )

    model_fit = model.fit(disp=False)

    forecast_obj = model_fit.get_forecast(steps=6)

    forecast_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    return monthly, forecast_mean, conf_int