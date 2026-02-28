import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_forecast(crop_name):

    if crop_name.lower() == "paddy":
        files = ["paddy_2023.csv", "paddy_2024.csv", "paddy_2025.csv"]

    elif crop_name.lower() == "coffee":
        files = ["Coffee_2023.csv", "Coffee_2024.csv", "Coffee_2025.csv"]

    else:
        return None, None

    dfs = []
    for file in files:
        df = pd.read_csv(file, skiprows=1)
        dfs.append(df)

    df = pd.concat(dfs)

    df = df[["Price Date", "Modal Price"]]
    df["Price Date"] = pd.to_datetime(df["Price Date"], dayfirst=True)
    df["Modal Price"] = df["Modal Price"].str.replace(",", "").astype(float)

    df = df.sort_values("Price Date")

    monthly = df.groupby(pd.Grouper(key="Price Date", freq="ME"))["Modal Price"].mean()

    model = SARIMAX(monthly,
                    order=(1,1,1),
                    seasonal_order=(1,1,1,12))

    model_fit = model.fit(disp=False)

    forecast = model_fit.get_forecast(steps=6)
    forecast_mean = forecast.predicted_mean

    current_price = monthly.iloc[-1]
    future_price = forecast_mean.iloc[-1]

    price_change = ((future_price - current_price) / current_price) * 100

    return float(future_price), float(price_change)