import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("krishi_ai_dataset.csv")

# Normalize helper
def normalize(series):
    return 100 * (series - series.min()) / (series.max() - series.min())

# 1️⃣ Disease Score (use disease_risk directly for now)
data['disease_score'] = data['disease_risk'] * 100

# 2️⃣ Yield Stress Score
avg_yield = data['predicted_yield'].mean()
yield_deviation = avg_yield - data['predicted_yield']
yield_deviation[yield_deviation < 0] = 0
data['yield_stress'] = normalize(yield_deviation)

# 3️⃣ Soil Stress (low moisture = high stress)
max_moisture = data['soil_moisture'].max()
data['soil_stress'] = normalize(max_moisture - data['soil_moisture'])

# 4️⃣ Weather Stress (high temp + low rainfall)
weather_raw = data['temperature'] - data['rainfall'] * 0.1
data['weather_stress'] = normalize(weather_raw)

# Final KRI
data['KRI'] = (
    0.30 * data['disease_score'] +
    0.30 * data['yield_stress'] +
    0.20 * data['soil_stress'] +
    0.20 * data['weather_stress']
)

# Categorize Risk
def categorize(kri):
    if kri < 40:
        return "Low"
    elif kri < 70:
        return "Moderate"
    else:
        return "High"

data['Risk_Level'] = data['KRI'].apply(categorize)

print("Sample Output:")
print(data[['predicted_yield', 'KRI', 'Risk_Level']].head())

print("\nRisk Distribution:")
print(data['Risk_Level'].value_counts())