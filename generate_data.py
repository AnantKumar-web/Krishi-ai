import numpy as np
import pandas as pd

np.random.seed(42)

n = 2000

soil_types = ['Sandy', 'Loamy', 'Clay']
crop_stages = ['Early', 'Mid', 'Late']

data = pd.DataFrame({
    'soil_type': np.random.choice(soil_types, n),
    'soil_moisture': np.random.uniform(10, 60, n),
    'temperature': np.random.uniform(15, 45, n),
    'rainfall': np.random.uniform(0, 200, n),
    'crop_stage': np.random.choice(crop_stages, n),
    'fertilizer': np.random.uniform(0, 150, n),
    'humidity': np.random.uniform(30, 90, n)
})

# Disease Risk Logic
disease_prob = (
    0.03 * data['temperature'] +
    0.04 * data['humidity'] +
    0.01 * data['rainfall']
)

disease_prob = disease_prob / disease_prob.max()

data['disease_risk'] = (disease_prob > 0.6).astype(int)

# Yield Logic
yield_base = (
    5 +
    0.02 * data['soil_moisture'] +
    0.01 * data['fertilizer'] -
    0.03 * data['temperature']
)

# Soil effect
soil_bonus = data['soil_type'].map({
    'Loamy': 1.0,
    'Clay': 0.5,
    'Sandy': -0.5
})

data['predicted_yield'] = yield_base + soil_bonus

data.to_csv("krishi_ai_dataset.csv", index=False)

print("Dataset Generated Successfully")