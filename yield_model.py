import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv("krishi_ai_dataset.csv")

# Encode categorical variables
le_soil = LabelEncoder()
le_stage = LabelEncoder()

data['soil_type'] = le_soil.fit_transform(data['soil_type'])
data['crop_stage'] = le_stage.fit_transform(data['crop_stage'])

# Features and Target
X = data.drop(['predicted_yield', 'disease_risk'], axis=1)
y = data['predicted_yield']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Linear Regression Model
# ---------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred_lin = lin_model.predict(X_test)

rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)

print("=== Linear Regression ===")
print("RMSE:", rmse_lin)
print("R2 Score:", r2_lin)

# ---------------------------
# Random Forest Regressor
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\n=== Random Forest Regressor ===")
print("RMSE:", rmse_rf)
print("R2 Score:", r2_rf)
import joblib

joblib.dump(lin_model, "yield_linear_model.pkl")
joblib.dump(rf_model, "yield_rf_model.pkl")