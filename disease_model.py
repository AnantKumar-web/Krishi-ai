import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data = pd.read_csv("krishi_ai_dataset.csv")

# Encode categorical variables
le_soil = LabelEncoder()
le_stage = LabelEncoder()

data['soil_type'] = le_soil.fit_transform(data['soil_type'])
data['crop_stage'] = le_stage.fit_transform(data['crop_stage'])

# Features and Target
X = data.drop(['disease_risk', 'predicted_yield'], axis=1)
y = data['disease_risk']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Logistic Regression Model
# ---------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# ---------------------------
# Random Forest Model
# ---------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
import joblib

joblib.dump(log_model, "disease_model.pkl")
joblib.dump(rf_model, "disease_rf_model.pkl")