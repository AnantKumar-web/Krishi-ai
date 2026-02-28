import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("krishi_ai_dataset.csv")

print("Shape of dataset:", data.shape)
print(data.head())
print(data.describe())

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
print("Disease Risk Distribution:")
print(data['disease_risk'].value_counts())