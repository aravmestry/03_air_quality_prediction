import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load data
# ----------------------------

df = pd.read_csv(OUTPUT_DIR / "air_quality_clean.csv")

# ----------------------------
# Features
# ----------------------------

features = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "season",
    "weekend",
    "hour_sin",
    "hour_cos"
]

X = df[features]
y = df["PM2.5"]

# ----------------------------
# Train/Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ----------------------------
# Random Forest
# ----------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# ----------------------------
# Metrics
# ----------------------------

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n========================")
print("Random Forest Results")
print("========================")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.4f}")

# ----------------------------
# Feature Importance
# ----------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nTop Features\n")
print(importance)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

joblib.dump(
    model,
    MODEL_DIR / "random_forest.pkl"
)

print("\nModel saved successfully.")