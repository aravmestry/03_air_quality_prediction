import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

FIGURE_DIR.mkdir(exist_ok=True)

# --------------------------
# Load Data
# --------------------------

df = pd.read_csv(OUTPUT_DIR / "air_quality_clean.csv")

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = joblib.load(MODEL_DIR / "random_forest.pkl")

pred = model.predict(X_test)

# --------------------------
# Metrics
# --------------------------

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nEvaluation")
print("-------------------")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.4f}")

# --------------------------
# Figure 1
# Observed vs Predicted
# --------------------------

plt.figure(figsize=(7,7))

plt.scatter(
    y_test,
    pred,
    alpha=0.3,
    s=10
)

lims = [
    min(y_test.min(), pred.min()),
    max(y_test.max(), pred.max())
]

plt.plot(lims, lims, 'r--')

plt.xlabel("Observed PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Observed vs Predicted PM2.5")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "06_observed_vs_predicted.png", dpi=300)
plt.close()

# --------------------------
# Figure 2
# Residual Plot
# --------------------------

residuals = y_test - pred

plt.figure(figsize=(8,6))

plt.scatter(
    pred,
    residuals,
    alpha=0.3,
    s=10
)

plt.axhline(0,color="red",linestyle="--")

plt.xlabel("Predicted PM2.5")
plt.ylabel("Residual")

plt.title("Residual Plot")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "07_residual_plot.png", dpi=300)
plt.close()

# --------------------------
# Figure 3
# Residual Histogram
# --------------------------

plt.figure(figsize=(8,5))

plt.hist(
    residuals,
    bins=60
)

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.title("Residual Distribution")

plt.tight_layout()

plt.savefig(FIGURE_DIR / "08_residual_histogram.png", dpi=300)

plt.close()

# --------------------------
# Figure 4
# Feature Importance
# --------------------------

importance = pd.DataFrame({
    "Feature":features,
    "Importance":model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(8,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR/"09_feature_importance.png",
    dpi=300
)

plt.close()

print("\nEvaluation figures created successfully.")