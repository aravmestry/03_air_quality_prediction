import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "air_quality_clean.csv")

features = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "season", "weekend", "hour_sin", "hour_cos"
]

X = df[features]
y = df["PM2.5"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42
    )
}

results = []
best_model = None
best_r2 = -999
best_name = None

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

print("\nMODEL COMPARISON")
print(results_df)

results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

with open(OUTPUT_DIR / "best_model_name.txt", "w") as f:
    f.write(best_name)

plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylabel("R² Score")
plt.title("Model Comparison for PM2.5 Prediction")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "10_model_comparison_r2.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.ylabel("RMSE")
plt.title("Model Comparison by RMSE")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "11_model_comparison_rmse.png", dpi=300)
plt.close()

print(f"\nBest model saved: {best_name}")