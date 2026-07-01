import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_validate

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "air_quality_clean.csv")

features = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "season", "weekend", "hour_sin", "hour_cos"
]

X = df[features]
y = df["PM2.5"]

model = ExtraTreesRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error"
}

results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=True
)

summary = pd.DataFrame({
    "Fold": range(1, 6),
    "Train_R2": results["train_r2"],
    "Test_R2": results["test_r2"],
    "Test_MAE": -results["test_mae"],
    "Test_RMSE": -results["test_rmse"]
})

print("\n5-Fold Cross Validation Results")
print(summary)

print("\nSummary")
print(f"Mean Test R²   : {summary['Test_R2'].mean():.4f}")
print(f"Std Test R²    : {summary['Test_R2'].std():.4f}")
print(f"Mean Test MAE  : {summary['Test_MAE'].mean():.3f}")
print(f"Mean Test RMSE : {summary['Test_RMSE'].mean():.3f}")

summary.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)

print("\nSaved to outputs/cross_validation_results.csv")