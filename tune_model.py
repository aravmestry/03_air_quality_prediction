import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

df = pd.read_csv(OUTPUT_DIR / "air_quality_clean.csv")

features = [
    "PM10","SO2","NO2","CO","O3",
    "TEMP","PRES","DEWP","RAIN","WSPM",
    "season","weekend","hour_sin","hour_cos"
]

X = df[features]
y = df["PM2.5"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

param_grid = {
    "n_estimators":[300,500,700],
    "max_depth":[None,20,30,40],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4],
    "max_features":["sqrt","log2",None]
}

model = ExtraTreesRegressor(
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train)

best = search.best_estimator_

pred = best.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n==============================")
print("OPTIMIZED EXTRA TREES")
print("==============================")
print(search.best_params_)
print()
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.5f}")

joblib.dump(best, MODEL_DIR / "best_extra_trees.pkl")