import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")
MODEL_DIR = Path("models")

FIGURE_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "air_quality_clean.csv")

features = [
    "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
    "season", "weekend", "hour_sin", "hour_cos"
]

X = df[features]

model = load(MODEL_DIR / "best_model.pkl")

# Smaller sample so SHAP does not crash
X_sample = X.sample(n=500, random_state=42)

print("Loaded model successfully.")
print("Running SHAP on sample size:", X_sample.shape)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# SHAP summary plot
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    show=False,
    max_display=14
)

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "12_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# SHAP bar plot
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False,
    max_display=14
)

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "13_shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved:")
print("figures/12_shap_summary.png")
print("figures/13_shap_feature_importance.png")