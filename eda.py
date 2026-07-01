import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUTPUT_DIR / "air_quality_clean.csv")

df["datetime"] = pd.to_datetime(df["datetime"])

print("Dataset loaded:", df.shape)

# Figure 1: PM2.5 over time
daily_pm25 = (
    df.groupby("datetime")["PM2.5"]
    .mean()
    .rolling(24 * 30)
    .mean()
)

plt.figure(figsize=(14, 6))
plt.plot(daily_pm25)
plt.title("Average PM2.5 Concentration Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_pm25_time_series.png", dpi=300)
plt.close()

# Figure 2: Correlation heatmap
pollution_weather_cols = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

corr = df[pollution_weather_cols].corr()

plt.figure(figsize=(11, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Matrix of Air Pollution and Meteorological Variables")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_correlation_heatmap.png", dpi=300)
plt.close()

# Figure 3: PM2.5 distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["PM2.5"], bins=60, kde=True)
plt.title("Distribution of PM2.5 Concentrations")
plt.xlabel("PM2.5 Concentration")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "03_pm25_distribution.png", dpi=300)
plt.close()

# Figure 4: Seasonal PM2.5
season_labels = {
    1: "Winter",
    2: "Spring",
    3: "Summer",
    4: "Fall"
}

df["season_name"] = df["season"].map(season_labels)

plt.figure(figsize=(9, 5))
sns.boxplot(x="season_name", y="PM2.5", data=df)
plt.title("Seasonal Distribution of PM2.5 Concentrations")
plt.xlabel("Season")
plt.ylabel("PM2.5 Concentration")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_pm25_by_season.png", dpi=300)
plt.close()

# Figure 5: PM2.5 by station
station_avg = (
    df.groupby("station")["PM2.5"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 6))
station_avg.plot(kind="bar")
plt.title("Average PM2.5 Concentration by Monitoring Station")
plt.xlabel("Station")
plt.ylabel("Average PM2.5 Concentration")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_pm25_by_station.png", dpi=300)
plt.close()

print("EDA complete. Figures saved in the figures folder.")
print("Saved figures:")
for file in sorted(FIGURE_DIR.glob("*.png")):
    print(file)