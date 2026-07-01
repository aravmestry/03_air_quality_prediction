import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

csv_files = list(DATA_DIR.rglob("*.csv"))

print(f"Found {len(csv_files)} CSV files")

all_data = []

for file in csv_files:
    print(f"Loading: {file.name}")
    df = pd.read_csv(file)

    # Extract station name from file name
    station = file.name.replace("PRSA_Data_", "").replace("_20130301-20170228.csv", "")
    df["station"] = station

    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

print("\nCombined dataset shape:")
print(data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum().sort_values(ascending=False))

print("\nSummary statistics:")
print(data.describe())

# Create datetime column
data["datetime"] = pd.to_datetime(
    data[["year", "month", "day", "hour"]]
)

# Sort by station and time
data = data.sort_values(["station", "datetime"])

# Save combined raw dataset
data.to_csv("outputs/combined_air_quality_raw.csv", index=False)

print("\nSaved combined dataset to outputs/combined_air_quality_raw.csv")
print("Done.")