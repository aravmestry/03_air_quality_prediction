import pandas as pd
from pathlib import Path

# -----------------------------
# Load Combined Dataset
# -----------------------------

df = pd.read_csv("outputs/combined_air_quality_raw.csv")

print("="*60)
print("ORIGINAL DATASET")
print("="*60)
print(df.shape)

# -----------------------------
# Create Datetime
# -----------------------------

df["datetime"] = pd.to_datetime(
    df[["year","month","day","hour"]]
)

# -----------------------------
# Drop useless column
# -----------------------------

if "No" in df.columns:
    df = df.drop(columns=["No"])

# -----------------------------
# Missing Values
# -----------------------------

print("\nMissing Values Before Cleaning\n")
print(df.isna().sum())

# Fill missing numerical values with median

numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing wind direction

if "wd" in df.columns:
    df["wd"] = df["wd"].fillna(df["wd"].mode()[0])

print("\nMissing Values After Cleaning\n")
print(df.isna().sum())

# -----------------------------
# Feature Engineering
# -----------------------------

df["season"] = (
    (df["month"] % 12 + 3) // 3
)

df["weekend"] = (
    df["datetime"]
    .dt.dayofweek
    .isin([5,6])
    .astype(int)
)

df["hour_sin"] = (
    __import__("numpy").sin(
        2*__import__("numpy").pi*df["hour"]/24
    )
)

df["hour_cos"] = (
    __import__("numpy").cos(
        2*__import__("numpy").pi*df["hour"]/24
    )
)

# -----------------------------
# Save
# -----------------------------

Path("outputs").mkdir(exist_ok=True)

df.to_csv(
    "outputs/air_quality_clean.csv",
    index=False
)

print("\nSaved cleaned dataset.")
print(df.shape)
print(df.head())