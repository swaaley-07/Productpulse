
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_dataset.parquet"

df = pd.read_parquet(DATA_PATH)

print("\n1. DATASET OVERVIEW")
print("-" * 50)
print("Shape:", df.shape)
print("Unique products:", df["item_id"].nunique())
print("Unique stores:", df["store_id"].nunique())
print("Prediction dates:", sorted(df["prediction_date"].dt.strftime("%Y-%m-%d").unique()))

print("\n2. TARGET DISTRIBUTION BY CATEGORY")
print("-" * 50)
print(pd.crosstab(
    df["category"],
    df["future_sales_active"],
    normalize="index"
).round(3))

print("\n3. TARGET DISTRIBUTION BY PREDICTION DATE")
print("-" * 50)
print(pd.crosstab(
    df["prediction_date"].dt.strftime("%Y-%m-%d"),
    df["future_sales_active"],
    normalize="index"
).round(3))

print("\n4. SALES HISTORY FOR EACH TARGET CLASS")
print("-" * 50)
print(df.groupby("future_sales_active")[
    ["units_last_28d", "units_last_90d",
     "active_days_last_28d", "active_days_last_90d"]
].median().round(2))

print("\n5. CHECK FOR MISSING VALUES")
print("-" * 50)
print(df.isna().sum())

print("\n6. CHECK DUPLICATES")
print("-" * 50)
print(
    "Duplicate product-store-date rows:",
    df.duplicated(["item_id", "store_id", "prediction_date"]).sum()
)
