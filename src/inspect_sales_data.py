
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

sales_path = RAW / "sales_train_evaluation.csv"
calendar_path = RAW / "calendar.csv"
prices_path = RAW / "sell_prices.csv"

print("Checking dataset files...")

for path in [sales_path, calendar_path, prices_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    print(f"Found: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

print("\nReading sales data...")
sales = pd.read_csv(sales_path)

print("\nReading calendar...")
calendar = pd.read_csv(calendar_path)

print("\nReading prices...")
prices = pd.read_csv(prices_path)

print("\n========== DATASET SUMMARY ==========")
print("Sales shape:", sales.shape)
print("Calendar shape:", calendar.shape)
print("Prices shape:", prices.shape)

print("\nSales columns (first 10):")
print(sales.columns[:10].tolist())

day_columns = [c for c in sales.columns if c.startswith("d_")]
print("\nNumber of daily sales columns:", len(day_columns))

print("\nUnique products:", sales["item_id"].nunique())
print("Unique stores:", sales["store_id"].nunique())
print("Calendar date range:", calendar["date"].min(), "to", calendar["date"].max())
print("Price records:", len(prices))

print("\nSample product records:")
print(sales[["item_id", "dept_id", "cat_id", "store_id"]].head())

print("\nTotal recorded units sold:", sales[day_columns].to_numpy().sum())

print("\nInspection complete.")
