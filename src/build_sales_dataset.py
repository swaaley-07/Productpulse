
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

print("Loading M5 data...")
sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
calendar = pd.read_csv(RAW / "calendar.csv")
prices = pd.read_csv(RAW / "sell_prices.csv")

day_columns = [c for c in sales.columns if c.startswith("d_")]
calendar = calendar.set_index("d").loc[day_columns].reset_index()
dates = pd.to_datetime(calendar["date"])

# Use fixed historical prediction dates.
# Each requires 90 days of past data and 90 days of future data.
prediction_days = [500, 700, 900, 1100, 1300, 1500, 1700]

print("Preparing price lookup...")
prices["store_item"] = prices["store_id"] + "_" + prices["item_id"]
price_lookup = prices.set_index(
    ["store_id", "item_id", "wm_yr_wk"]
)["sell_price"]

rows = []

print("Building product-store observations...")

for _, product in sales.iterrows():
    item = product["item_id"]
    store = product["store_id"]
    category = product["cat_id"]

    values = product[day_columns].to_numpy(dtype=np.float32)

    for end_day in prediction_days:
        # Python slice end_day-89:end_day gives 90 past days.
        start = end_day - 89
        past = values[start:end_day + 1]

        # The future target covers the 90 days after end_day.
        future = values[end_day + 1:end_day + 91]

        if len(past) != 90 or len(future) != 90:
            continue

        # Skip series with no sales in the observed past.
        if past.sum() == 0:
            continue

        active_days_28 = int((past[-28:] > 0).sum())
        active_days_90 = int((past > 0).sum())
        units_28 = float(past[-28:].sum())
        units_90 = float(past.sum())

        # Compare the latest 28 days with the preceding 28 days.
        previous_28 = float(past[-56:-28].sum())
        recent_28 = units_28
        sales_change = (
            (recent_28 - previous_28) / (previous_28 + 1)
        )

        future_units = float(future.sum())
        future_active = int(future_units > 0)

        rows.append({
            "item_id": item,
            "store_id": store,
            "category": category,
            "prediction_date": dates.iloc[end_day],
            "units_last_28d": units_28,
            "units_last_90d": units_90,
            "active_days_last_28d": active_days_28,
            "active_days_last_90d": active_days_90,
            "sales_change_recent_vs_previous_28d": sales_change,
            "future_units_next_90d": future_units,
            "future_sales_active": future_active,
        })

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No observations were created. Check day indexes.")

# Save the complete supervised dataset.
output = PROCESSED / "sales_longevity_dataset.parquet"
df.to_parquet(output, index=False)

print("\nDataset created successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nTarget distribution:")
print(df["future_sales_active"].value_counts())
print("\nTarget proportions:")
print(df["future_sales_active"].value_counts(normalize=True).round(3))
print("\nPrediction date range:")
print(df["prediction_date"].min(), "to", df["prediction_date"].max())
print("\nSaved to:", output)
print("\nNote: prices are not yet included in this first version.")
