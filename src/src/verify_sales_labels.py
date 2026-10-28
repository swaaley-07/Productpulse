
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data" / "processed" / "sales_longevity_dataset.parquet"

print("Loading data...")
sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
calendar = pd.read_csv(RAW / "calendar.csv")
df = pd.read_parquet(DATA)

day_columns = [c for c in sales.columns if c.startswith("d_")]

# Map each sales day to its real calendar date.
calendar_dates = calendar.set_index("d").loc[day_columns, "date"]
calendar_dates = pd.to_datetime(calendar_dates)

print("\nVerifying five randomly selected observations...")
samples = df.sample(5, random_state=42)
all_match = True

for _, row in samples.iterrows():
    product = sales[
        (sales["item_id"] == row["item_id"])
        & (sales["store_id"] == row["store_id"])
    ]

    if product.empty:
        print("ERROR: Product-store combination not found.")
        all_match = False
        continue

    values = product[day_columns].iloc[0].to_numpy(dtype=np.float64)

    matching_days = np.flatnonzero(
        calendar_dates.to_numpy() == np.datetime64(row["prediction_date"])
    )

    if len(matching_days) == 0:
        print("ERROR: Prediction date not found in calendar.")
        all_match = False
        continue

    idx = int(matching_days[0])

    past = values[idx - 89:idx + 1]
    future = values[idx + 1:idx + 91]

    stored_past = float(row["units_last_90d"])
    stored_future = float(row["future_units_next_90d"])

    actual_past = float(past.sum())
    actual_future = float(future.sum())

    past_match = np.isclose(stored_past, actual_past)
    future_match = np.isclose(stored_future, actual_future)

    print(f"\nProduct: {row['item_id']} | Store: {row['store_id']}")
    print("Prediction date:", pd.Timestamp(row["prediction_date"]).date())
    print(f"Past 90 days: stored={stored_past}, actual={actual_past}")
    print(f"Future 90 days: stored={stored_future}, actual={actual_future}")
    print("Result:", "PASS" if past_match and future_match else "FAIL")

    if not (past_match and future_match):
        all_match = False

print("\n" + "=" * 50)

if all_match:
    print("PASS: All five sampled observations match.")
else:
    print("FAIL: At least one observation does not match.")
    print("Do not train the model until the issue is fixed.")
