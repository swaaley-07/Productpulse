
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
calendar = pd.read_csv(RAW / "calendar.csv")

day_columns = [c for c in sales.columns if c.startswith("d_")]
dates = pd.to_datetime(calendar.set_index("d")["date"])

print("Sales history:", dates.min().date(), "to", dates.max().date())
print("Product-store series:", len(sales))
print("Daily observations per series:", len(day_columns))

results = []

for _, row in sales.iterrows():
    values = row[day_columns].to_numpy(dtype=np.int32)

    positive_days = np.flatnonzero(values > 0)
    if len(positive_days) == 0:
        continue

    first = int(positive_days[0])
    last = int(positive_days[-1])

    # Only measure gaps after the first sale and before the last sale.
    active_positions = np.flatnonzero(values[first:last + 1] > 0)
    gaps = np.diff(active_positions) - 1

    results.append({
        "item_id": row["item_id"],
        "store_id": row["store_id"],
        "first_sale_day": first,
        "last_sale_day": last,
        "days_between_first_and_last": last - first,
        "max_internal_zero_gap": int(gaps.max()) if len(gaps) else 0,
        "total_units": int(values.sum()),
    })

summary = pd.DataFrame(results)

print("\nProducts with at least one recorded sale:", len(summary))
print("\nLongest internal zero-sales gap (days):")
print(summary["max_internal_zero_gap"].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
).round(1))

print("\nSensitivity to inactivity thresholds:")
for threshold in [28, 56, 90, 120]:
    count = (summary["max_internal_zero_gap"] >= threshold).sum()
    print(
        f"{threshold:>3} days: {count:>6} series "
        f"({count / len(summary):.1%}) have an internal gap this long"
    )

out = ROOT / "reports" / "sales_gap_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out, index=False)

print(f"\nSaved product-store summary to: {out}")
