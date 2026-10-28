
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
calendar = pd.read_csv(RAW / "calendar.csv")

day_columns = [c for c in sales.columns if c.startswith("d_")]
dates = pd.to_datetime(calendar.set_index("d").loc[day_columns, "date"])

results = []

for _, row in sales.iterrows():
    values = row[day_columns].to_numpy(dtype=np.int32)
    positive = np.flatnonzero(values > 0)

    if len(positive) == 0:
        continue

    first = int(positive[0])
    last = int(positive[-1])

    # Longest internal gap between positive-sales days
    active = np.flatnonzero(values[first:last + 1] > 0)
    gaps = np.diff(active) - 1
    max_gap = int(gaps.max()) if len(gaps) else 0

    # Days observed after the last positive sale
    trailing_zero_days = len(values) - 1 - last

    results.append({
        "item_id": row["item_id"],
        "store_id": row["store_id"],
        "category": row["cat_id"],
        "first_sale_date": dates.iloc[first],
        "last_sale_date": dates.iloc[last],
        "max_internal_gap": max_gap,
        "trailing_zero_days": trailing_zero_days,
        "total_units": int(values.sum()),
    })

df = pd.DataFrame(results)

print("Total product-store series:", len(df))
print("\nTrailing zero-sales days:")
print(df["trailing_zero_days"].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95]
).round(1))

print("\nSensitivity by inactivity threshold:")
for threshold in [90, 120, 180]:
    internal = (df["max_internal_gap"] >= threshold).mean()
    trailing = (df["trailing_zero_days"] >= threshold).mean()
    print(
        f"{threshold} days | internal gaps: {internal:.1%} | "
        f"trailing gaps: {trailing:.1%}"
    )

print("\nCategory-level trailing inactivity:")
for category, group in df.groupby("category"):
    print(
        f"{category}: "
        f"{len(group)} series, "
        f"{(group['trailing_zero_days'] >= 90).mean():.1%} "
        "have at least 90 trailing zero-sales days"
    )

out = ROOT / "reports" / "inactivity_threshold_analysis.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
