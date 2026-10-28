
from pathlib import Path
import pandas as pd

# Project paths: this script is inside src/src/
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data" / "processed" / "sales_longevity_dataset.parquet"

# Load data
sales = pd.read_csv(
    RAW / "sales_train_evaluation.csv",
    nrows=1
)
calendar = pd.read_csv(RAW / "calendar.csv")
df = pd.read_parquet(DATA)

# Match sales day columns to calendar dates
day_columns = [
    column for column in sales.columns
    if column.startswith("d_")
]

sales_days = calendar[
    calendar["d"].isin(day_columns)
].copy()

sales_days["date"] = pd.to_datetime(sales_days["date"])
sales_days = sales_days.sort_values("date")

print("\nSALES DAY ALIGNMENT")
print("=" * 55)
print(
    "First sales day:",
    sales_days.iloc[0]["d"],
    sales_days.iloc[0]["date"].date()
)
print(
    "Last sales day:",
    sales_days.iloc[-1]["d"],
    sales_days.iloc[-1]["date"].date()
)
print("Number of sales days:", len(sales_days))

# Check past and future windows
print("\nPREDICTION DATE CHECK")
print("=" * 55)

for date in sorted(pd.to_datetime(df["prediction_date"].unique())):
    past_start = date - pd.Timedelta(days=89, unit="days")
    future_end = date + pd.Timedelta(days=90, unit="days")

    print(
        f"Prediction: {date.date()} | "
        f"Past window: {past_start.date()} to {date.date()} | "
        f"Future window ends: {future_end.date()}"
    )

# Check prediction dates against the sales calendar
print("\nCHECK PREDICTION DATES EXIST IN CALENDAR")
print("=" * 55)

calendar_dates = set(sales_days["date"])

for date in sorted(pd.to_datetime(df["prediction_date"].unique())):
    print(date.date(), "->", date in calendar_dates)

# Display target counts
print("\nCHECK TARGET COUNTS")
print("=" * 55)

print("Total rows:", len(df))
print(
    "Target under 20 units:",
    (df["future_units_next_90d"] < 20).sum()
)
print(
    "Target 20+ units:",
    (df["future_units_next_90d"] >= 20).sum()
)

print("\nValidation checks completed.")
