
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

INPUT_PATH = PROCESSED / "sales_longevity_dataset.parquet"
OUTPUT_PATH = PROCESSED / "sales_longevity_with_prices.parquet"

print("Loading dataset and price data...")

df = pd.read_parquet(INPUT_PATH)
calendar = pd.read_csv(RAW / "calendar.csv")
prices = pd.read_csv(RAW / "sell_prices.csv")

df["prediction_date"] = pd.to_datetime(df["prediction_date"])
calendar["date"] = pd.to_datetime(calendar["date"])

# Map each prediction date to its M5 retail week.
date_to_week = (
    calendar[["date", "wm_yr_wk"]]
    .drop_duplicates("date")
)

df = df.merge(
    date_to_week,
    left_on="prediction_date",
    right_on="date",
    how="left",
    validate="many_to_one",
).drop(columns=["date"])

# Join the price for the prediction week.
current_prices = prices.rename(
    columns={"sell_price": "price_at_prediction"}
)

df = df.merge(
    current_prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left",
    validate="many_to_one",
)

# Create a chronological index for the retail weeks.
weeks = sorted(calendar["wm_yr_wk"].dropna().unique())
week_position = {week: i for i, week in enumerate(weeks)}

df["week_position"] = df["wm_yr_wk"].map(week_position)

# Prepare prices from four retail weeks earlier.
prior_prices = prices.copy()
prior_prices["week_position"] = prior_prices["wm_yr_wk"].map(
    week_position
)

prior_prices["target_position"] = (
    prior_prices["week_position"] + 4
)

prior_prices = prior_prices.rename(
    columns={"sell_price": "price_4_weeks_prior"}
)

prior_prices = prior_prices[
    [
        "store_id",
        "item_id",
        "target_position",
        "price_4_weeks_prior",
    ]
]

# Join earlier prices to the prediction-week observation.
df["target_position"] = df["week_position"]

df = df.merge(
    prior_prices,
    on=["store_id", "item_id", "target_position"],
    how="left",
    validate="many_to_one",
)

# Calculate percentage price change.
df["price_change_4w_pct"] = (
    (
        df["price_at_prediction"]
        - df["price_4_weeks_prior"]
    )
    / df["price_4_weeks_prior"].where(
        df["price_4_weeks_prior"] != 0
    )
)

# Remove temporary columns.
df = df.drop(
    columns=[
        "wm_yr_wk",
        "week_position",
        "target_position",
    ]
)

# Save as a separate dataset. Keep the original unchanged.
df.to_parquet(OUTPUT_PATH, index=False)

print("\nPRICE FEATURE DATASET CREATED")
print("=" * 50)
print("Rows:", len(df))
print("Columns:", len(df.columns))

print(
    "Current price available:",
    round(df["price_at_prediction"].notna().mean() * 100, 2),
    "%"
)

print(
    "Earlier price available:",
    round(df["price_4_weeks_prior"].notna().mean() * 100, 2),
    "%"
)

print(
    "Missing current prices:",
    int(df["price_at_prediction"].isna().sum())
)

print(
    "Missing earlier prices:",
    int(df["price_4_weeks_prior"].isna().sum())
)

print("Saved to:", OUTPUT_PATH)

