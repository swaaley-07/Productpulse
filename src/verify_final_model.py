
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "sales_longevity_final_candidate.joblib"
DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_with_prices.parquet"

print("Loading saved model...")
model = joblib.load(MODEL_PATH)

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
df["prediction_date"] = pd.to_datetime(df["prediction_date"])

numeric_features = [
    "units_last_28d",
    "units_last_90d",
    "active_days_last_28d",
    "active_days_last_90d",
    "sales_change_recent_vs_previous_28d",
    "price_at_prediction",
    "price_change_4w_pct",
]

categorical_features = ["category"]
features = numeric_features + categorical_features

test_date = df["prediction_date"].max()
sample = df[df["prediction_date"] == test_date].head(10).copy()

X_sample = sample[features]
predictions = model.predict(X_sample)
probabilities = model.predict_proba(X_sample)[:, 1]

results = sample[
    ["item_id", "store_id", "prediction_date", "future_units_next_90d"]
].copy()

results["predicted_class"] = predictions
results["predicted_low_sales_probability"] = probabilities

results["predicted_class"] = results["predicted_class"].map({
    0: "20+ future units",
    1: "Fewer than 20 future units",
})

print("\nSAMPLE PREDICTIONS")
print(results.to_string(index=False))

output_path = ROOT / "reports" / "sample_predictions.csv"
results.to_csv(output_path, index=False)

print("\nPASS: Saved model loaded and generated predictions.")
print("Saved sample predictions to:", output_path)
