
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import joblib

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_with_prices.parquet"
MODEL_PATH = ROOT / "models" / "sales_longevity_final_candidate.joblib"
OUTPUT_DIR = ROOT / "visualisation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "units_last_28d",
    "units_last_90d",
    "active_days_last_28d",
    "active_days_last_90d",
    "sales_change_recent_vs_previous_28d",
    "price_at_prediction",
    "price_change_4w_pct",
    "category",
]

print("Loading dataset and model...")
df = pd.read_parquet(DATA_PATH)
df["prediction_date"] = pd.to_datetime(df["prediction_date"])
model = joblib.load(MODEL_PATH)

# Match the final model's held-out evaluation date.
test_date = df["prediction_date"].max()
test = df[df["prediction_date"] == test_date].copy()

X_test = test[FEATURES]
y_test = (test["future_units_next_90d"] < 20).astype(int)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("Held-out date:", test_date.date())
print("Test rows:", len(test))

# 1. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["20+ future units", "Fewer than 20 units"],
    values_format=",",
)
plt.title("ProductPulse: Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

# 2. ROC curve
RocCurveDisplay.from_predictions(y_test, probabilities)
plt.title("ProductPulse: ROC Curve")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
plt.close()

# 3. Precision-recall curve
PrecisionRecallDisplay.from_predictions(y_test, probabilities)
plt.title("ProductPulse: Precision-Recall Curve")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=200)
plt.close()

# 4. Feature importance chart, if available
importance_path = OUTPUT_DIR / "feature_importance.png"

if importance_path.exists():
    print("Existing feature importance chart found.")
else:
    print(
        "Feature importance chart not found. "
        "Run src/feature_importance.py to generate it."
    )

print("\nCharts saved to:", OUTPUT_DIR)
print("Created:")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- precision_recall_curve.png")
