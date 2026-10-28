
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_with_prices.parquet"
REPORT_DIR = ROOT / "reports"
VIS_DIR = ROOT / "visualisation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
df["prediction_date"] = pd.to_datetime(df["prediction_date"])
df["target"] = (df["future_units_next_90d"] < 20).astype(int)

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

# Train on earlier dates and evaluate on the latest date.
dates = sorted(df["prediction_date"].dropna().unique())
test_date = dates[-1]

train = df[df["prediction_date"] < test_date]
test = df[df["prediction_date"] == test_date]

X_train = train[features]
y_train = train["target"]
X_test = test[features]
y_test = test["target"]

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_features,
    ),
    (
        "category",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_features,
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

print("Training model...")
model.fit(X_train, y_train)

print("Calculating permutation importance...")
importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

importance_df = pd.DataFrame({
    "feature": features,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std,
}).sort_values("importance_mean", ascending=False)

output_csv = REPORT_DIR / "feature_importance.csv"
importance_df.to_csv(output_csv, index=False)

print("\nFEATURE IMPORTANCE")
print(importance_df.round(5).to_string(index=False))

# Save a chart.
plot_df = importance_df.sort_values("importance_mean")

plt.figure(figsize=(10, 6))
plt.barh(plot_df["feature"], plot_df["importance_mean"])
plt.xlabel("Decrease in PR-AUC when feature is shuffled")
plt.title("Permutation Feature Importance")
plt.tight_layout()

chart_path = VIS_DIR / "feature_importance.png"
plt.savefig(chart_path, dpi=200)
plt.close()

print("\nSaved table to:", output_csv)
print("Saved chart to:", chart_path)
