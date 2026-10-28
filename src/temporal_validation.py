
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT / "data" / "processed" / "sales_longevity_with_prices.parquet"
)
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

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

# Each split trains on earlier dates and tests on a later date.
dates = sorted(df["prediction_date"].dropna().unique())

if len(dates) < 4:
    raise ValueError("Not enough prediction dates for temporal validation.")

# Use expanding training windows, testing each later date individually.
# The first two dates provide initial training data.
test_dates = dates[2:]

results = []

for test_date in test_dates:
    train = df[df["prediction_date"] < test_date].copy()
    test = df[df["prediction_date"] == test_date].copy()

    if train.empty or test.empty:
        continue

    X_train = train[features]
    y_train = train["target"]
    X_test = test[features]
    y_test = test["target"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
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

    print(
        f"\nTesting {pd.Timestamp(test_date).date()} "
        f"| Train rows: {len(train)} | Test rows: {len(test)}"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "test_date": pd.Timestamp(test_date).date().isoformat(),
        "train_rows": len(train),
        "test_rows": len(test),
        "positive_rate": y_test.mean(),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": (
            roc_auc_score(y_test, probabilities)
            if y_test.nunique() == 2 else float("nan")
        ),
    })

results_df = pd.DataFrame(results)

output_path = REPORT_DIR / "temporal_validation.csv"
results_df.to_csv(output_path, index=False)

print("\nTEMPORAL VALIDATION RESULTS")
print("=" * 90)
print(results_df.round(4).to_string(index=False))

if not results_df.empty:
    print("\nAVERAGE METRICS ACROSS TEST DATES")
    metric_columns = [
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
    ]
    print(results_df[metric_columns].mean().round(4).to_string())

print("\nSaved report to:", output_path)
