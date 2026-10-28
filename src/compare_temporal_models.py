
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_with_prices.parquet"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df["prediction_date"] = pd.to_datetime(df["prediction_date"])
df["target"] = (df["future_units_next_90d"] < 20).astype(int)

sales_features = [
    "units_last_28d",
    "units_last_90d",
    "active_days_last_28d",
    "active_days_last_90d",
    "sales_change_recent_vs_previous_28d",
]
price_features = sales_features + [
    "price_at_prediction",
    "price_change_4w_pct",
]
categorical_features = ["category"]

dates = sorted(df["prediction_date"].dropna().unique())
test_dates = dates[2:]
results = []

for test_date in test_dates:
    train = df[df["prediction_date"] < test_date]
    test = df[df["prediction_date"] == test_date]

    for model_name, numeric_features in [
        ("Random Forest - Sales only", sales_features),
        ("Random Forest - Sales + prices", price_features),
    ]:
        features = numeric_features + categorical_features

        preprocessor = ColumnTransformer([
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
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

        X_train = train[features]
        y_train = train["target"]
        X_test = test[features]
        y_test = test["target"]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append({
            "test_date": pd.Timestamp(test_date).date().isoformat(),
            "model": model_name,
            "train_rows": len(train),
            "test_rows": len(test),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "pr_auc": average_precision_score(y_test, probabilities),
            "roc_auc": roc_auc_score(y_test, probabilities),
        })

    print("Completed:", pd.Timestamp(test_date).date())

results_df = pd.DataFrame(results)
results_path = REPORT_DIR / "temporal_model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\nAVERAGE PERFORMANCE BY MODEL")
metrics = [
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
]
print(
    results_df.groupby("model")[metrics]
    .mean()
    .round(4)
    .to_string()
)

print("\nRESULTS BY TEST DATE")
print(
    results_df[
        ["test_date", "model", "f1", "pr_auc", "roc_auc"]
    ].round(4).to_string(index=False)
)

print("\nSaved report to:", results_path)
