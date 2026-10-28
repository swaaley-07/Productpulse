
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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

print("Loading price-feature dataset...")
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

dates = sorted(df["prediction_date"].unique())
train_dates = dates[:-2]
test_dates = dates[-2:]

train = df[df["prediction_date"].isin(train_dates)]
test = df[df["prediction_date"].isin(test_dates)]

X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]

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

models = {
    "Logistic Regression + prices": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest + prices": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

results = []

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Test dates:", test_dates)

for name, classifier in models.items():
    print(f"\nTraining {name}...")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_test, probabilities),
        "roc_auc": (
            roc_auc_score(y_test, probabilities)
            if y_test.nunique() == 2 else float("nan")
        ),
    })

results_df = pd.DataFrame(results).set_index("model")
output_path = REPORT_DIR / "price_model_comparison.csv"
results_df.to_csv(output_path)

print("\nPRICE MODEL COMPARISON")
print(results_df.round(4).to_string())
print("\nSaved results to:", output_path)
