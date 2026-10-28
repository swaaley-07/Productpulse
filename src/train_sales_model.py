
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    balanced_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_dataset.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
df["prediction_date"] = pd.to_datetime(df["prediction_date"])

# Target: 1 means fewer than 20 units in the next 90 days.
df["low_future_sales"] = (
    df["future_units_next_90d"] < 20
).astype(int)

# Features available at prediction time only.
numeric_features = [
    "units_last_28d",
    "units_last_90d",
    "active_days_last_28d",
    "active_days_last_90d",
    "sales_change_recent_vs_previous_28d",
]

categorical_features = ["category"]

feature_columns = numeric_features + categorical_features

# Hold out the latest two prediction dates for testing.
dates = sorted(df["prediction_date"].unique())

if len(dates) < 3:
    raise ValueError("Need at least three prediction dates for a time-based split.")

test_dates = dates[-2:]
train_dates = dates[:-2]

train_df = df[df["prediction_date"].isin(train_dates)].copy()
test_df = df[df["prediction_date"].isin(test_dates)].copy()

X_train = train_df[feature_columns]
y_train = train_df["low_future_sales"]

X_test = test_df[feature_columns]
y_test = test_df["low_future_sales"]

print("\nTIME-BASED SPLIT")
print("Training dates:", train_dates[0], "to", train_dates[-1])
print("Testing dates:", test_dates[0], "to", test_dates[-1])
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))
print("Training positive rate:", round(y_train.mean(), 4))
print("Testing positive rate:", round(y_test.mean(), 4))

# Preprocessing and baseline classifier.
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
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    target_names=["20+ future units", "Fewer than 20 units"],
    zero_division=0,
))

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

print("\nSUMMARY METRICS")
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, predictions), 4
))
print("PR-AUC (average precision):", round(
    average_precision_score(y_test, probabilities), 4
))

if y_test.nunique() == 2:
    print("ROC-AUC:", round(
        roc_auc_score(y_test, probabilities), 4
    ))
else:
    print("ROC-AUC unavailable: test set contains only one class.")

# Save model and evaluation report.
model_path = MODEL_DIR / "sales_longevity_baseline.joblib"
joblib.dump(model, model_path)

report_path = REPORT_DIR / "baseline_evaluation.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Sales Longevity Baseline Evaluation\n")
    file.write("=" * 40 + "\n")
    file.write(f"Training dates: {train_dates[0]} to {train_dates[-1]}\n")
    file.write(f"Testing dates: {test_dates[0]} to {test_dates[-1]}\n")
    file.write(f"Training rows: {len(train_df)}\n")
    file.write(f"Testing rows: {len(test_df)}\n\n")
    file.write(classification_report(
        y_test,
        predictions,
        target_names=["20+ future units", "Fewer than 20 units"],
        zero_division=0,
    ))
    file.write("\nConfusion matrix:\n")
    file.write(str(confusion_matrix(y_test, predictions)))
    file.write(
        f"\n\nBalanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}\n"
    )
    file.write(
        f"PR-AUC (average precision): "
        f"{average_precision_score(y_test, probabilities):.4f}\n"
    )
    if y_test.nunique() == 2:
        file.write(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}\n")

print("\nSaved model to:", model_path)
print("Saved evaluation report to:", report_path)
print("\nBaseline training and evaluation complete.")
