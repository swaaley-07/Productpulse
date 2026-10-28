
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_with_prices.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
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

dates = sorted(df["prediction_date"].dropna().unique())
test_date = dates[-1]

train = df[df["prediction_date"] < test_date].copy()
test = df[df["prediction_date"] == test_date].copy()

X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]

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

print("Training rows:", len(train))
print("Test rows:", len(test))
print("Held-out date:", pd.Timestamp(test_date).date())

print("\nTraining final candidate model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

report = classification_report(
    y_test,
    predictions,
    target_names=["20+ future units", "Fewer than 20 future units"],
    zero_division=0,
)

metrics = {
    "held_out_date": str(pd.Timestamp(test_date).date()),
    "training_rows": len(train),
    "test_rows": len(test),
    "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
    "f1": f1_score(y_test, predictions, zero_division=0),
    "pr_auc": average_precision_score(y_test, probabilities),
    "roc_auc": roc_auc_score(y_test, probabilities),
}

print("\nFINAL EVALUATION")
print(report)
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print("\nSummary metrics:")
for name, value in metrics.items():
    print(f"{name}: {value}")

model_path = MODEL_DIR / "sales_longevity_final_candidate.joblib"
joblib.dump(model, model_path)

report_path = REPORT_DIR / "final_candidate_evaluation.txt"
with open(report_path, "w", encoding="utf-8") as file:
    file.write("FINAL CANDIDATE EVALUATION\n\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(str(confusion_matrix(y_test, predictions)))
    file.write("\n\nSummary metrics:\n")
    for name, value in metrics.items():
        file.write(f"{name}: {value}\n")

print("\nSaved model to:", model_path)
print("Saved evaluation to:", report_path)
