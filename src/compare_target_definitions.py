
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_dataset.parquet"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)

thresholds = [0, 4, 10, 20]

results = []

for threshold in thresholds:
    if threshold == 0:
        target = df["future_units_next_90d"] == 0
        label = "Zero future sales"
    else:
        target = df["future_units_next_90d"] < threshold
        label = f"Fewer than {threshold} future units"

    results.append({
        "definition": label,
        "positive_cases": int(target.sum()),
        "negative_cases": int((~target).sum()),
        "positive_percentage": round(target.mean() * 100, 2),
        "negative_percentage": round((~target).mean() * 100, 2),
    })

summary = pd.DataFrame(results)

print("\nTARGET DEFINITION COMPARISON")
print("=" * 75)
print(summary.to_string(index=False))

output = REPORTS / "target_definition_comparison.csv"
summary.to_csv(output, index=False)

print(f"\nSaved report to: {output}")
