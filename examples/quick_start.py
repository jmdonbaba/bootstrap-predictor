"""
BootstrapPredictor Quick Start — synthetic time series data.

Scenario: Predict next year's sales for 10 stores using historical data.
Demonstrates: train → CI prediction → sensitivity → CV.
"""

import numpy as np
import pandas as pd
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")

from bootstrap_predictor import BootstrapPredictor

# ================================================================
# 1. Generate synthetic panel data
# ================================================================
print("=" * 60)
print("1. Generating synthetic panel data (10 stores x 5 years)")
print("=" * 60)

np.random.seed(42)
n_entities = 10
n_years = 5
rows = []

for store_id in range(n_entities):
    base = np.random.uniform(50, 200)
    trend = np.random.uniform(-5, 10)
    for t in range(n_years):
        year = 2020 + t
        sales = base + trend * t + np.random.normal(0, 5)
        customers = sales * 0.3 + np.random.normal(0, 10)
        rows.append({
            "store": f"S{store_id:02d}",
            "year": year,
            "sales": max(0, sales),
            "customers": max(0, customers),
        })

df = pd.DataFrame(rows)
print(f"Shape: {df.shape}")
print(df.head(6).to_string())
print()

# ================================================================
# 2. Train + predict with CI
# ================================================================
print("=" * 60)
print("2. Train BootstrapPredictor + predict with 95% CI")
print("=" * 60)

# Features and target
X = pd.get_dummies(df[["customers", "year", "store"]], columns=["store"])
y = df[["sales"]]

# Split: first 4 years train, last year test
train_mask = df["year"] < 2024
test_mask = df["year"] == 2024

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

bp = BootstrapPredictor(random_state=42)
bp.fit(X_train, y_train.values.ravel())

result = bp.predict_with_ci(X_test, n_bootstrap=50)

print(f"\nPredictions for {len(result.point_estimate)} stores (2024):")
print(f"{'Store':<8}{'Actual':>8}{'Predicted':>10}{'CI_Lower':>10}{'CI_Upper':>10}")
print("-" * 48)
for i, (_, row) in enumerate(df[test_mask].iterrows()):
    print(f"{row['store']:<8}{row['sales']:>8.1f}{result.point_estimate[i,0]:>10.1f}"
          f"{result.ci_lower[i,0]:>10.1f}{result.ci_upper[i,0]:>10.1f}")
print()

# ================================================================
# 3. Feature importance
# ================================================================
print("=" * 60)
print("3. Feature Importance")
print("=" * 60)
imp = bp.feature_importance()
print(imp.head(8).to_string())
print()

# ================================================================
# 4. Sensitivity analysis
# ================================================================
print("=" * 60)
print("4. Sensitivity Analysis (perturb 'customers' by 1-20%)")
print("=" * 60)
sens = bp.sensitivity(X_test, feature="customers")
print(sens.head(10).to_string())
print()

# ================================================================
# 5. Time series CV
# ================================================================
print("=" * 60)
print("5. Time Series Cross-Validation")
print("=" * 60)
cv = bp.time_series_cv(X_train, y_train.values.ravel(), n_splits=3)
print(cv.to_string())
print()

# ================================================================
# 6. Report
# ================================================================
bp.summary()

# ================================================================
# 7. Plot
# ================================================================
fig = bp.plot(result=result, sens_df=sens)
fig.savefig("bp_report.png", dpi=150, bbox_inches="tight")
print("\nReport figure saved to bp_report.png")
