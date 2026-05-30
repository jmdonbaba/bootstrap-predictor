# bootstrap-predictor

ML regression with **bootstrap confidence intervals** — turn any sklearn model into an uncertainty-aware forecaster.

Zero extra dependencies: `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## Install

```bash
pip install bootstrap-predictor
# or
git clone https://github.com/YOUR_USERNAME/bootstrap-predictor.git
cd bootstrap-predictor
pip install -e .
```

## Quick Start

```python
from bootstrap_predictor import BootstrapPredictor

bp = BootstrapPredictor()
bp.fit(X_train, y_train)

# Predict with 95% confidence intervals
result = bp.predict_with_ci(X_new, n_bootstrap=100)
# result.point_estimate  — shape (n, n_targets)
# result.ci_lower        — lower bound
# result.ci_upper        — upper bound

# Sensitivity analysis
sens = bp.sensitivity(X, feature="price", pct_range=(0.01, 0.20))
# → DataFrame: pct, mean_abs_change

# Time series cross-validation
cv = bp.time_series_cv(X, y, n_splits=5)
# → DataFrame: fold, target, R2, RMSE, MAE

# Report & plot
bp.summary()
bp.plot(result=result, sens_df=sens)
```

## Why Bootstrap CI?

Standard ML models give point predictions — no measure of uncertainty. Bootstrap resampling quantifies model variance:

1. Resample training data with replacement (n times)
2. Train a model on each bootstrap sample
3. Predict on new data with all models
4. Build percentile-based confidence intervals from the ensemble

This captures **model uncertainty** — if the training data changed slightly, how much would predictions shift?

## API

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train model, store data for bootstrap |
| `predict(X)` | Point predictions |
| `predict_with_ci(X, n_bootstrap=100)` | Predict with bootstrap CIs |
| `feature_importance()` | Feature importance DataFrame |
| `permutation_importance(X, y)` | Permutation-based importance |
| `sensitivity(X, feature, pct_range)` | Perturbation sensitivity |
| `time_series_cv(X, y, n_splits=5)` | Expanding-window CV |
| `summary()` | Print full report |
| `plot(result, sens_df)` | Diagnostic figures |

## Dependencies

- Python >= 3.8
- numpy, pandas, scikit-learn, matplotlib

## License

MIT
