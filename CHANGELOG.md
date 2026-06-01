# Changelog

All notable changes to bootstrap-predictor will be documented in this file.

## [0.1.0] — 2026-06-02

### Added
- `BootstrapPredictor` class wrapping any sklearn-compatible regressor
- Bootstrap confidence intervals via `predict_with_ci()`
- Model-native feature importance (`feature_importance()`)
- Permutation feature importance (`permutation_importance()`)
- Single-feature sensitivity analysis (`sensitivity()`)
- Expanding-window time series cross-validation (`time_series_cv()`)
- Console summary report (`summary()`)
- Multi-panel diagnostic plots (`plot()`)
- `PredictionResult` dataclass with `to_dataframe()` export
- Optional `store_samples` parameter to control memory usage
- Full type annotations (PEP 561)
- CI pipeline (GitHub Actions: test matrix + ruff lint)
- Bilingual README (English + Chinese)
