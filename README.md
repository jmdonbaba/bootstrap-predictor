# bootstrap-predictor · 带置信区间的 ML 预测框架

[English](#english) | [中文](#中文)

---

## English

Turn any sklearn model into an **uncertainty-aware forecaster** with bootstrap confidence intervals. Zero extra dependencies — `numpy, pandas, scikit-learn, matplotlib` only.

### Install

```bash
pip install bootstrap-predictor
# or from source
git clone https://github.com/jmdonbaba/bootstrap-predictor.git
cd bootstrap-predictor
pip install -e .
```

### Quick Start

```python
from bootstrap_predictor import BootstrapPredictor

bp = BootstrapPredictor()
bp.fit(X_train, y_train)

# Predict with 95% bootstrap confidence intervals
result = bp.predict_with_ci(X_new, n_bootstrap=100)
# result.point_estimate  — shape (n_samples, n_targets)
# result.ci_lower        — lower bound
# result.ci_upper        — upper bound
# result.to_dataframe()  — export to pandas DataFrame

# Sensitivity analysis: how predictions change when perturbing a feature
sens = bp.sensitivity(X, feature="price", pct_range=(0.01, 0.20))
# -> DataFrame with pct, mean_abs_change

# Time series cross-validation (expanding window)
cv = bp.time_series_cv(X, y, n_splits=5)
# -> DataFrame with fold, target, R2, RMSE, MAE

# Report & diagnostic plots
bp.summary()
bp.plot(result=result, sens_df=sens)
```

### Why Bootstrap CI?

Standard ML models give point predictions with no measure of uncertainty. Bootstrap resampling quantifies model variance:

1. Resample training data with replacement (n times)
2. Train a model on each bootstrap sample
3. Predict on new data with all models
4. Build percentile-based confidence intervals from the ensemble

This captures **model uncertainty** — if the training data changed slightly, how much would predictions shift?

### API

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Train model, store data for bootstrap |
| `predict(X)` | Point predictions |
| `predict_with_ci(X, n_bootstrap=100)` | Predict with bootstrap CIs |
| `feature_importance()` | Feature importance DataFrame |
| `permutation_importance(X, y)` | Permutation-based importance (any estimator) |
| `sensitivity(X, feature, pct_range)` | Perturb feature & measure prediction change |
| `time_series_cv(X, y, n_splits=5)` | Expanding-window cross-validation |
| `summary()` | Print full report |
| `plot(result, sens_df)` | Diagnostic figures |

### Dependencies

- Python ≥ 3.8
- numpy ≥ 1.20, pandas ≥ 1.3, scikit-learn ≥ 1.0, matplotlib ≥ 3.4

### License

MIT

---

## 中文

将任意 sklearn 模型升级为**带置信区间的预测器**。通过 Bootstrap 重采样量化模型不确定性。零额外依赖 — 仅需 `numpy, pandas, scikit-learn, matplotlib`。

### 安装

```bash
pip install bootstrap-predictor
# 或从源码安装
git clone https://github.com/jmdonbaba/bootstrap-predictor.git
cd bootstrap-predictor
pip install -e .
```

### 快速开始

```python
from bootstrap_predictor import BootstrapPredictor

bp = BootstrapPredictor()
bp.fit(X_train, y_train)

# 预测 + 95% Bootstrap 置信区间
result = bp.predict_with_ci(X_new, n_bootstrap=100)
# result.point_estimate  — 点估计 (n_samples, n_targets)
# result.ci_lower        — 置信下界
# result.ci_upper        — 置信上界
# result.to_dataframe()  — 导出为 pandas DataFrame

# 敏感性分析: 扰动某个特征，观察预测变化
sens = bp.sensitivity(X, feature="price", pct_range=(0.01, 0.20))
# -> DataFrame: pct (扰动百分比), mean_abs_change (平均绝对变化)

# 时间序列交叉验证 (扩展窗口，保持时间顺序)
cv = bp.time_series_cv(X, y, n_splits=5)
# -> DataFrame: fold (折), target (目标变量), R2, RMSE, MAE

# 报告与诊断图
bp.summary()
bp.plot(result=result, sens_df=sens)
```

### Bootstrap CI 原理

标准 ML 模型只给出点预测，没有不确定性度量。Bootstrap 重采样量化了模型方差：

1. 从训练数据中有放回地抽样 n 次
2. 每次用不同样本训练一个模型
3. 用所有模型对新数据做预测
4. 从预测分布中构建百分位置信区间

这衡量的是**模型不确定性**——如果训练数据略有变化，预测结果会有多大波动？

### API 概览

| 方法 | 说明 |
|------|------|
| `fit(X, y)` | 训练模型并存储数据用于 Bootstrap |
| `predict(X)` | 点预测 |
| `predict_with_ci(X, n_bootstrap=100)` | 预测 + Bootstrap 置信区间 |
| `feature_importance()` | 特征重要性 (DataFrame) |
| `permutation_importance(X, y)` | 置换特征重要性 (适用于任意模型) |
| `sensitivity(X, feature, pct_range)` | 扰动特征 → 测量预测变化 |
| `time_series_cv(X, y, n_splits=5)` | 扩展窗口时间序列交叉验证 |
| `summary()` | 打印完整报告 |
| `plot(result, sens_df)` | 生成诊断图 |

### 环境依赖

- Python ≥ 3.8
- numpy ≥ 1.20, pandas ≥ 1.3, scikit-learn ≥ 1.0, matplotlib ≥ 3.4

### 开源协议

MIT
