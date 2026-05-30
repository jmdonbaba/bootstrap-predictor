"""
BootstrapPredictor — ML regression with bootstrap uncertainty quantification.

Workflow
--------
1. fit(X, y)                              → train model
2. predict_with_ci(X_new, n_bootstrap)    → point estimates + CIs
3. sensitivity(X, feature)                → feature perturbation analysis
4. time_series_cv(X, y)                   → temporal cross-validation
5. summary() / plot()                     → report & diagnostics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dataclasses import dataclass, field

from .viz import plot_predictor_report


@dataclass
class PredictionResult:
    """Result of predict_with_ci()"""
    point_estimate: np.ndarray       # (n_samples, n_targets)
    ci_lower: np.ndarray             # (n_samples, n_targets)
    ci_upper: np.ndarray             # (n_samples, n_targets)
    ci_level: float = 0.95
    bootstrap_samples: np.ndarray = field(default=None, repr=False)

    def __repr__(self):
        return (f"PredictionResult(n={len(self.point_estimate)}, "
                f"ci={self.ci_level*100:.0f}%, "
                f"targets={self.point_estimate.shape[1]})")

    def to_dataframe(self, sample_index=None, target_names=None):
        """Export to DataFrame with columns: point, ci_lower, ci_upper per target"""
        n = len(self.point_estimate)
        if sample_index is None:
            sample_index = range(n)
        if target_names is None:
            target_names = [f"y{i}" for i in range(self.point_estimate.shape[1])]
        dfs = []
        for j, name in enumerate(target_names):
            dfs.append(pd.DataFrame({
                "sample": sample_index,
                "target": name,
                "point": self.point_estimate[:, j],
                "ci_lower": self.ci_lower[:, j],
                "ci_upper": self.ci_upper[:, j],
            }))
        return pd.concat(dfs, ignore_index=True)


class BootstrapPredictor:
    """ML regressor with bootstrap confidence intervals.

    Parameters
    ----------
    estimator : sklearn regressor, default=RandomForestRegressor()
        Any sklearn-compatible regressor. Wrapped models (multi-output)
        are supported.
    random_state : int, default=42
    n_jobs : int, default=-1
        Parallel jobs for bootstrap training. -1 = all cores.
    """

    def __init__(self, estimator=None, random_state=42, n_jobs=-1):
        if estimator is None:
            estimator = RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                random_state=random_state, n_jobs=n_jobs,
            )
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.feature_names_ = None
        self.target_names_ = None
        self._X_train = None
        self._y_train = None
        self._is_fitted = False
        self.cv_results_ = None

    # ================================================================
    # Step 1: Train
    # ================================================================
    def fit(self, X, y):
        """Train the estimator and store data for bootstrap.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        self
        """
        self.feature_names_ = self._extract_names(X)
        self.target_names_ = self._extract_names(y, prefix="y")

        self._X_train = self._to_array(X)
        self._y_train = self._to_array(y)
        if self._y_train.ndim == 1:
            self._y_train = self._y_train.reshape(-1, 1)

        self.estimator.fit(self._X_train, self._y_train)
        self._is_fitted = True
        return self

    # ================================================================
    # Step 2: Predict with CI
    # ================================================================
    def predict(self, X):
        """Point prediction only (no CI)."""
        self._check_fitted()
        X = self._to_array(X)
        pred = self.estimator.predict(X)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def predict_with_ci(self, X, n_bootstrap=100, alpha=0.05):
        """Predict with bootstrap confidence intervals.

        Trains n_bootstrap models on resampled training data,
        then computes percentile-based CIs from the ensemble.

        Requires fit() to have been called first (stores training data).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        n_bootstrap : int, default=100
        alpha : float, default=0.05

        Returns
        -------
        PredictionResult
        """
        self._check_fitted()
        X_new = self._to_array(X)
        X_tr = self._X_train
        y_tr = self._y_train
        n_train = len(X_tr)
        n_new = len(X_new)
        n_targets = y_tr.shape[1]

        point = self.estimator.predict(X_new)
        if point.ndim == 1:
            point = point.reshape(-1, 1)

        boot_samples = np.zeros((n_bootstrap, n_new, n_targets))
        rng = np.random.RandomState(self.random_state)
        y_flat = y_tr.ravel() if y_tr.shape[1] == 1 else y_tr

        print(f"Bootstrapping: {n_bootstrap} iterations ", end="", flush=True)
        for i in range(n_bootstrap):
            idx = rng.choice(n_train, size=n_train, replace=True)
            Xb, yb = X_tr[idx], y_flat[idx]

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from sklearn.ensemble import RandomForestRegressor as RF
                m = RF(n_estimators=100, max_depth=12, min_samples_leaf=5,
                       random_state=self.random_state + i, n_jobs=self.n_jobs)
                m.fit(Xb, yb)
                pb = m.predict(X_new)
            if pb.ndim == 1:
                pb = pb.reshape(-1, 1)
            boot_samples[i] = pb

            if (i + 1) % 20 == 0:
                print(".", end="", flush=True)
        print(" done")

        lo = np.percentile(boot_samples, alpha / 2 * 100, axis=0)
        hi = np.percentile(boot_samples, (1 - alpha / 2) * 100, axis=0)

        return PredictionResult(
            point_estimate=point,
            ci_lower=lo,
            ci_upper=hi,
            ci_level=1 - alpha,
            bootstrap_samples=boot_samples,
        )

    # ================================================================
    # Step 3: Feature Importance
    # ================================================================
    def feature_importance(self):
        """Return feature importance DataFrame.

        For tree-based models: mean decrease in impurity.
        For linear models: absolute coefficient values.
        """
        self._check_fitted()
        names = self.feature_names_ or [f"X{i}" for i in range(self._n_features())]

        if hasattr(self.estimator, "feature_importances_"):
            imp = self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):
            imp = np.abs(self.estimator.coef_).flatten()
        else:
            raise AttributeError(
                "Estimator has no feature_importances_ or coef_. "
                "Use permutation_importance() instead."
            )

        return pd.DataFrame({
            "feature": names[:len(imp)],
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def permutation_importance(self, X, y, n_repeats=5):
        """Permutation-based feature importance (works for any estimator)."""
        self._check_fitted()
        from sklearn.inspection import permutation_importance as pi
        names = self.feature_names_ or [f"X{i}" for i in range(self._to_array(X).shape[1])]
        r = pi(self.estimator, self._to_array(X), y,
               n_repeats=n_repeats, random_state=self.random_state, n_jobs=self.n_jobs)
        return pd.DataFrame({
            "feature": names,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # ================================================================
    # Step 4: Sensitivity Analysis
    # ================================================================
    def sensitivity(self, X, feature, pct_range=(0.01, 0.20), n_steps=20):
        """Sensitivity analysis: how predictions change when perturbing a feature.

        Parameters
        ----------
        X : array-like
            Reference data to perturb.
        feature : str or int
            Feature name (str for DataFrame) or column index.
        pct_range : tuple (min_pct, max_pct)
            Perturbation range as fraction (0.01 to 0.20 = 1% to 20%).
        n_steps : int, default=20

        Returns
        -------
        pd.DataFrame with columns: pct, mean_abs_change
        """
        self._check_fitted()
        X_ref = self._to_array(X)
        idx = self._resolve_feature_index(feature)

        base_pred = self.estimator.predict(X_ref)
        if base_pred.ndim == 1:
            base_pred = base_pred.reshape(-1, 1)

        rows = []
        for pct in np.linspace(pct_range[0], pct_range[1], n_steps):
            Xp = X_ref.copy()
            Xp[:, idx] *= (1 + pct)
            pred_p = self.estimator.predict(Xp)
            if pred_p.ndim == 1:
                pred_p = pred_p.reshape(-1, 1)
            change = np.mean(np.abs(pred_p - base_pred))
            rows.append({"pct": round(pct * 100, 1), "mean_abs_change": change})

        return pd.DataFrame(rows)

    # ================================================================
    # Step 5: Time Series CV
    # ================================================================
    def time_series_cv(self, X, y, n_splits=5):
        """Time series cross-validation (expanding window).

        Returns
        -------
        pd.DataFrame with columns: fold, target, R2, RMSE, MAE
        """
        # Trains from scratch per fold — does not use the stored estimator
        X = self._to_array(X)
        y = self._to_array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        tnames = self.target_names_ or [f"y{i}" for i in range(y.shape[1])]
        results = []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            m = RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_leaf=5,
                random_state=self.random_state, n_jobs=self.n_jobs,
            )
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            for j, name in enumerate(tnames):
                results.append({
                    "fold": fold + 1,
                    "target": name,
                    "R2": r2_score(y_te[:, j], y_pred[:, j]),
                    "RMSE": np.sqrt(mean_squared_error(y_te[:, j], y_pred[:, j])),
                    "MAE": mean_absolute_error(y_te[:, j], y_pred[:, j]),
                })

        self.cv_results_ = pd.DataFrame(results)
        return self.cv_results_

    # ================================================================
    # Step 6: Summary
    # ================================================================
    def summary(self):
        """Print full report: feature importance + CV summary."""
        self._check_fitted()
        print("=" * 60)
        print("BootstrapPredictor Report")
        print("=" * 60)

        print("\n[Feature Importance]")
        try:
            imp = self.feature_importance()
            for _, r in imp.iterrows():
                print(f"  {r['feature']:<20s} {r['importance']:.4f}")
        except AttributeError:
            print("  (not available for this estimator)")

        if self.cv_results_ is not None:
            print("\n[Time Series CV Summary]")
            cv_s = self.cv_results_.groupby("target")[["R2", "RMSE", "MAE"]].mean()
            print(cv_s.to_string())

        print("=" * 60)

    # ================================================================
    # Step 7: Plot
    # ================================================================
    def plot(self, result=None, sens_df=None):
        """Generate diagnostic plots.

        Parameters
        ----------
        result : PredictionResult, optional
            From predict_with_ci(). If provided, shows CI plot.
        sens_df : pd.DataFrame, optional
            From sensitivity(). If provided, shows sensitivity curve.
        """
        self._check_fitted()
        return plot_predictor_report(
            predictor=self,
            result=result,
            sens_df=sens_df,
        )

    # ================================================================
    # Internals
    # ================================================================
    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call .fit() or .fit_predict_with_ci() first")

    @staticmethod
    def _to_array(x):
        if hasattr(x, "values"):
            x = x.values
        return np.asarray(x, dtype=float)

    def _extract_names(self, obj, prefix="X"):
        if hasattr(obj, "columns"):
            return list(obj.columns)
        if hasattr(obj, "name") and obj.name is not None:
            return [obj.name]
        return None

    def _n_features(self):
        if hasattr(self.estimator, "n_features_in_"):
            return self.estimator.n_features_in_
        if hasattr(self.estimator, "coef_"):
            return self.estimator.coef_.shape[1]
        return 0

    def _resolve_feature_index(self, feature):
        if isinstance(feature, str) and self.feature_names_:
            return self.feature_names_.index(feature)
        return int(feature)
