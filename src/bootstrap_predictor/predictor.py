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

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from .viz import plot_predictor_report

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

__all__ = ["BootstrapPredictor", "PredictionResult"]


@dataclass
class PredictionResult:
    """Result of predict_with_ci()."""

    point_estimate: np.ndarray  # (n_samples, n_targets)
    ci_lower: np.ndarray  # (n_samples, n_targets)
    ci_upper: np.ndarray  # (n_samples, n_targets)
    ci_level: float = 0.95
    bootstrap_samples: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"PredictionResult(n={len(self.point_estimate)}, "
            f"ci={self.ci_level * 100:.0f}%, "
            f"targets={self.point_estimate.shape[1]})"
        )

    def to_dataframe(
        self,
        sample_index: Optional[Union[Sequence, np.ndarray]] = None,
        target_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Export to DataFrame with columns: sample, target, point, ci_lower, ci_upper.

        Parameters
        ----------
        sample_index : sequence, optional
            Labels for each sample (default: 0, 1, 2, ...).
        target_names : list of str, optional
            Names for each target column (default: y0, y1, ...).

        Returns
        -------
        pd.DataFrame
        """
        n = len(self.point_estimate)
        if sample_index is None:
            sample_index = list(range(n))
        else:
            sample_index = list(sample_index)
        if target_names is None:
            target_names = [f"y{i}" for i in range(self.point_estimate.shape[1])]
        dfs = []
        for j, name in enumerate(target_names):
            dfs.append(
                pd.DataFrame(
                    {
                        "sample": sample_index,
                        "target": name,
                        "point": self.point_estimate[:, j],
                        "ci_lower": self.ci_lower[:, j],
                        "ci_upper": self.ci_upper[:, j],
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)


class BootstrapPredictor:
    """ML regressor with bootstrap confidence intervals.

    Parameters
    ----------
    estimator : sklearn regressor, optional
        Any sklearn-compatible regressor. Defaults to a tuned
        ``RandomForestRegressor(n_estimators=200, max_depth=12,
        min_samples_leaf=5)``.
    random_state : int, default=42
        Seed for reproducibility across bootstrap and CV.
    n_jobs : int, default=-1
        Parallel jobs for the default estimator and permutation importance.
        ``-1`` means all cores.
    """

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if estimator is None:
            estimator = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        self.estimator: BaseEstimator = estimator
        self.random_state: int = random_state
        self.n_jobs: int = n_jobs

        self.feature_names_: Optional[list[str]] = None
        self.target_names_: Optional[list[str]] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._is_fitted: bool = False
        self.cv_results_: Optional[pd.DataFrame] = None
        self._n_features_in_: Optional[int] = None

    def __repr__(self) -> str:
        est_name = type(self.estimator).__name__
        state = "fitted" if self._is_fitted else "unfitted"
        return f"BootstrapPredictor(estimator={est_name}, {state})"

    # ================================================================
    # Step 1: Train
    # ================================================================
    def fit(self, X: "ArrayLike", y: "ArrayLike") -> "BootstrapPredictor":
        """Train the estimator and store data for bootstrap.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features. DataFrame column names are captured as
            ``feature_names_``.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Series name is captured as ``target_names_``.

        Returns
        -------
        self : BootstrapPredictor
        """
        self.feature_names_ = self._extract_names(X)
        self.target_names_ = self._extract_names(y)

        self._X_train = self._to_array(X)
        self._y_train = self._to_array(y)
        if self._y_train.ndim == 1:
            self._y_train = self._y_train.reshape(-1, 1)

        self._n_features_in_ = self._X_train.shape[1]

        self._validate_input(self._X_train, "X")
        self._validate_input(self._y_train, "y")

        self.estimator.fit(
            self._X_train,
            self._y_train.ravel()
            if self._y_train.shape[1] == 1
            else self._y_train,
        )
        self._is_fitted = True
        return self

    # ================================================================
    # Step 2: Predict with CI
    # ================================================================
    def predict(self, X: "ArrayLike") -> np.ndarray:
        """Point predictions without confidence intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        pred : np.ndarray of shape (n_samples, n_targets)
        """
        self._check_fitted()
        X_arr = self._to_array(X)
        pred = self.estimator.predict(X_arr)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def predict_with_ci(
        self,
        X: "ArrayLike",
        n_bootstrap: int = 100,
        alpha: float = 0.05,
        store_samples: bool = True,
    ) -> PredictionResult:
        """Predict with bootstrap confidence intervals.

        Trains ``n_bootstrap`` cloned models on resampled training data,
        then computes percentile-based CIs from the ensemble.

        Requires ``fit()`` to have been called first (stores training data).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.
        n_bootstrap : int, default=100
            Number of bootstrap iterations. Must be >= 1.
        alpha : float, default=0.05
            Significance level (0.05 → 95% CI). Must be in (0, 1).
        store_samples : bool, default=True
            If False, the ``bootstrap_samples`` attribute on the result is
            ``None``, saving memory when only CI bounds are needed.

        Returns
        -------
        PredictionResult
        """
        self._check_fitted()
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

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

        logger.info("Bootstrapping: %d iterations", n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.choice(n_train, size=n_train, replace=True)
            Xb = X_tr[idx]
            yb = y_tr[idx]
            if y_tr.shape[1] == 1:
                yb = yb.ravel()

            m = clone(self.estimator)
            try:
                m.set_params(random_state=self.random_state + i)
            except (ValueError, AttributeError):
                pass

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                m.fit(Xb, yb)

            pb = m.predict(X_new)
            if pb.ndim == 1:
                pb = pb.reshape(-1, 1)
            boot_samples[i] = pb

            if (i + 1) % max(1, n_bootstrap // 5) == 0:
                logger.info("  Bootstrap progress: %d/%d", i + 1, n_bootstrap)

        logger.info("Bootstrapping complete (%d iterations).", n_bootstrap)

        lo = np.percentile(boot_samples, alpha / 2 * 100, axis=0)
        hi = np.percentile(boot_samples, (1 - alpha / 2) * 100, axis=0)

        return PredictionResult(
            point_estimate=point,
            ci_lower=lo,
            ci_upper=hi,
            ci_level=1 - alpha,
            bootstrap_samples=boot_samples if store_samples else None,
        )

    # ================================================================
    # Step 3: Feature Importance
    # ================================================================
    def feature_importance(self) -> pd.DataFrame:
        """Model-native feature importance.

        - Tree-based models: mean decrease in impurity.
        - Linear models (single-output): absolute coefficient values.
        - Multi-output linear models: raises with hint to use
          ``permutation_importance()``.
        - Other estimators: raises if neither ``feature_importances_`` nor
          ``coef_`` is available.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``, sorted descending.
        """
        self._check_fitted()
        names = self.feature_names_ or [
            f"X{i}" for i in range(self._n_features())
        ]

        if hasattr(self.estimator, "feature_importances_"):
            imp = self.estimator.feature_importances_
        elif hasattr(self.estimator, "coef_"):
            coef = self.estimator.coef_
            if coef.ndim == 2 and coef.shape[0] > 1:
                raise AttributeError(
                    "Multi-output linear model detected (coef_ is 2D). "
                    "Use permutation_importance() for per-target importance."
                )
            imp = np.abs(coef).flatten()
        else:
            raise AttributeError(
                "Estimator has no feature_importances_ or coef_. "
                "Use permutation_importance() instead."
            )

        if len(imp) != len(names):
            names = [f"X{i}" for i in range(len(imp))]

        return (
            pd.DataFrame({"feature": names[: len(imp)], "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def permutation_importance(
        self, X: "ArrayLike", y: "ArrayLike", n_repeats: int = 5
    ) -> pd.DataFrame:
        """Permutation-based feature importance (model-agnostic).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
        n_repeats : int, default=5
            Number of times each feature is permuted. Must be >= 1.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance_mean``, ``importance_std``,
            sorted by mean descending.
        """
        self._check_fitted()
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")

        X_arr = self._to_array(X)
        y_arr = self._to_array(y)
        names = self.feature_names_ or [f"X{i}" for i in range(X_arr.shape[1])]
        r = sk_permutation_importance(
            self.estimator,
            X_arr,
            y_arr,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        return (
            pd.DataFrame(
                {
                    "feature": names,
                    "importance_mean": r.importances_mean,
                    "importance_std": r.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

    # ================================================================
    # Step 4: Sensitivity Analysis
    # ================================================================
    def sensitivity(
        self,
        X: "ArrayLike",
        feature: Union[str, int],
        pct_range: tuple[float, float] = (0.01, 0.20),
        n_steps: int = 20,
    ) -> pd.DataFrame:
        """How predictions change when perturbing a single feature.

        Incrementally shifts *feature* upward by a percentage (1%–20% by
        default) and measures the mean absolute change in predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Reference data to perturb.
        feature : str or int
            Feature name (str for DataFrame columns) or column index.
        pct_range : tuple (min_pct, max_pct), default=(0.01, 0.20)
            Perturbation range as fractions (0.01–0.20 = 1%–20%).
        n_steps : int, default=20
            Number of evenly-spaced perturbation levels. Must be >= 1.

        Returns
        -------
        pd.DataFrame
            Columns: ``pct`` (percentage), ``mean_abs_change``.
        """
        self._check_fitted()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if pct_range[0] >= pct_range[1]:
            raise ValueError("pct_range[0] must be < pct_range[1]")

        X_ref = self._to_array(X)
        idx = self._resolve_feature_index(feature, X)

        base_pred = self.estimator.predict(X_ref)
        if base_pred.ndim == 1:
            base_pred = base_pred.reshape(-1, 1)

        rows: list[dict[str, float]] = []
        for pct in np.linspace(pct_range[0], pct_range[1], n_steps):
            Xp = X_ref.copy()
            Xp[:, idx] *= 1.0 + pct
            pred_p = self.estimator.predict(Xp)
            if pred_p.ndim == 1:
                pred_p = pred_p.reshape(-1, 1)
            change = float(np.mean(np.abs(pred_p - base_pred)))
            rows.append({"pct": round(pct * 100, 1), "mean_abs_change": change})

        return pd.DataFrame(rows)

    # ================================================================
    # Step 5: Time Series CV
    # ================================================================
    def time_series_cv(
        self, X: "ArrayLike", y: "ArrayLike", n_splits: int = 5
    ) -> pd.DataFrame:
        """Expanding-window time series cross-validation.

        Trains a fresh clone of the estimator on each fold so the user's model
        choice is respected.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
        n_splits : int, default=5
            Number of train/test splits. Must be >= 2.

        Returns
        -------
        pd.DataFrame
            Columns: ``fold``, ``target``, ``R2``, ``RMSE``, ``MAE``.
            Also stored as ``self.cv_results_``.
        """
        self._check_fitted()
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        X_arr = self._to_array(X)
        y_arr = self._to_array(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        tnames = self.target_names_ or [f"y{j}" for j in range(y_arr.shape[1])]
        results: list[dict[str, Any]] = []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_arr)):
            X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
            y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]

            m = clone(self.estimator)
            try:
                m.set_params(random_state=self.random_state)
            except (ValueError, AttributeError):
                pass

            m.fit(
                X_tr,
                y_tr.ravel()
                if y_tr.ndim == 2 and y_tr.shape[1] == 1
                else y_tr,
            )
            y_pred = m.predict(X_te)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)

            for j, name in enumerate(tnames):
                results.append(
                    {
                        "fold": fold + 1,
                        "target": name,
                        "R2": r2_score(y_te[:, j], y_pred[:, j]),
                        "RMSE": float(
                            np.sqrt(mean_squared_error(y_te[:, j], y_pred[:, j]))
                        ),
                        "MAE": mean_absolute_error(y_te[:, j], y_pred[:, j]),
                    }
                )

        self.cv_results_ = pd.DataFrame(results)
        return self.cv_results_

    # ================================================================
    # Step 6: Summary
    # ================================================================
    def summary(self) -> None:
        """Print a formatted console report.

        Includes feature importance (if available) and time-series CV
        averages (if ``time_series_cv()`` was called beforehand).
        """
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
    def plot(
        self,
        result: Optional[PredictionResult] = None,
        sens_df: Optional[pd.DataFrame] = None,
    ) -> "Figure":
        """Generate diagnostic matplotlib figure.

        Panels are chosen automatically:
        - Feature importance (always, if available)
        - Prediction error bars with CIs (if *result* given)
        - Sensitivity curve (if *sens_df* given)
        - Time-series CV R² per fold (if ``time_series_cv()`` was called)

        Parameters
        ----------
        result : PredictionResult, optional
            From ``predict_with_ci()``.
        sens_df : pd.DataFrame, optional
            From ``sensitivity()``.

        Returns
        -------
        matplotlib.figure.Figure
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
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first")

    @staticmethod
    def _to_array(x: "ArrayLike") -> np.ndarray:
        """Convert array-like to a float64 ndarray."""
        if hasattr(x, "values"):
            x = x.values
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _validate_input(arr: np.ndarray, name: str = "input") -> None:
        if arr.size == 0:
            raise ValueError(f"{name} is empty")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or inf values")

    def _extract_names(self, obj: Any) -> Optional[list[str]]:
        """Extract column/Series names from a DataFrame or Series."""
        if hasattr(obj, "columns"):
            return list(obj.columns)
        if hasattr(obj, "name") and obj.name is not None:
            return [str(obj.name)]
        return None

    def _n_features(self) -> int:
        """Determine feature count from the fitted state."""
        if self._n_features_in_ is not None:
            return self._n_features_in_
        if hasattr(self.estimator, "n_features_in_"):
            return int(self.estimator.n_features_in_)
        if hasattr(self.estimator, "coef_"):
            return int(self.estimator.coef_.shape[-1])
        if self._X_train is not None:
            return self._X_train.shape[1]
        raise RuntimeError(
            "Cannot determine n_features. Call fit() first, "
            "or use an estimator that exposes n_features_in_ or coef_."
        )

    def _resolve_feature_index(
        self, feature: Union[str, int], X: Optional["ArrayLike"] = None
    ) -> int:
        """Resolve a feature name (str) or index (int) to a column index."""
        if isinstance(feature, str):
            if X is not None and hasattr(X, "columns"):
                try:
                    return list(X.columns).index(feature)
                except ValueError:
                    pass
            if self.feature_names_:
                return self.feature_names_.index(feature)
            raise ValueError(
                f"Feature '{feature}' not found. "
                "Pass a DataFrame with named columns, or use a column index."
            )
        return int(feature)
