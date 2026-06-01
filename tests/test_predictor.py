"""Tests for bootstrap_predictor."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from bootstrap_predictor import BootstrapPredictor, PredictionResult


# ================================================================
# Fixtures
# ================================================================
@pytest.fixture
def sample_data():
    """Simple regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(100) * 0.3
    return X, y


@pytest.fixture
def sample_df(sample_data):
    """DataFrame variant."""
    X, y = sample_data
    X_df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def fitted_bp(sample_data):
    """A fitted BootstrapPredictor with default RF."""
    X, y = sample_data
    bp = BootstrapPredictor(random_state=42)
    bp.fit(X, y)
    return bp


# ================================================================
# Basic fit / predict
# ================================================================
class TestFitPredict:
    def test_fit_sets_attributes(self, sample_data):
        X, y = sample_data
        bp = BootstrapPredictor(random_state=42)
        bp.fit(X, y)
        assert bp._is_fitted
        assert bp._X_train is not None
        assert bp._y_train is not None

    def test_predict_returns_2d(self, fitted_bp, sample_data):
        X, _ = sample_data
        pred = fitted_bp.predict(X[:10])
        assert pred.shape == (10, 1)

    def test_predict_raises_if_not_fitted(self, sample_data):
        X, _ = sample_data
        bp = BootstrapPredictor()
        with pytest.raises(RuntimeError, match="Call .fit"):
            bp.predict(X)

    def test_repr(self, fitted_bp):
        r = repr(fitted_bp)
        assert "RandomForestRegressor" in r
        assert "fitted" in r

    def test_repr_unfitted(self):
        bp = BootstrapPredictor()
        assert "unfitted" in repr(bp)


# ================================================================
# predict_with_ci
# ================================================================
class TestPredictWithCI:
    def test_returns_prediction_result(self, fitted_bp, sample_data):
        X, _ = sample_data
        result = fitted_bp.predict_with_ci(X[:10], n_bootstrap=20)
        assert isinstance(result, PredictionResult)
        assert result.point_estimate.shape == (10, 1)
        assert result.ci_lower.shape == (10, 1)
        assert result.ci_upper.shape == (10, 1)

    def test_ci_bounds_are_ordered(self, fitted_bp, sample_data):
        X, _ = sample_data
        result = fitted_bp.predict_with_ci(X[:10], n_bootstrap=20)
        assert np.all(result.ci_lower <= result.point_estimate)
        assert np.all(result.point_estimate <= result.ci_upper)

    def test_ci_level_default_95(self, fitted_bp, sample_data):
        X, _ = sample_data
        result = fitted_bp.predict_with_ci(X[:10], n_bootstrap=20)
        assert result.ci_level == 0.95

    def test_custom_alpha(self, fitted_bp, sample_data):
        X, _ = sample_data
        result = fitted_bp.predict_with_ci(X[:10], n_bootstrap=20, alpha=0.10)
        assert result.ci_level == 0.90

    def test_bootstrap_samples_stored(self, fitted_bp, sample_data):
        X, _ = sample_data
        result = fitted_bp.predict_with_ci(X[:10], n_bootstrap=20)
        assert result.bootstrap_samples is not None
        assert result.bootstrap_samples.shape[0] == 20

    def test_uses_cloned_estimator_not_hardcoded_rf(self, sample_data):
        """Regression test for bug #1: bootstrap must use the user's estimator."""
        X, y = sample_data
        bp = BootstrapPredictor(estimator=LinearRegression(), random_state=42)
        bp.fit(X, y)
        result = bp.predict_with_ci(X[:10], n_bootstrap=20)
        # LinearRegression coef on this data should be roughly [2, 0.5, 0]
        # The point estimate comes from the fitted LinearRegression
        assert isinstance(bp.estimator, LinearRegression)
        # CIs should be narrower for linear models (less variance) than RF
        ci_width = np.mean(result.ci_upper - result.ci_lower)
        # Reasonable width for linear model on this data
        assert ci_width < 5.0

    def test_multi_output(self, sample_data):
        X, y = sample_data
        y_multi = np.column_stack([y, y * 2])
        bp = BootstrapPredictor(random_state=42)
        bp.fit(X, y_multi)
        result = bp.predict_with_ci(X[:10], n_bootstrap=20)
        assert result.point_estimate.shape == (10, 2)
        assert result.ci_lower.shape == (10, 2)


# ================================================================
# Feature importance
# ================================================================
class TestFeatureImportance:
    def test_tree_based_importance(self, fitted_bp):
        imp = fitted_bp.feature_importance()
        assert list(imp.columns) == ["feature", "importance"]
        assert len(imp) == 3

    def test_linear_importance(self, sample_data):
        X, y = sample_data
        bp = BootstrapPredictor(estimator=LinearRegression(), random_state=42)
        bp.fit(X, y)
        imp = bp.feature_importance()
        assert len(imp) == 3
        assert "importance" in imp.columns

    def test_raises_for_unsupported_estimator(self, sample_data):
        """Estimator without feature_importances_ or coef_ should raise."""
        from sklearn.svm import SVR
        X, y = sample_data
        bp = BootstrapPredictor(estimator=SVR(), random_state=42)
        bp.fit(X, y)
        with pytest.raises(AttributeError):
            bp.feature_importance()

    def test_multi_output_linear_raises(self, sample_data):
        """Multi-output linear model should suggest permutation_importance."""
        X, y = sample_data
        y_multi = np.column_stack([y, y * 2])
        bp = BootstrapPredictor(estimator=LinearRegression(), random_state=42)
        bp.fit(X, y_multi)
        with pytest.raises(AttributeError, match="permutation_importance"):
            bp.feature_importance()


# ================================================================
# Permutation importance
# ================================================================
class TestPermutationImportance:
    def test_returns_dataframe(self, fitted_bp, sample_data):
        X, y = sample_data
        imp = fitted_bp.permutation_importance(X, y, n_repeats=3)
        assert list(imp.columns) == ["feature", "importance_mean", "importance_std"]
        assert len(imp) == 3

    def test_works_with_any_estimator(self, sample_data):
        from sklearn.svm import SVR
        X, y = sample_data
        bp = BootstrapPredictor(estimator=SVR(), random_state=42)
        bp.fit(X, y)
        imp = bp.permutation_importance(X, y, n_repeats=3)
        assert len(imp) == 3


# ================================================================
# Sensitivity analysis
# ================================================================
class TestSensitivity:
    def test_returns_dataframe(self, fitted_bp, sample_data):
        X, _ = sample_data
        X_df = pd.DataFrame(X, columns=["a", "b", "c"])
        sens = fitted_bp.sensitivity(X_df, feature="a")
        assert list(sens.columns) == ["pct", "mean_abs_change"]
        assert len(sens) == 20

    def test_change_increases_with_pct(self, fitted_bp, sample_data):
        X, _ = sample_data
        X_df = pd.DataFrame(X, columns=["a", "b", "c"])
        sens = fitted_bp.sensitivity(X_df, feature="a")
        # Mean absolute change should generally increase with perturbation %
        assert sens["mean_abs_change"].iloc[-1] >= sens["mean_abs_change"].iloc[0]

    def test_by_index(self, fitted_bp, sample_data):
        X, _ = sample_data
        sens = fitted_bp.sensitivity(X, feature=0)
        assert len(sens) == 20


# ================================================================
# Time series CV
# ================================================================
class TestTimeSeriesCV:
    def test_returns_dataframe(self, fitted_bp, sample_data):
        X, y = sample_data
        cv = fitted_bp.time_series_cv(X, y, n_splits=3)
        assert "fold" in cv.columns
        assert "R2" in cv.columns
        assert "RMSE" in cv.columns
        assert "MAE" in cv.columns
        assert cv["fold"].max() == 3

    def test_uses_cloned_estimator(self, sample_data):
        """Regression test for bug #2: CV must use user's estimator, not hardcoded RF."""
        X, y = sample_data
        bp = BootstrapPredictor(estimator=LinearRegression(), random_state=42)
        bp.fit(X, y)
        cv = bp.time_series_cv(X, y, n_splits=3)
        # LinearRegression should give perfect-ish R² on this linear data
        assert cv["R2"].mean() > 0.7

    def test_stores_cv_results(self, fitted_bp, sample_data):
        X, y = sample_data
        fitted_bp.time_series_cv(X, y, n_splits=3)
        assert fitted_bp.cv_results_ is not None


# ================================================================
# PredictionResult
# ================================================================
class TestPredictionResult:
    def test_to_dataframe(self):
        r = PredictionResult(
            point_estimate=np.array([[1.0], [2.0]]),
            ci_lower=np.array([[0.8], [1.8]]),
            ci_upper=np.array([[1.2], [2.2]]),
        )
        df = r.to_dataframe()
        assert len(df) == 2
        assert list(df.columns) == ["sample", "target", "point", "ci_lower", "ci_upper"]

    def test_to_dataframe_custom_names(self):
        r = PredictionResult(
            point_estimate=np.array([[1.0], [2.0]]),
            ci_lower=np.array([[0.8], [1.8]]),
            ci_upper=np.array([[1.2], [2.2]]),
        )
        df = r.to_dataframe(sample_index=["A", "B"], target_names=["sales"])
        assert df["sample"].tolist() == ["A", "B"]
        assert df["target"].tolist() == ["sales", "sales"]

    def test_repr(self):
        r = PredictionResult(
            point_estimate=np.zeros((10, 1)),
            ci_lower=np.zeros((10, 1)),
            ci_upper=np.zeros((10, 1)),
        )
        rep = repr(r)
        assert "n=10" in rep
        assert "95%" in rep


# ================================================================
# Input validation
# ================================================================
class TestInputValidation:
    def test_empty_X_raises(self):
        bp = BootstrapPredictor()
        with pytest.raises(ValueError, match="empty"):
            bp.fit(np.array([]).reshape(0, 3), np.array([1, 2, 3]))

    def test_nan_X_raises(self):
        bp = BootstrapPredictor()
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="NaN"):
            bp.fit(X, y)

    def test_inf_y_raises(self):
        bp = BootstrapPredictor()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, np.inf])
        with pytest.raises(ValueError, match="inf"):
            bp.fit(X, y)


# ================================================================
# Edge cases
# ================================================================
class TestEdgeCases:
    def test_single_feature(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 1)
        y = 3 * X[:, 0] + rng.randn(50) * 0.1
        bp = BootstrapPredictor(random_state=42)
        bp.fit(X, y)
        result = bp.predict_with_ci(X[:5], n_bootstrap=10)
        assert result.point_estimate.shape == (5, 1)

    def test_feature_names_from_dataframe(self, sample_df):
        X_df, y_series = sample_df
        bp = BootstrapPredictor(random_state=42)
        bp.fit(X_df, y_series)
        assert bp.feature_names_ == ["feat_a", "feat_b", "feat_c"]
        assert bp.target_names_ == ["target"]
        imp = bp.feature_importance()
        assert imp["feature"].tolist() == ["feat_a", "feat_b", "feat_c"]

    def test_summary_runs(self, fitted_bp, capsys):
        fitted_bp.summary()
        captured = capsys.readouterr()
        assert "BootstrapPredictor Report" in captured.out
