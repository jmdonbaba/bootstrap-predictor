"""
bootstrap-predictor: ML prediction with bootstrap confidence intervals.

Core API
--------
BootstrapPredictor    — train → predict_with_ci → sensitivity → CV

Usage
-----
from bootstrap_predictor import BootstrapPredictor

bp = BootstrapPredictor()
bp.fit(X_train, y_train)
result = bp.predict_with_ci(X_new, n_bootstrap=100)
# result.point_estimate, result.ci_lower, result.ci_upper
"""

from .predictor import BootstrapPredictor, PredictionResult

__version__ = "0.1.0"
__all__ = ["BootstrapPredictor", "PredictionResult"]
