"""Diagnostic plots for BootstrapPredictor — matplotlib only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .predictor import BootstrapPredictor, PredictionResult


def plot_predictor_report(
    predictor: "BootstrapPredictor",
    result: Optional["PredictionResult"] = None,
    sens_df: Optional[pd.DataFrame] = None,
    figsize: tuple[float, float] = (12, 8),
) -> "Figure":
    """2-4 panel diagnostic report.

    Panels depend on what is provided:
    - Feature importance (always, if available)
    - CV metrics (if cv was run)
    - Prediction with CI (if result is provided)
    - Sensitivity curve (if sens_df is provided)
    """

    n_panels = 1
    has_ci = result is not None
    has_sens = sens_df is not None
    has_cv = predictor.cv_results_ is not None
    if has_ci:
        n_panels += 1
    if has_sens:
        n_panels += 1
    if has_cv:
        n_panels += 1

    # Determine layout
    ncols = min(n_panels, 2)
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    panel = 0

    # ---- Panel 1: Feature Importance ----
    ax = axes[panel]
    try:
        imp = predictor.feature_importance()
        top = imp.head(15)
        ax.barh(range(len(top)), top["importance"].values[::-1], color="steelblue")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
    except AttributeError:
        ax.text(
            0.5,
            0.5,
            "Not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Feature Importance (N/A)")
    panel += 1

    # ---- Panel 2: Prediction with CI ----
    if has_ci:
        ax = axes[panel]
        n_show = min(20, len(result.point_estimate))
        y = result.point_estimate[:n_show, 0]
        lo = result.ci_lower[:n_show, 0]
        hi = result.ci_upper[:n_show, 0]
        x = np.arange(n_show)
        ax.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            fmt="o",
            capsize=3,
            markersize=5,
            elinewidth=1,
            color="steelblue",
        )
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Prediction")
        ax.set_title(f"Predictions with {result.ci_level * 100:.0f}% CI")
        ax.axhline(y=np.mean(y), color="gray", linestyle="--", alpha=0.5)
        panel += 1

    # ---- Panel 3: Sensitivity ----
    if has_sens:
        ax = axes[panel]
        ax.plot(
            sens_df["pct"],
            sens_df["mean_abs_change"],
            marker="o",
            markersize=3,
            linewidth=2,
            color="darkorange",
        )
        ax.set_xlabel("Perturbation (%)")
        ax.set_ylabel("Mean |Change| in Prediction")
        ax.set_title("Sensitivity Analysis")
        ax.grid(True, alpha=0.3)
        panel += 1

    # ---- Panel 4: CV ----
    if has_cv:
        ax = axes[panel]
        cv_s = predictor.cv_results_
        targets = cv_s["target"].unique()
        x = cv_s[cv_s["target"] == targets[0]]["fold"].values
        for tname in targets:
            td = cv_s[cv_s["target"] == tname]
            ax.plot(x, td["R2"].values, marker="o", label=f"{tname}")
        ax.set_xlabel("Fold")
        ax.set_ylabel("R²")
        ax.set_title("Time Series CV (R² per fold)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        panel += 1

    # Hide unused axes
    for i in range(panel, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig
