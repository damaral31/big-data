"""Plotting helpers.

All functions take prepared dataframes/arrays and return a Matplotlib Figure so
the notebook can display them inline and/or save them to ``reports/figures``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import config as cfg

sns.set_style("whitegrid")


def save(fig, filename: str) -> Path:
    path = cfg.FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def patient_timeline(events: pd.DataFrame, icustay_id: int):
    """Reproduce the assignment's per-patient plot.

    X = fraction of a day from ICU admission, Y = measured value, colour = concept.
    ``events`` columns: day_fraction, value, concept.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for concept, grp in events.groupby("concept"):
        ax.scatter(grp["day_fraction"], grp["value"], s=18, alpha=0.6, label=concept)
    ax.set_xlabel("Time since ICU admission (days)")
    ax.set_ylabel("Measured value")
    ax.set_title(f"ICU chart events over time — ICUSTAY_ID {icustay_id}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def los_distribution(y_days: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(y_days, bins=60, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].axvline(y_days.mean(), color="red", ls="--", label=f"mean {y_days.mean():.1f}d")
    axes[0].axvline(y_days.median(), color="green", ls="--",
                    label=f"median {y_days.median():.1f}d")
    axes[0].set(xlabel="ICU LOS (days)", ylabel="count", title="LOS distribution")
    axes[0].legend()
    axes[1].hist(np.log1p(y_days), bins=60, color="darkorange",
                 edgecolor="black", alpha=0.8)
    axes[1].set(xlabel="log(1 + LOS days)", ylabel="count",
                title="LOS distribution (log scale)")
    fig.tight_layout()
    return fig


def predicted_vs_actual(y_true, y_pred, model_name: str):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lims = [0, max(y_true.max(), y_pred.max())]
    axes[0].scatter(y_true, y_pred, s=12, alpha=0.4)
    axes[0].plot(lims, lims, "r--", lw=2)
    axes[0].set(xlabel="actual LOS (days)", ylabel="predicted LOS (days)",
                title=f"{model_name}: predicted vs actual")
    axes[1].scatter(y_pred, resid, s=12, alpha=0.4)
    axes[1].axhline(0, color="red", ls="--", lw=2)
    axes[1].set(xlabel="predicted LOS (days)", ylabel="residual (days)",
                title=f"{model_name}: residuals")
    fig.tight_layout()
    return fig


def model_comparison(cv_df: pd.DataFrame, metric_col: str, title: str,
                     lower_is_better: bool):
    df = cv_df.sort_values(metric_col, ascending=lower_is_better)
    std_col = metric_col.replace("_mean", "_std")
    err = df[std_col] if std_col in df.columns else None
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(df) + 1.5))
    ax.barh(df.index, df[metric_col], xerr=err, capsize=4,
            color="steelblue", alpha=0.8)
    ax.set(xlabel=metric_col, title=title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def feature_importance(names, importances, top_n: int = 20):
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(order) + 1.5))
    ax.barh([names[i] for i in order], [importances[i] for i in order],
            color="seagreen", alpha=0.85)
    ax.invert_yaxis()
    ax.set(xlabel="importance", title=f"Top {top_n} feature importances")
    fig.tight_layout()
    return fig


def confusion(cm: np.ndarray, labels):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="predicted", ylabel="actual", title="Confusion matrix")
    fig.tight_layout()
    return fig


def profiling(profile_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(profile_df) + 1.5))
    ax.barh(profile_df["phase"], profile_df["seconds"], color="slateblue",
            alpha=0.85)
    ax.invert_yaxis()
    ax.set(xlabel="seconds", title="Execution time by phase")
    fig.tight_layout()
    return fig
