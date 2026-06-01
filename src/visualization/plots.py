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


def los_window_justification(y_hours, windows=(6, 24, 48, 72), clip_h: int = 168):
    """Histogram of ICU LOS in hours with candidate prediction windows marked.

    The mass to the LEFT of each line is the fraction of stays that would be
    *excluded* by that window (they end before a full window of data exists).
    Computed on the UNFILTERED stay durations so the exclusions are visible.
    """
    h = np.asarray(y_hours, float)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(np.clip(h, 0, clip_h), bins=70, color="steelblue", edgecolor="black", alpha=0.8)
    for w, c in zip(windows, ["#888888", "firebrick", "darkorange", "seagreen"]):
        excl = float(np.mean(h < w)) * 100
        ax.axvline(w, color=c, ls="--", lw=2, label=f"{w}h  (exclui {excl:.1f}%)")
    ax.set(xlabel="duracao da estadia na UCI (horas, recortado a 168h)",
           ylabel="numero de estadias",
           title="Distribuicao da duracao das estadias e janelas candidatas")
    ax.legend(title="janela candidata")
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


def variance_scree(explained: np.ndarray, top: int = 10):
    n = min(top, len(explained))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n + 1), explained[:n] * 100, color="steelblue", alpha=0.85)
    ax.plot(range(1, n + 1), np.cumsum(explained[:n]) * 100, "o-", color="firebrick",
            label="cumulative")
    ax.set(xlabel="principal component", ylabel="variance explained (%)",
           title="PCA scree")
    ax.legend()
    fig.tight_layout()
    return fig


def embedding_scatter(coords, color_values, title: str, cbar_label: str = "LOS (days)",
                      categorical: bool = False):
    """2-D scatter (PCA or t-SNE) coloured by a value (LOS) or a category."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    if categorical:
        cats = pd.Series(color_values).astype("category")
        for c in cats.cat.categories:
            m = cats.values == c
            ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.5, label=str(c))
        ax.legend(title=cbar_label, fontsize=8)
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=np.clip(color_values, 0, 30),
                        s=10, alpha=0.5, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=cbar_label)
    ax.set(title=title, xlabel="dim 1", ylabel="dim 2")
    fig.tight_layout()
    return fig


def silhouette_plot(k_scores: dict):
    ks, sc = list(k_scores), list(k_scores.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, sc, "o-", color="slateblue")
    best = max(k_scores, key=k_scores.get)
    ax.axvline(best, color="firebrick", ls="--", label=f"best k={best}")
    ax.set(xlabel="number of clusters (k)", ylabel="mean silhouette",
           title="KMeans silhouette vs k")
    ax.legend()
    fig.tight_layout()
    return fig


def feature_distributions_by_class(X: pd.DataFrame, y_clf: pd.Series,
                                   features: list, order=("short", "medium", "long")):
    """Small-multiples violin/box of the top features, split by LOS bucket."""
    feats = [f for f in features if f in X.columns][:9]
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    df = X.copy()
    df["__bucket"] = pd.Categorical(y_clf.values, categories=list(order), ordered=True)
    for ax, f in zip(axes, feats):
        data = [df.loc[df["__bucket"] == b, f].dropna() for b in order]
        ax.boxplot(data, labels=list(order), showfliers=False)
        ax.set_title(f, fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[len(feats):]:
        ax.axis("off")
    fig.suptitle("Distribution of top features by LOS bucket", fontweight="bold")
    fig.tight_layout()
    return fig


def patient_vs_population(timeline: pd.DataFrame, concept: str,
                          pop_mean: float, pop_std: float, icustay_id: int):
    """One patient's trajectory for a concept vs the cohort's typical band.

    Line = the patient's measured values over the first day; shaded area = the
    cohort mean +/- 1 SD of that concept (so you can see if the patient runs
    high/low/unstable relative to the population).
    """
    g = timeline[timeline["concept"] == concept].sort_values("day_fraction")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhspan(pop_mean - pop_std, pop_mean + pop_std, color="steelblue", alpha=0.15,
               label="cohort mean +/- 1 SD")
    ax.axhline(pop_mean, color="steelblue", ls="--", lw=1, label="cohort mean")
    ax.plot(g["day_fraction"], g["value"], "o-", color="firebrick", ms=4,
            label=f"patient {icustay_id}")
    ax.set(xlabel="time since ICU admission (days)", ylabel=concept,
           title=f"{concept}: patient trajectory vs cohort")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def missingness_bar(missing_fraction: pd.Series, top: int = 30):
    s = missing_fraction.sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(s) + 1.5))
    ax.barh(s.index, s.values * 100, color="darkorange", alpha=0.85)
    ax.invert_yaxis()
    ax.set(xlabel="% missing", title=f"Missingness of top-{top} features")
    fig.tight_layout()
    return fig


def feature_category_bar(category_counts: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(category_counts) + 1.5))
    ax.barh(category_counts.index, category_counts.values, color="seagreen", alpha=0.85)
    ax.invert_yaxis()
    for i, v in enumerate(category_counts.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set(xlabel="number of features", title="Feature inventory by category")
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
