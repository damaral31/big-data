"""Evaluation metrics for the regression and classification framings."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score,
)

# ordinal order of the LOS buckets (short < medium < long)
CLASS_ORDER = ("short", "medium", "long")


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MedAE": float(np.median(np.abs(y_true - y_pred))),
        "within_1d": float(np.mean(np.abs(y_true - y_pred) <= 1.0)),
        "within_2d": float(np.mean(np.abs(y_true - y_pred) <= 2.0)),
    }


def error_by_los_range(y_true, y_pred) -> pd.DataFrame:
    """MAE / count broken down by true-LOS band -- shows where the model fails."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, np.inf)]
    rows = []
    for lo, hi in bands:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "LOS_range_days": f"{lo}-{'inf' if hi == np.inf else hi}",
            "n": int(m.sum()),
            "MAE": mean_absolute_error(y_true[m], y_pred[m]),
            "mean_pred": float(y_pred[m].mean()),
            "mean_true": float(y_true[m].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred, y_proba=None,
                           labels=CLASS_ORDER) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels,
                             zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro",
                                           labels=labels, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro",
                                     labels=labels, zero_division=0),
        # quadratic-weighted kappa treats the buckets as ordinal (short<med<long)
        "kappa_quadratic": cohen_kappa_score(y_true, y_pred, labels=labels,
                                             weights="quadratic"),
    }
    if y_proba is not None:
        try:
            out["roc_auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro",
                labels=labels)
        except Exception:
            out["roc_auc_ovr"] = float("nan")
    return out
