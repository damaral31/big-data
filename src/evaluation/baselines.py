"""Published benchmark numbers for ICU LOS on MIMIC-III.

These are *reference baselines from the literature* -- we compare our models
against them in the Evaluation phase rather than only against our own trivial
baselines. They also calibrate expectations: continuous-LOS R^2 is low, which is
why most papers prefer the ordinal/bucketed framing.

Numbers are quoted from the cited papers; see reports/REPORT.md for full
citations. Hours converted to days at /24 where noted.
"""
from __future__ import annotations

import pandas as pd

# (method, framing, metric, value, source)
_ROWS = [
    ("Linear regression (baseline)", "regression",      "MAE_days", 4.85,
     "Harutyunyan 2019 (MAD 116.4h)"),
    ("Channel-wise LSTM",            "regression",      "MAE_days", 3.92,
     "Harutyunyan 2019 (MAD 94.0h)"),
    ("Std ML, first-24h",            "regression",      "R2",       0.04,
     "first-24h MIMIC study (R^2~0.04)"),
    ("Logistic regression (base)",  "ordinal/bucket",  "kappa_quadratic", 0.34,
     "Harutyunyan 2019"),
    ("Channel-wise LSTM",            "ordinal/bucket",  "kappa_quadratic", 0.43,
     "Harutyunyan 2019"),
    ("GRU-D / RF (LOS>3d)",          "binary>3d",       "AUROC",    0.74,
     "MIMIC-Extract 2020"),
    ("GRU-D / RF (LOS>7d)",          "binary>7d",       "AUROC",    0.76,
     "MIMIC-Extract 2020"),
]


def literature_baselines() -> pd.DataFrame:
    return pd.DataFrame(
        _ROWS, columns=["method", "framing", "metric", "value", "source"])
