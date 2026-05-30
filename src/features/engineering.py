"""Feature engineering: long aggregates + demographics -> model matrix.

Design rules (all chosen to avoid target leakage):

* Every numeric feature comes from the FIRST-WINDOW aggregates only
  (computed in BigQuery with ``charttime < intime + window``).  Nothing derived
  from OUTTIME/DISCHTIME or whole-stay statistics enters the matrix.
* Measurement *counts* are kept, but they are counts *within the fixed window*
  -- a legitimate severity proxy (sicker patients are monitored more closely),
  not a proxy for total stay length.
* Imputation is NOT done here.  Missing values are left as NaN and handled
  inside the model :class:`~sklearn.pipeline.Pipeline` so imputers are fit on
  training folds only.
* Age is clipped to ``AGE_CAP`` to neutralise the MIMIC-III >89y / ~300y shift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as cfg

logger = logging.getLogger(__name__)

# feature-name suffix -> column in the long aggregates frame
_STAT_TO_COL = {"mean": "mean", "min": "min", "max": "max", "std": "std", "count": "n"}

_CATEGORICAL = ["admission_type", "first_careunit", "insurance"]


@dataclass
class FeatureMatrix:
    X: pd.DataFrame             # numeric feature matrix (may contain NaN)
    y_reg: pd.Series            # ICU LOS in days (regression target)
    y_clf: pd.Series            # short/medium/long bucket (classification target)
    groups: pd.Series           # subject_id, for grouped CV / splitting
    feature_names: list[str]


def _pivot_aggregates(aggregates: pd.DataFrame) -> pd.DataFrame:
    """Long (icustay, concept, stats) -> wide (one column per concept x stat)."""
    parts = []
    for stat in cfg.AGG_FUNCS:
        col = _STAT_TO_COL[stat]
        wide = aggregates.pivot(index="icustay_id", columns="concept", values=col)
        wide.columns = [f"{c}_{stat}" for c in wide.columns]
        parts.append(wide)
    out = pd.concat(parts, axis=1)

    # window-capped intensity features (not leaky: bounded by the fixed window)
    counts = aggregates.pivot(index="icustay_id", columns="concept", values="n")
    out["n_measurements_total"] = counts.sum(axis=1)
    out["n_concepts_observed"] = counts.notna().sum(axis=1)
    return out


def _encode_demographics(demographics: pd.DataFrame) -> pd.DataFrame:
    df = demographics.set_index("icustay_id").copy()
    df["age_years"] = df["age_years"].clip(upper=cfg.AGE_CAP)
    df["is_male"] = (df["gender"] == "M").astype("int8")

    cats = [c for c in _CATEGORICAL if c in df.columns]
    dummies = pd.get_dummies(df[cats], prefix=cats, dummy_na=False, dtype="int8")
    return pd.concat([df[["age_years", "is_male"]], dummies], axis=1)


def build_feature_matrix(data) -> FeatureMatrix:
    """Assemble the model-ready matrix from a :class:`CohortData` object."""
    vitals = _pivot_aggregates(data.aggregates)
    demo = _encode_demographics(data.demographics)

    target = data.cohort.set_index("icustay_id")[["subject_id", "los_icu_days"]]

    # inner-join keeps only stays that have BOTH a target and >=1 charted vital
    X = vitals.join(demo, how="inner").join(target, how="inner")
    X = X[X["los_icu_days"].notna()]

    # drop engineered columns that are almost always missing
    feat_cols = [c for c in X.columns if c not in ("subject_id", "los_icu_days")]
    missing = X[feat_cols].isna().mean()
    keep = missing[missing <= cfg.MAX_MISSING_FRACTION].index.tolist()
    dropped = sorted(set(feat_cols) - set(keep))
    if dropped:
        logger.info("Dropping %d high-missing features: %s", len(dropped), dropped)

    y_reg = X["los_icu_days"].astype(float)
    y_clf = pd.cut(
        y_reg,
        bins=[0, cfg.LOS_SHORT_DAYS, cfg.LOS_MEDIUM_DAYS, np.inf],
        labels=list(cfg.LOS_CLASS_LABELS),
        include_lowest=True,
    ).astype(str)
    groups = X["subject_id"].astype(int)
    X_out = X[keep].astype("float32")

    logger.info(
        "Feature matrix: %d stays x %d features | LOS mean=%.2f median=%.2f days",
        len(X_out), X_out.shape[1], y_reg.mean(), y_reg.median(),
    )
    logger.info("Class balance: %s", y_clf.value_counts().to_dict())
    return FeatureMatrix(X_out, y_reg, y_clf, groups, keep)
