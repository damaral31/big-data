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


def _pivot_aggregates(aggregates: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Long (icustay, concept, stats) -> wide (one column per concept x stat).

    ``prefix`` namespaces the columns (e.g. ``"lab_"`` for LABEVENTS) so vital and
    lab concepts that share a name (e.g. glucose) do not collide.
    """
    parts = []
    for stat in cfg.AGG_FUNCS:
        col = _STAT_TO_COL[stat]
        wide = aggregates.pivot(index="icustay_id", columns="concept", values=col)
        wide.columns = [f"{prefix}{c}_{stat}" for c in wide.columns]
        parts.append(wide)
    out = pd.concat(parts, axis=1)

    # window-capped intensity features (not leaky: bounded by the fixed window)
    counts = aggregates.pivot(index="icustay_id", columns="concept", values="n")
    out[f"{prefix}n_measurements_total"] = counts.sum(axis=1)
    out[f"{prefix}n_concepts_observed"] = counts.notna().sum(axis=1)
    return out


def _encode_demographics(demographics: pd.DataFrame) -> pd.DataFrame:
    df = demographics.set_index("icustay_id").copy()
    df["age_years"] = df["age_years"].clip(upper=cfg.AGE_CAP)
    df["is_male"] = (df["gender"] == "M").astype("int8")

    cats = [c for c in _CATEGORICAL if c in df.columns]
    dummies = pd.get_dummies(df[cats], prefix=cats, dummy_na=False, dtype="int8")
    return pd.concat([df[["age_years", "is_male"]], dummies], axis=1)


def _measured_indicators(aggregates: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """One 0/1 column per concept: was it charted in the window at all?

    In MIMIC missingness is *informative* (missing-not-at-random): the decision to
    order a lactate or an extra vital is itself a severity signal. These indicators
    capture that, and -- crucially -- they are built BEFORE imputation, which would
    otherwise erase the "was it measured" information.
    """
    pres = aggregates.pivot(index="icustay_id", columns="concept", values="n")
    pres = pres.notna().astype("int8")
    pres.columns = [f"{prefix}{c}_measured" for c in pres.columns]
    return pres


def build_feature_matrix(data, include_labs: bool = True,
                         add_missing_indicators: bool = True) -> FeatureMatrix:
    """Assemble the model-ready matrix from a :class:`CohortData` object.

    With ``include_labs`` and lab aggregates present, LABEVENTS features are
    *left*-joined onto the vitals matrix (prefix ``lab_``). The left join keeps
    the stay set identical to the vitals-only matrix, so a with-vs-without-labs
    comparison is strictly apples-to-apples (same rows, extra columns).

    ``add_missing_indicators`` appends one 0/1 "was-measured" column per concept
    (informative missingness; see :func:`_measured_indicators`).
    """
    vitals = _pivot_aggregates(data.aggregates)
    demo = _encode_demographics(data.demographics)

    target = data.cohort.set_index("icustay_id")[["subject_id", "los_icu_days"]]

    # inner-join keeps only stays that have BOTH a target and >=1 charted vital
    X = vitals.join(demo, how="inner").join(target, how="inner")
    X = X[X["los_icu_days"].notna()]

    lab_cols: set[str] = set()
    labs = getattr(data, "lab_aggregates", None)
    if include_labs and labs is not None and len(labs):
        lab_wide = _pivot_aggregates(labs, prefix="lab_")
        X = X.join(lab_wide, how="left")  # left: same rows, lab columns added
        lab_cols = set(lab_wide.columns)
        logger.info("Added %d LABEVENTS feature columns", len(lab_cols))

    if add_missing_indicators:
        ind = _measured_indicators(data.aggregates)
        if include_labs and labs is not None and len(labs):
            ind = ind.join(_measured_indicators(labs, prefix="lab_"), how="outer")
        # reindex to current stays; absent => not measured => 0
        ind = ind.reindex(X.index).fillna(0).astype("int8")
        X = X.join(ind, how="left")
        logger.info("Added %d informative-missingness indicators", ind.shape[1])

    # drop engineered columns that are almost always missing (labs get a more
    # lenient threshold because they are sampled less often than vitals)
    feat_cols = [c for c in X.columns if c not in ("subject_id", "los_icu_days")]
    missing = X[feat_cols].isna().mean()
    keep = [
        c for c in feat_cols
        if missing[c] <= (cfg.MAX_MISSING_FRACTION_LABS if c in lab_cols
                          else cfg.MAX_MISSING_FRACTION)
    ]
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


def categorize_features(feature_names: list[str]) -> pd.Series:
    """Group feature columns into human-readable categories (counts).

    Used in the notebook to report 'how many attributes and the distribution of
    attribute categories'.
    """
    def cat(name: str) -> str:
        if name.endswith("_measured"):
            return "lab_missingness" if name.startswith("lab_") else "vital_missingness"
        if name.startswith("lab_"):
            return "lab_intensity" if "n_measurements" in name or "n_concepts" in name \
                else "lab_statistic"
        if name in ("n_measurements_total", "n_concepts_observed"):
            return "vital_intensity"
        if name in ("age_years", "is_male") or name.startswith(
                ("admission_type_", "first_careunit_", "insurance_")):
            return "demographic"
        return "vital_statistic"

    counts = pd.Series([cat(n) for n in feature_names]).value_counts()
    counts.name = "n_features"
    return counts
