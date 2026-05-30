"""Clearly-labelled SYNTHETIC data generator.

This exists so the full pipeline can be developed, unit-tested and demonstrated
without PhysioNet credentials or a live BigQuery connection.  It returns data in
*exactly* the same shape as the BigQuery loaders (:mod:`src.data.loader`) so the
downstream feature-engineering / modelling code is identical on both paths.

IMPORTANT: results produced on synthetic data are NOT real findings.  Every
function here stamps ``source = "SYNTHETIC"`` and the notebook prints a loud
banner.  The generator embeds a *modest, honest* signal (LOS depends weakly on a
latent severity that also drives vitals + on age/admission type) so model
metrics land in the realistic regime rather than at a misleadingly high R².
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as cfg
from .concepts import CONCEPT_VALID_RANGES

# Probability that a given concept was charted at all for a stay (mimics the
# sparse, ragged reality of CHARTEVENTS: not every patient gets every vital).
_CONCEPT_PRESENCE = {
    "heart_rate": 0.98, "sbp": 0.95, "dbp": 0.95, "mbp": 0.80,
    "resp_rate": 0.95, "spo2": 0.97, "temp_f": 0.85, "temp_c": 0.25,
    "glucose": 0.70, "gcs_total": 0.60, "fio2": 0.45,
    "weight_kg": 0.55, "tidal_volume": 0.30,
}

# Baseline mean per concept and how strongly latent severity shifts it.
# (concept: (healthy_mean, severity_slope))
_CONCEPT_SEVERITY = {
    "heart_rate": (82, 14), "sbp": (122, -10), "dbp": (68, -4),
    "mbp": (84, -7), "resp_rate": (17, 5), "spo2": (98, -4),
    "temp_f": (98.6, 1.2), "temp_c": (37.0, 0.6), "glucose": (130, 35),
    "gcs_total": (14, -4), "fio2": (40, 18), "weight_kg": (82, 0),
    "tidal_volume": (450, -20),
}


def generate(n_stays: int = cfg.SYNTHETIC_N_STAYS, seed: int = cfg.RANDOM_STATE):
    """Return ``(cohort_df, aggregates_long_df, demographics_df)`` -- synthetic."""
    rng = np.random.default_rng(seed)

    # --- patients: ~1.4 stays per subject so grouped splitting is exercised ----
    n_subjects = int(n_stays / 1.4)
    subject_ids = 10_000 + rng.integers(0, n_subjects, size=n_stays)
    icustay_ids = np.arange(200_000, 200_000 + n_stays)
    hadm_ids = 100_000 + rng.integers(0, n_subjects, size=n_stays)

    # --- latent severity drives both LOS and the vitals --------------------- #
    severity = rng.normal(0, 1, size=n_stays)

    # demographics
    age = np.clip(rng.normal(64, 17, size=n_stays), 18, 91)
    # ~3% are the >89 cohort whose DOB is shifted ~300y in real MIMIC
    shifted = rng.random(n_stays) < 0.03
    age[shifted] = rng.normal(300, 0.4, size=shifted.sum())
    gender = rng.choice(["M", "F"], size=n_stays, p=[0.56, 0.44])
    adm_type = rng.choice(
        ["EMERGENCY", "URGENT", "ELECTIVE", "NEWBORN"],
        size=n_stays, p=[0.74, 0.04, 0.21, 0.01],
    )
    careunit = rng.choice(
        ["MICU", "SICU", "CCU", "CSRU", "TSICU"], size=n_stays,
        p=[0.38, 0.20, 0.16, 0.16, 0.10],
    )

    # admission type nudges LOS (emergencies stay longer than electives)
    adm_effect = np.select(
        [adm_type == "EMERGENCY", adm_type == "URGENT", adm_type == "ELECTIVE"],
        [0.25, 0.15, -0.20], default=0.0,
    )
    age_effect = 0.004 * (np.clip(age, 18, 91) - 64)

    # right-skewed LOS (log-normal), correlated with severity -> realistic, modest
    log_los = 1.0 + 0.45 * severity + adm_effect + age_effect \
        + rng.normal(0, 0.55, size=n_stays)
    los_days = np.clip(np.exp(log_los), cfg.MIN_LOS_HOURS / 24.0, cfg.MAX_LOS_DAYS)

    cohort = pd.DataFrame({
        "icustay_id": icustay_ids,
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "intime": pd.Timestamp("2150-01-01"),  # placeholder; not used as a feature
        "los_icu_days": los_days,
    })

    demographics = pd.DataFrame({
        "icustay_id": icustay_ids,
        "subject_id": subject_ids,
        "hadm_id": hadm_ids,
        "gender": gender,
        "admission_type": adm_type,
        "admission_location": "EMERGENCY ROOM ADMIT",
        "insurance": rng.choice(["Medicare", "Private", "Medicaid"], size=n_stays),
        "ethnicity": "UNKNOWN",
        "first_careunit": careunit,
        "age_years": age,
    })

    # --- per-concept first-window aggregates (long format) ------------------ #
    rows = []
    for i in range(n_stays):
        s = severity[i]
        n_meas_base = max(2.0, 8 + 4 * s)  # sicker patients measured more often
        for concept, present_p in _CONCEPT_PRESENCE.items():
            if rng.random() > present_p:
                continue
            lo, hi = CONCEPT_VALID_RANGES[concept]
            base, slope = _CONCEPT_SEVERITY[concept]
            mean = base + slope * s + rng.normal(0, abs(slope) * 0.4 + 1)
            mean = float(np.clip(mean, lo, hi))
            spread = abs(rng.normal(0, abs(slope) * 0.5 + 2)) + 1e-3
            vmin = float(np.clip(mean - spread, lo, hi))
            vmax = float(np.clip(mean + spread, lo, hi))
            n = int(max(1, rng.poisson(n_meas_base)))
            rows.append((icustay_ids[i], concept, n, mean, vmin, vmax,
                         spread / 2.0))

    aggregates = pd.DataFrame(
        rows, columns=["icustay_id", "concept", "n", "mean", "min", "max", "std"]
    )
    return cohort, aggregates, demographics


def patient_timeline(icustay_id: int = 200_000, window_hours: int = 48,
                     seed: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Illustrative raw (day_fraction, value, concept) stream for ONE stay.

    Used only so the per-patient EDA plot renders on the synthetic path; on the
    BigQuery path the real :func:`src.data.sql.patient_timeline_query` is used.
    """
    rng = np.random.default_rng(seed + int(icustay_id))
    rows = []
    for concept in ["heart_rate", "sbp", "resp_rate", "spo2", "glucose"]:
        lo, hi = CONCEPT_VALID_RANGES[concept]
        base, slope = _CONCEPT_SEVERITY[concept]
        n = rng.integers(20, 60)
        t = np.sort(rng.uniform(0, window_hours / 24.0, size=n))
        vals = np.clip(base + rng.normal(0, abs(slope) * 0.4 + 2, size=n), lo, hi)
        for ti, vi in zip(t, vals):
            rows.append((float(ti), float(vi), concept))
    return pd.DataFrame(rows, columns=["day_fraction", "value", "concept"])
