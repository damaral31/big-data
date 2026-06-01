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
from .lab_concepts import CONCEPT_VALID_RANGES as LAB_VALID_RANGES

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

# Lab presence (labs are sampled less universally than vitals) and how strongly
# a second latent factor -- "organ dysfunction" -- shifts each.  organ is only
# partly correlated with the vitals-driven severity, so labs carry information
# the vitals alone do not -> the with-labs model genuinely improves.
_LAB_PRESENCE = {
    "creatinine": 0.92, "bun": 0.90, "sodium": 0.92, "potassium": 0.92,
    "chloride": 0.85, "bicarbonate": 0.85, "anion_gap": 0.70, "glucose": 0.88,
    "hemoglobin": 0.90, "hematocrit": 0.90, "wbc": 0.88, "platelets": 0.88,
    "lactate": 0.45, "bilirubin": 0.55, "inr": 0.60, "ptt": 0.55,
    "albumin": 0.45, "calcium": 0.80, "magnesium": 0.78, "ph": 0.40,
}
_LAB_ORGAN = {  # concept: (healthy_mean, organ_slope)
    "creatinine": (1.0, 0.9), "bun": (18, 9), "sodium": (139, -1.5),
    "potassium": (4.1, 0.3), "chloride": (104, -2), "bicarbonate": (24, -2.2),
    "anion_gap": (12, 2.0), "glucose": (128, 24), "hemoglobin": (12.2, -1.2),
    "hematocrit": (36, -3.5), "wbc": (9, 4.5), "platelets": (235, -42),
    "lactate": (1.6, 1.7), "bilirubin": (0.8, 0.9), "inr": (1.1, 0.5),
    "ptt": (32, 8), "albumin": (3.6, -0.5), "calcium": (8.8, -0.4),
    "magnesium": (2.0, 0.1), "ph": (7.40, -0.05),
}


def generate(n_stays: int = cfg.SYNTHETIC_N_STAYS, seed: int = cfg.RANDOM_STATE):
    """Return ``(cohort_df, aggregates_long_df, demographics_df)`` -- synthetic."""
    rng = np.random.default_rng(seed)

    # --- patients: ~1.4 stays per subject so grouped splitting is exercised ----
    n_subjects = int(n_stays / 1.4)
    subject_ids = 10_000 + rng.integers(0, n_subjects, size=n_stays)
    icustay_ids = np.arange(200_000, 200_000 + n_stays)
    hadm_ids = 100_000 + rng.integers(0, n_subjects, size=n_stays)

    # --- two latent factors ------------------------------------------------- #
    # severity drives the vitals + LOS; organ (renal/metabolic dysfunction)
    # drives the labs + LOS and is only partly correlated with severity, so labs
    # add predictive value beyond the vitals.
    severity = rng.normal(0, 1, size=n_stays)
    organ = 0.55 * severity + 0.83 * rng.normal(0, 1, size=n_stays)

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

    # right-skewed LOS (log-normal): vitals-severity + organ-dysfunction + admin
    log_los = 1.0 + 0.38 * severity + 0.22 * organ + adm_effect + age_effect \
        + rng.normal(0, 0.52, size=n_stays)
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

    # --- per-lab first-window aggregates (long format) ---------------------- #
    lab_rows = []
    for i in range(n_stays):
        o = organ[i]
        n_lab_base = max(1.0, 2.0 + 1.4 * o)  # labs drawn far less often than vitals
        for concept, present_p in _LAB_PRESENCE.items():
            if rng.random() > present_p:
                continue
            lo, hi = LAB_VALID_RANGES[concept]
            base, slope = _LAB_ORGAN[concept]
            mean = base + slope * o + rng.normal(0, abs(slope) * 0.45 + 1e-2)
            mean = float(np.clip(mean, lo, hi))
            spread = abs(rng.normal(0, abs(slope) * 0.5 + 1e-2)) + 1e-3
            vmin = float(np.clip(mean - spread, lo, hi))
            vmax = float(np.clip(mean + spread, lo, hi))
            n = int(max(1, rng.poisson(n_lab_base)))
            lab_rows.append((icustay_ids[i], concept, n, mean, vmin, vmax,
                             spread / 2.0))

    lab_aggregates = pd.DataFrame(
        lab_rows, columns=["icustay_id", "concept", "n", "mean", "min", "max", "std"]
    )
    return cohort, aggregates, demographics, lab_aggregates


def all_los_hours(n_stays: int = cfg.SYNTHETIC_N_STAYS, seed: int = cfg.RANDOM_STATE):
    """Synthetic UNFILTERED ICU LOS in hours (includes sub-24h stays).

    Mirrors :func:`src.data.sql.all_los_hours_query` for the Phase-1 window plot;
    ~20% of stays fall under 24h so the window-exclusion effect is visible.
    """
    rng = np.random.default_rng(seed + 1)
    los_days = np.exp(rng.normal(0.7, 0.9, size=n_stays))   # median ~2d, right-skewed
    hours = np.clip(los_days * 24.0, 1.0, cfg.MAX_LOS_DAYS * 24.0)
    return pd.Series(hours, name="los_hours")


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
