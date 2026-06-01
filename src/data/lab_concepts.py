"""LABEVENTS ITEMID harmonization (the D_LABITEMS dictionary).

Laboratory results live in LABEVENTS, keyed by SUBJECT_ID / HADM_ID (note: there
is **no ICUSTAY_ID** -- labs are recorded at the hospital-admission level, so the
aggregation query joins on HADM_ID and time-windows against the ICU INTIME; see
:func:`src.data.sql.window_lab_aggregates_query`).

ITEMIDs come from D_LABITEMS (the 50000 range) and are *different* from the
CHARTEVENTS / D_ITEMS dictionary used in :mod:`src.data.concepts`.

Unlike CHARTEVENTS, LABEVENTS has **no ERROR column** -- it has a FLAG
('abnormal') which is clinically meaningful, NOT a data-quality flag, so we must
never filter on it.  We only drop NULL values and physiologically impossible ones.

The labs below are the high-yield panel for severity/LOS: renal function,
electrolytes, perfusion (lactate), haematology and liver/coagulation.
Reference: MIMIC-III D_LABITEMS, https://mimic.mit.edu/docs/iii/tables/d_labitems/
"""
from __future__ import annotations

# concept name -> list of equivalent LABEVENTS ITEMIDs
CONCEPT_ITEMIDS: dict[str, list[int]] = {
    "creatinine":  [50912],
    "bun":         [51006],            # urea nitrogen (renal function)
    "sodium":      [50983, 50824],
    "potassium":   [50971, 50822],
    "chloride":    [50902, 50806],
    "bicarbonate": [50882],
    "anion_gap":   [50868],
    "glucose":     [50931],            # serum glucose (lab; prefixed lab_ in features)
    "hemoglobin":  [51222],
    "hematocrit":  [51221],
    "wbc":         [51301, 51300],     # white blood cell count
    "platelets":   [51265],
    "lactate":     [50813],            # perfusion / shock marker
    "bilirubin":   [50885],            # total bilirubin (liver)
    "inr":         [51237],            # coagulation
    "ptt":         [51275],
    "albumin":     [50862],
    "calcium":     [50893],            # total calcium
    "magnesium":   [50960],
    "ph":          [50820],            # arterial blood gas pH
}

ITEMID_TO_CONCEPT: dict[int, str] = {
    itemid: concept
    for concept, itemids in CONCEPT_ITEMIDS.items()
    for itemid in itemids
}
ALL_ITEMIDS: list[int] = sorted(ITEMID_TO_CONCEPT)

# generous plausibility ranges; values outside are treated as charting errors
CONCEPT_VALID_RANGES: dict[str, tuple[float, float]] = {
    "creatinine":  (0.1, 50),
    "bun":         (1, 300),
    "sodium":      (80, 200),
    "potassium":   (1, 12),
    "chloride":    (50, 200),
    "bicarbonate": (2, 60),
    "anion_gap":   (1, 50),
    "glucose":     (5, 2000),
    "hemoglobin":  (1, 30),
    "hematocrit":  (5, 80),
    "wbc":         (0.1, 200),
    "platelets":   (1, 2000),
    "lactate":     (0.1, 50),
    "bilirubin":   (0.1, 80),
    "inr":         (0.1, 30),
    "ptt":         (10, 250),
    "albumin":     (0.5, 8),
    "calcium":     (2, 20),
    "magnesium":   (0.1, 10),
    "ph":          (6.5, 8.0),
}
