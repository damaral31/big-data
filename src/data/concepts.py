"""CareVue / MetaVision ITEMID harmonization for CHARTEVENTS.

MIMIC-III stitches together two source ICU systems:

* **CareVue**   (Philips, pre-2008)  -> ITEMIDs < 220000
* **MetaVision** (iMDsoft, 2008+)    -> ITEMIDs >= 220000

The *same* clinical concept (e.g. heart rate) therefore has several ITEMIDs
(211 in CareVue, 220045 in MetaVision, ...).  If you aggregate raw ITEMIDs you
end up with duplicate, half-empty columns and you cannot compare patients
across the two systems.  We map the most common bedside vitals to a single
canonical concept name and aggregate on that.

The list below covers the high-yield vitals that actually live in CHARTEVENTS.
Laboratory results (creatinine, BUN, sodium, ...) live in LABEVENTS, which this
project does not load, so they are intentionally excluded.

References:
* MIMIC-III schema / D_ITEMS, https://mimic.mit.edu/docs/iii/tables/d_items/
* mimic-code issue #472 (CareVue vs MetaVision itemids).
"""
from __future__ import annotations

# concept name -> list of equivalent ITEMIDs (CareVue first, MetaVision after)
CONCEPT_ITEMIDS: dict[str, list[int]] = {
    "heart_rate":        [211, 220045],
    "sbp":               [51, 442, 455, 6701, 220179, 220050],
    "dbp":               [8368, 8440, 8441, 8555, 220180, 220051],
    "mbp":               [456, 52, 6702, 443, 220052, 220181, 225312],
    "resp_rate":         [615, 618, 220210, 224690],
    "spo2":              [646, 220277],
    "temp_c":            [676, 223762],
    "temp_f":            [678, 223761],
    "glucose":           [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
    "gcs_total":         [198, 226755],          # CareVue total GCS / MetaVision calc
    "fio2":              [223835, 3420, 3422, 190],
    "weight_kg":         [763, 224639, 226512],
    "tidal_volume":      [681, 682, 683, 224685, 224684, 224686],
}

# inverse map: itemid -> canonical concept (built once at import)
ITEMID_TO_CONCEPT: dict[int, str] = {
    itemid: concept
    for concept, itemids in CONCEPT_ITEMIDS.items()
    for itemid in itemids
}

# all itemids we care about (used to push a WHERE filter into BigQuery)
ALL_ITEMIDS: list[int] = sorted(ITEMID_TO_CONCEPT)

# Plausibility ranges per concept; values outside are treated as charting errors
# and dropped before aggregation.  Ranges are deliberately generous.
CONCEPT_VALID_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate":   (10, 300),
    "sbp":          (20, 300),
    "dbp":          (5, 200),
    "mbp":          (10, 250),
    "resp_rate":    (1, 80),
    "spo2":         (10, 100),
    "temp_c":       (25, 45),
    "temp_f":       (77, 113),
    "glucose":      (10, 2000),
    "gcs_total":    (3, 15),
    "fio2":         (0.21, 100),   # accepts both fraction(<=1) and percent
    "weight_kg":    (20, 400),
    "tidal_volume": (50, 2000),
}
