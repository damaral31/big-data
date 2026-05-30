"""BigQuery SQL builders.

The whole point of this project is that CHARTEVENTS has ~330 million rows
(4.2 GB compressed).  We never download it.  Instead we push the heavy lifting
-- cohort selection, first-window filtering, error/range cleaning, ITEMID
harmonization and per-concept aggregation -- into BigQuery and pull back a
*compact* table with at most a few hundred-thousand rows
(one per icustay x concept).

All queries are generated from :mod:`src.config` and :mod:`src.data.concepts`
so there is a single source of truth for cohort rules and itemid mappings.
"""
from __future__ import annotations

from .. import config as cfg
from .concepts import CONCEPT_ITEMIDS, CONCEPT_VALID_RANGES


def _concept_map_cte() -> str:
    """Build an inline (itemid -> concept, valid-range) lookup table.

    Emitted as ``UNNEST([STRUCT(...)])`` so no extra table needs to exist in
    BigQuery and the mapping always matches :mod:`src.data.concepts`.
    """
    rows = []
    for concept, itemids in CONCEPT_ITEMIDS.items():
        lo, hi = CONCEPT_VALID_RANGES[concept]
        for itemid in itemids:
            rows.append(
                f"STRUCT({itemid} AS itemid, '{concept}' AS concept, "
                f"{float(lo)} AS lo, {float(hi)} AS hi)"
            )
    return "SELECT * FROM UNNEST([\n        " + ",\n        ".join(rows) + "\n    ])"


def cohort_query(limit: int | None = None) -> str:
    """One row per qualifying ICU stay with the regression target (LOS in days).

    Cohort rules (all leak-safe -- they use OUTTIME only to *define* the label
    and to *select* rows, never as a feature):
      * OUTTIME present
      * stay length >= MIN_LOS_HOURS  (so a full first window exists)
      * stay length <= MAX_LOS_DAYS   (drop erroneous/extreme stays)
    """
    limit_sql = f"\nLIMIT {int(limit)}" if limit else ""
    return f"""
SELECT
    i.icustay_id,
    i.subject_id,
    i.hadm_id,
    i.intime,
    TIMESTAMP_DIFF(i.outtime, i.intime, SECOND) / 86400.0 AS los_icu_days
FROM `{cfg.ICUSTAYS_TABLE}` i
WHERE i.outtime IS NOT NULL
  AND TIMESTAMP_DIFF(i.outtime, i.intime, HOUR) >= {cfg.MIN_LOS_HOURS}
  AND TIMESTAMP_DIFF(i.outtime, i.intime, SECOND) / 86400.0 <= {cfg.MAX_LOS_DAYS}
ORDER BY i.icustay_id{limit_sql}
""".strip()


def window_aggregates_query(
    window_hours: int = cfg.PREDICTION_WINDOW_HOURS,
    limit: int | None = None,
) -> str:
    """Per-(icustay, concept) aggregates over the FIRST ``window_hours``.

    This is the expensive query and the one that demonstrates big-data handling:
    the 330M-row scan, the error/range filtering and the GROUP BY all execute in
    BigQuery; only the aggregated result (long format) is returned.

    Leakage controls baked into the WHERE clause:
      * charttime in [intime, intime + window) -- nothing after the window
      * error IS NULL OR error = 0             -- drop clinician-flagged errors
      * valuenum BETWEEN concept lo/hi          -- drop impossible values
    """
    cohort = cohort_query(limit=limit)
    return f"""
WITH concept_map AS (
    {_concept_map_cte()}
),
cohort AS (
    {cohort}
),
events AS (
    SELECT
        c.icustay_id,
        m.concept,
        c.valuenum
    FROM `{cfg.CHARTEVENTS_TABLE}` c
    JOIN cohort co        ON c.icustay_id = co.icustay_id
    JOIN concept_map m    ON c.itemid     = m.itemid
    WHERE c.valuenum IS NOT NULL
      AND (c.error IS NULL OR c.error = 0)
      AND c.charttime >= co.intime
      AND c.charttime <  TIMESTAMP_ADD(co.intime, INTERVAL {int(window_hours)} HOUR)
      AND c.valuenum BETWEEN m.lo AND m.hi
)
SELECT
    icustay_id,
    concept,
    COUNT(*)        AS n,
    AVG(valuenum)   AS mean,
    MIN(valuenum)   AS min,
    MAX(valuenum)   AS max,
    STDDEV(valuenum) AS std
FROM events
GROUP BY icustay_id, concept
""".strip()


def demographics_query(limit: int | None = None) -> str:
    """Static, admission-time demographics for the cohort.

    Age is computed from ADMITTIME - DOB.  MIMIC-III shifts the DOB of patients
    older than 89 by ~300 years, so ages come back as ~300 for them; we leave
    the raw value here and clip it to ``AGE_CAP`` during feature engineering.
    Only admission-time fields are selected -- nothing known only at discharge.
    """
    cohort = cohort_query(limit=limit)
    return f"""
WITH cohort AS (
    {cohort}
)
SELECT
    co.icustay_id,
    co.subject_id,
    co.hadm_id,
    p.gender,
    a.admission_type,
    a.admission_location,
    a.insurance,
    a.ethnicity,
    i.first_careunit,
    DATETIME_DIFF(a.admittime, p.dob, DAY) / 365.25 AS age_years
FROM cohort co
JOIN `{cfg.ICUSTAYS_TABLE}`  i ON co.icustay_id = i.icustay_id
JOIN `{cfg.ADMISSIONS_TABLE}` a ON co.hadm_id    = a.hadm_id
JOIN `{cfg.PATIENTS_TABLE}`   p ON co.subject_id = p.subject_id
""".strip()


def patient_timeline_query(icustay_id: int, window_hours: int | None = None) -> str:
    """Raw (charttime, value, concept) events for ONE stay, for the per-patient plot."""
    window_sql = ""
    if window_hours:
        window_sql = (
            f"  AND c.charttime < TIMESTAMP_ADD(i.intime, "
            f"INTERVAL {int(window_hours)} HOUR)\n"
        )
    return f"""
WITH concept_map AS (
    {_concept_map_cte()}
)
SELECT
    c.charttime,
    TIMESTAMP_DIFF(c.charttime, i.intime, SECOND) / 86400.0 AS day_fraction,
    m.concept,
    c.itemid,
    c.valuenum AS value
FROM `{cfg.CHARTEVENTS_TABLE}` c
JOIN `{cfg.ICUSTAYS_TABLE}` i ON c.icustay_id = i.icustay_id
JOIN concept_map m           ON c.itemid     = m.itemid
WHERE c.icustay_id = {int(icustay_id)}
  AND c.valuenum IS NOT NULL
  AND (c.error IS NULL OR c.error = 0)
  AND c.charttime >= i.intime
{window_sql}ORDER BY c.charttime
""".strip()
