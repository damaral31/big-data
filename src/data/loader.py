"""Data-loading facade: BigQuery first, synthetic fallback.

``load_cohort`` is the single entry point used by the notebook.  It returns the
same three frames regardless of source so everything downstream is identical:

    cohort        : icustay_id, subject_id, hadm_id, intime, los_icu_days
    aggregates    : icustay_id, concept, n, mean, min, max, std   (long format)
    demographics  : icustay_id, subject_id, hadm_id, gender, admission_type, ...

It also reports which source was used so the notebook can print an honest banner.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .. import config as cfg
from . import sql, synthetic

logger = logging.getLogger(__name__)

try:  # optional dependency -- only needed for the real path
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


@dataclass
class CohortData:
    cohort: pd.DataFrame
    aggregates: pd.DataFrame          # CHARTEVENTS vitals (long)
    demographics: pd.DataFrame
    source: str                       # "BIGQUERY" or "SYNTHETIC"
    lab_aggregates: pd.DataFrame = field(default_factory=pd.DataFrame)  # LABEVENTS (long)

    @property
    def is_real(self) -> bool:
        return self.source == "BIGQUERY"


class BigQueryClient:
    """Thin wrapper around the BigQuery client with parquet result caching."""

    def __init__(self):
        if bigquery is None:
            raise RuntimeError("google-cloud-bigquery is not installed")
        if cfg.GCP_CREDENTIALS_PATH:
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.GCP_CREDENTIALS_PATH
        self.client = bigquery.Client(project=cfg.GCP_PROJECT_ID)
        logger.info("BigQuery client ready (project=%s)", cfg.GCP_PROJECT_ID)

    def _cache_path(self, query: str, tag: str) -> Path:
        h = hashlib.md5(query.encode()).hexdigest()[:10]
        return cfg.CACHE_DIR / f"{tag}_{h}.parquet"

    def run(self, query: str, tag: str) -> pd.DataFrame:
        cache = self._cache_path(query, tag)
        if cfg.CACHE_QUERIES and cache.exists():
            logger.info("Loading cached %s (%s)", tag, cache.name)
            return pd.read_parquet(cache)

        logger.info("Running BigQuery job: %s", tag)
        df = self.client.query(query).to_dataframe()
        df.columns = [c.lower() for c in df.columns]
        logger.info("  -> %s rows", len(df))
        if cfg.CACHE_QUERIES:
            df.to_parquet(cache, index=False)
        return df


def _try_bigquery(window_hours: int, limit: int | None,
                  include_labs: bool) -> CohortData | None:
    if bigquery is None or cfg.GCP_CREDENTIALS_PATH is None:
        logger.info("BigQuery unavailable (no client or no credentials).")
        return None
    try:
        client = BigQueryClient()
        cohort = client.run(sql.cohort_query(limit), "cohort")
        aggregates = client.run(
            sql.window_aggregates_query(window_hours, limit), "agg")
        demographics = client.run(sql.demographics_query(limit), "demo")
        labs = pd.DataFrame()
        if include_labs:
            labs = client.run(
                sql.window_lab_aggregates_query(window_hours, limit), "lab_agg")
        return CohortData(cohort, aggregates, demographics, "BIGQUERY",
                          lab_aggregates=labs)
    except Exception as exc:  # pragma: no cover - network/credential dependent
        logger.warning("BigQuery path failed (%s); falling back to synthetic.", exc)
        return None


def load_all_los_hours(use_bigquery: bool = True,
                       limit: int | None = None) -> pd.Series:
    """Unfiltered ICU LOS in hours (for the Phase-1 window-justification plot)."""
    if use_bigquery and bigquery is not None and cfg.GCP_CREDENTIALS_PATH is not None:
        try:
            client = BigQueryClient()
            df = client.run(sql.all_los_hours_query(limit), "all_los_hours")
            return df["los_hours"]
        except Exception as exc:  # pragma: no cover
            logger.warning("all_los_hours BigQuery failed (%s); using synthetic.", exc)
    return synthetic.all_los_hours(limit or cfg.SYNTHETIC_N_STAYS)


def load_cohort(
    use_bigquery: bool = True,
    window_hours: int = cfg.PREDICTION_WINDOW_HOURS,
    limit: int | None = cfg.DEV_ICUSTAY_LIMIT,
    include_labs: bool = cfg.INCLUDE_LABS,
) -> CohortData:
    """Load cohort + first-window vital aggregates + demographics + lab aggregates.

    Tries BigQuery when ``use_bigquery`` and credentials exist; otherwise (or on
    any failure) returns a clearly-labelled synthetic cohort of the same shape.
    ``include_labs`` controls whether the LABEVENTS aggregation is loaded.
    """
    if use_bigquery:
        data = _try_bigquery(window_hours, limit, include_labs)
        if data is not None:
            return data

    logger.warning("Using SYNTHETIC data -- results are for demonstration only.")
    n = limit or cfg.SYNTHETIC_N_STAYS
    cohort, aggregates, demographics, labs = synthetic.generate(n_stays=n)
    if not include_labs:
        labs = pd.DataFrame()
    return CohortData(cohort, aggregates, demographics, "SYNTHETIC",
                      lab_aggregates=labs)
