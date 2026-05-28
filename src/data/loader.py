# Data loading from BigQuery and MIMIC-III
import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None

from config import (
    GCP_CREDENTIALS_PATH, GCP_PROJECT_ID, CHARTEVENTS_TABLE,
    D_ITEMS_TABLE, ADMISSIONS_TABLE, ICUSTAYS_TABLE, PATIENTS_TABLE,
    CACHE_DIR, CACHE_QUERIES, VERBOSE
)

logging.basicConfig(level=logging.INFO if VERBOSE else logging.WARNING)
logger = logging.getLogger(__name__)

class MIMICDataLoader:
    """Load and manage MIMIC-III data from BigQuery"""

    def __init__(self):
        """Initialize BigQuery client"""
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize BigQuery client with authentication"""
        if not os.path.exists(GCP_CREDENTIALS_PATH):
            logger.warning(f"GCP credentials not found at {GCP_CREDENTIALS_PATH}")
            logger.info("Attempting to use default credentials...")

        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
            self.client = bigquery.Client(project=GCP_PROJECT_ID)
            logger.info(f"BigQuery client initialized for project {GCP_PROJECT_ID}")
        except Exception as e:
            logger.warning(f"Failed to initialize BigQuery client: {e}")
            logger.info("Data loading will be unavailable")

    def _get_cache_path(self, query_name: str) -> Path:
        """Get cache file path for query"""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"{query_name}_{datetime.now().strftime('%Y%m%d')}.parquet"

    def _load_cached_query(self, query_name: str) -> Optional[pd.DataFrame]:
        """Load cached query result"""
        if not CACHE_QUERIES:
            return None

        cache_path = self._get_cache_path(query_name)
        if cache_path.exists():
            logger.info(f"Loading cached query: {query_name}")
            return pd.read_parquet(cache_path)
        return None

    def _cache_query_result(self, query_name: str, df: pd.DataFrame):
        """Cache query result"""
        if not CACHE_QUERIES:
            return

        cache_path = self._get_cache_path(query_name)
        df.to_parquet(cache_path)
        logger.info(f"Cached query result: {query_name} ({len(df)} rows)")

    def query_bigquery(self, query: str, query_name: str = "query") -> pd.DataFrame:
        """Execute BigQuery query with caching"""
        # Try to load from cache first
        cached_df = self._load_cached_query(query_name)
        if cached_df is not None:
            return cached_df

        if self.client is None:
            raise RuntimeError("BigQuery client not initialized")

        logger.info(f"Executing BigQuery query: {query_name}")
        df = self.client.query(query).to_dataframe()
        logger.info(f"Query returned {len(df)} rows")

        self._cache_query_result(query_name, df)
        return df

    def get_patient_demographics(self, subject_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load patient demographics"""
        if subject_ids:
            ids_str = ",".join(map(str, subject_ids))
            where_clause = f"WHERE subject_id IN ({ids_str})"
        else:
            where_clause = ""

        query = f"""
        SELECT DISTINCT
            subject_id,
            gender,
            dob,
            dod
        FROM `{PATIENTS_TABLE}`
        {where_clause}
        """

        return self.query_bigquery(query, "patient_demographics")

    def get_admissions(self, subject_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load admission records"""
        if subject_ids:
            ids_str = ",".join(map(str, subject_ids))
            where_clause = f"WHERE subject_id IN ({ids_str})"
        else:
            where_clause = ""

        query = f"""
        SELECT
            hadm_id,
            subject_id,
            admittime,
            dischtime,
            admission_type,
            TIMESTAMP_DIFF(dischtime, admittime, HOUR) / 24.0 AS length_of_stay
        FROM `{ADMISSIONS_TABLE}`
        {where_clause}
        ORDER BY subject_id, admittime
        """

        return self.query_bigquery(query, "admissions")

    def get_icu_stays(self, subject_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load ICU stay records"""
        if subject_ids:
            ids_str = ",".join(map(str, subject_ids))
            where_clause = f"WHERE subject_id IN ({ids_str})"
        else:
            where_clause = ""

        query = f"""
        SELECT
            icustay_id,
            hadm_id,
            subject_id,
            intime,
            outtime,
            TIMESTAMP_DIFF(outtime, intime, HOUR) / 24.0 AS los_icu_days
        FROM `{ICUSTAYS_TABLE}`
        {where_clause}
        ORDER BY subject_id, intime
        """

        return self.query_bigquery(query, "icu_stays")

    def get_chart_events(self, icustay_ids: List[int],
                         value_only: bool = True) -> pd.DataFrame:
        """Load chart events for specific ICU stays"""
        ids_str = ",".join(map(str, icustay_ids))

        value_filter = "AND c.VALUENUM IS NOT NULL" if value_only else ""

        query = f"""
        SELECT
            c.icustay_id,
            c.hadm_id,
            c.subject_id,
            c.charttime,
            c.itemid,
            c.valuenum AS value,
            c.valueuom,
            d.label,
            d.category
        FROM `{CHARTEVENTS_TABLE}` c
        LEFT JOIN `{D_ITEMS_TABLE}` d ON c.itemid = d.itemid
        WHERE c.icustay_id IN ({ids_str})
        {value_filter}
        ORDER BY c.charttime
        """

        return self.query_bigquery(query, f"chart_events_{len(icustay_ids)}_stays")

    def get_items_catalog(self) -> pd.DataFrame:
        """Load item catalog (measurements reference)"""
        query = f"""
        SELECT DISTINCT
            itemid,
            label,
            abbreviation,
            category,
            unitname
        FROM `{D_ITEMS_TABLE}`
        ORDER BY category, label
        """

        return self.query_bigquery(query, "items_catalog")

    def sample_patient_data(self, n_samples: int = 100,
                            min_los_hours: float = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sample patients for analysis (useful for development)"""
        query = f"""
        WITH sampled_stays AS (
            SELECT
                i.subject_id,
                i.hadm_id,
                i.icustay_id,
                i.intime,
                i.outtime,
                TIMESTAMP_DIFF(i.outtime, i.intime, HOUR) AS los_hours
            FROM `{ICUSTAYS_TABLE}` i
            WHERE TIMESTAMP_DIFF(i.outtime, i.intime, HOUR) >= {min_los_hours}
            ORDER BY RAND()
            LIMIT {n_samples}
        )
        SELECT * FROM sampled_stays
        """

        return self.query_bigquery(query, f"sample_stays_{n_samples}")
