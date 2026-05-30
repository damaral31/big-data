"""Central configuration for the MIMIC-III ICU Length-of-Stay pipeline.

Every tunable lives here so the notebook and the ``src`` modules share a single
source of truth.  Nothing in this file performs I/O or touches the network.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # parquet cache of (expensive) BQ results
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (DATA_DIR, CACHE_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Google Cloud / BigQuery
# --------------------------------------------------------------------------- #
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "big-data-497416")
MIMIC_DATASET = os.environ.get("MIMIC_DATASET", "mimic_prod")

# Locate a service-account key without hard-coding one machine's layout.
_KEY_CANDIDATES = [
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
    "./gcp_key.json",
    str(PROJECT_ROOT / "gcp_key.json"),
    str(Path.home() / "gcp_key.json"),
]
GCP_CREDENTIALS_PATH = next((p for p in _KEY_CANDIDATES if p and os.path.exists(p)), None)


def table(name: str) -> str:
    """Fully-qualified BigQuery table id, e.g. ``project.dataset.icustays``."""
    return f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.{name}"


CHARTEVENTS_TABLE = table("chartevents")
D_ITEMS_TABLE = table("d_items")
ADMISSIONS_TABLE = table("admissions")
ICUSTAYS_TABLE = table("icustays")
PATIENTS_TABLE = table("patients")

# --------------------------------------------------------------------------- #
# Cohort / target definition
# --------------------------------------------------------------------------- #
# Target = ICU length of stay in DAYS (ICUSTAYS.OUTTIME - INTIME), one row per
# ICUSTAY_ID.  We predict at the PREDICTION_WINDOW_HOURS mark, so the cohort is
# restricted to stays that are still ongoing at that point (see MIN_LOS_HOURS).
PREDICTION_WINDOW_HOURS = 24      # only data from the first N hours feeds the model
MIN_LOS_HOURS = 24                # exclude stays shorter than the window (truncated)
MAX_LOS_DAYS = 60                 # clip absurd/erroneous stays (data-quality guard)

# Classification buckets (days). Edges are [0, short), [short, medium), [medium, inf)
LOS_SHORT_DAYS = 3
LOS_MEDIUM_DAYS = 7
LOS_CLASS_LABELS = ("short", "medium", "long")

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
# Per-concept aggregates kept from the first-window measurements.
AGG_FUNCS = ("mean", "min", "max", "std", "count")
# Drop engineered columns missing in more than this fraction of stays.
MAX_MISSING_FRACTION = 0.60
AGE_CAP = 90.0                    # MIMIC shifts DOB of >89y patients ~300y; cap it.

# --------------------------------------------------------------------------- #
# Modelling / validation
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20                  # grouped hold-out by SUBJECT_ID
CV_FOLDS = 5                      # GroupKFold for cross-validation
HP_SEARCH_ITER = 20              # RandomizedSearchCV budget for the tuned model
N_JOBS = -1

# --------------------------------------------------------------------------- #
# Development / sampling
# --------------------------------------------------------------------------- #
# Cap rows pulled when developing against BigQuery (None = full cohort).
DEV_ICUSTAY_LIMIT = None
SYNTHETIC_N_STAYS = 4000          # size of the labelled synthetic fallback cohort
CACHE_QUERIES = True
VERBOSE = True
