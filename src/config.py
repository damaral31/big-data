# Configuration for MIMIC-III LOS Prediction Project
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# GCP Configuration
GCP_PROJECT_ID = "big-data-497416"
# Try to find gcp_key.json in multiple locations
_locations = [
    "./gcp_key.json",  # Root directory
    "../gcp_key.json",  # Parent directory (if running from notebooks/)
    "gcp_key.json",  # Current directory
    os.path.join(os.path.expanduser("~"), "gcp_key.json"),  # Home directory
    os.path.join(PROJECT_ROOT, "gcp_key.json"),  # Project root
]

GCP_CREDENTIALS_PATH = None
for path in _locations:
    if os.path.exists(path):
        GCP_CREDENTIALS_PATH = os.path.abspath(path)
        break

if GCP_CREDENTIALS_PATH is None:
    GCP_CREDENTIALS_PATH = "./gcp_key.json"  # Default fallback

# BigQuery Configuration
MIMIC_DATASET = "mimic_prod"
CHARTEVENTS_TABLE = f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.chartevents"
D_ITEMS_TABLE = f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.d_items"
ADMISSIONS_TABLE = f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.admissions"
ICUSTAYS_TABLE = f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.icustays"
PATIENTS_TABLE = f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.patients"

# Data Processing Parameters
CHUNK_SIZE = 100000  # Rows per batch processing
CACHE_QUERIES = True  # Cache BigQuery results locally
USE_SPARK = False     # Use PySpark for processing (can be enabled)

# Target Variable
TARGET_VARIABLE = "length_of_stay"  # Prediction target
LOS_THRESHOLD_SHORT = 3    # Days: short stay
LOS_THRESHOLD_MEDIUM = 7   # Days: medium stay
# Long stay: > 7 days

# Feature Engineering
WINDOW_SIZE_HOURS = 24     # Initial window for feature extraction (1 day)
LOOKBACK_WINDOW = [6, 12, 24, 48, 72]  # Lookback windows in hours
MISSING_VALUE_THRESHOLD = 0.5  # Drop features with >50% missing

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
CV_FOLDS = 5

# Hyperparameter tuning
HP_SEARCH_CV_FOLDS = 3
HP_SEARCH_ITERATIONS = 20

# Performance profiling
PROFILE_EXECUTION = True
VERBOSE = True
