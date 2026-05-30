"""Load the MIMIC-III tables this project needs into BigQuery.

Pipeline per table:
  1. stream the .csv.gz from the U.Porto mirror straight into a GCS bucket
     (no large local download), then
  2. create/replace a BigQuery table from that GCS object.

Tables loaded (everything the LOS pipeline queries):
  patients, admissions, icustays, d_items, chartevents (~4.2 GB, ~330M rows).

Schemas use NULLABLE for non-key columns on purpose: the official MIMIC DDL
marks several columns NOT NULL, but the public CSV mirror contains rows that
violate a few of those (e.g. ADMISSION_LOCATION, ICUSTAY_ID in CHARTEVENTS),
which makes a strict load fail. Keys stay REQUIRED.

Usage:
    export GCP_PROJECT_ID=...   GOOGLE_APPLICATION_CREDENTIALS=./gcp_key.json
    python import_tables.py                  # load everything
    python import_tables.py icustays d_items # load a subset
"""
from __future__ import annotations

import os
import sys

import requests
from google.cloud import bigquery, storage

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "big-data-497416")
MIMIC_DATASET = os.environ.get("MIMIC_DATASET", "mimic_prod")
BUCKET_NAME = os.environ.get("MIMIC_BUCKET", "big-data-visty-up")
BASE_URL = "https://www.dcc.fc.up.pt/~ines/MIMIC-III"

bq = bigquery.Client(project=GCP_PROJECT_ID)
gcs = storage.Client(project=GCP_PROJECT_ID)

S = bigquery.SchemaField
REQ, NUL = "REQUIRED", "NULLABLE"
TS, INT, FLT, STR = "TIMESTAMP", "INTEGER", "FLOAT", "STRING"

SCHEMAS: dict[str, list] = {
    "patients": [
        S("ROW_ID", INT, REQ), S("SUBJECT_ID", INT, REQ), S("GENDER", STR, NUL),
        S("DOB", TS, NUL), S("DOD", TS, NUL), S("DOD_HOSP", TS, NUL),
        S("DOD_SSN", TS, NUL), S("EXPIRE_FLAG", INT, NUL),
    ],
    "admissions": [
        S("ROW_ID", INT, REQ), S("SUBJECT_ID", INT, REQ), S("HADM_ID", INT, REQ),
        S("ADMITTIME", TS, NUL), S("DISCHTIME", TS, NUL), S("DEATHTIME", TS, NUL),
        S("ADMISSION_TYPE", STR, NUL), S("ADMISSION_LOCATION", STR, NUL),
        S("DISCHARGE_LOCATION", STR, NUL), S("INSURANCE", STR, NUL),
        S("LANGUAGE", STR, NUL), S("RELIGION", STR, NUL),
        S("MARITAL_STATUS", STR, NUL), S("ETHNICITY", STR, NUL),
        S("EDREGTIME", TS, NUL), S("EDOUTTIME", TS, NUL), S("DIAGNOSIS", STR, NUL),
        S("HOSPITAL_EXPIRE_FLAG", INT, NUL), S("HAS_CHARTEVENTS_DATA", INT, NUL),
    ],
    "icustays": [
        S("ROW_ID", INT, REQ), S("SUBJECT_ID", INT, REQ), S("HADM_ID", INT, REQ),
        S("ICUSTAY_ID", INT, REQ), S("DBSOURCE", STR, NUL),
        S("FIRST_CAREUNIT", STR, NUL), S("LAST_CAREUNIT", STR, NUL),
        S("FIRST_WARDID", INT, NUL), S("LAST_WARDID", INT, NUL),
        S("INTIME", TS, NUL), S("OUTTIME", TS, NUL), S("LOS", FLT, NUL),
    ],
    "d_items": [
        S("ROW_ID", INT, REQ), S("ITEMID", INT, REQ), S("LABEL", STR, NUL),
        S("ABBREVIATION", STR, NUL), S("DBSOURCE", STR, NUL),
        S("LINKSTO", STR, NUL), S("CATEGORY", STR, NUL), S("UNITNAME", STR, NUL),
        S("PARAM_TYPE", STR, NUL), S("CONCEPTID", INT, NUL),
    ],
    "chartevents": [
        S("ROW_ID", INT, REQ), S("SUBJECT_ID", INT, REQ), S("HADM_ID", INT, NUL),
        S("ICUSTAY_ID", INT, NUL), S("ITEMID", INT, NUL), S("CHARTTIME", TS, NUL),
        S("STORETIME", TS, NUL), S("CGID", INT, NUL), S("VALUE", STR, NUL),
        S("VALUENUM", FLT, NUL), S("VALUEUOM", STR, NUL), S("WARNING", INT, NUL),
        S("ERROR", INT, NUL), S("RESULTSTATUS", STR, NUL), S("STOPPED", STR, NUL),
    ],
}

LOAD_ORDER = ["patients", "admissions", "icustays", "d_items", "chartevents"]


def stream_to_gcs(table: str) -> str:
    blob_name = f"{table.upper()}.csv.gz"
    blob = gcs.bucket(BUCKET_NAME).blob(blob_name)
    if blob.exists():
        print(f"  · gs://{BUCKET_NAME}/{blob_name} already present, reusing")
        return f"gs://{BUCKET_NAME}/{blob_name}"
    url = f"{BASE_URL}/{blob_name}"
    print(f"  ↳ streaming {url} -> gs://{BUCKET_NAME}/{blob_name}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        blob.upload_from_file(r.raw, content_type="application/gzip")
    return f"gs://{BUCKET_NAME}/{blob_name}"


def load_table(table: str) -> None:
    print(f"[{table}]")
    uri = stream_to_gcs(table)
    table_id = f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.{table}"
    job_config = bigquery.LoadJobConfig(
        schema=SCHEMAS[table],
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        allow_quoted_newlines=True,
        write_disposition="WRITE_TRUNCATE",
        max_bad_records=100,  # tolerate a handful of malformed rows in the mirror
    )
    job = bq.load_table_from_uri(uri, table_id, job_config=job_config)
    job.result()
    n = bq.get_table(table_id).num_rows
    print(f"  ✓ {table_id}: {n:,} rows\n")


def main(tables: list[str]) -> None:
    bq.create_dataset(MIMIC_DATASET, exists_ok=True)
    for t in tables:
        if t not in SCHEMAS:
            print(f"!! unknown table '{t}', skipping")
            continue
        load_table(t)
    print("Done.")


if __name__ == "__main__":
    requested = [t.lower() for t in sys.argv[1:]] or LOAD_ORDER
    main(requested)
