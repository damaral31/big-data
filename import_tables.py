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

Tables already present in BigQuery (with rows) are skipped; pass --force to
reload them.

Usage:
    export GCP_PROJECT_ID=...   GOOGLE_APPLICATION_CREDENTIALS=./gcp_key.json
    python import_tables.py                  # load everything (skip what exists)
    python import_tables.py icustays d_items # load a subset
    python import_tables.py --force          # reload everything, even if present
"""
from __future__ import annotations

import os
import sys
import time

import requests
from google.cloud import bigquery, storage

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; we fall back to periodic prints
    tqdm = None


class _ProgressReader:
    """File-like wrapper that reports bytes as a stream is read (for uploads).

    ``google-cloud-storage`` reads from the file object in chunks while uploading;
    we forward ``read`` and tick a tqdm bar (or print every ~50 MB if tqdm is
    absent). Total size comes from the HTTP ``Content-Length`` header.
    """

    def __init__(self, raw, total: int | None, desc: str):
        self._raw = raw
        self._total = total
        self._seen = 0
        self._last = 0
        self._bar = (tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                          desc=desc, leave=False) if tqdm else None)

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self._seen += len(chunk)
        if self._bar is not None:
            self._bar.update(len(chunk))
        elif self._seen - self._last >= 50 * 1024 * 1024:  # every ~50 MB
            self._last = self._seen
            mb = self._seen / 1e6
            pct = f" ({100 * self._seen / self._total:.0f}%)" if self._total else ""
            tot = f" / {self._total / 1e6:.0f} MB" if self._total else ""
            print(f"      uploaded {mb:.0f} MB{tot}{pct}", flush=True)
        return chunk

    def tell(self) -> int:
        return self._seen

    def readable(self) -> bool:
        return True

    def close(self):
        if self._bar is not None:
            self._bar.close()

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "big-data-497416")
MIMIC_DATASET = os.environ.get("MIMIC_DATASET", "mimic_prod")
BUCKET_NAME = os.environ.get("MIMIC_BUCKET", "big-data-visty-up")
BASE_URL = "https://www.dcc.fc.up.pt/~ines/MIMIC-III"

# Point the clients at a service-account key if one isn't already configured via
# GOOGLE_APPLICATION_CREDENTIALS (mirrors src/config.py credential discovery).
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    for _cand in ("./gcp_key.json", os.path.join(os.path.dirname(__file__), "gcp_key.json"),
                  os.path.join(os.path.expanduser("~"), "gcp_key.json")):
        if os.path.exists(_cand):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(_cand)
            print(f"Using credentials: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
            break
    else:
        print("WARNING: no gcp_key.json found and GOOGLE_APPLICATION_CREDENTIALS unset; "
              "BigQuery auth will fail. Place gcp_key.json at the project root.")

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
    "d_labitems": [
        S("ROW_ID", INT, REQ), S("ITEMID", INT, REQ), S("LABEL", STR, NUL),
        S("FLUID", STR, NUL), S("CATEGORY", STR, NUL), S("LOINC_CODE", STR, NUL),
    ],
    "labevents": [
        S("ROW_ID", INT, REQ), S("SUBJECT_ID", INT, REQ), S("HADM_ID", INT, NUL),
        S("ITEMID", INT, NUL), S("CHARTTIME", TS, NUL), S("VALUE", STR, NUL),
        S("VALUENUM", FLT, NUL), S("VALUEUOM", STR, NUL), S("FLAG", STR, NUL),
    ],
}

LOAD_ORDER = ["patients", "admissions", "icustays", "d_items", "chartevents",
              "d_labitems", "labevents"]


def stream_to_gcs(table: str) -> str:
    blob_name = f"{table.upper()}.csv.gz"
    blob = gcs.bucket(BUCKET_NAME).blob(blob_name)
    if blob.exists():
        print(f"  · gs://{BUCKET_NAME}/{blob_name} already present, reusing")
        return f"gs://{BUCKET_NAME}/{blob_name}"
    url = f"{BASE_URL}/{blob_name}"
    print(f"  > streaming {url} -> gs://{BUCKET_NAME}/{blob_name}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        r.raw.decode_content = False           # upload the raw .gz bytes as-is
        reader = _ProgressReader(r.raw, total, f"  upload {blob_name}")
        try:
            blob.upload_from_file(reader, content_type="application/gzip")
        finally:
            reader.close()
    return f"gs://{BUCKET_NAME}/{blob_name}"


def bq_row_count(table_id: str) -> int | None:
    """Return the table's row count, or None if it doesn't exist yet."""
    try:
        return bq.get_table(table_id).num_rows
    except Exception:
        return None


def load_table(table: str, idx: int = 0, n_tables: int = 0, force: bool = False) -> None:
    tag = f"[{idx}/{n_tables}] {table}" if n_tables else f"[{table}]"
    print(tag)
    table_id = f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.{table}"
    if not force:
        existing = bq_row_count(table_id)
        if existing:
            print(f"  · already in BigQuery ({existing:,} rows), skipping (use --force to reload)\n")
            return
    uri = stream_to_gcs(table)
    job_config = bigquery.LoadJobConfig(
        schema=SCHEMAS[table],
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        allow_quoted_newlines=True,
        write_disposition="WRITE_TRUNCATE",
        max_bad_records=100,  # tolerate a handful of malformed rows in the mirror
    )
    job = bq.load_table_from_uri(uri, table_id, job_config=job_config)
    start = time.time()
    while not job.done():                      # poll so the user sees it is alive
        print(f"\r  indexing in BigQuery... {time.time() - start:4.0f}s", end="", flush=True)
        time.sleep(2)
    job.result()                               # surface any load error
    n = bq.get_table(table_id).num_rows
    print(f"\r  OK {table_id}: {n:,} rows  ({time.time() - start:.0f}s)        \n")


def main(tables: list[str], force: bool = False) -> None:
    bq.create_dataset(MIMIC_DATASET, exists_ok=True)
    valid = [t for t in tables if t in SCHEMAS]
    for bad in [t for t in tables if t not in SCHEMAS]:
        print(f"!! unknown table '{bad}', skipping")
    for i, t in enumerate(valid, 1):
        load_table(t, i, len(valid), force=force)
    print("Done.")


if __name__ == "__main__":
    argv = [a.lower() for a in sys.argv[1:]]
    force = any(a in ("--force", "-f") for a in argv)
    requested = [a for a in argv if not a.startswith("-")] or LOAD_ORDER
    main(requested, force=force)
