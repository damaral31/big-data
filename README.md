# MIMIC-III — ICU Length-of-Stay Prediction

Predict a patient's **ICU length of stay (LOS, in days)** from the **first 24 hours** of
bedside chart events in MIMIC-III, using BigQuery to handle the 330M-row CHARTEVENTS table
and scikit-learn / XGBoost / LightGBM for modelling.

This is a university Big-Data course project. The emphasis is on doing the data engineering
and the evaluation **honestly and at scale**, not on chasing an unrealistic score.

---

## What the pipeline does

1. **Scale.** CHARTEVENTS (~330M rows / 4.2 GB) and LABEVENTS (~27M rows) are never
   downloaded. Cohort selection, first-24h filtering, error/range cleaning, CareVue↔MetaVision
   ITEMID harmonization and per-concept aggregation are all pushed into **BigQuery SQL**
   ([`src/data/sql.py`](src/data/sql.py)); only compact aggregated tables (one row per
   ICU stay × concept) are downloaded.
2. **Leak-free features.** Only first-24h data feeds the model; measurement counts are
   window-capped (severity proxy, not a stay-length proxy); age is clipped to 90 to undo
   MIMIC's >89y / ~300y DOB shift; imputation is fit inside CV folds.
   ([`src/features/engineering.py`](src/features/engineering.py))
3. **LABEVENTS ablation.** Every model is trained **without** and **with** the lab feature
   block (renal/electrolyte/perfusion/haematology/coagulation panels, ~20 concepts via
   [`src/data/lab_concepts.py`](src/data/lab_concepts.py)) on identical patients/folds, so the
   report quantifies exactly what labs add. Labs join on `HADM_ID` (LABEVENTS has no
   `ICUSTAY_ID`) and are time-windowed against the ICU `INTIME`; their 'abnormal' FLAG is kept,
   not filtered.
4. **Honest validation.** Train/test split is **grouped by `SUBJECT_ID`** (no patient in
   both sides; [`src/data/splits.py`](src/data/splits.py)). Models are compared with
   **GroupKFold** CV against trivial baselines, and **both** framings are reported:
   - **regression** — predict LOS in days (MAE / RMSE / R²),
   - **classification** — short (<3d) / medium (3–7d) / long (>7d), with accuracy, macro-F1,
     quadratic-weighted Cohen's κ and one-vs-rest AUROC.
5. **Profiling.** Every phase is timed; the notebook prints a per-phase table and total
   runtime ([`src/evaluation/profiling.py`](src/evaluation/profiling.py)).

> **A realistic-expectations note:** continuous ICU-LOS is hard. Published R² is commonly
> **~0.05–0.30**, which is why the literature usually prefers the bucketed/ordinal framing
> (e.g. Harutyunyan et al. 2019 report quadratic κ ≈ 0.43). If you see R² near 0.7 on this
> task, suspect leakage. We report modest numbers *and* beat trivial baselines — the lift is
> what matters.

---

## Project layout

```
src/
  config.py                 # all tunables (window, thresholds, CV, paths, GCP)
  data/
    concepts.py             # CareVue/MetaVision chart ITEMID -> concept + valid ranges
    lab_concepts.py         # LABEVENTS (D_LABITEMS) ITEMID -> lab concept + ranges
    sql.py                  # BigQuery query builders, incl. lab aggregation (big-data core)
    loader.py               # BigQuery-first loader w/ parquet cache + synthetic fallback
    synthetic.py            # clearly-labelled synthetic cohort (vitals + labs, same shape)
    splits.py               # grouped (by SUBJECT_ID) train/test split + GroupKFold
  features/engineering.py   # vital + lab aggregates + demographics -> leak-free matrix
  models/registry.py        # sklearn Pipelines for regression & classification
  evaluation/
    metrics.py              # regression + classification metrics, error-by-LOS-band
    harness.py              # GroupKFold CV, hold-out eval, RandomizedSearchCV
    profiling.py            # execution-time profiler
  visualization/plots.py    # per-patient timeline, distributions, comparisons, etc.

import_tables.py            # load patients/admissions/icustays/d_items/chartevents/d_labitems/labevents
main.ipynb                  # the annotated, runnable orchestrator
reports/REPORT.md           # methodology + results write-up (fill in numbers from a real run)
requirements.txt
```

---

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows; use bin/activate on *nix
pip install -r requirements.txt
jupyter notebook main.ipynb
```

The notebook runs **out of the box on labelled synthetic data** (clearly banner-flagged) so
you can see the whole pipeline without credentials.

### Running on the real MIMIC-III data (BigQuery)

1. Put a GCP service-account key at `./gcp_key.json` (or set
   `GOOGLE_APPLICATION_CREDENTIALS`). Set `GCP_PROJECT_ID` / `MIMIC_DATASET` /
   `MIMIC_BUCKET` env vars if they differ from the defaults in `src/config.py`.
2. Load the tables once: `python import_tables.py`
   (streams each `.csv.gz` from the U.Porto mirror → GCS → BigQuery).
3. In `main.ipynb` cell 0, leave `USE_BIGQUERY = True` and run all.

Results from the synthetic path are **demonstrations, not findings** — the notebook says so
loudly. Only numbers produced with `DATA SOURCE: BIGQUERY` go in your report.

---

## Key design choices (and why)

| Choice | Rationale |
|---|---|
| Target = **ICU LOS days**, predicted **at 24h** | Early enough to inform staffing/beds, late enough for a full day of vitals. Cohort = stays still in ICU at 24h. |
| Aggregate **in BigQuery**, train locally | Turns a 4.2 GB scan into a few-hundred-thousand-row download; the actual "big data" answer. |
| **Tree ensembles** (RF / HistGB / XGBoost / LightGBM) | SOTA on tabular clinical data at this cohort size (~40–60k stays); interpretable; deep learning is unwarranted here. |
| **Grouped** split + CV by SUBJECT_ID | A patient can have many stays; random splitting leaks patient signal. |
| Report **regression *and* classification** vs **baselines** | LOS regression is intrinsically low-R²; classification is the standard, more useful framing; baselines prove the lift is real. |

## References

- Johnson et al. (2016), *MIMIC-III, a freely accessible critical care database*, Sci. Data.
- Harutyunyan et al. (2019), *Multitask learning and benchmarking with clinical time series*
  (LOS benchmark; quadratic-κ framing).
- Wang et al. (2020), *MIMIC-Extract* (first-24h feature pipeline; leakage controls).
- Grinsztajn et al. (2022), *Why do tree-based models still outperform deep learning on
  tabular data?*
