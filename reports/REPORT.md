# Predicting ICU Length of Stay from the First 24 Hours — MIMIC-III
### A CRISP-DM report

> Source document for the PDF deliverable; mirrors `main.ipynb`. Bracketed `[…]` values are
> filled from a real BigQuery run. Synthetic-run numbers (§Eval note) are **not** findings.
> Methodology: CRISP-DM (Wirth & Hipp, 2000) — the six phases below.

---

## 1. Business Understanding

**Goal.** Predict ICU **length of stay (LOS, days)** *at the 24-hour mark* — early enough to
support bed/staffing/step-down decisions, late enough that a full day of data exists.

**Target (why).** ICU LOS = `OUTTIME − INTIME` (per `ICUSTAY_ID`): matches the chart-event
data, is the canonical benchmark target (Harutyunyan 2019), and `LOS` exists precomputed in
ICUSTAYS. Patients with multiple stays are handled by grouped splitting, not by changing target.

**Window (why 24h).** 6h → too few labs resulted; 72h → decisions delayed and more short stays
excluded. Cohort = stays still in the ICU at 24h (`MIN_LOS_HOURS = 24`), LOS clipped at 60d.

**Two framings.** Regression (days, the literal ask, low-R²) **and** ordinal classification
(short<3d / medium 3–7d / long>7d, the robust standard). Both reported.

**Why these feature categories.** First-24h **vitals** (haemodynamics, respiration, neuro),
**labs** (renal, electrolytes, perfusion, haematology, coagulation) and **demographics/admin** —
the variables available at hour 24 that the literature finds predictive of deterioration.

## 2. Data Understanding

**Dataset (MIMIC-III v1.4).**

| Table | Rows | Use |
|---|---|---|
| CHARTEVENTS | ~330 M (4.2 GB gz) | first-24h vitals |
| LABEVENTS | ~27 M | first-24h labs |
| ICUSTAYS / ADMISSIONS / PATIENTS | 61.5k / 59k / 46.5k | cohort, target, demographics |
| D_ITEMS / D_LABITEMS | ~12.5k / ~750 | ITEMID → concept |

**EDA (notebook §2):** LOS is strongly right-skewed (median [..] d, long minority class);
per-patient event timelines + a "patient vs cohort mean ± 1 SD" band; **missing-value analysis**
showing labs are absent far more often than vitals.

**Informative missingness.** Missingness is not random — ordering a lactate signals concern. We
therefore (Phase 3) add a 0/1 *was-measured* feature per concept, capturing the decision itself.

## 3. Data Preparation

**Leakage controls — what is hidden or altered:**

| Variable | Action | Why |
|---|---|---|
| `OUTTIME`/`DISCHTIME` | hidden (define target only) | knowing discharge = the answer |
| events after `INTIME+24h` | hidden (SQL window) | unavailable at prediction time |
| measurement counts | capped to 24h window | full-stay counts encode LOS |
| age (`DOB`) | clipped to 90 | MIMIC shifts >89y DOB ~300y |
| `CHARTEVENTS.ERROR=1` | dropped | clinician-flagged errors |
| `HOSPITAL_EXPIRE_FLAG`, `DISCHARGE_LOCATION`, `DEATHTIME`, precomputed `LOS` | hidden | post-discharge / target copy |

**Feature engineering** ([engineering.py](../src/features/engineering.py)): ITEMID
harmonization (CareVue↔MetaVision vitals; D_LABITEMS labs, `lab_` prefixed), per-concept
mean/min/max/std/count, window-capped intensity, demographics, and the `*_measured` indicators.
Imputation/scaling are in-pipeline (fit per fold). Two matrices on identical rows (without/with
labs) for an apples-to-apples ablation.

**Feature inventory** (notebook §3.3): [..] features total — categories: vital statistics,
lab statistics, intensity, missingness indicators, demographics.

**Unsupervised structure exploration** (§3.4, *exploratory, not predictive*): PCA scree + 2-D
projection, t-SNE (visualization only), KMeans with **silhouette** to pick k. Expectation and
finding: clinical cohorts cluster weakly (silhouette [..], typically <0.3) and clusters align
only loosely with LOS — confirming a *supervised* model is the right tool.

**Split:** `GroupShuffleSplit`/`GroupKFold` on `SUBJECT_ID` (no patient on both sides).

## 4. Modeling

**Baselines.** Trivial (mean/majority) + **published** ([baselines.py](../src/evaluation/baselines.py)):
Harutyunyan 2019 linear κ≈0.34 / LSTM κ≈0.43, MIMIC-Extract LOS>7d AUROC≈0.76, first-24h
regression R²≈0.04.

**Model zoo / proposed SOTA.** Ridge/Logistic, RandomForest, HistGB, **XGBoost**, **LightGBM** —
gradient-boosted trees are SOTA for medium tabular data (Grinsztajn 2022); they match/beat deep
nets here, with interpretability. Sequence DL (GRU/temporal CNN on hourly bins) is the only DL
worth trying and is left as future work (modest LOS gains).

**Tuning.** `RandomizedSearchCV` (GroupKFold) over the boosting family.

**LABEVENTS ablation.** Every model cross-validated without vs with labs on identical folds.

## 5. Evaluation

| Framing | Metric | No labs | With labs | Δ | Literature |
|---|---|---|---|---|---|
| Regression (tuned, hold-out) | MAE (d) | [..] | [..] | **[..]** | LSTM 3.92 |
| Regression | R² | [..] | [..] | **[..]** | ~0.04 |
| Classification (best) | quadratic κ | [..] | [..] | **[..]** | LSTM 0.43 |
| Classification | macro-AUROC | [..] | [..] | **[..]** | >7d 0.76 |

Labs were **[..]%** of the tuned model's importance (missingness indicators **[..]%**); the gain
concentrates in tree models (the linear model barely benefits — labs add non-linear signal).
**Distributions of the top features by LOS bucket** and **feature importance** are in §5.2. The
error-by-band table shows the universal failure mode: **long stays under-predicted**. Confusion
matrix in §5.3.

> Synthetic demonstration (not findings): labs improve boosting κ by ≈+0.02–0.03, cut hold-out
> MAE ≈0.05 d; tuned-model lab importance ≈51%; best silhouette ≈0.1–0.2 (weak structure).

## 6. Deployment considerations

**Profiling** (§6.1): total [..] s — BigQuery load, feature build, unsupervised, CV ablation,
HP search.

**MapReduce — used via the engine, not hand-rolled.** The workload is filter-then-aggregate
over 330M+27M rows; BigQuery's Dremel engine runs it as a distributed map (`WHERE`/`JOIN`) /
reduce (`GROUP BY AVG/MIN/MAX/COUNT`) plan. We push that down in SQL and download only
aggregates. A bespoke Spark/Hadoop job would duplicate this with more overhead — over-engineering
here; it *would* be the choice for flat files with no warehouse.

**Multiprocessing.** scikit-learn `n_jobs=-1` (joblib) parallelises CV folds and RF trees across
cores; XGBoost/LightGBM are internally multithreaded; BigQuery parallelises server-side; parquet
caching amortises re-runs. Feature engineering is left single-process (small, vectorised matrix —
pool overhead would hurt).

**Limitations / future work.** Medications (INPUTEVENTS) and notes unused; *remaining*-LOS
re-scored daily is more deployable; pre-ICU lab window (`intime−6h`) for admission labs; SHAP;
sequence DL; temporal (admission-time) split across the CareVue→MetaVision era change.

### References
Wirth & Hipp (2000) CRISP-DM · Johnson et al. (2016) MIMIC-III · Harutyunyan et al. (2019)
benchmark · Wang et al. (2020) MIMIC-Extract · Grinsztajn et al. (2022) tabular DL ·
Sharafoddini et al. (2019) informative missingness · Dean & Ghemawat (2008) MapReduce ·
Melnik et al. (2010) Dremel.
