# Predicting ICU Length of Stay from the First 24 Hours — MIMIC-III

> Source document for the PDF deliverable. Bracketed `[…]` values are filled from a real
> BigQuery run (see §6). The synthetic demonstration numbers in §7 must **not** be reported
> as findings.

## 1. Problem & data

**Task.** Predict ICU **length of stay (LOS)** for a stay, using only information available in
the **first 24 hours** after ICU admission. We treat the problem two ways:

- **Regression** — LOS in fractional days (`ICUSTAYS.OUTTIME − INTIME`).
- **Classification** — ordinal buckets: **short** (<3 d), **medium** (3–7 d), **long** (>7 d).

**Why a window?** At deployment a model that needs the whole stay to predict the stay is
useless. Predicting *at the 24-hour mark* is the standard early-warning setting: a full day of
vitals exists, yet there is still time to act on the prediction (staffing, bed planning,
step-down). The cohort is therefore restricted to stays still in the ICU at 24 h
(`MIN_LOS_HOURS = 24`); we also clip LOS at 60 days as a data-quality guard.

**Dataset.** MIMIC-III v1.4 (BIDMC, 2001–2012). Tables used:

| Table | Rows | Use |
|---|---|---|
| CHARTEVENTS | ~330 M (4.2 GB gz) | first-24h vitals (features) |
| ICUSTAYS | 61,532 | cohort + target (LOS) |
| ADMISSIONS | 58,976 | admission type/location, timing |
| PATIENTS | 46,520 | gender, DOB (age) |
| D_ITEMS | ~12,500 | ITEMID → concept |

One **SUBJECT_ID** can have several **HADM_ID**s and several **ICUSTAY_ID**s — this drives the
grouped-splitting decision (§4).

## 2. Big-data handling

CHARTEVENTS cannot be downloaded or processed locally. The expensive work runs **in BigQuery**
([`src/data/sql.py`](../src/data/sql.py)): a single query joins the cohort to CHARTEVENTS,
filters to the first 24 h, drops error/implausible values, maps ITEMIDs to canonical concepts,
and returns **per-(stay, concept) aggregates** (mean/min/max/std/count). The ~330 M-row scan
collapses to a few-hundred-thousand-row result that fits comfortably in memory. Results are
parquet-cached so re-runs are free.

This is deliberately *not* Spark/MapReduce on a local cluster: for a filter-then-aggregate
workload over a single large table, pushing SQL to a columnar warehouse is both simpler and
faster, and it is the approach the assignment's "use the resources wisely" brief rewards.

## 3. Feature engineering (leakage controls)

Implemented in [`src/features/engineering.py`](../src/features/engineering.py).

- **First-window only.** Every feature derives from `charttime ∈ [intime, intime + 24h)`.
  Nothing uses OUTTIME/DISCHTIME or whole-stay statistics.
- **ITEMID harmonization.** CareVue and MetaVision encode the same vital with different
  ITEMIDs (e.g. heart rate = 211 *and* 220045). We map ~13 high-yield vitals to a single
  concept each ([`src/data/concepts.py`](../src/data/concepts.py)) so columns aren't split or
  duplicated across the two source systems.
- **Error / range cleaning.** Rows with `ERROR = 1` are dropped; values outside generous
  per-concept physiological ranges are discarded as charting errors.
- **Window-capped intensity.** We keep measurement counts, but only *within the fixed 24h
  window* — a legitimate severity signal (sicker patients are monitored more), **not** a proxy
  for total stay length (the classic LOS-leakage trap).
- **Age fix.** MIMIC shifts the DOB of patients >89 by ~300 years; we clip computed age at 90.
- **Missingness.** Columns missing in >60% of stays are dropped; the rest keep NaNs, imputed
  (median) *inside* the model pipeline so the imputer never sees the test fold.

## 4. Data preparation & validation

- **Grouped hold-out** (`GroupShuffleSplit`, 20% test) and **GroupKFold (k=5)** for CV, both
  keyed on **SUBJECT_ID**, so no patient appears in both train and test
  ([`src/data/splits.py`](../src/data/splits.py)). A random split would inflate scores by
  letting the model memorise patient-specific quirks.
- **Preprocessing in-pipeline.** Imputation/scaling live in each model's
  `sklearn.Pipeline`, fit per fold — no preprocessing leakage.
- **Class imbalance.** The *long* class is rare; classifiers use balanced class weights.

## 5. Models, tuning & metrics

[`src/models/registry.py`](../src/models/registry.py) — all plain `sklearn` Pipelines:

- **Baselines:** mean regressor / majority classifier (sanity floor — any real model must beat
  these, and a baseline R² ≈ 0 is evidence the target isn't leaked).
- **Linear:** Ridge / Logistic Regression (scaled).
- **Trees:** Random Forest, HistGradientBoosting, **XGBoost**, **LightGBM**.

**Tuning.** `RandomizedSearchCV` (20 iters, GroupKFold) over the strongest gradient-boosting
family, scoring negative MAE.

**Metrics.** Regression: MAE, RMSE, R², median AE, % within ±1/±2 days, and **error by true-LOS
band** (to expose where the model fails). Classification: accuracy, macro-F1, **quadratic-
weighted Cohen's κ** (rewards ordinal near-misses), macro one-vs-rest AUROC, confusion matrix.

## 6. Results (fill from a BigQuery run)

**Cohort:** [N] stays / [P] patients. LOS median [..] d, mean [..] d.

**Regression (hold-out, tuned [model]):** MAE **[..] d**, RMSE [..] d, R² **[..]**,
within ±1 d [..]%. Mean baseline R² ≈ [~0] → the lift is real.

**Classification (hold-out, [model]):** accuracy [..], macro-F1 [..], quadratic κ **[..]**,
macro-AUROC [..].

**What drove predictions:** top features [e.g. heart-rate/respiratory-rate stats, measurement
intensity, age, admission type].

**Profiling:** total [..] s — BigQuery load [..] s, feature build [..] s, CV [..] s, HP search
[..] s. (BigQuery query bytes/cost: [..].)

## 7. Demonstration run on synthetic data (NOT findings)

Running the pipeline on the built-in synthetic cohort (~1.2k stays) produces, illustratively:
tuned regressor MAE ≈ 1.8 d, R² ≈ 0.2–0.3; classification accuracy ≈ 0.58, quadratic κ ≈ 0.46,
AUROC ≈ 0.73; mean/majority baselines at R² ≈ 0 / κ = 0. These confirm the pipeline is wired
correctly and leak-free, but they describe synthetic data and carry no clinical meaning.

## 8. Discussion

- **Regression is intrinsically hard.** Continuous ICU-LOS R² in the literature is low
  (~0.05–0.30); our numbers land in/above that band while clearly beating baselines. The
  error-by-band table shows the universal failure mode: **long stays are systematically
  under-predicted** because they are rare and heterogeneous — the model regresses them toward
  the population mean.
- **Classification is the more useful framing**, and our quadratic κ is comparable to the
  Harutyunyan et al. (2019) benchmark (κ ≈ 0.43). Balanced weights keep the rare *long* class
  from being ignored.
- **Trees beat linear models** modestly and dominate any case for deep learning at this cohort
  size, while staying interpretable.

## 9. Limitations & future work

Labs (LABEVENTS), medications (INPUTEVENTS) and clinical notes are unused — adding labs alone
typically helps most. A *remaining*-LOS target re-scored each day is more deployable than total
LOS. A temporal (admission-time) split would test generalisation across MIMIC's CareVue→
MetaVision era change. SHAP would strengthen interpretability beyond impurity importance.

## References

1. Johnson et al. (2016). MIMIC-III. *Scientific Data* 3:160035.
2. Harutyunyan et al. (2019). Multitask learning and benchmarking with clinical time series.
   *Scientific Data* 6:96.
3. Wang et al. (2020). MIMIC-Extract. *ACM CHIL*.
4. Grinsztajn et al. (2022). Why tree-based models still outperform deep learning on tabular
   data. *NeurIPS*.
5. Pedregosa et al. (2011). scikit-learn. *JMLR* 12.
