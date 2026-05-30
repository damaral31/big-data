"""Cross-validation, hold-out evaluation and hyper-parameter search.

All cross-validation uses :class:`sklearn.model_selection.GroupKFold` keyed on
SUBJECT_ID so no patient is split across folds.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate

from .. import config as cfg
from ..data.splits import grouped_cv
from ..models import registry

# ---- scoring dictionaries ------------------------------------------------- #
REG_SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
}

_KAPPA = make_scorer(cohen_kappa_score, weights="quadratic")


def _auc_ovr(estimator, X, y):
    proba = estimator.predict_proba(X)
    return roc_auc_score(y, proba, multi_class="ovr", average="macro")


CLF_SCORING = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "kappa_quadratic": _KAPPA,
    "roc_auc_ovr": _auc_ovr,
}


def _summarise(cv_out: dict, scoring: dict) -> dict:
    row = {}
    for name in scoring:
        scores = cv_out[f"test_{name}"]
        # neg_* scorers come back negative -> flip sign for readability
        sign = -1.0 if name in ("MAE", "RMSE") else 1.0
        row[f"{name}_mean"] = float(np.mean(sign * scores))
        row[f"{name}_std"] = float(np.std(sign * scores))
    row["fit_time_s"] = float(np.mean(cv_out["fit_time"]))
    return row


def cross_validate_models(models, X, y, groups, task: str,
                          profiler=None) -> pd.DataFrame:
    """Run GroupKFold CV for every model; return one row per model."""
    scoring = REG_SCORING if task == "regression" else CLF_SCORING
    cv = grouped_cv()
    rows = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        out = cross_validate(
            model, X, y, groups=groups, cv=cv, scoring=scoring,
            n_jobs=cfg.N_JOBS, return_train_score=False, error_score="raise",
        )
        rows[name] = _summarise(out, scoring)
        if profiler is not None:
            profiler.record(f"cv:{task}:{name}", time.perf_counter() - t0)
    sort_key = "MAE_mean" if task == "regression" else "kappa_quadratic_mean"
    ascending = task == "regression"
    return pd.DataFrame(rows).T.sort_values(sort_key, ascending=ascending)


def holdout_fit_predict(model, X_train, y_train, X_test, want_proba=False):
    """Fit on train, predict on test; return (y_pred, y_proba_or_None, seconds)."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if want_proba else None
    return y_pred, proba, fit_s


def tune_regressor(X_train, y_train, groups_train, profiler=None):
    """RandomizedSearchCV (GroupKFold) over the strongest regressor family."""
    name, base, space = registry.tuned_regressor_search_space()
    search = RandomizedSearchCV(
        base, space, n_iter=cfg.HP_SEARCH_ITER, scoring="neg_mean_absolute_error",
        cv=grouped_cv(), random_state=cfg.RANDOM_STATE, n_jobs=cfg.N_JOBS,
        error_score="raise",
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train, groups=groups_train)
    if profiler is not None:
        profiler.record("hp_search", time.perf_counter() - t0)
    return name, search.best_estimator_, search.best_params_, -search.best_score_
