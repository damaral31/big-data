"""Model zoo -- plain scikit-learn Pipelines (no custom estimator wrappers).

Each entry is a ``sklearn.pipeline.Pipeline`` so that imputation/scaling are
fit *inside* cross-validation folds (no preprocessing leakage), and so the
objects compose cleanly with ``cross_validate`` / ``RandomizedSearchCV`` /
``GroupKFold`` -- the bug that made the previous custom ``BaseModel`` silently
unusable.

Linear / instance models get median-imputation + standardisation.
Tree ensembles get median-imputation only (scale-invariant).  XGBoost and
LightGBM handle NaN natively but the imputer is harmless and keeps the matrix
uniform.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as cfg

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

RS = cfg.RANDOM_STATE
NJ = cfg.N_JOBS


def _scaled(estimator) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _tree(estimator) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def build_regressors() -> dict[str, Pipeline]:
    models = {
        "baseline_mean": _scaled(DummyRegressor(strategy="mean")),
        "ridge": _scaled(Ridge(alpha=1.0, random_state=RS)),
        "random_forest": _tree(RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=4,
            n_jobs=NJ, random_state=RS)),
        "hist_gb": _tree(HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=RS)),
    }
    if HAS_XGB:
        models["xgboost"] = _tree(XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=NJ, random_state=RS))
    if HAS_LGBM:
        models["lightgbm"] = _tree(LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, n_jobs=NJ,
            random_state=RS, verbose=-1))
    return models


def build_classifiers() -> dict[str, Pipeline]:
    models = {
        "baseline_majority": _scaled(DummyClassifier(strategy="most_frequent")),
        "logistic": _scaled(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RS)),
        "random_forest": _tree(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=4, class_weight="balanced",
            n_jobs=NJ, random_state=RS)),
        "hist_gb": _tree(HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced",
            random_state=RS)),
    }
    if HAS_LGBM:
        models["lightgbm"] = _tree(LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", n_jobs=NJ, random_state=RS, verbose=-1))
    if HAS_XGB:
        # XGBClassifier has no class_weight; imbalance handled via balanced
        # sample weights passed at fit time (see evaluation.tuning).
        models["xgboost"] = _tree(XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=NJ, random_state=RS))
    return models


# Hyper-parameter search space for the tuned model (LightGBM if present, else
# HistGB).  Keys are addressed through the pipeline's ``model__`` prefix.
def tuned_regressor_search_space():
    if HAS_LGBM:
        base = _tree(LGBMRegressor(n_jobs=NJ, random_state=RS, verbose=-1))
        space = {
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__min_child_samples": [10, 20, 40],
        }
        return "lightgbm", base, space
    base = _tree(HistGradientBoostingRegressor(random_state=RS))
    space = {
        "model__max_iter": [200, 400],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 40],
    }
    return "hist_gb", base, space
