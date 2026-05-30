"""Patient-grouped splitting.

A single SUBJECT_ID can have several ICU stays.  If stays from the same patient
land in both train and test, the model can memorise patient-specific quirks and
the scores are optimistic.  Every split here groups by SUBJECT_ID so a patient
is entirely in train or entirely in test/validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .. import config as cfg


def grouped_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = cfg.TEST_SIZE,
    random_state: int = cfg.RANDOM_STATE,
):
    """Hold-out split with no patient appearing on both sides."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx], groups.iloc[test_idx],
    )


def grouped_cv(n_splits: int = cfg.CV_FOLDS) -> GroupKFold:
    """GroupKFold for cross-validation / hyper-parameter search."""
    return GroupKFold(n_splits=n_splits)


def assert_no_group_leakage(groups_train: pd.Series, groups_test: pd.Series) -> None:
    """Sanity check used in the notebook -- raises if a patient straddles split."""
    overlap = set(np.unique(groups_train)) & set(np.unique(groups_test))
    if overlap:
        raise AssertionError(
            f"{len(overlap)} subject_id(s) appear in both train and test")
