"""MIMIC-III ICU Length-of-Stay prediction pipeline.

Public entry points used by the notebook::

    from src import config
    from src.data.loader import load_cohort
    from src.data.splits import (grouped_train_test_split, grouped_cv,
                                 assert_no_group_leakage)
    from src.features.engineering import build_feature_matrix
    from src.models import registry
    from src.evaluation import harness, metrics
    from src.evaluation.profiling import Profiler
    from src.visualization import plots
"""
from . import config

__all__ = ["config"]
__version__ = "2.0.0"
