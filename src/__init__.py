"""
MIMIC-III Length of Stay Prediction - ML Pipeline
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "ML Team"
__description__ = "Comprehensive ML pipeline for ICU length of stay prediction"

# Import main components for easy access
from data.loader import MIMICDataLoader
from data.preprocessor import MIMICPreprocessor
from features.engineering import FeatureEngineer
from models.base import ModelFactory, BaseModel
from evaluation.metrics import (
    RegressionMetrics,
    ClassificationMetrics,
    PerformanceProfiler,
    ResultsAnalyzer,
    ModelComparison
)
from visualization.plots import MIMICVisualizer

__all__ = [
    'MIMICDataLoader',
    'MIMICPreprocessor',
    'FeatureEngineer',
    'ModelFactory',
    'BaseModel',
    'RegressionMetrics',
    'ClassificationMetrics',
    'PerformanceProfiler',
    'ResultsAnalyzer',
    'ModelComparison',
    'MIMICVisualizer',
]
