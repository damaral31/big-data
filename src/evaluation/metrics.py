# Evaluation metrics and profiling for LOS prediction
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time
from datetime import datetime
import json

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
    )
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False


class RegressionMetrics:
    """Calculate regression metrics for LOS prediction"""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score"""
        return r2_score(y_true, y_pred)

    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Median Absolute Error"""
        return np.median(np.abs(y_true - y_pred))

    @staticmethod
    def mean_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Percentage Error (signed)"""
        mask = y_true != 0
        return np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]) * 100

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all regression metrics"""
        return {
            'MAE': RegressionMetrics.mae(y_true, y_pred),
            'RMSE': RegressionMetrics.rmse(y_true, y_pred),
            'MAPE': RegressionMetrics.mape(y_true, y_pred),
            'R2': RegressionMetrics.r2(y_true, y_pred),
            'Median AE': RegressionMetrics.median_absolute_error(y_true, y_pred),
            'Mean PE': RegressionMetrics.mean_percentage_error(y_true, y_pred),
        }


class ClassificationMetrics:
    """Calculate classification metrics for LOS prediction"""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate all classification metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # ROC-AUC for binary or OvR multiclass
        if len(np.unique(y_true)) == 2 and y_proba is not None:
            try:
                metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['ROC-AUC'] = np.nan

        return metrics

    @staticmethod
    def confusion_matrix_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get confusion matrix and classification report"""
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        return f"Confusion Matrix:\n{cm}\n\n{report}"


class PerformanceProfiler:
    """Profile execution time and memory usage"""

    def __init__(self):
        self.profiles = {}
        self.start_times = {}

    def start(self, phase: str):
        """Start timing a phase"""
        self.start_times[phase] = time.time()
        logger.info(f"Starting: {phase}")

    def end(self, phase: str) -> float:
        """End timing a phase"""
        if phase not in self.start_times:
            logger.warning(f"Phase {phase} not started")
            return 0

        elapsed = time.time() - self.start_times[phase]
        self.profiles[phase] = elapsed
        logger.info(f"Completed: {phase} ({elapsed:.2f}s)")
        return elapsed

    def get_report(self) -> pd.DataFrame:
        """Get profiling report"""
        df = pd.DataFrame(list(self.profiles.items()), columns=['Phase', 'Time (seconds)'])
        df['Time (%)'] = (df['Time (seconds)'] / df['Time (seconds)'].sum()) * 100
        df = df.sort_values('Time (seconds)', ascending=False)
        return df

    def print_report(self):
        """Print profiling report"""
        report = self.get_report()
        logger.info(f"\n{report.to_string(index=False)}")

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return self.profiles.copy()


class ResultsAnalyzer:
    """Analyze and interpret model results"""

    @staticmethod
    def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Analyze prediction errors"""
        errors = y_true - y_pred
        abs_errors = np.abs(errors)

        return {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mean_abs_error': float(np.mean(abs_errors)),
            'min_error': float(np.min(abs_errors)),
            'max_error': float(np.max(abs_errors)),
            'median_error': float(np.median(abs_errors)),
            'error_0': float(np.sum(abs_errors < 0.5) / len(errors)),  # Within 0.5 days
            'error_1': float(np.sum(abs_errors < 1.0) / len(errors)),  # Within 1 day
            'error_2': float(np.sum(abs_errors < 2.0) / len(errors)),  # Within 2 days
            'error_5': float(np.sum(abs_errors < 5.0) / len(errors)),  # Within 5 days
        }

    @staticmethod
    def analyze_by_los_range(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Analyze errors by length of stay range"""
        ranges = [(0, 2), (2, 5), (5, 10), (10, 20), (20, np.inf)]
        results = []

        for min_los, max_los in ranges:
            mask = (y_true >= min_los) & (y_true < max_los)
            if mask.sum() == 0:
                continue

            y_true_range = y_true[mask]
            y_pred_range = y_pred[mask]

            results.append({
                'LOS Range': f"{min_los}-{max_los if max_los != np.inf else '+'}",
                'Count': mask.sum(),
                'MAE': mean_absolute_error(y_true_range, y_pred_range),
                'RMSE': np.sqrt(mean_squared_error(y_true_range, y_pred_range)),
                'R2': r2_score(y_true_range, y_pred_range),
            })

        return pd.DataFrame(results)

    @staticmethod
    def generate_report(model_name: str, metrics: Dict, errors: Dict,
                       by_range: pd.DataFrame) -> str:
        """Generate comprehensive evaluation report"""
        report = f"""
================================================================================
MODEL EVALUATION REPORT: {model_name}
================================================================================

OVERALL METRICS:
{json.dumps(metrics, indent=2)}

ERROR ANALYSIS:
{json.dumps(errors, indent=2)}

ERROR BY LOS RANGE:
{by_range.to_string(index=False)}

================================================================================
"""
        return report


class ModelComparison:
    """Compare multiple models"""

    def __init__(self):
        self.results = []

    def add_result(self, model_name: str, metrics: Dict, training_time: float):
        """Add model result"""
        result = {
            'Model': model_name,
            'Training Time (s)': training_time,
            **metrics
        }
        self.results.append(result)

    def get_comparison_df(self) -> pd.DataFrame:
        """Get comparison dataframe"""
        df = pd.DataFrame(self.results)
        return df.sort_values('RMSE' if 'RMSE' in df.columns else 'Accuracy', ascending=False)

    def print_comparison(self):
        """Print model comparison"""
        df = self.get_comparison_df()
        logger.info(f"\n{df.to_string(index=False)}")

    def get_best_model(self, metric: str = 'RMSE') -> str:
        """Get best model by metric"""
        df = self.get_comparison_df()
        if metric in df.columns:
            if metric in ['MAE', 'RMSE', 'MAPE', 'Error']:
                best_idx = df[metric].idxmin()
            else:
                best_idx = df[metric].idxmax()
            return df.loc[best_idx, 'Model']
        return df.iloc[0]['Model']
