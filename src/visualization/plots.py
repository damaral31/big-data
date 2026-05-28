# Visualization module for MIMIC-III analysis
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class MIMICVisualizer:
    """Create visualizations for MIMIC-III analysis"""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(self, filename: str):
        """Save figure to output directory"""
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure: {filepath}")
        plt.close()

    def plot_chart_events_timeline(self, chart_events_df: pd.DataFrame,
                                   icustay_id: int,
                                   title: str = None):
        """Plot chart events timeline for a patient (like the example in assignment)"""
        df = chart_events_df[chart_events_df['icustay_id'] == icustay_id].copy()

        if len(df) == 0:
            logger.warning(f"No events found for ICU stay {icustay_id}")
            return

        # Convert charttime to hours from start
        df['charttime'] = pd.to_datetime(df['charttime'])
        start_time = df['charttime'].min()
        df['hours_from_start'] = (df['charttime'] - start_time).dt.total_seconds() / 3600

        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot each item type with different color
        for label, group in df.groupby('label'):
            ax.scatter(group['hours_from_start'] / 24, group['value'],
                      label=label, s=50, alpha=0.6)

        ax.set_xlabel('Time (Days)', fontsize=12)
        ax.set_ylabel('Measurement Value', fontsize=12)
        ax.set_title(title or f'ICU Events Timeline - ICUSTAY_ID {icustay_id}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()

        self.save_fig(f'timeline_{icustay_id}.png')

    def plot_length_of_stay_distribution(self, los_df: pd.DataFrame,
                                        figsize: tuple = (12, 6)):
        """Plot LOS distribution"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Histogram
        axes[0].hist(los_df['length_of_stay'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Length of Stay (Days)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Length of Stay Distribution', fontsize=12, fontweight='bold')
        axes[0].axvline(los_df['length_of_stay'].mean(), color='red', linestyle='--',
                       label=f"Mean: {los_df['length_of_stay'].mean():.1f}")
        axes[0].axvline(los_df['length_of_stay'].median(), color='green', linestyle='--',
                       label=f"Median: {los_df['length_of_stay'].median():.1f}")
        axes[0].legend()

        # Box plot
        axes[1].boxplot(los_df['length_of_stay'], vert=True)
        axes[1].set_ylabel('Length of Stay (Days)', fontsize=11)
        axes[1].set_title('Length of Stay Box Plot', fontsize=12, fontweight='bold')

        plt.tight_layout()
        self.save_fig('los_distribution.png')

    def plot_model_predictions_vs_actual(self, y_true: np.ndarray,
                                        y_pred: np.ndarray,
                                        model_name: str = "Model"):
        """Plot predictions vs actual values"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=30)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual LOS (Days)', fontsize=11)
        axes[0].set_ylabel('Predicted LOS (Days)', fontsize=11)
        axes[0].set_title(f'{model_name}: Predictions vs Actual', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted LOS (Days)', fontsize=11)
        axes[1].set_ylabel('Residuals (Days)', fontsize=11)
        axes[1].set_title(f'{model_name}: Residual Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_fig(f'predictions_{model_name.replace(" ", "_").lower()}.png')

    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             metric: str = 'RMSE'):
        """Plot model comparison"""
        if metric not in comparison_df.columns:
            logger.warning(f"Metric {metric} not found in comparison results")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by metric
        df = comparison_df.sort_values(metric)
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(df)))

        bars = ax.barh(df['Model'], df[metric], color=colors)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f'Model Comparison: {metric}', fontsize=12, fontweight='bold')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df[metric])):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                   va='center', ha='left', fontsize=10)

        plt.tight_layout()
        self.save_fig(f'model_comparison_{metric.lower()}.png')

    def plot_feature_importance(self, feature_importance: dict,
                               top_n: int = 15):
        """Plot feature importances"""
        if not feature_importance:
            logger.warning("No feature importance data available")
            return

        # Get top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars = ax.barh(range(len(features)), importances, color=colors)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=12, fontweight='bold')

        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax.text(imp, bar.get_y() + bar.get_height() / 2, f'{imp:.4f}',
                   va='center', ha='left', fontsize=9)

        plt.tight_layout()
        self.save_fig('feature_importance.png')

    def plot_error_distribution(self, y_true: np.ndarray,
                               y_pred: np.ndarray,
                               model_name: str = "Model"):
        """Plot error distribution"""
        errors = np.abs(y_true - y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(np.mean(errors), color='red', linestyle='--', lw=2,
                       label=f"Mean: {np.mean(errors):.2f}")
        axes[0].axvline(np.median(errors), color='green', linestyle='--', lw=2,
                       label=f"Median: {np.median(errors):.2f}")
        axes[0].set_xlabel('Absolute Error (Days)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title(f'{model_name}: Error Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()

        # Cumulative distribution
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1].plot(sorted_errors, cumulative, linewidth=2)
        axes[1].axvline(1.0, color='red', linestyle='--', alpha=0.7, label='1 day error')
        axes[1].axvline(2.0, color='orange', linestyle='--', alpha=0.7, label='2 day error')
        axes[1].axvline(5.0, color='green', linestyle='--', alpha=0.7, label='5 day error')
        axes[1].set_xlabel('Absolute Error (Days)', fontsize=11)
        axes[1].set_ylabel('Cumulative Probability', fontsize=11)
        axes[1].set_title(f'{model_name}: Cumulative Error Distribution', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_fig(f'error_distribution_{model_name.replace(" ", "_").lower()}.png')

    def plot_performance_profiling(self, profile_times: dict):
        """Plot execution time profiling"""
        if not profile_times:
            return

        phases = list(profile_times.keys())
        times = list(profile_times.values())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
        bars = axes[0].bar(range(len(phases)), times, color=colors, edgecolor='black')
        axes[0].set_xticks(range(len(phases)))
        axes[0].set_xticklabels(phases, rotation=45, ha='right')
        axes[0].set_ylabel('Time (seconds)', fontsize=11)
        axes[0].set_title('Execution Time by Phase', fontsize=12, fontweight='bold')

        # Add value labels
        for bar, time in zip(bars, times):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.1f}s', ha='center', va='bottom', fontsize=9)

        # Pie chart
        axes[1].pie(times, labels=phases, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Time Distribution', fontsize=12, fontweight='bold')

        plt.tight_layout()
        self.save_fig('performance_profiling.png')

    def plot_correlation_heatmap(self, features_df: pd.DataFrame,
                                target_col: str = 'target'):
        """Plot feature correlation heatmap"""
        # Select numeric columns
        numeric_df = features_df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 30:
            # Too many features, select top features
            numeric_df = numeric_df.iloc[:, :30]

        fig, ax = plt.subplots(figsize=(12, 10))
        corr = numeric_df.corr()
        sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
                   ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

        plt.tight_layout()
        self.save_fig('correlation_heatmap.png')
