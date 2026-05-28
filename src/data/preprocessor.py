# Data preprocessing for MIMIC-III data
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MIMICPreprocessor:
    """Preprocess MIMIC-III data for ML pipeline"""

    @staticmethod
    def validate_data(df: pd.DataFrame, required_cols: list) -> bool:
        """Validate that required columns exist"""
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        return True

    @staticmethod
    def parse_timestamps(df: pd.DataFrame, timestamp_cols: list) -> pd.DataFrame:
        """Parse timestamp columns"""
        df = df.copy()
        for col in timestamp_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_datetime(df[col])
        return df

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop',
                              threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values.

        Args:
            df: Input dataframe
            strategy: 'drop' columns with high missing, 'forward_fill' for time series
            threshold: Drop columns with > threshold missing (for drop strategy)
        """
        df = df.copy()

        if strategy == 'drop':
            # Drop columns with too many missing values
            missing_pct = df.isnull().sum() / len(df)
            cols_to_drop = missing_pct[missing_pct > threshold].index
            logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing")
            df = df.drop(columns=cols_to_drop)

            # Drop rows with remaining NaNs in critical columns
            critical_cols = [c for c in df.columns if c not in ['value', 'label']]
            df = df.dropna(subset=critical_cols)

        elif strategy == 'forward_fill':
            # Forward fill for time series data
            df = df.sort_values('charttime').fillna(method='ffill')

        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str,
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numerical column.

        Args:
            df: Input dataframe
            column: Column to clean
            method: 'iqr' for interquartile range, 'zscore' for z-score
            threshold: IQR multiplier or z-score threshold
        """
        if column not in df.columns or df[column].dtype == 'object':
            return df

        df = df.copy()
        col_data = df[column]

        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            before = len(df)
            df = df[(col_data >= lower) & (col_data <= upper)]
            logger.info(f"Removed {before - len(df)} outliers from {column} (IQR method)")

        elif method == 'zscore':
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            before = len(df)
            df = df[z_scores < threshold]
            logger.info(f"Removed {before - len(df)} outliers from {column} (zscore method)")

        return df

    @staticmethod
    def normalize_values(df: pd.DataFrame, method: str = 'minmax',
                        numeric_only: bool = True) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize numerical values.

        Returns:
            Normalized dataframe and normalization parameters for inverse transform
        """
        df = df.copy()
        norm_params = {}

        if numeric_only:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = df.columns

        for col in numeric_cols:
            if col not in df.columns:
                continue

            valid_data = df[col].dropna()
            if len(valid_data) == 0:
                continue

            if method == 'minmax':
                min_val = valid_data.min()
                max_val = valid_data.max()
                if max_val == min_val:
                    df[col] = 0
                else:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                norm_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}

            elif method == 'standard':
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                if std_val == 0:
                    df[col] = 0
                else:
                    df[col] = (df[col] - mean_val) / std_val
                norm_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'standard'}

        return df, norm_params

    @staticmethod
    def create_target_variable(admissions_df: pd.DataFrame,
                               classification: bool = False,
                               short_threshold: int = 3,
                               medium_threshold: int = 7) -> pd.DataFrame:
        """
        Create target variable for prediction.

        Args:
            admissions_df: Admissions dataframe with length_of_stay column (days)
            classification: If True, create classes (short/medium/long)
            short_threshold: Days threshold for short stay
            medium_threshold: Days threshold for medium stay
        """
        df = admissions_df.copy()

        if not 'length_of_stay' in df.columns:
            raise ValueError("length_of_stay column not found in admissions data")

        if classification:
            # Multi-class classification
            df['los_class'] = pd.cut(
                df['length_of_stay'],
                bins=[0, short_threshold, medium_threshold, np.inf],
                labels=['short', 'medium', 'long'],
                include_lowest=True
            )
            logger.info(f"Created classification target: {df['los_class'].value_counts().to_dict()}")
            return df[['hadm_id', 'subject_id', 'los_class']].rename(
                columns={'los_class': 'target'}
            )
        else:
            # Regression: predict days
            logger.info(f"Length of stay statistics:\n{df['length_of_stay'].describe()}")
            return df[['hadm_id', 'subject_id', 'length_of_stay']].rename(
                columns={'length_of_stay': 'target'}
            )

    @staticmethod
    def clean_chart_events(chart_events_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate chart events data"""
        df = chart_events_df.copy()

        # Parse timestamps
        df = MIMICPreprocessor.parse_timestamps(df, ['charttime'])

        # Remove rows with null values and icustay_id
        df = df.dropna(subset=['icustay_id', 'charttime', 'itemid'])

        # Handle value column - remove completely null values
        if 'value' in df.columns:
            # Keep only rows with numeric values
            df = df[pd.notna(df['value'])]

            # Remove obvious invalid values (negative for most vital signs)
            # Keep it general - specific validation per itemid in feature engineering
            logger.info(f"Cleaned chart events: {len(df)} records")

        return df

    @staticmethod
    def aggregate_chart_events(chart_events_df: pd.DataFrame,
                              aggregation: str = 'mean') -> pd.DataFrame:
        """
        Aggregate chart events to reduce dimensionality.

        Args:
            chart_events_df: Raw chart events
            aggregation: 'mean', 'median', 'min', 'max', 'std'
        """
        df = chart_events_df.copy()

        if aggregation == 'mean':
            agg_func = 'mean'
        elif aggregation == 'median':
            agg_func = 'median'
        elif aggregation == 'min':
            agg_func = 'min'
        elif aggregation == 'max':
            agg_func = 'max'
        elif aggregation == 'std':
            agg_func = 'std'
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Group by ICU stay and item
        aggregated = df.groupby(['icustay_id', 'hadm_id', 'subject_id', 'itemid', 'label']).agg({
            'value': agg_func,
            'charttime': ['min', 'max', 'count']
        }).reset_index()

        aggregated.columns = ['icustay_id', 'hadm_id', 'subject_id', 'itemid', 'label',
                             f'value_{aggregation}', 'first_time', 'last_time', 'count']

        logger.info(f"Aggregated from {len(df)} records to {len(aggregated)} aggregates")
        return aggregated
