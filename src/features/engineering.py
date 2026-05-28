# Feature engineering for MIMIC-III LOS prediction
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Extract and engineer features from MIMIC-III chart events"""

    # Common vital signs and lab values with reasonable ranges (for validation)
    VITAL_SIGN_RANGES = {
        'Heart Rate': (30, 200),
        'Respiratory Rate': (5, 60),
        'Temperature Fahrenheit': (95, 107),
        'Systolic Blood Pressure': (50, 250),
        'Diastolic Blood Pressure': (20, 150),
        'Mean Blood Pressure': (30, 180),
        'SpO2': (50, 100),
        'Glucose': (50, 600),
        'Potassium': (1, 8),
        'Sodium': (100, 160),
        'Hemoglobin': (5, 20),
        'Hematocrit': (10, 60),
        'Creatinine': (0.3, 15),
        'BUN': (5, 150),
    }

    @staticmethod
    def validate_vital_sign(label: str, value: float) -> bool:
        """Check if vital sign value is in reasonable range"""
        if label not in FeatureEngineer.VITAL_SIGN_RANGES:
            return True  # No validation rule

        min_val, max_val = FeatureEngineer.VITAL_SIGN_RANGES[label]
        return min_val <= value <= max_val

    @staticmethod
    def extract_temporal_features(chart_events_df: pd.DataFrame,
                                  icu_stays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from chart events.

        Features:
        - Time from ICU admission
        - Measurement frequency
        - Measurement gaps
        """
        df = chart_events_df.copy()

        # Merge with ICU stay times
        icu_times = icu_stays_df[['icustay_id', 'intime', 'outtime']].copy()
        df = df.merge(icu_times, on='icustay_id', how='left')

        # Parse timestamps
        df['charttime'] = pd.to_datetime(df['charttime'])
        df['intime'] = pd.to_datetime(df['intime'])
        df['outtime'] = pd.to_datetime(df['outtime'])

        # Hours from ICU admission
        df['hours_from_admission'] = (df['charttime'] - df['intime']).dt.total_seconds() / 3600
        df = df[df['hours_from_admission'] >= 0]  # Remove measurements before admission

        # Hours until discharge
        df['hours_until_discharge'] = (df['outtime'] - df['charttime']).dt.total_seconds() / 3600

        # Day of stay
        df['day_of_stay'] = (df['hours_from_admission'] // 24).astype(int) + 1

        logger.info(f"Extracted temporal features for {df['icustay_id'].nunique()} ICU stays")
        return df

    @staticmethod
    def aggregate_by_time_window(chart_events_df: pd.DataFrame,
                                window_hours: int = 24) -> pd.DataFrame:
        """
        Aggregate measurements into time windows.

        Returns dataframe with one row per ICU stay per time window
        """
        df = chart_events_df.copy()

        if 'charttime' in df.columns:
            df['charttime'] = pd.to_datetime(df['charttime'])

        # Create time windows
        df['time_window'] = (df['hours_from_admission'] // window_hours).astype(int)

        # Aggregate by window
        agg_dict = {
            'value': ['mean', 'min', 'max', 'std', 'count'],
            'hours_from_admission': 'first',
            'hours_until_discharge': 'first',
            'day_of_stay': 'first',
        }

        aggregated = df.groupby(
            ['icustay_id', 'hadm_id', 'subject_id', 'itemid', 'label', 'time_window']
        ).agg(agg_dict).reset_index()

        aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns.values]
        logger.info(f"Created {len(aggregated)} features from time-windowed aggregation")

        return aggregated

    @staticmethod
    def pivot_to_features(aggregated_df: pd.DataFrame,
                         target_col: str = 'value_mean',
                         feature_prefix: str = '') -> pd.DataFrame:
        """
        Pivot aggregated data to wide format (one row per sample, one column per feature).
        """
        df = aggregated_df.copy()

        # Create feature name: label + suffix
        df['feature_name'] = df['label'] + '_' + target_col.split('_', 1)[1]

        # Pivot wider
        pivot_cols = ['icustay_id', 'hadm_id', 'subject_id', 'time_window']
        features = df.pivot_table(
            index=pivot_cols,
            columns='feature_name',
            values=target_col,
            aggfunc='first'
        ).reset_index()

        logger.info(f"Created feature matrix with {len(features)} samples and {features.shape[1]-4} features")
        return features

    @staticmethod
    def extract_first_day_features(chart_events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from first 24 hours of ICU stay.
        This is a common approach for early LOS prediction.
        """
        df = chart_events_df.copy()

        # Filter to first 24 hours
        first_day = df[df['hours_from_admission'] <= 24].copy()

        logger.info(f"Extracted first day data: {len(first_day)} measurements for {first_day['icustay_id'].nunique()} stays")

        # Aggregate by vital sign/lab value
        features = first_day.groupby(['icustay_id', 'hadm_id', 'subject_id', 'label']).agg({
            'value': ['mean', 'min', 'max', 'std', 'count'],
            'hours_from_admission': 'mean'
        }).reset_index()

        features.columns = ['icustay_id', 'hadm_id', 'subject_id', 'label',
                           'value_mean', 'value_min', 'value_max', 'value_std', 'value_count',
                           'hours_avg']

        # Filter features with enough measurements
        features = features[features['value_count'] >= 2]

        # Create wide format
        result = features.pivot_table(
            index=['icustay_id', 'hadm_id', 'subject_id'],
            columns='label',
            values='value_mean',
            aggfunc='first'
        ).reset_index()

        logger.info(f"Created first-day features: {result.shape[1]-3} features")
        return result

    @staticmethod
    def extract_demographic_features(patients_df: pd.DataFrame,
                                    admissions_df: pd.DataFrame,
                                    icu_stays_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract demographic and static features.
        """
        # Merge demographic data
        df = icu_stays_df.copy()
        df = df.merge(admissions_df[['hadm_id', 'admission_type']], on='hadm_id', how='left')
        df = df.merge(patients_df[['subject_id', 'gender']], on='subject_id', how='left')

        # Age at admission
        df['intime'] = pd.to_datetime(df['intime'])
        admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
        admissions_time = admissions_df[['hadm_id', 'admittime']].drop_duplicates()
        df = df.merge(admissions_time, on='hadm_id', how='left')

        if 'dob' in patients_df.columns:
            patients_dob = patients_df[['subject_id', 'dob']].copy()
            patients_dob['dob'] = pd.to_datetime(patients_dob['dob'])
            df = df.merge(patients_dob, on='subject_id', how='left')
            df['age_at_admission'] = (df['admittime'] - df['dob']).dt.days / 365.25
        else:
            df['age_at_admission'] = 50  # Default if DOB not available

        # Length of ICU stay
        df['los_icu_hours'] = (df['outtime'] - df['intime']).dt.total_seconds() / 3600

        # Encode categorical features
        df['is_male'] = (df['gender'] == 'M').astype(int)
        if 'admission_type' in df.columns:
            admission_dummies = pd.get_dummies(df['admission_type'], prefix='admission')
            df = pd.concat([df, admission_dummies], axis=1)

        # Select numeric features
        feature_cols = ['age_at_admission', 'los_icu_hours', 'is_male']
        admission_cols = [c for c in df.columns if c.startswith('admission_')]
        feature_cols.extend(admission_cols)

        result = df[['icustay_id', 'hadm_id', 'subject_id'] + feature_cols].copy()
        logger.info(f"Created demographic features: {len(feature_cols)} features")

        return result

    @staticmethod
    def combine_features(vital_features_df: pd.DataFrame,
                        demographic_features_df: pd.DataFrame,
                        fill_method: str = 'forward_fill') -> pd.DataFrame:
        """Combine all feature sets"""
        df = vital_features_df.merge(
            demographic_features_df,
            on=['icustay_id', 'hadm_id', 'subject_id'],
            how='inner'
        )

        # Handle missing values in vital signs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                if fill_method == 'forward_fill':
                    df[col] = df[col].fillna(method='ffill')
                elif fill_method == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif fill_method == 'zero':
                    df[col] = df[col].fillna(0)

        logger.info(f"Combined features: {df.shape}")
        return df

    @staticmethod
    def select_top_features(features_df: pd.DataFrame,
                           targets_df: pd.DataFrame,
                           n_features: int = 20,
                           method: str = 'correlation') -> List[str]:
        """
        Select top N most important features using correlation or other methods.
        """
        # Merge features with target
        df = features_df.merge(targets_df, on='hadm_id', how='inner')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['icustay_id', 'hadm_id', 'subject_id']]

        if method == 'correlation':
            correlations = df[numeric_cols + ['target']].corr()['target'].abs().sort_values(ascending=False)
            top_features = correlations[1:n_features+1].index.tolist()
            logger.info(f"Top features by correlation: {top_features[:5]}")

        return top_features
