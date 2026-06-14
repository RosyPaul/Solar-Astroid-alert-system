"""
Solar Flare Risk Prediction - Feature Engineering
Reads cleaned flare data, builds the target label (24-hour hazard window)
and final feature set for model training.
"""

import os
import json
import logging
from datetime import timedelta

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'flares_clean.csv')
FEATURES_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'flares_features.csv')
ENCODER_PATH = os.path.join(BASE_DIR, 'model', 'label_encoder.joblib')
FEATURES_LIST_PATH = os.path.join(BASE_DIR, 'model', 'features.json')

FEATURES = [
    'flare_class_encoded',
    'flare_intensity',
    'duration_minutes',
    'activeRegionNum',
    'has_linked_events',
    'hour',
    'month',
    'dayofweek'
]


def load_processed_data(path: str) -> pd.DataFrame:
    """Load cleaned flare data from CSV."""
    try:
        df = pd.read_csv(path, parse_dates=['beginTime'])
        logger.debug(f"Loaded processed data: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load processed data from {path}: {e}")
        raise


def check_next_24hrs(idx, df):
    """Check if a significant flare (M/X class) occurs within 24 hours after this one."""
    current_time = df.loc[idx, 'beginTime']
    next_24 = current_time + timedelta(hours=24)

    future_flares = df[
        (df['beginTime'] > current_time) &
        (df['beginTime'] <= next_24) &
        (df['is_significant'] == 1)
    ]
    return 1 if len(future_flares) > 0 else 0


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Define significant flares and build the 24-hour hazard target label."""
    try:
        df['is_significant'] = df['flare_class'].apply(lambda x: 1 if x in ['M', 'X'] else 0)

        logger.debug("Building target labels (this may take a moment)...")
        df['target'] = [check_next_24hrs(i, df) for i in range(len(df))]

        logger.debug(f"Target distribution:\n{df['target'].value_counts()}")
        logger.debug(f"Hazardous %: {df['target'].mean() * 100:.1f}%")
        return df
    except Exception as e:
        logger.error(f"Failed to build target labels: {e}")
        raise


def build_features(df: pd.DataFrame):
    """Encode categorical features and extract time-based features."""
    try:
        # Encode flare_class (C, M, X -> 0, 1, 2)
        le = LabelEncoder()
        df['flare_class_encoded'] = le.fit_transform(df['flare_class'])

        # Time-based features
        df['hour'] = df['beginTime'].dt.hour
        df['month'] = df['beginTime'].dt.month
        df['dayofweek'] = df['beginTime'].dt.dayofweek

        X = df[FEATURES]
        y = df['target']

        logger.debug(f"Final feature matrix shape: {X.shape}")
        return df, X, y, le
    except Exception as e:
        logger.error(f"Feature building failed: {e}")
        raise


def save_outputs(df: pd.DataFrame, le: LabelEncoder):
    """Save feature dataframe, label encoder, and feature list for downstream steps."""
    try:
        os.makedirs(os.path.dirname(FEATURES_DATA_PATH), exist_ok=True)
        df.to_csv(FEATURES_DATA_PATH, index=False)
        logger.debug(f"Feature data saved to {FEATURES_DATA_PATH}")

        os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
        joblib.dump(le, ENCODER_PATH)
        logger.debug(f"Label encoder saved to {ENCODER_PATH}")

        with open(FEATURES_LIST_PATH, 'w') as f:
            json.dump(FEATURES, f)
        logger.debug(f"Feature list saved to {FEATURES_LIST_PATH}")

    except Exception as e:
        logger.error(f"Failed to save feature engineering outputs: {e}")
        raise


def main():
    try:
        df = load_processed_data(PROCESSED_DATA_PATH)
        df = build_target(df)
        df, X, y, le = build_features(df)
        save_outputs(df, le)
        logger.debug("Feature engineering completed successfully.")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    main()