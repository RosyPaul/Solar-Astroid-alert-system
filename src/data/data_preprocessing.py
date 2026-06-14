"""
Solar Flare Risk Prediction - Data Preprocessing
Reads raw flare data, cleans and structures it, saves as processed CSV
for the feature engineering step.
"""

import os
import logging
from datetime import timedelta

import pandas as pd


# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logger = logging.getLogger('data_preprocessing')
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
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'flares_raw.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'flares_clean.csv')


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw flare data from CSV."""
    try:
        df = pd.read_csv(path)
        logger.debug(f"Loaded raw data: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load raw data from {path}: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns, parse types, handle missing values."""
    try:
        df = df[['flrID', 'beginTime', 'peakTime', 'endTime', 'classType',
                 'sourceLocation', 'activeRegionNum', 'linkedEvents']].copy()
        logger.debug(f"Selected columns. Shape: {df.shape}")

        # Split flare class and intensity, e.g. "M5.2" -> class "M", intensity 5.2
        df['flare_class'] = df['classType'].str[0]
        df['flare_intensity'] = df['classType'].str[1:].astype(float)
        logger.debug("Split classType into flare_class and flare_intensity")

        # Parse timestamps
        df['beginTime'] = pd.to_datetime(df['beginTime'])
        df['peakTime'] = pd.to_datetime(df['peakTime'])
        df['endTime'] = pd.to_datetime(df['endTime'])

        # Duration of the flare
        df['duration_minutes'] = (df['endTime'] - df['beginTime']).dt.total_seconds() / 60

        # Whether the flare has linked events (CMEs, etc.)
        df['has_linked_events'] = df['linkedEvents'].apply(lambda x: 1 if pd.notna(x) and x not in ('[]', '', None) else 0)

        # Drop columns not needed further
        df = df.drop(columns=['flrID', 'linkedEvents', 'classType'])

        # Handle missing values
        df['duration_minutes'] = df['duration_minutes'].fillna(df['duration_minutes'].median())
        df['activeRegionNum'] = df['activeRegionNum'].fillna(0)
        df = df.drop(columns=['endTime', 'peakTime'])

        # Sort chronologically
        df = df.sort_values('beginTime').reset_index(drop=True)
        df['date'] = df['beginTime'].dt.date

        logger.debug(f"Cleaning complete. Final shape: {df.shape}")
        logger.debug(f"Missing values:\n{df.isnull().sum()}")
        return df

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise


def save_processed_data(df: pd.DataFrame, path: str):
    """Save cleaned data to CSV."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug(f"Processed data saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        raise


def main():
    try:
        df = load_raw_data(RAW_DATA_PATH)
        df = clean_data(df)
        save_processed_data(df, PROCESSED_DATA_PATH)
        logger.debug("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()