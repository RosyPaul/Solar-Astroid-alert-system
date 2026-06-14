"""
Solar Flare Risk Prediction - Data Ingestion
Fetches latest NASA DONKI flare data and saves it as raw CSV
for the data cleaning/preprocessing step.
"""

import os
import logging
from datetime import date

import requests
import pandas as pd
from dotenv import load_dotenv


# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logger = logging.getLogger('data_ingestion')
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
# Fetch data from NASA DONKI API
# ----------------------------------------------------------------------
load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'flares_raw.csv')


def fetch_solar_flares(start_date, end_date):
    """Fetch solar flare data from NASA DONKI API."""
    try:
        url = "https://api.nasa.gov/DONKI/FLR"
        params = {"startDate": start_date, "endDate": end_date, "api_key": NASA_API_KEY}
        logger.debug(f"Requesting NASA DONKI API from {start_date} to {end_date}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from NASA API: {e}")
        raise


def save_raw_data(df: pd.DataFrame, path: str):
    """Save raw fetched data to CSV."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.debug(f"Raw data saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save raw data: {e}")
        raise


def main():
    try:
        if not NASA_API_KEY:
            logger.error("NASA_API_KEY not found. Check your .env file.")
            raise ValueError("Missing NASA_API_KEY")

        flares = fetch_solar_flares("2020-01-01", str(date.today()))
        logger.debug(f"Total flares fetched: {len(flares)}")

        df = pd.DataFrame(flares)
        logger.debug(f"Raw dataframe shape: {df.shape}")

        save_raw_data(df, RAW_DATA_PATH)
        logger.debug("Data ingestion completed successfully.")

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()