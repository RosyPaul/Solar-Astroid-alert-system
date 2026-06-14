"""
Solar Flare Risk Prediction - Training Pipeline
Fetches latest NASA DONKI flare data, engineers features, trains XGBoost,
logs experiment to MLflow, and saves model artifacts for the prediction API.
"""

import os
import json
from datetime import date, timedelta

import requests
import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost


# ----------------------------------------------------------------------
# 1. Fetch data from NASA DONKI API
# ----------------------------------------------------------------------
load_dotenv()
NASA_API_KEY = os.getenv("NASA_API_KEY")


def fetch_solar_flares(start_date, end_date):
    url = "https://api.nasa.gov/DONKI/FLR"
    params = {"startDate": start_date, "endDate": end_date, "api_key": NASA_API_KEY}
    response = requests.get(url, params=params)
    return response.json()


print("Fetching solar flare data from NASA DONKI...")
flares = fetch_solar_flares("2020-01-01", str(date.today()))
print(f"Total flares fetched: {len(flares)}")

df = pd.DataFrame(flares)


# ----------------------------------------------------------------------
# 2. Feature engineering (matches notebook cells 8-26)
# ----------------------------------------------------------------------
df = df[['flrID', 'beginTime', 'peakTime', 'endTime', 'classType',
         'sourceLocation', 'activeRegionNum', 'linkedEvents']]

# Split flare class and intensity, e.g. "M5.2" -> class "M", intensity 5.2
df['flare_class'] = df['classType'].str[0]
df['flare_intensity'] = df['classType'].str[1:].astype(float)

# Parse timestamps
df['beginTime'] = pd.to_datetime(df['beginTime'])
df['peakTime'] = pd.to_datetime(df['peakTime'])
df['endTime'] = pd.to_datetime(df['endTime'])

# Duration of the flare
df['duration_minutes'] = (df['endTime'] - df['beginTime']).dt.total_seconds() / 60

# Whether the flare has linked events (CMEs, etc.)
df['has_linked_events'] = df['linkedEvents'].apply(lambda x: 1 if x else 0)

# Drop columns not needed further
df = df.drop(columns=['flrID', 'linkedEvents', 'classType'])

# Handle missing values
df['duration_minutes'] = df['duration_minutes'].fillna(df['duration_minutes'].median())
df['activeRegionNum'] = df['activeRegionNum'].fillna(0)
df = df.drop(columns=['endTime', 'peakTime'])

# Sort chronologically
df = df.sort_values('beginTime').reset_index(drop=True)
df['date'] = df['beginTime'].dt.date

# Define "significant" flares (M or X class)
df['is_significant'] = df['flare_class'].apply(lambda x: 1 if x in ['M', 'X'] else 0)


# Target: will a significant flare occur in the next 24 hours?
def check_next_24hrs(idx, df):
    current_time = df.loc[idx, 'beginTime']
    next_24 = current_time + timedelta(hours=24)

    future_flares = df[
        (df['beginTime'] > current_time) &
        (df['beginTime'] <= next_24) &
        (df['is_significant'] == 1)
    ]
    return 1 if len(future_flares) > 0 else 0


print("Building target labels...")
df['target'] = [check_next_24hrs(i, df) for i in range(len(df))]
print(df['target'].value_counts())
print(f"Hazardous %: {df['target'].mean() * 100:.1f}%")

# Encode flare_class (C, M, X -> 0, 1, 2)
le = LabelEncoder()
df['flare_class_encoded'] = le.fit_transform(df['flare_class'])

# Time-based features
df['hour'] = df['beginTime'].dt.hour
df['month'] = df['beginTime'].dt.month
df['dayofweek'] = df['beginTime'].dt.dayofweek

# Final feature set
features = [
    'flare_class_encoded',
    'flare_intensity',
    'duration_minutes',
    'activeRegionNum',
    'has_linked_events',
    'hour',
    'month',
    'dayofweek'
]

X = df[features]
y = df['target']

print(X.shape)


# ----------------------------------------------------------------------
# 3. Train/test split + handle class imbalance
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos


# ----------------------------------------------------------------------
# 4. Train XGBoost with MLflow tracking
# ----------------------------------------------------------------------
mlflow.set_experiment("solar_flare_prediction")

with mlflow.start_run():

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {auc_score:.4f}")

    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "scale_pos_weight": scale
    })
    mlflow.log_metric("roc_auc", auc_score)
    mlflow.xgboost.log_model(model, "model")


# ----------------------------------------------------------------------
# 5. Save artifacts for the prediction API (app/predict.py expects these)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Save raw data (DVC will track this)
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
df.to_csv(os.path.join(BASE_DIR, 'data', 'solar_flare_data.csv'), index=False)

# Save model artifacts (DVC will track these)
os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)
joblib.dump(model, os.path.join(BASE_DIR, 'model', 'flare_model.joblib'))
joblib.dump(le, os.path.join(BASE_DIR, 'model', 'label_encoder.joblib'))

with open(os.path.join(BASE_DIR, 'model', 'features.json'), 'w') as f:
    json.dump(features, f)

print("Training complete. Model and artifacts saved.")