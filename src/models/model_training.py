"""
Solar Flare Risk Prediction - Model Training
Reads engineered features, trains XGBoost with MLflow tracking,
and saves the trained model artifact.
"""

import os
import logging

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost


# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logger = logging.getLogger('model_training')
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
FEATURES_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'flares_features.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'flare_model.joblib')

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


def load_features(path: str):
    """Load engineered feature dataframe and split into X, y."""
    try:
        df = pd.read_csv(path)
        X = df[FEATURES]
        y = df['target']
        logger.debug(f"Loaded features: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to load features from {path}: {e}")
        raise

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("solar_flare_prediction")

def train_model(X, y):
    """Split data, train XGBoost with MLflow tracking, return trained model."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.debug(f"Train/test split: train={X_train.shape}, test={X_test.shape}")

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale = neg / pos
        logger.debug(f"Class imbalance scale_pos_weight={scale:.4f}")

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

            logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
            logger.debug(f"ROC-AUC Score: {auc_score:.4f}")

            mlflow.log_params({
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "scale_pos_weight": scale
            })
            mlflow.log_metric("roc_auc", auc_score)
            mlflow.xgboost.log_model(model, "model")

        return model

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def save_model(model, path: str):
    """Save trained model to disk."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.debug(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise


def main():
    try:
        X, y = load_features(FEATURES_DATA_PATH)
        model = train_model(X, y)
        save_model(model, MODEL_PATH)
        logger.debug("Model training completed successfully.")
    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()