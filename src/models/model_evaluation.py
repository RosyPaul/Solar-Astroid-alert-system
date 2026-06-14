"""
Solar Flare Risk Prediction - Model Evaluation
Loads the trained model and test data, computes evaluation metrics,
logs them to MLflow, and saves a metrics report.
"""

import os
import json
import logging

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import mlflow


# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logger = logging.getLogger('model_evaluation')
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
METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'metrics.json')

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

ALERT_THRESHOLD = 0.4


def load_data_and_model(features_path: str, model_path: str):
    """Load feature data and trained model."""
    try:
        df = pd.read_csv(features_path)
        X = df[FEATURES]
        y = df['target']
        logger.debug(f"Loaded features: X={X.shape}, y={y.shape}")

        model = joblib.load(model_path)
        logger.debug(f"Loaded model from {model_path}")

        return X, y, model
    except Exception as e:
        logger.error(f"Failed to load data/model: {e}")
        raise


def evaluate_model(model, X, y):
    """Recreate the same test split used in training and compute metrics."""
    try:
        # Same split params as model_training.py, so this is the same held-out test set
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Apply same alert threshold as the prediction API
        y_pred_threshold = (y_prob >= ALERT_THRESHOLD).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred_threshold),
            "recall": recall_score(y_test, y_pred_threshold),
            "f1_score": f1_score(y_test, y_pred_threshold),
        }

        cm = confusion_matrix(y_test, y_pred_threshold)
        report = classification_report(y_test, y_pred_threshold, output_dict=True)

        logger.debug(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.debug(f"Precision: {metrics['precision']:.4f}")
        logger.debug(f"Recall: {metrics['recall']:.4f}")
        logger.debug(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.debug(f"Confusion Matrix:\n{cm}")
        logger.debug(f"Classification report:\n{classification_report(y_test, y_pred_threshold)}")

        return metrics, cm, report

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


def save_metrics(metrics: dict, cm, report: dict, path: str):
    """Save evaluation metrics to a JSON report and log to MLflow."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        output = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }

        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.debug(f"Metrics saved to {path}")

        # Log to MLflow as a separate run for evaluation
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("solar_flare_prediction")
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(path)

    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        raise


def main():
    try:
        X, y, model = load_data_and_model(FEATURES_DATA_PATH, MODEL_PATH)
        metrics, cm, report = evaluate_model(model, X, y)
        save_metrics(metrics, cm, report, METRICS_PATH)
        logger.debug("Model evaluation completed successfully.")
    except Exception as e:
        logger.error(f"Model evaluation pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()