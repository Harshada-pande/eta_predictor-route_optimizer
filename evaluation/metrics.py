"""
Evaluation metrics for ETA prediction.

Includes:
- Regression metrics: MSE, RMSE, MAE, R²
- Delay-risk classification metrics:
  Accuracy, Error Rate, Precision, Recall, F1-score
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics for ETA prediction.

    Returns:
        dict containing MSE, RMSE, MAE, and R².
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def delay_risk_metrics(y_true, y_pred, threshold=None):
    """
    Evaluate ETA prediction as a delay-risk classification problem.

    A trip is classified as 'long/delay-risk' when its duration
    is greater than or equal to the threshold.

    If threshold is not provided, the 75th percentile of actual
    trip durations is used.

    Returns:
        dict containing threshold, accuracy, error rate,
        precision, recall, and F1-score.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if threshold is None:
        threshold = np.percentile(y_true, 75)

    actual_long = y_true >= threshold
    predicted_long = y_pred >= threshold

    accuracy = accuracy_score(actual_long, predicted_long)
    error_rate = 1 - accuracy
    precision = precision_score(
        actual_long,
        predicted_long,
        zero_division=0,
    )
    recall = recall_score(
        actual_long,
        predicted_long,
        zero_division=0,
    )
    f1 = f1_score(
        actual_long,
        predicted_long,
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def evaluate_predictions(y_true, y_pred, threshold=None):
    """
    Run both regression and delay-risk evaluation.

    Returns:
        Combined dictionary of all evaluation metrics.
    """

    results = {}

    results.update(
        regression_metrics(y_true, y_pred)
    )

    results.update(
        delay_risk_metrics(
            y_true,
            y_pred,
            threshold=threshold,
        )
    )

    return results