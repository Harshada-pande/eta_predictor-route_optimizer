import numpy as np
import pytest

from evaluation.metrics import (
    regression_metrics,
    delay_risk_metrics,
    evaluate_predictions,
)


def test_regression_metrics():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([12, 18, 33, 37])

    metrics = regression_metrics(y_true, y_pred)

    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_perfect_predictions():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([10, 20, 30, 40])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["mse"] == pytest.approx(0)
    assert metrics["rmse"] == pytest.approx(0)
    assert metrics["mae"] == pytest.approx(0)
    assert metrics["r2"] == pytest.approx(1)


def test_delay_risk_metrics():
    y_true = np.array([20, 25, 30, 40, 50])
    y_pred = np.array([21, 24, 31, 38, 48])

    metrics = delay_risk_metrics(
        y_true,
        y_pred,
        threshold=32,
    )

    assert "threshold" in metrics
    assert "accuracy" in metrics
    assert "error_rate" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

    assert metrics["threshold"] == 32
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["error_rate"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1


def test_evaluate_predictions_combines_metrics():
    y_true = np.array([20, 25, 30, 40, 50])
    y_pred = np.array([21, 24, 31, 38, 48])

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        threshold=32,
    )

    # Regression metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    # Delay-risk metrics
    assert "threshold" in metrics
    assert "accuracy" in metrics
    assert "error_rate" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics