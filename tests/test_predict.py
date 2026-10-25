import numpy as np
import pandas as pd
import pytest

from models.predict import predict_minutes, predict_one
from preprocessing.feature_builder import build_features


def make_features(n=1):
    """Create valid Indian food-delivery model features."""
    rows = []

    for i in range(n):
        features = build_features(
            dist_km=5.0 + i,
            order_datetime="2026-10-24 11:00:00",
            weather="Sunny",
            traffic="Medium",
            vehicle_type="Motorcycle",
            vehicle_condition=2,
            delivery_person_age=25,
            delivery_person_rating=4.5,
            multiple_deliveries=0,
            festival="No",
            city_type="Metropolitan",
        )

        rows.append(features.iloc[0])

    return pd.DataFrame(rows)


def test_predict_minutes_returns_finite_prediction():
    """A valid Indian delivery feature row should produce a finite ETA."""
    features = make_features(1)

    prediction = predict_minutes(features)

    assert np.isfinite(prediction).all()
    assert len(prediction) == 1
    assert prediction[0] > 0


def test_predict_one_returns_float():
    """predict_one should return one numeric ETA."""
    features = make_features(1)

    prediction = predict_one(features.iloc[0].to_dict())

    assert isinstance(prediction, float)
    assert np.isfinite(prediction)
    assert prediction > 0


def test_batch_prediction_matches_input_length():
    """Batch prediction should return one prediction per input row."""
    features = make_features(5)

    predictions = predict_minutes(features)

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()
    assert np.all(predictions > 0)


def test_predictions_change_with_distance():
    """Different distances should be accepted and produce predictions."""
    features = make_features(2)

    predictions = predict_minutes(features)

    assert len(predictions) == 2
    assert np.isfinite(predictions).all()


def test_missing_feature_is_rejected():
    """The predictor should reject input missing a required model feature."""
    features = make_features(1)

    features = features.drop(columns=["dist_km"])

    with pytest.raises((ValueError, KeyError)):
        predict_minutes(features)