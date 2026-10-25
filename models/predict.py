"""
ETA prediction interface for the Indian food-delivery model.

The trained XGBoost model expects 27 features.
This module provides a simple prediction interface without exposing
the 27 model columns directly to the application.

Higher-level application code can generate the feature values from:
- user inputs
- weather API
- traffic API
- Indian festival/calendar data
- derived date/time features
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Paths and cached artifacts
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_MODEL = None
_FEATURE_COLUMNS = None


def _load_artifacts():
    """Load the trained model and feature list once."""

    global _MODEL, _FEATURE_COLUMNS

    if _MODEL is None:
        with open(
            PROJECT_ROOT / "configs" / "config.yaml",
            "r",
            encoding="utf-8",
        ) as f:
            cfg = yaml.safe_load(f)

        model_path = PROJECT_ROOT / cfg["paths"]["best_model"]
        feature_path = PROJECT_ROOT / cfg["paths"]["feature_list"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature list not found: {feature_path}"
            )

        _MODEL = joblib.load(model_path)

        with open(feature_path, "r", encoding="utf-8") as f:
            _FEATURE_COLUMNS = json.load(f)

    return _MODEL, _FEATURE_COLUMNS


# ---------------------------------------------------------------------
# Public prediction function
# ---------------------------------------------------------------------

def predict_minutes(features: pd.DataFrame | dict[str, Any]) -> np.ndarray:
    """
    Predict delivery time in minutes.

    Parameters
    ----------
    features:
        A pandas DataFrame or dictionary containing the model features.

        The input must contain the same 27 feature columns used during
        training.

    Returns
    -------
    np.ndarray
        Predicted delivery time in minutes.
    """

    model, feature_columns = _load_artifacts()

    # Convert a single dictionary into a one-row DataFrame.
    if isinstance(features, dict):
        features = pd.DataFrame([features])

    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            "features must be a pandas DataFrame or dictionary."
        )

    # Check that all expected features exist.
    missing_features = [
        column
        for column in feature_columns
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(missing_features)
        )

    # IMPORTANT:
    # Select columns in exactly the same order used during training.
    X = features[feature_columns]

    predictions = model.predict(X)

    # Delivery time cannot be negative.
    predictions = np.maximum(predictions, 0.5)

    return np.asarray(predictions, dtype=float)


# ---------------------------------------------------------------------
# Convenience function for one prediction
# ---------------------------------------------------------------------

def predict_one(features: dict[str, Any]) -> float:
    """
    Predict ETA for one delivery and return a single float.
    """

    prediction = predict_minutes(features)

    return float(prediction[0])