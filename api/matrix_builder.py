"""
Build the pairwise ETA matrix for the Indian food-delivery model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from models.predict import predict_one
from preprocessing.feature_builder import build_features


Coordinate = tuple[float, float]
# Distance range represented in the training data.
TRAINING_DISTANCE_MIN_KM = 1.47
TRAINING_DISTANCE_MAX_KM = 20.97

def find_clipped_legs(
    coordinates: Sequence[Coordinate],
) -> list[dict]:
    """
    Find directed legs whose geographic distance falls outside
    the model's training distance range.
    """
    coords = np.asarray(coordinates, dtype=float)
    clipped_legs = []

    for i in range(len(coords)):
        for j in range(len(coords)):

            if i == j:
                continue

            distance_km = haversine_km(
                coords[i, 0],
                coords[i, 1],
                coords[j, 0],
                coords[j, 1],
            )

            clipped_distance, was_clipped = clip_distance_km(
                distance_km
            )

            if was_clipped:
                clipped_legs.append(
                    {
                        "from_index": i,
                        "to_index": j,
                        "original_distance_km": distance_km,
                        "clipped_distance_km": clipped_distance,
                    }
                )
    return clipped_legs














def clip_distance_km(distance_km: float) -> tuple[float, bool]:
    """
    Clip a geographic distance to the model's supported training range.

    Returns
    -------
    tuple[float, bool]
        (distance used by the model, whether clipping occurred)
    """

    if distance_km < TRAINING_DISTANCE_MIN_KM:
        return TRAINING_DISTANCE_MIN_KM, True

    if distance_km > TRAINING_DISTANCE_MAX_KM:
        return TRAINING_DISTANCE_MAX_KM, True

    return distance_km, False

def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate great-circle distance between two coordinates in km.
    """

    earth_radius_km = 6371.0

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a),
    )

    return float(earth_radius_km * c)


def build_eta_matrix(
    coordinates: Sequence[Coordinate],
    departure: str | datetime,
    contexts: Sequence[dict],
) -> np.ndarray:
    """
    Build an n x n ETA matrix.

    coordinates[0] is the depot.
    coordinates[1:] are delivery stops.

    Each stop has its own model context.

    For every directed pair i -> j:
        1. Calculate Haversine distance.
        2. Build the model's 27 features.
        3. Predict ETA using XGBoost.

    Diagonal entries are zero.
    """

    coords = np.asarray(
        coordinates,
        dtype=float,
    )

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "coordinates must contain (latitude, longitude) pairs."
        )

    if len(coords) == 0:
        raise ValueError(
            "At least one coordinate is required."
        )

    if not np.isfinite(coords).all():
        raise ValueError(
            "Coordinates must be finite numbers."
        )

    if np.any((coords[:, 0] < -90) | (coords[:, 0] > 90)):
        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if np.any((coords[:, 1] < -180) | (coords[:, 1] > 180)):
        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    n = len(coords)

    if len(contexts) != n:
        raise ValueError(
            f"Expected {n} contexts, received {len(contexts)}."
        )

    matrix = np.zeros(
        (n, n),
        dtype=float,
    )

    departure_dt = (
        datetime.fromisoformat(
            departure.replace("Z", "+00:00")
        )
        if isinstance(departure, str)
        else departure
    )

    # Build every directed edge.
    for i in range(n):

        for j in range(n):

            if i == j:
                continue

            distance_km = haversine_km(
                coords[i, 0],
                coords[i, 1],
                coords[j, 0],
                coords[j, 1],
            )

            distance_km, _ = clip_distance_km(distance_km)

            # Use the destination stop's delivery context.
            context = contexts[j].copy()

            features = build_features(
                dist_km=distance_km,
                order_datetime=departure_dt,
                **context,
            )

            matrix[i, j] = predict_one(
                features.iloc[0].to_dict()
            )

    return matrix