"""
Graph and time-matrix utilities.

Phase 1 starts with a simple distance-based travel-time matrix:
Haversine distance at an assumed average speed of 30 km/h.

Later, this matrix can be replaced by ETA predictions from
the trained XGBoost model without changing the optimizer.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


EARTH_RADIUS_KM = 6371.0
AVERAGE_SPEED_KMH = 30.0


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate great-circle distance between two latitude/longitude points.

    Returns distance in kilometres.
    """

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return float(EARTH_RADIUS_KM * c)


def build_distance_matrix(
    coordinates: Sequence[tuple[float, float]],
    speed_kmh: float = AVERAGE_SPEED_KMH,
) -> np.ndarray:
    """
    Build an n x n travel-time matrix.

    Parameters
    ----------
    coordinates:
        Sequence of (latitude, longitude) pairs.

    speed_kmh:
        Assumed average travel speed.
        Default = 30 km/h.

    Returns
    -------
    np.ndarray
        n x n matrix containing travel time in minutes.

    Formula
    -------
    time_minutes = distance_km / speed_kmh * 60
    """

    if speed_kmh <= 0:
        raise ValueError("speed_kmh must be greater than zero.")

    coordinates = list(coordinates)
    n = len(coordinates)

    if n == 0:
        return np.empty((0, 0), dtype=float)

    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        lat1, lon1 = coordinates[i]

        for j in range(n):
            if i == j:
                continue

            lat2, lon2 = coordinates[j]

            distance_km = haversine_km(
                lat1,
                lon1,
                lat2,
                lon2,
            )

            travel_time_minutes = (
                distance_km / speed_kmh
            ) * 60.0

            matrix[i, j] = travel_time_minutes

    return matrix