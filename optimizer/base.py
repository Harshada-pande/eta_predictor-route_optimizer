"""
Common utilities for route optimization.

The optimizer works on a precomputed time matrix.
Each matrix entry time_matrix[i][j] is the travel time
from stop i to stop j, in minutes.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


def route_cost(time_matrix: np.ndarray, route: Sequence[int]) -> float:
    """
    Calculate the total travel time of a route.

    Parameters
    ----------
    time_matrix:
        n x n matrix where time_matrix[i][j] is the travel
        time from stop i to stop j, in minutes.

    route:
        Ordered list of stop indices, e.g. [0, 3, 2, 1].

    Returns
    -------
    float
        Total route travel time in minutes.

    Notes
    -----
    The route does NOT automatically return to the depot.
    If a return to the depot is required later, the route
    can explicitly include the depot again.
    """

    matrix = np.asarray(time_matrix, dtype=float)
    route = list(route)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("time_matrix must be a square matrix.")

    if len(route) <= 1:
        return 0.0

    n = matrix.shape[0]

    if any(stop < 0 or stop >= n for stop in route):
        raise ValueError("Route contains an invalid stop index.")

    total = 0.0

    for i in range(len(route) - 1):
        total += matrix[route[i], route[i + 1]]

    return float(total)