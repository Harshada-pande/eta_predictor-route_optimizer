"""
Hill Climbing route optimizer.

The optimizer works entirely on a precomputed time matrix.
The ETA model is NOT called inside the search loop.

Neighbourhoods:
1. Swap
2. 2-opt

The algorithm repeatedly moves to the best improving neighbour.
When no neighbour improves the current route, it stops at a local optimum.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from optimizer.base import route_cost


def swap_neighbors(route: Sequence[int]):
    """
    Generate neighbours by swapping two stops.

    The start node is kept fixed.
    """
    route = list(route)

    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            neighbour = route.copy()
            neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
            yield neighbour


def two_opt_neighbors(route: Sequence[int]):
    """
    Generate neighbours using 2-opt.

    The start node is kept fixed.

    2-opt reverses a section of the route.
    """
    route = list(route)

    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            neighbour = route.copy()
            neighbour[i:j + 1] = reversed(neighbour[i:j + 1])
            yield neighbour


def all_neighbors(route: Sequence[int]):
    """Generate both swap and 2-opt neighbours."""
    yield from swap_neighbors(route)
    yield from two_opt_neighbors(route)


def solve(
    time_matrix: np.ndarray,
    start: int = 0,
    initial_route: list[int] | None = None,
) -> list[int]:
    """
    Hill Climbing route optimizer.

    Parameters
    ----------
    time_matrix : np.ndarray
        Precomputed travel-time matrix in minutes.

    start : int
        Starting/depot node. Default is 0.

    initial_route : list[int] | None
        Optional starting route.
        If None, the route starts as [start, 1, 2, 3, ...].

    Returns
    -------
    list[int]
        Locally optimal route found by Hill Climbing.

    Notes
    -----
    The ETA/ML model is NOT called inside this function.
    The time matrix must already be precomputed.
    """

    matrix = np.asarray(time_matrix, dtype=float)

    # Validate matrix
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("time_matrix must be a square matrix.")

    n = matrix.shape[0]

    # Validate start
    if start < 0 or start >= n:
        raise ValueError("Invalid start node.")

    # Create initial route
    if initial_route is None:
        route = [start] + [node for node in range(n) if node != start]
    else:
        route = list(initial_route)

        # Validate initial route
        if len(route) != n:
            raise ValueError("initial_route must contain every node exactly once.")

        if route[0] != start:
            raise ValueError("initial_route must start with the start node.")

        if sorted(route) != list(range(n)):
            raise ValueError(
                "initial_route must contain every node exactly once."
            )

    current_cost = route_cost(matrix, route)

    while True:
        best_route = route
        best_cost = current_cost

        # Check all swap and 2-opt neighbours
        for neighbor in all_neighbors(route):
            neighbor_cost = route_cost(matrix, neighbor)

            # Only accept an improving move
            if neighbor_cost < best_cost:
                best_route = neighbor
                best_cost = neighbor_cost

        # No improvement → local optimum reached
        if best_cost >= current_cost:
            break

        route = best_route
        current_cost = best_cost

    return route