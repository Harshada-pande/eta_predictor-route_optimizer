"""
Exact route optimizer using Held-Karp dynamic programming.

Finds the globally optimal open route for the supplied time matrix.

The route:
- starts at `start`
- visits every other stop exactly once
- does NOT return to the depot
- can work with asymmetric travel-time matrices

Time complexity:
    O(2^m * m^2)

Memory complexity:
    O(2^m * m)

where m = number of stops excluding the starting point.
"""

from __future__ import annotations

import numpy as np


MAX_EXACT_STOPS = 15


def solve(time_matrix: np.ndarray, start: int = 0) -> list[int]:
    """
    Find the minimum-cost route visiting every stop exactly once.

    Parameters
    ----------
    time_matrix:
        Square matrix where matrix[i][j] is the travel time
        from stop i to stop j.

    start:
        Starting node. By default, node 0 is the depot.

    Returns
    -------
    list[int]
        Globally optimal route.
    """

    matrix = np.asarray(time_matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("time_matrix must be a square matrix.")

    n = matrix.shape[0]

    if n == 0:
        raise ValueError("time_matrix cannot be empty.")

    if not 0 <= start < n:
        raise ValueError("start must be a valid node index.")

    if n == 1:
        return [start]

    stops = [i for i in range(n) if i != start]
    m = len(stops)

    if m > MAX_EXACT_STOPS:
        raise ValueError(
            f"Exact solver supports at most {MAX_EXACT_STOPS} "
            f"non-start stops (got {m})."
        )

    # Travel time from the starting node to each stop.
    from_start = matrix[start, stops]

    # Travel times between non-start stops.
    between = matrix[np.ix_(stops, stops)]

    # dp[mask][j] =
    # minimum cost to start at `start`,
    # visit exactly the stops represented by mask,
    # and finish at stops[j].
    size = 1 << m

    dp = np.full((size, m), np.inf)

    # Parent table used to reconstruct the optimal route.
    parent = np.full((size, m), -1, dtype=np.int16)

    # Base cases:
    # start -> each individual stop
    for j in range(m):
        dp[1 << j, j] = from_start[j]

    # Build solutions for increasingly large subsets.
    for mask in range(1, size):

        for j in range(m):

            bit = 1 << j

            # Stop j must be part of this subset.
            if not mask & bit:
                continue

            # Single-stop route already initialized above.
            if mask == bit:
                continue

            previous_mask = mask ^ bit

            previous_costs = dp[previous_mask]

            candidates = previous_costs + between[:, j]

            previous_j = int(np.argmin(candidates))

            dp[mask, j] = candidates[previous_j]

            parent[mask, j] = previous_j

    # All stops have been visited.
    full_mask = size - 1

    # Choose the cheapest possible final stop.
    final_j = int(np.argmin(dp[full_mask]))

    # Reconstruct route backwards.
    route = []

    mask = full_mask
    j = final_j

    while j != -1:

        route.append(stops[j])

        previous_j = int(parent[mask, j])

        mask ^= 1 << j

        j = previous_j

    route.reverse()

    return [start] + route