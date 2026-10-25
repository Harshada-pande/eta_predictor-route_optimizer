from __future__ import annotations

import itertools

from optimizer.base import route_cost
# pyrefly: ignore[missing-import]
from optimizer.eta_matrix import build_eta_matrix
from optimizer.hill_climbing import solve


if __name__ == "__main__":

    stops = [
        (40.7580, -73.9855),  # Stop 0
        (40.6413, -73.7781),  # Stop 1
        (40.7306, -73.9352),  # Stop 2
        (40.7484, -73.9857),  # Stop 3
    ]

    departure = "2016-06-15 08:30:00"

    # Build real ETA matrix using XGBoost.
    matrix = build_eta_matrix(
        stops,
        departure,
    )

    # Hill Climbing result.
    hill_route = solve(
        matrix,
        start=0,
    )

    hill_cost = route_cost(
        matrix,
        hill_route,
    )

    # Brute-force all routes with stop 0 fixed.
    remaining = [1, 2, 3]

    best_route = None
    best_cost = float("inf")

    print("\nALL POSSIBLE ROUTES")
    print("=" * 60)

    for permutation in itertools.permutations(remaining):

        route = [0] + list(permutation)

        cost = route_cost(
            matrix,
            route,
        )

        print(
            f"{route} -> {cost:.2f} minutes"
        )

        if cost < best_cost:
            best_cost = cost
            best_route = route

    print("\nCOMPARISON")
    print("=" * 60)

    print(
        f"Hill Climbing : {hill_route} "
        f"-> {hill_cost:.2f} min"
    )

    print(
        f"Brute Force   : {best_route} "
        f"-> {best_cost:.2f} min"
    )

    if abs(hill_cost - best_cost) < 1e-6:
        print("\nRESULT: Hill Climbing found the optimal route.")
    else:
        print("\nRESULT: Hill Climbing did not find the global optimum.")