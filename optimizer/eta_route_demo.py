from __future__ import annotations

from optimizer.base import route_cost
from api.matrix_builder import build_eta_matrix
from optimizer.hill_climbing import solve


if __name__ == "__main__":

    # Four example stops in NYC.
    stops = [
        (40.7580, -73.9855),  # Stop 0
        (40.6413, -73.7781),  # Stop 1
        (40.7306, -73.9352),  # Stop 2
        (40.7484, -73.9857),  # Stop 3
    ]

    departure = "2016-06-15 08:30:00"

    # Step 1: Build ETA matrix using the trained ML model.
    eta_matrix = build_eta_matrix(
        stops,
        departure,
    )

    print("\nETA MATRIX")
    print("=" * 50)
    print(eta_matrix)

    # Step 2: Run Hill Climbing.
    route = solve(
        eta_matrix,
        start=0,
    )

    # Step 3: Calculate final route cost.
    total_cost = route_cost(
        eta_matrix,
        route,
    )

    print("\nOPTIMIZED ROUTE")
    print("=" * 50)
    print("Route:", route)
    print(f"Total predicted travel time: {total_cost:.2f} minutes")

    # Step 4: Show each leg.
    print("\nROUTE LEGS")
    print("=" * 50)

    for i in range(len(route) - 1):
        source = route[i]
        destination = route[i + 1]
        travel_time = eta_matrix[source, destination]

        print(
            f"Stop {source} -> Stop {destination}: "
            f"{travel_time:.2f} minutes"
        )