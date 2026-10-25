import itertools

import numpy as np

from optimizer.base import route_cost
from optimizer.hill_climbing import solve
from optimizer.local_optima_demo import run_local_optima_demo


def brute_force_best_route(time_matrix, start=0):
    """
    Find the globally optimal route by checking every possible
    ordering of the remaining stops.
    """

    n = len(time_matrix)

    remaining = [
        node for node in range(n)
        if node != start
    ]

    best_route = None
    best_cost = float("inf")

    for permutation in itertools.permutations(remaining):
        route = [start] + list(permutation)
        cost = route_cost(time_matrix, route)

        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route, best_cost


def test_hill_climbing_returns_valid_route():
    """Hill climbing should return every stop exactly once."""

    matrix = np.array([
        [0, 4, 2, 7, 3],
        [4, 0, 5, 1, 6],
        [2, 5, 0, 6, 4],
        [7, 1, 6, 0, 2],
        [3, 6, 4, 2, 0],
    ], dtype=float)

    route = solve(matrix, start=0)

    assert route[0] == 0
    assert sorted(route) == list(range(5))


def test_hill_climbing_matches_brute_force_on_small_case():
    """
    For this small case, compare hill climbing with exhaustive
    brute-force search.
    """

    matrix = np.array([
        [0, 4, 2, 7, 3],
        [4, 0, 5, 1, 6],
        [2, 5, 0, 6, 4],
        [7, 1, 6, 0, 2],
        [3, 6, 4, 2, 0],
    ], dtype=float)

    hill_route = solve(matrix, start=0)
    hill_cost = route_cost(matrix, hill_route)

    brute_route, brute_cost = brute_force_best_route(
        matrix,
        start=0,
    )

    assert hill_cost == brute_cost

def test_local_optima_demo_produces_different_results():
    results = run_local_optima_demo(
        num_starts=50,
        n_stops=10,
        seed=42,
    )

    # We should have exactly 50 experiments.
    assert len(results) == 50

    # Every final route should contain all 10 stops.
    for result in results:
        route = result["final_route"]

        assert route[0] == 0
        assert sorted(route) == list(range(10))
        assert result["final_cost"] > 0

    # Different starting routes should produce different
    # local-optimum costs.
    costs = [result["final_cost"] for result in results]

    assert min(costs) < max(costs)