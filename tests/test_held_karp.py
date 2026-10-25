import itertools

import numpy as np
import pytest

from optimizer.held_karp import MAX_EXACT_STOPS, solve


def brute_force_solve(time_matrix, start=0):
    """
    Simple brute-force reference implementation.

    Used only for testing Held-Karp.
    """
    matrix = np.asarray(time_matrix, dtype=float)
    n = matrix.shape[0]

    stops = [i for i in range(n) if i != start]

    best_route = None
    best_cost = float("inf")

    for permutation in itertools.permutations(stops):
        route = [start] + list(permutation)

        cost = sum(
            matrix[route[i], route[i + 1]]
            for i in range(len(route) - 1)
        )

        if cost < best_cost:
            best_cost = cost
            best_route = route

    return best_route, best_cost


def route_cost(matrix, route):
    return sum(
        matrix[route[i], route[i + 1]]
        for i in range(len(route) - 1)
    )


def test_held_karp_finds_known_optimal_route():
    matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ], dtype=float)

    route = solve(matrix)

    expected_route = [0, 1, 3, 2]

    assert route == expected_route
    assert route_cost(matrix, route) == pytest.approx(65)


def test_held_karp_matches_brute_force_on_asymmetric_matrix():
    matrix = np.array([
        [0, 8, 3, 9],
        [4, 0, 7, 2],
        [6, 5, 0, 4],
        [3, 6, 8, 0],
    ], dtype=float)

    held_karp_route = solve(matrix)

    brute_route, brute_cost = brute_force_solve(matrix)

    assert route_cost(matrix, held_karp_route) == pytest.approx(brute_cost)
    assert set(held_karp_route) == set(brute_route)
    assert held_karp_route[0] == 0


def test_held_karp_matches_brute_force_on_symmetric_matrix():
    matrix = np.array([
        [0, 4, 7, 3, 8],
        [4, 0, 6, 5, 2],
        [7, 6, 0, 4, 3],
        [3, 5, 4, 0, 6],
        [8, 2, 3, 6, 0],
    ], dtype=float)

    held_karp_route = solve(matrix)

    brute_route, brute_cost = brute_force_solve(matrix)

    assert route_cost(matrix, held_karp_route) == pytest.approx(brute_cost)
    assert held_karp_route[0] == 0


def test_held_karp_supports_different_start_node():
    matrix = np.array([
        [0, 8, 3, 9],
        [4, 0, 7, 2],
        [6, 5, 0, 4],
        [3, 6, 8, 0],
    ], dtype=float)

    start = 2

    held_karp_route = solve(matrix, start=start)

    brute_route, brute_cost = brute_force_solve(
        matrix,
        start=start,
    )

    assert held_karp_route[0] == start
    assert route_cost(matrix, held_karp_route) == pytest.approx(
        brute_cost
    )
    assert set(held_karp_route) == set(brute_route)


def test_held_karp_visits_every_node_exactly_once():
    matrix = np.array([
        [0, 5, 8, 6],
        [5, 0, 4, 7],
        [8, 4, 0, 3],
        [6, 7, 3, 0],
    ], dtype=float)

    route = solve(matrix)

    assert len(route) == len(matrix)
    assert len(set(route)) == len(matrix)
    assert set(route) == set(range(len(matrix)))
    assert route[0] == 0


def test_single_node_returns_start():
    matrix = np.array([[0]], dtype=float)

    assert solve(matrix) == [0]


def test_empty_matrix_is_rejected():
    matrix = np.empty((0, 0))

    with pytest.raises(
        ValueError,
        match="time_matrix cannot be empty",
    ):
        solve(matrix)


def test_non_square_matrix_is_rejected():
    matrix = np.zeros((3, 4))

    with pytest.raises(
        ValueError,
        match="time_matrix must be a square matrix",
    ):
        solve(matrix)


def test_invalid_start_is_rejected():
    matrix = np.zeros((4, 4))

    with pytest.raises(
        ValueError,
        match="start must be a valid node index",
    ):
        solve(matrix, start=4)


def test_exact_solver_stop_limit():
    # MAX_EXACT_STOPS refers to non-start nodes.
    # Therefore MAX_EXACT_STOPS + 1 total nodes is allowed.
    allowed_matrix = np.zeros(
        (MAX_EXACT_STOPS + 1, MAX_EXACT_STOPS + 1)
    )

    route = solve(allowed_matrix)

    assert len(route) == MAX_EXACT_STOPS + 1

    # One additional non-start node should be rejected.
    too_large_matrix = np.zeros(
        (MAX_EXACT_STOPS + 2, MAX_EXACT_STOPS + 2)
    )

    with pytest.raises(
        ValueError,
        match="Exact solver supports at most",
    ):
        solve(too_large_matrix)


@pytest.mark.parametrize("seed", range(10))
def test_held_karp_matches_brute_force_on_random_matrices(seed):
    """
    Compare Held-Karp with brute force on small random
    asymmetric matrices.
    """
    rng = np.random.default_rng(seed)

    n = 6

    matrix = rng.integers(
        1,
        30,
        size=(n, n),
    ).astype(float)

    np.fill_diagonal(matrix, 0)

    held_karp_route = solve(matrix)

    brute_route, brute_cost = brute_force_solve(matrix)

    assert route_cost(matrix, held_karp_route) == pytest.approx(
        brute_cost
    )
    assert held_karp_route[0] == 0
    assert set(held_karp_route) == set(brute_route)