"""
Solver registry.

Provides a common interface for route optimization algorithms.
Future algorithms such as Tabu Search and Beam Search can be
registered without changing the rest of the application.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from optimizer.hill_climbing import solve as hill_climbing_solve
from optimizer.held_karp import solve as held_karp_solve


Solver = Callable[[np.ndarray, int], list[int]]


SOLVERS: dict[str, Solver] = {
    "hill_climbing": hill_climbing_solve,
    "held_karp": held_karp_solve,
    "exact": held_karp_solve,
}


def get_solver(name: str) -> Solver:
    """
    Return the solver associated with the given name.
    """

    name = name.lower().strip()

    if name not in SOLVERS:
        available = ", ".join(sorted(SOLVERS))
        raise ValueError(
            f"Unknown solver '{name}'. Available solvers: {available}"
        )

    return SOLVERS[name]


def available_solvers() -> list[str]:
    """Return the names of all registered solvers."""
    return sorted(SOLVERS)