from __future__ import annotations

import random
import numpy as np
import matplotlib.pyplot as plt

from optimizer.base import route_cost
from optimizer.hill_climbing import solve


def generate_time_matrix(
    n: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a reproducible asymmetric travel-time matrix.

    The matrix represents precomputed travel times between stops.
    """
    rng = np.random.default_rng(seed)

    matrix = rng.uniform(5, 60, size=(n, n))

    # A stop to itself takes zero time.
    np.fill_diagonal(matrix, 0)

    return matrix


def run_local_optima_demo(
    num_starts: int = 50,
    n_stops: int = 10,
    seed: int = 42,
):
    """
    Run Hill Climbing from multiple random starting routes.

    Returns:
        list of dictionaries containing:
        start route, final route and final cost.
    """

    matrix = generate_time_matrix(n=n_stops, seed=seed)

    rng = random.Random(seed)

    results = []

    nodes = list(range(1, n_stops))

    for i in range(num_starts):
        shuffled_nodes = nodes.copy()
        rng.shuffle(shuffled_nodes)

        initial_route = [0] + shuffled_nodes

        final_route = solve(
            matrix,
            start=0,
            initial_route=initial_route,
        )

        final_cost = route_cost(matrix, final_route)

        results.append(
            {
                "run": i + 1,
                "initial_route": initial_route,
                "final_route": final_route,
                "final_cost": final_cost,
            }
        )

    return results


if __name__ == "__main__":

    results = run_local_optima_demo(
        num_starts=50,
        n_stops=10,
        seed=42,
    )

    print("\nHILL CLIMBING — LOCAL OPTIMA DEMONSTRATION")
    print("=" * 60)

    for result in results:
        print(
            f"Run {result['run']:2d}: "
            f"Final cost = {result['final_cost']:.2f} min"
        )

    costs = [result["final_cost"] for result in results]

    print("\n" + "=" * 60)
    print(f"Number of starts : {len(results)}")
    print(f"Best cost        : {min(costs):.2f} min")
    print(f"Worst cost       : {max(costs):.2f} min")
    print(f"Average cost     : {np.mean(costs):.2f} min")
    print(f"Unique final costs: {len(set(round(c, 6) for c in costs))}")

    print("\nBest route:")
    best_result = min(results, key=lambda x: x["final_cost"])

    print(best_result["final_route"])

    print("\nWorst route:")
    worst_result = max(results, key=lambda x: x["final_cost"])

    print(worst_result["final_route"])
        # Plot final costs for all random starting routes
    runs = [result["run"] for result in results]
    costs = [result["final_cost"] for result in results]

    best_cost = min(costs)
    worst_cost = max(costs)

    plt.figure(figsize=(10, 5))

    plt.plot(
        runs,
        costs,
        marker="o",
        linewidth=1,
        label="Final Hill Climbing Cost",
    )

    plt.axhline(
        best_cost,
        linestyle="--",
        linewidth=1,
        label=f"Best = {best_cost:.2f} min",
    )

    plt.axhline(
        worst_cost,
        linestyle="--",
        linewidth=1,
        label=f"Worst = {worst_cost:.2f} min",
    )

    plt.xlabel("Random Starting Route")
    plt.ylabel("Final Route Cost (minutes)")
    plt.title("Hill Climbing: Effect of Different Starting Routes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "docs/local_optima_demo.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print("\nPlot saved to: docs/local_optima_demo.png")