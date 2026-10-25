"""
Benchmark Hill Climbing against Held-Karp on real model-generated ETA matrices.

Experiment:
- 100 random Pune route cases
- 10 delivery stops per case
- fixed random seed for reproducibility
- real XGBoost ETA predictions
- random Hill Climbing initial route
- Held-Karp as the exact optimum
- reports suboptimality, optimality gap, runtime, and clipped-leg share
"""

from __future__ import annotations

import random
import time
from datetime import datetime

import numpy as np

from api.matrix_builder import build_eta_matrix, haversine_km
from optimizer.hill_climbing import solve as hill_climbing
from optimizer.held_karp import solve as held_karp


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

SEED = 42
CASES = 100
DELIVERY_STOPS = 10

# Pune-area coordinate bounds used only to generate benchmark cases.
PUNE_LAT_MIN = 18.45
PUNE_LAT_MAX = 18.65
PUNE_LON_MIN = 73.75
PUNE_LON_MAX = 73.95

# Model training distance range.
TRAINING_DISTANCE_MIN = 1.47
TRAINING_DISTANCE_MAX = 20.97


# ---------------------------------------------------------------------------
# Random route generation
# ---------------------------------------------------------------------------

def generate_coordinates(rng: random.Random) -> list[tuple[float, float]]:
    """
    Generate one Pune depot + delivery-stop coordinate set.
    """

    coordinates = [
        (
            rng.uniform(PUNE_LAT_MIN, PUNE_LAT_MAX),
            rng.uniform(PUNE_LON_MIN, PUNE_LON_MAX),
        )
    ]

    for _ in range(DELIVERY_STOPS):
        coordinates.append(
            (
                rng.uniform(PUNE_LAT_MIN, PUNE_LAT_MAX),
                rng.uniform(PUNE_LON_MIN, PUNE_LON_MAX),
            )
        )

    return coordinates


def generate_contexts(rng: random.Random) -> list[dict]:
    """
    Generate one route-level rider context plus stop-level contexts.

    The rider information is shared by every stop.
    Weather, traffic, and multiple deliveries vary by stop.
    """

    rider = {
        "vehicle_type": rng.choice(
            [
                "Motorcycle",
                "Scooter",
                "Bicycle",
                "Electric Scooter",
            ]
        ),
        "vehicle_condition": rng.randint(0, 3),
        "delivery_person_age": rng.randint(20, 39),
        "delivery_person_rating": round(
            rng.uniform(2.5, 5.0),
            1,
        ),
    }

    festival = rng.choice(["No", "Yes", "Unknown"])
    city_type = rng.choice(
        [
            "Metropolitan",
            "Semi-Urban",
            "Urban",
        ]
    )

    contexts = [
        {
            **rider,
            "weather": "Sunny",
            "traffic": "Medium",
            "multiple_deliveries": 0,
            "festival": festival,
            "city_type": city_type,
        }
    ]

    for _ in range(DELIVERY_STOPS):
        contexts.append(
            {
                **rider,
                "weather": rng.choice(
                    [
                        "Sunny",
                        "Cloudy",
                        "Fog",
                        "Sandstorms",
                        "Stormy",
                        "Windy",
                        "Unknown",
                    ]
                ),
                "traffic": rng.choice(
                    [
                        "Unknown",
                        "Low",
                        "Medium",
                        "High",
                        "Jam",
                    ]
                ),
                "multiple_deliveries": rng.randint(0, 3),
                "festival": festival,
                "city_type": city_type,
            }
        )

    return contexts


# ---------------------------------------------------------------------------
# Distance clipping measurement
# ---------------------------------------------------------------------------

def count_clipped_legs(
    coordinates: list[tuple[float, float]],
) -> tuple[int, int]:
    """
    Count directed legs whose raw distance falls outside the
    model's training distance range.

    Each i -> j direction is counted.
    """

    clipped = 0
    total = 0

    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            if i == j:
                continue

            distance = haversine_km(
                coordinates[i][0],
                coordinates[i][1],
                coordinates[j][0],
                coordinates[j][1],
            )

            total += 1

            if (
                distance < TRAINING_DISTANCE_MIN
                or distance > TRAINING_DISTANCE_MAX
            ):
                clipped += 1

    return clipped, total


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(SEED)

    np.random.seed(SEED)

    results = []

    print("=" * 70)
    print("HILL CLIMBING vs HELD-KARP BENCHMARK")
    print("=" * 70)
    print(f"Cases:           {CASES}")
    print(f"Delivery stops:  {DELIVERY_STOPS}")
    print(f"Random seed:     {SEED}")
    print()

    for case_number in range(1, CASES + 1):

        coordinates = generate_coordinates(rng)
        contexts = generate_contexts(rng)

        departure = datetime(
            2026,
            10,
            24,
            rng.randint(8, 20),
            rng.choice([0, 15, 30, 45]),
        )

        # ---------------------------------------------------------------
        # Build real ETA matrix using XGBoost.
        # ---------------------------------------------------------------

        matrix_start = time.perf_counter()

        matrix = build_eta_matrix(
            coordinates=coordinates,
            departure=departure,
            contexts=contexts,
        )

        matrix_build_time = time.perf_counter() - matrix_start

        # ---------------------------------------------------------------
        # Measure distance clipping.
        # ---------------------------------------------------------------

        clipped_legs, total_legs = count_clipped_legs(coordinates)

        # ---------------------------------------------------------------
        # Random starting solution for Hill Climbing.
        # ---------------------------------------------------------------

        initial_route = [0] + rng.sample(
            list(range(1, DELIVERY_STOPS + 1)),
            DELIVERY_STOPS,
        )

        # ---------------------------------------------------------------
        # Hill Climbing.
        # ---------------------------------------------------------------

        hill_start = time.perf_counter()

        hill_route = hill_climbing(
            matrix,
            start=0,
            initial_route=initial_route,
        )

        hill_runtime = time.perf_counter() - hill_start

        hill_cost = sum(
            matrix[hill_route[i], hill_route[i + 1]]
            for i in range(len(hill_route) - 1)
        )

        # ---------------------------------------------------------------
        # Held-Karp exact solution.
        # ---------------------------------------------------------------

        exact_start = time.perf_counter()

        exact_route = held_karp(
            matrix,
            start=0,
        )

        exact_runtime = time.perf_counter() - exact_start

        exact_cost = sum(
            matrix[exact_route[i], exact_route[i + 1]]
            for i in range(len(exact_route) - 1)
        )

        # ---------------------------------------------------------------
        # Compare solutions.
        # ---------------------------------------------------------------

        if exact_cost > 0:
            gap_percent = (
                (hill_cost - exact_cost)
                / exact_cost
                * 100
            )
        else:
            gap_percent = 0.0

        suboptimal = hill_cost > exact_cost + 1e-9

        results.append(
            {
                "case": case_number,
                "hill_cost": hill_cost,
                "exact_cost": exact_cost,
                "gap_percent": gap_percent,
                "suboptimal": suboptimal,
                "hill_runtime": hill_runtime,
                "exact_runtime": exact_runtime,
                "matrix_build_time": matrix_build_time,
                "clipped_legs": clipped_legs,
                "total_legs": total_legs,
            }
        )

        if case_number % 10 == 0:
            print(
                f"Completed {case_number}/{CASES} cases"
            )

    # -------------------------------------------------------------------
    # Aggregate results
    # -------------------------------------------------------------------

    suboptimal_count = sum(
        result["suboptimal"]
        for result in results
    )

    suboptimal_percent = (
        suboptimal_count / CASES * 100
    )

    gaps = [
        result["gap_percent"]
        for result in results
    ]

    mean_gap = float(np.mean(gaps))
    max_gap = float(np.max(gaps))

    hill_times = [
        result["hill_runtime"]
        for result in results
    ]

    exact_times = [
        result["exact_runtime"]
        for result in results
    ]

    matrix_times = [
        result["matrix_build_time"]
        for result in results
    ]

    total_clipped = sum(
        result["clipped_legs"]
        for result in results
    )

    total_legs = sum(
        result["total_legs"]
        for result in results
    )

    clipped_share = (
        total_clipped / total_legs * 100
        if total_legs
        else 0.0
    )

    mean_hill_time_ms = np.mean(hill_times) * 1000
    mean_exact_time_ms = np.mean(exact_times) * 1000
    mean_matrix_time_ms = np.mean(matrix_times) * 1000

    # -------------------------------------------------------------------
    # Write markdown report.
    # -------------------------------------------------------------------

    report_lines = [
        "# Route Optimization Benchmark",
        "",
        "## Experiment setup",
        "",
        f"- Cases: **{CASES}**",
        f"- Delivery stops per case: **{DELIVERY_STOPS}**",
        f"- Total nodes including depot: **{DELIVERY_STOPS + 1}**",
        f"- Random seed: **{SEED}**",
        "- ETA matrix: **real XGBoost model predictions**",
        "- Hill Climbing: **randomized initial route**",
        "- Exact solver: **Held-Karp dynamic programming**",
        "- Route type: **open route; no return to depot**",
        "",
        "## Results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Cases | {CASES} |",
        f"| Stops per case | {DELIVERY_STOPS} |",
        f"| Hill Climbing suboptimal cases | {suboptimal_count} / {CASES} |",
        f"| Hill Climbing suboptimality | {suboptimal_percent:.2f}% |",
        f"| Mean optimality gap | {mean_gap:.2f}% |",
        f"| Maximum optimality gap | {max_gap:.2f}% |",
        f"| Mean Hill Climbing runtime | {mean_hill_time_ms:.3f} ms |",
        f"| Mean Held-Karp runtime | {mean_exact_time_ms:.3f} ms |",
        f"| Mean ETA matrix build time | {mean_matrix_time_ms:.3f} ms |",
        f"| Clipped directed legs | {total_clipped} / {total_legs} |",
        f"| Clipped-leg share | {clipped_share:.2f}% |",
        "",
        "## Interpretation",
        "",
        "Held-Karp is treated as the exact reference solution for these "
        "benchmark cases.",
        "",
        "A Hill Climbing case is considered suboptimal when its route cost "
        "is greater than the Held-Karp route cost by more than 1e-9.",
        "",
        "The optimality gap is calculated as:",
        "",
        "`(Hill Climbing cost - Held-Karp cost) / Held-Karp cost × 100`",
        "",
        "The clipped-leg share measures how often generated geographic "
        "distances fall outside the model's training distance range of "
        f"{TRAINING_DISTANCE_MIN:.2f}–{TRAINING_DISTANCE_MAX:.2f} km. "
        "These cases are important because distance clipping can make "
        "different geographic legs receive similar model inputs.",
        "",
        "## Per-case results",
        "",
        "| Case | Hill Cost | Exact Cost | Gap % | Suboptimal | HC ms | HK ms | Clipped | Total Legs |",
        "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]

    for result in results:
        report_lines.append(
            f"| {result['case']} "
            f"| {result['hill_cost']:.3f} "
            f"| {result['exact_cost']:.3f} "
            f"| {result['gap_percent']:.3f} "
            f"| {'Yes' if result['suboptimal'] else 'No'} "
            f"| {result['hill_runtime'] * 1000:.3f} "
            f"| {result['exact_runtime'] * 1000:.3f} "
            f"| {result['clipped_legs']} "
            f"| {result['total_legs']} |"
        )

    report_path = "docs/benchmark_results.md"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(report_lines)
            + "\n"
        )

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Suboptimal cases:  {suboptimal_count}/{CASES}")
    print(f"Suboptimality:     {suboptimal_percent:.2f}%")
    print(f"Mean gap:          {mean_gap:.2f}%")
    print(f"Maximum gap:       {max_gap:.2f}%")
    print(f"Mean HC runtime:   {mean_hill_time_ms:.3f} ms")
    print(f"Mean HK runtime:   {mean_exact_time_ms:.3f} ms")
    print(f"Clipped-leg share: {clipped_share:.2f}%")
    print()
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
