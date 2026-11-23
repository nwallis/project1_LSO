from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from provided_functions import (
    cheapest_grid_routes,
    load_instance,
    plot_grid_routes,
)


def primal_heuristic(
    width: int,
    height: int,
    contact_pairs: np.ndarray,
    candidate_routes: list[tuple[int, list[tuple[int, int, int]], float]],
    u: np.ndarray,
    ergodic_usage: np.ndarray,
    all_terminals: set[tuple[int, int, int]],
    rng: np.random.Generator,
) -> tuple[int, list[list[tuple[int, int, int]] | None]]:
    """
    Task 2: Two-stage primal feasibility heuristic to recover vertex-disjoint routes.

    Stage 1: Greedily selects non-overlapping paths from the current Lagrangean
             subproblem solution while protecting all contact-pair terminals.
    Stage 2: Re-routes remaining unconnected pairs sequentially using shortest-path
             queries where occupied nodes and foreign terminals are blocked (inf cost).
    """
    k = len(contact_pairs)
    num_nodes = 2 * width * height
    best_count = -1
    best_routes: list[list[tuple[int, int, int]] | None] = [None] * k

    # Precompute scaled congestion-aware weights for Stage 2 repair.
    # Scaling ensures every valid simple path has total cost <= 0.99 < 1.0,
    # preventing cheapest_grid_routes from aborting valid paths at its 1.0 cutoff.
    max_raw_weight = 1.0 + float(np.max(u)) + 0.5 * float(np.max(ergodic_usage))
    scale = 0.99 / (num_nodes * max_raw_weight)
    base_repair_costs = (1.0 + u + 0.5 * ergodic_usage) * scale

    # Evaluate two complementary greedy priority rules to avoid deterministic traps:
    #   Order A: Shortest path length and lowest historical (ergodic) congestion first
    #   Order B: Dual cost with small random perturbation for tie-breaking diversity
    orderings = [
        sorted(
            candidate_routes,
            key=lambda item: (
                len(item[1]),
                sum(ergodic_usage[x, y, z] for x, y, z in item[1]),
            ),
        ),
        sorted(
            candidate_routes,
            key=lambda item: (item[2] + 0.05 * rng.random(), len(item[1])),
        ),
    ]

    for ordered_candidates in orderings:
        used_nodes: set[tuple[int, int, int]] = set()
        selected_routes: list[list[tuple[int, int, int]] | None] = [None] * k
        connected_count = 0

        # --- Stage 1: Conflict-free greedy selection from subproblem paths ---
        for l, route, _ in ordered_candidates:
            s_l = tuple(int(v) for v in contact_pairs[l, 0])
            t_l = tuple(int(v) for v in contact_pairs[l, 1])
            foreign_terminals = all_terminals - {s_l, t_l}

            if any(node in used_nodes or node in foreign_terminals for node in route):
                continue

            selected_routes[l] = route
            used_nodes.update(route)
            connected_count += 1

        # --- Stage 2: Congestion-aware repair for unconnected pairs ---
        unconnected = [l for l in range(k) if selected_routes[l] is None]
        # Route pairs with larger 2D Manhattan distance first
        unconnected.sort(
            key=lambda l: (
                abs(int(contact_pairs[l, 0, 0]) - int(contact_pairs[l, 1, 0]))
                + abs(int(contact_pairs[l, 0, 1]) - int(contact_pairs[l, 1, 1]))
            ),
            reverse=True,
        )

        if unconnected:
            repair_costs = base_repair_costs.copy()
            for x, y, z in used_nodes:
                repair_costs[x, y, z] = np.inf
            for x, y, z in all_terminals:
                repair_costs[x, y, z] = np.inf

            for l in unconnected:
                s_l = tuple(int(v) for v in contact_pairs[l, 0])
                t_l = tuple(int(v) for v in contact_pairs[l, 1])

                # Temporarily allow pair l's own start and end terminals
                repair_costs[s_l] = base_repair_costs[s_l]
                repair_costs[t_l] = base_repair_costs[t_l]

                repaired = cheapest_grid_routes(
                    width, height, repair_costs, contact_pairs[l : l + 1]
                )[0]

                if repaired is not None:
                    selected_routes[l] = repaired
                    used_nodes.update(repaired)
                    connected_count += 1
                    for x, y, z in repaired:
                        repair_costs[x, y, z] = np.inf
                else:
                    # Re-lock terminals if pair l could not be routed
                    repair_costs[s_l] = np.inf
                    repair_costs[t_l] = np.inf

        if connected_count > best_count:
            best_count = connected_count
            best_routes = selected_routes
            if best_count == k:
                break

    return best_count, best_routes


def solve_vlsi_lagrangean(
    width: int,
    height: int,
    contact_pairs: np.ndarray,
    max_iter: int = 1000,
    theta_init: float = 2.0,
    use_deflection: bool = True,
    seed: int = 42,
) -> dict:
    """
    Solves the VLSI routing problem via Lagrangean relaxation and projected/deflected
    subgradient optimization (Task 1) coupled with primal recovery (Task 2).
    """
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    k = len(contact_pairs)

    # Collect all terminal pads (starts and ends) across all contact pairs
    all_terminals: set[tuple[int, int, int]] = set()
    for l in range(k):
        all_terminals.add(tuple(int(v) for v in contact_pairs[l, 0]))
        all_terminals.add(tuple(int(v) for v in contact_pairs[l, 1]))

    # Step 1: Initialize dual variables u^0 >= 0 of shape (width, height, 2) and theta_0
    u = np.zeros((width, height, 2), dtype=np.float64)
    theta = theta_init

    # Ergodic node visitation frequency and subgradient deflection memory
    ergodic_usage = np.zeros((width, height, 2), dtype=np.float64)
    prev_direction = np.zeros((width, height, 2), dtype=np.float64)

    best_dual = float("inf")
    best_primal = 0
    best_primal_routes: list[list[tuple[int, int, int]] | None] = [None] * k

    dual_history: list[float] = []
    primal_history: list[int] = []
    stagnation_count = 0

    for t in range(max_iter):
        # Step 2: Solve the k independent Lagrangean subproblems for u = u^t
        raw_routes = cheapest_grid_routes(width, height, u, contact_pairs)

        subproblem_sum = 0.0
        node_usage = np.zeros((width, height, 2), dtype=np.int64)
        candidate_routes: list[tuple[int, list[tuple[int, int, int]], float]] = []

        for l, route in enumerate(raw_routes):
            if route is not None:
                route_cost = float(sum(u[x, y, z] for x, y, z in route))
                candidate_routes.append((l, route, route_cost))

                # Activate pair l iff its exact unperturbed route cost is < 1
                if route_cost < 1.0:
                    subproblem_sum += 1.0 - route_cost
                    for x, y, z in route:
                        node_usage[x, y, z] += 1

        # Update ergodic average of node usage across iterations 0..t
        inv_t = 1.0 / (t + 1)
        ergodic_usage = (1.0 - inv_t) * ergodic_usage + inv_t * node_usage

        # Evaluate dual upper bound h(u^t)
        h_u = float(np.sum(u)) + subproblem_sum
        if h_u < best_dual - 1e-9:
            best_dual = h_u
            stagnation_count = 0
        else:
            stagnation_count += 1

        # Task 2: Recover a conflict-free primal solution (pessimistic lower bound)
        primal_val, primal_routes = primal_heuristic(
            width=width,
            height=height,
            contact_pairs=contact_pairs,
            candidate_routes=candidate_routes,
            u=u,
            ergodic_usage=ergodic_usage,
            all_terminals=all_terminals,
            rng=rng,
        )
        if primal_val > best_primal:
            best_primal = primal_val
            best_primal_routes = primal_routes

        dual_history.append(h_u)
        primal_history.append(best_primal)

        # Termination check 1: All k pairs routed without collision -> global optimum
        if best_primal == k:
            break

        # Termination check 2: Integer duality gap closed (floor(best_dual) <= best_primal)
        if math.floor(best_dual + 1e-7) <= best_primal:
            break

        # Step 3: Compute conditional (projected) subgradient gamma^t
        gamma = 1.0 - node_usage
        gamma = np.where((u <= 1e-12) & (gamma > 0.0), 0.0, gamma)

        # Optional Camerini-Fratta-Maffioli deflection to reduce zig-zagging
        if use_deflection and t > 0:
            dot_gd = float(np.sum(gamma * prev_direction))
            norm_prev_sq = float(np.sum(prev_direction ** 2))
            if dot_gd < 0.0 and norm_prev_sq > 1e-12:
                beta = min(0.8, -1.5 * dot_gd / norm_prev_sq)
            else:
                beta = 0.0
            direction = gamma + beta * prev_direction
            direction = np.where((u <= 1e-12) & (direction > 0.0), 0.0, direction)
        else:
            direction = gamma

        prev_direction = direction

        norm_sq = float(np.sum(direction ** 2))
        if norm_sq <= 1e-12:
            # Zero projected subgradient implies dual optimality
            break

        # Polyak step length using best_primal as the lower bound
        gap = max(0.0, h_u - best_primal)
        alpha = theta * gap / norm_sq

        # Step 4: Update dual variables u^{t+1} = max(0, u^t - alpha_t * direction^t)
        u = np.maximum(0.0, u - alpha * direction)

        # Step 5: Update step-size parameter theta_t
        if stagnation_count >= 15:
            theta *= 0.85
            stagnation_count = 0
        elif (t + 1) % 10 == 0:
            theta *= 0.95

    elapsed_time = time.perf_counter() - start_time
    return {
        "best_dual": best_dual,
        "best_primal": best_primal,
        "k": k,
        "best_routes": best_primal_routes,
        "dual_history": dual_history,
        "primal_history": primal_history,
        "elapsed_time": elapsed_time,
        "iterations": len(dual_history),
    }


def conclude_feasibility(best_dual: float, best_primal: int, k: int) -> str:
    """Returns a rigorous mathematical conclusion on instance solvability."""
    dual_floor = math.floor(best_dual + 1e-7)
    if best_primal == k:
        return f"FEASIBLE — All {k} pairs connected without collision."
    if dual_floor < k:
        return f"PROVEN INFEASIBLE — Dual upper bound floor({best_dual:.4f}) = {dual_floor} < {k}."
    return (
        f"INCONCLUSIVE — Best primal = {best_primal}/{k}, "
        f"dual upper bound = {best_dual:.4f}."
    )


def evaluate_instance(
    instance_path: str | Path,
    num_runs: int = 5,
    max_iter: int = 1000,
    output_dir: str | Path | None = "results",
    show_plots: bool = False,
) -> dict:
    """
    Runs multiple trials on a single instance to compute average runtime,
    prints Task 3 summary statistics, and generates convergence + routing plots.
    """
    instance_path = Path(instance_path)
    width, height, contact_pairs = load_instance(instance_path)

    runs = [
        solve_vlsi_lagrangean(
            width=width,
            height=height,
            contact_pairs=contact_pairs,
            max_iter=max_iter,
            seed=42 + run_idx,
        )
        for run_idx in range(num_runs)
    ]

    avg_time = float(np.mean([r["elapsed_time"] for r in runs]))
    best_run = max(runs, key=lambda r: (r["best_primal"], -r["best_dual"]))
    conclusion = conclude_feasibility(
        best_run["best_dual"], best_run["best_primal"], best_run["k"]
    )

    print(f"Instance : {instance_path.name} ({width}x{height} grid, k={best_run['k']})")
    print(f"  Optimal / Best Dual   : {best_run['best_dual']:.4f}")
    print(f"  Best Primal Objective : {best_run['best_primal']} / {best_run['k']}")
    print(f"  Iterations (best run) : {best_run['iterations']}")
    print(f"  Avg Runtime ({num_runs} runs)  : {avg_time:.4f} s")
    print(f"  Conclusion            : {conclusion}\n")

    # Generate Task 3 plots: (Left) Dual & Primal vs. Iteration; (Right) Best Primal Routing
    fig, (ax_conv, ax_grid) = plt.subplots(1, 2, figsize=(13, 5))

    iters = np.arange(1, len(best_run["dual_history"]) + 1)
    ax_conv.plot(
        iters,
        best_run["dual_history"],
        label="Lagrangean Dual $h(u^t)$ (Upper Bound)",
        linewidth=1.5,
    )
    ax_conv.step(
        iters,
        best_run["primal_history"],
        where="post",
        label="Best Primal Solution (Lower Bound)",
        linewidth=2.0,
    )
    ax_conv.axhline(
        best_run["k"],
        color="gray",
        linestyle="--",
        label=f"Target k = {best_run['k']}",
    )
    ax_conv.set_xlabel("Iteration $t$")
    ax_conv.set_ylabel("Objective Value")
    ax_conv.set_title(f"Convergence — {instance_path.name}")
    ax_conv.legend()
    ax_conv.grid(True, alpha=0.3)

    plot_grid_routes(width, height, best_run["best_routes"], axes=ax_grid)
    ax_grid.set_title(
        f"Best Primal Routing ({best_run['best_primal']}/{best_run['k']} connected)"
    )

    plt.tight_layout()
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path / f"{instance_path.stem}_results.png", dpi=200)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return best_run


if __name__ == "__main__":
    instances_dir = Path("instances")
    json_files = sorted(instances_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError("No .json files found in the 'instances/' directory.")

    print(f"Evaluating {len(json_files)} VLSI routing instances...\n" + "=" * 65)
    for json_file in json_files:
        evaluate_instance(
            instance_path=json_file,
            num_runs=5,
            max_iter=1000,
            output_dir="results",
            show_plots=False,
        )