#!/usr/bin/env python3
"""Deterministic CPU evaluation for DFlash capture observability solvers.

The output is synthetic.  It checks algorithm behavior on XOR synergy,
conditionally independent sensors, deterministic duplicates, a depth regime
change, and an informative capture whose synchronization latency is uneconomic.
It does not claim anything about the production 5/14/24/33/42 capture set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from dflash_capture_observability import (
    CaptureProblem,
    DinkelbachResult,
    SelectionResult,
    adaptive_dinkelbach_select,
    adaptive_context_values,
    bundle_density_greedy,
    bundle_submodularity_ratio,
    canonical_subset,
    dinkelbach_select,
    exact_select,
    fano_block_bits,
    fixed_current_result,
    greedy_select,
    kill_criterion_net_upper_bound,
    make_duplicate_network,
    make_l45_capture_problem,
    make_naive_bayes_network,
    make_small_problem_from_network,
    make_xor_synergy_network,
    minimum_submodularity_gap,
    prefix_acceptance_fano_bits,
)


def _selection_from_dinkelbach(result: DinkelbachResult, method: str) -> SelectionResult:
    return SelectionResult(
        method=method,
        selected=result.selected,
        objective_value=result.throughput_tokens_per_ms,
        evaluations=result.inner_evaluations,
        feasible=True,
    )


def _metric_row(
    problem: CaptureProblem,
    result: SelectionResult,
    *,
    optimized_objective: str,
    optimum_value: float | None = None,
) -> dict[str, Any]:
    selected = result.selected
    information = problem.information(selected)
    accepted = problem.accepted_tokens(selected)
    round_time = problem.round_time_ms(selected)
    byte_cost, latency, effective = problem.selected_costs(selected)
    objective_value = information if optimized_objective == "information" else accepted / round_time
    return {
        "scenario": problem.name,
        "n_candidates": len(problem.layers),
        "max_captures": problem.constraints.max_items,
        "optimized_objective": optimized_objective,
        "method": result.method,
        "selected_layers": " ".join(str(x) for x in selected),
        "selected_count": len(selected),
        "feasible": int(problem.feasible(selected)),
        "information_bits": information,
        "bayes_accepted_tokens": accepted,
        "round_time_ms": round_time,
        "throughput_tokens_per_second": 1000.0 * accepted / round_time,
        "capture_bytes": byte_cost,
        "capture_latency_ms": latency,
        "capture_effective_ms": effective,
        "objective_value": objective_value,
        "fraction_of_optimum": (
            objective_value / optimum_value
            if optimum_value is not None and optimum_value > 0 else ""
        ),
        "oracle_evaluations": result.evaluations,
    }


def _evaluate_problem(
    problem: CaptureProblem,
    *,
    exact: bool,
    include_current: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    information_results = [
        greedy_select(problem, problem.information, method="greedy_information"),
        greedy_select(
            problem,
            problem.information,
            density=True,
            method="cost_aware_greedy",
        ),
        bundle_density_greedy(
            problem,
            problem.information,
            max_bundle_size=2,
            method="pair_bundle_greedy",
        ),
    ]
    if include_current:
        information_results.insert(0, fixed_current_result(problem, problem.information))
    information_optimum = exact_select(problem, problem.information) if exact else None
    if information_optimum is not None:
        information_results.append(information_optimum)
    optimum_information = (
        information_optimum.objective_value if information_optimum is not None else None
    )
    rows.extend(
        _metric_row(
            problem,
            result,
            optimized_objective="information",
            optimum_value=optimum_information,
        )
        for result in information_results
    )

    def bundle_inner(value):
        return bundle_density_greedy(
            problem,
            value,
            max_bundle_size=2,
            method="dinkelbach_pair_inner",
        )

    proposed = dinkelbach_select(problem, bundle_inner)
    by_method = {result.method: result for result in information_results}
    throughput_results = [
        SelectionResult(
            "greedy_information",
            by_method["greedy_information"].selected,
            problem.accepted_tokens(by_method["greedy_information"].selected)
            / problem.round_time_ms(by_method["greedy_information"].selected),
            by_method["greedy_information"].evaluations,
            by_method["greedy_information"].feasible,
        ),
        SelectionResult(
            "cost_aware_greedy",
            by_method["cost_aware_greedy"].selected,
            problem.accepted_tokens(by_method["cost_aware_greedy"].selected)
            / problem.round_time_ms(by_method["cost_aware_greedy"].selected),
            by_method["cost_aware_greedy"].evaluations,
            by_method["cost_aware_greedy"].feasible,
        ),
        _selection_from_dinkelbach(proposed, "dinkelbach_pair_bundle"),
    ]
    if include_current:
        throughput_results.insert(0, fixed_current_result(
            problem, lambda s: problem.accepted_tokens(s) / problem.round_time_ms(s)
        ))
    throughput_optimum: SelectionResult | None = None
    if exact:
        exact_dinkelbach = dinkelbach_select(
            problem,
            lambda value: exact_select(problem, value, method="exact_inner"),
        )
        throughput_optimum = _selection_from_dinkelbach(
            exact_dinkelbach, "true_optimum"
        )
        throughput_results.append(throughput_optimum)
    optimum_throughput = (
        throughput_optimum.objective_value if throughput_optimum is not None else None
    )
    rows.extend(
        _metric_row(
            problem,
            result,
            optimized_objective="throughput",
            optimum_value=optimum_throughput,
        )
        for result in throughput_results
    )
    return rows


def _restrict_problem(
    base: CaptureProblem,
    *,
    name: str,
    layers: Sequence[int],
) -> CaptureProblem:
    selected_layers = canonical_subset(layers)
    if not set(selected_layers).issubset(base.layers):
        raise ValueError("restricted layers must be candidates in the base problem")
    return CaptureProblem(
        name=name,
        layers=selected_layers,
        costs={layer: base.costs[layer] for layer in selected_layers},
        constraints=base.constraints,
        information=base.information,
        accepted_tokens=base.accepted_tokens,
        round_time_ms=base.round_time_ms,
        bandwidth_bytes_per_ms=base.bandwidth_bytes_per_ms,
        current_layers=base.current_layers,
    )


def _context_problem(base: CaptureProblem, information, regime_name: str) -> CaptureProblem:
    @lru_cache(maxsize=None)
    def info(selected):
        return information.regime_information(regime_name, canonical_subset(selected))

    @lru_cache(maxsize=None)
    def accepted(selected):
        return 0.35 + 7.65 * (1.0 - math.exp(-0.47 * info(selected)))

    @lru_cache(maxsize=None)
    def time_ms(selected):
        subset = canonical_subset(selected)
        capture = sum(base.effective_cost(layer) for layer in subset)
        return 34.0 + capture + 5.5 * (8.0 - accepted(subset))

    return CaptureProblem(
        name=f"{base.name}/{regime_name}",
        layers=base.layers,
        costs=base.costs,
        constraints=base.constraints,
        information=info,
        accepted_tokens=accepted,
        round_time_ms=time_ms,
        bandwidth_bytes_per_ms=base.bandwidth_bytes_per_ms,
        current_layers=base.current_layers,
    )


def _adaptive_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base, information = make_l45_capture_problem("change_point")
    context_probs = {regime.name: regime.probability for regime in information.regimes}
    context_problems = {
        name: _context_problem(base, information, name)
        for name in context_probs
    }

    fixed = dinkelbach_select(
        base,
        lambda value: bundle_density_greedy(
            base, value, max_bundle_size=2, method="fixed_inner"
        ),
    )
    adaptive = adaptive_dinkelbach_select(
        context_probs,
        context_problems,
        lambda _name, problem, value: bundle_density_greedy(
            problem, value, max_bundle_size=2, method="adaptive_inner"
        ),
    )
    rows.append({
        "scenario": "change_point_l45",
        "policy": "fixed",
        "mapping": json.dumps({"all": list(fixed.selected)}, sort_keys=True),
        "accepted_tokens": fixed.accepted_tokens,
        "round_time_ms": fixed.round_time_ms,
        "throughput_tokens_per_second": 1000 * fixed.throughput_tokens_per_ms,
        "adaptivity_ratio": 1.0,
    })
    rows.append({
        "scenario": "change_point_l45",
        "policy": "cheap_context_adaptive",
        "mapping": json.dumps({k: list(v) for k, v in adaptive.policy.items()}, sort_keys=True),
        "accepted_tokens": adaptive.expected_accepted_tokens,
        "round_time_ms": adaptive.expected_round_time_ms,
        "throughput_tokens_per_second": 1000 * adaptive.throughput_tokens_per_ms,
        "adaptivity_ratio": adaptive.throughput_tokens_per_ms / fixed.throughput_tokens_per_ms,
    })

    # Two disjoint five-capture actions.  A mismatch causes an M-cost verify and
    # fallback path.  The adaptive/fixed ratio is M+1 and is unbounded in M.
    for mismatch_time in (10.0, 100.0, 1000.0):
        actions = ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
        probabilities = {"early": 0.5, "late": 0.5}
        accepted = {}
        times = {}
        for context, matched in zip(probabilities, actions):
            for action in actions:
                accepted[(context, action)] = 1.0 if action == matched else 0.0
                times[(context, action)] = 1.0 if action == matched else mismatch_time
        result = adaptive_context_values(probabilities, actions, accepted, times)
        rows.append({
            "scenario": f"arbitrary_gap_M{int(mismatch_time)}",
            "policy": "fixed_vs_adaptive",
            "mapping": json.dumps(result, sort_keys=True),
            "accepted_tokens": result["adaptive_accepted"],
            "round_time_ms": result["adaptive_time"],
            "throughput_tokens_per_second": 1000 * result["adaptive_throughput"],
            "adaptivity_ratio": result["adaptivity_ratio"],
        })

    actions = ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
    probabilities = {"x0": 0.4, "x1": 0.6}
    accepted = {}
    times = {}
    for context in probabilities:
        accepted[(context, actions[0])] = 2.0
        times[(context, actions[0])] = 2.0
        accepted[(context, actions[1])] = 1.0
        times[(context, actions[1])] = 2.0
    no_help = adaptive_context_values(probabilities, actions, accepted, times)
    rows.append({
        "scenario": "common_argmax_no_help",
        "policy": "fixed_vs_adaptive",
        "mapping": json.dumps(no_help, sort_keys=True),
        "accepted_tokens": no_help["adaptive_accepted"],
        "round_time_ms": no_help["adaptive_time"],
        "throughput_tokens_per_second": 1000 * no_help["adaptive_throughput"],
        "adaptivity_ratio": no_help["adaptivity_ratio"],
    })
    return rows


def evaluate() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    xor_network = make_xor_synergy_network()
    naive_network = make_naive_bayes_network((0.07, 0.13, 0.20, 0.28, 0.35))
    duplicate_network = make_duplicate_network()

    small_problems = (
        make_small_problem_from_network("xor_synergy_small", xor_network, max_items=2),
        make_small_problem_from_network(
            "conditional_independent_small", naive_network, max_items=3
        ),
        make_small_problem_from_network("near_duplicate_small", duplicate_network, max_items=2),
        make_small_problem_from_network(
            "informative_but_expensive_small",
            make_naive_bayes_network((0.01, 0.16, 0.19, 0.23, 0.27)),
            max_items=2,
            expensive_layer=1,
        ),
    )
    rows: list[dict[str, Any]] = []
    for problem in small_problems:
        rows.extend(_evaluate_problem(problem, exact=True, include_current=False))
    restricted_base, _ = make_l45_capture_problem("change_point")
    restricted = _restrict_problem(
        restricted_base,
        name="l45_restricted_small",
        layers=(5, 8, 10, 14, 24, 33, 38, 42),
    )
    rows.extend(_evaluate_problem(restricted, exact=True, include_current=True))
    for kind in ("near_duplicate", "change_point", "expensive_late"):
        problem, _ = make_l45_capture_problem(kind)
        rows.extend(_evaluate_problem(problem, exact=False, include_current=True))

    naive_gap, naive_witness = minimum_submodularity_gap(
        naive_network.layers, naive_network.mutual_information
    )
    xor_gap, xor_witness = minimum_submodularity_gap(
        xor_network.layers, xor_network.mutual_information
    )
    duplicate_gap, duplicate_witness = minimum_submodularity_gap(
        duplicate_network.layers, duplicate_network.mutual_information
    )
    near_duplicate_l45, _ = make_l45_capture_problem("near_duplicate")
    checks = {
        "conditional_independent_min_submodularity_gap": naive_gap,
        "conditional_independent_witness": [
            list(naive_witness[0]),
            list(naive_witness[1]),
            naive_witness[2],
        ],
        "xor_min_submodularity_gap": xor_gap,
        "xor_witness": [list(xor_witness[0]), list(xor_witness[1]), xor_witness[2]],
        "duplicate_min_submodularity_gap": duplicate_gap,
        "duplicate_witness": [
            list(duplicate_witness[0]),
            list(duplicate_witness[1]),
            duplicate_witness[2],
        ],
        "xor_gamma_q1": bundle_submodularity_ratio(
            xor_network.layers,
            xor_network.mutual_information,
            max_items=2,
            max_bundle_size=1,
        ),
        "xor_gamma_q2": bundle_submodularity_ratio(
            xor_network.layers,
            xor_network.mutual_information,
            max_items=2,
            max_bundle_size=2,
        ),
        "duplicate_information": {
            "z1": duplicate_network.mutual_information((1,)),
            "z2": duplicate_network.mutual_information((2,)),
            "z1_z2": duplicate_network.mutual_information((1, 2)),
        },
        "l45_near_duplicate_information": {
            "z23": near_duplicate_l45.information((23,)),
            "z24": near_duplicate_l45.information((24,)),
            "z23_z24": near_duplicate_l45.information((23, 24)),
            "conditional_gain_z24_given_z23": (
                near_duplicate_l45.information((23, 24))
                - near_duplicate_l45.information((23,))
            ),
        },
        "fano_block_bits_uniform_256_error_005": fano_block_bits(8.0, 256, 0.05),
        "fano_prefix_bits_binary_k8_mean65": prefix_acceptance_fano_bits(
            block_length=8, expected_prefix=6.5, alphabet_size=2
        ),
        "kill_example_net_upper": kill_criterion_net_upper_bound(
            accepted_gain_upper=0.08,
            added_time_lower_ms=2.0,
            baseline_throughput_lower_tokens_per_ms=0.05,
        ),
        "synthetic_only": True,
    }
    return rows, _adaptive_rows(), checks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scratch/dflash-capture-observability"),
    )
    args = parser.parse_args(argv)
    rows, adaptive, checks = evaluate()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "summary.csv", rows)
    _write_csv(args.out_dir / "adaptive.csv", adaptive)
    (args.out_dir / "theorem-checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "summary_rows": len(rows),
        "adaptive_rows": len(adaptive),
        "checks": checks,
        "out_dir": str(args.out_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
