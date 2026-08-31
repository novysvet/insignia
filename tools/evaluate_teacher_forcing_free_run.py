#!/usr/bin/env python3
"""Reproduce the finite-state and anytime-certification results for Problem 1."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from teacher_forcing_free_run import (
    AdversarySearchResult,
    bernoulli_mixture_cs_upper,
    binary_kl_upper,
    corpus_coverage_hole_model,
    direct_kl,
    enumerate_trajectory_laws,
    forward_metrics,
    greedy_metric_counterexample,
    pinsker_upper,
    ppl_reallocation_counterexample,
    rare_history_cascade,
    search_high_failure_pairs,
    simulate_anytime_coverage,
)


def encode_fraction(value: Fraction | float) -> str | float:
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    return float(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def cascade_record(horizon: int, hazard: Fraction) -> dict[str, Any]:
    metrics = forward_metrics(rare_history_cascade(horizon, hazard, vocab_size=8))
    p_failure = float(metrics.p_failure)
    q_failure = float(metrics.q_failure)
    return {
        "horizon": horizon,
        "hazard": str(hazard),
        "mean_teacher_kl_per_token": -math.log(float(1 - hazard)),
        "total_teacher_trajectory_kl": metrics.trajectory_kl,
        "per_step_ppl_ratio": float(1 / (1 - hazard)),
        "ppl_delta_fraction": float(1 / (1 - hazard) - 1),
        "top1_agreement": 1.0,
        "p_failure": encode_fraction(metrics.p_failure),
        "q_failure": encode_fraction(metrics.q_failure),
        "q_failure_float": q_failure,
        "sharp_binary_kl_upper": binary_kl_upper(p_failure, metrics.trajectory_kl),
        "pinsker_upper": pinsker_upper(p_failure, metrics.trajectory_kl),
        "symbolic_trajectory_kl": (
            metrics.symbolic_trajectory_kl.to_json_dict()
            if metrics.symbolic_trajectory_kl is not None else None
        ),
    }


def exact_path_law_payload() -> dict[str, Any]:
    model = rare_history_cascade(4, Fraction(1, 5), vocab_size=2)
    laws = enumerate_trajectory_laws(model)
    metrics = forward_metrics(model)
    paths = []
    for tokens in sorted(laws.p):
        paths.append({
            "tokens": "".join(str(token) for token in tokens),
            "p": encode_fraction(laws.p[tokens]),
            "q": encode_fraction(laws.q[tokens]),
            "failure": laws.failure[tokens],
        })
    return {
        "vocabulary_size": 2,
        "horizon": 4,
        "paths": paths,
        "p_mass": encode_fraction(sum(laws.p.values(), Fraction(0))),
        "q_mass": encode_fraction(sum(laws.q.values(), Fraction(0))),
        "direct_trajectory_kl": direct_kl(laws.p, laws.q),
        "chain_rule_trajectory_kl": metrics.trajectory_kl,
        "q_failure": encode_fraction(metrics.q_failure),
        "symbolic_trajectory_kl": metrics.symbolic_trajectory_kl.to_json_dict(),
    }


def search_rows(rows: list[AdversarySearchResult]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/teacher-forcing-free-run"),
    )
    parser.add_argument("--search-trials", type=int, default=250)
    parser.add_argument("--coverage-trials", type=int, default=5_000)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    hazard = Fraction(1, 30)
    cascades = [cascade_record(horizon, hazard) for horizon in (50, 256)]
    write_json(args.output / "exact-cascades.json", cascades)
    write_json(args.output / "exact-path-law.json", exact_path_law_payload())

    adversaries = search_high_failure_pairs(
        budget=0.5,
        horizon=12,
        vocab_size=3,
        state_count=4,
        trials=args.search_trials,
        strength_grid=64,
        seed=7,
        keep=20,
    )
    write_csv(args.output / "adversary-search.csv", search_rows(adversaries))

    coverage_specs = [
        (0.0, 0.02),
        (0.02, 0.08),
        (0.10, 0.20),
        (0.50, None),
    ]
    coverage_rows: list[dict[str, Any]] = []
    for index, (true_failure, threshold) in enumerate(coverage_specs):
        result = simulate_anytime_coverage(
            true_failure=true_failure,
            alpha=0.05,
            max_samples=300,
            trials=args.coverage_trials,
            certify_threshold=threshold,
            seed=100 + index,
        )
        row = asdict(result)
        row["certify_threshold"] = threshold
        coverage_rows.append(row)
    write_csv(args.output / "confidence-coverage.csv", coverage_rows)

    zero_failure_rows = [
        {
            "candidate_trajectories": n,
            "observed_failures": 0,
            "alpha": 0.05,
            "anytime_upper": bernoulli_mixture_cs_upper(0, n, alpha=0.05),
        }
        for n in (100, 500, 1_000, 5_000, 10_000)
    ]
    write_csv(args.output / "zero-failure-confidence-widths.csv", zero_failure_rows)

    ppl_example = ppl_reallocation_counterexample(epsilon=1e-6)
    greedy_example = greedy_metric_counterexample(margin=1e-6, background=1e6)
    coverage_hole_metrics = forward_metrics(
        corpus_coverage_hole_model(Fraction(1, 100))
    )
    coverage_hole = {
        "corpus_state": "logged",
        "deployment_state": "deployment",
        "corpus_mean_local_kl": 0.0,
        "deployment_trajectory_kl": coverage_hole_metrics.trajectory_kl,
        "p_failure": encode_fraction(coverage_hole_metrics.p_failure),
        "q_failure": encode_fraction(coverage_hole_metrics.q_failure),
        "deployment_to_corpus_density_cap": "infinity",
    }
    write_json(args.output / "ppl-reallocation-counterexample.json", ppl_example)
    write_json(args.output / "greedy-metric-counterexample.json", greedy_example)
    write_json(args.output / "corpus-coverage-hole.json", coverage_hole)

    best = adversaries[0] if adversaries else None
    summary = {
        "status": "reproduced",
        "exact_path_chain_rule_error": abs(
            exact_path_law_payload()["direct_trajectory_kl"]
            - exact_path_law_payload()["chain_rule_trajectory_kl"]
        ),
        "cascade_50": cascades[0],
        "cascade_256": cascades[1],
        "best_random_adversary": asdict(best) if best is not None else None,
        "zero_failure_anytime_ucb": {
            str(row["candidate_trajectories"]): row["anytime_upper"]
            for row in zero_failure_rows
        },
        "coverage": coverage_rows,
        "ppl_reallocation": ppl_example,
        "corpus_coverage_hole": coverage_hole,
        "greedy_metric_flip": greedy_example,
        "decision": {
            "stochastic_teacher_forcing": (
                "Use inverse binary KL only after summing local KL under the "
                "actual exact sampling occupancy and supplying exact-policy failure."
            ),
            "greedy": (
                "Use per-step top-1 equality or the equivalent pairwise margin/error "
                "inequalities; softmax KL is not a greedy-policy certificate."
            ),
            "minimum_new_log": (
                "Randomized direct candidate trajectories with predictable source "
                "propensity and a trajectory failure flag."
            ),
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
