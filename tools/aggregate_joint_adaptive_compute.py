#!/usr/bin/env python3
"""Pool independent joint adaptive-compute simulation replications.

The per-run evaluator writes raw counters.  This script sums those counters,
reports ratio-of-sums estimates, keeps the replicate dispersion, and adds
Wilson intervals for the two event-rate constraints.  It also copies the
static finite-model artifacts from the first replication into one final output
directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

QUALITY_BUDGETS = {
    "ppl_loss_per_token": 0.0045,
    "hard_violation_probability": 0.035,
    "catastrophe_per_token": 0.00035,
}
STATIC_FILES = (
    "exact-policy.json",
    "model-summary.json",
    "fixed-sweep.csv",
    "voi-table.csv",
    "counterexamples.json",
)
RAW_SUM_FIELDS = (
    "rounds",
    "committed_tokens",
    "wall_time_ms",
    "bytes_mb_total",
    "ppl_loss_total",
    "hard_ends",
    "hard_violations",
    "catastrophes",
    "measurement_count",
    "fallback_count",
    "measurement_time_ms",
    "guard_time_ms",
    "cache_level_sum",
    "io_queue_sum",
    "hidden_hard_count",
    "debt_and_queue2_count",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _as_number(value: str) -> float:
    return float(value)


def wilson_interval(successes: float, trials: float, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli event rate."""

    if trials <= 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def mean_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float, float, float]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if finite.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(finite.mean())
    if finite.size == 1:
        return mean, math.nan, math.nan, math.nan
    se = float(finite.std(ddof=1) / math.sqrt(finite.size))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, finite.size - 1))
    return mean, se, mean - critical * se, mean + critical * se


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def aggregate(run_dirs: Sequence[Path], out_dir: Path) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("at least two replication directories are required")
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    replicate_rows: list[dict[str, Any]] = []
    action_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    action_start_totals: dict[tuple[str, str], int] = defaultdict(int)
    seeds: list[int] = []
    exact_metadata: dict[str, Any] | None = None

    expected_keys: set[tuple[str, str]] | None = None
    for replication, run_dir in enumerate(run_dirs):
        summary_path = run_dir / "summary.csv"
        evaluation_path = run_dir / "evaluation.json"
        if not summary_path.exists() or not evaluation_path.exists():
            raise FileNotFoundError(f"missing replication artifacts under {run_dir}")
        metadata = json.loads(evaluation_path.read_text())
        seeds.append(int(metadata["seed"]))
        if exact_metadata is None:
            exact_metadata = metadata
        rows = _read_csv(summary_path)
        keys = {(row["scenario"], row["policy"]) for row in rows}
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"scenario/policy mismatch in {run_dir}")
        for row in rows:
            parsed: dict[str, Any] = dict(row)
            for field in RAW_SUM_FIELDS:
                parsed[field] = _as_number(row[field])
            for field in (
                "committed_tokens_per_second",
                "ppl_loss_per_token",
                "hard_violation_probability",
                "catastrophe_per_token",
                "bytes_per_token_mb",
                "measurement_fraction",
                "fallback_fraction",
                "mean_cache_level",
                "mean_io_queue",
                "hidden_hard_fraction",
                "debt_and_queue2_fraction",
            ):
                parsed[field] = _as_number(row[field])
            parsed["replication"] = replication
            parsed["seed"] = int(metadata["seed"])
            grouped[(row["scenario"], row["policy"])].append(parsed)
            replicate_rows.append(parsed)

        action_path = run_dir / "action-frequency.csv"
        for row in _read_csv(action_path):
            key = (row["scenario"], row["policy"], row["action"])
            count = int(row["count"])
            action_counts[key] += count
            action_start_totals[(row["scenario"], row["policy"])] += count

    summary_rows: list[dict[str, Any]] = []
    for (scenario, policy), rows in sorted(grouped.items()):
        totals = {field: sum(float(row[field]) for row in rows) for field in RAW_SUM_FIELDS}
        rounds = totals["rounds"]
        tokens = totals["committed_tokens"]
        time_ms = totals["wall_time_ms"]
        hard_ends = totals["hard_ends"]
        hard_violations = totals["hard_violations"]
        catastrophes = totals["catastrophes"]
        tps = 1000.0 * _ratio(tokens, time_ms)
        ppl = _ratio(totals["ppl_loss_total"], tokens)
        hard = _ratio(hard_violations, hard_ends)
        catastrophe = _ratio(catastrophes, tokens)
        hard_lo, hard_hi = wilson_interval(hard_violations, hard_ends)
        cat_lo, cat_hi = wilson_interval(catastrophes, tokens)
        tps_mean, tps_se, tps_lo, tps_hi = mean_interval(
            row["committed_tokens_per_second"] for row in rows
        )
        ppl_mean, ppl_se, ppl_lo, ppl_hi = mean_interval(
            row["ppl_loss_per_token"] for row in rows
        )
        hard_mean, hard_se, hard_t_lo, hard_t_hi = mean_interval(
            row["hard_violation_probability"] for row in rows
        )
        row_out = {
            "scenario": scenario,
            "policy": policy,
            "replications": len(rows),
            "rounds": int(rounds),
            "committed_tokens": int(tokens),
            "wall_time_ms": time_ms,
            "ppl_loss_total": totals["ppl_loss_total"],
            "hard_ends": int(hard_ends),
            "hard_violations": int(hard_violations),
            "catastrophes": int(catastrophes),
            "measurement_count": int(totals["measurement_count"]),
            "fallback_count": int(totals["fallback_count"]),
            "committed_tokens_per_second": tps,
            "replicate_mean_tokens_per_second": tps_mean,
            "tokens_per_second_se": tps_se,
            "tokens_per_second_ci95_low": tps_lo,
            "tokens_per_second_ci95_high": tps_hi,
            "ppl_loss_per_token": ppl,
            "replicate_mean_ppl_loss_per_token": ppl_mean,
            "ppl_loss_per_token_se": ppl_se,
            "ppl_loss_per_token_ci95_low": ppl_lo,
            "ppl_loss_per_token_ci95_high": ppl_hi,
            "hard_violation_probability": hard,
            "hard_violation_wilson95_low": hard_lo,
            "hard_violation_wilson95_high": hard_hi,
            "replicate_mean_hard_violation_probability": hard_mean,
            "hard_violation_replicate_se": hard_se,
            "hard_violation_replicate_ci95_low": hard_t_lo,
            "hard_violation_replicate_ci95_high": hard_t_hi,
            "catastrophe_per_token": catastrophe,
            "catastrophe_wilson95_low": cat_lo,
            "catastrophe_wilson95_high": cat_hi,
            "bytes_per_token_mb": _ratio(totals["bytes_mb_total"], tokens),
            "measurement_fraction": _ratio(totals["measurement_count"], rounds),
            "fallback_fraction": _ratio(totals["fallback_count"], rounds),
            "measurement_time_fraction": _ratio(totals["measurement_time_ms"], time_ms),
            "guard_time_fraction": _ratio(totals["guard_time_ms"], time_ms),
            "mean_cache_level": _ratio(totals["cache_level_sum"], rounds),
            "mean_io_queue": _ratio(totals["io_queue_sum"], rounds),
            "hidden_hard_fraction": _ratio(totals["hidden_hard_count"], rounds),
            "debt_and_queue2_fraction": _ratio(totals["debt_and_queue2_count"], rounds),
            "ood_latched_replication_fraction": sum(
                str(row.get("ood_latched", "False")).lower() == "true" for row in rows
            ) / len(rows),
            "ppl_constraint_met_point": ppl <= QUALITY_BUDGETS["ppl_loss_per_token"],
            "ppl_constraint_met_ci95": (
                math.isfinite(ppl_hi) and ppl_hi <= QUALITY_BUDGETS["ppl_loss_per_token"]
            ),
            "hard_constraint_met_point": hard <= QUALITY_BUDGETS["hard_violation_probability"],
            "hard_constraint_met_wilson95": (
                math.isfinite(hard_hi)
                and hard_hi <= QUALITY_BUDGETS["hard_violation_probability"]
            ),
            "catastrophe_constraint_met_point": catastrophe
            <= QUALITY_BUDGETS["catastrophe_per_token"],
            "catastrophe_constraint_met_wilson95": (
                math.isfinite(cat_hi)
                and cat_hi <= QUALITY_BUDGETS["catastrophe_per_token"]
            ),
        }
        summary_rows.append(row_out)

    action_rows: list[dict[str, Any]] = []
    for (scenario, policy, action), count in sorted(action_counts.items()):
        total = action_start_totals[(scenario, policy)]
        action_rows.append(
            {
                "scenario": scenario,
                "policy": policy,
                "action": action,
                "count": count,
                "fraction_of_action_starts": count / total,
            }
        )

    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "replicate-summary.csv", replicate_rows)
    _write_csv(out_dir / "action-frequency.csv", action_rows)

    first = run_dirs[0]
    for name in STATIC_FILES:
        source = first / name
        if source.exists():
            shutil.copy2(source, out_dir / name)

    by_key = {(row["scenario"], row["policy"]): row for row in summary_rows}
    calibrated_robust = by_key[("calibrated", "robust_safe_joint")]
    calibrated_fixed = by_key[("calibrated", "best_fixed")]
    calibrated_exact = by_key[("calibrated", "exact_joint")]
    calibrated_safe = by_key[("calibrated", "safe_exact_fixed")]
    baseline_rho = calibrated_fixed["committed_tokens_per_second"] / 1000.0
    robust_gain = (
        calibrated_robust["committed_tokens_per_second"]
        / calibrated_fixed["committed_tokens_per_second"]
        - 1.0
    )
    robust_transformed_gain_per_round = (
        calibrated_robust["committed_tokens"] / calibrated_robust["rounds"]
        - baseline_rho
        * calibrated_robust["wall_time_ms"]
        / calibrated_robust["rounds"]
    )
    fallback = calibrated_robust["fallback_fraction"]
    joint_y = calibrated_exact["committed_tokens"] / calibrated_exact["rounds"]
    joint_t = calibrated_exact["wall_time_ms"] / calibrated_exact["rounds"]
    safe_y = calibrated_safe["committed_tokens"] / calibrated_safe["rounds"]
    safe_t = calibrated_safe["wall_time_ms"] / calibrated_safe["rounds"]
    guard = calibrated_robust["guard_time_fraction"] * calibrated_robust["wall_time_ms"] / calibrated_robust["rounds"]
    mixed_y = (1.0 - fallback) * joint_y + fallback * safe_y
    mixed_t = (1.0 - fallback) * joint_t + fallback * safe_t + guard
    joint_rate = 1000.0 * joint_y / joint_t
    safe_rate = 1000.0 * mixed_y / mixed_t
    observed_robust_rate = calibrated_robust["committed_tokens_per_second"]
    price = {
        "joint_tokens_per_second": joint_rate,
        "observed_robust_tokens_per_second": observed_robust_rate,
        "observed_absolute_price_tokens_per_second": joint_rate - observed_robust_rate,
        "observed_relative_price": (joint_rate - observed_robust_rate) / joint_rate,
        "state_agnostic_mixture_tokens_per_second": safe_rate,
        "state_agnostic_mixture_absolute_price_tokens_per_second": joint_rate - safe_rate,
        "state_agnostic_mixture_relative_price": (joint_rate - safe_rate) / joint_rate,
        "fallback_fraction": fallback,
        "guard_ms_per_round": guard,
    }
    kill = {
        "controller_cost_exceeds_predicted_saving": robust_transformed_gain_per_round <= 0.0,
        "safe_exploration_lacks_overlap": fallback >= 0.95,
        "no_material_robust_gain_over_best_fixed": robust_gain < 0.03,
        "robust_gain_over_best_fixed": robust_gain,
        "robust_transformed_gain_per_round": robust_transformed_gain_per_round,
        "kill_controller": False,
    }
    kill["kill_controller"] = any(
        bool(value)
        for key, value in kill.items()
        if key not in {
            "robust_gain_over_best_fixed",
            "robust_transformed_gain_per_round",
            "kill_controller",
        }
    )

    payload = {
        "synthetic_only": True,
        "replications": len(run_dirs),
        "seeds": seeds,
        "quality_budgets": QUALITY_BUDGETS,
        "best_fixed_action": exact_metadata["best_fixed_action"] if exact_metadata else None,
        "exact_finite_result": exact_metadata["exact_finite_result"] if exact_metadata else None,
        "simulation": summary_rows,
        "throughput_price_of_safety": price,
        "kill_criteria": kill,
        "confidence_note": (
            "Wilson intervals treat event counts as Bernoulli trials; replicate t intervals "
            "describe synthetic-run variability and are not hardware certificates."
        ),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _discover_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("run-*/summary.csv"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="directory containing run-*/summary.csv")
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run_dirs = list(args.run_dir)
    if args.root:
        run_dirs.extend(_discover_runs(args.root))
    run_dirs = sorted(dict.fromkeys(path.resolve() for path in run_dirs))
    if not run_dirs:
        parser.error("supply --root or one or more --run-dir values")
    payload = aggregate(run_dirs, args.out_dir)
    print(f"pooled {payload['replications']} replications")
    print(f"best fixed action: {payload['best_fixed_action']}")
    print("scenario             policy                    tok/s      ppl/tok    hard [95% U]     cat/tok   fallback")
    for row in payload["simulation"]:
        print(
            f"{row['scenario']:20s} {row['policy']:24s} "
            f"{row['committed_tokens_per_second']:8.3f} "
            f"{row['ppl_loss_per_token']:11.6f} "
            f"{row['hard_violation_probability']:7.4f} [{row['hard_violation_wilson95_high']:7.4f}] "
            f"{row['catastrophe_per_token']:9.6f} "
            f"{row['fallback_fraction']:9.4f}"
        )
    print("kill criteria:")
    for key, value in payload["kill_criteria"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
