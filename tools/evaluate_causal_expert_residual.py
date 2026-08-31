#!/usr/bin/env python3
"""Evaluate synthetic causal expert predictor/residual schemes.

All outputs are generated from explicit synthetic parameters.  The script does
not treat repository hardware notes as a residual-correlation trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal_expert_residual import (  # noqa: E402
    HardwareCostModel,
    PrefixPolicySolution,
    SyntheticExpertTrace,
    SyntheticParameters,
    arbitrary_map_lower_bound,
    build_prefix_table_from_residuals,
    conditional_entropy_bits,
    conditional_mutual_information_bits,
    conditional_rate_distortion_curve,
    decode_xor_residual_all,
    demo_cache_problem,
    encode_categories,
    encode_xor_residual_container,
    find_fp32_additive_counterexample,
    find_fp32_fma_additive_counterexample,
    make_synthetic_trace,
    mutual_information_bits,
    predict_system_costs,
    quantize_symmetric,
    representation_ledger,
    solve_prefix_policy,
    solve_small_cache_problem,
)

FAMILIES = (
    "independent_random",
    "exact_shared_basis",
    "shared_basis_sparse_residual",
    "slowly_drifting",
    "route_only_adversary",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row_slices(rows: int, chunks: int) -> tuple[slice, ...]:
    bounds = np.linspace(0, rows, chunks + 1, dtype=np.int64)
    return tuple(slice(int(bounds[j]), int(bounds[j + 1])) for j in range(chunks))


def _bayes_mse(values: np.ndarray, context: np.ndarray) -> float:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    c = encode_categories(context)
    if y.size != c.size:
        raise ValueError("Bayes-MSE value/context length mismatch")
    total = 0.0
    for category in np.unique(c):
        selected = y[c == category]
        total += float(np.sum((selected - np.mean(selected)) ** 2))
    return total / y.size


def _scalar_probe(
    trace: SyntheticExpertTrace,
    token_slice: slice,
    coordinates: int,
    *,
    scale: float | None = None,
    levels: int,
) -> dict[str, Any]:
    output_dim = trace.residual_outputs.shape[1]
    coordinates = min(coordinates, output_dim)
    values = trace.residual_outputs[token_slice, :coordinates].astype(np.float64)
    if scale is None:
        scale = float(np.sqrt(np.mean(values * values)))
    scale = max(scale, 1e-12)
    normalized = values / scale
    quantized = quantize_symmetric(normalized, levels=levels, scale=1.0)
    tokens = values.shape[0]
    coordinate = np.tile(np.arange(coordinates, dtype=np.int64), tokens)
    route = np.repeat(trace.routes[token_slice], coordinates)
    top = np.repeat(trace.previous_top_route[token_slice], coordinates)
    margin = np.repeat(trace.previous_margin_bin[token_slice], coordinates)
    symbols = quantized.symbols.reshape(-1)
    normalized_flat = normalized.reshape(-1)
    base_context = encode_categories(coordinate, route)
    logit_context = encode_categories(coordinate, route, margin)
    return {
        "scale": scale,
        "centers": quantized.centers,
        "symbols": symbols,
        "normalized": normalized_flat,
        "coordinate": coordinate,
        "route": route,
        "top": top,
        "margin": margin,
        "none_context": coordinate,
        "route_context": base_context,
        "logit_context": logit_context,
    }


def _information_rows(
    trace: SyntheticExpertTrace,
    train_slice: slice,
    test_slice: slice,
    coordinates: int,
    levels: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    train = _scalar_probe(trace, train_slice, coordinates, levels=levels)
    test = _scalar_probe(
        trace,
        test_slice,
        coordinates,
        scale=float(train["scale"]),
        levels=levels,
    )
    symbols = test["symbols"]
    base = test["route_context"]
    with_logits = test["logit_context"]
    cmi = conditional_mutual_information_bits(symbols, test["margin"], base)
    base_mse = _bayes_mse(test["normalized"], base)
    logit_mse = _bayes_mse(test["normalized"], with_logits)
    info = {
        "family": trace.family,
        "heldout_tokens": trace.routes[test_slice].size,
        "probe_coordinates": min(coordinates, trace.residual_outputs.shape[1]),
        "route_prediction_accuracy": float(
            np.mean(trace.previous_top_route[test_slice] == trace.routes[test_slice])
        ),
        "route_information_bits": mutual_information_bits(
            trace.routes[test_slice], trace.previous_top_route[test_slice]
        ),
        "residual_entropy_bits_per_probe_scalar": conditional_entropy_bits(
            symbols, test["none_context"]
        ),
        "residual_entropy_given_route_bits": conditional_entropy_bits(symbols, base),
        "residual_entropy_given_route_and_prior_logit_bits": conditional_entropy_bits(
            symbols, with_logits
        ),
        "prior_logit_incremental_information_bits": cmi,
        "optimal_log_loss_reduction_bits": cmi,
        "base_bayes_mse_normalized": base_mse,
        "logit_bayes_mse_normalized": logit_mse,
        "squared_error_reduction": base_mse - logit_mse,
        "residual_rms_scale": float(train["scale"]),
    }
    rd_rows: list[dict[str, Any]] = []
    betas = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    contexts = {
        "coordinate-only": test["none_context"],
        "coordinate+route": base,
        "coordinate+route+prior-logit": with_logits,
    }
    for name, context in contexts.items():
        curve = conditional_rate_distortion_curve(
            symbols,
            context,
            test["centers"],
            betas=betas,
            tolerance=1e-7,
            max_iterations=200,
        )
        for point in curve:
            rd_rows.append(
                {
                    "family": trace.family,
                    "side_information": name,
                    "beta": point.beta,
                    "normalized_distortion": point.distortion,
                    "rate_bits_per_probe_scalar": point.rate_bits,
                }
            )
    return info, test, rd_rows


def _chunk_contributions(trace: SyntheticExpertTrace, token_slice: slice) -> np.ndarray:
    routes = trace.routes[token_slice]
    activations = trace.activations[token_slice]
    samples = routes.size
    output_dim = trace.weights.shape[1]
    chunks = int(trace.parameters["chunks"])
    result = np.zeros((samples, chunks, output_dim), dtype=np.float64)
    delta = trace.weights.astype(np.float64) - trace.predictor_weights.astype(np.float64)
    for chunk, rows in enumerate(_row_slices(output_dim, chunks)):
        selected = delta[routes, rows, :]
        values = np.einsum("toi,ti->to", selected, activations.astype(np.float64), optimize=False)
        result[:, chunk, rows] = values
    return result


def _policy_dict(solution: PrefixPolicySolution) -> dict[str, Any]:
    components = []
    for probability, policy in solution.components:
        histogram: dict[str, int] = {}
        for prefix in policy.prefixes:
            histogram[str(prefix)] = histogram.get(str(prefix), 0) + 1
        components.append(
            {
                "probability": probability,
                "prefixes_by_context": list(policy.prefixes),
                "prefix_histogram": histogram,
                "expected_cost": policy.expected_cost,
                "constraint_value": policy.constraint_value,
                "expected_distortion": policy.expected_distortion,
                "selective_risk_numerator": policy.selective_risk_numerator,
                "approximate_probability": policy.approximate_probability,
            }
        )
    return {
        "mode": solution.mode,
        "bound": solution.bound,
        "expected_cost": solution.expected_cost,
        "constraint_value": solution.constraint_value,
        "components": components,
    }


def _stopping_result(
    trace: SyntheticExpertTrace,
    container: Any,
    train_slice: slice,
    test_slice: slice,
) -> dict[str, Any]:
    train_chunks = _chunk_contributions(trace, train_slice)
    test_chunks = _chunk_contributions(trace, test_slice)
    chunks = train_chunks.shape[1]
    route_train = trace.routes[train_slice]
    counts = np.bincount(route_train, minlength=container.stats.experts).astype(np.float64)
    counts /= counts.sum()
    read_cost = np.zeros(chunks, dtype=np.float64)
    for chunk in range(chunks):
        read_cost[chunk] = sum(
            counts[expert] * container.descriptors[expert * chunks + chunk].extent_bytes
            for expert in range(container.stats.experts)
        )
    energy = np.mean(np.sum(train_chunks * train_chunks, axis=2), axis=0)
    ratio = np.divide(
        energy,
        read_cost,
        out=np.where(energy > 0.0, np.full_like(energy, np.inf), np.zeros_like(energy)),
        where=read_cost > 0.0,
    )
    priority = np.argsort(-ratio, kind="stable")
    train_chunks = train_chunks[:, priority, :]
    test_chunks = test_chunks[:, priority, :]
    read_cost = read_cost[priority]
    energy = energy[priority]
    context = encode_categories(
        trace.routes[test_slice] % 2, trace.previous_margin_bin[test_slice] % 2
    )
    full_tail = np.linalg.norm(np.sum(test_chunks, axis=1), axis=1)
    nonzero = full_tail[full_tail > 0]
    threshold = float(np.median(nonzero)) if nonzero.size else 0.0
    table = build_prefix_table_from_residuals(
        test_chunks,
        context,
        read_cost,
        tail_error_threshold=threshold,
    )
    baseline_distortion = float(
        np.sum(table.context_probability * table.expected_distortion[:, 0])
    )
    expected_solutions = []
    for fraction in (0.50, 0.10, 0.01, 0.0):
        bound = baseline_distortion * fraction
        solution = solve_prefix_policy(
            table,
            mode="expected_distortion",
            bound=bound,
            allow_randomization=True,
        )
        expected_solutions.append(
            {"distortion_fraction": fraction, **_policy_dict(solution)}
        )
    selective = solve_prefix_policy(
        table,
        mode="selective_risk",
        bound=0.0,
        selective_alpha=0.05,
        allow_randomization=True,
    )
    return {
        "family": trace.family,
        "priority_original_chunk_indices": priority.tolist(),
        "mean_chunk_read_bytes_in_priority_order": read_cost.tolist(),
        "mean_chunk_energy_in_priority_order": energy.tolist(),
        "tail_error_threshold": threshold,
        "predictor_only_expected_distortion": baseline_distortion,
        "expected_error_solutions": expected_solutions,
        "selective_risk_alpha": 0.05,
        "selective_risk_solution": _policy_dict(selective),
    }


def _rate_at_distortion(rows: Sequence[Mapping[str, Any]], target: float) -> float:
    points = sorted(
        (
            (float(row["normalized_distortion"]), float(row["rate_bits_per_probe_scalar"]))
            for row in rows
        ),
        reverse=True,
    )
    if not points:
        raise ValueError("empty rate-distortion curve")
    if target >= points[0][0]:
        return 0.0
    if target <= points[-1][0]:
        return points[-1][1]
    for (d0, r0), (d1, r1) in zip(points, points[1:]):
        if d0 >= target >= d1:
            if abs(d0 - d1) <= 1e-15:
                return min(r0, r1)
            weight = (d0 - target) / (d0 - d1)
            return r0 + weight * (r1 - r0)
    raise AssertionError("target distortion was not bracketed")


def _fixed_distortion_rows(rd_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_rows = [row for row in rd_rows if row["family"] == family]
        for target in (0.25, 0.10, 0.01):
            output: dict[str, Any] = {
                "family": family,
                "normalized_distortion": target,
            }
            for side in (
                "coordinate-only",
                "coordinate+route",
                "coordinate+route+prior-logit",
            ):
                selected = [row for row in family_rows if row["side_information"] == side]
                output[f"rate_{side}_bits"] = _rate_at_distortion(selected, target)
            result.append(output)
    return result


def _cache_results() -> list[dict[str, Any]]:
    scenarios = (
        ("tight-exact", 10, 8, 0.0),
        ("tight-approximate", 10, 8, 0.03),
        ("balanced-exact", 18, 10, 0.0),
        ("balanced-approximate", 18, 10, 0.03),
        ("roomy-exact", 30, 16, 0.0),
    )
    rows = []
    for name, ram, vram, distortion in scenarios:
        problem = demo_cache_problem(
            ram_capacity=ram, vram_capacity=vram, max_distortion=distortion
        )
        solution = solve_small_cache_problem(problem)
        rows.append(
            {
                "scenario": name,
                "problem": _jsonable(problem),
                "solution": _jsonable(solution),
            }
        )
    return rows


def _engine_residency_rows(
    ranks: Iterable[int], bytes_per_value: Iterable[int], full_record_bytes: int
) -> list[dict[str, Any]]:
    rows = []
    layers = 42
    experts = 288
    # One 4096-input basis can be shared by gate/up; down uses a 2048-input
    # basis.  Coefficients cover gate 2048, up 2048, and down 4096 outputs.
    for rank in ranks:
        for value_bytes in bytes_per_value:
            per_layer_basis_values = rank * (4096 + 2048)
            per_layer_coefficient_values = experts * rank * (2048 + 2048 + 4096)
            total = layers * (per_layer_basis_values + per_layer_coefficient_values) * value_bytes
            rows.append(
                {
                    "rank": rank,
                    "bytes_per_predictor_value": value_bytes,
                    "layers": layers,
                    "experts_per_layer": experts,
                    "predictor_resident_bytes": total,
                    "predictor_resident_gib": total / (1 << 30),
                    "equivalent_full_experts_displaced": total / full_record_bytes,
                }
            )
    return rows


def _engine_cost_rows(
    family: str,
    ledger: Any,
    trace: SyntheticExpertTrace,
    *,
    rank: int,
    full_record_bytes: int,
    cache_hit_rate: float,
    hardware: HardwareCostModel,
    prediction_miss_probability: float,
) -> list[dict[str, Any]]:
    rows = []
    residual_nnz_fraction = float(
        np.count_nonzero(trace.weights != trace.predictor_weights) / trace.weights.size
    )
    # GLM routed expert has two 2048x4096 projections and one 4096x2048
    # projection.  Operation counts use two scalar operations per contribution.
    full_gpu_ops_per_expert = 2.0 * (2 * 2048 * 4096 + 4096 * 2048)
    predictor_gpu_ops_per_expert = 2.0 * rank * 3 * (4096 + 2048)
    requests_per_layer = 8
    requests_per_token = 336
    read_ratio = ledger.mean_read_ratio

    def make(scope: str, requests: int, mode: str, initial_fraction: float) -> dict[str, Any]:
        misses = requests * (1.0 - cache_hit_rate)
        if mode == "full-record":
            read_fraction = 1.0
            read_bytes = misses * full_record_bytes
            gpu_ops = requests * full_gpu_ops_per_expert
            cpu_ops = 0.0
            synchronizations = misses
        else:
            read_fraction = initial_fraction + prediction_miss_probability * (1.0 - initial_fraction)
            read_bytes = misses * full_record_bytes * read_ratio * read_fraction
            gpu_ops = requests * (
                predictor_gpu_ops_per_expert
                + full_gpu_ops_per_expert * residual_nnz_fraction * read_fraction
            )
            cpu_ops = misses * full_record_bytes * ledger.changed_byte_fraction * read_fraction
            synchronizations = misses * ledger.chunks * read_fraction
        io_us = read_bytes / (hardware.ssd_gib_per_s * (1 << 30)) * 1e6
        h2d_us = read_bytes / (hardware.h2d_gib_per_s * (1 << 30)) * 1e6
        gpu_us = gpu_ops / (hardware.gpu_tflop_per_s * 1e12) * 1e6
        cpu_us = cpu_ops / (hardware.cpu_gop_per_s * 1e9) * 1e6
        sync_us = synchronizations * hardware.synchronization_us
        overlapped = max(io_us + h2d_us, gpu_us + cpu_us) + sync_us
        deadline = hardware.layer_deadline_us if scope == "layer" else hardware.layer_deadline_us * 42
        return {
            "family": family,
            "scope": scope,
            "mode": mode,
            "requests": requests,
            "cache_hit_rate_parameter": cache_hit_rate,
            "prediction_miss_probability_parameter": prediction_miss_probability,
            "initial_residual_fraction": initial_fraction,
            "effective_residual_fraction": read_fraction,
            "synthetic_exact_read_ratio": read_ratio,
            "synthetic_residual_nnz_fraction": residual_nnz_fraction,
            "read_bytes": read_bytes,
            "io_us": io_us,
            "h2d_us": h2d_us,
            "gpu_operations": gpu_ops,
            "gpu_us": gpu_us,
            "cpu_operations": cpu_ops,
            "cpu_us": cpu_us,
            "synchronization_us": sync_us,
            "overlapped_total_us": overlapped,
            "deadline_us_parameter": deadline,
            "deadline_met": overlapped <= deadline,
            "full_record_bytes_parameter": full_record_bytes,
            "predictor_rank_parameter": rank,
        }

    for scope, requests in (("layer", requests_per_layer), ("token", requests_per_token)):
        rows.append(make(scope, requests, "full-record", 1.0))
        rows.append(make(scope, requests, "predictor+all-residual", 1.0))
        rows.append(make(scope, requests, "predictor+causal-half", 0.5))
    return rows


def _proposed_trace_schema() -> dict[str, Any]:
    return {
        "status": "proposed pre-kernel schema; no real trace is included",
        "version": 1,
        "record_key": [
            "run_id",
            "generation_id",
            "token_index",
            "layer",
            "expert",
            "router_rank",
        ],
        "identity": [
            "model_sha256",
            "engine_commit",
            "config_sha256",
            "source_expert_sha256",
            "router_weight_f32_bits",
        ],
        "availability_timestamps_ns": [
            "token_start",
            "prior_logits_available",
            "hidden_summary_available",
            "route_selected",
            "read_issued",
            "host_read_complete",
            "h2d_complete",
            "predictor_start",
            "predictor_complete",
            "correction_start",
            "correction_complete",
            "layer_complete",
            "token_commit",
        ],
        "causal_context": [
            "prior_routes",
            "prior_logit_dump_join_key",
            "prior_top_ids_values",
            "prior_logit_margin_entropy",
            "expert_input_dump_join_key",
            "fixed_seed_hidden_summary",
            "cache_tier_slot_recency",
            "inflight_and_queue_state",
        ],
        "exact_targets": [
            "gate_preactivation_dump_join_key",
            "up_preactivation_dump_join_key",
            "swiglu_activation_dump_join_key",
            "down_output_dump_join_key",
            "weighted_expert_contribution_dump_join_key",
            "predictor_output_dump_join_key",
            "canonical_chunk_contribution_dump_join_keys",
            "layer_output_f32_hash",
            "target_logits_dump_join_key",
            "selected_token",
            "next_recurrent_state_hash",
        ],
        "cost": [
            "resident_predictor_bytes_by_tier",
            "ssd_requested_completed_padded_useful_bytes",
            "h2d_bytes",
            "cpu_decode_ops_and_ns",
            "gpu_predictor_residual_duplicate_ops_and_ns",
            "synchronization_count_and_ns",
            "deadline_ns",
            "deadline_slack_ns",
            "prediction_miss_rescue_crc_cache_displacement_flags",
        ],
    }


def _artifact_readme() -> str:
    return """# Causal expert residual synthetic artifact

This directory is generated by:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
  python3 tools/evaluate_causal_expert_residual.py
```

It contains no real GLM trace and makes no claim about real route or residual
correlation. All generator and cost parameters are recorded in
`configuration.json`.

- `representation-summary.csv`: exact `CER1` storage/read ledgers and byte
  round-trip status.
- `information-summary.csv`: held-out route information, conditional residual
  entropy, log-loss gain, and Bayes-MSE gain.
- `conditional-rate-distortion.csv`: conditional Blahut-Arimoto points.
- `rate-at-fixed-distortion.csv`: time-sharing interpolation at three normalized
  distortions.
- `stopping-policies.json`: exact small contextual prefix optima.
- `cache-optima.json`: exact exhaustive representation, basis, and residual-
  chunk tier-placement optima.
- `engine-predictor-residency.csv`: parametric all-layer predictor residency.
- `engine-parametric-cost.csv`: parametric engine-scale cost projection.
- `synthetic-system-cost.csv`: native synthetic-dimension cost projection.
- `trace-schema.json`: proposed causally timestamped real-trace fields.
- `containers/*.cer`: exact random-access sample residual files.
- `fp32-*-counterexample.json`: arithmetic-order counterexamples.
- `arbitrary-map-lower-bound.json`: finite arbitrary-map lower-bound instance.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/causal-expert-residual"),
    )
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--activation-states", type=int, default=8)
    parser.add_argument("--rd-coordinates", type=int, default=4)
    parser.add_argument("--quantization-levels", type=int, default=9)
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--full-record-bytes", type=int, default=(12 << 20) + (1536 << 10))
    parser.add_argument("--engine-predictor-rank", type=int, default=32)
    parser.add_argument("--cache-hit-rate", type=float, default=0.0)
    parser.add_argument("--prediction-miss-probability", type=float, default=0.10)
    parser.add_argument("--ssd-gib-per-s", type=float, default=4.0)
    parser.add_argument("--h2d-gib-per-s", type=float, default=24.0)
    parser.add_argument("--gpu-tflop-per-s", type=float, default=20.0)
    parser.add_argument("--cpu-gop-per-s", type=float, default=50.0)
    parser.add_argument("--synchronization-us", type=float, default=5.0)
    parser.add_argument("--layer-deadline-us", type=float, default=5000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    container_dir = args.output / "containers"
    container_dir.mkdir(exist_ok=True)
    parameters = SyntheticParameters(
        seed=args.seed,
        experts=args.experts,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        rank=args.rank,
        chunks=args.chunks,
        tokens=args.tokens,
        activation_states=args.activation_states,
        route_repeat_probability=0.75,
        route_logit_accuracy=0.90,
        logit_activation_accuracy=0.80,
        sparse_chunk_probability=0.35,
        sparse_entry_probability=0.12,
        residual_scale=0.08,
        drift_scale=0.003,
    )
    parameters.validate()
    if args.tokens % 2:
        raise ValueError("tokens must be even for the held-out split")
    split = args.tokens // 2
    train_slice = slice(0, split)
    test_slice = slice(split, args.tokens)
    hardware = HardwareCostModel(
        ssd_gib_per_s=args.ssd_gib_per_s,
        h2d_gib_per_s=args.h2d_gib_per_s,
        gpu_tflop_per_s=args.gpu_tflop_per_s,
        cpu_gop_per_s=args.cpu_gop_per_s,
        synchronization_us=args.synchronization_us,
        layer_deadline_us=args.layer_deadline_us,
    )

    representation_rows: list[dict[str, Any]] = []
    information_rows: list[dict[str, Any]] = []
    rd_rows: list[dict[str, Any]] = []
    synthetic_cost_rows: list[dict[str, Any]] = []
    engine_cost_rows: list[dict[str, Any]] = []
    stopping_results: list[dict[str, Any]] = []

    for family_index, family in enumerate(FAMILIES):
        family_parameters = SyntheticParameters(**{
            **asdict(parameters),
            "seed": parameters.seed + 100 * family_index,
            "logit_activation_accuracy": 0.0 if family == "route_only_adversary" else parameters.logit_activation_accuracy,
        })
        trace = make_synthetic_trace(family, family_parameters)
        container = encode_xor_residual_container(
            trace.weights,
            trace.predictor_weights,
            chunks=args.chunks,
            alignment=args.alignment,
        )
        decoded = decode_xor_residual_all(container, trace.predictor_weights)
        exact = bool(np.array_equal(decoded.view(np.uint32), trace.weights.view(np.uint32)))
        if not exact:
            raise AssertionError(f"lossless round trip failed for {family}")
        (container_dir / f"{family}.cer").write_bytes(container.blob)
        ledger = representation_ledger(trace, container)
        representation_rows.append(
            {
                "family": family,
                **asdict(ledger),
                "alignment": args.alignment,
                "zero_chunks": container.stats.zero_chunks,
                "sparse_chunks": container.stats.sparse_chunks,
                "raw_chunks": container.stats.raw_chunks,
                "exact_byte_roundtrip": exact,
            }
        )
        info, _, family_rd_rows = _information_rows(
            trace,
            train_slice,
            test_slice,
            args.rd_coordinates,
            args.quantization_levels,
        )
        information_rows.append(info)
        rd_rows.extend(family_rd_rows)
        stopping_results.append(
            _stopping_result(trace, container, train_slice, test_slice)
        )
        residual_nnz = float(
            np.count_nonzero(trace.weights != trace.predictor_weights) / trace.weights.shape[0]
        )
        baseline, scheme = predict_system_costs(
            ledger,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            rank=args.rank,
            residual_nnz_per_expert=residual_nnz,
            requests=336,
            cache_hit_rate=args.cache_hit_rate,
            residual_chunk_fraction=1.0,
            prediction_miss_probability=args.prediction_miss_probability,
            hardware=hardware,
        )
        synthetic_cost_rows.extend(
            [{"family": family, **asdict(cost)} for cost in (baseline, scheme)]
        )
        engine_cost_rows.extend(
            _engine_cost_rows(
                family,
                ledger,
                trace,
                rank=args.engine_predictor_rank,
                full_record_bytes=args.full_record_bytes,
                cache_hit_rate=args.cache_hit_rate,
                hardware=hardware,
                prediction_miss_probability=args.prediction_miss_probability,
            )
        )

    lower_bound = arbitrary_map_lower_bound(
        args.experts,
        args.input_dim,
        args.output_dim * 32,
        resident_bits=0,
    )
    counterexample = find_fp32_additive_counterexample(seed=args.seed)
    fma_counterexample = find_fp32_fma_additive_counterexample(seed=args.seed + 10)
    cache_results = _cache_results()
    residency_rows = _engine_residency_rows(
        ranks=(8, 16, 32, 64),
        bytes_per_value=(1, 2, 4),
        full_record_bytes=args.full_record_bytes,
    )

    configuration = {
        "artifact_kind": "synthetic-only; no real model trace",
        "synthetic_parameters": asdict(parameters),
        "alignment": args.alignment,
        "heldout_split": {"train_tokens": split, "test_tokens": args.tokens - split},
        "rate_distortion_probe_coordinates": args.rd_coordinates,
        "quantization_levels": args.quantization_levels,
        "hardware_cost_parameters": asdict(hardware),
        "engine_projection_parameters": {
            "full_record_bytes": args.full_record_bytes,
            "predictor_rank": args.engine_predictor_rank,
            "cache_hit_rate": args.cache_hit_rate,
            "prediction_miss_probability": args.prediction_miss_probability,
        },
    }
    _write_json(args.output / "configuration.json", configuration)
    _write_json(args.output / "trace-schema.json", _proposed_trace_schema())
    (args.output / "README.md").write_text(_artifact_readme())
    _write_json(args.output / "arbitrary-map-lower-bound.json", lower_bound)
    _write_json(args.output / "fp32-additive-counterexample.json", counterexample)
    _write_json(args.output / "fp32-fma-additive-counterexample.json", fma_counterexample)
    _write_json(args.output / "stopping-policies.json", stopping_results)
    _write_json(args.output / "cache-optima.json", cache_results)
    _write_csv(args.output / "representation-summary.csv", representation_rows)
    _write_csv(args.output / "information-summary.csv", information_rows)
    _write_csv(args.output / "conditional-rate-distortion.csv", rd_rows)
    _write_csv(
        args.output / "rate-at-fixed-distortion.csv", _fixed_distortion_rows(rd_rows)
    )
    _write_csv(args.output / "synthetic-system-cost.csv", synthetic_cost_rows)
    _write_csv(args.output / "engine-parametric-cost.csv", engine_cost_rows)
    _write_csv(args.output / "engine-predictor-residency.csv", residency_rows)
    print(json.dumps({
        "output": str(args.output),
        "families": list(FAMILIES),
        "representation_rows": len(representation_rows),
        "rate_distortion_rows": len(rd_rows),
        "cache_scenarios": len(cache_results),
        "all_exact_round_trips": all(row["exact_byte_roundtrip"] for row in representation_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
