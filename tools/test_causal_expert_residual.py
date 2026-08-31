#!/usr/bin/env python3
"""Hardware-free tests for causal expert predictor/residual coding."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from causal_expert_residual import (  # noqa: E402
    BasisPlacement,
    CacheItem,
    CacheOption,
    CacheProblem,
    ChunkPlacement,
    ResidualChunkObject,
    PrefixActionTable,
    SyntheticParameters,
    additive_correction_f32_dot,
    arbitrary_map_lower_bound,
    canonical_f32_dot,
    canonical_f32_fma_dot,
    conditional_mutual_information_bits,
    conditional_rate_distortion_curve,
    decode_xor_residual_all,
    decode_xor_residual_chunk,
    demo_cache_problem,
    encode_xor_residual_container,
    entropy_bits,
    f32_bits,
    find_fp32_additive_counterexample,
    find_fp32_fma_additive_counterexample,
    make_synthetic_trace,
    mutual_information_bits,
    prefix_then_continue_f32_dot,
    prefix_then_continue_f32_fma_dot,
    quantize_symmetric,
    representation_ledger,
    solve_prefix_policy,
    solve_small_cache_problem,
)


class NoFreeLunchTests(unittest.TestCase):
    def test_counting_and_exhaustive_query_bound(self) -> None:
        bound = arbitrary_map_lower_bound(8, 32, 16, resident_bits=256)
        self.assertEqual(bound.independent_cells, 256)
        self.assertEqual(bound.total_representation_bits, 4096)
        self.assertEqual(bound.external_bits_for_exhaustive_queries, 3840)
        self.assertEqual(bound.mean_external_bits_per_query, 15.0)

    def test_entropy_helpers(self) -> None:
        left = np.tile([0, 0, 1, 1], 16)
        right = np.tile([0, 1, 0, 1], 16)
        self.assertAlmostEqual(entropy_bits(left), 1.0, places=12)
        self.assertAlmostEqual(mutual_information_bits(left, right), 0.0, places=12)
        self.assertAlmostEqual(
            conditional_mutual_information_bits(left, left, right), 1.0, places=12
        )


class SyntheticTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = SyntheticParameters(
            seed=91,
            experts=4,
            input_dim=12,
            output_dim=8,
            rank=3,
            chunks=2,
            tokens=1024,
            activation_states=4,
            route_logit_accuracy=0.9,
            logit_activation_accuracy=0.9,
        )

    def test_exact_shared_basis_has_zero_residual(self) -> None:
        trace = make_synthetic_trace("exact_shared_basis", self.parameters)
        self.assertTrue(
            np.array_equal(trace.weights.view(np.uint32), trace.predictor_weights.view(np.uint32))
        )
        self.assertTrue(np.all(trace.residual_outputs == 0.0))

    def test_adversarial_context_predicts_routes_not_residuals(self) -> None:
        trace = make_synthetic_trace("route_only_adversary", self.parameters)
        quantized = quantize_symmetric(trace.residual_outputs, levels=7)
        residual = quantized.symbols.reshape(-1)
        route = np.repeat(trace.routes, self.parameters.output_dim)
        top = np.repeat(trace.previous_top_route, self.parameters.output_dim)
        self.assertGreater(mutual_information_bits(route, top), 1.0)
        self.assertAlmostEqual(mutual_information_bits(residual, route), 0.0, places=12)
        self.assertAlmostEqual(mutual_information_bits(residual, top), 0.0, places=12)
        self.assertAlmostEqual(
            conditional_mutual_information_bits(residual, top, route), 0.0, places=12
        )

    def test_declared_logit_hint_can_have_incremental_information(self) -> None:
        parameters = SyntheticParameters(**{
            **self.parameters.__dict__,
            "route_logit_accuracy": 1.0,
            "logit_activation_accuracy": 1.0,
        })
        trace = make_synthetic_trace("independent_random", parameters)
        quantized = quantize_symmetric(trace.residual_outputs, levels=7)
        residual = quantized.symbols.reshape(-1)
        route = np.repeat(trace.routes, parameters.output_dim)
        margin = np.repeat(trace.previous_margin_bin, parameters.output_dim)
        self.assertGreater(
            conditional_mutual_information_bits(residual, margin, route), 0.02
        )


class ResidualContainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = SyntheticParameters(
            seed=7,
            experts=4,
            input_dim=16,
            output_dim=16,
            rank=4,
            chunks=4,
            tokens=256,
            activation_states=4,
        )

    def test_exact_round_trip_and_random_access(self) -> None:
        trace = make_synthetic_trace("shared_basis_sparse_residual", self.parameters)
        first = encode_xor_residual_container(
            trace.weights, trace.predictor_weights, chunks=4, alignment=64
        )
        second = encode_xor_residual_container(
            trace.weights, trace.predictor_weights, chunks=4, alignment=64
        )
        self.assertEqual(first.blob, second.blob)
        decoded = decode_xor_residual_all(first, trace.predictor_weights)
        self.assertTrue(np.array_equal(decoded.view(np.uint32), trace.weights.view(np.uint32)))
        rows_per_chunk = self.parameters.output_dim // self.parameters.chunks
        chunk = decode_xor_residual_chunk(
            first,
            trace.predictor_weights[2, rows_per_chunk : 2 * rows_per_chunk],
            2,
            1,
        )
        expected = trace.weights[2, rows_per_chunk : 2 * rows_per_chunk]
        self.assertTrue(np.array_equal(chunk.view(np.uint32), expected.view(np.uint32)))

    def test_exact_basis_needs_no_payload(self) -> None:
        trace = make_synthetic_trace("exact_shared_basis", self.parameters)
        container = encode_xor_residual_container(
            trace.weights, trace.predictor_weights, chunks=4, alignment=64
        )
        ledger = representation_ledger(trace, container)
        self.assertEqual(container.stats.zero_chunks, 16)
        self.assertEqual(container.stats.payload_extent_bytes, 0)
        self.assertEqual(ledger.mean_read_ratio, 0.0)

    def test_corruption_is_detected_by_chunk_crc(self) -> None:
        trace = make_synthetic_trace("independent_random", self.parameters)
        container = encode_xor_residual_container(
            trace.weights, trace.predictor_weights, chunks=4, alignment=64
        )
        descriptor = next(d for d in container.descriptors if d.stored_bytes)
        damaged = bytearray(container.blob)
        damaged[descriptor.offset] ^= 0x01
        with self.assertRaisesRegex(ValueError, "CRC"):
            decode_xor_residual_all(bytes(damaged), trace.predictor_weights)


class RateDistortionTests(unittest.TestCase):
    def test_lossless_endpoint_and_perfect_side_information(self) -> None:
        symbols = np.tile(np.array([0, 1], dtype=np.int64), 128)
        centers = np.array([0.0, 1.0])
        none = np.zeros_like(symbols)
        curve_none = conditional_rate_distortion_curve(
            symbols, none, centers, betas=(0.0, 1.0, 10.0)
        )
        curve_side = conditional_rate_distortion_curve(
            symbols, symbols, centers, betas=(0.0, 1.0, 10.0)
        )
        lossless_none = min(curve_none, key=lambda point: point.distortion)
        lossless_side = min(curve_side, key=lambda point: point.distortion)
        zero_rate_none = max(curve_none, key=lambda point: point.distortion)
        self.assertAlmostEqual(lossless_none.rate_bits, 1.0, places=9)
        self.assertAlmostEqual(lossless_side.rate_bits, 0.0, places=9)
        self.assertAlmostEqual(zero_rate_none.distortion, 0.5, places=9)


class PrefixStoppingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = PrefixActionTable(
            context_probability=np.array([1.0]),
            cumulative_cost=np.array([[0.0, 1.0, 3.0]]),
            expected_distortion=np.array([[4.0, 1.0, 0.0]]),
            tail_risk=np.array([[0.8, 0.2, 0.0]]),
        )

    def test_deterministic_expected_distortion_solution(self) -> None:
        solution = solve_prefix_policy(
            self.table, mode="expected_distortion", bound=1.0, allow_randomization=False
        )
        self.assertAlmostEqual(solution.expected_cost, 1.0)
        self.assertEqual(solution.components[0][1].prefixes, (1,))

    def test_two_policy_randomization_is_exact(self) -> None:
        solution = solve_prefix_policy(
            self.table, mode="expected_distortion", bound=0.5, allow_randomization=True
        )
        self.assertAlmostEqual(solution.expected_cost, 2.0, places=9)
        self.assertEqual(len(solution.components), 2)
        self.assertAlmostEqual(sum(weight for weight, _ in solution.components), 1.0)

    def test_selective_risk_linearization(self) -> None:
        table = PrefixActionTable(
            context_probability=np.array([1.0]),
            cumulative_cost=np.array([[0.0, 1.0, 3.0]]),
            expected_distortion=np.array([[4.0, 1.0, 0.0]]),
            tail_risk=np.array([[0.05, 0.2, 0.0]]),
        )
        solution = solve_prefix_policy(
            table,
            mode="selective_risk",
            bound=0.0,
            selective_alpha=0.1,
            allow_randomization=False,
        )
        self.assertEqual(solution.components[0][1].prefixes, (0,))


class CacheTests(unittest.TestCase):
    def test_demo_problem_is_solved_by_exhaustive_enumeration(self) -> None:
        problem = demo_cache_problem(ram_capacity=18, vram_capacity=10, max_distortion=0.03)
        solution = solve_small_cache_problem(problem)
        self.assertEqual(solution.enumerated_configurations, 6591)
        self.assertLessEqual(solution.ram_bytes, problem.ram_capacity)
        self.assertLessEqual(solution.vram_bytes, problem.vram_capacity)
        self.assertLessEqual(solution.expected_distortion, problem.max_expected_distortion)
        self.assertAlmostEqual(solution.expected_cost, 0.5654, places=12)
        self.assertTrue(solution.chunk_choices)

    def test_residual_chunk_competes_for_vram_bytes(self) -> None:
        chunk = ResidualChunkObject(
            "c0",
            1.0,
            (
                ChunkPlacement("disk", 0, 0, 9.0),
                ChunkPlacement("vram", 0, 1, 0.0),
            ),
        )
        problem = CacheProblem(
            items=(
                CacheItem(
                    "e0",
                    1.0,
                    (
                        CacheOption("full", 0, 3, 4.0),
                        CacheOption(
                            "residual",
                            0,
                            0,
                            0.5,
                            requires_basis=True,
                            residual_chunks=(chunk,),
                        ),
                    ),
                ),
            ),
            basis_placements=(
                BasisPlacement("none", 0, 0),
                BasisPlacement("vram-basis", 0, 2),
            ),
            ram_capacity=0,
            vram_capacity=3,
        )
        solution = solve_small_cache_problem(problem)
        self.assertEqual(solution.basis, "vram-basis")
        self.assertEqual(solution.choices, (("e0", "residual"),))
        self.assertEqual(solution.chunk_choices, (("e0", "c0", "vram"),))
        self.assertEqual(solution.vram_bytes, 3)
        self.assertAlmostEqual(solution.expected_cost, 0.5, places=12)

    def test_basis_dependency_is_enforced(self) -> None:
        problem = CacheProblem(
            items=(
                CacheItem(
                    "e0",
                    1.0,
                    (
                        CacheOption("full", 0, 4, 5.0),
                        CacheOption("residual", 0, 1, 1.0, requires_basis=True),
                    ),
                ),
            ),
            basis_placements=(
                BasisPlacement("none", 0, 0),
                BasisPlacement("vram-basis", 0, 2),
            ),
            ram_capacity=0,
            vram_capacity=3,
        )
        solution = solve_small_cache_problem(problem)
        self.assertEqual(solution.basis, "vram-basis")
        self.assertEqual(solution.choices, (("e0", "residual"),))


class FP32OrderTests(unittest.TestCase):
    def test_prefix_continuation_preserves_every_accumulation_bit(self) -> None:
        rng = np.random.default_rng(17)
        weights = rng.normal(size=32).astype(np.float32)
        activation = rng.normal(size=32).astype(np.float32)
        reference = canonical_f32_dot(weights, activation)
        for split in range(33):
            continued = prefix_then_continue_f32_dot(weights, activation, split)
            self.assertEqual(f32_bits(reference), f32_bits(continued))

    def test_fma_prefix_continuation_preserves_every_accumulation_bit(self) -> None:
        rng = np.random.default_rng(23)
        weights = rng.normal(size=32).astype(np.float32)
        activation = rng.normal(size=32).astype(np.float32)
        reference = canonical_f32_fma_dot(weights, activation)
        for split in range(33):
            continued = prefix_then_continue_f32_fma_dot(weights, activation, split)
            self.assertEqual(f32_bits(reference), f32_bits(continued))

    def test_separately_reduced_fma_correction_changes_rounding(self) -> None:
        counterexample = find_fp32_fma_additive_counterexample(
            seed=19, width=8, attempts=10_000
        )
        self.assertNotEqual(counterexample.canonical_bits, counterexample.additive_bits)

    def test_separately_reduced_additive_correction_changes_rounding(self) -> None:
        counterexample = find_fp32_additive_counterexample(seed=9, width=8, attempts=10_000)
        self.assertNotEqual(counterexample.canonical_bits, counterexample.additive_bits)
        predictor = np.asarray(counterexample.predictor_weights, dtype=np.float32)
        residual = np.asarray(counterexample.residual_weights, dtype=np.float32)
        activation = np.asarray(counterexample.activation, dtype=np.float32)
        result = additive_correction_f32_dot(predictor, residual, activation)
        self.assertEqual(f32_bits(result), counterexample.additive_bits)


if __name__ == "__main__":
    unittest.main()
