#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from dflash_capture_observability import (
    CaptureCost,
    CaptureProblem,
    SelectionConstraints,
    adaptive_dinkelbach_select,
    adaptive_context_values,
    bundle_density_greedy,
    bundle_submodularity_ratio,
    canonical_subset,
    dinkelbach_select,
    exact_select,
    fano_block_bits,
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


class CaptureObservabilityTests(unittest.TestCase):
    def test_mutual_information_is_monotone(self) -> None:
        network = make_naive_bayes_network()
        values = {
            subset: network.mutual_information(subset)
            for size in range(len(network.layers) + 1)
            for subset in itertools.combinations(network.layers, size)
        }
        for subset, value in values.items():
            for layer in network.layers:
                if layer not in subset:
                    larger = canonical_subset((*subset, layer))
                    self.assertGreaterEqual(values[larger] + 1e-11, value)

    def test_conditional_independence_is_submodular(self) -> None:
        network = make_naive_bayes_network((0.08, 0.17, 0.29, 0.37))
        gap, _ = minimum_submodularity_gap(
            network.layers, network.mutual_information
        )
        self.assertGreaterEqual(gap, -1e-10)

    def test_xor_is_smallest_synergy_witness(self) -> None:
        network = make_xor_synergy_network()
        self.assertAlmostEqual(network.mutual_information((1,)), 0.0, places=11)
        self.assertAlmostEqual(network.mutual_information((2,)), 0.0, places=11)
        self.assertAlmostEqual(network.mutual_information((1, 2)), 1.0, places=11)
        self.assertAlmostEqual(network.submodularity_gap((), (1,), 2), -1.0)

    def test_deterministic_duplicate_adds_no_information(self) -> None:
        network = make_duplicate_network()
        single = network.mutual_information((1,))
        self.assertAlmostEqual(single, network.mutual_information((2,)))
        self.assertAlmostEqual(single, network.mutual_information((1, 2)))
        gap, _ = minimum_submodularity_gap(
            network.layers, network.mutual_information
        )
        self.assertGreaterEqual(gap, -1e-10)

    def test_pair_bundle_recovers_xor_optimum(self) -> None:
        network = make_xor_synergy_network()
        problem = make_small_problem_from_network(
            "xor", network, max_items=2
        )
        greedy = greedy_select(problem, problem.information)
        pair = bundle_density_greedy(problem, problem.information)
        optimum = exact_select(problem, problem.information)
        self.assertEqual(greedy.selected, (3, 4))
        self.assertEqual(pair.selected, (1, 2))
        self.assertEqual(pair.selected, optimum.selected)
        self.assertLess(greedy.objective_value, optimum.objective_value)

    def test_bundle_ratio_detects_interaction_order(self) -> None:
        network = make_xor_synergy_network()
        gamma_one = bundle_submodularity_ratio(
            network.layers,
            network.mutual_information,
            max_items=2,
            max_bundle_size=1,
        )
        gamma_two = bundle_submodularity_ratio(
            network.layers,
            network.mutual_information,
            max_items=2,
            max_bundle_size=2,
        )
        self.assertAlmostEqual(gamma_one, 0.0)
        self.assertAlmostEqual(gamma_two, 1.0)

    def test_nonuniform_hard_costs_are_enforced(self) -> None:
        layers = (1, 2, 3)
        costs = {
            1: CaptureCost(10, 5.0),
            2: CaptureCost(10, 1.0),
            3: CaptureCost(10, 1.0),
        }
        value_map = {
            (): 0.0,
            (1,): 10.0,
            (2,): 4.0,
            (3,): 3.0,
            (1, 2): 12.0,
            (1, 3): 11.0,
            (2, 3): 7.0,
            (1, 2, 3): 13.0,
        }
        value = lambda s: value_map[canonical_subset(s)]
        problem = CaptureProblem(
            "hard_cost",
            layers,
            costs,
            SelectionConstraints(max_items=2, max_latency_ms=2.0),
            value,
            lambda s: 1 + value(s),
            lambda s: 1 + sum(costs[x].latency_ms for x in s),
            bandwidth_bytes_per_ms=1000,
            current_layers=(),
        )
        optimum = exact_select(problem, value)
        self.assertEqual(optimum.selected, (2, 3))
        self.assertTrue(problem.feasible(optimum.selected))
        self.assertFalse(problem.feasible((1,)))

    def test_dinkelbach_exact_matches_direct_ratio(self) -> None:
        network = make_naive_bayes_network((0.04, 0.18, 0.24, 0.31))
        problem = make_small_problem_from_network(
            "ratio", network, max_items=3
        )
        direct = exact_select(
            problem,
            lambda s: problem.accepted_tokens(s) / problem.round_time_ms(s),
        )
        dinkelbach = dinkelbach_select(
            problem, lambda value: exact_select(problem, value)
        )
        self.assertEqual(direct.selected, dinkelbach.selected)
        self.assertAlmostEqual(
            direct.objective_value, dinkelbach.throughput_tokens_per_ms, places=12
        )
        self.assertLessEqual(abs(dinkelbach.transformed_residual), 1e-9)

    def test_information_can_lose_throughput(self) -> None:
        network = make_naive_bayes_network((0.01, 0.16, 0.19, 0.23, 0.27))
        problem = make_small_problem_from_network(
            "expensive", network, max_items=2, expensive_layer=1
        )
        information_greedy = greedy_select(problem, problem.information)
        throughput_optimum = exact_select(
            problem,
            lambda s: problem.accepted_tokens(s) / problem.round_time_ms(s),
        )
        self.assertIn(1, information_greedy.selected)
        self.assertNotIn(1, throughput_optimum.selected)
        self.assertGreater(
            throughput_optimum.objective_value,
            problem.accepted_tokens(information_greedy.selected)
            / problem.round_time_ms(information_greedy.selected),
        )

    def test_l45_expensive_strongest_layer_is_rejected_by_ratio(self) -> None:
        problem, _ = make_l45_capture_problem("expensive_late")
        strongest = max(
            problem.layers, key=lambda layer: problem.information((layer,))
        )
        self.assertEqual(strongest, 42)
        information_greedy = greedy_select(problem, problem.information)
        self.assertIn(42, information_greedy.selected)
        throughput = dinkelbach_select(
            problem,
            lambda value: bundle_density_greedy(problem, value),
        )
        self.assertNotIn(42, throughput.selected)
        self.assertLessEqual(len(throughput.selected), 5)

    def test_l45_near_duplicate_pair_has_tiny_conditional_gain(self) -> None:
        problem, _ = make_l45_capture_problem("near_duplicate")
        first = problem.information((23,))
        pair = problem.information((23, 24))
        self.assertGreater(first, 1.0)
        self.assertLess(pair - first, 0.02)

    def test_current_l45_locations_are_evaluated_verbatim(self) -> None:
        problem, _ = make_l45_capture_problem("change_point")
        self.assertEqual(problem.current_subset(), (5, 14, 24, 33, 42))
        self.assertTrue(problem.feasible(problem.current_subset()))

    def test_adaptivity_can_be_arbitrarily_better(self) -> None:
        actions = ((1, 2, 3, 4, 5), (6, 7, 8, 9, 10))
        probabilities = {"a": 0.5, "b": 0.5}
        mismatch = 1000.0
        accepted = {}
        times = {}
        for context, matched in zip(probabilities, actions):
            for action in actions:
                accepted[(context, action)] = float(action == matched)
                times[(context, action)] = 1.0 if action == matched else mismatch
        result = adaptive_context_values(probabilities, actions, accepted, times)
        self.assertAlmostEqual(result["adaptivity_ratio"], mismatch + 1)

    def test_common_argmax_means_no_adaptive_value(self) -> None:
        actions = ((1,), (2,))
        probabilities = {"a": 0.25, "b": 0.75}
        accepted = {}
        times = {}
        for context in probabilities:
            accepted[(context, (1,))] = 2.0
            times[(context, (1,))] = 2.0
            accepted[(context, (2,))] = 1.0
            times[(context, (2,))] = 2.0
        result = adaptive_context_values(probabilities, actions, accepted, times)
        self.assertAlmostEqual(result["adaptivity_ratio"], 1.0)
        self.assertEqual(result["fixed_action"], [1])

    def test_adaptive_dinkelbach_uses_one_global_ratio(self) -> None:
        layers = (1, 2)
        costs = {layer: CaptureCost(1.0, 0.0) for layer in layers}
        constraints = SelectionConstraints(max_items=1)
        tables = {
            "a": {
                (): (0.0, 1.0),
                (1,): (0.1, 0.01),
                (2,): (8.0, 1.0),
            },
            "b": {
                (): (0.0, 100.0),
                (1,): (0.0, 100.0),
                (2,): (0.0, 100.0),
            },
        }
        problems = {}
        for context, table in tables.items():
            problems[context] = CaptureProblem(
                context,
                layers,
                costs,
                constraints,
                lambda s, t=table: t[canonical_subset(s)][0],
                lambda s, t=table: t[canonical_subset(s)][0],
                lambda s, t=table: t[canonical_subset(s)][1],
                bandwidth_bytes_per_ms=1_000.0,
                current_layers=(),
            )
        probabilities = {"a": 0.5, "b": 0.5}
        adaptive = adaptive_dinkelbach_select(
            probabilities,
            problems,
            lambda _context, problem, value: exact_select(problem, value),
        )
        candidates = (
            (
                sum(probabilities[x] * tables[x][policy[x]][0] for x in probabilities)
                / sum(probabilities[x] * tables[x][policy[x]][1] for x in probabilities),
                policy,
            )
            for policy in (
                {"a": a, "b": b}
                for a in ((), (1,), (2,))
                for b in ((), (1,), (2,))
            )
        )
        brute_force = max(candidates, key=lambda item: item[0])
        self.assertAlmostEqual(adaptive.throughput_tokens_per_ms, brute_force[0])
        self.assertEqual(dict(adaptive.policy), brute_force[1])
        local_ratio_choice = exact_select(
            problems["a"],
            lambda s: problems["a"].accepted_tokens(s) / problems["a"].round_time_ms(s),
        )
        self.assertEqual(local_ratio_choice.selected, (1,))
        self.assertEqual(adaptive.policy["a"], (2,))

    def test_fano_bounds(self) -> None:
        exact = fano_block_bits(8.0, 256, 0.0)
        loose = fano_block_bits(8.0, 256, 0.20)
        self.assertAlmostEqual(exact, 8.0)
        self.assertGreater(loose, 0.0)
        self.assertLess(loose, exact)
        self.assertAlmostEqual(fano_block_bits(1.0, 2, 1.0), 0.0)
        prefix = prefix_acceptance_fano_bits(
            block_length=8, expected_prefix=6.5, alphabet_size=2
        )
        self.assertGreater(prefix, 0.0)
        self.assertLessEqual(prefix, 8.0)

    def test_kill_criterion(self) -> None:
        net_upper = kill_criterion_net_upper_bound(
            accepted_gain_upper=0.08,
            added_time_lower_ms=2.0,
            baseline_throughput_lower_tokens_per_ms=0.05,
        )
        self.assertLess(net_upper, 0.0)

    def test_trace_schema_accepts_minimum_record(self) -> None:
        schema_path = Path(__file__).with_name("schemas") / "dflash_capture_trace.schema.json"
        schema = json.loads(schema_path.read_text())
        record = {
            "schema_version": "dflash-capture-v1",
            "request_id": "r",
            "round_id": 0,
            "policy": {
                "policy_id": "explore",
                "policy_version": "1",
                "exploration_cohort": "pilot",
                "joint_action_probability": 0.1
            },
            "cheap_context": {
                "feature_schema_id": "cheap-v1",
                "available_before_selection_ns": 1,
                "values": [0.2, 3.0]
            },
            "conditioning_state": {
                "representation_schema_id": "drafter-state-v1",
                "available_before_selection_ns": 1,
                "values_or_reference": "sha256:example",
                "drafter_checkpoint_id": "dflash2-test"
            },
            "candidate_layers": [5, 14, 24, 33, 42],
            "selected_captures": [{
                "selection_order": 0,
                "layer": 5,
                "conditional_selection_probability": 0.5,
                "encoder_id": "rank16",
                "encoded_bits": 256,
                "logical_bytes": 8192,
                "transferred_bytes": 32,
                "capture_ready_ns": 2,
                "transfer_start_ns": 3,
                "transfer_end_ns": 4,
                "critical_path_latency_ns": 2,
                "observation": [0.1, -0.2]
            }],
            "draft": {
                "block_token_ids": [10, 11],
                "block_width": 2,
                "drafter_start_ns": 5,
                "drafter_end_ns": 6
            },
            "target_outcome": {
                "exact_block_token_ids": [10, 12],
                "verification_prefix_length": 1,
                "accepted_draft_tokens": 1,
                "committed_tokens": 1,
                "fallback_used": False
            },
            "timing": {
                "round_start_ns": 1,
                "round_end_ns": 10,
                "verify_start_ns": 7,
                "verify_end_ns": 9,
                "capture_critical_path_ns": 2,
                "fallback_ns": 0
            }
        }
        errors = list(Draft202012Validator(schema).iter_errors(record))
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
