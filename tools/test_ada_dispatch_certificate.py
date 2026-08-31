#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import ada_dispatch_certificate as adc


class OccupancyTests(unittest.TestCase):
    def test_lut_cta4_and_cta8_both_reach_full_resident_warps(self) -> None:
        cta4 = adc.calculate_occupancy(
            adc.KernelMetadata("cta4", 128, 40, 2048, block_barriers=1)
        )
        cta8 = adc.calculate_occupancy(
            adc.KernelMetadata("cta8", 256, 40, 2048, block_barriers=1)
        )
        self.assertTrue(cta4.feasible)
        self.assertTrue(cta8.feasible)
        self.assertEqual(cta4.resident_blocks_per_sm, 12)
        self.assertEqual(cta8.resident_blocks_per_sm, 6)
        self.assertEqual(cta4.resident_warps_per_sm, 48)
        self.assertEqual(cta8.resident_warps_per_sm, 48)
        self.assertAlmostEqual(cta4.theoretical_occupancy, 1.0)
        self.assertAlmostEqual(cta8.theoretical_occupancy, 1.0)
        self.assertIsNone(cta4.achieved_occupancy)

    def test_real_tensor_and_imma_resource_points(self) -> None:
        tensor = adc.calculate_occupancy(
            adc.KernelMetadata("tc", 256, 48, 46_080, block_barriers=1)
        )
        imma = adc.calculate_occupancy(
            adc.KernelMetadata("imma", 256, 62, 4096, block_barriers=1)
        )
        self.assertEqual(tensor.resident_blocks_per_sm, 2)
        self.assertEqual(tensor.limiting_resources, ("shared_memory",))
        self.assertAlmostEqual(tensor.theoretical_occupancy, 16 / 48)
        self.assertEqual(imma.allocated_registers_per_warp, 2048)
        self.assertEqual(imma.allocated_registers_per_block, 16_384)
        self.assertEqual(imma.resident_blocks_per_sm, 4)
        self.assertAlmostEqual(imma.theoretical_occupancy, 32 / 48)

    def test_unknown_barrier_is_exposed_and_runtime_match_closes_it(self) -> None:
        metadata = adc.KernelMetadata(
            "barrier_kernel", 128, 40, 2048, block_barriers=1
        )
        conditional = adc.calculate_occupancy(metadata)
        self.assertEqual(
            conditional.resource_model_status, "conditional_barrier_capacity_unknown"
        )
        self.assertEqual(conditional.unmodeled_resource_limits, ("barriers",))

        exact = adc.calculate_occupancy(
            metadata,
            assumptions=adc.AllocationAssumptions(block_barriers_per_sm=16),
        )
        self.assertEqual(exact.resource_model_status, "exact_under_recorded_microconstants")
        self.assertEqual(exact.resident_blocks_per_sm, 12)

        validation = adc.validate_runtime_residency(metadata, 12)
        self.assertTrue(validation["match"])
        self.assertTrue(validation["certificate_eligible"])
        self.assertEqual(validation["status"], "validated_against_cuda_runtime")
        mismatch = adc.validate_runtime_residency(metadata, 11)
        self.assertFalse(mismatch["certificate_eligible"])

    def test_grid_underfill_is_not_achieved_occupancy(self) -> None:
        result = adc.calculate_occupancy(
            adc.KernelMetadata(
                "one-block-per-sm-grid", 128, 40, 2048, grid_blocks=56, sm_count=56
            )
        )
        self.assertAlmostEqual(result.first_wave_occupancy_upper_bound or 0.0, 4 / 48)
        self.assertAlmostEqual(result.tail_wave_occupancy_upper_bound or 0.0, 4 / 48)
        self.assertIsNone(result.achieved_occupancy)

    def test_infeasible_kernel(self) -> None:
        result = adc.calculate_occupancy(
            adc.KernelMetadata("bad", 2048, 300, 200_000)
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.resident_blocks_per_sm, 0)

    def test_ptxas_parser(self) -> None:
        text = """
ptxas info    : Compiling entry function '_Z6kernelv' for 'sm_89'
ptxas info    : Function properties for _Z6kernelv
    16 bytes stack frame, 8 bytes spill stores, 12 bytes spill loads
ptxas info    : Used 62 registers, 4096 bytes smem, 392 bytes cmem[0]
"""
        parsed = adc.parse_ptxas_verbose(text)
        self.assertEqual(parsed["_Z6kernelv"]["registers_per_thread"], 62)
        self.assertEqual(parsed["_Z6kernelv"]["static_shared_bytes"], 4096)
        self.assertEqual(parsed["_Z6kernelv"]["stack_bytes"], 16)
        self.assertEqual(parsed["_Z6kernelv"]["spill_store_bytes"], 8)
        self.assertEqual(parsed["_Z6kernelv"]["spill_load_bytes"], 12)


class AnalyticalModelTests(unittest.TestCase):
    def test_roofline_cannot_order_dependency_counterexample(self) -> None:
        example = adc.dependency_mlp_counterexample()
        counts = adc.RooflineCounts(**example["counts"])
        bound_a = adc.roofline_lower_bound_us(counts, 1e14, 5e11)
        bound_b = adc.roofline_lower_bound_us(counts, 1e14, 5e11)
        self.assertEqual(bound_a, bound_b)
        self.assertEqual(example["cold_winner"], "memory_parallel")
        self.assertEqual(example["hot_winner"], "compute_parallel")

    def test_analytically_favored_tablefree_kernel_is_measured_slower(self) -> None:
        example = adc.analytically_favored_but_slower_example()
        self.assertEqual(example["analytical_preference"], "tablefree")
        self.assertEqual(example["measured_winner"], "lut_dp4a")
        self.assertGreater(example["tablefree_slowdown_fraction"], 0.25)

    def test_resource_favorite_is_reliably_rejected_under_heavy_tailed_noise(self) -> None:
        stress = adc.evaluate_analytical_reversal_stress(trials=20, seed=900)
        self.assertEqual(stress["analytical_favorite"], "cta8")
        self.assertEqual(stress["true_winner"], "cta4")
        self.assertEqual(stress["empirical_family_misselection_probability"], 0.0)
        self.assertEqual(stress["empirical_epsilon_violation_probability"], 0.0)
        self.assertEqual(stress["direct_certificate_completion_probability"], 1.0)


class SequentialExperimentTests(unittest.TestCase):
    def test_anytime_radius_decreases_and_direct_decision_certifies(self) -> None:
        alpha = 0.05 / 24
        r100 = adc.anytime_hoeffding_radius(100, 0.75, alpha)
        r1000 = adc.anytime_hoeffding_radius(1000, 0.75, alpha)
        self.assertGreater(r100, r1000)
        samples = adc.StateSamples([-0.60] * 1000)
        design = adc.SequentialDesign(
            alpha=0.05,
            epsilon_us=0.10,
            contrast_bound_us=0.75,
            min_blocks_per_state=2,
        )
        decision, interval, reason = adc.direct_decision(samples, design, alpha)
        self.assertEqual(decision, 8)
        self.assertLessEqual(interval[1], design.epsilon_us)
        self.assertEqual(reason, "cta8_certified")

    def test_model_misspecification_is_visible(self) -> None:
        gaps = adc.synthetic_gap_scenario("misspecified")
        design = adc.SequentialDesign(
            alpha=0.05,
            epsilon_us=0.12,
            contrast_bound_us=0.75,
            max_blocks=4500,
            min_blocks_per_state=3,
        )
        direct = adc.run_structured_bai(gaps, "independent", design, seed=19)
        low_rank = adc.run_structured_bai(gaps, "low_rank", design, seed=19)
        self.assertTrue(direct.completed_direct_certificate)
        self.assertFalse(direct.any_epsilon_violation)
        self.assertFalse(low_rank.completed_direct_certificate)
        self.assertGreaterEqual(
            low_rank.mean_simple_regret_us, direct.mean_simple_regret_us
        )


class RobustDispatchTests(unittest.TestCase):
    def test_minimax_and_distribution_shift(self) -> None:
        local = {
            "cta4": adc.LatencyInterval(10.0, 10.2),
            "cta8": adc.LatencyInterval(9.7, 10.0),
        }
        remote = {
            "cta4": adc.LatencyInterval(9.8, 10.1),
            "cta8": adc.LatencyInterval(10.0, 10.4),
        }
        local_choice, _ = adc.minimax_kernel(local)
        remote_choice, _ = adc.minimax_kernel(remote)
        self.assertEqual(local_choice, "cta8")
        self.assertEqual(remote_choice, "cta4")
        portable, objective = adc.portable_minimax_kernel(
            {"local": local, "remote": remote},
            nominal_machine_probabilities={"local": 0.5, "remote": 0.5},
            distribution_l1_radius=2.0,
        )
        self.assertIn(portable, {"cta4", "cta8"})
        self.assertEqual(set(objective), {"cta4", "cta8"})
        self.assertEqual(
            adc.interval_regret_upper("only", {"only": adc.LatencyInterval(1.0, 2.0)}),
            0.0,
        )

    def test_separate_table_roi_rule(self) -> None:
        lower_saving = adc.latency_saving_lower_bound(
            adc.LatencyInterval(10.30, 10.45),
            adc.LatencyInterval(9.85, 10.00),
        )
        self.assertAlmostEqual(lower_saving, 0.30)
        result = adc.separate_tables_dominate(
            certified_saving_lower_us_per_call=lower_saving,
            expected_calls=10_000_000,
            extra_tuning_cost_us=1_000_000,
            extra_code_and_icache_cost_us=500_000,
        )
        self.assertTrue(result["separate_tables_dominate"])
        self.assertGreater(result["margin_us"], 0)


class CompileSearchTests(unittest.TestCase):
    def test_finite_compile_search_classifies_every_candidate(self) -> None:
        candidates = (
            adc.CompileCandidate(
                "incumbent",
                9.0,
                100,
                metadata=adc.KernelMetadata("incumbent", 256, 40, 2048),
                measured_interval=adc.LatencyInterval(10.0, 10.2),
            ),
            adc.CompileCandidate("static_loser", 10.3, 100),
            adc.CompileCandidate(
                "spill",
                8.0,
                100,
                metadata=adc.KernelMetadata(
                    "spill", 256, 80, 0, spill_load_bytes=16, spill_store_bytes=16
                ),
            ),
        )
        cert = adc.compile_measure_stopping_certificate(
            candidates,
            incumbent="incumbent",
            incumbent_upper_us=10.2,
            epsilon_us=0.1,
            expected_lifetime_calls=1000,
            remaining_measurement_cost_us=100,
        )
        self.assertTrue(cert["complete"])
        self.assertEqual(cert["classifications"]["static_loser"], "static_lower_bound_eliminated")
        self.assertEqual(cert["classifications"]["spill"], "spill_gate_rejected")

    def test_uncompiled_candidate_can_be_abandoned_by_candidate_specific_roi(self) -> None:
        candidates = (
            adc.CompileCandidate(
                "incumbent",
                9.0,
                0,
                metadata=adc.KernelMetadata("incumbent", 256, 40, 2048),
                measured_interval=adc.LatencyInterval(10.0, 10.2),
            ),
            adc.CompileCandidate("optimistic", 9.9, 50_000),
        )
        cert = adc.compile_measure_stopping_certificate(
            candidates,
            incumbent="incumbent",
            incumbent_upper_us=10.2,
            epsilon_us=0.1,
            expected_lifetime_calls=1000,
            remaining_measurement_cost_us=1_000,
        )
        self.assertTrue(cert["complete"])
        self.assertEqual(cert["classifications"]["optimistic"], "roi_abandoned")
        self.assertEqual(cert["candidate_remaining_cost_us"]["optimistic"], 51_000)


class DagSchedulerTests(unittest.TestCase):
    def test_exact_schedule_beats_per_kernel_greedy(self) -> None:
        problem = adc.demo_dag_problem()
        exact = adc.exact_dag_schedule(problem)
        greedy = adc.greedy_dag_schedule(problem)
        self.assertLess(exact.total_latency_us, greedy.total_latency_us)
        self.assertIn("fused_gate_activation_cta8", exact.actions)
        self.assertLessEqual(exact.peak_workspace_bytes, problem.workspace_budget_bytes)
        self.assertEqual(set(exact.covered_nodes), set(problem.nodes))
        self.assertAlmostEqual(exact.total_latency_us, 23.92)

    def test_workspace_can_make_problem_infeasible(self) -> None:
        problem = adc.demo_dag_problem()
        impossible = adc.dataclasses.replace(problem, workspace_budget_bytes=1)
        with self.assertRaises(ValueError):
            adc.exact_dag_schedule(impossible)

    def test_topological_solver_remains_exact_with_negative_cache_credit(self) -> None:
        problem = adc.DagProblem(
            nodes=("a", "b"),
            predecessors={"a": frozenset(), "b": frozenset({"a"})},
            output_bytes={"a": 0, "b": 0},
            actions=(
                adc.DagAction("a_fast", frozenset({"a"}), 2.0, cache_tag_out="x"),
                adc.DagAction("a_slow", frozenset({"a"}), 3.0, cache_tag_out="y"),
                adc.DagAction("b_from_x", frozenset({"b"}), 3.0, cache_tag_out="done_x"),
                adc.DagAction("b_from_y", frozenset({"b"}), 3.0, cache_tag_out="done_y"),
            ),
            workspace_budget_bytes=1,
            launch_overhead_us=0.0,
            transition_cost_us={
                ("x", "done_x"): 0.0,
                ("x", "done_y"): 100.0,
                ("y", "done_x"): 100.0,
                ("y", "done_y"): -2.5,
            },
        )
        schedule = adc.exact_dag_schedule(problem)
        self.assertEqual(schedule.actions, ("a_slow", "b_from_y"))
        self.assertAlmostEqual(schedule.total_latency_us, 3.5)


class ArtifactTests(unittest.TestCase):
    def test_certificate_refuses_remote_interpolation(self) -> None:
        cert = adc.finalize_certificate(adc.build_dispatch_certificate())
        remote = cert["dispatch"]["rtx_4070_ti_super_oc"]
        self.assertEqual(remote["status"], "unmeasured")
        self.assertIsNone(remote["table"])
        self.assertEqual(len(cert["certificate_hash"]), 64)
        header = adc.emit_cpp_dispatch_header(cert)
        self.assertIn("kRtx4070SuperTableCertified = false", header)
        self.assertIn("kRtx4070TiSuperTableCertified = false", header)

    def test_write_evaluation_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp)
            # Avoid the Monte Carlo cost in this file-output smoke test.
            cert = adc.finalize_certificate(adc.build_dispatch_certificate())
            (output / "dispatch-certificate.json").write_text(
                json.dumps(cert), encoding="utf-8"
            )
            (output / "dispatch-table.hpp").write_text(
                adc.emit_cpp_dispatch_header(cert), encoding="utf-8"
            )
            self.assertTrue((output / "dispatch-certificate.json").is_file())
            self.assertTrue((output / "dispatch-table.hpp").is_file())


if __name__ == "__main__":
    unittest.main()
