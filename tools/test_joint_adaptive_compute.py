#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

from joint_adaptive_compute import (
    ActionKernel,
    ComputeAction,
    ControllerState,
    ExpectedMetrics,
    FiniteControllerModel,
    LogitObservation,
    SyntheticParameters,
    draft_checkpoint_counterexample,
    enumerate_compute_actions,
    every_heuristic_locally_sensible_case,
    expert_route_counterexample,
    lyapunov_performance_bound,
    metric_distribution,
    myopic_value_of_information,
    posterior_from_logit_observation,
    robust_safety_certificate,
    solve_occupation_lp,
    throughput_price_of_safety,
)


class JointAdaptiveComputeTests(unittest.TestCase):
    def test_logit_likelihood_moves_posterior_monotonically(self) -> None:
        p = SyntheticParameters()
        prior = 0.25
        easy_evidence = LogitObservation(2, 0)
        hard_evidence = LogitObservation(0, 1)
        q_easy = posterior_from_logit_observation(prior, easy_evidence, p.likelihood)
        q_hard = posterior_from_logit_observation(prior, hard_evidence, p.likelihood)
        self.assertLess(q_easy, prior)
        self.assertGreater(q_hard, prior)
        self.assertGreater(q_hard, q_easy)

    def test_exact_metric_preserves_prior_in_expectation(self) -> None:
        p = SyntheticParameters()
        for prior in (0.1, 0.4, 0.9):
            branches = metric_distribution(prior, p)
            self.assertAlmostEqual(sum(prob for _, prob, _ in branches), 1.0)
            posterior_mean = sum(prob * posterior for _, prob, posterior in branches)
            self.assertAlmostEqual(posterior_mean, prior, places=12)

    def test_gross_value_of_information_is_nonnegative(self) -> None:
        p = SyntheticParameters()
        for prior in p.belief_grid:
            result = myopic_value_of_information(
                prior,
                cache_level=1,
                io_queue=0,
                quality_debt=0,
                candidate_actions=enumerate_compute_actions(),
                parameters=p,
                throughput_tokens_per_ms=0.031,
                hard_multiplier=75.0,
            )
            self.assertGreaterEqual(result["gross_voi"], -1e-10)

    def test_time_normalized_lp_and_two_constraints(self) -> None:
        p = SyntheticParameters(belief_grid=(0.5,), cache_levels=2, io_levels=2)
        state = ControllerState(0, 0, 0, 0, "ready")
        risky = ActionKernel(
            "risk",
            ((0, 1.0),),
            committed=2.0,
            time_ms=1.0,
            bytes_mb=0.0,
            ppl_loss=0.20,
            hard_end=1.0,
            hard_violation=0.10,
            catastrophe=0.0,
        )
        safe = ActionKernel(
            "safe",
            ((0, 1.0),),
            committed=1.0,
            time_ms=1.0,
            bytes_mb=0.0,
            ppl_loss=0.0,
            hard_end=1.0,
            hard_violation=0.0,
            catastrophe=0.0,
        )
        model = FiniteControllerModel((state,), ((risky, safe),), p)
        result = solve_occupation_lp(
            model,
            max_ppl_loss_per_token=0.05,
            max_hard_violation_probability=0.05,
            max_catastrophe_per_token=None,
        )
        policy = {row.action: row.probability for row in result.policy}
        # PPL permits risk probability 1/3; the hard constraint permits 1/2.
        self.assertAlmostEqual(policy["risk"], 1 / 3, places=7)
        self.assertAlmostEqual(policy["safe"], 2 / 3, places=7)
        self.assertAlmostEqual(result.ppl_loss_per_committed_token, 0.05, places=8)
        self.assertLess(result.hard_violation_probability, 0.05)
        self.assertAlmostEqual(result.committed_tokens_per_second, 4000 / 3, places=6)

    def test_pairwise_counterexamples_destroy_separability(self) -> None:
        expert = expert_route_counterexample()
        checkpoint = draft_checkpoint_counterexample()
        unstable = every_heuristic_locally_sensible_case()
        self.assertFalse(expert["separable"])
        self.assertNotEqual(expert["myopic_choice"], expert["dynamic_choice"])
        self.assertFalse(checkpoint["separable"])
        self.assertLess(
            checkpoint["independent_combination"]["tps"],
            checkpoint["joint_choice"]["tps"],
        )
        self.assertTrue(unstable["all_individual_improve"])
        self.assertTrue(unstable["combination_worse_than_baseline"])

    def test_safety_certificate_uses_incremental_value_not_raw_time(self) -> None:
        candidate = ExpectedMetrics(
            committed=4.0,
            time_ms=120.0,
            bytes_mb=0.0,
            ppl_loss=0.0,
            hard_end=0.1,
            hard_violation=0.0,
            catastrophe=0.0,
            harm_probability=0.0,
            next_hard_if_easy=0.1,
            next_hard_if_hard=0.8,
        )
        exact = ExpectedMetrics(
            committed=1.0,
            time_ms=60.0,
            bytes_mb=0.0,
            ppl_loss=0.0,
            hard_end=0.1,
            hard_violation=0.0,
            catastrophe=0.0,
            harm_probability=0.0,
            next_hard_if_easy=0.1,
            next_hard_if_hard=0.8,
        )
        certificate = robust_safety_certificate(
            candidate,
            exact,
            throughput_tokens_per_ms=0.02,
            controller_and_guard_ms=1.0,
            ppl_radius=0.0,
            hard_risk_radius=0.0,
            catastrophe_radius=0.0,
            time_radius_ms=0.0,
            ppl_envelope_per_token=0.01,
            hard_envelope=0.01,
            catastrophe_envelope_per_token=0.01,
            uncertainty_radius=0.0,
            max_uncertainty_radius=0.1,
            overlap=1.0,
            min_overlap=0.5,
        )
        self.assertLess(certificate["saving_lcb_ms"], 0.0)
        self.assertGreater(certificate["economic_gain_lcb"], 0.0)
        self.assertTrue(certificate["allowed"])


    def test_lyapunov_bound_charges_model_and_belief_error(self) -> None:
        bound = lyapunov_performance_bound(
            drift_constant=2.0,
            V=20.0,
            tau_min_ms=5.0,
            model_error=0.01,
            belief_diameter=0.02,
            belief_value_lipschitz=3.0,
            planning_error=0.005,
            horizon=100,
            terminal_virtual_queues=(2.0, 5.0),
        )
        expected_error = 0.005 + 2 * 0.01 + 2 * 3.0 * 0.02
        self.assertAlmostEqual(bound["action_value_error"], expected_error)
        self.assertAlmostEqual(
            bound["transformed_objective_gap_per_decision"],
            2.0 / 20.0 + expected_error,
        )
        self.assertEqual(
            bound["positive_average_constraint_violation_bounds"],
            (0.02, 0.05),
        )

    def test_throughput_price_of_safety(self) -> None:
        result = throughput_price_of_safety(
            joint_tokens_per_round=3.0,
            joint_ms_per_round=80.0,
            exact_tokens_per_round=1.0,
            exact_ms_per_round=50.0,
            fallback_fraction=0.2,
            guard_ms_per_round=0.2,
        )
        self.assertGreater(result["joint_tokens_per_second"], result["safe_tokens_per_second"])
        self.assertGreaterEqual(result["relative_price"], 0.0)
        self.assertLess(result["relative_price"], 1.0)


if __name__ == "__main__":
    unittest.main()
