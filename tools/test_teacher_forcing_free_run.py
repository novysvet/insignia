#!/usr/bin/env python3
from __future__ import annotations

import math
import random
import unittest
from fractions import Fraction

from teacher_forcing_free_run import (
    FiniteStateAR,
    MixtureRecord,
    SequentialMixtureCertifier,
    bernoulli_mixture_cs_upper,
    binary_kl,
    binary_kl_upper,
    corpus_coverage_hole_model,
    density_cap_trajectory_kl_bound,
    direct_kl,
    enumerate_stopped_laws,
    enumerate_trajectory_laws,
    forward_metrics,
    greedy_metric_counterexample,
    greedy_pairwise_slacks,
    greedy_tokens_identical,
    likelihood_cap_event_upper,
    ppl_reallocation_counterexample,
    random_finite_state_adversary,
    rare_history_cascade,
    renyi_event_upper,
    search_high_failure_pairs,
    simulate_anytime_coverage,
    simultaneous_policy_ucb,
)


def small_rational_model() -> FiniteStateAR:
    states = ("s0", "s1", "failure")
    transition = {
        "s0": ("s0", "s1"),
        "s1": ("s1", "failure"),
        "failure": ("failure", "failure"),
    }
    p_row = {
        "s0": (Fraction(3, 4), Fraction(1, 4)),
        "s1": (Fraction(2, 3), Fraction(1, 3)),
        "failure": (Fraction(3, 5), Fraction(2, 5)),
    }
    q_row = {
        "s0": (Fraction(2, 3), Fraction(1, 3)),
        "s1": (Fraction(1, 2), Fraction(1, 2)),
        "failure": (Fraction(1, 2), Fraction(1, 2)),
    }
    return FiniteStateAR(
        vocab=("a", "b"),
        states=states,
        initial_state="s0",
        transition=transition,
        p_kernels=(p_row, p_row, p_row),
        q_kernels=(q_row, q_row, q_row),
        failure_states=frozenset({"failure"}),
    )


class TeacherForcingFreeRunTests(unittest.TestCase):
    def test_exact_chain_rule_matches_enumerated_path_law(self) -> None:
        model = small_rational_model()
        metrics = forward_metrics(model)
        laws = enumerate_trajectory_laws(model)
        self.assertAlmostEqual(metrics.trajectory_kl, direct_kl(laws.p, laws.q), places=13)
        self.assertAlmostEqual(
            metrics.trajectory_kl, sum(metrics.teacher_kl_by_step), places=13
        )
        self.assertIsNotNone(metrics.symbolic_trajectory_kl)
        self.assertAlmostEqual(
            float(metrics.symbolic_trajectory_kl), metrics.trajectory_kl, places=13
        )
        self.assertEqual(sum(laws.p.values(), Fraction(0)), Fraction(1))
        self.assertEqual(sum(laws.q.values(), Fraction(0)), Fraction(1))

    def test_stopped_chain_rule_matches_enumerated_stopped_law(self) -> None:
        model = small_rational_model()
        metrics = forward_metrics(model)
        stopped = enumerate_stopped_laws(model)
        self.assertAlmostEqual(metrics.stopped_kl, direct_kl(stopped.p, stopped.q), places=13)
        self.assertLess(metrics.stopped_kl, metrics.trajectory_kl)
        self.assertIsNotNone(metrics.symbolic_stopped_kl)
        self.assertAlmostEqual(
            float(metrics.symbolic_stopped_kl), metrics.stopped_kl, places=13
        )

    def test_random_rational_chain_rules(self) -> None:
        for seed in range(5):
            model = random_finite_state_adversary(
                horizon=4,
                vocab_size=3,
                state_count=3,
                rng=random.Random(seed),
                attack_strength=Fraction(1, 8),
            )
            metrics = forward_metrics(model)
            full = enumerate_trajectory_laws(model)
            stopped = enumerate_stopped_laws(model)
            self.assertAlmostEqual(
                metrics.trajectory_kl, direct_kl(full.p, full.q), places=12
            )
            self.assertAlmostEqual(
                metrics.stopped_kl, direct_kl(stopped.p, stopped.q), places=12
            )

    def test_inverse_binary_kl_is_attained_by_two_point_law(self) -> None:
        p = Fraction(1, 4)
        q = Fraction(1, 2)
        model = FiniteStateAR(
            vocab=("fail", "safe"),
            states=("start", "failure", "safe"),
            initial_state="start",
            transition={
                "start": ("failure", "safe"),
                "failure": ("failure", "failure"),
                "safe": ("safe", "safe"),
            },
            p_kernels=({
                "start": (p, 1 - p),
                "failure": (Fraction(1), Fraction(0)),
                "safe": (Fraction(1), Fraction(0)),
            },),
            q_kernels=({
                "start": (q, 1 - q),
                "failure": (Fraction(1), Fraction(0)),
                "safe": (Fraction(1), Fraction(0)),
            },),
            failure_states=frozenset({"failure"}),
        )
        metrics = forward_metrics(model)
        self.assertEqual(metrics.p_failure, p)
        self.assertEqual(metrics.q_failure, q)
        self.assertAlmostEqual(metrics.trajectory_kl, binary_kl(float(p), float(q)))
        self.assertAlmostEqual(binary_kl_upper(float(p), metrics.trajectory_kl), float(q))

    def test_rational_cascade_vocabulary_eight_horizon_fifty(self) -> None:
        model = rare_history_cascade(50, Fraction(1, 30), vocab_size=8)
        metrics = forward_metrics(model)
        expected = Fraction(1) - Fraction(29, 30) ** 50
        self.assertEqual(metrics.p_failure, Fraction(0))
        self.assertEqual(metrics.q_failure, expected)
        self.assertEqual(sum(metrics.q_first_failure, Fraction(0)), expected)
        self.assertEqual(sum(metrics.q_terminal_state.values(), Fraction(0)), Fraction(1))
        self.assertAlmostEqual(metrics.trajectory_kl, -50 * math.log(29 / 30), places=13)
        self.assertAlmostEqual(
            binary_kl_upper(0.0, metrics.trajectory_kl), float(expected), places=13
        )

    def test_cascade_passes_three_point_five_percent_local_ppl_gate(self) -> None:
        metrics = forward_metrics(rare_history_cascade(50, Fraction(1, 30)))
        self.assertLess(float(Fraction(30, 29) - 1), 0.035)
        self.assertGreater(float(metrics.q_failure), 0.81)

    def test_density_likelihood_and_renyi_bounds(self) -> None:
        self.assertAlmostEqual(
            density_cap_trajectory_kl_bound([0.1, 0.2], [2.0, 3.0]), 0.8
        )
        self.assertAlmostEqual(likelihood_cap_event_upper(0.02, 4.0), 0.08)
        self.assertAlmostEqual(renyi_event_upper(0.25, 0.0, 2.0), 0.5)

    def test_zero_ppl_delta_can_hide_probability_reallocation(self) -> None:
        example = ppl_reallocation_counterexample(epsilon=1e-6)
        self.assertEqual(example["ppl_ratio"], 1.0)
        self.assertTrue(example["top1_agrees"])
        self.assertEqual(example["p_failure"], 0.0)
        self.assertGreater(example["q_failure"], 0.48)

    def test_corpus_coverage_hole_has_zero_logged_kl(self) -> None:
        model = corpus_coverage_hole_model(Fraction(1, 100))
        metrics = forward_metrics(model)
        logged_p = model.p_kernels[0]["logged"]
        logged_q = model.q_kernels[0]["logged"]
        self.assertEqual(logged_p, logged_q)
        self.assertEqual(metrics.q_failure, Fraction(99, 100))
        self.assertAlmostEqual(metrics.trajectory_kl, math.log(100))

    def test_greedy_pairwise_condition_is_exact(self) -> None:
        exact = [[2.0, 1.0, 0.0], [3.0, 1.0, -2.0]]
        good = [[1.8, 1.2, 0.1], [2.5, 1.4, -1.0]]
        bad = [[1.8, 1.2, 0.1], [1.0, 1.1, -1.0]]
        self.assertTrue(greedy_tokens_identical(exact, good))
        self.assertTrue(all(v > 0 for v in greedy_pairwise_slacks(exact, good)))
        self.assertFalse(greedy_tokens_identical(exact, bad))
        self.assertLess(greedy_pairwise_slacks(exact, bad)[1], 0.0)

    def test_average_logit_metrics_do_not_certify_greedy(self) -> None:
        example = greedy_metric_counterexample(margin=1e-6, background=1e6)
        self.assertNotEqual(example["exact_top1"], example["candidate_top1"])
        self.assertGreaterEqual(example["raw_cosine"], 1.0 - 1e-12)
        self.assertGreaterEqual(example["centered_cosine"], 1.0 - 1e-12)
        self.assertLess(example["mse"], 1e-12)
        self.assertLess(example["pairwise_slack"], 0.0)

    def test_bernoulli_mixture_confidence_sequence_shrinks(self) -> None:
        u100 = bernoulli_mixture_cs_upper(0, 100, alpha=0.05)
        u1000 = bernoulli_mixture_cs_upper(0, 1000, alpha=0.05)
        self.assertLess(u1000, u100)
        self.assertLess(u1000, 0.01)

    def test_anytime_coverage_under_early_stopping(self) -> None:
        result = simulate_anytime_coverage(
            true_failure=0.02,
            alpha=0.05,
            max_samples=300,
            trials=1000,
            certify_threshold=0.08,
            seed=42,
        )
        self.assertGreaterEqual(result.coverage, 0.95)
        self.assertGreater(result.stopped_early, 900)

    def test_mixture_certifier_and_policy_selection_penalty(self) -> None:
        certifier = SequentialMixtureCertifier(alpha=0.05)
        rng = random.Random(3)
        for i in range(120):
            source = ("P", "Q", "I")[i % 3]
            # The three trajectory laws are identical in this smoke test, so
            # every importance weight is one.  Source propensities are logged.
            certifier.add(MixtureRecord(
                source=source,
                failure=int(rng.random() < 0.01),
                log_p=0.0,
                log_q=0.0,
                log_intervention=0.0,
                exact_probability=0.3,
                candidate_probability=0.4,
                intervention_probability=0.3,
            ))
        bounds = certifier.bounds()
        self.assertEqual(bounds["records"], 120)
        self.assertEqual(bounds["candidate_records"], 40)
        self.assertLessEqual(bounds["combined_ucb"], 1.0)
        loose = simultaneous_policy_ucb(
            [0.0] * 100, [1.0] * 100, alpha=0.05, prior_weight=0.01
        )
        tight = simultaneous_policy_ucb(
            [0.0] * 100, [1.0] * 100, alpha=0.05, prior_weight=0.5
        )
        self.assertGreater(loose, tight)

    def test_adversary_search_respects_budget_and_sharp_bound(self) -> None:
        rows = search_high_failure_pairs(
            budget=0.5,
            horizon=10,
            vocab_size=3,
            state_count=4,
            trials=20,
            strength_grid=32,
            keep=5,
            seed=11,
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertLessEqual(row.trajectory_kl, 0.5 + 1e-12)
            self.assertLessEqual(row.q_failure, row.binary_upper + 1e-12)
        self.assertGreater(rows[0].failure_increase, 0.25)


if __name__ == "__main__":
    unittest.main()
