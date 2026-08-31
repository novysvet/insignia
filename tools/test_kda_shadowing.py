#!/usr/bin/env python3
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kda_shadowing import (
    StepPair,
    exact_error_terms,
    approximate_propagator_error_terms,
    false_confidence_examples,
    fp8_block_quantize,
    fp8_rank_obstruction_demo,
    fp8_scale_discontinuity_demo,
    kda_driven_energy_terms,
    kda_energy_terms,
    kda_left_gate,
    lazy_boundary_reset_schedule,
    lazy_reset_schedule,
    make_adversarial_tight,
    make_contractive_kda,
    make_diagonal_shared_directions,
    make_fp8_kda,
    make_marginal,
    make_switching_nonnormal,
    recurrence_step,
    simulate_scenario,
    solve_reset_bruteforce,
    solve_reset_dp,
    symbolic_checks,
    write_artifacts,
)


class KdaShadowingTests(unittest.TestCase):
    def test_exact_decompositions_numeric(self) -> None:
        rng = np.random.default_rng(11)
        d = 4
        S = rng.normal(size=(d, d))
        hat_S = S + rng.normal(scale=1e-3, size=(d, d))
        A = rng.normal(scale=.2, size=(d, d))
        B = rng.normal(scale=.2, size=(d, d))
        hat_A = A + rng.normal(scale=1e-4, size=(d, d))
        hat_B = B + rng.normal(scale=1e-4, size=(d, d))
        u, v = rng.normal(size=d), rng.normal(size=d)
        hat_u = u + rng.normal(scale=1e-4, size=d)
        hat_v = v + rng.normal(scale=1e-4, size=d)
        step = StepPair(A, B, .3, u, v, hat_A, hat_B, .3002, hat_u, hat_v)
        S_next = recurrence_step(S, A, B, .3, u, v)
        pre = recurrence_step(hat_S, hat_A, hat_B, .3002, hat_u, hat_v)
        rounding = rng.normal(scale=1e-6, size=(d, d))
        hat_next = pre + rounding
        exact = exact_error_terms(S, hat_S, S_next, hat_next, step, rounding)
        approx = approximate_propagator_error_terms(
            S, hat_S, S_next, hat_next, step, rounding
        )
        target = hat_next - S_next
        np.testing.assert_allclose(exact.reconstructed, target, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(approx.reconstructed, target, rtol=1e-11, atol=1e-11)

    def test_symbolic_checks(self) -> None:
        result = symbolic_checks()
        self.assertEqual(result["max_symbolic_dimension"], 4)
        for small_d in map(str, range(1, 5)):
            self.assertIn(small_d, result["dimension_sweep"])
            self.assertTrue(all(result["dimension_sweep"][small_d].values()))
        self.assertTrue(result["exact_propagator_decomposition_zero"])
        self.assertTrue(result["approximate_propagator_decomposition_zero"])
        self.assertTrue(result["kda_energy_identity_zero"])
        self.assertTrue(result["kda_driven_energy_identity_zero"])
        self.assertGreater(result["switch_pair_max_eigenvalue"], 1.0)
        self.assertEqual(result["rank_one_prequantized_state_rank"], 1)
        self.assertEqual(result["rank_two_quantized_state_rank"], 2)
        self.assertFalse(result["rank_one_update_can_match_quantized_state"])

    def test_kda_energy_identity(self) -> None:
        rng = np.random.default_rng(3)
        for d in (2, 4, 8):
            for _ in range(20):
                k = rng.normal(size=d)
                k /= np.linalg.norm(k)
                beta = float(rng.random())
                decay = rng.uniform(.2, 1.0, size=d)
                X = rng.normal(size=(d, d))
                terms = kda_energy_terms(X, k, beta, decay)
                self.assertLess(terms["identity_error"], 2e-10)
                self.assertLessEqual(terms["after"], terms["before"] + 2e-10)
                self.assertLessEqual(np.linalg.norm(kda_left_gate(k, beta, decay), 2),
                                     max(decay) + 1e-10)
                v = rng.normal(size=d)
                driven = kda_driven_energy_terms(X, k, beta, decay, v)
                self.assertLess(driven["identity_error"], 2e-10)
                self.assertLessEqual(
                    driven["after"],
                    (max(decay) * np.linalg.norm(X, "fro")) ** 2
                    + beta * np.linalg.norm(v) ** 2 + 2e-10,
                )

    def test_all_proposed_bounds_cover_observed_error(self) -> None:
        scenarios = (
            make_contractive_kda(d=8, steps=20),
            make_fp8_kda(d=8, steps=20),
            make_marginal(d=8, steps=20),
            make_switching_nonnormal(d=4, steps=12),
            make_adversarial_tight(d=4, steps=12),
            make_diagonal_shared_directions(d=4, steps=20),
        )
        for scenario in scenarios:
            rows, summary = simulate_scenario(scenario)
            for name, violation in summary["bound_violations"].items():
                self.assertLessEqual(violation, 1e-9, (scenario.name, name, violation))
            for row in rows:
                self.assertLessEqual(row["observed"], row["operator_bound"] + 1e-9)
                self.assertLessEqual(row["observed"], row["online_bound"] + 1e-9)
                self.assertLessEqual(row["observed"], row["online_realized_bound"] + 1e-9)
                self.assertLessEqual(row["observed"], row["transition_bound"] + 1e-9)

    def test_dimension_32_smoke(self) -> None:
        rows, summary = simulate_scenario(make_contractive_kda(d=32, steps=4))
        self.assertEqual(summary["dimension"], 32)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(
            row["observed"] <= row["online_bound"] + 1e-9 for row in rows
        ))

    def test_fp8_realized_residual_certificate_beats_worst_cell(self) -> None:
        rows, summary = simulate_scenario(make_fp8_kda(d=8, steps=20))
        self.assertTrue(all(
            row["observed"] <= row["online_realized_bound"] + 1e-9 for row in rows
        ))
        self.assertLess(
            summary["final_online_realized_bound"],
            summary["final_online_bound"],
        )

    def test_operator_bound_is_attained(self) -> None:
        rows, _ = simulate_scenario(make_adversarial_tight(d=4, steps=10))
        for row in rows:
            self.assertAlmostEqual(row["observed"], row["operator_bound"], places=13)
            self.assertAlmostEqual(row["observed"], row["online_bound"], places=13)

    def test_diagonal_bound_removes_false_exponential(self) -> None:
        rows, summary = simulate_scenario(make_diagonal_shared_directions(d=4, steps=20))
        self.assertAlmostEqual(rows[-1]["observed"], rows[-1]["diagonal_bound"], places=14)
        self.assertGreater(rows[-1]["operator_bound"] / rows[-1]["diagonal_bound"], 1000)
        self.assertGreater(summary["max_individual_operator_norm"], 1.0)

    def test_switching_nonnormal_spectral_radius_failure(self) -> None:
        scenario = make_switching_nonnormal(d=2, steps=10)
        _, summary = simulate_scenario(scenario)
        initial = np.linalg.norm(scenario.hat_S0 - scenario.S0)
        self.assertLess(summary["max_individual_spectral_radius"], 1.0)
        self.assertGreater(summary["peak_observed"] / initial, 1000)

    def test_false_confidence_examples(self) -> None:
        examples = false_confidence_examples()
        self.assertLess(examples["spectral_radius"]["max_individual"], 1.0)
        self.assertGreater(examples["spectral_radius"]["amplification"], 1000)
        self.assertEqual(examples["readout_sketch"]["current_readout_error"], 0.0)
        self.assertEqual(examples["readout_sketch"]["later_readout_error"], 1.0)
        self.assertLess(examples["mean_decay"]["mean_decay"], .3)
        self.assertEqual(examples["mean_decay"]["error_after"], 1.0)

    def test_fp8_rounding_bound_and_scale_discontinuity(self) -> None:
        rng = np.random.default_rng(19)
        for shape in ((4, 4), (8, 8), (2, 13)):
            values = rng.normal(scale=20.0, size=shape)
            restored, info = fp8_block_quantize(values, power_of_two_scale=True)
            actual = np.linalg.norm(restored - values)
            self.assertLessEqual(actual, info["rounding_bound"] + 1e-10)
            self.assertGreater(info["scale"], 0)
        demo = fp8_scale_discontinuity_demo(1e-7)
        self.assertEqual(demo["low_scale"], 1.0)
        self.assertEqual(demo["high_scale"], 2.0)
        self.assertGreater(demo["local_lipschitz_ratio"], 1000)
        obstruction = fp8_rank_obstruction_demo()
        self.assertEqual(obstruction["source_rank"], 1)
        self.assertEqual(obstruction["quantized_rank"], 2)
        self.assertNotEqual(obstruction["quantized_determinant"], 0.0)

    def test_reset_dp_matches_bruteforce_and_causal_lazy(self) -> None:
        rng = np.random.default_rng(23)
        for n in range(1, 13):
            for _ in range(30):
                gains = rng.uniform(.4, 1.4, size=n).tolist()
                local = rng.uniform(.01, .2, size=n).tolist()
                budget = .35
                dp = solve_reset_dp(gains, local, budget, .7, .1)
                brute = solve_reset_bruteforce(gains, local, budget, .7, .1)
                lazy = lazy_reset_schedule(gains, local, budget, .7, .1)
                self.assertAlmostEqual(dp.net_saving, brute.net_saving)
                self.assertEqual(dp.resets, brute.resets)
                self.assertEqual(lazy.resets, brute.resets)
                self.assertAlmostEqual(lazy.net_saving, brute.net_saving)
                self.assertTrue(all(peak <= budget + 1e-12 for peak in lazy.segment_peaks))

    def test_reset_dp_with_restricted_boundaries(self) -> None:
        gains = [.93, 1.0, .88, 1.0, .97, .82, 1.0, .9, .95, 1.0]
        local = [.08, .11, .07, .13, .09, .06, .14, .08, .1, .07]
        legal = [0, 2, 4, 6, 8, 10]
        dp = solve_reset_dp(
            gains, local, .32, .45, .12,
            allowed_reset_boundaries=legal,
        )
        brute = solve_reset_bruteforce(
            gains, local, .32, .45, .12,
            allowed_reset_boundaries=legal,
        )
        lazy = lazy_boundary_reset_schedule(
            gains, local, .32, .45, .12,
            allowed_reset_boundaries=legal,
        )
        self.assertEqual(dp.resets, brute.resets)
        self.assertEqual(lazy.resets, brute.resets)
        self.assertAlmostEqual(dp.net_saving, brute.net_saving)
        self.assertAlmostEqual(lazy.net_saving, brute.net_saving)
        self.assertTrue(set(dp.boundaries).issubset(set(legal)))
        self.assertTrue(set(lazy.boundaries).issubset(set(legal)))

    def test_artifact_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            payload = write_artifacts(output, d=4, steps=8)
            self.assertTrue((output / "trace.csv").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "reset-demo.json").is_file())
            self.assertTrue((output / "fp8-scale-demo.json").is_file())
            self.assertTrue((output / "fp8-rank-demo.json").is_file())
            self.assertEqual(payload["symbolic_checks"]["max_symbolic_dimension"], 4)
            self.assertTrue(all(
                all(checks.values())
                for checks in payload["symbolic_checks"]["dimension_sweep"].values()
            ))


if __name__ == "__main__":
    unittest.main()
