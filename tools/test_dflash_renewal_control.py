#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

from dflash_renewal_control import (
    Action, Outcome, SemiMarkovModel, build_demo_model,
    nonthreshold_counterexample, retry_threshold,
    safe_retry_competitive_bound, sequential_rows_from_prefix,
    solve_occupation_lp, stop_continue_threshold,
)


class RenewalControlTests(unittest.TestCase):
    def test_zero_token_ratio_of_expectations(self) -> None:
        model = SemiMarkovModel(("s",), {"s": (
            Action("variable", (
                Outcome(.5, "s", 0, 1, 1), Outcome(.5, "s", 2, 2, 3))),
            Action("steady", (Outcome(1, "s", 1, 1, 2.1),)),
        )})
        result = solve_occupation_lp(model)
        self.assertAlmostEqual(result.accepted_tokens_per_second, 500)
        self.assertEqual(result.policy[0].action, "variable")
        self.assertNotAlmostEqual(result.accepted_tokens_per_second, 1000/3)

    def test_trajectory_constraint_randomizes(self) -> None:
        model = SemiMarkovModel(("s",), {"s": (
            Action("risk", (Outcome(1, "s", 2, 2, 1,
                                            trajectory_end=1,
                                            trajectory_violation=.1),)),
            Action("safe", (Outcome(1, "s", 1, 1, 1,
                                            trajectory_end=1,
                                            trajectory_violation=0),)),
        )})
        result = solve_occupation_lp(model,
            max_trajectory_violation_probability=.05)
        p = {e.action: e.probability for e in result.policy}
        self.assertAlmostEqual(p["risk"], .5)
        self.assertAlmostEqual(p["safe"], .5)
        self.assertAlmostEqual(result.trajectory_violation_probability, .05)

    def test_damage_constraint(self) -> None:
        model = SemiMarkovModel(("s",), {"s": (
            Action("risk", (Outcome(1, "s", 4, 4, 4, damage=.4),)),
            Action("safe", (Outcome(1, "s", 1, 1, 2),)),
        )})
        result = solve_occupation_lp(model, max_damage_per_accepted_token=.05)
        self.assertLessEqual(result.damage_per_accepted_token, .05 + 1e-9)

    def test_thresholds(self) -> None:
        self.assertAlmostEqual(retry_threshold(
            throughput_tokens_per_ms=.002, marginal_retry_ms=100,
            bad_state_gain_tokens=1), .2)
        self.assertAlmostEqual(stop_continue_threshold(
            throughput_tokens_per_ms=.002, marginal_verify_ms=100,
            success_gain_tokens=1), .2)

    def test_nonthreshold(self) -> None:
        self.assertEqual([r["retry"] for r in nonthreshold_counterexample()],
                         [True, False, True])

    def test_sequential_rows(self) -> None:
        self.assertEqual(sequential_rows_from_prefix(0, 4), 0)
        self.assertEqual(sequential_rows_from_prefix(3, 4), 3)
        self.assertEqual(sequential_rows_from_prefix(3, 4, known_first=False), 4)

    def test_safe_bound(self) -> None:
        ratio = safe_retry_competitive_bound(
            exact_round_ms=680, controller_snapshot_restore_ms=10)
        self.assertAlmostEqual(ratio, 2 + 10/680)
        self.assertLess(ratio, 2.02)

    def test_demo(self) -> None:
        result = solve_occupation_lp(build_demo_model(),
                                     max_damage_per_accepted_token=.01)
        self.assertTrue(math.isfinite(result.accepted_tokens_per_second))
        self.assertGreater(result.accepted_tokens_per_second, 0)
        self.assertTrue(result.policy)


if __name__ == "__main__":
    unittest.main()

