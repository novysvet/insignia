#!/usr/bin/env python3
from __future__ import annotations

import itertools
import math
import unittest

from anytime_ab import (
    CellRule,
    DecisionEngine,
    EscalationModel,
    EscalationStage,
    MixtureBettingEProcess,
    PairObservation,
    ProtocolConfig,
    TwoSidedAnchorAlarm,
    bounded_log_ratio_score,
    comparison_score,
    default_escalation_model,
    derive_randomization,
    epoch_alpha,
    polynomial_alpha,
    seed_commitment,
    solve_escalation_dp,
)
from simulate_anytime_ab import (
    HostileWSLSimulator,
    MethodConfig,
    SimulationConfig,
    decide_anytime,
    decide_three_run_median,
)


class BettingTests(unittest.TestCase):
    def test_comparison_ties_are_neutral(self) -> None:
        self.assertEqual(comparison_score(.98, .98, lower_is_better=True), 0)
        self.assertEqual(comparison_score(.97, .98, lower_is_better=True), 1)
        self.assertEqual(comparison_score(.99, .98, lower_is_better=True), -1)

    def test_exact_fair_coin_crossing_bound(self) -> None:
        # Exhaust every path.  The event is pathwise optional stopping, not a
        # fixed-n tail probability.
        alpha = 0.20
        horizon = 14
        crossing_probability = 0.0
        for bits in itertools.product((0, 1), repeat=horizon):
            process = MixtureBettingEProcess()
            crossed = False
            for bit in bits:
                process.update(1.0 if bit else -1.0)
                crossed = crossed or process.crossed(alpha)
            if crossed:
                crossing_probability += 0.5**horizon
        self.assertLessEqual(crossing_probability, alpha + 1e-12)
        self.assertGreater(crossing_probability, 0)

    def test_summable_allocations(self) -> None:
        total = .05
        spent = sum(polynomial_alpha(total, k) for k in range(100_000))
        self.assertLessEqual(spent, total)
        self.assertAlmostEqual(epoch_alpha(.01, 0), .01 * 6 / math.pi**2)

    def test_two_sided_anchor_alarm_has_anytime_union_bound(self) -> None:
        alpha = 0.20
        horizon = 12
        crossing_probability = 0.0
        for bits in itertools.product((0, 1), repeat=horizon):
            alarm = TwoSidedAnchorAlarm(.5, .5, alpha)
            crossed = False
            for bit in bits:
                crossed = alarm.update(float(bit)) or crossed
            if crossed:
                crossing_probability += 0.5**horizon
        self.assertLessEqual(crossing_probability, alpha + 1e-12)

    def test_hmac_randomization_is_replayable_and_bound_to_case(self) -> None:
        secret = bytes(range(32))
        common = dict(
            protocol_hash="1" * 64,
            candidate_id="B",
            epoch=0,
            pair_id="p1",
            cell="decode",
        )
        first = derive_randomization(secret, case_id="short", **common)
        second = derive_randomization(secret, case_id="short", **common)
        changed = derive_randomization(secret, case_id="long", **common)
        self.assertEqual(first, second)
        self.assertEqual(first.seed_commitment, seed_commitment(secret))
        self.assertNotEqual(first.message_sha256, changed.message_sha256)

    def test_bounded_log_score_is_bounded_and_centered_at_threshold(self) -> None:
        self.assertAlmostEqual(
            bounded_log_ratio_score(
                .98, .98, minimum_value=1.0, maximum_value=100.0
            ),
            0.0,
        )
        for ratio in (0.01, 0.1, 1.0, 10.0, 100.0):
            score = bounded_log_ratio_score(
                ratio, .98, minimum_value=1.0, maximum_value=100.0
            )
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)


class DecisionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProtocolConfig(
            version="test",
            candidate_id="B",
            candidate_alpha=.05,
            reject_beta=.05,
            max_campaign_seconds=10_000,
            cells=(
                CellRule("x", "full", "latency", "lower", .98, 1.02,
                         min_pairs=6, max_pairs=80),
                CellRule("y", "full", "latency", "lower", .98, 1.02,
                         min_pairs=6, max_pairs=80),
            ),
        )

    @staticmethod
    def observation(pair: int, cell: str, *, epoch: int = 0,
                    a: float | None = 100, b: float | None = 90,
                    parity: bool = True, code: str = "ok") -> PairObservation:
        return PairObservation(
            pair_id=str(pair), epoch=epoch, cell=cell, case_id="c", order="AB",
            a_value=a, b_value=b, parity_ok=parity, prep_ok=True,
            validity_code=code, cost_seconds=1.0,
        )

    def test_intersection_union_requires_every_cell(self) -> None:
        engine = DecisionEngine(self.config)
        for i in range(40):
            engine.update(self.observation(i, "x"))
        self.assertNotEqual(engine.status(), "PROMOTE")
        for i in range(40):
            engine.update(self.observation(i, "y"))
        self.assertEqual(engine.status(), "PROMOTE")

    def test_epochs_do_not_pool(self) -> None:
        engine = DecisionEngine(self.config)
        for i in range(40):
            engine.update(self.observation(i, "x", epoch=0))
        for i in range(40):
            engine.update(self.observation(i, "y", epoch=1))
        self.assertNotEqual(engine.status(), "PROMOTE")

    def test_outcome_dependent_exclusion_is_invalid(self) -> None:
        engine = DecisionEngine(self.config)
        engine.update(self.observation(
            1, "x", a=100, b=500,
            code="prestart_infrastructure_fault"))
        self.assertEqual(engine.status(), "INVALID")

    def test_parity_failure_rejects(self) -> None:
        engine = DecisionEngine(self.config)
        engine.update(self.observation(1, "x", parity=False))
        self.assertEqual(engine.status(), "REJECT")

    def test_first_terminal_decision_is_frozen(self) -> None:
        engine = DecisionEngine(self.config)
        pair = 0
        while engine.status() == "CONTINUE" and pair < 80:
            engine.update(self.observation(pair, "x", a=100, b=90))
            engine.update(self.observation(pair, "y", a=100, b=90))
            pair += 1
        self.assertEqual(engine.status(), "PROMOTE")
        stopped_after = engine.first_terminal_after_records
        engine.update(self.observation(999, "x", a=100, b=200))
        self.assertEqual(engine.status(), "PROMOTE")
        self.assertEqual(engine.first_terminal_after_records, stopped_after)
        self.assertEqual(engine.post_terminal_records, 1)


class EscalationTests(unittest.TestCase):
    def test_exact_dp_is_between_certified_bounds(self) -> None:
        result = solve_escalation_dp(default_escalation_model())
        self.assertGreaterEqual(result.value + 1e-9, result.lower_bound)
        self.assertLessEqual(result.value, result.perfect_information_upper_bound + 1e-9)
        self.assertAlmostEqual(
            result.certified_gap,
            result.perfect_information_upper_bound - result.lower_bound,
        )
        self.assertTrue(result.action.startswith("MEASURE:"))

    def test_escalation_model_rejects_prerequisite_cycles(self) -> None:
        model = EscalationModel(
            theta=("x",),
            prior=(1.0,),
            promote_utility=(1.0,),
            stages=(
                EscalationStage("a", 1.0, ("y",), ((1.0,),), prerequisites=("b",)),
                EscalationStage("b", 1.0, ("y",), ((1.0,),), prerequisites=("a",)),
            ),
        )
        with self.assertRaises(ValueError):
            model.validate()


class SimulatorTests(unittest.TestCase):
    def test_replay_seed_zero_promotes_slower_arm_by_three_medians(self) -> None:
        config = SimulationConfig(true_ratio=1.05, max_pairs=80, guarded=True)
        records = HostileWSLSimulator(0, config).campaign()
        method = MethodConfig(max_pairs=80)
        three = decide_three_run_median(records, method)
        anytime = decide_anytime(records[:3], method)
        self.assertEqual(three.status, "PROMOTE")
        self.assertGreater(config.true_ratio, 1.0)
        self.assertNotEqual(anytime.status, "PROMOTE")
        self.assertTrue(all(
            run.cache_before == 0
            for pair in records[:3]
            for run in pair.runs if run.measured
        ))

    def test_simulation_is_replayable(self) -> None:
        config = SimulationConfig(true_ratio=.98, max_pairs=5)
        first = HostileWSLSimulator(123, config).campaign()
        second = HostileWSLSimulator(123, config).campaign()
        self.assertEqual(
            [p.flat_dict() for p in first],
            [p.flat_dict() for p in second],
        )


if __name__ == "__main__":
    unittest.main()
