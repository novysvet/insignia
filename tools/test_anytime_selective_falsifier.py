#!/usr/bin/env python3
from __future__ import annotations

import itertools
import inspect
import math
import unittest

from anytime_selective_falsifier import (
    AuditCaptureBoundary,
    AuditRecord,
    CalibrationRow,
    FeatureAcquisition,
    FeatureOutcome,
    FeatureState,
    RandomAuditLedger,
    aggregate_block_severity,
    build_certificate_fingerprint,
    build_runtime_state_fingerprint,
    canonical_fingerprint,
    certificate_reset_reason,
    counter_based_audit_uniform,
    design_then_certify_threshold_and_audit_plan,
    indistinguishable_environment_demo,
    optimal_audit_probability,
    pathwise_reset_reserve,
    select_historical_screen_threshold_and_audit_plan,
    solve_sequential_feature_acquisition,
)
from evaluate_anytime_selective_falsifier import (
    Context,
    GuardedRiskModel,
    RidgeRiskModel,
    estimated_economic_gains,
)


class AnytimeSelectiveFalsifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fingerprint = canonical_fingerprint({"model": "unit", "loss": "block-v1"})

    def make_record(
        self,
        round_id: int,
        *,
        q: float,
        audited: bool,
        label: float | None = None,
        bound: float = 1.0,
    ) -> AuditRecord:
        uniform = q / 2.0 if audited else q + (1.0 - q) / 2.0
        return AuditRecord(
            round_id=round_id,
            audit_probability=q,
            severity_bound=bound,
            audited=audited,
            audit_uniform=uniform,
            state_fingerprint=self.fingerprint,
            runtime_state_fingerprint=canonical_fingerprint({
                "certificate": self.fingerprint, "round": round_id, "q": q
            }),
            support_key="unit",
            label=label,
            label_due_round=round_id + 1 if audited and label is None else None,
        )

    def test_no_overlap_worlds_are_observationally_identical(self) -> None:
        demo = indistinguishable_environment_demo(24)
        self.assertTrue(demo["logs_identical"])
        self.assertEqual(demo["good_log_sha256"], demo["bad_log_sha256"])
        self.assertEqual(demo["good_deployment"]["selective_risk"], 0.0)
        self.assertEqual(demo["bad_deployment"]["selective_risk"], 1.0)
        self.assertNotEqual(
            demo["good_deployment"]["final_state"],
            demo["bad_deployment"]["final_state"],
        )

    def test_audit_capture_one_step_factor_is_a_supermartingale(self) -> None:
        for q in (0.08, 0.23, 0.61, 1.0):
            for bound in (0.2, 1.0):
                for ratio in (0.05, 0.33, 0.80, 0.97):
                    rho = ratio * q / bound
                    lam = -math.log1p(-rho * bound / q) / bound
                    for k in range(31):
                        value = bound * k / 30.0
                        expected_factor = math.exp(rho * value) * (
                            1.0 - q + q * math.exp(-lam * value)
                        )
                        self.assertLessEqual(expected_factor, 1.0 + 2e-12)

    def test_small_exact_enumeration_respects_time_uniform_error_budget(self) -> None:
        # Exact enumeration over audit coins, with deterministic adaptive-safe
        # losses, checks the proposal component of the capture CS.  The ledger
        # splits delta in half, so the proposal crossing budget is delta/2.
        q = 0.28
        losses = (0.0, 0.06, 0.0, 0.18, 0.0, 0.04, 0.0)
        delta = 0.20
        crossing_probability = 0.0
        for path in itertools.product((False, True), repeat=len(losses)):
            probability = 1.0
            ledger = RandomAuditLedger(
                delta=delta,
                epsilon=1.0,
                startup_budget=20.0,
                q_min=q,
                state_fingerprint=self.fingerprint,
            )
            true_total = 0.0
            crossed = False
            for t, (audited, loss) in enumerate(zip(path, losses)):
                probability *= q if audited else 1.0 - q
                ledger.append(self.make_record(t, q=q, audited=audited))
                if audited:
                    ledger.resolve(t, loss)
                true_total += loss
                crossed |= true_total > ledger.snapshot().proposal_loss_upper + 1e-12
            if crossed:
                crossing_probability += probability
        self.assertLessEqual(crossing_probability, delta / 2.0 + 1e-10)

    def test_exact_enumeration_allows_feedback_adaptive_losses_and_propensities(self) -> None:
        delta = 0.20
        crossing_probability = 0.0
        horizon = 7
        for path in itertools.product((False, True), repeat=horizon):
            probability = 1.0
            ledger = RandomAuditLedger(
                delta=delta,
                epsilon=1.0,
                startup_budget=20.0,
                q_min=0.18,
                state_fingerprint=self.fingerprint,
            )
            debt = 0.0
            true_total = 0.0
            crossed = False
            for t, audited in enumerate(path):
                # Both q_t and Y_t are functions of the prior audit/action
                # history, but are frozen before the current coin.
                q = 0.18 + 0.20*debt
                loss = min(1.0, 0.01 + 0.58*debt + 0.02*(t % 2))
                probability *= q if audited else 1.0 - q
                ledger.append(self.make_record(t, q=q, audited=audited))
                if audited:
                    ledger.resolve(t, loss)
                    debt *= 0.35
                else:
                    debt = min(1.0, 0.72*debt + 0.45*loss + 0.08)
                true_total += loss
                crossed |= true_total > ledger.snapshot().proposal_loss_upper + 1e-12
            if crossed:
                crossing_probability += probability
        self.assertLessEqual(crossing_probability, delta / 2.0 + 1e-10)

    def test_delayed_labels_are_consumed_only_in_decision_order(self) -> None:
        ledger = RandomAuditLedger(
            delta=0.1,
            epsilon=0.5,
            startup_budget=4.0,
            q_min=0.2,
            state_fingerprint=self.fingerprint,
        )
        ledger.append(self.make_record(0, q=0.2, audited=True))
        ledger.append(self.make_record(1, q=0.2, audited=True))
        ledger.append(self.make_record(2, q=0.2, audited=False))
        ledger.resolve(1, 0.0)
        blocked = ledger.snapshot()
        self.assertEqual(blocked.processed_prefix, 0)
        self.assertEqual(blocked.pending_audits, 1)
        self.assertAlmostEqual(blocked.pending_fast_bound, 1.0)
        ledger.resolve(0, 0.0)
        drained = ledger.snapshot()
        self.assertEqual(drained.processed_prefix, 3)
        self.assertEqual(drained.pending_audits, 0)
        self.assertAlmostEqual(drained.pending_fast_bound, 0.0)

    def test_bad_propensity_or_fingerprint_invalidates_fail_closed(self) -> None:
        ledger = RandomAuditLedger(
            delta=0.1,
            epsilon=0.1,
            startup_budget=2.0,
            q_min=0.2,
            state_fingerprint=self.fingerprint,
        )
        self.assertIsNone(ledger.snapshot().invalid_reason)
        bad = AuditRecord(
            round_id=0,
            audit_probability=0.2,
            severity_bound=1.0,
            audited=True,
            audit_uniform=0.9,
            state_fingerprint=self.fingerprint,
            runtime_state_fingerprint=canonical_fingerprint({"bad": 0}),
            support_key="unit",
            label_due_round=1,
        )
        with self.assertRaises(ValueError):
            ledger.append(bad)
        self.assertFalse(ledger.can_commit_fast(severity_bound=1.0))
        self.assertIsNotNone(ledger.invalid_reason)

        other = RandomAuditLedger(
            delta=0.1,
            epsilon=0.1,
            startup_budget=2.0,
            q_min=0.2,
            state_fingerprint=self.fingerprint,
        )
        wrong = self.make_record(0, q=0.2, audited=False)
        wrong = AuditRecord(**{**wrong.__dict__, "state_fingerprint": "wrong"})
        with self.assertRaises(ValueError):
            other.append(wrong)
        self.assertFalse(other.snapshot().certificate_valid)

    def test_pathwise_gate_implication_on_the_coverage_event(self) -> None:
        ledger = RandomAuditLedger(
            delta=0.1,
            epsilon=0.12,
            startup_budget=2.0,
            q_min=0.25,
            state_fingerprint=self.fingerprint,
        )
        committed = 0.0
        fast_commits = 0
        # The implication is deterministic: when the pre-commit upper endpoint
        # covers the next committed loss and passes the budget gate, the actual
        # post-commit loss cannot exceed the budget.
        for t, loss in enumerate((0.0, 0.1, 0.0, 0.8, 0.0, 0.2, 0.0, 0.0)):
            may_commit = ledger.can_commit_fast(severity_bound=1.0)
            if may_commit:
                prospective = ledger.prospective_upper_if_fast(severity_bound=1.0)
                ledger.append(self.make_record(t, q=0.25, audited=False))
                committed += loss
                fast_commits += 1
                if committed <= prospective + 1e-12:
                    self.assertLessEqual(
                        committed,
                        ledger.risk_budget(fast_commits) + 1e-12,
                    )
            else:
                ledger.append(self.make_record(t, q=1.0, audited=True))
                ledger.resolve(t, loss)

    def test_reset_carries_unspent_global_reserve_without_minting_budget(self) -> None:
        carry = pathwise_reset_reserve(
            global_startup_budget=10.0,
            epsilon=0.1,
            closed_fast_commits=(20, 30),
            closed_committed_uppers=(5.0, 6.0),
        )
        self.assertAlmostEqual(carry, 4.0)
        self.assertIsNone(pathwise_reset_reserve(
            global_startup_budget=2.0,
            epsilon=0.05,
            closed_fast_commits=(10,),
            closed_committed_uppers=(4.0,),
        ))

    def test_cost_aware_audit_probability_has_expected_monotonicity(self) -> None:
        cheap = optimal_audit_probability(
            audit_opportunity_cost=25.0,
            information_price=20.0,
            q_min=0.05,
            q_max=0.8,
        )
        expensive = optimal_audit_probability(
            audit_opportunity_cost=400.0,
            information_price=20.0,
            q_min=0.05,
            q_max=0.8,
        )
        more_information = optimal_audit_probability(
            audit_opportunity_cost=400.0,
            information_price=80.0,
            q_min=0.05,
            q_max=0.8,
        )
        self.assertGreater(cheap, expensive)
        self.assertGreater(more_information, expensive)
        self.assertEqual(
            optimal_audit_probability(
                audit_opportunity_cost=-1.0,
                information_price=20.0,
                q_min=0.05,
                q_max=0.8,
            ),
            0.8,
        )

    def test_audit_cost_estimator_has_no_realized_outcome_argument(self) -> None:
        parameters = inspect.signature(estimated_economic_gains).parameters
        self.assertEqual(tuple(parameters), ("context", "cfg", "severity_score"))

    def test_one_shot_heldout_certificate_accepts_safe_and_rejects_bad(self) -> None:
        design = [
            CalibrationRow(0.01, 0.0, 130.0, -30.0, support_key="cell")
            for _ in range(900)
        ]
        safe_certificate = [
            CalibrationRow(0.01, 0.0, 130.0, -30.0, support_key="cell")
            for _ in range(900)
        ]
        plan = design_then_certify_threshold_and_audit_plan(
            design,
            safe_certificate,
            thresholds=(0.02,),
            information_prices=(10.0,),
            epsilon=0.08,
            delta=0.05,
            q_min=0.1,
            q_max=0.5,
            min_selected_mass=50.0,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertLessEqual(plan.heldout_risk_upper, 0.08)
        self.assertEqual(plan.supported_keys, ("cell",))

        bad_certificate = [
            CalibrationRow(0.01, 0.3, 130.0, -30.0, support_key="cell")
            for _ in range(900)
        ]
        self.assertIsNone(design_then_certify_threshold_and_audit_plan(
            design,
            bad_certificate,
            thresholds=(0.02,),
            information_prices=(10.0,),
            epsilon=0.08,
            delta=0.05,
            q_min=0.1,
            q_max=0.5,
            min_selected_mass=50.0,
        ))

    def test_adaptive_historical_screen_has_no_coverage_role_and_drops_bad_cells(self) -> None:
        design = [
            CalibrationRow(0.01, 0.01, 120.0, -20.0, support_key="safe")
            for _ in range(180)
        ] + [
            CalibrationRow(0.01, 0.30, 120.0, -20.0, support_key="bad")
            for _ in range(180)
        ]
        later = [
            CalibrationRow(0.01, 0.01, 120.0, -20.0, support_key="safe")
            for _ in range(180)
        ] + [
            CalibrationRow(0.01, 0.30, 120.0, -20.0, support_key="bad")
            for _ in range(180)
        ]
        plan = select_historical_screen_threshold_and_audit_plan(
            design,
            later,
            thresholds=(0.02,),
            information_prices=(10.0,),
            epsilon=0.08,
            q_min=0.1,
            q_max=0.5,
            empirical_margin=0.01,
            min_selected_mass=40.0,
            min_support_count=4,
            local_screen_mass=30.0,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.supported_keys, ("safe",))
        self.assertIn("no_coverage_claim", plan.screening_method)

    def test_reset_guard_score_never_undercuts_launch_sentinel(self) -> None:
        context = Context(
            round_id=0,
            phase=0,
            hardness=0.2,
            cache_fraction=0.8,
            route_entropy=0.3,
            divergence_proxy=0.1,
            previous_score=0.0,
        )
        sentinel = RidgeRiskModel((0.20,) + (0.0,)*9, alpha=1.0)
        lower_reset = RidgeRiskModel((0.05,) + (0.0,)*9, alpha=1.0)
        higher_reset = RidgeRiskModel((0.35,) + (0.0,)*9, alpha=1.0)
        self.assertAlmostEqual(
            GuardedRiskModel(lower_reset, sentinel).score(context), 0.20
        )
        self.assertAlmostEqual(
            GuardedRiskModel(higher_reset, sentinel).score(context), 0.35
        )

    def test_block_loss_is_one_coupled_severity(self) -> None:
        severity = aggregate_block_severity(
            (0.0, 0.02, 0.08, 0.16),
            row_weights=(1.0, 1.2, 1.5, 2.0),
            collapse=False,
        )
        self.assertAlmostEqual(severity, 0.02*1.2 + 0.08*1.5 + 0.16*2.0)
        self.assertEqual(aggregate_block_severity((0.0, 0.0), collapse=True), 1.0)

    def test_sequential_feature_acquisition_never_uses_unsafe_fast_leaf(self) -> None:
        states = {
            "root": FeatureState(
                "root", fast_gain_ms=120.0, risk_upper=0.4, certificate_valid=False,
                acquisitions=(FeatureAcquisition(
                    "partial_exact", 10.0,
                    (FeatureOutcome(0.8, "safe"), FeatureOutcome(0.2, "unsafe")),
                ),),
            ),
            "safe": FeatureState("safe", fast_gain_ms=100.0, risk_upper=0.03, certificate_valid=True),
            "unsafe": FeatureState("unsafe", fast_gain_ms=150.0, risk_upper=0.5, certificate_valid=True),
        }
        value, entries = solve_sequential_feature_acquisition(states, root="root", epsilon=0.08)
        by_state = {entry.state: entry.action for entry in entries}
        self.assertAlmostEqual(value, 70.0)
        self.assertEqual(by_state["root"], "acquire:partial_exact")
        self.assertEqual(by_state["safe"], "fast")
        self.assertEqual(by_state["unsafe"], "exact")

        cyclic = {
            "a": FeatureState("a", 0.0, 1.0, False, (
                FeatureAcquisition("loop", 1.0, (FeatureOutcome(1.0, "a"),)),
            )),
        }
        with self.assertRaises(ValueError):
            solve_sequential_feature_acquisition(cyclic, root="a", epsilon=0.08)

    def test_counter_based_audit_coin_is_replayable_and_domain_separated(self) -> None:
        key = b"unit-test-secret"
        first = counter_based_audit_uniform(
            key, request_id="r", round_id=7, state_fingerprint=self.fingerprint
        )
        second = counter_based_audit_uniform(
            key, request_id="r", round_id=7, state_fingerprint=self.fingerprint
        )
        other = counter_based_audit_uniform(
            key,
            request_id="r",
            round_id=7,
            state_fingerprint=self.fingerprint,
            domain="other-domain",
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertGreater(first, 0.0)
        self.assertLess(first, 1.0)


    def test_certificate_fingerprint_binds_audit_and_reset_semantics(self) -> None:
        kwargs = dict(
            controller_digest="controller",
            feature_schema_digest="features",
            verifier_digest="verifier",
            loss_schema_digest="loss",
            threshold=0.05,
            information_price=10.0,
            audit_protocol_version="audit-v1",
            block_semantics_version="block-v1",
            cache_transition_version="cache-v1",
            calibration_epoch="epoch-0",
            supported_keys=("cell",),
            max_severity=1.0,
            q_min=0.1,
            q_max=0.5,
            cost_model_digest="cost-v1",
            reset_policy_digest="reset-v1",
        )
        base = build_certificate_fingerprint(**kwargs)
        self.assertNotEqual(
            base, build_certificate_fingerprint(**{**kwargs, "q_max": 0.6})
        )
        self.assertNotEqual(
            base, build_certificate_fingerprint(
                **{**kwargs, "cost_model_digest": "cost-v2"}
            )
        )
        self.assertNotEqual(
            base, build_certificate_fingerprint(
                **{**kwargs, "reset_policy_digest": "reset-v2"}
            )
        )

    def test_runtime_fingerprint_binds_propensity_and_state(self) -> None:
        kwargs = dict(
            certificate_fingerprint=self.fingerprint,
            request_id="req",
            round_id=3,
            prefix_digest=canonical_fingerprint({"prefix": 1}),
            target_logits_digest=canonical_fingerprint({"target": 1}),
            draft_logits_digest=canonical_fingerprint({"draft": 1}),
            route_state_digest=canonical_fingerprint({"route": 1}),
            cache_state_digest=canonical_fingerprint({"cache": 1}),
            hidden_summary_digest=canonical_fingerprint({"hidden": 1}),
            history_digest=canonical_fingerprint({"history": 1}),
            support_key="cell",
            score=0.02,
            threshold=0.03,
            audit_probability=0.2,
            severity_bound=1.0,
        )
        first = build_runtime_state_fingerprint(**kwargs)
        second = build_runtime_state_fingerprint(**{**kwargs, "audit_probability": 0.21})
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_reset_reason_is_fail_closed(self) -> None:
        reason = certificate_reset_reason(
            expected_fingerprint=self.fingerprint,
            observed_fingerprint="changed",
            support_key="cell",
            supported_keys=("cell",),
            audit_probability=0.2,
            q_min=0.1,
            oldest_pending_delay=0,
            max_label_delay=4,
            drift_alarm=False,
            model_modified=False,
            online_risk_upper=0.01,
            epsilon=0.08,
        )
        self.assertEqual(reason, "state_fingerprint_mismatch")
        self.assertIsNone(certificate_reset_reason(
            expected_fingerprint=self.fingerprint,
            observed_fingerprint=self.fingerprint,
            support_key="cell",
            supported_keys=("cell",),
            audit_probability=0.2,
            q_min=0.1,
            oldest_pending_delay=1,
            max_label_delay=4,
            drift_alarm=False,
            model_modified=False,
            online_risk_upper=0.01,
            epsilon=0.08,
        ))


if __name__ == "__main__":
    unittest.main()
