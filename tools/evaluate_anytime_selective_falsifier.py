#!/usr/bin/env python3
"""CPU-only evaluation of a feedback-safe selective falsifier.

The controlled process contains all failure modes requested by Problem 8:
continuous covariate drift, one abrupt change point, policy-dependent future
contexts and caches, selectively observed counterfactual labels, outcome-
dependent label delay, abstention, and a rare persistent catastrophic mode.
DFlash rows are generated as one causally coupled block and receive one audit
coin and one severity label.

Compared policies:

* ``naive_calibration`` reuses its calibration rows to tune a threshold;
* ``split_conformal`` freezes a nominal residual percentile;
* ``importance_weighted_nominal`` randomizes audits but repeatedly peeks at a
  fixed-time IID bound;
* ``anytime_random_audit`` freezes an arbitrarily tuned historical policy, then
  uses fresh logged propensities, an audit-capture e-process, fail-closed support
  checks, and a safe exact-shadow reset with a summable error allocation and the
  unspent global pathwise reserve.

This artifact is a controlled mathematical test, not a hardware throughput
claim for GLM-5.3-Flash.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence
from collections import deque

import numpy as np

from anytime_selective_falsifier import (
    AuditRecord,
    CalibrationRow,
    RandomAuditLedger,
    ThresholdAuditPlan,
    aggregate_block_severity,
    build_certificate_fingerprint,
    build_runtime_state_fingerprint,
    canonical_fingerprint,
    counter_based_audit_uniform,
    indistinguishable_environment_demo,
    optimal_audit_probability,
    pathwise_reset_reserve,
    runtime_support_key,
    select_historical_screen_threshold_and_audit_plan,
)

MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def counter_uniform(seed: int, index: int, channel: int) -> float:
    value = (int(seed) & MASK64) ^ ((int(index) + 1) * 0xD2B74407B1CE6E93 & MASK64)
    value ^= ((int(channel) + 17) * 0xCA5A826395121157) & MASK64
    return (_splitmix64(value) + 0.5) / 2**64


def counter_normal(seed: int, index: int, channel: int) -> float:
    u1 = max(counter_uniform(seed, index, channel), 1e-15)
    u2 = counter_uniform(seed, index, channel + 1)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class EvaluationConfig:
    epsilon: float = 0.060
    delta: float = 0.050
    max_severity: float = 1.0
    q_min: float = 0.18
    q_max: float = 0.48
    startup_budget: float = 60.0
    rounds: int = 12000
    change_round: int = 4200
    reset_rows: int = 2400
    calibration_rows: int = 15000
    prompt_length: int = 48
    max_label_delay: int = 10
    online_monitor_weight: float = 1800.0
    min_risk_prefix_weight: float = 100.0
    cache_shadow_price_ms: float = 180.0
    initial_empirical_screen_margin: float = 0.004
    reset_empirical_screen_margin: float = 0.015
    # Zero disables support-miss-triggered retraining.  Unsupported states still
    # fail closed to exact.  The controlled experiment resets on the declared
    # traffic change point, label-delay failure, or an anytime risk alarm.
    support_reset_streak: int = 0


@dataclass
class ProcessState:
    cache_fraction: float = 0.74
    divergence_debt: float = 0.0
    catastrophic_mode: float = 0.0
    previous_score: float = 0.0


@dataclass(frozen=True)
class Context:
    round_id: int
    phase: int
    hardness: float
    cache_fraction: float
    route_entropy: float
    divergence_proxy: float
    previous_score: float


@dataclass(frozen=True)
class PotentialOutcome:
    severity: float
    catastrophic: bool
    first_bad_row: int | None
    row_losses: tuple[float, ...]
    exact_ms: float
    fast_ms: float
    audit_ms: float
    next_cache_exact: float
    next_cache_fast: float
    next_cache_audit: float


@dataclass(frozen=True)
class RawCalibrationRow:
    context: Context
    outcome: PotentialOutcome


class ControlledFeedbackProcess:
    def __init__(self, *, seed: int, config: EvaluationConfig) -> None:
        self.seed = int(seed)
        self.config = config
        self.state = ProcessState()

    def context(self, round_id: int) -> Context:
        cfg = self.config
        phase = int(round_id >= cfg.change_round)
        prompt_id = round_id // cfg.prompt_length
        prompt_u = counter_uniform(self.seed, prompt_id, 10)
        prompt_hardness = 0.04 + 0.72 * prompt_u**1.65
        within_prompt = (round_id % cfg.prompt_length) / max(cfg.prompt_length - 1, 1)
        slow_drift = 0.055 * (round_id / max(cfg.rounds, 1)) + 0.025 * math.sin(round_id / 311.0)
        phase_shift = 0.035 * phase
        noise = 0.035 * counter_normal(self.seed, round_id, 20)
        hardness = float(np.clip(
            prompt_hardness
            + slow_drift
            + phase_shift
            + 0.22 * self.state.divergence_debt
            + 0.17 * self.state.catastrophic_mode
            + 0.035 * within_prompt
            + noise,
            0.0,
            1.0,
        ))
        entropy = float(np.clip(
            0.12
            + 0.62 * hardness
            + 0.24 * self.state.divergence_debt
            + 0.030 * phase
            + 0.045 * counter_normal(self.seed, round_id, 22),
            0.0,
            1.0,
        ))
        divergence = float(np.clip(
            0.04
            + 0.63 * self.state.divergence_debt
            + 0.29 * self.state.catastrophic_mode
            + 0.12 * hardness
            + 0.035 * counter_normal(self.seed, round_id, 24),
            0.0,
            1.0,
        ))
        return Context(
            round_id=round_id,
            phase=phase,
            hardness=hardness,
            cache_fraction=self.state.cache_fraction,
            route_entropy=entropy,
            divergence_proxy=divergence,
            previous_score=self.state.previous_score,
        )

    def potential(self, context: Context) -> PotentialOutcome:
        h = context.hardness
        e = context.route_entropy
        d = context.divergence_proxy
        miss = 1.0 - context.cache_fraction
        phase = context.phase
        mode = self.state.catastrophic_mode

        p_cat = sigmoid(-11.2 + 4.1*h + 2.7*e + 3.0*d + 3.8*mode + 5.90*phase)
        p_hard = sigmoid(-8.2 + 3.3*h + 1.8*e + 2.4*d + 2.0*mode + 2.45*phase)
        p_benign = sigmoid(-4.2 + 2.0*h + 0.9*e + 1.1*d + 0.8*miss + 0.55*phase)
        draw = counter_uniform(self.seed, context.round_id, 40)
        catastrophic = draw < p_cat
        hard_failure = (not catastrophic) and draw < p_cat + p_hard
        benign = (not catastrophic and not hard_failure) and draw < p_cat + p_hard + p_benign

        first_bad: int | None = None
        row_losses = np.zeros(4, dtype=float)
        if catastrophic or hard_failure or benign:
            first_bad = min(3, int(counter_uniform(self.seed, context.round_id, 41) * 4))
            if hard_failure:
                base = 0.055 + 0.025 * h
            elif benign:
                base = 0.007 + 0.006 * h
            else:
                base = 0.018
            for row in range(first_bad, 4):
                # Later labels are consequences of the first divergence, not
                # independent examples.
                row_losses[row] = base * (1.0 + 0.65 * (row - first_bad))
        elif counter_uniform(self.seed, context.round_id, 42) < 0.12:
            row_losses[-1] = 0.0025 * (1.0 + h)
        severity = aggregate_block_severity(
            row_losses,
            row_weights=(1.0, 1.25, 1.55, 1.95),
            collapse=catastrophic,
            collapse_weight=1.0,
            bound=1.0,
        )

        jitter = 7.0 * counter_normal(self.seed, context.round_id, 50)
        exact_ms = max(360.0, 610.0 + 135.0*h + 92.0*miss + 30.0*e + jitter)
        fast_ms = max(120.0, 205.0 + 58.0*h + 46.0*miss + 24.0*e + 0.35*jitter)
        rollback = 34.0 + 48.0*miss + 22.0*e
        audit_ms = exact_ms + 0.62*fast_ms + rollback
        next_exact = float(np.clip(context.cache_fraction + 0.115 - 0.025*e, 0.0, 1.0))
        next_fast = float(np.clip(
            context.cache_fraction - 0.052 - 0.10*severity - 0.035*e,
            0.0,
            1.0,
        ))
        next_audit = float(np.clip(
            context.cache_fraction + 0.048 - 0.045*e - 0.018*miss,
            0.0,
            1.0,
        ))
        return PotentialOutcome(
            severity=severity,
            catastrophic=catastrophic,
            first_bad_row=first_bad,
            row_losses=tuple(float(x) for x in row_losses),
            exact_ms=exact_ms,
            fast_ms=fast_ms,
            audit_ms=audit_ms,
            next_cache_exact=next_exact,
            next_cache_fast=next_fast,
            next_cache_audit=next_audit,
        )

    def transition(self, context: Context, outcome: PotentialOutcome, action: str, score: float) -> None:
        if action == "fast":
            self.state.cache_fraction = outcome.next_cache_fast
            self.state.divergence_debt = float(np.clip(
                0.80*self.state.divergence_debt + 0.42*outcome.severity + 0.025*context.hardness,
                0.0,
                1.0,
            ))
            if outcome.catastrophic:
                self.state.catastrophic_mode = 1.0
            else:
                self.state.catastrophic_mode *= 0.965
        elif action in {"audit", "reset_shadow"}:
            self.state.cache_fraction = outcome.next_cache_audit
            self.state.divergence_debt *= 0.69
            recover = counter_uniform(self.seed, context.round_id, 60) < 0.30
            self.state.catastrophic_mode *= 0.18 if recover else 0.78
        elif action == "exact":
            self.state.cache_fraction = outcome.next_cache_exact
            self.state.divergence_debt *= 0.61
            recover = counter_uniform(self.seed, context.round_id, 61) < 0.38
            self.state.catastrophic_mode *= 0.12 if recover else 0.72
        else:
            raise ValueError(action)
        if (context.round_id + 1) % self.config.prompt_length == 0:
            self.state.divergence_debt *= 0.62
            self.state.catastrophic_mode *= 0.72
        self.state.previous_score = float(np.clip(score, 0.0, 1.0))


FEATURE_NAMES = (
    "bias",
    "hardness",
    "route_entropy",
    "cache_miss",
    "divergence_proxy",
    "previous_score",
    "hardness_x_divergence",
    "entropy_x_cache_miss",
    "hardness_sq",
    "divergence_sq",
)


def feature_vector(context: Context) -> np.ndarray:
    h = context.hardness
    e = context.route_entropy
    m = 1.0 - context.cache_fraction
    d = context.divergence_proxy
    p = context.previous_score
    return np.asarray((1.0, h, e, m, d, p, h*d, e*m, h*h, d*d), dtype=float)


@dataclass(frozen=True)
class RidgeRiskModel:
    coefficients: tuple[float, ...]
    alpha: float

    def score(self, context: Context) -> float:
        value = float(np.dot(np.asarray(self.coefficients), feature_vector(context)))
        return float(np.clip(value, 0.0, 1.0))

    @property
    def digest(self) -> str:
        payload = json.dumps({"coefficients": self.coefficients, "alpha": self.alpha}, sort_keys=True)
        import hashlib
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class GuardedRiskModel:
    """Reset score that cannot undercut the immutable launch sentinel."""

    adaptive: RidgeRiskModel
    sentinel: RidgeRiskModel

    def score(self, context: Context) -> float:
        return max(self.adaptive.score(context), self.sentinel.score(context))

    @property
    def digest(self) -> str:
        import hashlib
        payload = json.dumps({
            "kind": "max-reset-sentinel",
            "adaptive": self.adaptive.digest,
            "sentinel": self.sentinel.digest,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def severity_training_weights(rows: Sequence[RawCalibrationRow]) -> np.ndarray:
    """Loss weights used only to fit the tiny severity score.

    A collapse has unit severity and must dominate benign row mismatches during
    model fitting.  The certificate does not rely on this model being correct;
    weighting only makes the frozen proposal policy economically less reckless.
    """

    y = np.asarray([r.outcome.severity for r in rows], dtype=float)
    weights = np.ones_like(y)
    weights[(y >= 0.05) & (y < 0.99)] = 10.0
    weights[y >= 0.99] = 160.0
    return weights


def fit_ridge(rows: Sequence[RawCalibrationRow], alpha: float) -> RidgeRiskModel:
    x = np.vstack([feature_vector(r.context) for r in rows])
    y = np.asarray([r.outcome.severity for r in rows], dtype=float)
    weights = severity_training_weights(rows)
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        x.T @ (weights[:, None] * x) + penalty,
        x.T @ (weights * y),
    )
    return RidgeRiskModel(tuple(float(v) for v in coefficients), float(alpha))


def choose_ridge_model(
    train_rows: Sequence[RawCalibrationRow],
    design_rows: Sequence[RawCalibrationRow],
    alphas: Sequence[float] = (0.01, 0.05, 0.20, 1.0, 5.0),
) -> RidgeRiskModel:
    candidates = [fit_ridge(train_rows, alpha) for alpha in alphas]
    def loss(model: RidgeRiskModel) -> float:
        pred = np.asarray([model.score(r.context) for r in design_rows])
        y = np.asarray([r.outcome.severity for r in design_rows])
        weights = severity_training_weights(design_rows)
        return float(np.average((pred-y)**2, weights=weights))
    return min(candidates, key=loss)


def estimated_economic_gains(
    context: Context,
    cfg: EvaluationConfig,
    *,
    severity_score: float,
) -> tuple[float, float]:
    """Predict exact-relative gains from pre-coin information only.

    The realized audit cost and realized cache transition are used for reported
    throughput, but they cannot be used to choose ``q_t``: both may reveal the
    fast potential outcome.  This deterministic cost model uses the causal
    context and the already-frozen controller score.  Consequently the audit
    propensity is predictable and the certificate weights are label-free.
    """

    h = context.hardness
    e = context.route_entropy
    miss = 1.0 - context.cache_fraction
    predicted_severity = float(np.clip(severity_score, 0.0, cfg.max_severity))
    exact_ms = max(360.0, 610.0 + 135.0*h + 92.0*miss + 30.0*e)
    fast_ms = max(120.0, 205.0 + 58.0*h + 46.0*miss + 24.0*e)
    rollback_ms = 34.0 + 48.0*miss + 22.0*e
    audit_ms = exact_ms + 0.62*fast_ms + rollback_ms
    next_exact = float(np.clip(context.cache_fraction + 0.115 - 0.025*e, 0.0, 1.0))
    next_fast = float(np.clip(
        context.cache_fraction - 0.052 - 0.10*predicted_severity - 0.035*e,
        0.0,
        1.0,
    ))
    next_audit = float(np.clip(
        context.cache_fraction + 0.048 - 0.045*e - 0.018*miss,
        0.0,
        1.0,
    ))
    future_fast = cfg.cache_shadow_price_ms * (next_fast - next_exact)
    future_audit = cfg.cache_shadow_price_ms * (next_audit - next_exact)
    fast_gain = exact_ms - fast_ms + future_fast
    audit_gain = exact_ms - audit_ms + future_audit
    return float(fast_gain), float(audit_gain)


def simulation_support_key(context: Context, score: float) -> str:
    """Coarse predeclared support cell for the synthetic runtime."""

    return runtime_support_key(
        score=score,
        cache_fraction=context.cache_fraction,
        route_entropy=context.route_entropy,
        divergence_proxy=context.divergence_proxy,
        score_bins=(0.02, 0.05, 0.10),
        cache_bins=(),
        entropy_bins=(),
        divergence_bins=(),
    )


def as_calibration_rows(
    rows: Sequence[RawCalibrationRow],
    model: RidgeRiskModel | GuardedRiskModel,
    cfg: EvaluationConfig,
) -> list[CalibrationRow]:
    result: list[CalibrationRow] = []
    for row in rows:
        score = model.score(row.context)
        fast_gain, audit_gain = estimated_economic_gains(
            row.context,
            cfg,
            severity_score=score,
        )
        support = simulation_support_key(row.context, score)
        result.append(CalibrationRow(
            score=score,
            severity=row.outcome.severity,
            fast_gain_ms=fast_gain,
            audit_gain_ms=audit_gain,
            severity_bound=cfg.max_severity,
            support_key=support,
        ))
    return result


def collect_exact_shadow_rows(
    *,
    seed: int,
    count: int,
    cfg: EvaluationConfig,
    start_round: int = 0,
    initial_state: ProcessState | None = None,
    pilot_feedback: bool = True,
) -> list[RawCalibrationRow]:
    calibration_cfg = replace(cfg, rounds=max(count, 1), change_round=start_round + count + 1)
    process = ControlledFeedbackProcess(seed=seed, config=calibration_cfg)
    if initial_state is not None:
        process.state = ProcessState(**asdict(initial_state))
    rows: list[RawCalibrationRow] = []
    for offset in range(count):
        round_id = start_round + offset
        context = process.context(round_id)
        outcome = process.potential(context)
        rows.append(RawCalibrationRow(context, outcome))
        if pilot_feedback:
            pilot_uniform = counter_uniform(seed, round_id, 72)
            pilot_action = "fast" if pilot_uniform < 0.58 else "reset_shadow"
            pilot_score = float(np.clip(
                0.010 + 0.035*context.hardness + 0.055*context.divergence_proxy
                + 0.020*(1-context.cache_fraction),
                0.0, 1.0,
            ))
            process.transition(context, outcome, pilot_action, score=pilot_score)
        else:
            process.transition(context, outcome, "reset_shadow", score=0.0)
    return rows


@dataclass(frozen=True)
class CalibrationBundle:
    our_model: RidgeRiskModel
    our_plan: ThresholdAuditPlan
    naive_model: RidgeRiskModel
    naive_threshold: float
    naive_reported_risk: float
    conformal_model: RidgeRiskModel
    conformal_residual_quantile: float
    importance_threshold: float
    importance_initial_count: int
    importance_initial_sum: float
    importance_initial_sum_sq: float
    initial_rows: int


def _score_quantile_grid(scores: Sequence[float], count: int = 9) -> tuple[float, ...]:
    values = np.asarray(scores, dtype=float)
    qs = np.linspace(0.12, 0.88, count)
    return tuple(sorted(set(float(np.quantile(values, q)) for q in qs)))


def build_calibration_bundle(cfg: EvaluationConfig, seed: int = 20260831) -> CalibrationBundle:
    rows = collect_exact_shadow_rows(seed=seed, count=cfg.calibration_rows, cfg=cfg)
    n = len(rows)
    cut1, cut2 = n // 3, 2*n // 3
    train, design, later_screen = rows[:cut1], rows[cut1:cut2], rows[cut2:]

    our_model = choose_ridge_model(train, design)
    design_cal = as_calibration_rows(design, our_model, cfg)
    screen_cal = as_calibration_rows(later_screen, our_model, cfg)
    thresholds = _score_quantile_grid([r.score for r in design_cal], 10)
    information_prices = (10.0, 18.0, 32.0, 56.0, 96.0, 160.0)
    plan = select_historical_screen_threshold_and_audit_plan(
        design_cal,
        screen_cal,
        thresholds=thresholds,
        information_prices=information_prices,
        epsilon=cfg.epsilon,
        q_min=cfg.q_min,
        q_max=cfg.q_max,
        empirical_margin=cfg.initial_empirical_screen_margin,
        min_selected_mass=300.0,
        min_support_count=8,
    )
    if plan is None:
        raise RuntimeError("historical stress screen found no admissible plan")

    # Naive calibration selects and evaluates on the same full sample.
    naive_model = choose_ridge_model(rows[:n//2], rows[n//2:])
    naive_cal = as_calibration_rows(rows, naive_model, cfg)
    naive_candidates = _score_quantile_grid([r.score for r in naive_cal], 18)
    naive_options: list[tuple[float, float, float]] = []
    for threshold in naive_candidates:
        selected = [r for r in naive_cal if r.score <= threshold]
        if not selected:
            continue
        risk = float(np.mean([r.severity for r in selected]))
        reward = sum(r.fast_gain_ms for r in selected) / len(naive_cal)
        if risk <= cfg.epsilon:
            naive_options.append((reward, threshold, risk))
    if not naive_options:
        raise RuntimeError("naive calibration found no threshold")
    _, naive_threshold, naive_risk = max(naive_options)

    # Split conformal uses a train/calibration split and a nominal 90% residual
    # quantile.  It is not an anytime selective-risk certificate.
    half = n // 2
    conformal_model = choose_ridge_model(rows[:half//2], rows[half//2:half])
    residuals = np.asarray([
        r.outcome.severity - conformal_model.score(r.context) for r in rows[half:]
    ])
    alpha = 0.10
    rank = min(len(residuals)-1, math.ceil((len(residuals)+1)*(1-alpha))-1)
    conformal_q = float(np.sort(residuals)[max(rank, 0)])

    importance_rows = collect_exact_shadow_rows(seed=20260831, count=2500, cfg=cfg)
    importance_initial = [
        row.outcome.severity for row in importance_rows
        if naive_model.score(row.context) <= naive_threshold
    ]

    return CalibrationBundle(
        our_model=our_model,
        our_plan=plan,
        naive_model=naive_model,
        naive_threshold=float(naive_threshold),
        naive_reported_risk=float(naive_risk),
        conformal_model=conformal_model,
        conformal_residual_quantile=conformal_q,
        importance_threshold=float(naive_threshold),
        importance_initial_count=len(importance_initial),
        importance_initial_sum=float(sum(importance_initial)),
        importance_initial_sum_sq=float(sum(y*y for y in importance_initial)),
        initial_rows=n,
    )


@dataclass
class PendingLabel:
    due_round: int
    ledger: RandomAuditLedger | None
    round_id: int
    severity: float
    consumer: str
    audit_probability: float = 1.0


@dataclass
class EpochState:
    epoch_index: int
    model: RidgeRiskModel | GuardedRiskModel
    plan: ThresholdAuditPlan
    fingerprint: str
    ledger: RandomAuditLedger
    delta: float
    start_round: int
    traffic_phase: int
    active: bool = True
    closed_committed_upper: float | None = None
    closed_fast_commits: int | None = None


@dataclass
class RunMetrics:
    seed: int
    policy: str
    rounds: int
    total_ms: float = 0.0
    matched_exact_ms: float = 0.0
    fast_commits: int = 0
    audits: int = 0
    exact_actions: int = 0
    reset_shadow_actions: int = 0
    audit_cost_ms: float = 0.0
    reset_cost_ms: float = 0.0
    committed_loss: float = 0.0
    committed_catastrophes: int = 0
    intended_loss: float = 0.0
    intended_weight: float = 0.0
    proposals: int = 0
    any_mean_risk_violation: bool = False
    any_false_safe: bool = False
    any_local_risk_violation: bool = False
    any_local_false_safe: bool = False
    max_local_intended_risk: float = 0.0
    any_reported_coverage_failure: bool = False
    any_committed_coverage_failure: bool = False
    any_pathwise_budget_violation: bool = False
    resets: int = 0
    first_reset_round: int | None = None
    labels_observed: int = 0
    max_pending_labels: int = 0
    final_reported_upper: float = math.inf
    final_committed_upper: float = math.inf
    certificate_available_rounds: int = 0
    support_abstentions: int = 0
    reset_rounds: list[int] = field(default_factory=list)
    reset_reasons: list[str] = field(default_factory=list)

    def finalize(self, cfg: EvaluationConfig) -> dict[str, Any]:
        fast_risk = self.committed_loss / max(self.fast_commits, 1)
        intended_risk = self.intended_loss / max(self.intended_weight, 1e-12)
        throughput = 1000.0 * self.rounds / self.total_ms
        exact_throughput = 1000.0 * self.rounds / self.matched_exact_ms
        return {
            **asdict(self),
            "fast_selective_risk": fast_risk,
            "intended_selective_risk": intended_risk,
            "abstention_fraction": (self.exact_actions + self.reset_shadow_actions) / self.rounds,
            "audit_fraction": self.audits / self.rounds,
            "fast_commit_fraction": self.fast_commits / self.rounds,
            "throughput_tokens_per_second": throughput,
            "matched_exact_tokens_per_second": exact_throughput,
            "throughput_reward_tokens_per_second": throughput - exact_throughput,
            "saved_ms_per_round": (self.matched_exact_ms - self.total_ms) / self.rounds,
            "audit_cost_ms_per_round": self.audit_cost_ms / self.rounds,
            "reset_cost_ms_per_round": self.reset_cost_ms / self.rounds,
            "epsilon": cfg.epsilon,
            "delta": cfg.delta,
            "startup_budget": cfg.startup_budget,
        }


def fingerprint_for_epoch(
    model: RidgeRiskModel | GuardedRiskModel,
    plan: ThresholdAuditPlan,
    *,
    epoch_index: int,
    cfg: EvaluationConfig,
) -> str:
    import hashlib
    feature_digest = hashlib.sha256("|".join(FEATURE_NAMES).encode()).hexdigest()
    cost_model_digest = canonical_fingerprint({
        "name": "synthetic-causal-audit-cost-v2",
        "cache_shadow_price_ms": cfg.cache_shadow_price_ms,
        "inputs": (
            "hardness", "cache_fraction", "route_entropy",
            "divergence_proxy", "previous_score", "severity_score",
        ),
        "forbidden_inputs": (
            "realized_severity", "realized_fast_ms", "realized_exact_ms",
            "realized_audit_ms", "realized_cache_transition",
        ),
    })
    reset_policy_digest = canonical_fingerprint({
        "name": "fail-closed-monotone-sentinel-reset-v2",
        "reset_rows": cfg.reset_rows,
        "reset_empirical_screen_margin": cfg.reset_empirical_screen_margin,
        "support_reset_streak": cfg.support_reset_streak,
        "score_guard": "max(reset_model, launch_sentinel)",
        "threshold_cap": "reset_threshold<=launch_threshold",
        "pathwise_reserve": "global-unspent-only",
    })
    return build_certificate_fingerprint(
        controller_digest=model.digest,
        feature_schema_digest=feature_digest,
        verifier_digest="synthetic-exact-block-verifier-v1",
        loss_schema_digest="severity:block-top1-0.01-hard-0.2-collapse-1.0:v1",
        threshold=plan.threshold,
        information_price=plan.information_price,
        audit_protocol_version="bernoulli-counter-coin-v1",
        block_semantics_version="four-row-stop-coupled-v1",
        cache_transition_version="synthetic-cache-feedback-v1",
        calibration_epoch=f"epoch-{epoch_index}",
        supported_keys=plan.supported_keys,
        max_severity=cfg.max_severity,
        q_min=cfg.q_min,
        q_max=cfg.q_max,
        cost_model_digest=cost_model_digest,
        reset_policy_digest=reset_policy_digest,
    )


def make_epoch(
    model: RidgeRiskModel | GuardedRiskModel,
    plan: ThresholdAuditPlan,
    *,
    epoch_index: int,
    start_round: int,
    cfg: EvaluationConfig,
    traffic_phase: int = 0,
    startup_budget: float | None = None,
) -> EpochState:
    # Summable allocation: delta_e = delta / 2^(e+2), leaving half for other
    # operational alarms.  The old ledger is retained after a reset.
    epoch_delta = cfg.delta / (2 ** (epoch_index + 2))
    fingerprint = fingerprint_for_epoch(model, plan, epoch_index=epoch_index, cfg=cfg)
    ledger = RandomAuditLedger(
        delta=epoch_delta,
        epsilon=cfg.epsilon,
        startup_budget=(cfg.startup_budget if startup_budget is None else startup_budget),
        max_severity=cfg.max_severity,
        q_min=cfg.q_min,
        state_fingerprint=fingerprint,
    )
    return EpochState(
        epoch_index=epoch_index,
        model=model,
        plan=plan,
        fingerprint=fingerprint,
        ledger=ledger,
        delta=epoch_delta,
        start_round=start_round,
        traffic_phase=int(traffic_phase),
    )


def rebuild_epoch_from_reset(
    rows: Sequence[RawCalibrationRow],
    *,
    epoch_index: int,
    start_round: int,
    cfg: EvaluationConfig,
    startup_budget: float,
    sentinel_model: RidgeRiskModel,
    sentinel_threshold: float,
) -> EpochState | None:
    if len(rows) < cfg.reset_rows:
        return None
    n = len(rows)
    cut1, cut2 = n // 3, 2*n // 3
    train, design, later_screen = rows[:cut1], rows[cut1:cut2], rows[cut2:]
    adaptive_model = choose_ridge_model(train, design)
    model = GuardedRiskModel(adaptive=adaptive_model, sentinel=sentinel_model)
    design_cal = as_calibration_rows(design, model, cfg)
    screen_cal = as_calibration_rows(later_screen, model, cfg)
    raw_thresholds = _score_quantile_grid([r.score for r in design_cal], 6)
    thresholds = tuple(sorted(set(
        t for t in (*raw_thresholds, float(sentinel_threshold))
        if t <= sentinel_threshold + 1e-12
    )))
    if not thresholds:
        return None
    plan = select_historical_screen_threshold_and_audit_plan(
        design_cal,
        screen_cal,
        thresholds=thresholds,
        information_prices=(18.0, 40.0, 80.0, 140.0),
        epsilon=cfg.epsilon,
        q_min=cfg.q_min,
        q_max=cfg.q_max,
        empirical_margin=cfg.reset_empirical_screen_margin,
        min_selected_mass=80.0,
        min_support_count=4,
    )
    if plan is None:
        return None
    return make_epoch(
        model, plan, epoch_index=epoch_index, start_round=start_round, cfg=cfg,
        traffic_phase=rows[-1].context.phase, startup_budget=startup_budget,
    )


def aggregate_epoch_bounds(epochs: Sequence[EpochState]) -> tuple[float, float, float]:
    intended_upper = 0.0
    intended_weight = 0.0
    committed_upper = 0.0
    for epoch in epochs:
        snapshot = epoch.ledger.snapshot()
        intended_upper += snapshot.intended_loss_upper
        intended_weight += snapshot.intended_fast_weight
        committed_upper += (
            epoch.closed_committed_upper
            if epoch.closed_committed_upper is not None
            else snapshot.committed_loss_upper
        )
    return intended_upper, intended_weight, committed_upper


def pathwise_carry_for_new_epoch(
    epochs: Sequence[EpochState], cfg: EvaluationConfig
) -> float | None:
    """Carry only unspent global pathwise budget into a reset epoch."""

    uppers: list[float] = []
    commits: list[int] = []
    for epoch in epochs:
        snapshot = epoch.ledger.snapshot()
        uppers.append(
            epoch.closed_committed_upper
            if epoch.closed_committed_upper is not None
            else snapshot.committed_loss_upper
        )
        commits.append(
            epoch.closed_fast_commits
            if epoch.closed_fast_commits is not None
            else snapshot.fast_commits
        )
    return pathwise_reset_reserve(
        global_startup_budget=cfg.startup_budget,
        epsilon=cfg.epsilon,
        closed_fast_commits=commits,
        closed_committed_uppers=uppers,
    )


def runtime_fingerprint_for_round(
    *,
    seed: int,
    context: Context,
    certificate_fingerprint: str,
    score: float,
    threshold: float,
    audit_probability: float,
    severity_bound: float,
    support_key: str,
) -> str:
    context_digest = canonical_fingerprint(asdict(context))
    return build_runtime_state_fingerprint(
        certificate_fingerprint=certificate_fingerprint,
        request_id=f"synthetic-{seed}",
        round_id=context.round_id,
        prefix_digest=canonical_fingerprint({
            "request": seed, "round": context.round_id, "previous_score": context.previous_score
        }),
        target_logits_digest=canonical_fingerprint({
            "synthetic_target_summary": (context.hardness, context.route_entropy)
        }),
        draft_logits_digest=canonical_fingerprint({
            "synthetic_draft_summary": (context.divergence_proxy, context.previous_score)
        }),
        route_state_digest=canonical_fingerprint({"route_entropy": context.route_entropy}),
        cache_state_digest=canonical_fingerprint({"cache_fraction": context.cache_fraction}),
        hidden_summary_digest=context_digest,
        history_digest=canonical_fingerprint({
            "phase": context.phase, "round": context.round_id
        }),
        support_key=support_key,
        score=score,
        threshold=threshold,
        audit_probability=audit_probability,
        severity_bound=severity_bound,
    )


def _oldest_pending_delay(pending: Sequence[PendingLabel], current_round: int) -> int:
    if not pending:
        return 0
    oldest_origin = min(item.round_id for item in pending)
    return max(0, current_round - oldest_origin)


def run_policy(
    policy: str,
    *,
    seed: int,
    cfg: EvaluationConfig,
    bundle: CalibrationBundle,
) -> dict[str, Any]:
    process = ControlledFeedbackProcess(seed=seed, config=cfg)
    metrics = RunMetrics(seed=seed, policy=policy, rounds=cfg.rounds)
    pending: list[PendingLabel] = []
    importance_count = 0
    importance_sum = 0.0
    importance_sum_sq = 0.0
    reset_rows: list[RawCalibrationRow] = []
    local_window: deque[tuple[float, float, bool]] = deque()
    local_window_weight = 0.0
    local_window_loss = 0.0
    in_reset = False
    support_miss_streak = 0
    epochs: list[EpochState] = []
    active_epoch: EpochState | None = None
    if policy == "anytime_random_audit":
        active_epoch = make_epoch(bundle.our_model, bundle.our_plan, epoch_index=0, start_round=0, cfg=cfg)
        epochs.append(active_epoch)
    elif policy == "importance_weighted_nominal":
        # Reuse one frozen calibration subset as IID pseudo-evidence, then
        # repeatedly peek at the same nominal bound.
        importance_count = bundle.importance_initial_count
        importance_sum = bundle.importance_initial_sum
        importance_sum_sq = bundle.importance_initial_sum_sq

    for t in range(cfg.rounds):
        # Resolve outcome-dependent delayed labels.  Out-of-order arrivals are
        # accepted by the API; each anytime ledger only processes its longest
        # decision-ordered resolved prefix.
        due = [item for item in pending if item.due_round <= t]
        pending = [item for item in pending if item.due_round > t]
        for item in sorted(due, key=lambda x: (x.due_round, -x.round_id)):
            if item.ledger is not None:
                item.ledger.resolve(item.round_id, item.severity)
            if item.consumer == "importance":
                ipw_value = item.severity / item.audit_probability
                importance_sum += ipw_value
                importance_sum_sq += ipw_value*ipw_value
            metrics.labels_observed += 1
        metrics.max_pending_labels = max(metrics.max_pending_labels, len(pending))

        context = process.context(t)
        outcome = process.potential(context)
        metrics.matched_exact_ms += outcome.exact_ms
        action = "exact"
        score = 1.0
        certificate_claimed = False
        reported_upper = math.inf
        audit_probability = 0.0
        support_key = "invalid"
        audit_uniform = 0.9999999999999999
        runtime_fingerprint: str | None = None

        if policy == "naive_calibration":
            score = bundle.naive_model.score(context)
            certificate_claimed = score <= bundle.naive_threshold
            reported_upper = bundle.naive_reported_risk
            if certificate_claimed:
                action = "fast"

        elif policy == "split_conformal":
            score = bundle.conformal_model.score(context)
            row_upper = float(np.clip(score + bundle.conformal_residual_quantile, 0.0, 1.0))
            reported_upper = row_upper
            certificate_claimed = row_upper <= cfg.epsilon
            if certificate_claimed:
                action = "fast"

        elif policy == "importance_weighted_nominal":
            score = bundle.naive_model.score(context)
            if importance_count > 1:
                mean = importance_sum / importance_count
                variance = max(0.0, importance_sum_sq/importance_count - mean*mean)
                # Pointwise Gaussian plug-in interval, repeatedly peeked.  It
                # ignores heavy inverse-propensity tails, feedback, and delayed
                # outcome selection.
                width = 1.645*math.sqrt(variance/importance_count)
                reported_upper = min(1.0, mean + width)
            certificate_claimed = score <= bundle.importance_threshold and reported_upper <= cfg.epsilon
            if certificate_claimed:
                audit_probability = 0.20
                audit_uniform = counter_uniform(
                    seed, t, 900 + POLICY_AUDIT_CHANNEL[policy]
                )
                action = "audit" if audit_uniform < audit_probability else "fast"

        elif policy == "anytime_random_audit":
            if in_reset:
                action = "reset_shadow"
                score = 1.0
            elif active_epoch is None or not active_epoch.active:
                action = "exact"
            else:
                score = active_epoch.model.score(context)
                support_key = simulation_support_key(context, score)
                fast_gain, audit_gain = estimated_economic_gains(
                    context,
                    cfg,
                    severity_score=score,
                )
                audit_probability = optimal_audit_probability(
                    audit_opportunity_cost=fast_gain-audit_gain,
                    information_price=active_epoch.plan.information_price,
                    severity_bound=cfg.max_severity,
                    q_min=cfg.q_min,
                    q_max=cfg.q_max,
                )
                runtime_fingerprint = runtime_fingerprint_for_round(
                    seed=seed,
                    context=context,
                    certificate_fingerprint=active_epoch.fingerprint,
                    score=score,
                    threshold=active_epoch.plan.threshold,
                    audit_probability=audit_probability,
                    severity_bound=cfg.max_severity,
                    support_key=support_key,
                )
                # In production the epoch key is held by an audit broker and
                # precommitted by hash.  The synthetic key is deterministic so
                # the CPU artifact is exactly replayable.
                audit_uniform = counter_based_audit_uniform(
                    f"synthetic-audit-key-{seed}".encode(),
                    request_id=f"synthetic-{seed}",
                    round_id=t,
                    state_fingerprint=runtime_fingerprint,
                )
                snap = active_epoch.ledger.snapshot()
                local_online_alarm = (
                    snap.intended_fast_weight >= cfg.online_monitor_weight
                    and snap.intended_selective_risk_upper > cfg.epsilon
                )
                delay_alarm = _oldest_pending_delay(
                    [p for p in pending if p.ledger is active_epoch.ledger], t
                ) > cfg.max_label_delay
                support_ok = support_key in set(active_epoch.plan.supported_keys)
                threshold_ok = score <= active_epoch.plan.threshold
                pathwise_ok = active_epoch.ledger.can_commit_fast(
                    severity_bound=cfg.max_severity
                )
                if support_ok:
                    support_miss_streak = 0
                else:
                    support_miss_streak += 1
                    metrics.support_abstentions += 1
                reset_reason = None
                if context.phase != active_epoch.traffic_phase:
                    reset_reason = "traffic_epoch_change_point"
                elif (
                    cfg.support_reset_streak > 0
                    and support_miss_streak >= cfg.support_reset_streak
                ):
                    reset_reason = "persistent_unsupported_runtime_state"
                elif delay_alarm:
                    reset_reason = "audit_label_delay_exceeded"
                elif local_online_alarm:
                    reset_reason = "anytime_risk_upper_crossed"
                if reset_reason is not None:
                    active_epoch.active = False
                    active_epoch.closed_committed_upper = snap.committed_loss_upper
                    active_epoch.closed_fast_commits = snap.fast_commits
                    in_reset = True
                    reset_rows = []
                    metrics.resets += 1
                    metrics.reset_rounds.append(t)
                    metrics.reset_reasons.append(reset_reason)
                    if metrics.first_reset_round is None:
                        metrics.first_reset_round = t
                    action = "reset_shadow"
                elif threshold_ok and support_ok and pathwise_ok:
                    certificate_claimed = True
                    action = "audit" if audit_uniform < audit_probability else "fast"
                else:
                    action = "exact"

                reported_upper = snap.intended_selective_risk_upper
                metrics.final_committed_upper = snap.committed_loss_upper
        else:
            raise ValueError(policy)

        if certificate_claimed:
            metrics.certificate_available_rounds += 1

        if policy == "importance_weighted_nominal" and certificate_claimed:
            importance_count += 1
            # Every proposal contributes an observed IPW value of zero until an
            # audit label arrives.  A delayed audited label then adds Y/q.

        # The policy's intended-commit target is predictable before the coin.
        if certificate_claimed:
            q_for_target = audit_probability if policy in {
                "importance_weighted_nominal", "anytime_random_audit"
            } else 0.0
            intended_weight = 1.0 - q_for_target
            weighted_loss = intended_weight * outcome.severity
            metrics.intended_weight += intended_weight
            metrics.intended_loss += weighted_loss
            metrics.proposals += 1
            local_window.append((intended_weight, weighted_loss, certificate_claimed))
            local_window_weight += intended_weight
            local_window_loss += weighted_loss
            while local_window_weight > 420.0 and local_window:
                old_weight, old_loss, _ = local_window.popleft()
                local_window_weight -= old_weight
                local_window_loss -= old_loss

        if action == "fast":
            metrics.total_ms += outcome.fast_ms
            metrics.fast_commits += 1
            metrics.committed_loss += outcome.severity
            metrics.committed_catastrophes += int(outcome.catastrophic)
        elif action == "audit":
            metrics.total_ms += outcome.audit_ms
            metrics.audits += 1
            metrics.audit_cost_ms += outcome.audit_ms
        elif action == "reset_shadow":
            metrics.total_ms += outcome.audit_ms
            metrics.reset_shadow_actions += 1
            metrics.reset_cost_ms += outcome.audit_ms
            reset_rows.append(RawCalibrationRow(context, outcome))
        else:
            metrics.total_ms += outcome.exact_ms
            metrics.exact_actions += 1

        if policy == "importance_weighted_nominal" and action == "audit":
            delay = min(cfg.max_label_delay, 1 + int(4*context.hardness) + 3*int(outcome.catastrophic))
            pending.append(PendingLabel(
                t+delay, None, t, outcome.severity, "importance", audit_probability
            ))

        if policy == "anytime_random_audit" and certificate_claimed and active_epoch is not None:
            audited = action == "audit"
            assert runtime_fingerprint is not None
            delay = (
                min(
                    cfg.max_label_delay,
                    1 + int(4*context.hardness) + 3*int(outcome.catastrophic),
                )
                if audited else 0
            )
            record = AuditRecord(
                round_id=t,
                audit_probability=audit_probability,
                severity_bound=cfg.max_severity,
                audited=audited,
                audit_uniform=audit_uniform,
                state_fingerprint=active_epoch.fingerprint,
                runtime_state_fingerprint=runtime_fingerprint,
                support_key=support_key,
                label=None,
                label_due_round=t+delay if audited else None,
            )
            active_epoch.ledger.append(record)
            if audited:
                pending.append(PendingLabel(
                    due_round=t+delay,
                    ledger=active_epoch.ledger,
                    round_id=t,
                    severity=outcome.severity,
                    consumer="anytime",
                    audit_probability=audit_probability,
                ))

        process.transition(context, outcome, action, score)

        # Complete a safe reset only after three disjoint chronological pieces.
        if policy == "anytime_random_audit" and in_reset and len(reset_rows) >= cfg.reset_rows:
            new_index = len(epochs)
            carried_reserve = pathwise_carry_for_new_epoch(epochs, cfg)
            new_epoch = (
                rebuild_epoch_from_reset(
                    reset_rows,
                    epoch_index=new_index,
                    start_round=t+1,
                    cfg=cfg,
                    startup_budget=carried_reserve,
                    sentinel_model=bundle.our_model,
                    sentinel_threshold=bundle.our_plan.threshold,
                )
                if carried_reserve is not None else None
            )
            in_reset = False
            reset_rows = []
            if new_epoch is not None:
                active_epoch = new_epoch
                epochs.append(new_epoch)
            else:
                active_epoch = None

        true_intended_risk = metrics.intended_loss / max(metrics.intended_weight, 1e-12)
        if metrics.intended_weight >= cfg.min_risk_prefix_weight:
            violation = true_intended_risk > cfg.epsilon + 1e-12
            metrics.any_mean_risk_violation |= violation
            metrics.any_false_safe |= violation and certificate_claimed
        if local_window_weight >= 180.0:
            local_risk = local_window_loss / max(local_window_weight, 1e-12)
            metrics.max_local_intended_risk = max(metrics.max_local_intended_risk, local_risk)
            local_violation = local_risk > cfg.epsilon + 1e-12
            metrics.any_local_risk_violation |= local_violation
            metrics.any_local_false_safe |= local_violation and certificate_claimed

        if policy == "naive_calibration" and metrics.intended_weight > 0:
            metrics.any_reported_coverage_failure |= true_intended_risk > reported_upper + 1e-12
        elif policy == "split_conformal" and certificate_claimed:
            metrics.any_reported_coverage_failure |= outcome.severity > reported_upper + 1e-12
        elif policy == "importance_weighted_nominal" and metrics.intended_weight > 0 and math.isfinite(reported_upper):
            metrics.any_reported_coverage_failure |= true_intended_risk > reported_upper + 1e-12
        elif policy == "anytime_random_audit":
            intended_upper, intended_weight, committed_upper = aggregate_epoch_bounds(epochs)
            true_intended_total = metrics.intended_loss
            metrics.any_reported_coverage_failure |= true_intended_total > intended_upper + 1e-10
            metrics.any_committed_coverage_failure |= metrics.committed_loss > committed_upper + 1e-10
            metrics.final_reported_upper = intended_upper / max(intended_weight, 1e-12)
            metrics.final_committed_upper = committed_upper

        metrics.any_pathwise_budget_violation |= (
            metrics.committed_loss
            > cfg.startup_budget + cfg.epsilon*metrics.fast_commits + 1e-12
        )

    # Resolve every remaining label so final reported bounds use complete data.
    for item in sorted(pending, key=lambda x: (x.due_round, x.round_id)):
        if item.ledger is not None:
            item.ledger.resolve(item.round_id, item.severity)
        if item.consumer == "importance":
            ipw_value = item.severity / item.audit_probability
            importance_sum += ipw_value
            importance_sum_sq += ipw_value*ipw_value
        metrics.labels_observed += 1
    if policy == "anytime_random_audit":
        intended_upper, intended_weight, committed_upper = aggregate_epoch_bounds(epochs)
        metrics.final_reported_upper = intended_upper / max(intended_weight, 1e-12)
        metrics.final_committed_upper = committed_upper
        metrics.any_reported_coverage_failure |= metrics.intended_loss > intended_upper + 1e-10
        metrics.any_committed_coverage_failure |= metrics.committed_loss > committed_upper + 1e-10
    elif policy == "importance_weighted_nominal":
        if importance_count > 1:
            mean = importance_sum / importance_count
            variance = max(0.0, importance_sum_sq/importance_count - mean*mean)
            metrics.final_reported_upper = min(
                1.0, mean + 1.645*math.sqrt(variance/importance_count)
            )
    elif policy == "naive_calibration":
        metrics.final_reported_upper = bundle.naive_reported_risk
    elif policy == "split_conformal":
        metrics.final_reported_upper = bundle.conformal_residual_quantile
    return metrics.finalize(cfg)


POLICY_AUDIT_CHANNEL = {
    "naive_calibration": 1,
    "split_conformal": 2,
    "importance_weighted_nominal": 3,
    "anytime_random_audit": 4,
}


POLICIES = (
    "naive_calibration",
    "split_conformal",
    "importance_weighted_nominal",
    "anytime_random_audit",
)


def summarize_runs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    boolean_fields = (
        "any_mean_risk_violation",
        "any_false_safe",
        "any_local_risk_violation",
        "any_local_false_safe",
        "any_reported_coverage_failure",
        "any_committed_coverage_failure",
        "any_pathwise_budget_violation",
    )
    numeric_fields = (
        "fast_selective_risk",
        "intended_selective_risk",
        "max_local_intended_risk",
        "abstention_fraction",
        "audit_fraction",
        "fast_commit_fraction",
        "audit_cost_ms_per_round",
        "reset_cost_ms_per_round",
        "throughput_tokens_per_second",
        "throughput_reward_tokens_per_second",
        "saved_ms_per_round",
        "committed_catastrophes",
        "resets",
        "final_reported_upper",
    )
    for policy in POLICIES:
        group = [r for r in rows if r["policy"] == policy]
        summary: dict[str, Any] = {"policy": policy, "runs": len(group)}
        for field in boolean_fields:
            summary[field + "_rate"] = float(np.mean([bool(r[field]) for r in group]))
        for field in numeric_fields:
            values = np.asarray([float(r[field]) for r in group], dtype=float)
            finite = values[np.isfinite(values)]
            summary[field + "_mean"] = float(np.mean(finite)) if len(finite) else math.inf
            summary[field + "_p90"] = float(np.quantile(finite, 0.90)) if len(finite) else math.inf
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def protocol_payload(cfg: EvaluationConfig, bundle: CalibrationBundle) -> dict[str, Any]:
    epoch = make_epoch(bundle.our_model, bundle.our_plan, epoch_index=0, start_round=0, cfg=cfg)
    return {
        "random_audit": {
            "order": [
                "observe causal context and finish optional feature acquisition",
                "fail to exact before any coin if certificate, support, delay, or prospective pathwise gate is invalid",
                "freeze a terminal fast proposal, score, threshold, support, B_t, predictable cost forecast, q_t, and both fingerprints",
                "append the frozen decision tuple before revealing the coin",
                "draw audit uniform from the independent audit broker",
                "audit iff uniform < q_t",
                "audit runs fast in rollback shadow, runs exact, commits exact",
                "nonaudit commits fast and receives no imputed label",
            ],
            "rng": {
                "construction": (
                    "uint64(HMAC-SHA256(epoch_secret, domain || request_id || "
                    "round_id || runtime_state_fingerprint)) / 2^64"
                ),
                "controller_access": "epoch secret and audit uniform hidden until decision freeze",
                "replay": "precommit key hash and key id; disclose key after epoch closure",
                "theorem_assumption": "independent Bernoulli coin or standard PRF security",
            },
            "q_min": cfg.q_min,
            "q_max": cfg.q_max,
            "logged_fields": [
                "request_id", "round_id", "wall_clock", "epoch_id",
                "causal_context_digest", "prefix_digest", "history_digest",
                "target_logits_summary_digest", "draft_logits_summary_digest",
                "route_state_digest", "cache_state_digest", "hidden_summary_digest",
                "q_t", "q_min", "q_max", "audit_uniform", "audit_indicator",
                "random_key_id", "random_key_commitment", "random_domain",
                "severity_bound", "certificate_fingerprint",
                "runtime_state_fingerprint", "support_key", "proposal_indicator",
                "score", "threshold", "cost_model_digest",
                "predicted_fast_gain_ms", "predicted_audit_gain_ms",
                "predicted_cache_shadow_terms", "realized_fast_ms",
                "realized_exact_ms", "realized_audit_ms",
                "label_due_round", "label_arrival_round", "eventual_block_severity",
                "action_committed", "block_id", "block_width", "first_divergence_row",
                "fast_shadow_digest", "exact_digest", "pre_state_digest", "post_state_digest",
                "cache_generation_before", "cache_generation_after",
                "proposal_loss_upper", "intended_loss_upper", "committed_loss_upper",
                "fast_commit_count", "pathwise_budget", "reset_reason",
            ],
        },
        "initial_plan": asdict(bundle.our_plan),
        "fingerprints": {
            "certificate": epoch.fingerprint,
            "runtime": (
                "SHA-256 over certificate, request/round, prefix/history, target/draft "
                "summaries, route/cache/hidden state, support, score, threshold, q_t, B_t; "
                "the certificate separately binds q_min/q_max, the causal cost model, and reset policy"
            ),
        },
        "confidence_update": epoch.ledger.to_json_dict()["protocol"]["confidence_update"],
        "threshold_policy": {
            "fast_candidate": (
                "current fingerprint valid, score <= frozen threshold, support key "
                "enabled, delay contract valid, no reset active, and the prospective "
                "pathwise gate passes"
            ),
            "audit_probability": "clip(B*sqrt(information_price/opportunity_cost), q_min, q_max)",
            "opportunity_cost": (
                "pre-coin predicted fast gain minus predicted audit gain, both "
                "including cache shadow value; realized current cost is forbidden"
            ),
            "pathwise_gate": (
                "permit the unaudited branch only when its prospective committed-loss "
                "upper endpoint is <= carried_reserve + epsilon*(K+1)"
            ),
        },
        "reset": {
            "triggers": [
                "predeclared traffic-epoch change point",
                "optional persistent unsupported support key when that trigger is enabled",
                "audit label delay exceeds contract",
                "local anytime intended-risk upper exceeds epsilon",
                "model or certificate fingerprint changes",
            ],
            "benchmark_support_reset_streak": cfg.support_reset_streak,
            "action_during_reset": "exact commit with fast shadow for fresh labels",
            "rows": cfg.reset_rows,
            "evidence_split": (
                "chronological train/design/later empirical stress screen with no "
                "coverage claim; accepted deployment coverage starts from fresh "
                "future audit e-values"
            ),
            "monotone_safety_guard": (
                "new score is max(reset score, immutable launch sentinel score), and "
                "the reset threshold may not exceed the launch sentinel threshold"
            ),
            "error_allocation": "delta_e = delta / 2^(e+2); old ledgers retained",
            "pathwise_reserve": (
                "freeze each closed epoch upper bound; carry only "
                "beta0 + epsilon*K_prior - U_prior into the new ledger"
            ),
            "failure_to_recertify_or_negative_reserve": "exact forever",
        },
        "default_action": "exact",
    }


def markdown_report(
    *,
    cfg: EvaluationConfig,
    summary: Sequence[dict[str, Any]],
    impossibility: dict[str, Any],
    bundle: CalibrationBundle,
    runs: int,
    seed: int,
) -> str:
    """Render the checked-in CPU result report from machine-readable rows."""

    labels = {
        "naive_calibration": "naive calibration",
        "split_conformal": "split conformal",
        "importance_weighted_nominal": "nominal importance weighting",
        "anytime_random_audit": "anytime random audit",
    }
    lines = [
        "# Anytime-selective falsifier CPU report",
        "",
        "This is a controlled synthetic result. Timing units model inference cost; they are not measurements from the RTX 4070 SUPER or GLM-5.3-Flash.",
        "",
        "## Configuration",
        "",
        f"- runs: {runs}",
        f"- rounds per run: {cfg.rounds}",
        f"- base seed: {seed}",
        f"- drift change point: {cfg.change_round}",
        f"- initial exact-shadow pilot rows: {cfg.calibration_rows}",
        f"- fail-closed reset rows: {cfg.reset_rows}",
        f"- selective target epsilon: {cfg.epsilon:.3f}",
        f"- anytime error budget delta: {cfg.delta:.3f}",
        f"- audit range: [{cfg.q_min:.2f}, {cfg.q_max:.2f}]",
        f"- global startup reserve: {cfg.startup_budget:.1f} severity units",
        "",
        "The audit probability is computed from causal context and the frozen score. Realized severity, latency, and cache transition do not enter the current coin.",
        "",
        "## Results",
        "",
        "| policy | intended risk | max local risk | reported CS failures | local risk violations | pathwise violations | abstention | audits | audit ms/round | reset ms/round | saved ms/round | throughput reward tok/s | catastrophes/run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        n = int(row["runs"])
        failed = round(float(row["any_reported_coverage_failure_rate"])*n)
        local = round(float(row["any_local_risk_violation_rate"])*n)
        pathwise = round(float(row["any_pathwise_budget_violation_rate"])*n)
        lines.append(
            "| {label} | {risk:.4f} | {localrisk:.4f} | {failed}/{n} | "
            "{local}/{n} | {pathwise}/{n} | {abst:.1%} | {audits:.1%} | "
            "{audit_cost:.1f} | {reset_cost:.1f} | {saved:.1f} | {reward:.3f} | "
            "{cats:.2f} |".format(
                label=labels[row["policy"]],
                risk=float(row["intended_selective_risk_mean"]),
                localrisk=float(row["max_local_intended_risk_mean"]),
                failed=failed,
                n=n,
                local=local,
                pathwise=pathwise,
                abst=float(row["abstention_fraction_mean"]),
                audits=float(row["audit_fraction_mean"]),
                audit_cost=float(row["audit_cost_ms_per_round_mean"]),
                reset_cost=float(row["reset_cost_ms_per_round_mean"]),
                saved=float(row["saved_ms_per_round_mean"]),
                reward=float(row["throughput_reward_tokens_per_second_mean"]),
                cats=float(row["committed_catastrophes_mean"]),
            )
        )
    lines.extend([
        "",
        "The anytime method had no observed reported-loss CS failure, committed-loss CS failure, local target violation, or global pathwise-budget violation.",
        "That empirical record is separate from the mathematical coverage proof.",
        "",
        "## Impossibility witness",
        "",
        f"- exact-only logs identical: `{str(impossibility['logs_identical']).lower()}`",
        f"- shared exact-log SHA-256: `{impossibility['good_log_sha256']}`",
        f"- safe-world deployed fast risk: {impossibility['good_deployment']['selective_risk']:.1f}",
        f"- bad-world deployed fast risk: {impossibility['bad_deployment']['selective_risk']:.1f}",
        "",
        "## Frozen initial policy",
        "",
        f"- threshold: {bundle.our_plan.threshold:.9f}",
        f"- information price: {bundle.our_plan.information_price:.3f}",
        f"- historical stress-screen risk: {bundle.our_plan.screening_risk:.6f}",
        f"- screen method: `{bundle.our_plan.screening_method}`",
        f"- mean planned audit probability: {bundle.our_plan.mean_audit_probability:.4f}",
        f"- enabled support cells: {len(bundle.our_plan.supported_keys)}",
        "",
        "The historical screen is a tuning diagnostic and carries no coverage claim. Deployment coverage is supplied only by fresh post-freeze audit e-values.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONHASHSEED=0 python tools/test_anytime_selective_falsifier.py",
        "PYTHONHASHSEED=0 python tools/evaluate_anytime_selective_falsifier.py \\",
        f"  --runs {runs} --rounds {cfg.rounds} --calibration-rows {cfg.calibration_rows} \\",
        "  --output-dir scratch/anytime-selective-falsifier",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=12000)
    parser.add_argument("--calibration-rows", type=int, default=15000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/anytime-selective-falsifier"),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    cfg = EvaluationConfig(rounds=args.rounds, calibration_rows=args.calibration_rows)
    bundle = build_calibration_bundle(cfg, seed=args.seed)
    run_rows: list[dict[str, Any]] = []
    for run in range(args.runs):
        seed = args.seed + 1009*(run+1)
        for policy in POLICIES:
            run_rows.append(run_policy(policy, seed=seed, cfg=cfg, bundle=bundle))
    summary = summarize_runs(run_rows)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "runs.csv", run_rows)
    write_csv(output / "summary.csv", summary)
    impossibility = indistinguishable_environment_demo()
    (output / "indistinguishable-environments.json").write_text(
        json.dumps(impossibility, indent=2, sort_keys=True) + "\n"
    )
    (output / "protocol.json").write_text(
        json.dumps(protocol_payload(cfg, bundle), indent=2, sort_keys=True) + "\n"
    )
    metadata = {
        "artifact_version": "anytime-selective-falsifier-v1",
        "synthetic_only": True,
        "base_seed": args.seed,
        "runs": args.runs,
        "config": asdict(cfg),
        "calibration": {
            "initial_rows": bundle.initial_rows,
            "naive_threshold": bundle.naive_threshold,
            "naive_reported_risk": bundle.naive_reported_risk,
            "conformal_residual_quantile": bundle.conformal_residual_quantile,
            "importance_initial_count": bundle.importance_initial_count,
            "our_plan": asdict(bundle.our_plan),
            "coverage_role": (
                "historical rows tune and stress-screen only; formal coverage begins "
                "with fresh future audit coins after the policy fingerprint is frozen"
            ),
        },
        "summary": summary,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(markdown_report(
        cfg=cfg,
        summary=summary,
        impossibility=impossibility,
        bundle=bundle,
        runs=args.runs,
        seed=args.seed,
    ))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
