#!/usr/bin/env python3
"""CPU simulation for the joint adaptive-compute controller.

The simulator has hidden token difficulty, a request-level hard/easy variable,
a controlled cache and I/O queue, stochastic DFlash acceptance, and correlated
quality bursts.  It compares:

* a conservative exact fixed action;
* the best feasible fixed tuple from the finite model;
* independent coordinate thresholds;
* a one-step value-of-information controller;
* a scalable primal-dual joint controller;
* the exact finite occupation-measure policy;
* a robust guard around the exact joint policy.

All numbers are synthetic.  The script writes deterministic CSV/JSON artifacts
for a given seed and never claims hardware measurements.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

from joint_adaptive_compute import (
    ComputeAction,
    ControllerState,
    ExactSolveResult,
    FiniteControllerModel,
    LogitLikelihood,
    LogitObservation,
    PolicyEntry,
    SyntheticParameters,
    action_harm_probability,
    bayes_binary,
    build_finite_model,
    cache_transition_distribution,
    draft_checkpoint_counterexample,
    enumerate_compute_actions,
    every_heuristic_locally_sensible_case,
    expected_metrics,
    expert_route_counterexample,
    hidden_transition_probabilities,
    io_transition_distribution,
    metric_distribution,
    myopic_value_of_information,
    policy_map,
    posterior_from_logit_observation,
    prefix_distribution,
    project_belief,
    robust_safety_certificate,
    safe_exact_action,
    solve_occupation_lp,
    throughput_price_of_safety,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    hard_after_easy_delta: float = 0.0
    hard_after_hard_delta: float = 0.0
    harm_multiplier: float = 1.0
    time_multiplier: float = 1.0
    timing_sigma: float = 0.045
    observation_likelihood: LogitLikelihood | None = None
    metric_sensitivity: float | None = None
    metric_false_positive: float | None = None
    adversarial_interactions: bool = False
    queue_burst_probability: float = 0.0


def scenarios(parameters: SyntheticParameters) -> tuple[Scenario, ...]:
    shifted_likelihood = replace(
        parameters.likelihood,
        margin_mu_easy=2.05,
        margin_mu_hard=1.10,
        margin_sigma_hard=1.05,
        churn_theta_easy=0.18,
        churn_theta_hard=0.39,
    )
    return (
        Scenario("calibrated"),
        Scenario(
            "distribution_shift",
            hard_after_easy_delta=0.045,
            hard_after_hard_delta=0.085,
            harm_multiplier=1.55,
            time_multiplier=1.16,
            timing_sigma=0.085,
            observation_likelihood=shifted_likelihood,
            metric_sensitivity=0.82,
            metric_false_positive=0.16,
            queue_burst_probability=0.035,
        ),
        Scenario(
            "interaction_trap",
            hard_after_easy_delta=0.015,
            hard_after_hard_delta=0.030,
            harm_multiplier=1.15,
            time_multiplier=1.03,
            timing_sigma=0.055,
            adversarial_interactions=True,
            queue_burst_probability=0.018,
        ),
    )


@dataclass
class EnvironmentState:
    difficulty: int
    request_hard: int
    cache_level: int
    io_queue: int
    quality_debt: int
    request_index: int = 0
    block_in_request: int = 0


@dataclass(frozen=True)
class MeasurementResult:
    metric: int
    time_ms: float
    bytes_mb: float


@dataclass(frozen=True)
class RoundResult:
    committed: int
    time_ms: float
    bytes_mb: float
    ppl_loss: float
    hard_end: int
    hard_violation: int
    catastrophe: int
    harm: int
    accepted_prefix: int
    next_observation: LogitObservation
    request_end: bool
    hidden_hard: int
    cache_level: int
    io_queue: int
    quality_debt: int
    nominal_time_ms: float


class JointSyntheticEnvironment:
    def __init__(
        self,
        parameters: SyntheticParameters,
        scenario: Scenario,
        seed: int,
    ) -> None:
        self.parameters = parameters
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        request_hard = int(self.rng.random() < parameters.hard_request_prior)
        difficulty = int(
            self.rng.random() < (0.66 if request_hard else 0.12)
        )
        self.state = EnvironmentState(
            difficulty=difficulty,
            request_hard=request_hard,
            cache_level=0,
            io_queue=0,
            quality_debt=0,
        )

    @property
    def likelihood(self) -> LogitLikelihood:
        return self.scenario.observation_likelihood or self.parameters.likelihood

    def _sample_observation(self, difficulty: int) -> LogitObservation:
        observations = self.likelihood.observations()
        probabilities = np.asarray(
            [self.likelihood.probability(difficulty, o) for o in observations],
            dtype=float,
        )
        probabilities /= probabilities.sum()
        return observations[int(self.rng.choice(len(observations), p=probabilities))]

    def measure(self) -> MeasurementResult:
        sensitivity = (
            self.parameters.metric_sensitivity
            if self.scenario.metric_sensitivity is None
            else self.scenario.metric_sensitivity
        )
        false_positive = (
            self.parameters.metric_false_positive
            if self.scenario.metric_false_positive is None
            else self.scenario.metric_false_positive
        )
        probability_one = sensitivity if self.state.difficulty else false_positive
        metric = int(self.rng.random() < probability_one)
        time_ms = (
            self.parameters.measurement_ms
            * self.scenario.time_multiplier
            * self.rng.lognormal(0.0, self.scenario.timing_sigma / 2.0)
        )
        return MeasurementResult(metric, float(time_ms), self.parameters.measurement_bytes_mb)

    @staticmethod
    def _sample_discrete(
        rng: np.random.Generator, distribution: Sequence[tuple[int, float]]
    ) -> int:
        values = [item[0] for item in distribution]
        probabilities = np.asarray([item[1] for item in distribution], dtype=float)
        probabilities /= probabilities.sum()
        return int(rng.choice(values, p=probabilities))

    def step(self, action: ComputeAction) -> RoundResult:
        s = self.state
        parameters = self.parameters
        difficulty = s.difficulty

        # One latent stress draw couples acceptance, quality loss, and future
        # debt.  Debt then persists, producing serially correlated damage.
        stress_probability = 0.055 + 0.19 * difficulty + 0.10 * s.quality_debt
        stress = bool(self.rng.random() < stress_probability)
        base_distribution = prefix_distribution(float(difficulty), action)
        if stress:
            p = max(0.08, 1.0 - base_distribution[0] - 0.18)
            width = action.draft_k
            stressed = np.empty(width + 1, dtype=float)
            stressed[0] = 1.0 - p
            for prefix in range(1, width):
                stressed[prefix] = p**prefix * (1.0 - p)
            stressed[width] = p**width
            distribution = stressed / stressed.sum()
        else:
            distribution = base_distribution
        accepted_prefix = int(
            self.rng.choice(np.arange(action.draft_k + 1), p=distribution)
        )
        committed = max(1, accepted_prefix)

        nominal = expected_metrics(
            float(difficulty),
            s.cache_level,
            s.io_queue,
            s.quality_debt,
            action,
            parameters,
        )
        harm_probability = action_harm_probability(
            float(difficulty),
            s.cache_level,
            s.io_queue,
            s.quality_debt,
            action,
        )
        if stress:
            harm_probability *= 2.4
        harm_probability *= self.scenario.harm_multiplier

        low_approx = action.expert_k == 4 and action.verify_policy == "approx_delta"
        long_delta = action.draft_k == 4 and action.verify_policy == "approx_delta"
        overloaded_prefetch = action.prefetch_budget and s.io_queue >= 1
        if self.scenario.adversarial_interactions:
            if low_approx:
                harm_probability *= 1.85
            if low_approx and action.precision == "fp8":
                harm_probability *= 1.55
            if long_delta and overloaded_prefetch:
                harm_probability *= 1.25
        harm_probability = float(np.clip(harm_probability, 0.0, 0.70))
        harm = int(self.rng.random() < harm_probability)

        continuous = max(
            0.0,
            nominal.ppl_loss
            - nominal.harm_probability * (0.22 + 0.22 * difficulty),
        )
        continuous *= committed / max(nominal.committed, 1e-9)
        harm_severity = (
            (0.22 + 0.22 * difficulty)
            * self.rng.lognormal(0.0, 0.16)
            if harm
            else 0.0
        )
        ppl_loss = continuous + harm_severity
        catastrophe_probability = (
            0.003 + 0.020 * difficulty + 0.006 * s.quality_debt
        )
        catastrophe = int(harm and self.rng.random() < catastrophe_probability)

        debt_before_end = int(s.quality_debt or harm)
        request_end = bool(self.rng.random() < parameters.request_end_probability)
        hard_end = int(request_end and s.request_hard)
        hard_violation = 0
        if hard_end and debt_before_end:
            severe_probability = 0.20 + 0.55 * difficulty
            hard_violation = int(self.rng.random() < severe_probability)

        # Time is conditioned weakly on the realized prefix and strongly on
        # the controlled queue.  The interaction trap adds a rollback/queue
        # storm specifically to the independently sensible long-delta/prefetch
        # combination.
        prefix_factor = 0.78 + 0.22 * committed / max(nominal.committed, 1e-9)
        interaction_factor = 1.0
        if self.scenario.adversarial_interactions:
            if long_delta and overloaded_prefetch:
                interaction_factor *= 1.90
            if low_approx and s.cache_level == 0:
                interaction_factor *= 1.18
        time_ms = (
            nominal.time_ms
            * prefix_factor
            * interaction_factor
            * self.scenario.time_multiplier
            * self.rng.lognormal(0.0, self.scenario.timing_sigma)
        )
        bytes_mb = nominal.bytes_mb * (0.90 + 0.20 * self.rng.random())

        p_hard_easy, p_hard_hard = hidden_transition_probabilities(
            float(difficulty), action, parameters
        )
        if self.scenario.adversarial_interactions and low_approx:
            p_hard_easy += 0.10
            p_hard_hard += 0.13
        p_hard_easy += self.scenario.hard_after_easy_delta
        p_hard_hard += self.scenario.hard_after_hard_delta
        p_next_hard = p_hard_hard if difficulty else p_hard_easy
        p_next_hard = float(np.clip(p_next_hard, 0.01, 0.995))

        next_cache = self._sample_discrete(
            self.rng,
            cache_transition_distribution(
                float(difficulty), s.cache_level, s.io_queue, action, parameters
            ),
        )
        next_io = self._sample_discrete(
            self.rng,
            io_transition_distribution(s.io_queue, action, parameters),
        )
        if self.rng.random() < self.scenario.queue_burst_probability:
            next_io = min(parameters.io_levels - 1, next_io + 1)

        if request_end:
            request_hard = int(self.rng.random() < parameters.hard_request_prior)
            next_difficulty = int(
                self.rng.random() < (0.66 if request_hard else 0.12)
            )
            next_debt = 0
            request_index = s.request_index + 1
            block_in_request = 0
        else:
            request_hard = s.request_hard
            next_difficulty = int(self.rng.random() < p_next_hard)
            next_debt = debt_before_end
            request_index = s.request_index
            block_in_request = s.block_in_request + 1

        next_observation = self._sample_observation(next_difficulty)
        self.state = EnvironmentState(
            difficulty=next_difficulty,
            request_hard=request_hard,
            cache_level=next_cache,
            io_queue=next_io,
            quality_debt=next_debt,
            request_index=request_index,
            block_in_request=block_in_request,
        )
        return RoundResult(
            committed=committed,
            time_ms=float(time_ms),
            bytes_mb=float(bytes_mb),
            ppl_loss=float(ppl_loss),
            hard_end=hard_end,
            hard_violation=hard_violation,
            catastrophe=catastrophe,
            harm=harm,
            accepted_prefix=accepted_prefix,
            next_observation=next_observation,
            request_end=request_end,
            hidden_hard=difficulty,
            cache_level=next_cache,
            io_queue=next_io,
            quality_debt=next_debt,
            nominal_time_ms=nominal.time_ms,
        )


class BeliefPolicy:
    name = "base"
    guard_ms = 0.0

    def __init__(self, parameters: SyntheticParameters, seed: int) -> None:
        self.parameters = parameters
        self.rng = np.random.default_rng(seed)
        self.q_hard = parameters.hard_request_prior
        self.phase = "ready"
        self.measurements = 0
        self.compute_decisions = 0
        self.fallbacks = 0
        self.ood_latched = False

    def choose(
        self, cache_level: int, io_queue: int, quality_debt: int
    ) -> str | ComputeAction:
        raise NotImplementedError

    def apply_metric(self, metric: int) -> None:
        sensitivity = self.parameters.metric_sensitivity
        false_positive = self.parameters.metric_false_positive
        likelihood_h = sensitivity if metric else 1.0 - sensitivity
        likelihood_e = false_positive if metric else 1.0 - false_positive
        self.q_hard = bayes_binary(self.q_hard, likelihood_h, likelihood_e)
        self.phase = "measured"
        self.measurements += 1

    def observe(self, action: ComputeAction, result: RoundResult) -> None:
        if result.request_end:
            prior = self.parameters.hard_request_prior
        else:
            p_easy, p_hard = hidden_transition_probabilities(
                self.q_hard, action, self.parameters
            )
            prior = (1.0 - self.q_hard) * p_easy + self.q_hard * p_hard
        self.q_hard = posterior_from_logit_observation(
            prior, result.next_observation, self.parameters.likelihood
        )
        self.phase = "ready"
        self.compute_decisions += 1


class FixedPolicy(BeliefPolicy):
    def __init__(
        self,
        parameters: SyntheticParameters,
        seed: int,
        action: ComputeAction,
        name: str,
    ) -> None:
        super().__init__(parameters, seed)
        self.action = action
        self.name = name

    def choose(self, cache_level: int, io_queue: int, quality_debt: int) -> ComputeAction:
        return self.action


class IndependentThresholdPolicy(BeliefPolicy):
    name = "independent_thresholds"

    def choose(self, cache_level: int, io_queue: int, quality_debt: int) -> ComputeAction:
        q = self.q_hard
        expert_k = 8 if (q >= 0.58 or quality_debt) else 4
        draft_k = 2 if q >= 0.48 else 4
        verify = "exact_full" if (q >= 0.66 or quality_debt) else "approx_delta"
        precision = "bf16" if q >= 0.72 else "fp8"
        prefetch = int(cache_level <= 1 and io_queue <= 1)
        return ComputeAction(expert_k, draft_k, verify, precision, prefetch)


class MyopicVOIPolicy(BeliefPolicy):
    name = "myopic_voi"

    def __init__(
        self,
        parameters: SyntheticParameters,
        seed: int,
        throughput_tokens_per_ms: float,
        multipliers: Mapping[str, float],
    ) -> None:
        super().__init__(parameters, seed)
        self.actions = enumerate_compute_actions()
        self.action_by_name = {a.name: a for a in self.actions}
        self.throughput = throughput_tokens_per_ms
        self.multipliers = multipliers
        self.cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    def _voi(self, cache_level: int, io_queue: int, quality_debt: int) -> dict[str, Any]:
        belief_index = project_belief(self.q_hard, self.parameters.belief_grid)
        key = (belief_index, cache_level, io_queue, quality_debt)
        if key not in self.cache:
            q = self.parameters.belief_grid[belief_index]
            self.cache[key] = myopic_value_of_information(
                q,
                cache_level,
                io_queue,
                quality_debt,
                self.actions,
                self.parameters,
                throughput_tokens_per_ms=self.throughput,
                ppl_multiplier=self.multipliers.get("ppl_loss_per_token", 0.0),
                hard_multiplier=self.multipliers.get("hard_violation_probability", 0.0),
                catastrophe_multiplier=self.multipliers.get("catastrophe_per_token", 0.0),
            )
        return self.cache[key]

    def _best_action_for_posterior(
        self, cache_level: int, io_queue: int, quality_debt: int
    ) -> ComputeAction:
        result = self._voi(cache_level, io_queue, quality_debt)
        return self.action_by_name[result["no_measure_action"]]

    def choose(
        self, cache_level: int, io_queue: int, quality_debt: int
    ) -> str | ComputeAction:
        if self.phase == "ready":
            voi = self._voi(cache_level, io_queue, quality_debt)
            if voi["measure"]:
                return "measure_exact_metric"
        return self._best_action_for_posterior(cache_level, io_queue, quality_debt)


class ExactJointPolicy(BeliefPolicy):
    name = "exact_joint"

    def __init__(
        self,
        parameters: SyntheticParameters,
        seed: int,
        result: ExactSolveResult,
    ) -> None:
        super().__init__(parameters, seed)
        self.policy = policy_map(result)
        self.action_by_name = {a.name: a for a in enumerate_compute_actions()}
        self.safe = safe_exact_action()

    def _label(self, cache_level: int, io_queue: int, quality_debt: int) -> str:
        state = ControllerState(
            project_belief(self.q_hard, self.parameters.belief_grid),
            cache_level,
            io_queue,
            quality_debt,
            self.phase,
        )
        return state.label(self.parameters.belief_grid)

    def choose(
        self, cache_level: int, io_queue: int, quality_debt: int
    ) -> str | ComputeAction:
        entries = self.policy.get(self._label(cache_level, io_queue, quality_debt))
        if not entries:
            return self.safe
        names = [entry[0] for entry in entries]
        probabilities = np.asarray([entry[1] for entry in entries], dtype=float)
        probabilities /= probabilities.sum()
        name = names[int(self.rng.choice(len(names), p=probabilities))]
        if name == "measure_exact_metric":
            return name
        return self.action_by_name.get(name, self.safe)


class RobustSafeJointPolicy(ExactJointPolicy):
    name = "robust_safe_joint"

    def __init__(
        self,
        parameters: SyntheticParameters,
        seed: int,
        result: ExactSolveResult,
        throughput_tokens_per_ms: float,
    ) -> None:
        super().__init__(parameters, seed, result)
        self.throughput = throughput_tokens_per_ms
        self.surprise = 0.0
        self.overlap = 0.92
        self.last_certificate: dict[str, Any] | None = None
        self.guard_ms = parameters.guard_ms

    def choose(
        self, cache_level: int, io_queue: int, quality_debt: int
    ) -> str | ComputeAction:
        proposal = super().choose(cache_level, io_queue, quality_debt)
        if proposal == "measure_exact_metric":
            if self.ood_latched or self.overlap < 0.35:
                return self.safe
            return proposal
        assert isinstance(proposal, ComputeAction)
        if proposal.is_safe_exact:
            return proposal
        q = self.parameters.belief_grid[
            project_belief(self.q_hard, self.parameters.belief_grid)
        ]
        metrics = expected_metrics(
            q, cache_level, io_queue, quality_debt, proposal, self.parameters
        )
        safe_metrics = expected_metrics(
            q, cache_level, io_queue, quality_debt, self.safe, self.parameters
        )
        entropy = -q * math.log(max(q, 1e-12)) - (1.0 - q) * math.log(
            max(1.0 - q, 1e-12)
        )
        uncertainty = 0.018 + 0.032 * entropy + 0.018 * min(self.surprise, 5.0)
        if self.ood_latched:
            uncertainty = 1.0
        certificate = robust_safety_certificate(
            metrics,
            safe_metrics,
            throughput_tokens_per_ms=self.throughput,
            controller_and_guard_ms=self.guard_ms,
            ppl_radius=0.00030 + 0.00035 * self.surprise,
            hard_risk_radius=0.00012 + 0.00018 * self.surprise,
            catastrophe_radius=0.000005 + 0.000006 * self.surprise,
            time_radius_ms=1.0 + 1.2 * self.surprise,
            ppl_envelope_per_token=0.0045,
            hard_envelope=0.035,
            catastrophe_envelope_per_token=0.00035,
            uncertainty_radius=uncertainty,
            max_uncertainty_radius=0.20,
            overlap=self.overlap,
            min_overlap=0.35,
        )
        self.last_certificate = certificate
        if not certificate["allowed"]:
            self.fallbacks += 1
            return self.safe
        return proposal

    def observe(self, action: ComputeAction, result: RoundResult) -> None:
        q_before = self.q_hard
        p_easy, p_hard = hidden_transition_probabilities(
            q_before, action, self.parameters
        )
        prior = (
            self.parameters.hard_request_prior
            if result.request_end
            else (1.0 - q_before) * p_easy + q_before * p_hard
        )
        probability = (
            (1.0 - prior)
            * self.parameters.likelihood.probability(0, result.next_observation)
            + prior
            * self.parameters.likelihood.probability(1, result.next_observation)
        )
        nll_excess = max(0.0, -math.log(max(probability, 1e-12)) - 3.8)
        timing_ratio = result.time_ms / max(result.nominal_time_ms, 1e-9)
        timing_excess = max(0.0, timing_ratio - 1.45)
        self.surprise = 0.90 * self.surprise + nll_excess + 1.5 * timing_excess
        self.overlap = min(
            0.98,
            max(
                0.0,
                0.995 * self.overlap
                + 0.005 * 0.92
                - 0.025 * (nll_excess + timing_excess),
            ),
        )
        if self.surprise > 8.0 or self.overlap < 0.35:
            self.ood_latched = True
        super().observe(action, result)


class PrimalDualJointPolicy(BeliefPolicy):
    """Scalable drift-plus-penalty approximation with virtual queues."""

    name = "primal_dual_joint"

    def __init__(
        self,
        parameters: SyntheticParameters,
        seed: int,
        initial_throughput_tokens_per_ms: float,
        warm_shadow_prices: Mapping[str, float],
        V: float = 18.0,
    ) -> None:
        super().__init__(parameters, seed)
        self.actions = enumerate_compute_actions()
        self.V = V
        self.rho = initial_throughput_tokens_per_ms
        self.q_ppl = 0.02 * warm_shadow_prices.get("ppl_loss_per_token", 0.0)
        self.virtual_hard = 0.02 * warm_shadow_prices.get("hard_violation_probability", 0.0)
        self.q_cat = 0.02 * warm_shadow_prices.get("catastrophe_per_token", 0.0)
        self.total_tokens = 0.0
        self.total_time = 0.0
        self.action_cache: dict[tuple[int, int, int, int], ComputeAction] = {}

    def _score(
        self,
        q: float,
        cache_level: int,
        io_queue: int,
        quality_debt: int,
        action: ComputeAction,
    ) -> float:
        m = expected_metrics(
            q, cache_level, io_queue, quality_debt, action, self.parameters
        )
        g_ppl = m.ppl_loss - 0.0045 * m.committed
        g_hard = m.hard_violation - 0.035 * m.hard_end
        g_cat = m.catastrophe - 0.00035 * m.committed
        future_cache = 0.38 * action.prefetch_budget * (2 - cache_level)
        future_route = 0.25 * float(action.expert_k == 8) * q
        queue_penalty = 0.55 * io_queue * action.prefetch_budget
        return (
            self.V * (m.committed - self.rho * m.time_ms)
            - 120.0 * self.q_ppl * g_ppl
            - 30.0 * self.virtual_hard * g_hard
            - 2500.0 * self.q_cat * g_cat
            + future_cache
            + future_route
            - queue_penalty
        )

    def choose(
        self, cache_level: int, io_queue: int, quality_debt: int
    ) -> ComputeAction:
        belief_index = project_belief(self.q_hard, self.parameters.belief_grid)
        q = self.parameters.belief_grid[belief_index]
        key = (belief_index, cache_level, io_queue, quality_debt)
        # Recompute after queue updates; the cache stores only the static action
        # order and is intentionally not used as a stale decision cache.
        _ = key
        return max(
            self.actions,
            key=lambda action: self._score(
                q, cache_level, io_queue, quality_debt, action
            ),
        )

    def observe(self, action: ComputeAction, result: RoundResult) -> None:
        self.q_ppl = max(
            0.0,
            self.q_ppl + 100.0 * (result.ppl_loss - 0.0045 * result.committed),
        )
        self.virtual_hard = max(
            0.0,
            self.virtual_hard
            + 20.0 * (result.hard_violation - 0.035 * result.hard_end),
        )
        self.q_cat = max(
            0.0,
            self.q_cat + 5000.0 * (result.catastrophe - 0.00035 * result.committed),
        )
        self.total_tokens += result.committed
        self.total_time += result.time_ms
        if self.total_time > 0:
            observed_rho = self.total_tokens / self.total_time
            self.rho = 0.995 * self.rho + 0.005 * observed_rho
        super().observe(action, result)


@dataclass(frozen=True)
class RunSummary:
    scenario: str
    policy: str
    rounds: int
    committed_tokens: int
    wall_time_ms: float
    bytes_mb_total: float
    ppl_loss_total: float
    hard_ends: int
    hard_violations: int
    catastrophes: int
    measurement_count: int
    fallback_count: int
    measurement_time_ms: float
    guard_time_ms: float
    cache_level_sum: float
    io_queue_sum: float
    hidden_hard_count: int
    debt_and_queue2_count: int
    committed_tokens_per_second: float
    ppl_loss_per_token: float
    hard_violation_probability: float
    catastrophe_per_token: float
    bytes_per_token_mb: float
    measurement_fraction: float
    fallback_fraction: float
    mean_cache_level: float
    mean_io_queue: float
    hidden_hard_fraction: float
    debt_and_queue2_fraction: float
    ppl_constraint_met: bool
    hard_constraint_met: bool
    catastrophe_constraint_met: bool
    ood_latched: bool


def run_policy(
    policy: BeliefPolicy,
    environment: JointSyntheticEnvironment,
    rounds: int,
) -> tuple[RunSummary, Counter[str]]:
    committed = 0
    wall_time_ms = 0.0
    bytes_mb = 0.0
    ppl_loss = 0.0
    hard_ends = 0
    hard_violations = 0
    catastrophes = 0
    cache_sum = 0.0
    io_sum = 0.0
    hidden_hard = 0
    debt_queue2 = 0
    measurement_time_ms = 0.0
    action_counts: Counter[str] = Counter()

    for _ in range(rounds):
        state = environment.state
        choice = policy.choose(
            state.cache_level, state.io_queue, state.quality_debt
        )
        if choice == "measure_exact_metric":
            measurement = environment.measure()
            wall_time_ms += measurement.time_ms
            measurement_time_ms += measurement.time_ms
            bytes_mb += measurement.bytes_mb
            action_counts["measure_exact_metric"] += 1
            policy.apply_metric(measurement.metric)
            state = environment.state
            choice = policy.choose(
                state.cache_level, state.io_queue, state.quality_debt
            )
            if choice == "measure_exact_metric":
                # A policy cannot acquire the same exact metric twice in one
                # block.  This is a controller bug, so fail closed.
                choice = safe_exact_action()
                policy.fallbacks += 1
        assert isinstance(choice, ComputeAction)
        result = environment.step(choice)
        result_time = result.time_ms + policy.guard_ms
        committed += result.committed
        wall_time_ms += result_time
        bytes_mb += result.bytes_mb
        ppl_loss += result.ppl_loss
        hard_ends += result.hard_end
        hard_violations += result.hard_violation
        catastrophes += result.catastrophe
        cache_sum += result.cache_level
        io_sum += result.io_queue
        hidden_hard += result.hidden_hard
        debt_queue2 += int(result.quality_debt and result.io_queue == 2)
        action_counts[choice.name] += 1
        policy.observe(choice, result)

    hard_probability = hard_violations / hard_ends if hard_ends else math.nan
    summary = RunSummary(
        scenario=environment.scenario.name,
        policy=policy.name,
        rounds=rounds,
        committed_tokens=committed,
        wall_time_ms=wall_time_ms,
        bytes_mb_total=bytes_mb,
        ppl_loss_total=ppl_loss,
        hard_ends=hard_ends,
        hard_violations=hard_violations,
        catastrophes=catastrophes,
        measurement_count=policy.measurements,
        fallback_count=policy.fallbacks,
        measurement_time_ms=measurement_time_ms,
        guard_time_ms=policy.guard_ms * rounds,
        cache_level_sum=cache_sum,
        io_queue_sum=io_sum,
        hidden_hard_count=hidden_hard,
        debt_and_queue2_count=debt_queue2,
        committed_tokens_per_second=1000.0 * committed / wall_time_ms,
        ppl_loss_per_token=ppl_loss / committed,
        hard_violation_probability=hard_probability,
        catastrophe_per_token=catastrophes / committed,
        bytes_per_token_mb=bytes_mb / committed,
        measurement_fraction=policy.measurements / rounds,
        fallback_fraction=policy.fallbacks / rounds,
        mean_cache_level=cache_sum / rounds,
        mean_io_queue=io_sum / rounds,
        hidden_hard_fraction=hidden_hard / rounds,
        debt_and_queue2_fraction=debt_queue2 / rounds,
        ppl_constraint_met=ppl_loss / committed <= 0.0045,
        hard_constraint_met=(hard_probability <= 0.035 if math.isfinite(hard_probability) else False),
        catastrophe_constraint_met=catastrophes / committed <= 0.00035,
        ood_latched=policy.ood_latched,
    )
    return summary, action_counts


def evaluate_fixed_action_model(
    model: FiniteControllerModel, action_name: str
) -> dict[str, Any]:
    """Exact stationary evaluation for one fixed compute tuple.

    Compute actions always return to a ``ready`` state, so the recurrent chain
    for a fixed tuple contains only the ready half of the finite state space.
    Power iteration on that sparse chain is much cheaper than solving a fresh
    occupation LP for every tuple.
    """

    ready_global = [i for i, state in enumerate(model.states) if state.phase == "ready"]
    local = {global_index: i for i, global_index in enumerate(ready_global)}
    n = len(ready_global)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    committed = np.empty(n)
    time_ms = np.empty(n)
    ppl_loss = np.empty(n)
    hard_end = np.empty(n)
    hard_violation = np.empty(n)
    catastrophe = np.empty(n)
    for i, global_index in enumerate(ready_global):
        choices = [a for a in model.actions[global_index] if a.name == action_name]
        if len(choices) != 1:
            raise ValueError(f"action {action_name} unavailable in ready state")
        action = choices[0]
        for target, probability in action.transitions:
            if target not in local:
                raise ValueError("compute action unexpectedly enters measured phase")
            rows.append(i)
            cols.append(local[target])
            data.append(probability)
        committed[i] = action.committed
        time_ms[i] = action.time_ms
        ppl_loss[i] = action.ppl_loss
        hard_end[i] = action.hard_end
        hard_violation[i] = action.hard_violation
        catastrophe[i] = action.catastrophe
    transition = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    # Solve (P^T-I) pi = 0 with one row replaced by sum(pi)=1.
    stationary_system = (transition.T - eye(n, format="csr")).tolil()
    stationary_system[-1, :] = np.ones(n)
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    distribution = np.asarray(spsolve(stationary_system.tocsr(), rhs), dtype=float)
    distribution = np.clip(distribution, 0.0, None)
    distribution /= distribution.sum()
    token_rate = float(distribution @ committed)
    time_rate = float(distribution @ time_ms)
    ppl_rate = float(distribution @ ppl_loss)
    hard_end_rate = float(distribution @ hard_end)
    hard_violation_rate = float(distribution @ hard_violation)
    catastrophe_rate = float(distribution @ catastrophe)
    ppl_ratio = ppl_rate / max(token_rate, 1e-12)
    hard_ratio = hard_violation_rate / max(hard_end_rate, 1e-12)
    catastrophe_ratio = catastrophe_rate / max(token_rate, 1e-12)
    feasible = (
        ppl_ratio <= 0.0045 + 1e-10
        and hard_ratio <= 0.035 + 1e-10
        and catastrophe_ratio <= 0.00035 + 1e-10
    )
    return {
        "action": action_name,
        "feasible": feasible,
        "committed_tokens_per_second": 1000.0 * token_rate / time_rate,
        "ppl_loss_per_token": ppl_ratio,
        "hard_violation_probability": hard_ratio,
        "catastrophe_per_token": catastrophe_ratio,
    }


def select_best_fixed(
    model: FiniteControllerModel,
) -> tuple[ComputeAction, list[dict[str, Any]]]:
    action_by_name = {a.name: a for a in enumerate_compute_actions()}
    rows = [evaluate_fixed_action_model(model, name) for name in action_by_name]
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        raise RuntimeError("no feasible fixed action")
    best_row = max(feasible, key=lambda row: row["committed_tokens_per_second"])
    return action_by_name[best_row["action"]], rows




def load_static_plan(
    static_dir: Path,
) -> tuple[ExactSolveResult, ComputeAction, list[dict[str, Any]], dict[str, Any]]:
    """Load one previously solved finite controller for cheap replications."""

    exact_payload = json.loads((static_dir / "exact-policy.json").read_text())
    exact_result = ExactSolveResult(
        committed_tokens_per_second=float(exact_payload["committed_tokens_per_second"]),
        utility_per_second=float(exact_payload["utility_per_second"]),
        bytes_per_second_mb=float(exact_payload["bytes_per_second_mb"]),
        ppl_loss_per_committed_token=float(
            exact_payload["ppl_loss_per_committed_token"]
        ),
        hard_violation_probability=float(
            exact_payload["hard_violation_probability"]
        ),
        catastrophe_per_committed_token=float(
            exact_payload["catastrophe_per_committed_token"]
        ),
        decision_starts_per_second=float(
            exact_payload["decision_starts_per_second"]
        ),
        measurement_fraction=float(exact_payload["measurement_fraction"]),
        expected_sojourn_ms=float(exact_payload["expected_sojourn_ms"]),
        policy=tuple(PolicyEntry(**row) for row in exact_payload["policy"]),
        variable_labels=tuple(
            (str(row[0]), str(row[1])) for row in exact_payload["variable_labels"]
        ),
        raw_occupation=tuple(float(value) for value in exact_payload["raw_occupation"]),
        constraint_shadow_prices={
            str(key): float(value)
            for key, value in exact_payload["constraint_shadow_prices"].items()
        },
        solver_message=str(exact_payload["solver_message"]),
    )
    fixed_rows: list[dict[str, Any]] = []
    with (static_dir / "fixed-sweep.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            fixed_rows.append(
                {
                    "action": row["action"],
                    "feasible": row["feasible"].lower() == "true",
                    "committed_tokens_per_second": float(
                        row["committed_tokens_per_second"]
                    ),
                    "ppl_loss_per_token": float(row["ppl_loss_per_token"]),
                    "hard_violation_probability": float(
                        row["hard_violation_probability"]
                    ),
                    "catastrophe_per_token": float(row["catastrophe_per_token"]),
                }
            )
    feasible = [row for row in fixed_rows if row["feasible"]]
    if not feasible:
        raise RuntimeError("static plan contains no feasible fixed action")
    best_name = max(
        feasible, key=lambda row: row["committed_tokens_per_second"]
    )["action"]
    actions = {action.name: action for action in enumerate_compute_actions()}
    if best_name not in actions:
        raise RuntimeError(f"unknown fixed action in static plan: {best_name}")
    model_summary = json.loads((static_dir / "model-summary.json").read_text())
    return exact_result, actions[best_name], fixed_rows, model_summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_without_gc(
    *,
    rounds: int,
    seed: int,
    out_dir: Path,
    static_dir: Path | None = None,
) -> dict[str, Any]:
    parameters = SyntheticParameters()
    if static_dir is None:
        model = build_finite_model(parameters)
        exact_result = solve_occupation_lp(model)
        best_fixed, fixed_rows = select_best_fixed(model)
        model_summary = model.compact_json_dict()
    else:
        exact_result, best_fixed, fixed_rows, model_summary = load_static_plan(
            static_dir
        )
    safe = safe_exact_action()
    baseline_rho = max(
        row["committed_tokens_per_second"]
        for row in fixed_rows
        if row["feasible"]
    ) / 1000.0

    summaries: list[RunSummary] = []
    action_rows: list[dict[str, Any]] = []
    scenario_list = scenarios(parameters)
    for scenario_index, scenario in enumerate(scenario_list):
        policy_factories = (
            lambda s: FixedPolicy(parameters, s, safe, "safe_exact_fixed"),
            lambda s: FixedPolicy(parameters, s, best_fixed, "best_fixed"),
            lambda s: IndependentThresholdPolicy(parameters, s),
            lambda s: MyopicVOIPolicy(
                parameters,
                s,
                baseline_rho,
                exact_result.constraint_shadow_prices,
            ),
            lambda s: PrimalDualJointPolicy(
                parameters,
                s,
                baseline_rho,
                exact_result.constraint_shadow_prices,
            ),
            lambda s: ExactJointPolicy(parameters, s, exact_result),
            lambda s: RobustSafeJointPolicy(
                parameters, s, exact_result, baseline_rho
            ),
        )
        for policy_index, factory in enumerate(policy_factories):
            run_seed = seed + 10_000 * scenario_index + 211 * policy_index
            environment = JointSyntheticEnvironment(parameters, scenario, run_seed)
            policy = factory(run_seed + 97)
            summary, counts = run_policy(policy, environment, rounds)
            summaries.append(summary)
            total_actions = sum(counts.values())
            for action, count in sorted(counts.items()):
                action_rows.append(
                    {
                        "scenario": scenario.name,
                        "policy": policy.name,
                        "action": action,
                        "count": count,
                        "fraction_of_action_starts": count / total_actions,
                    }
                )
            del policy, environment, counts

    summary_rows = [asdict(summary) for summary in summaries]
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "action-frequency.csv", action_rows)
    write_csv(out_dir / "fixed-sweep.csv", fixed_rows)

    voi_rows = []
    for q in parameters.belief_grid:
        voi = myopic_value_of_information(
            q,
            cache_level=1,
            io_queue=0,
            quality_debt=0,
            candidate_actions=enumerate_compute_actions(),
            parameters=parameters,
            throughput_tokens_per_ms=baseline_rho,
            hard_multiplier=exact_result.constraint_shadow_prices.get(
                "hard_violation_probability", 0.0
            ),
        )
        voi_rows.append(
            {
                "posterior_hard": q,
                "gross_voi": voi["gross_voi"],
                "measurement_time_value": voi["measurement_time_value"],
                "net_voi": voi["net_voi"],
                "measure": voi["measure"],
                "no_measure_action": voi["no_measure_action"],
                "metric0_action": voi["branches"][0]["action"],
                "metric1_action": voi["branches"][1]["action"],
            }
        )
    write_csv(out_dir / "voi-table.csv", voi_rows)

    exact_json = exact_result.to_json_dict()
    (out_dir / "exact-policy.json").write_text(json.dumps(exact_json, indent=2) + "\n")
    (out_dir / "model-summary.json").write_text(
        json.dumps(model_summary, indent=2) + "\n"
    )
    counterexamples = {
        "expert_route": expert_route_counterexample(),
        "draft_checkpoint": draft_checkpoint_counterexample(),
        "locally_sensible_unstable": every_heuristic_locally_sensible_case(),
    }
    (out_dir / "counterexamples.json").write_text(
        json.dumps(counterexamples, indent=2) + "\n"
    )

    calibrated = {
        row.policy: row
        for row in summaries
        if row.scenario == "calibrated"
    }
    exact_row = calibrated["exact_joint"]
    robust_row = calibrated["robust_safe_joint"]
    safe_row = calibrated["safe_exact_fixed"]
    safety_price = throughput_price_of_safety(
        joint_tokens_per_round=exact_row.committed_tokens / exact_row.rounds,
        joint_ms_per_round=exact_row.wall_time_ms / exact_row.rounds,
        exact_tokens_per_round=safe_row.committed_tokens / safe_row.rounds,
        exact_ms_per_round=safe_row.wall_time_ms / safe_row.rounds,
        fallback_fraction=robust_row.fallback_fraction,
        guard_ms_per_round=parameters.guard_ms,
    )

    best_fixed_row = calibrated["best_fixed"]
    robust_improvement = (
        robust_row.committed_tokens_per_second
        - best_fixed_row.committed_tokens_per_second
    ) / best_fixed_row.committed_tokens_per_second
    robust_transformed_gain_per_round = (
        robust_row.committed_tokens / robust_row.rounds
        - baseline_rho * robust_row.wall_time_ms / robust_row.rounds
    )
    kill = {
        "controller_cost_exceeds_predicted_saving": (
            robust_transformed_gain_per_round <= 0.0
        ),
        "safe_exploration_lacks_overlap": robust_row.fallback_fraction >= 0.95,
        "no_material_robust_gain_over_best_fixed": robust_improvement < 0.03,
        "robust_gain_over_best_fixed": robust_improvement,
        "robust_transformed_gain_per_round": robust_transformed_gain_per_round,
        "kill_controller": False,
    }
    kill["kill_controller"] = any(
        value for key, value in kill.items() if key not in {"robust_gain_over_best_fixed", "robust_transformed_gain_per_round", "kill_controller"}
    )

    payload = {
        "synthetic_only": True,
        "seed": seed,
        "rounds_per_policy_scenario": rounds,
        "best_fixed_action": best_fixed.name,
        "exact_finite_result": {
            "committed_tokens_per_second": exact_result.committed_tokens_per_second,
            "ppl_loss_per_token": exact_result.ppl_loss_per_committed_token,
            "hard_violation_probability": exact_result.hard_violation_probability,
            "catastrophe_per_token": exact_result.catastrophe_per_committed_token,
            "measurement_fraction": exact_result.measurement_fraction,
            "constraint_shadow_prices": dict(exact_result.constraint_shadow_prices),
        },
        "simulation": summary_rows,
        "throughput_price_of_safety": safety_price,
        "kill_criteria": kill,
        "counterexamples": counterexamples,
    }
    (out_dir / "evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def evaluate(
    *,
    rounds: int,
    seed: int,
    out_dir: Path,
    static_dir: Path | None = None,
) -> dict[str, Any]:
    # The finite kernel is a large acyclic tuple graph.  CPython's cyclic GC
    # can spend minutes rescanning it even though reference counting is enough.
    # Disable cyclic scans for the deterministic evaluation, including failure
    # paths, then restore the caller's setting.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        return _evaluate_without_gc(
            rounds=rounds,
            seed=seed,
            out_dir=out_dir,
            static_dir=static_dir,
        )
    finally:
        if gc_was_enabled:
            gc.enable()


def print_compact(payload: Mapping[str, Any]) -> None:
    print("synthetic CPU evaluation")
    print(f"best fixed action: {payload['best_fixed_action']}")
    print("scenario             policy                    tok/s      ppl/tok    hard-fail   cat/tok   fallback")
    for row in payload["simulation"]:
        print(
            f"{row['scenario']:20s} {row['policy']:24s} "
            f"{row['committed_tokens_per_second']:8.3f} "
            f"{row['ppl_loss_per_token']:11.6f} "
            f"{row['hard_violation_probability']:11.6f} "
            f"{row['catastrophe_per_token']:9.6f} "
            f"{row['fallback_fraction']:9.4f}"
        )
    print("kill criteria:")
    for key, value in payload["kill_criteria"].items():
        print(f"  {key}: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=12012)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scratch/joint-adaptive-compute"),
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        help=(
            "reuse exact-policy.json, fixed-sweep.csv, and model-summary.json "
            "from a completed run"
        ),
    )
    args = parser.parse_args(argv)
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    payload = evaluate(
        rounds=args.rounds,
        seed=args.seed,
        out_dir=args.out_dir,
        static_dir=args.static_dir,
    )
    print_compact(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
