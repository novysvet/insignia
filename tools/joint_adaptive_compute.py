#!/usr/bin/env python3
"""Exact finite-state joint adaptive-compute controller for Insignia.

The module implements a finite belief-state constrained semi-Markov decision
process (CSMDP).  It is deliberately CPU-only and synthetic: every transition
parameter is exposed, and no private-model measurement is presented as fact.

The finite model contains:

* a posterior probability that the next block is in a hard regime;
* endogenous expert-cache and I/O-queue states;
* a one-bit correlated quality-debt automaton;
* an explicit pre-action exact-metric stage with positive time and zero tokens;
* the joint action tuple (expert_k, draft_k, verification, precision, prefetch).

The exact solver uses time-normalized occupation measures.  The objective is
committed tokens per wall-clock second (optionally minus a byte price), with a
global PPL-loss constraint, a separate rare hard-answer chance constraint, and
an optional catastrophic-risk constraint.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Observation and action schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class ControllerState:
    """A point in the finite belief-state discretization."""

    belief_index: int
    cache_level: int
    io_queue: int
    quality_debt: int
    phase: str  # "ready" permits measurement; "measured" does not.

    def label(self, grid: Sequence[float]) -> str:
        return (
            f"q={grid[self.belief_index]:.3f}/c={self.cache_level}/"
            f"io={self.io_queue}/z={self.quality_debt}/{self.phase}"
        )


@dataclass(frozen=True, order=True)
class ComputeAction:
    expert_k: int
    draft_k: int
    verify_policy: str  # exact_full | approx_delta
    precision: str  # bf16 | fp8
    prefetch_budget: int  # 0 | 1 synthetic unit

    @property
    def name(self) -> str:
        verify = "exact" if self.verify_policy == "exact_full" else "approx"
        return (
            f"k{self.expert_k}_d{self.draft_k}_{verify}_"
            f"{self.precision}_pf{self.prefetch_budget}"
        )

    @property
    def is_safe_exact(self) -> bool:
        return (
            self.expert_k == 8
            and self.verify_policy == "exact_full"
            and self.precision == "bf16"
            and self.draft_k == 2
            and self.prefetch_budget == 0
        )


@dataclass(frozen=True)
class LogitObservation:
    """Discretized previous-logit evidence used by the finite model."""

    margin_bin: int  # 0 low, 1 middle, 2 high
    churn_high: int  # 0 if <=2 of 8 routes changed, else 1

    @property
    def name(self) -> str:
        return f"m{self.margin_bin}_r{self.churn_high}"


@dataclass(frozen=True)
class ActionKernel:
    name: str
    transitions: tuple[tuple[int, float], ...]
    committed: float
    time_ms: float
    bytes_mb: float
    ppl_loss: float
    hard_end: float
    hard_violation: float
    catastrophe: float
    measurement: bool = False

    def validate(self, n_states: int) -> None:
        if self.time_ms <= 0:
            raise ValueError(f"non-positive sojourn time for {self.name}")
        if min(
            self.committed,
            self.bytes_mb,
            self.ppl_loss,
            self.hard_end,
            self.hard_violation,
            self.catastrophe,
        ) < -_EPS:
            raise ValueError(f"negative counter in {self.name}")
        if self.hard_violation > self.hard_end + 1e-9:
            raise ValueError(f"hard_violation exceeds hard_end in {self.name}")
        total = 0.0
        for target, probability in self.transitions:
            if not 0 <= target < n_states:
                raise ValueError(f"bad next state {target}")
            if probability < -_EPS:
                raise ValueError(f"negative transition probability in {self.name}")
            total += probability
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"transitions for {self.name} sum to {total}")


@dataclass(frozen=True)
class FiniteControllerModel:
    states: tuple[ControllerState, ...]
    actions: tuple[tuple[ActionKernel, ...], ...]
    parameters: "SyntheticParameters"

    def validate(self) -> None:
        if not self.states or len(self.states) != len(set(self.states)):
            raise ValueError("states must be non-empty and unique")
        if len(self.actions) != len(self.states):
            raise ValueError("one action list is required per state")
        for choices in self.actions:
            if not choices:
                raise ValueError("every state needs at least one action")
            names = [choice.name for choice in choices]
            if len(names) != len(set(names)):
                raise ValueError("duplicate action name in state")
            for choice in choices:
                choice.validate(len(self.states))

    def compact_json_dict(self) -> dict[str, Any]:
        return {
            "parameters": asdict(self.parameters),
            "state_count": len(self.states),
            "action_start_count": sum(len(x) for x in self.actions),
            "states": [s.label(self.parameters.belief_grid) for s in self.states],
            "action_names": sorted({a.name for choices in self.actions for a in choices}),
        }


# ---------------------------------------------------------------------------
# Parametric likelihood and synthetic dynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogitLikelihood:
    """Likelihood for causal previous-logit features.

    Conditional on latent difficulty D, the top-1/top-2 logit margin M is
    Gaussian and route churn J is Binomial(8, theta_D).  The finite controller
    observes a three-bin margin and a low/high churn bit.  Conditional
    independence is an explicit modeling assumption, not an engine fact.
    """

    margin_mu_easy: float = 2.25
    margin_mu_hard: float = 0.65
    margin_sigma_easy: float = 0.70
    margin_sigma_hard: float = 0.85
    margin_cut_1: float = 0.90
    margin_cut_2: float = 1.75
    churn_n: int = 8
    churn_theta_easy: float = 0.13
    churn_theta_hard: float = 0.52
    churn_cut: int = 2

    @staticmethod
    def _normal_cdf(x: float, mu: float, sigma: float) -> float:
        return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))

    def margin_bin_probability(self, difficulty: int, margin_bin: int) -> float:
        mu = self.margin_mu_hard if difficulty else self.margin_mu_easy
        sigma = self.margin_sigma_hard if difficulty else self.margin_sigma_easy
        c1 = self._normal_cdf(self.margin_cut_1, mu, sigma)
        c2 = self._normal_cdf(self.margin_cut_2, mu, sigma)
        return (c1, c2 - c1, 1.0 - c2)[margin_bin]

    def churn_probability(self, difficulty: int, churn_high: int) -> float:
        theta = self.churn_theta_hard if difficulty else self.churn_theta_easy
        low = sum(
            math.comb(self.churn_n, j)
            * theta**j
            * (1.0 - theta) ** (self.churn_n - j)
            for j in range(self.churn_cut + 1)
        )
        return (low, 1.0 - low)[churn_high]

    def probability(self, difficulty: int, observation: LogitObservation) -> float:
        return self.margin_bin_probability(difficulty, observation.margin_bin) * self.churn_probability(
            difficulty, observation.churn_high
        )

    def observations(self) -> tuple[LogitObservation, ...]:
        return tuple(LogitObservation(m, r) for m in range(3) for r in range(2))

    def likelihood_table(self) -> dict[str, dict[str, float]]:
        return {
            observation.name: {
                "easy": self.probability(0, observation),
                "hard": self.probability(1, observation),
                "likelihood_ratio_hard_over_easy": self.probability(1, observation)
                / max(self.probability(0, observation), _EPS),
            }
            for observation in self.observations()
        }


@dataclass(frozen=True)
class SyntheticParameters:
    """All free parameters for the finite model and simulator.

    Values are chosen to create a small, reproducible control problem.  They
    are not measurements of the private Insignia model.
    """

    belief_grid: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
    cache_levels: int = 3
    io_levels: int = 3
    measurement_ms: float = 1.25
    measurement_bytes_mb: float = 0.125
    metric_sensitivity: float = 0.91
    metric_false_positive: float = 0.09
    request_end_probability: float = 0.085
    hard_request_prior: float = 0.24
    hard_request_given_hard_regime: float = 0.72
    base_hard_after_easy: float = 0.075
    base_hard_after_hard: float = 0.79
    controller_ms: float = 0.42
    guard_ms: float = 0.18
    checkpoint_delta_d4_penalty_ms: float = 16.0
    checkpoint_delta_d4_queue_penalty_ms: float = 8.0
    byte_weight_tokens_per_mb: float = 0.0
    likelihood: LogitLikelihood = LogitLikelihood()

    def validate(self) -> None:
        grid = self.belief_grid
        if not grid or any(not 0 < q < 1 for q in grid):
            raise ValueError("belief points must lie strictly inside (0,1)")
        if any(grid[i] >= grid[i + 1] for i in range(len(grid) - 1)):
            raise ValueError("belief grid must be strictly increasing")
        if self.cache_levels < 2 or self.io_levels < 2:
            raise ValueError("cache and queue need at least two levels")
        if self.measurement_ms <= 0:
            raise ValueError("measurement must have positive duration")


@dataclass(frozen=True)
class ExpectedMetrics:
    committed: float
    time_ms: float
    bytes_mb: float
    ppl_loss: float
    hard_end: float
    hard_violation: float
    catastrophe: float
    harm_probability: float
    next_hard_if_easy: float
    next_hard_if_hard: float


def enumerate_compute_actions() -> tuple[ComputeAction, ...]:
    actions = tuple(
        ComputeAction(*values)
        for values in itertools.product(
            (4, 8),
            (2, 4),
            ("approx_delta", "exact_full"),
            ("fp8", "bf16"),
            (0, 1),
        )
    )
    return tuple(sorted(actions, key=lambda a: a.name))


def safe_exact_action() -> ComputeAction:
    return ComputeAction(8, 2, "exact_full", "bf16", 0)


def bayes_binary(prior_hard: float, likelihood_hard: float, likelihood_easy: float) -> float:
    numerator = prior_hard * likelihood_hard
    denominator = numerator + (1.0 - prior_hard) * likelihood_easy
    if denominator <= _EPS:
        return prior_hard
    return float(np.clip(numerator / denominator, 1e-9, 1.0 - 1e-9))


def project_belief(q: float, grid: Sequence[float]) -> int:
    return min(range(len(grid)), key=lambda i: (abs(grid[i] - q), i))


def metric_distribution(prior_hard: float, parameters: SyntheticParameters) -> tuple[tuple[int, float, float], ...]:
    """Return (metric_bit, probability, posterior_hard)."""

    result = []
    for metric in (0, 1):
        lh = parameters.metric_sensitivity if metric else 1.0 - parameters.metric_sensitivity
        le = parameters.metric_false_positive if metric else 1.0 - parameters.metric_false_positive
        probability = prior_hard * lh + (1.0 - prior_hard) * le
        posterior = bayes_binary(prior_hard, lh, le)
        result.append((metric, probability, posterior))
    return tuple(result)


def posterior_from_logit_observation(
    prior_hard: float, observation: LogitObservation, likelihood: LogitLikelihood
) -> float:
    return bayes_binary(
        prior_hard,
        likelihood.probability(1, observation),
        likelihood.probability(0, observation),
    )


def _accept_probability(q_hard: float, action: ComputeAction) -> float:
    easy = 0.925
    hard = 0.615
    if action.expert_k == 8:
        easy += 0.018
        hard += 0.105
    if action.precision == "bf16":
        easy += 0.008
        hard += 0.050
    if action.verify_policy == "approx_delta":
        # Approximate checks let a small number of mismatches pass, increasing
        # apparent acceptance while moving the cost to the quality counters.
        easy += 0.018
        hard += 0.050
    if action.expert_k == 4 and action.verify_policy == "approx_delta":
        hard += 0.035
    p = (1.0 - q_hard) * easy + q_hard * hard
    return float(np.clip(p, 0.08, 0.985))


def prefix_distribution(q_hard: float, action: ComputeAction) -> np.ndarray:
    # Conditional expectations in a belief MDP must be affine in the belief.
    # Mix the latent easy/hard prefix laws, rather than taking powers of a
    # posterior-averaged row-survival probability.
    if 1e-12 < q_hard < 1.0 - 1e-12:
        return (
            (1.0 - q_hard) * prefix_distribution(0.0, action)
            + q_hard * prefix_distribution(1.0, action)
        )
    p = _accept_probability(float(q_hard >= 0.5), action)
    width = action.draft_k
    distribution = np.empty(width + 1, dtype=float)
    distribution[0] = 1.0 - p
    for prefix in range(1, width):
        distribution[prefix] = p**prefix * (1.0 - p)
    distribution[width] = p**width
    return distribution / distribution.sum()


def expected_committed_tokens(q_hard: float, action: ComputeAction) -> float:
    distribution = prefix_distribution(q_hard, action)
    # Empty draft rounds still commit one exact fallback token.
    return float(sum(probability * max(1, prefix) for prefix, probability in enumerate(distribution)))


def action_harm_probability(
    q_hard: float,
    cache_level: int,
    io_queue: int,
    quality_debt: int,
    action: ComputeAction,
) -> float:
    low_expert = float(action.expert_k == 4)
    fp8 = float(action.precision == "fp8")
    approximate = float(action.verify_policy == "approx_delta")
    long_draft = float(action.draft_k == 4)

    easy_hazard = 0.00012 * low_expert + 0.00008 * fp8 + 0.00018 * approximate
    hard_hazard = 0.0060 * low_expert + 0.0035 * fp8 + 0.0110 * approximate
    interaction = q_hard * (
        0.0180 * low_expert * approximate
        + 0.0100 * long_draft * approximate
        + 0.0050 * low_expert * fp8
        + 0.0040 * long_draft * low_expert
    )
    cache_route_mismatch = q_hard * low_expert * max(0, 1 - cache_level) * 0.0025
    queue_stress = float(action.prefetch_budget and io_queue == 2) * 0.0020
    debt_burst = quality_debt * (0.0010 + 0.0090 * q_hard)
    hazard = (1.0 - q_hard) * easy_hazard + q_hard * hard_hazard
    return float(np.clip(hazard + interaction + cache_route_mismatch + queue_stress + debt_burst, 0, 0.35))


def hidden_transition_probabilities(q_hard: float, action: ComputeAction, parameters: SyntheticParameters) -> tuple[float, float]:
    """Action-dependent P(H_{t+1}|D_t=easy/hard).

    The low-expert/approximate interaction is the explicit future-route
    coupling requested by the problem: it is larger than the sum of the two
    separate perturbations.
    """

    low_expert = float(action.expert_k == 4)
    approximate = float(action.verify_policy == "approx_delta")
    fp8 = float(action.precision == "fp8")
    long_draft = float(action.draft_k == 4)

    route_shift = (
        0.022 * low_expert
        + 0.018 * approximate
        + 0.010 * fp8
        + 0.008 * long_draft
        + 0.105 * low_expert * approximate
        + 0.030 * long_draft * approximate
    )
    recovery = 0.025 * float(action.expert_k == 8 and action.verify_policy == "exact_full")
    p_hard_if_easy = parameters.base_hard_after_easy + 0.42 * route_shift - recovery
    p_hard_if_hard = parameters.base_hard_after_hard + route_shift - 0.7 * recovery
    return (
        float(np.clip(p_hard_if_easy, 0.01, 0.97)),
        float(np.clip(p_hard_if_hard, 0.05, 0.995)),
    )


def expected_metrics(
    q_hard: float,
    cache_level: int,
    io_queue: int,
    quality_debt: int,
    action: ComputeAction,
    parameters: SyntheticParameters,
) -> ExpectedMetrics:
    # The hidden regime is binary, so every one-step expectation is the
    # posterior mixture of its latent-regime expectations.  This affine
    # construction is essential for a valid value-of-information calculation.
    if 1e-12 < q_hard < 1.0 - 1e-12:
        easy = expected_metrics(
            0.0, cache_level, io_queue, quality_debt, action, parameters
        )
        hard = expected_metrics(
            1.0, cache_level, io_queue, quality_debt, action, parameters
        )
        def mix(field: str) -> float:
            return (1.0 - q_hard) * float(getattr(easy, field)) + q_hard * float(getattr(hard, field))
        return ExpectedMetrics(
            committed=mix("committed"),
            time_ms=mix("time_ms"),
            bytes_mb=mix("bytes_mb"),
            ppl_loss=mix("ppl_loss"),
            hard_end=mix("hard_end"),
            hard_violation=mix("hard_violation"),
            catastrophe=mix("catastrophe"),
            harm_probability=mix("harm_probability"),
            next_hard_if_easy=easy.next_hard_if_easy,
            next_hard_if_hard=hard.next_hard_if_hard,
        )
    committed = expected_committed_tokens(q_hard, action)
    distribution = prefix_distribution(q_hard, action)
    expected_verified = float(sum(probability * max(1, prefix) for prefix, probability in enumerate(distribution)))

    cache_hit = (0.22, 0.58, 0.84)[cache_level]
    miss_fraction = 1.0 - cache_hit
    if action.prefetch_budget:
        miss_fraction *= 0.72 if io_queue < 2 else 0.94

    expert_factor = action.expert_k / 4.0
    precision_factor = 0.72 if action.precision == "fp8" else 1.0
    verify_factor = 0.70 if action.verify_policy == "approx_delta" else 1.0

    draft_ms = 4.3 + 2.05 * action.draft_k
    compute_ms = expected_verified * (
        9.5 + 8.2 * expert_factor * precision_factor * verify_factor
    )
    io_ms = expected_verified * miss_fraction * (
        12.0 + 3.3 * expert_factor
    ) * (1.0 + 0.32 * io_queue)
    checkpoint_ms = 1.6 + 0.65 * action.draft_k
    if action.verify_policy == "exact_full":
        checkpoint_ms += 1.3 * action.draft_k
    elif action.draft_k == 4:
        checkpoint_ms += parameters.checkpoint_delta_d4_penalty_ms
        checkpoint_ms += parameters.checkpoint_delta_d4_queue_penalty_ms * (io_queue / 2.0)
    prefetch_ms = 0.0
    if action.prefetch_budget:
        prefetch_ms = 1.3 + 2.5 * max(0, io_queue - 0.5)

    time_ms = (
        parameters.controller_ms
        + draft_ms
        + compute_ms
        + io_ms
        + checkpoint_ms
        + prefetch_ms
    )

    bytes_mb = (
        expected_verified * miss_fraction * (7.0 + 5.0 * expert_factor)
        + action.draft_k * (1.2 if action.verify_policy == "approx_delta" else 3.8)
        + 14.0 * action.prefetch_budget
    )

    harm = action_harm_probability(q_hard, cache_level, io_queue, quality_debt, action)
    low_expert = float(action.expert_k == 4)
    fp8 = float(action.precision == "fp8")
    approximate = float(action.verify_policy == "approx_delta")
    continuous_loss_per_token = (
        (0.00020 + 0.00180 * q_hard) * low_expert
        + (0.00012 + 0.00110 * q_hard) * fp8
        + (0.00010 + 0.00150 * q_hard) * approximate
    )
    ppl_loss = committed * continuous_loss_per_token + harm * (0.22 + 0.22 * q_hard)

    hard_request_probability = (
        (1.0 - q_hard) * parameters.hard_request_prior
        + q_hard * parameters.hard_request_given_hard_regime
    )
    hard_end = parameters.request_end_probability * hard_request_probability
    debt_before_end = quality_debt + (1 - quality_debt) * harm
    severe_given_debt = 0.20 + 0.55 * q_hard
    hard_violation = hard_end * debt_before_end * severe_given_debt
    catastrophe = harm * (0.003 + 0.020 * q_hard + 0.006 * quality_debt)

    next_easy, next_hard = hidden_transition_probabilities(q_hard, action, parameters)
    return ExpectedMetrics(
        committed=committed,
        time_ms=float(time_ms),
        bytes_mb=float(bytes_mb),
        ppl_loss=float(ppl_loss),
        hard_end=float(hard_end),
        hard_violation=float(hard_violation),
        catastrophe=float(catastrophe),
        harm_probability=float(harm),
        next_hard_if_easy=next_easy,
        next_hard_if_hard=next_hard,
    )


def cache_transition_distribution(
    q_hard: float,
    cache_level: int,
    io_queue: int,
    action: ComputeAction,
    parameters: SyntheticParameters,
) -> tuple[tuple[int, float], ...]:
    low_expert = float(action.expert_k == 4)
    approximate = float(action.verify_policy == "approx_delta")
    long_draft = float(action.draft_k == 4)
    p_down = (
        0.055
        + 0.18 * q_hard * low_expert
        + 0.19 * low_expert * approximate
        + 0.07 * long_draft * approximate
        + 0.09 * float(action.prefetch_budget and io_queue == parameters.io_levels - 1)
    )
    p_up = (
        0.06
        + 0.50 * action.prefetch_budget * (1.0 - 0.24 * io_queue)
        + 0.05 * float(action.expert_k == 8)
    )
    p_down = float(np.clip(p_down, 0.01, 0.85))
    p_up = float(np.clip(p_up, 0.01, 0.80))
    if p_down + p_up > 0.94:
        scale = 0.94 / (p_down + p_up)
        p_down *= scale
        p_up *= scale
    probabilities: dict[int, float] = {}
    for target, probability in (
        (max(0, cache_level - 1), p_down),
        (min(parameters.cache_levels - 1, cache_level + 1), p_up),
        (cache_level, 1.0 - p_down - p_up),
    ):
        probabilities[target] = probabilities.get(target, 0.0) + probability
    return tuple(sorted(probabilities.items()))


def io_transition_distribution(
    io_queue: int, action: ComputeAction, parameters: SyntheticParameters
) -> tuple[tuple[int, float], ...]:
    long_delta = float(action.draft_k == 4 and action.verify_policy == "approx_delta")
    load = 0.11 + 0.58 * action.prefetch_budget + 0.34 * long_delta
    load += 0.08 * float(action.expert_k == 8)
    p_up = float(np.clip(load * (1.0 - 0.15 * io_queue), 0.02, 0.82))
    p_down = float(np.clip(0.57 - 0.42 * load + 0.05 * io_queue, 0.04, 0.72))
    if p_up + p_down > 0.95:
        scale = 0.95 / (p_up + p_down)
        p_up *= scale
        p_down *= scale
    probabilities: dict[int, float] = {}
    for target, probability in (
        (min(parameters.io_levels - 1, io_queue + 1), p_up),
        (max(0, io_queue - 1), p_down),
        (io_queue, 1.0 - p_up - p_down),
    ):
        probabilities[target] = probabilities.get(target, 0.0) + probability
    return tuple(sorted(probabilities.items()))


def debt_transition_distribution(
    quality_debt: int, harm_probability: float, parameters: SyntheticParameters
) -> tuple[tuple[int, float], ...]:
    debt_before_end = quality_debt + (1 - quality_debt) * harm_probability
    p_debt = (1.0 - parameters.request_end_probability) * debt_before_end
    return ((0, 1.0 - p_debt), (1, p_debt))


def _combine_transitions(
    components: Iterable[tuple[tuple[Any, float], ...]],
) -> Iterable[tuple[tuple[Any, ...], float]]:
    for selections in itertools.product(*components):
        values = tuple(item[0] for item in selections)
        probability = math.prod(item[1] for item in selections)
        if probability > 1e-15:
            yield values, probability


def build_finite_model(parameters: SyntheticParameters | None = None) -> FiniteControllerModel:
    parameters = parameters or SyntheticParameters()
    parameters.validate()
    states = tuple(
        ControllerState(*values)
        for values in itertools.product(
            range(len(parameters.belief_grid)),
            range(parameters.cache_levels),
            range(parameters.io_levels),
            (0, 1),
            ("ready", "measured"),
        )
    )
    state_index = {state: i for i, state in enumerate(states)}
    compute_actions = enumerate_compute_actions()
    action_lists: list[tuple[ActionKernel, ...]] = []

    for state in states:
        q_hard = parameters.belief_grid[state.belief_index]
        choices: list[ActionKernel] = []
        if state.phase == "ready":
            metric_transitions: dict[int, float] = {}
            for _, probability, posterior in metric_distribution(q_hard, parameters):
                next_state = ControllerState(
                    project_belief(posterior, parameters.belief_grid),
                    state.cache_level,
                    state.io_queue,
                    state.quality_debt,
                    "measured",
                )
                idx = state_index[next_state]
                metric_transitions[idx] = metric_transitions.get(idx, 0.0) + probability
            choices.append(
                ActionKernel(
                    name="measure_exact_metric",
                    transitions=tuple(sorted(metric_transitions.items())),
                    committed=0.0,
                    time_ms=parameters.measurement_ms,
                    bytes_mb=parameters.measurement_bytes_mb,
                    ppl_loss=0.0,
                    hard_end=0.0,
                    hard_violation=0.0,
                    catastrophe=0.0,
                    measurement=True,
                )
            )

        for action in compute_actions:
            metrics = expected_metrics(
                q_hard,
                state.cache_level,
                state.io_queue,
                state.quality_debt,
                action,
                parameters,
            )
            predicted_hard = (
                (1.0 - q_hard) * metrics.next_hard_if_easy
                + q_hard * metrics.next_hard_if_hard
            )
            belief_transitions: list[tuple[int, float]] = []
            for observation in parameters.likelihood.observations():
                p_obs = (
                    (1.0 - predicted_hard)
                    * parameters.likelihood.probability(0, observation)
                    + predicted_hard
                    * parameters.likelihood.probability(1, observation)
                )
                posterior = posterior_from_logit_observation(
                    predicted_hard, observation, parameters.likelihood
                )
                belief_transitions.append(
                    (project_belief(posterior, parameters.belief_grid), p_obs)
                )
            # Projection can merge several observations into one belief point.
            merged_beliefs: dict[int, float] = {}
            for index, probability in belief_transitions:
                merged_beliefs[index] = merged_beliefs.get(index, 0.0) + probability

            cache_distribution = cache_transition_distribution(
                q_hard, state.cache_level, state.io_queue, action, parameters
            )
            io_distribution = io_transition_distribution(state.io_queue, action, parameters)
            debt_distribution = debt_transition_distribution(
                state.quality_debt, metrics.harm_probability, parameters
            )
            transitions: dict[int, float] = {}
            for (belief_index, cache, io, debt), probability in _combine_transitions(
                (
                    tuple(sorted(merged_beliefs.items())),
                    cache_distribution,
                    io_distribution,
                    debt_distribution,
                )
            ):
                next_state = ControllerState(
                    belief_index, cache, io, debt, "ready"
                )
                idx = state_index[next_state]
                transitions[idx] = transitions.get(idx, 0.0) + probability

            total = sum(transitions.values())
            if total <= _EPS:
                raise RuntimeError("zero transition mass")
            normalized = tuple(sorted((idx, p / total) for idx, p in transitions.items()))
            choices.append(
                ActionKernel(
                    name=action.name,
                    transitions=normalized,
                    committed=metrics.committed,
                    time_ms=metrics.time_ms,
                    bytes_mb=metrics.bytes_mb,
                    ppl_loss=metrics.ppl_loss,
                    hard_end=metrics.hard_end,
                    hard_violation=metrics.hard_violation,
                    catastrophe=metrics.catastrophe,
                )
            )
        action_lists.append(tuple(choices))

    model = FiniteControllerModel(states, tuple(action_lists), parameters)
    model.validate()
    return model


# ---------------------------------------------------------------------------
# Exact occupation-measure LP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyEntry:
    state: str
    action: str
    probability: float
    occupation_per_ms: float


@dataclass(frozen=True)
class ExactSolveResult:
    committed_tokens_per_second: float
    utility_per_second: float
    bytes_per_second_mb: float
    ppl_loss_per_committed_token: float
    hard_violation_probability: float
    catastrophe_per_committed_token: float
    decision_starts_per_second: float
    measurement_fraction: float
    expected_sojourn_ms: float
    policy: tuple[PolicyEntry, ...]
    variable_labels: tuple[tuple[str, str], ...]
    raw_occupation: tuple[float, ...]
    constraint_shadow_prices: Mapping[str, float]
    solver_message: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variable_labels"] = [list(x) for x in self.variable_labels]
        payload["constraint_shadow_prices"] = dict(self.constraint_shadow_prices)
        return payload


def _flatten_model(model: FiniteControllerModel):
    model.validate()
    labels: list[tuple[int, ActionKernel]] = []
    for state_index, choices in enumerate(model.actions):
        labels.extend((state_index, action) for action in choices)
    n_vars = len(labels)
    arrays = {
        field: np.empty(n_vars, dtype=float)
        for field in (
            "committed",
            "time_ms",
            "bytes_mb",
            "ppl_loss",
            "hard_end",
            "hard_violation",
            "catastrophe",
            "measurement",
        )
    }
    row: list[int] = []
    col: list[int] = []
    data: list[float] = []
    for variable, (source, action) in enumerate(labels):
        row.append(source)
        col.append(variable)
        data.append(1.0)
        for target, probability in action.transitions:
            row.append(target)
            col.append(variable)
            data.append(-probability)
        arrays["committed"][variable] = action.committed
        arrays["time_ms"][variable] = action.time_ms
        arrays["bytes_mb"][variable] = action.bytes_mb
        arrays["ppl_loss"][variable] = action.ppl_loss
        arrays["hard_end"][variable] = action.hard_end
        arrays["hard_violation"][variable] = action.hard_violation
        arrays["catastrophe"][variable] = action.catastrophe
        arrays["measurement"][variable] = float(action.measurement)
    flow = coo_matrix((data, (row, col)), shape=(len(model.states), n_vars)).tocsr()
    return labels, flow, arrays


def solve_occupation_lp(
    model: FiniteControllerModel,
    *,
    max_ppl_loss_per_token: float | None = 0.0045,
    max_hard_violation_probability: float | None = 0.035,
    max_catastrophe_per_token: float | None = 0.00035,
    byte_weight_tokens_per_mb: float | None = None,
) -> ExactSolveResult:
    """Solve the finite communicating CSMDP exactly by occupation measure."""

    byte_weight = (
        model.parameters.byte_weight_tokens_per_mb
        if byte_weight_tokens_per_mb is None
        else byte_weight_tokens_per_mb
    )
    if byte_weight < 0:
        raise ValueError("byte weight must be non-negative")
    for value in (
        max_ppl_loss_per_token,
        max_hard_violation_probability,
        max_catastrophe_per_token,
    ):
        if value is not None and value < 0:
            raise ValueError("quality budgets must be non-negative")

    labels, flow, arrays = _flatten_model(model)
    n_vars = len(labels)
    # One flow equation is redundant in a communicating chain.
    time_row = csr_matrix(arrays["time_ms"][None, :])
    A_eq = vstack([flow[:-1], time_row], format="csr")
    b_eq = np.r_[np.zeros(len(model.states) - 1), 1.0]

    inequalities: list[np.ndarray] = []
    names: list[str] = []
    if max_ppl_loss_per_token is not None:
        inequalities.append(
            arrays["ppl_loss"] - max_ppl_loss_per_token * arrays["committed"]
        )
        names.append("ppl_loss_per_token")
    if max_hard_violation_probability is not None:
        inequalities.append(
            arrays["hard_violation"]
            - max_hard_violation_probability * arrays["hard_end"]
        )
        names.append("hard_violation_probability")
    if max_catastrophe_per_token is not None:
        inequalities.append(
            arrays["catastrophe"]
            - max_catastrophe_per_token * arrays["committed"]
        )
        names.append("catastrophe_per_token")
    A_ub = csr_matrix(np.vstack(inequalities)) if inequalities else None
    b_ub = np.zeros(len(inequalities)) if inequalities else None

    utility = arrays["committed"] - byte_weight * arrays["bytes_mb"]
    result = linprog(
        -utility,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * n_vars,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"occupation LP failed: {result.message}")
    occupation = np.asarray(result.x, dtype=float)
    rates = {name: float(occupation @ values) for name, values in arrays.items()}
    starts = float(occupation.sum())
    if starts <= _EPS:
        raise RuntimeError("zero decision-start rate")

    policy_rows: list[PolicyEntry] = []
    for source, state in enumerate(model.states):
        positions = [i for i, (s, _) in enumerate(labels) if s == source]
        mass = float(occupation[positions].sum())
        if mass <= 1e-10:
            continue
        for i in positions:
            probability = float(occupation[i] / mass)
            if probability > 1e-8:
                policy_rows.append(
                    PolicyEntry(
                        state.label(model.parameters.belief_grid),
                        labels[i][1].name,
                        probability,
                        float(occupation[i]),
                    )
                )

    committed_rate = rates["committed"]
    hard_end_rate = rates["hard_end"]
    shadow_prices: dict[str, float] = {}
    if inequalities:
        marginals = np.asarray(result.ineqlin.marginals)
        # linprog minimizes -utility, so active resource prices have the
        # opposite sign of the reported inequality marginals.
        shadow_prices = {
            name: max(0.0, -float(marginal))
            for name, marginal in zip(names, marginals)
        }

    return ExactSolveResult(
        committed_tokens_per_second=1000.0 * committed_rate,
        utility_per_second=1000.0 * float(occupation @ utility),
        bytes_per_second_mb=1000.0 * rates["bytes_mb"],
        ppl_loss_per_committed_token=(
            rates["ppl_loss"] / committed_rate if committed_rate > _EPS else math.inf
        ),
        hard_violation_probability=(
            rates["hard_violation"] / hard_end_rate if hard_end_rate > _EPS else math.nan
        ),
        catastrophe_per_committed_token=(
            rates["catastrophe"] / committed_rate if committed_rate > _EPS else math.inf
        ),
        decision_starts_per_second=1000.0 * starts,
        measurement_fraction=rates["measurement"] / starts,
        expected_sojourn_ms=1.0 / starts,
        policy=tuple(policy_rows),
        variable_labels=tuple(
            (model.states[source].label(model.parameters.belief_grid), action.name)
            for source, action in labels
        ),
        raw_occupation=tuple(float(x) for x in occupation),
        constraint_shadow_prices=shadow_prices,
        solver_message=str(result.message),
    )


def policy_map(result: ExactSolveResult) -> dict[str, tuple[tuple[str, float], ...]]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for row in result.policy:
        grouped.setdefault(row.state, []).append((row.action, row.probability))
    return {state: tuple(entries) for state, entries in grouped.items()}


# ---------------------------------------------------------------------------
# Value of information, thresholds, safety bounds, and counterexamples
# ---------------------------------------------------------------------------


def one_step_transformed_value(
    metrics: ExpectedMetrics,
    *,
    throughput_tokens_per_ms: float,
    ppl_multiplier: float = 0.0,
    ppl_budget_per_token: float = 0.0,
    hard_multiplier: float = 0.0,
    hard_budget: float = 0.0,
    catastrophe_multiplier: float = 0.0,
    catastrophe_budget_per_token: float = 0.0,
) -> float:
    return (
        metrics.committed
        - throughput_tokens_per_ms * metrics.time_ms
        - ppl_multiplier
        * (metrics.ppl_loss - ppl_budget_per_token * metrics.committed)
        - hard_multiplier
        * (metrics.hard_violation - hard_budget * metrics.hard_end)
        - catastrophe_multiplier
        * (metrics.catastrophe - catastrophe_budget_per_token * metrics.committed)
    )


def myopic_value_of_information(
    q_hard: float,
    cache_level: int,
    io_queue: int,
    quality_debt: int,
    candidate_actions: Sequence[ComputeAction],
    parameters: SyntheticParameters,
    *,
    throughput_tokens_per_ms: float,
    ppl_multiplier: float = 0.0,
    hard_multiplier: float = 0.0,
    catastrophe_multiplier: float = 0.0,
    ppl_budget_per_token: float = 0.0045,
    hard_budget: float = 0.035,
    catastrophe_budget_per_token: float = 0.00035,
) -> dict[str, Any]:
    def value(q: float, action: ComputeAction) -> float:
        return one_step_transformed_value(
            expected_metrics(
                q, cache_level, io_queue, quality_debt, action, parameters
            ),
            throughput_tokens_per_ms=throughput_tokens_per_ms,
            ppl_multiplier=ppl_multiplier,
            ppl_budget_per_token=ppl_budget_per_token,
            hard_multiplier=hard_multiplier,
            hard_budget=hard_budget,
            catastrophe_multiplier=catastrophe_multiplier,
            catastrophe_budget_per_token=catastrophe_budget_per_token,
        )

    prior_values = {a.name: value(q_hard, a) for a in candidate_actions}
    no_measure_action = max(candidate_actions, key=lambda a: prior_values[a.name])
    no_measure_value = prior_values[no_measure_action.name]
    contingent_value = 0.0
    branches = []
    for metric, probability, posterior in metric_distribution(q_hard, parameters):
        branch_values = {a.name: value(posterior, a) for a in candidate_actions}
        branch_action = max(candidate_actions, key=lambda a: branch_values[a.name])
        branch_value = branch_values[branch_action.name]
        contingent_value += probability * branch_value
        branches.append(
            {
                "metric": metric,
                "probability": probability,
                "posterior_hard": posterior,
                "action": branch_action.name,
                "value": branch_value,
            }
        )
    gross_voi = contingent_value - no_measure_value
    net_voi = gross_voi - throughput_tokens_per_ms * parameters.measurement_ms
    return {
        "prior_hard": q_hard,
        "no_measure_action": no_measure_action.name,
        "no_measure_value": no_measure_value,
        "gross_voi": gross_voi,
        "measurement_time_value": throughput_tokens_per_ms * parameters.measurement_ms,
        "net_voi": net_voi,
        "measure": net_voi > 0,
        "branches": branches,
    }


def monotone_action_threshold(
    low_intercept: float,
    low_hard_slope: float,
    high_intercept: float,
    high_hard_slope: float,
) -> float:
    """Posterior threshold where an ordered high-compute action overtakes low."""

    denominator = high_hard_slope - low_hard_slope
    if denominator <= 0:
        raise ValueError("high action lacks increasing differences in difficulty")
    return (low_intercept - high_intercept) / denominator


def robust_safety_certificate(
    metrics: ExpectedMetrics,
    safe_metrics: ExpectedMetrics,
    *,
    throughput_tokens_per_ms: float,
    controller_and_guard_ms: float,
    ppl_radius: float,
    hard_risk_radius: float,
    catastrophe_radius: float,
    time_radius_ms: float,
    ppl_envelope_per_token: float,
    hard_envelope: float,
    catastrophe_envelope_per_token: float,
    uncertainty_radius: float,
    max_uncertainty_radius: float,
    overlap: float,
    min_overlap: float,
) -> dict[str, Any]:
    committed_lcb = max(0.0, metrics.committed - uncertainty_radius)
    time_ucb = metrics.time_ms + time_radius_ms + controller_and_guard_ms
    saving_lcb_ms = safe_metrics.time_ms - time_ucb
    ppl_ucb = (metrics.ppl_loss + ppl_radius) / max(committed_lcb, _EPS)
    hard_ucb = (metrics.hard_violation + hard_risk_radius) / max(metrics.hard_end, _EPS)
    catastrophe_ucb = (metrics.catastrophe + catastrophe_radius) / max(committed_lcb, _EPS)
    # Compare against exact fallback in Dinkelbach units.  A longer action can
    # still be better when it commits more tokens, so a raw-time saving is not
    # required.
    economic_gain_lcb = (
        committed_lcb
        - safe_metrics.committed
        - throughput_tokens_per_ms * (time_ucb - safe_metrics.time_ms)
    )
    allowed = (
        uncertainty_radius <= max_uncertainty_radius
        and overlap >= min_overlap
        and economic_gain_lcb > 0
        and ppl_ucb <= ppl_envelope_per_token
        and hard_ucb <= hard_envelope
        and catastrophe_ucb <= catastrophe_envelope_per_token
    )
    return {
        "allowed": allowed,
        "committed_lcb": committed_lcb,
        "time_ucb_ms": time_ucb,
        "saving_lcb_ms": saving_lcb_ms,
        "economic_gain_lcb": economic_gain_lcb,
        "ppl_ucb": ppl_ucb,
        "hard_violation_ucb": hard_ucb,
        "catastrophe_ucb": catastrophe_ucb,
        "uncertainty_radius": uncertainty_radius,
        "overlap": overlap,
    }


def lyapunov_performance_bound(
    *,
    drift_constant: float,
    V: float,
    tau_min_ms: float,
    model_error: float = 0.0,
    belief_diameter: float = 0.0,
    belief_value_lipschitz: float = 0.0,
    planning_error: float = 0.0,
    horizon: int | None = None,
    terminal_virtual_queues: Sequence[float] = (),
) -> dict[str, Any]:
    """Return the standard drift-plus-penalty gap and violation bounds.

    ``drift_constant`` is B = 1/2 sum_i G_i^2.  Uniform one-step model
    error, belief compression, and approximate maximization enter the action
    value error as

        eps_Q = planning + 2 * model + 2 * L_b * diameter.

    At the optimal Dinkelbach root, the resulting utility-rate gap is at most
    ``(B / V + eps_Q) / tau_min_ms`` utility units per millisecond.  The
    pathwise finite-horizon positive average violation is bounded by Q_i(T)/T.
    """

    values = (
        drift_constant,
        V,
        tau_min_ms,
        model_error,
        belief_diameter,
        belief_value_lipschitz,
        planning_error,
    )
    if any(value < 0 for value in values):
        raise ValueError("bound inputs must be non-negative")
    if V <= 0 or tau_min_ms <= 0:
        raise ValueError("V and tau_min_ms must be positive")
    if horizon is not None and horizon <= 0:
        raise ValueError("horizon must be positive")
    if any(queue < 0 for queue in terminal_virtual_queues):
        raise ValueError("virtual queues must be non-negative")

    action_value_error = (
        planning_error
        + 2.0 * model_error
        + 2.0 * belief_value_lipschitz * belief_diameter
    )
    transformed_gap = drift_constant / V + action_value_error
    result: dict[str, Any] = {
        "action_value_error": action_value_error,
        "transformed_objective_gap_per_decision": transformed_gap,
        "utility_rate_gap_per_ms": transformed_gap / tau_min_ms,
        "utility_rate_gap_per_second": 1000.0 * transformed_gap / tau_min_ms,
    }
    if horizon is not None:
        result["positive_average_constraint_violation_bounds"] = tuple(
            queue / horizon for queue in terminal_virtual_queues
        )
    return result


def throughput_price_of_safety(
    *,
    joint_tokens_per_round: float,
    joint_ms_per_round: float,
    exact_tokens_per_round: float,
    exact_ms_per_round: float,
    fallback_fraction: float,
    guard_ms_per_round: float,
) -> dict[str, float]:
    if not 0 <= fallback_fraction <= 1:
        raise ValueError("fallback fraction must lie in [0,1]")
    if min(joint_ms_per_round, exact_ms_per_round) <= 0 or guard_ms_per_round < 0:
        raise ValueError("invalid timing")
    mixed_tokens = (
        (1.0 - fallback_fraction) * joint_tokens_per_round
        + fallback_fraction * exact_tokens_per_round
    )
    mixed_time = (
        (1.0 - fallback_fraction) * joint_ms_per_round
        + fallback_fraction * exact_ms_per_round
        + guard_ms_per_round
    )
    joint_rate = 1000.0 * joint_tokens_per_round / joint_ms_per_round
    safe_rate = 1000.0 * mixed_tokens / mixed_time
    return {
        "joint_tokens_per_second": joint_rate,
        "safe_tokens_per_second": safe_rate,
        "absolute_price_tokens_per_second": joint_rate - safe_rate,
        "relative_price": (joint_rate - safe_rate) / max(joint_rate, _EPS),
    }


def expert_route_counterexample() -> dict[str, Any]:
    """Low expert count is myopically faster but poisons the next route."""

    rows = [
        {
            "expert_k": 4,
            "prefetch": 0,
            "current_tokens": 1.0,
            "current_ms": 1.0,
            "next_route": "cold-B",
            "next_tokens": 1.0,
            "next_ms": 10.0,
        },
        {
            "expert_k": 4,
            "prefetch": 1,
            "current_tokens": 1.0,
            "current_ms": 1.4,
            "next_route": "wrong-prefetch-B",
            "next_tokens": 1.0,
            "next_ms": 11.0,
        },
        {
            "expert_k": 8,
            "prefetch": 0,
            "current_tokens": 1.0,
            "current_ms": 2.0,
            "next_route": "warm-A",
            "next_tokens": 1.0,
            "next_ms": 1.0,
        },
        {
            "expert_k": 8,
            "prefetch": 1,
            "current_tokens": 1.0,
            "current_ms": 2.2,
            "next_route": "hot-A",
            "next_tokens": 1.0,
            "next_ms": 0.6,
        },
    ]
    for row in rows:
        row["myopic_tps"] = 1000.0 * row["current_tokens"] / row["current_ms"]
        row["two_block_tps"] = 1000.0 * (
            row["current_tokens"] + row["next_tokens"]
        ) / (row["current_ms"] + row["next_ms"])
    table = {(r["expert_k"], r["prefetch"]): r["two_block_tps"] for r in rows}
    cross_difference = table[(8, 1)] + table[(4, 0)] - table[(8, 0)] - table[(4, 1)]
    return {
        "rows": rows,
        "myopic_choice": max(rows, key=lambda r: r["myopic_tps"]),
        "dynamic_choice": max(rows, key=lambda r: r["two_block_tps"]),
        "cross_difference": cross_difference,
        "separable": math.isclose(cross_difference, 0.0, abs_tol=1e-12),
    }


def draft_checkpoint_counterexample() -> dict[str, Any]:
    rows = [
        {"draft_k": 2, "checkpoint": "delta", "committed": 1.8, "time_ms": 2.0},
        {"draft_k": 2, "checkpoint": "full", "committed": 1.8, "time_ms": 2.4},
        {"draft_k": 4, "checkpoint": "delta", "committed": 3.2, "time_ms": 11.0},
        {"draft_k": 4, "checkpoint": "full", "committed": 3.2, "time_ms": 3.6},
    ]
    for row in rows:
        row["tps"] = 1000.0 * row["committed"] / row["time_ms"]
    table = {(r["draft_k"], r["checkpoint"]): r["tps"] for r in rows}
    independent_draft = max(
        (r for r in rows if r["checkpoint"] == "full"), key=lambda r: r["tps"]
    )["draft_k"]
    independent_checkpoint = max(
        (r for r in rows if r["draft_k"] == 2), key=lambda r: r["tps"]
    )["checkpoint"]
    combination = next(
        r
        for r in rows
        if r["draft_k"] == independent_draft
        and r["checkpoint"] == independent_checkpoint
    )
    cross_difference = (
        table[(4, "full")]
        + table[(2, "delta")]
        - table[(4, "delta")]
        - table[(2, "full")]
    )
    return {
        "rows": rows,
        "independent_draft_choice": independent_draft,
        "independent_checkpoint_choice": independent_checkpoint,
        "independent_combination": combination,
        "joint_choice": max(rows, key=lambda r: r["tps"]),
        "cross_difference": cross_difference,
        "separable": math.isclose(cross_difference, 0.0, abs_tol=1e-12),
    }


def every_heuristic_locally_sensible_case() -> dict[str, Any]:
    """A static payoff table where one-at-a-time gains combine disastrously."""

    baseline = 100.0
    individual = {
        "expert_k_4": 104.0,
        "draft_k_4": 105.0,
        "approx_verify": 103.0,
        "fp8": 102.5,
        "prefetch_1": 104.5,
    }
    combined_nominal_without_interactions = baseline + sum(v - baseline for v in individual.values())
    interaction_penalties = {
        "low_expert_x_approx_future_route": 10.0,
        "draft4_x_delta_checkpoint": 15.0,
        "prefetch_x_checkpoint_queue": 9.0,
        "low_expert_x_fp8_quality_fallback": 7.0,
    }
    combined = combined_nominal_without_interactions - sum(interaction_penalties.values())
    return {
        "baseline": baseline,
        "one_at_a_time": individual,
        "additive_prediction": combined_nominal_without_interactions,
        "interaction_penalties": interaction_penalties,
        "actual_combination": combined,
        "all_individual_improve": all(v > baseline for v in individual.values()),
        "combination_worse_than_baseline": combined < baseline,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_solution(result: ExactSolveResult) -> None:
    print(f"committed throughput : {result.committed_tokens_per_second:.6f} tok/s")
    print(f"utility rate         : {result.utility_per_second:.6f} units/s")
    print(f"bytes                : {result.bytes_per_second_mb:.3f} MB/s")
    print(f"PPL loss/token       : {result.ppl_loss_per_committed_token:.8f}")
    print(f"hard violation       : {result.hard_violation_probability:.8f}")
    print(f"catastrophe/token    : {result.catastrophe_per_committed_token:.8f}")
    print(f"measurement fraction : {result.measurement_fraction:.6f}")
    print(f"mean decision ms     : {result.expected_sojourn_ms:.6f}")
    print("shadow prices:")
    for name, price in result.constraint_shadow_prices.items():
        print(f"  {name:32s} {price:.8f}")
    print("policy support:")
    for row in result.policy:
        print(f"  {row.state:38s} {row.action:32s} p={row.probability:.6f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve-demo", action="store_true")
    parser.add_argument("--write-model-summary", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--counterexamples", action="store_true")
    parser.add_argument("--likelihood-table", action="store_true")
    parser.add_argument("--voi", type=float, metavar="POSTERIOR_HARD")
    parser.add_argument("--max-ppl", type=float, default=0.0045)
    parser.add_argument("--max-hard", type=float, default=0.035)
    parser.add_argument("--max-catastrophe", type=float, default=0.00035)
    args = parser.parse_args(argv)

    parameters = SyntheticParameters()
    if args.counterexamples:
        print(
            json.dumps(
                {
                    "expert_route": expert_route_counterexample(),
                    "draft_checkpoint": draft_checkpoint_counterexample(),
                    "locally_sensible_unstable": every_heuristic_locally_sensible_case(),
                },
                indent=2,
            )
        )
        return 0
    if args.likelihood_table:
        print(json.dumps(parameters.likelihood.likelihood_table(), indent=2))
        return 0
    if args.voi is not None:
        print(
            json.dumps(
                myopic_value_of_information(
                    args.voi,
                    cache_level=1,
                    io_queue=0,
                    quality_debt=0,
                    candidate_actions=enumerate_compute_actions(),
                    parameters=parameters,
                    throughput_tokens_per_ms=0.035,
                    ppl_multiplier=12.0,
                    hard_multiplier=4.0,
                    catastrophe_multiplier=8.0,
                ),
                indent=2,
            )
        )
        return 0

    model = build_finite_model(parameters)
    if args.write_model_summary:
        args.write_model_summary.parent.mkdir(parents=True, exist_ok=True)
        args.write_model_summary.write_text(
            json.dumps(model.compact_json_dict(), indent=2) + "\n"
        )
    if args.solve_demo or args.json_out:
        result = solve_occupation_lp(
            model,
            max_ppl_loss_per_token=args.max_ppl,
            max_hard_violation_probability=args.max_hard,
            max_catastrophe_per_token=args.max_catastrophe,
        )
        _print_solution(result)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n")
        return 0
    parser.error(
        "choose --solve-demo, --write-model-summary, --counterexamples, "
        "--likelihood-table, or --voi"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
