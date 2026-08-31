#!/usr/bin/env python3
"""Finite-state certificates and adversaries for teacher forcing vs free run.

The module deliberately separates three regimes:

* stochastic decoding, where exact-policy occupancy KL composes by the path-space
  chain rule and the sharp event certificate is an inverse binary-KL bound;
* greedy decoding, where a pointwise logit-margin condition is necessary and
  sufficient along the common prefix;
* sequential certification, where candidate-policy trajectories are sampled in
  a predictable exact/candidate/intervention mixture and monitored by an
  anytime-valid confidence sequence.

Small models may use ``fractions.Fraction`` throughout.  Their trajectory laws,
state occupancies, first-failure probabilities, and KL coefficients are then
exact rationals; logarithms are retained as a symbolic rational linear
combination rather than being exponentiated into underflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from scipy.optimize import brentq
from scipy.special import betaln, logsumexp

Probability = float | Fraction
_EPS = 1e-12


def _is_fraction(value: object) -> bool:
    return isinstance(value, Fraction)


def _zero_like(value: Probability) -> Probability:
    return Fraction(0, 1) if _is_fraction(value) else 0.0


def _one_like(value: Probability) -> Probability:
    return Fraction(1, 1) if _is_fraction(value) else 1.0


def _probability_sum(values: Sequence[Probability]) -> Probability:
    if values and all(_is_fraction(v) for v in values):
        return sum(values, Fraction(0, 1))
    return float(sum(float(v) for v in values))


def _validate_distribution(values: Sequence[Probability], label: str) -> None:
    if not values:
        raise ValueError(f"{label}: empty distribution")
    if any(float(v) < -_EPS for v in values):
        raise ValueError(f"{label}: negative probability")
    total = _probability_sum(values)
    if _is_fraction(total):
        if total != 1:
            raise ValueError(f"{label}: probabilities sum to {total}, not 1")
    elif not math.isclose(float(total), 1.0, abs_tol=1e-10):
        raise ValueError(f"{label}: probabilities sum to {total}, not 1")


@dataclass
class SymbolicKL:
    """An exact expression ``sum coefficient * log(argument)``.

    Coefficients and arguments are rational.  This is exact for finite rational
    models even though KL itself is usually transcendental.  ``infinite`` marks
    a P-positive/Q-zero atom.
    """

    terms: MutableMapping[Fraction, Fraction]
    infinite: bool = False

    @classmethod
    def zero(cls) -> "SymbolicKL":
        return cls({})

    def add(self, coefficient: Fraction, numerator: Fraction,
            denominator: Fraction) -> None:
        if coefficient == 0 or numerator == 0:
            return
        if denominator == 0:
            self.infinite = True
            return
        argument = numerator / denominator
        if argument == 1:
            return
        self.terms[argument] = self.terms.get(argument, Fraction(0, 1)) + coefficient
        if self.terms[argument] == 0:
            del self.terms[argument]

    def __float__(self) -> float:
        if self.infinite:
            return math.inf
        return math.fsum(float(c) * math.log(float(r)) for r, c in self.terms.items())

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "infinite": self.infinite,
            "terms": [
                {
                    "coefficient": f"{c.numerator}/{c.denominator}",
                    "argument": f"{r.numerator}/{r.denominator}",
                }
                for r, c in sorted(self.terms.items(), key=lambda item: float(item[0]))
            ],
            "float": float(self),
        }

    def __str__(self) -> str:
        if self.infinite:
            return "infinity"
        if not self.terms:
            return "0"
        pieces = []
        for ratio, coefficient in sorted(self.terms.items(), key=lambda item: float(item[0])):
            pieces.append(f"({coefficient})*log({ratio})")
        return " + ".join(pieces)


@dataclass(frozen=True)
class FiniteStateAR:
    """A time-inhomogeneous finite-state autoregressive pair.

    ``transition[state][token]`` is deterministic.  Stochastic finite-state
    transitions can be represented by splitting tokens.  A path is a failure as
    soon as its post-token state belongs to ``failure_states``; augmenting state
    with a monitor supports format automata or cumulative-loss budgets.
    """

    vocab: tuple[str, ...]
    states: tuple[str, ...]
    initial_state: str
    transition: Mapping[str, tuple[str, ...]]
    p_kernels: tuple[Mapping[str, tuple[Probability, ...]], ...]
    q_kernels: tuple[Mapping[str, tuple[Probability, ...]], ...]
    failure_states: frozenset[str]

    @property
    def horizon(self) -> int:
        return len(self.p_kernels)

    @property
    def exact_rational(self) -> bool:
        values: list[Probability] = []
        for kernels in (self.p_kernels, self.q_kernels):
            for kernel in kernels:
                for state in self.states:
                    values.extend(kernel[state])
        return bool(values) and all(_is_fraction(v) for v in values)

    def validate(self) -> None:
        if not self.vocab or len(self.vocab) > 8:
            raise ValueError("vocabulary size must lie in [1, 8]")
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocabulary entries must be unique")
        if not self.states or len(set(self.states)) != len(self.states):
            raise ValueError("states must be non-empty and unique")
        state_set = set(self.states)
        if self.initial_state not in state_set:
            raise ValueError("initial state is unknown")
        if not self.p_kernels or len(self.p_kernels) != len(self.q_kernels):
            raise ValueError("P and Q must have the same positive horizon")
        if not self.failure_states.issubset(state_set):
            raise ValueError("unknown failure state")
        for state in self.states:
            next_states = self.transition.get(state)
            if next_states is None or len(next_states) != len(self.vocab):
                raise ValueError(f"transition row for {state!r} has wrong width")
            if any(next_state not in state_set for next_state in next_states):
                raise ValueError(f"transition row for {state!r} names unknown state")
        for label, kernels in (("P", self.p_kernels), ("Q", self.q_kernels)):
            for t, kernel in enumerate(kernels):
                if set(kernel) != state_set:
                    raise ValueError(f"{label}[{t}] does not cover every state")
                for state in self.states:
                    row = kernel[state]
                    if len(row) != len(self.vocab):
                        raise ValueError(f"{label}[{t},{state}] has wrong width")
                    _validate_distribution(row, f"{label}[{t},{state}]")


@dataclass(frozen=True)
class TrajectoryMetrics:
    p_failure: Probability
    q_failure: Probability
    p_first_failure: tuple[Probability, ...]
    q_first_failure: tuple[Probability, ...]
    p_terminal_state: Mapping[str, Probability]
    q_terminal_state: Mapping[str, Probability]
    trajectory_kl: float
    stopped_kl: float
    teacher_kl_by_step: tuple[float, ...]
    symbolic_trajectory_kl: SymbolicKL | None
    symbolic_stopped_kl: SymbolicKL | None

    def to_json_dict(self) -> dict[str, Any]:
        def encode_probability(value: Probability) -> float | str:
            if isinstance(value, Fraction):
                return f"{value.numerator}/{value.denominator}"
            return float(value)

        return {
            "p_failure": encode_probability(self.p_failure),
            "q_failure": encode_probability(self.q_failure),
            "p_first_failure": [encode_probability(v) for v in self.p_first_failure],
            "q_first_failure": [encode_probability(v) for v in self.q_first_failure],
            "p_terminal_state": {
                state: encode_probability(value)
                for state, value in self.p_terminal_state.items()
            },
            "q_terminal_state": {
                state: encode_probability(value)
                for state, value in self.q_terminal_state.items()
            },
            "trajectory_kl": self.trajectory_kl,
            "stopped_kl": self.stopped_kl,
            "teacher_kl_by_step": list(self.teacher_kl_by_step),
            "symbolic_trajectory_kl": (
                self.symbolic_trajectory_kl.to_json_dict()
                if self.symbolic_trajectory_kl is not None else None
            ),
            "symbolic_stopped_kl": (
                self.symbolic_stopped_kl.to_json_dict()
                if self.symbolic_stopped_kl is not None else None
            ),
        }


def _local_kl_float(p: Sequence[Probability], q: Sequence[Probability]) -> float:
    total = 0.0
    for pv, qv in zip(p, q):
        pf, qf = float(pv), float(qv)
        if pf == 0.0:
            continue
        if qf == 0.0:
            return math.inf
        total += pf * math.log(pf / qf)
    return max(0.0, total) if total > -1e-13 else total


def _add_local_symbolic(target: SymbolicKL, occupancy: Fraction,
                        p: Sequence[Probability], q: Sequence[Probability]) -> None:
    for pv, qv in zip(p, q):
        assert isinstance(pv, Fraction) and isinstance(qv, Fraction)
        if pv:
            target.add(occupancy * pv, pv, qv)


def _advance(
    model: FiniteStateAR,
    occupancy: Mapping[tuple[str, bool], Probability],
    kernel: Mapping[str, tuple[Probability, ...]],
) -> tuple[dict[tuple[str, bool], Probability], Probability]:
    sample = next(iter(occupancy.values()))
    zero = _zero_like(sample)
    next_occupancy: dict[tuple[str, bool], Probability] = {}
    first_failure = zero
    for (state, failed), mass in occupancy.items():
        if mass == 0:
            continue
        for token, probability in enumerate(kernel[state]):
            if probability == 0:
                continue
            next_state = model.transition[state][token]
            next_failed = failed or next_state in model.failure_states
            contribution = mass * probability
            key = (next_state, next_failed)
            next_occupancy[key] = next_occupancy.get(key, zero) + contribution
            if not failed and next_failed:
                first_failure += contribution
    return next_occupancy, first_failure


def forward_metrics(model: FiniteStateAR) -> TrajectoryMetrics:
    """Compute path KL and failure laws by exact finite-state dynamic programming."""

    model.validate()
    rational = model.exact_rational
    one: Probability = Fraction(1, 1) if rational else 1.0
    zero: Probability = Fraction(0, 1) if rational else 0.0
    p_occ: dict[tuple[str, bool], Probability] = {
        (model.initial_state, model.initial_state in model.failure_states): one
    }
    q_occ = dict(p_occ)
    p_first: list[Probability] = []
    q_first: list[Probability] = []
    kl_by_step: list[float] = []
    trajectory_kl = 0.0
    stopped_kl = 0.0
    symbolic = SymbolicKL.zero() if rational else None
    symbolic_stopped = SymbolicKL.zero() if rational else None

    for t in range(model.horizon):
        step_kl = 0.0
        for (state, failed), mass in p_occ.items():
            if mass == 0:
                continue
            local = _local_kl_float(model.p_kernels[t][state], model.q_kernels[t][state])
            if math.isinf(local):
                step_kl = math.inf
                trajectory_kl = math.inf
                if not failed:
                    stopped_kl = math.inf
            elif not math.isinf(trajectory_kl):
                contribution = float(mass) * local
                step_kl += contribution
                trajectory_kl += contribution
                if not failed:
                    stopped_kl += contribution
            if rational:
                assert isinstance(mass, Fraction)
                assert symbolic is not None and symbolic_stopped is not None
                _add_local_symbolic(
                    symbolic, mass, model.p_kernels[t][state], model.q_kernels[t][state]
                )
                if not failed:
                    _add_local_symbolic(
                        symbolic_stopped, mass,
                        model.p_kernels[t][state], model.q_kernels[t][state]
                    )
        kl_by_step.append(step_kl)
        p_occ, p_hit = _advance(model, p_occ, model.p_kernels[t])
        q_occ, q_hit = _advance(model, q_occ, model.q_kernels[t])
        p_first.append(p_hit)
        q_first.append(q_hit)

    def failure_mass(occupancy: Mapping[tuple[str, bool], Probability]) -> Probability:
        return sum((mass for (_, failed), mass in occupancy.items() if failed), zero)

    def terminal_state(occupancy: Mapping[tuple[str, bool], Probability]) -> dict[str, Probability]:
        result = {state: zero for state in model.states}
        for (state, _), mass in occupancy.items():
            result[state] += mass
        return result

    return TrajectoryMetrics(
        p_failure=failure_mass(p_occ),
        q_failure=failure_mass(q_occ),
        p_first_failure=tuple(p_first),
        q_first_failure=tuple(q_first),
        p_terminal_state=terminal_state(p_occ),
        q_terminal_state=terminal_state(q_occ),
        trajectory_kl=trajectory_kl,
        stopped_kl=stopped_kl,
        teacher_kl_by_step=tuple(kl_by_step),
        symbolic_trajectory_kl=symbolic,
        symbolic_stopped_kl=symbolic_stopped,
    )


@dataclass(frozen=True)
class EnumeratedLaws:
    p: Mapping[tuple[int, ...], Probability]
    q: Mapping[tuple[int, ...], Probability]
    failure: Mapping[tuple[int, ...], bool]


def enumerate_trajectory_laws(model: FiniteStateAR, *, max_paths: int = 1_000_000) -> EnumeratedLaws:
    """Materialize ``P^T`` and ``Q^T`` for small cases.

    Larger cases should use :func:`forward_metrics`, which has O(T |S| |V|)
    complexity rather than O(|V|^T).
    """

    model.validate()
    path_count = len(model.vocab) ** model.horizon
    if path_count > max_paths:
        raise ValueError(f"{path_count} paths exceed max_paths={max_paths}")
    rational = model.exact_rational
    one: Probability = Fraction(1, 1) if rational else 1.0
    p_law: dict[tuple[int, ...], Probability] = {}
    q_law: dict[tuple[int, ...], Probability] = {}
    failures: dict[tuple[int, ...], bool] = {}
    for tokens in product(range(len(model.vocab)), repeat=model.horizon):
        state = model.initial_state
        failed = state in model.failure_states
        p_mass: Probability = one
        q_mass: Probability = one
        for t, token in enumerate(tokens):
            p_mass *= model.p_kernels[t][state][token]
            q_mass *= model.q_kernels[t][state][token]
            state = model.transition[state][token]
            failed = failed or state in model.failure_states
        p_law[tokens] = p_mass
        q_law[tokens] = q_mass
        failures[tokens] = failed
    return EnumeratedLaws(p_law, q_law, failures)


def enumerate_stopped_laws(model: FiniteStateAR, *, max_paths: int = 1_000_000) -> EnumeratedLaws:
    """Aggregate full paths into variable-length transcripts stopped at failure."""

    full = enumerate_trajectory_laws(model, max_paths=max_paths)
    rational = model.exact_rational
    zero: Probability = Fraction(0, 1) if rational else 0.0
    p: dict[tuple[int, ...], Probability] = {}
    q: dict[tuple[int, ...], Probability] = {}
    failure: dict[tuple[int, ...], bool] = {}
    for tokens, p_mass in full.p.items():
        state = model.initial_state
        stopped: list[int] = []
        failed = state in model.failure_states
        if not failed:
            for token in tokens:
                stopped.append(token)
                state = model.transition[state][token]
                failed = failed or state in model.failure_states
                if failed:
                    break
        transcript = tuple(stopped)
        p[transcript] = p.get(transcript, zero) + p_mass
        q[transcript] = q.get(transcript, zero) + full.q[tokens]
        failure[transcript] = failed
    return EnumeratedLaws(p, q, failure)


def direct_kl(p: Mapping[Any, Probability], q: Mapping[Any, Probability]) -> float:
    total = 0.0
    for atom, probability in p.items():
        pf = float(probability)
        if pf == 0.0:
            continue
        qf = float(q.get(atom, 0.0))
        if qf == 0.0:
            return math.inf
        total += pf * math.log(pf / qf)
    return max(0.0, total) if total > -1e-13 else total


def binary_kl(p: float, q: float) -> float:
    """Bernoulli KL ``kl(p || q)`` with boundary conventions."""

    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("Bernoulli parameters must lie in [0,1]")
    if p == 0.0:
        return -math.log1p(-q) if q < 1.0 else math.inf
    if p == 1.0:
        return -math.log(q) if q > 0.0 else math.inf
    if q == 0.0 or q == 1.0:
        return math.inf
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def binary_kl_upper(p: float, d: float) -> float:
    """Sharp upper bound on q from ``kl(p || q) <= d``."""

    if not 0.0 <= p <= 1.0 or d < 0.0:
        raise ValueError("require p in [0,1] and d >= 0")
    if d == 0.0 or p == 1.0:
        return p
    if p == 0.0:
        return -math.expm1(-d)
    lower = p
    upper = math.nextafter(1.0, 0.0)
    return float(brentq(lambda q: binary_kl(p, q) - d, lower, upper,
                        xtol=1e-14, rtol=1e-14, maxiter=200))


def pinsker_upper(p: float, d: float) -> float:
    return min(1.0, p + math.sqrt(max(0.0, d) / 2.0))


def likelihood_cap_event_upper(p: float, cap: float) -> float:
    """Upper-bound ``Q(A)`` from ``P(A)=p`` and ``dQ/dP <= cap`` on A."""

    if not 0.0 <= p <= 1.0 or cap < 0.0:
        raise ValueError("require p in [0,1] and a non-negative cap")
    return min(1.0, cap * p)


def renyi_event_upper(p: float, divergence: float, order: float) -> float:
    """Stopped Hölder/Rényi event bound.

    If ``D_order(Q_tau || P_tau) <= divergence`` for ``order > 1``, then

    ``Q(A) <= exp((order-1)/order * divergence) * P(A)^((order-1)/order)``

    for every event measurable at the stopping time.
    """

    if not 0.0 <= p <= 1.0 or divergence < 0.0 or order <= 1.0:
        raise ValueError("require p in [0,1], divergence >= 0, and order > 1")
    exponent = (order - 1.0) / order
    if p == 0.0:
        return 0.0
    return min(1.0, math.exp(exponent * divergence) * p ** exponent)


def density_cap_trajectory_kl_bound(
    mean_local_kl_by_step: Sequence[float], density_caps: float | Sequence[float]
) -> float:
    """Transfer a corpus occupancy average to exact-policy occupancy.

    If ``d nu_t^P / d mu_t <= C_t``, then the trajectory KL is at most
    ``sum_t C_t E_mu_t[ell_t]``.
    """

    if isinstance(density_caps, (int, float)):
        caps = [float(density_caps)] * len(mean_local_kl_by_step)
    else:
        caps = [float(v) for v in density_caps]
    if len(caps) != len(mean_local_kl_by_step):
        raise ValueError("one density cap is required per time step")
    if any(c < 0.0 for c in caps) or any(v < 0.0 for v in mean_local_kl_by_step):
        raise ValueError("KL means and density caps must be non-negative")
    return math.fsum(c * v for c, v in zip(caps, mean_local_kl_by_step))


def ppl_reallocation_counterexample(*, retained_probability: float = 0.51,
                                    epsilon: float = 1e-6) -> dict[str, Any]:
    """Zero forced-token PPL delta with a large sampling-failure change.

    The forced token is token 0 under both laws.  P puts its remaining mass on
    a benign token, whereas Q moves nearly all of that mass to a failure token.
    Exact-token NLL, PPL, and top-1 therefore agree exactly even though the
    one-step free-run failure probabilities are far apart.
    """

    if not 0.5 < retained_probability < 1.0:
        raise ValueError("retained_probability must lie in (1/2,1)")
    remainder = 1.0 - retained_probability
    if not 0.0 < epsilon < remainder:
        raise ValueError("epsilon must lie in (0,1-retained_probability)")
    p = [retained_probability, remainder, 0.0]
    q = [retained_probability, epsilon, remainder - epsilon]
    return {
        "p": p,
        "q": q,
        "forced_token": 0,
        "p_failure": p[2],
        "q_failure": q[2],
        "ppl_ratio": 1.0,
        "top1_agrees": max(range(3), key=p.__getitem__) == max(
            range(3), key=q.__getitem__
        ),
        "kl_p_q": _local_kl_float(p, q),
    }


def rare_history_cascade(
    horizon: int,
    hazard: Probability = Fraction(1, 30),
    *,
    vocab_size: int = 2,
) -> FiniteStateAR:
    """A binary-KL-tight cascade with locally safe-looking exact histories.

    P always emits token 0 in ``safe``.  Q emits the same top-1 token with
    probability ``1-hazard`` and enters absorbing ``failure`` on token 1.
    Consequently P-failure is zero,

        D(P^T || Q^T) = -T log(1-hazard),
        Q(failure by T) = 1 - (1-hazard)^T,

    so the p=0 inverse binary-KL certificate is attained exactly.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 2 <= vocab_size <= 8:
        raise ValueError("vocab_size must lie in [2,8]")
    if not 0.0 < float(hazard) < 0.5:
        raise ValueError("hazard must lie in (0, 1/2) so top-1 still agrees")
    one: Probability = Fraction(1, 1) if isinstance(hazard, Fraction) else 1.0
    zero: Probability = Fraction(0, 1) if isinstance(hazard, Fraction) else 0.0
    safe_p = (one,) + (zero,) * (vocab_size - 1)
    safe_q = (one - hazard, hazard) + (zero,) * (vocab_size - 2)
    fail_row = (one,) + (zero,) * (vocab_size - 1)
    states = ("safe", "failure")
    transition = {
        "safe": ("safe", "failure") + ("safe",) * (vocab_size - 2),
        "failure": ("failure",) * vocab_size,
    }
    p_kernel = {"safe": safe_p, "failure": fail_row}
    q_kernel = {"safe": safe_q, "failure": fail_row}
    model = FiniteStateAR(
        vocab=tuple(f"v{i}" for i in range(vocab_size)),
        states=states,
        initial_state="safe",
        transition=transition,
        p_kernels=tuple(p_kernel for _ in range(horizon)),
        q_kernels=tuple(q_kernel for _ in range(horizon)),
        failure_states=frozenset({"failure"}),
    )
    model.validate()
    return model


def corpus_coverage_hole_model(
    candidate_safe_probability: Probability = Fraction(1, 100),
) -> FiniteStateAR:
    """A one-step finite-state example with zero logged-``mu`` KL.

    The returned deployment model starts in state ``deployment``.  A corpus
    that contains only state ``logged`` observes P=Q and hence local KL zero.
    On the actual deployment state, P is safe and Q fails with probability
    ``1-candidate_safe_probability``.  Thus no transfer is possible without a
    finite occupancy-density cap from deployment to the corpus law.
    """

    if not 0.0 < float(candidate_safe_probability) <= 1.0:
        raise ValueError("candidate_safe_probability must lie in (0,1]")
    rational = isinstance(candidate_safe_probability, Fraction)
    one: Probability = Fraction(1) if rational else 1.0
    zero: Probability = Fraction(0) if rational else 0.0
    states = ("logged", "deployment", "safe", "failure")
    transition = {
        "logged": ("safe", "failure"),
        "deployment": ("safe", "failure"),
        "safe": ("safe", "safe"),
        "failure": ("failure", "failure"),
    }
    p_kernel = {
        "logged": (one, zero),
        "deployment": (one, zero),
        "safe": (one, zero),
        "failure": (one, zero),
    }
    q_kernel = {
        "logged": (one, zero),
        "deployment": (
            candidate_safe_probability,
            one - candidate_safe_probability,
        ),
        "safe": (one, zero),
        "failure": (one, zero),
    }
    model = FiniteStateAR(
        vocab=("safe", "fail"),
        states=states,
        initial_state="deployment",
        transition=transition,
        p_kernels=(p_kernel,),
        q_kernels=(q_kernel,),
        failure_states=frozenset({"failure"}),
    )
    model.validate()
    return model


def random_finite_state_adversary(
    *,
    horizon: int,
    vocab_size: int,
    state_count: int,
    rng: random.Random,
    attack_strength: Fraction,
) -> FiniteStateAR:
    """Generate an exact-rational finite-state pair for adversarial search."""

    if not 2 <= vocab_size <= 8:
        raise ValueError("vocab_size must lie in [2,8]")
    if state_count < 2:
        raise ValueError("state_count must be at least 2")
    if not 0 <= attack_strength <= 1:
        raise ValueError("attack_strength must lie in [0,1]")
    states = tuple([f"s{i}" for i in range(state_count - 1)] + ["failure"])
    failure = "failure"
    transition: dict[str, tuple[str, ...]] = {}
    for state in states[:-1]:
        row = []
        for token in range(vocab_size):
            if token == vocab_size - 1:
                row.append(failure)
            else:
                row.append(states[rng.randrange(state_count - 1)])
        transition[state] = tuple(row)
    transition[failure] = (failure,) * vocab_size

    p_kernels: list[dict[str, tuple[Probability, ...]]] = []
    q_kernels: list[dict[str, tuple[Probability, ...]]] = []
    for _ in range(horizon):
        p_kernel: dict[str, tuple[Probability, ...]] = {}
        q_kernel: dict[str, tuple[Probability, ...]] = {}
        for state in states:
            if state == failure:
                p_row = (Fraction(1),) + (Fraction(0),) * (vocab_size - 1)
                q_row = p_row
            else:
                weights = [rng.randint(8, 40) for _ in range(vocab_size - 1)]
                # Most exact states have no direct failure edge; a few have a
                # small one.  This makes the search target failure introduced
                # by Q rather than merely selecting a high-failure P baseline.
                weights.append(rng.choice([0, 0, 0, 0, 1]))
                total = sum(weights)
                p_row = tuple(Fraction(w, total) for w in weights)
                attack = tuple(
                    Fraction(0) if token < vocab_size - 1 else Fraction(1)
                    for token in range(vocab_size)
                )
                q_row = tuple(
                    (1 - attack_strength) * pv + attack_strength * av
                    for pv, av in zip(p_row, attack)
                )
            p_kernel[state] = p_row
            q_kernel[state] = q_row
        p_kernels.append(p_kernel)
        q_kernels.append(q_kernel)
    model = FiniteStateAR(
        vocab=tuple(f"v{i}" for i in range(vocab_size)),
        states=states,
        initial_state=states[0],
        transition=transition,
        p_kernels=tuple(p_kernels),
        q_kernels=tuple(q_kernels),
        failure_states=frozenset({failure}),
    )
    model.validate()
    return model


@dataclass(frozen=True)
class AdversarySearchResult:
    trial: int
    model_seed: int
    attack_numerator: int
    attack_denominator: int
    trajectory_kl: float
    p_failure: float
    q_failure: float
    failure_increase: float
    binary_upper: float


def search_high_failure_pairs(
    *,
    budget: float,
    horizon: int = 12,
    vocab_size: int = 3,
    state_count: int = 4,
    trials: int = 250,
    strength_grid: int = 64,
    seed: int = 7,
    keep: int = 10,
) -> list[AdversarySearchResult]:
    """Search rational finite-state pairs under a P-occupancy KL budget."""

    if budget < 0.0 or trials <= 0 or strength_grid <= 0:
        raise ValueError("invalid search parameters")
    rng = random.Random(seed)
    results: list[AdversarySearchResult] = []
    for trial in range(trials):
        best: AdversarySearchResult | None = None
        # Reuse a seed so only attack strength changes within a trial.
        model_seed = rng.randrange(1 << 62)
        for numerator in range(strength_grid + 1):
            strength = Fraction(numerator, strength_grid)
            model_rng = random.Random(model_seed)
            model = random_finite_state_adversary(
                horizon=horizon,
                vocab_size=vocab_size,
                state_count=state_count,
                rng=model_rng,
                attack_strength=strength,
            )
            metrics = forward_metrics(model)
            if metrics.trajectory_kl > budget + 1e-12:
                break
            p_failure = float(metrics.p_failure)
            q_failure = float(metrics.q_failure)
            candidate = AdversarySearchResult(
                trial=trial,
                model_seed=model_seed,
                attack_numerator=numerator,
                attack_denominator=strength_grid,
                trajectory_kl=metrics.trajectory_kl,
                p_failure=p_failure,
                q_failure=q_failure,
                failure_increase=q_failure - p_failure,
                binary_upper=binary_kl_upper(p_failure, metrics.trajectory_kl),
            )
            if best is None or (candidate.failure_increase, candidate.q_failure) > (
                    best.failure_increase, best.q_failure):
                best = candidate
        if best is not None:
            results.append(best)
    results.sort(key=lambda item: (item.failure_increase, item.q_failure), reverse=True)
    return results[:keep]


def greedy_pairwise_slacks(
    exact_logits: Sequence[Sequence[float]],
    candidate_logits: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Return the weakest pairwise greedy slack on each exact-prefix step.

    For unique exact argmax ``i``, the candidate token is identical iff every
    ``(zP_i-zP_j) - ((zQ_j-zP_j)-(zQ_i-zP_i))`` is positive.  The returned
    value is the minimum over competitors.
    """

    if len(exact_logits) != len(candidate_logits):
        raise ValueError("exact and candidate must have the same step count")
    slacks: list[float] = []
    for step, (zp, zq) in enumerate(zip(exact_logits, candidate_logits)):
        if len(zp) != len(zq) or len(zp) < 2:
            raise ValueError(f"step {step}: incompatible logits")
        top = max(range(len(zp)), key=lambda j: zp[j])
        sorted_values = sorted((float(v) for v in zp), reverse=True)
        if not sorted_values[0] > sorted_values[1]:
            raise ValueError(f"step {step}: exact top logit is tied")
        e_top = float(zq[top]) - float(zp[top])
        minimum = math.inf
        for j in range(len(zp)):
            if j == top:
                continue
            margin = float(zp[top]) - float(zp[j])
            error_difference = (float(zq[j]) - float(zp[j])) - e_top
            minimum = min(minimum, margin - error_difference)
        slacks.append(minimum)
    return tuple(slacks)


def greedy_tokens_identical(
    exact_logits: Sequence[Sequence[float]],
    candidate_logits: Sequence[Sequence[float]],
) -> bool:
    return all(slack > 0.0 for slack in greedy_pairwise_slacks(exact_logits, candidate_logits))


def cosine(a: Sequence[float], b: Sequence[float], *, centered: bool = False) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("vectors must be non-empty and have equal length")
    av = [float(x) for x in a]
    bv = [float(x) for x in b]
    if centered:
        am = math.fsum(av) / len(av)
        bm = math.fsum(bv) / len(bv)
        av = [x - am for x in av]
        bv = [x - bm for x in bv]
    dot = math.fsum(x * y for x, y in zip(av, bv))
    na = math.sqrt(math.fsum(x * x for x in av))
    nb = math.sqrt(math.fsum(y * y for y in bv))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def mse(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("vectors must be non-empty and have equal length")
    return math.fsum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) / len(a)


def greedy_metric_counterexample(*, margin: float = 1e-6,
                                 background: float = 1e6) -> dict[str, Any]:
    """Near-perfect raw/centered cosine and tiny MSE with an argmax flip."""

    if margin <= 0.0 or background <= 0.0:
        raise ValueError("margin and background must be positive")
    exact = [0.0, -margin, -background]
    candidate = [-margin, 0.0, -background]
    return {
        "exact": exact,
        "candidate": candidate,
        "exact_top1": 0,
        "candidate_top1": 1,
        "margin": margin,
        "raw_cosine": cosine(exact, candidate),
        "centered_cosine": cosine(exact, candidate, centered=True),
        "mse": mse(exact, candidate),
        "pairwise_slack": greedy_pairwise_slacks([exact], [candidate])[0],
    }


def beta_binomial_log_evalue(
    successes: int,
    trials: int,
    null_mean: float,
    *,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> float:
    """Log beta-binomial mixture likelihood ratio for a Bernoulli null."""

    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0,trials]")
    if prior_a <= 0.0 or prior_b <= 0.0:
        raise ValueError("beta prior parameters must be positive")
    if null_mean < 0.0 or null_mean > 1.0:
        raise ValueError("null_mean must lie in [0,1]")
    numerator = float(betaln(successes + prior_a,
                             trials - successes + prior_b)
                      - betaln(prior_a, prior_b))
    if successes:
        if null_mean == 0.0:
            return math.inf
        numerator -= successes * math.log(null_mean)
    if trials - successes:
        if null_mean == 1.0:
            return math.inf
        numerator -= (trials - successes) * math.log1p(-null_mean)
    return numerator


@lru_cache(maxsize=200_000)
def _cached_bernoulli_mixture_upper(
    successes: int,
    trials: int,
    alpha: float,
    prior_a: float,
    prior_b: float,
) -> float:
    if trials == 0 or successes == trials:
        return 1.0
    threshold = math.log(1.0 / alpha)
    empirical = successes / trials

    def equation(q: float) -> float:
        return beta_binomial_log_evalue(
            successes, trials, q, prior_a=prior_a, prior_b=prior_b
        ) - threshold

    lower = max(empirical, 0.0)
    upper = math.nextafter(1.0, 0.0)
    if equation(lower) >= 0.0:
        # This can happen only through extreme floating-point roundoff at the
        # MLE; returning the empirical mean remains conservative.
        return lower
    return float(brentq(equation, lower, upper, xtol=1e-13, rtol=1e-13,
                        maxiter=200))


def bernoulli_mixture_cs_upper(
    successes: int,
    trials: int,
    *,
    alpha: float = 0.05,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> float:
    """Anytime-valid Bernoulli upper confidence sequence endpoint."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    return _cached_bernoulli_mixture_upper(
        int(successes), int(trials), float(alpha), float(prior_a), float(prior_b)
    )


def anytime_importance_ucb(
    weighted_failures: Sequence[float],
    weight_caps: Sequence[float],
    *,
    alpha: float = 0.05,
) -> float:
    """Alpha-spending Hoeffding UCB for predictable bounded importance data."""

    if len(weighted_failures) != len(weight_caps):
        raise ValueError("weighted observations and caps must have equal length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    n = len(weighted_failures)
    if n == 0:
        return 1.0
    for value, cap in zip(weighted_failures, weight_caps):
        if cap <= 0.0 or value < -_EPS or value > cap + _EPS:
            raise ValueError("each weighted observation must lie in [0,cap]")
    spending = 6.0 * alpha / (math.pi * math.pi * n * n)
    radius = math.sqrt(
        math.fsum(cap * cap for cap in weight_caps)
        * math.log(1.0 / spending) / (2.0 * n * n)
    )
    return min(1.0, math.fsum(weighted_failures) / n + radius)


@dataclass(frozen=True)
class MixtureRecord:
    source: str
    failure: int
    log_p: float
    log_q: float
    log_intervention: float
    exact_probability: float
    candidate_probability: float
    intervention_probability: float

    def validate(self) -> None:
        if self.source not in {"P", "Q", "I"}:
            raise ValueError("source must be P, Q, or I")
        if self.failure not in {0, 1}:
            raise ValueError("failure must be binary")
        weights = [self.exact_probability, self.candidate_probability,
                   self.intervention_probability]
        if any(weight < 0.0 for weight in weights):
            raise ValueError("mixture probabilities must be non-negative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-10):
            raise ValueError("mixture probabilities must sum to one")
        if self.candidate_probability <= 0.0:
            raise ValueError("candidate mixture probability must be positive")


class SequentialMixtureCertifier:
    """Certify Q failure from an adaptive P/Q/intervention trajectory mixture."""

    def __init__(self, *, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0,1)")
        self.alpha = alpha
        self.records: list[MixtureRecord] = []
        self.q_failures = 0
        self.q_trials = 0
        self.weighted_failures: list[float] = []
        self.weight_caps: list[float] = []

    def add(self, record: MixtureRecord) -> None:
        record.validate()
        mixture_log_density = float(logsumexp([
            math.log(record.exact_probability) + record.log_p
            if record.exact_probability > 0.0 else -math.inf,
            math.log(record.candidate_probability) + record.log_q,
            math.log(record.intervention_probability) + record.log_intervention
            if record.intervention_probability > 0.0 else -math.inf,
        ]))
        weight = math.exp(record.log_q - mixture_log_density)
        cap = 1.0 / record.candidate_probability
        if weight > cap * (1.0 + 1e-10):
            raise ArithmeticError("mixture importance weight exceeds 1/beta cap")
        self.records.append(record)
        self.weighted_failures.append(weight * record.failure)
        self.weight_caps.append(cap)
        if record.source == "Q":
            self.q_trials += 1
            self.q_failures += record.failure

    def bounds(self) -> dict[str, float | int]:
        direct = bernoulli_mixture_cs_upper(
            self.q_failures, self.q_trials, alpha=self.alpha / 2.0
        )
        importance = anytime_importance_ucb(
            self.weighted_failures, self.weight_caps, alpha=self.alpha / 2.0
        )
        return {
            "records": len(self.records),
            "candidate_records": self.q_trials,
            "candidate_failures": self.q_failures,
            "direct_candidate_ucb": direct,
            "importance_mixture_ucb": importance,
            "combined_ucb": min(direct, importance),
        }


def simultaneous_policy_ucb(
    weighted_failures: Sequence[float],
    weight_caps: Sequence[float],
    *,
    alpha: float,
    prior_weight: float,
) -> float:
    """Anytime UCB valid for one member of a countable policy class.

    Calling this with per-policy prior masses summing to at most one yields a
    simultaneous certificate for every policy and every sample size; a policy
    selected from the same log may then use its own endpoint.
    """

    if not 0.0 < prior_weight <= 1.0:
        raise ValueError("prior_weight must lie in (0,1]")
    return anytime_importance_ucb(
        weighted_failures, weight_caps, alpha=alpha * prior_weight
    )


@dataclass(frozen=True)
class CoverageResult:
    true_failure: float
    alpha: float
    max_samples: int
    trials: int
    violations: int
    coverage: float
    stopped_early: int


def simulate_anytime_coverage(
    *,
    true_failure: float,
    alpha: float = 0.05,
    max_samples: int = 250,
    trials: int = 2_000,
    certify_threshold: float | None = None,
    seed: int = 1,
) -> CoverageResult:
    """Monte Carlo check of beta-mixture CS coverage under adaptive stopping."""

    if not 0.0 <= true_failure <= 1.0:
        raise ValueError("true_failure must lie in [0,1]")
    rng = random.Random(seed)
    violations = 0
    stopped_early = 0
    for _ in range(trials):
        successes = 0
        violated = False
        for n in range(1, max_samples + 1):
            successes += int(rng.random() < true_failure)
            upper = bernoulli_mixture_cs_upper(successes, n, alpha=alpha)
            if true_failure > upper + 1e-12:
                violated = True
                break
            if certify_threshold is not None and upper <= certify_threshold:
                stopped_early += 1
                break
        if violated:
            violations += 1
    return CoverageResult(
        true_failure=true_failure,
        alpha=alpha,
        max_samples=max_samples,
        trials=trials,
        violations=violations,
        coverage=1.0 - violations / trials,
        stopped_early=stopped_early,
    )


def _fraction(text: str) -> Fraction:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _write_search_csv(path: Path, rows: Iterable[AdversarySearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(AdversarySearchResult.__annotations__),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cascade_parser = subparsers.add_parser("cascade", help="evaluate the tight cascade")
    cascade_parser.add_argument("--horizon", type=int, default=50)
    cascade_parser.add_argument("--hazard", type=_fraction, default=Fraction(1, 30))
    cascade_parser.add_argument("--vocab", type=int, default=2)

    search_parser = subparsers.add_parser("search", help="search finite-state adversaries")
    search_parser.add_argument("--budget", type=float, default=0.5)
    search_parser.add_argument("--horizon", type=int, default=12)
    search_parser.add_argument("--vocab", type=int, default=3)
    search_parser.add_argument("--states", type=int, default=4)
    search_parser.add_argument("--trials", type=int, default=250)
    search_parser.add_argument("--output", type=Path, default=None)

    coverage_parser = subparsers.add_parser("coverage", help="test anytime CS coverage")
    coverage_parser.add_argument("--failure", type=float, default=0.02)
    coverage_parser.add_argument("--alpha", type=float, default=0.05)
    coverage_parser.add_argument("--samples", type=int, default=250)
    coverage_parser.add_argument("--trials", type=int, default=2_000)
    coverage_parser.add_argument("--threshold", type=float, default=None)
    coverage_parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    if args.command == "cascade":
        model = rare_history_cascade(args.horizon, args.hazard, vocab_size=args.vocab)
        metrics = forward_metrics(model)
        payload = metrics.to_json_dict()
        payload.update({
            "horizon": args.horizon,
            "hazard": str(args.hazard),
            "per_step_ppl_ratio": float(1 / (1 - args.hazard)),
            "binary_kl_upper": binary_kl_upper(
                float(metrics.p_failure), metrics.trajectory_kl
            ),
            "pinsker_upper": pinsker_upper(
                float(metrics.p_failure), metrics.trajectory_kl
            ),
        })
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "search":
        rows = search_high_failure_pairs(
            budget=args.budget,
            horizon=args.horizon,
            vocab_size=args.vocab,
            state_count=args.states,
            trials=args.trials,
        )
        if args.output is not None:
            _write_search_csv(args.output, rows)
        print(json.dumps([asdict(row) for row in rows], indent=2))
        return 0
    if args.command == "coverage":
        result = simulate_anytime_coverage(
            true_failure=args.failure,
            alpha=args.alpha,
            max_samples=args.samples,
            trials=args.trials,
            certify_threshold=args.threshold,
            seed=args.seed,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
