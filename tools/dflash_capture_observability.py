#!/usr/bin/env python3
"""CPU reference solvers for DFlash target-layer capture selection.

The module has two exact information oracles:

* ``DiscreteLayeredBayesNet`` enumerates a finite joint law of (D, Y, Z_1,...).
* ``ConditionalGaussianInformation`` evaluates I(Y; Z_S | D) by log determinants.

Selection helpers include exact subset enumeration, singleton greedy, cost-aware
singleton greedy, q-bundle density greedy, and a Dinkelbach wrapper for renewal
throughput.  Nothing in this file assumes access to model weights or private
activation traces.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

_EPS = 1e-12
_CURRENT_CAPTURE_LAYERS = (5, 14, 24, 33, 42)

Subset = tuple[int, ...]
ValueFunction = Callable[[Subset], float]


def canonical_subset(items: Iterable[int]) -> Subset:
    return tuple(sorted(set(int(x) for x in items)))


def powerset(items: Sequence[int], max_size: int | None = None) -> Iterable[Subset]:
    ordered = tuple(items)
    limit = len(ordered) if max_size is None else min(max_size, len(ordered))
    for size in range(limit + 1):
        yield from itertools.combinations(ordered, size)


def _entropy_from_masses(masses: Iterable[float]) -> float:
    result = 0.0
    for mass in masses:
        if mass > 0:
            result -= mass * math.log2(mass)
    return result


def _longest_common_prefix(a: Sequence[Any], b: Sequence[Any]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


@dataclass(frozen=True)
class JointOutcome:
    probability: float
    d: Hashable
    y: Hashable
    observations: tuple[Hashable, ...]


class DiscreteLayeredBayesNet:
    """Finite joint distribution for exact conditional information and Bayes risk."""

    def __init__(self, layers: Sequence[int], outcomes: Sequence[JointOutcome]) -> None:
        self.layers = canonical_subset(layers)
        if len(self.layers) != len(tuple(layers)):
            raise ValueError("layers must be unique")
        self._position = {layer: i for i, layer in enumerate(self.layers)}
        self.outcomes = tuple(outcomes)
        if not self.outcomes:
            raise ValueError("joint law is empty")
        total = 0.0
        for row in self.outcomes:
            if row.probability < -_EPS:
                raise ValueError("negative probability")
            if len(row.observations) != len(self.layers):
                raise ValueError("observation width does not match layers")
            total += row.probability
        if not math.isclose(total, 1.0, abs_tol=1e-10):
            raise ValueError(f"joint probabilities sum to {total}")

    def _positions(self, selected: Iterable[int]) -> tuple[int, ...]:
        result = []
        for layer in canonical_subset(selected):
            if layer not in self._position:
                raise KeyError(f"unknown layer {layer}")
            result.append(self._position[layer])
        return tuple(result)

    def _z_key(self, row: JointOutcome, positions: Sequence[int]) -> tuple[Hashable, ...]:
        return tuple(row.observations[i] for i in positions)

    @lru_cache(maxsize=None)
    def mutual_information(self, selected: Subset) -> float:
        """Return I(Y; Z_selected | D) in bits."""
        positions = self._positions(selected)
        p_d: dict[Hashable, float] = defaultdict(float)
        p_dy: dict[tuple[Hashable, Hashable], float] = defaultdict(float)
        p_dz: dict[tuple[Hashable, tuple[Hashable, ...]], float] = defaultdict(float)
        p_dyz: dict[tuple[Hashable, Hashable, tuple[Hashable, ...]], float] = defaultdict(float)
        for row in self.outcomes:
            if row.probability <= 0:
                continue
            z = self._z_key(row, positions)
            p_d[row.d] += row.probability
            p_dy[(row.d, row.y)] += row.probability
            p_dz[(row.d, z)] += row.probability
            p_dyz[(row.d, row.y, z)] += row.probability
        result = 0.0
        for (d, y, z), mass in p_dyz.items():
            numerator = mass * p_d[d]
            denominator = p_dy[(d, y)] * p_dz[(d, z)]
            if mass > 0 and numerator > 0 and denominator > 0:
                result += mass * math.log2(numerator / denominator)
        return max(0.0, result) if result > -1e-10 else result

    @lru_cache(maxsize=None)
    def conditional_entropy_y(self, selected: Subset) -> float:
        positions = self._positions(selected)
        group_total: dict[tuple[Hashable, tuple[Hashable, ...]], float] = defaultdict(float)
        group_y: dict[tuple[Hashable, tuple[Hashable, ...], Hashable], float] = defaultdict(float)
        for row in self.outcomes:
            z = self._z_key(row, positions)
            group_total[(row.d, z)] += row.probability
            group_y[(row.d, z, row.y)] += row.probability
        result = 0.0
        for (d, z, _), mass in group_y.items():
            total = group_total[(d, z)]
            if mass > 0 and total > 0:
                result -= mass * math.log2(mass / total)
        return result

    @lru_cache(maxsize=None)
    def bayes_block_error(self, selected: Subset) -> float:
        positions = self._positions(selected)
        posterior_mass: dict[
            tuple[Hashable, tuple[Hashable, ...], Hashable], float
        ] = defaultdict(float)
        groups: set[tuple[Hashable, tuple[Hashable, ...]]] = set()
        for row in self.outcomes:
            z = self._z_key(row, positions)
            groups.add((row.d, z))
            posterior_mass[(row.d, z, row.y)] += row.probability
        success = 0.0
        for d, z in groups:
            success += max(
                mass for (gd, gz, _), mass in posterior_mass.items()
                if gd == d and gz == z
            )
        return min(1.0, max(0.0, 1.0 - success))

    @lru_cache(maxsize=None)
    def bayes_expected_prefix(self, selected: Subset) -> float:
        """Bayes-optimal expected exact-prefix length for sequence-valued Y."""
        positions = self._positions(selected)
        group_rows: dict[
            tuple[Hashable, tuple[Hashable, ...]],
            dict[tuple[Any, ...], float],
        ] = defaultdict(lambda: defaultdict(float))
        for row in self.outcomes:
            if isinstance(row.y, tuple):
                y = row.y
            elif isinstance(row.y, list):
                y = tuple(row.y)
            else:
                y = (row.y,)
            z = self._z_key(row, positions)
            group_rows[(row.d, z)][y] += row.probability
        expected = 0.0
        for distribution in group_rows.values():
            candidates = tuple(distribution)
            best = 0.0
            for prediction in candidates:
                score = sum(
                    mass * _longest_common_prefix(prediction, truth)
                    for truth, mass in distribution.items()
                )
                best = max(best, score)
            expected += best
        return expected

    def submodularity_gap(self, a: Iterable[int], b: Iterable[int], element: int) -> float:
        """Return Delta(element|a) - Delta(element|b)."""
        a_set, b_set = set(a), set(b)
        if not a_set.issubset(b_set) or element in b_set:
            raise ValueError("require a subset b and element outside b")
        ca, cb = canonical_subset(a_set), canonical_subset(b_set)
        return (
            self.mutual_information(canonical_subset((*ca, element)))
            - self.mutual_information(ca)
            - self.mutual_information(canonical_subset((*cb, element)))
            + self.mutual_information(cb)
        )


@dataclass(frozen=True)
class GaussianRegime:
    name: str
    probability: float
    loading: np.ndarray
    noise_covariance: np.ndarray
    target_covariance: np.ndarray

    def validate(self, n_layers: int) -> None:
        if self.probability < 0:
            raise ValueError("negative regime probability")
        if self.loading.ndim != 2 or self.loading.shape[0] != n_layers:
            raise ValueError("loading has wrong shape")
        target_dim = self.loading.shape[1]
        if self.noise_covariance.shape != (n_layers, n_layers):
            raise ValueError("noise covariance has wrong shape")
        if self.target_covariance.shape != (target_dim, target_dim):
            raise ValueError("target covariance has wrong shape")
        if np.linalg.eigvalsh(self.noise_covariance).min() <= 0:
            raise ValueError("noise covariance must be positive definite")
        if np.linalg.eigvalsh(self.target_covariance).min() <= 0:
            raise ValueError("target covariance must be positive definite")


class ConditionalGaussianInformation:
    """Closed-form I(Y; Z_S | D) for a finite observed regime D."""

    def __init__(self, layers: Sequence[int], regimes: Sequence[GaussianRegime]) -> None:
        self.layers = canonical_subset(layers)
        self._position = {layer: i for i, layer in enumerate(self.layers)}
        self.regimes = tuple(regimes)
        if not self.regimes:
            raise ValueError("at least one regime is required")
        total = sum(r.probability for r in self.regimes)
        if not math.isclose(total, 1.0, abs_tol=1e-10):
            raise ValueError(f"regime probabilities sum to {total}")
        for regime in self.regimes:
            regime.validate(len(self.layers))

    def _indices(self, selected: Iterable[int]) -> np.ndarray:
        result = []
        for layer in canonical_subset(selected):
            if layer not in self._position:
                raise KeyError(f"unknown layer {layer}")
            result.append(self._position[layer])
        return np.asarray(result, dtype=int)

    @staticmethod
    def _regime_mi(regime: GaussianRegime, indices: np.ndarray) -> float:
        if indices.size == 0:
            return 0.0
        loading = regime.loading[indices, :]
        noise = regime.noise_covariance[np.ix_(indices, indices)]
        total = noise + loading @ regime.target_covariance @ loading.T
        sign_total, log_total = np.linalg.slogdet(total)
        sign_noise, log_noise = np.linalg.slogdet(noise)
        if sign_total <= 0 or sign_noise <= 0:
            raise ValueError("selected covariance is not positive definite")
        return 0.5 * (log_total - log_noise) / math.log(2.0)

    @lru_cache(maxsize=None)
    def mutual_information(self, selected: Subset) -> float:
        indices = self._indices(selected)
        return sum(
            regime.probability * self._regime_mi(regime, indices)
            for regime in self.regimes
        )

    @lru_cache(maxsize=None)
    def regime_information(self, regime_name: str, selected: Subset) -> float:
        indices = self._indices(selected)
        for regime in self.regimes:
            if regime.name == regime_name:
                return self._regime_mi(regime, indices)
        raise KeyError(regime_name)


@dataclass(frozen=True)
class CaptureCost:
    bytes: float
    latency_ms: float


@dataclass(frozen=True)
class SelectionConstraints:
    max_items: int
    max_bytes: float | None = None
    max_latency_ms: float | None = None
    max_effective_ms: float | None = None

    def validate(self) -> None:
        if self.max_items < 0:
            raise ValueError("max_items must be non-negative")
        for value in (self.max_bytes, self.max_latency_ms, self.max_effective_ms):
            if value is not None and value < 0:
                raise ValueError("budgets must be non-negative")


@dataclass(frozen=True)
class SelectionResult:
    method: str
    selected: Subset
    objective_value: float
    evaluations: int
    feasible: bool


@dataclass(frozen=True)
class DinkelbachResult:
    selected: Subset
    accepted_tokens: float
    round_time_ms: float
    throughput_tokens_per_ms: float
    transformed_residual: float
    iterations: int
    inner_evaluations: int


@dataclass(frozen=True)
class AdaptiveDinkelbachResult:
    policy: Mapping[Hashable, Subset]
    expected_accepted_tokens: float
    expected_round_time_ms: float
    throughput_tokens_per_ms: float
    transformed_residual: float
    iterations: int
    inner_evaluations: int


@dataclass(frozen=True)
class CaptureProblem:
    name: str
    layers: Subset
    costs: Mapping[int, CaptureCost]
    constraints: SelectionConstraints
    information: ValueFunction
    accepted_tokens: ValueFunction
    round_time_ms: ValueFunction
    bandwidth_bytes_per_ms: float
    current_layers: Subset = _CURRENT_CAPTURE_LAYERS

    def __post_init__(self) -> None:
        self.constraints.validate()
        if self.bandwidth_bytes_per_ms <= 0:
            raise ValueError("bandwidth must be positive")
        if set(self.costs) != set(self.layers):
            raise ValueError("costs must cover every candidate layer")
        for cost in self.costs.values():
            if cost.bytes < 0 or cost.latency_ms < 0:
                raise ValueError("negative capture cost")

    def effective_cost(self, layer: int) -> float:
        cost = self.costs[layer]
        return cost.latency_ms + cost.bytes / self.bandwidth_bytes_per_ms

    def selected_costs(self, selected: Iterable[int]) -> tuple[float, float, float]:
        subset = canonical_subset(selected)
        byte_cost = sum(self.costs[layer].bytes for layer in subset)
        latency = sum(self.costs[layer].latency_ms for layer in subset)
        effective = sum(self.effective_cost(layer) for layer in subset)
        return byte_cost, latency, effective

    def feasible(self, selected: Iterable[int]) -> bool:
        subset = canonical_subset(selected)
        if not set(subset).issubset(self.layers):
            return False
        if len(subset) > self.constraints.max_items:
            return False
        byte_cost, latency, effective = self.selected_costs(subset)
        if self.constraints.max_bytes is not None and byte_cost > self.constraints.max_bytes + _EPS:
            return False
        if (self.constraints.max_latency_ms is not None and
                latency > self.constraints.max_latency_ms + _EPS):
            return False
        if (self.constraints.max_effective_ms is not None and
                effective > self.constraints.max_effective_ms + _EPS):
            return False
        return True

    def current_subset(self) -> Subset:
        return canonical_subset(layer for layer in self.current_layers if layer in self.layers)


def exact_select(
    problem: CaptureProblem,
    value: ValueFunction,
    *,
    method: str = "true_optimum",
) -> SelectionResult:
    best_set: Subset = ()
    best_value = value(())
    evaluations = 1
    for size in range(1, problem.constraints.max_items + 1):
        for subset in itertools.combinations(problem.layers, size):
            if not problem.feasible(subset):
                continue
            score = value(subset)
            evaluations += 1
            if score > best_value + _EPS or (
                math.isclose(score, best_value, abs_tol=_EPS) and subset < best_set
            ):
                best_set, best_value = subset, score
    return SelectionResult(method, best_set, best_value, evaluations, True)


def greedy_select(
    problem: CaptureProblem,
    value: ValueFunction,
    *,
    density: bool = False,
    method: str | None = None,
    fill_zero_gain: bool = False,
) -> SelectionResult:
    selected: Subset = ()
    current_value = value(selected)
    evaluations = 1
    while len(selected) < problem.constraints.max_items:
        candidate: tuple[float, float, int, float] | None = None
        selected_set = set(selected)
        for layer in problem.layers:
            if layer in selected_set:
                continue
            trial = canonical_subset((*selected, layer))
            if not problem.feasible(trial):
                continue
            trial_value = value(trial)
            evaluations += 1
            gain = trial_value - current_value
            denominator = problem.effective_cost(layer) if density else 1.0
            score = gain / max(denominator, _EPS)
            key = (score, gain, -layer, trial_value)
            if candidate is None or key > candidate:
                candidate = key
        if candidate is None:
            break
        score, gain, neg_layer, trial_value = candidate
        if gain <= _EPS and not fill_zero_gain:
            break
        layer = -int(neg_layer)
        selected = canonical_subset((*selected, layer))
        current_value = trial_value
    name = method or ("cost_aware_greedy" if density else "greedy_information")
    return SelectionResult(name, selected, current_value, evaluations, problem.feasible(selected))


def _candidate_bundles(remaining: Sequence[int], max_bundle_size: int) -> Iterable[Subset]:
    for size in range(1, min(max_bundle_size, len(remaining)) + 1):
        yield from itertools.combinations(remaining, size)


def bundle_density_greedy(
    problem: CaptureProblem,
    value: ValueFunction,
    *,
    max_bundle_size: int = 2,
    method: str = "pair_bundle_greedy",
    polish: bool = True,
) -> SelectionResult:
    """Greedily add the feasible bundle with highest marginal value per cost.

    Bundles expose low-order interactions such as XOR.  The optional polishing
    phase performs deterministic improving replacements of up to q selected
    layers by up to q unselected layers.  Since polishing only accepts strict
    improvements, any guarantee for the greedy prefix is preserved.
    """
    if max_bundle_size < 1:
        raise ValueError("max_bundle_size must be positive")
    selected: Subset = ()
    current_value = value(selected)
    evaluations = 1
    while len(selected) < problem.constraints.max_items:
        remaining = tuple(layer for layer in problem.layers if layer not in selected)
        best: tuple[float, float, tuple[int, ...], float] | None = None
        for bundle in _candidate_bundles(remaining, max_bundle_size):
            if len(selected) + len(bundle) > problem.constraints.max_items:
                continue
            trial = canonical_subset((*selected, *bundle))
            if not problem.feasible(trial):
                continue
            trial_value = value(trial)
            evaluations += 1
            gain = trial_value - current_value
            cost = sum(problem.effective_cost(layer) for layer in bundle)
            score = gain / max(cost, _EPS)
            key = (score, gain, tuple(-x for x in bundle), trial_value)
            if best is None or key > best:
                best = key
        if best is None or best[1] <= _EPS:
            break
        bundle = tuple(-x for x in best[2])
        selected = canonical_subset((*selected, *bundle))
        current_value = best[3]

    if polish and selected:
        improved = True
        while improved:
            improved = False
            selected_set = set(selected)
            unselected = tuple(layer for layer in problem.layers if layer not in selected_set)
            best_trial = selected
            best_value = current_value
            max_remove = min(max_bundle_size, len(selected))
            for remove_size in range(max_remove + 1):
                for removed in itertools.combinations(selected, remove_size):
                    base = tuple(x for x in selected if x not in removed)
                    max_add = min(
                        max_bundle_size,
                        problem.constraints.max_items - len(base),
                        len(unselected),
                    )
                    for add_size in range(max_add + 1):
                        if remove_size == 0 and add_size == 0:
                            continue
                        for added in itertools.combinations(unselected, add_size):
                            trial = canonical_subset((*base, *added))
                            if trial == selected or not problem.feasible(trial):
                                continue
                            trial_value = value(trial)
                            evaluations += 1
                            if trial_value > best_value + _EPS or (
                                math.isclose(trial_value, best_value, abs_tol=_EPS)
                                and trial < best_trial
                            ):
                                best_trial, best_value = trial, trial_value
            if best_value > current_value + _EPS:
                selected, current_value, improved = best_trial, best_value, True
    return SelectionResult(method, selected, current_value, evaluations, problem.feasible(selected))


def fixed_current_result(problem: CaptureProblem, value: ValueFunction) -> SelectionResult:
    selected = problem.current_subset()
    return SelectionResult(
        "current_5", selected, value(selected), 1, problem.feasible(selected)
    )


def dinkelbach_select(
    problem: CaptureProblem,
    inner_selector: Callable[[ValueFunction], SelectionResult],
    *,
    initial_throughput_tokens_per_ms: float | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 64,
) -> DinkelbachResult:
    """Optimize E[A]/E[T] using a supplied additive-objective selector."""
    empty_time = problem.round_time_ms(())
    if empty_time <= 0:
        raise ValueError("round time must be positive")
    rho = (
        problem.accepted_tokens(()) / empty_time
        if initial_throughput_tokens_per_ms is None
        else initial_throughput_tokens_per_ms
    )
    total_evaluations = 0
    selected: Subset = ()
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        def transformed(subset: Subset, trial_rho: float = rho) -> float:
            return problem.accepted_tokens(subset) - trial_rho * problem.round_time_ms(subset)

        result = inner_selector(transformed)
        total_evaluations += result.evaluations
        selected = result.selected
        accepted = problem.accepted_tokens(selected)
        round_time = problem.round_time_ms(selected)
        if round_time <= 0:
            raise ValueError("round time must be positive")
        residual = accepted - rho * round_time
        new_rho = accepted / round_time
        if abs(residual) <= tolerance or abs(new_rho - rho) <= tolerance:
            return DinkelbachResult(
                selected, accepted, round_time, new_rho, residual,
                iteration, total_evaluations,
            )
        rho = new_rho
    return DinkelbachResult(
        selected,
        problem.accepted_tokens(selected),
        problem.round_time_ms(selected),
        problem.accepted_tokens(selected) / problem.round_time_ms(selected),
        residual,
        max_iterations,
        total_evaluations,
    )


def adaptive_dinkelbach_select(
    probabilities: Mapping[Hashable, float],
    problems: Mapping[Hashable, CaptureProblem],
    inner_selector: Callable[[Hashable, CaptureProblem, ValueFunction], SelectionResult],
    *,
    initial_throughput_tokens_per_ms: float | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 64,
) -> AdaptiveDinkelbachResult:
    """Optimize a context-dependent set policy under one renewal ratio.

    The cheap context is observed before capture selection.  At a trial ratio
    ``rho``, the additive inner problem separates by context, but every context
    must use the same ``rho``.  Optimizing each context's ratio independently is
    generally not equivalent to optimizing the aggregate renewal ratio.
    """
    if not probabilities:
        raise ValueError("at least one context is required")
    if set(probabilities) != set(problems):
        raise ValueError("probabilities and problems must have the same contexts")
    if any(probability < 0 for probability in probabilities.values()):
        raise ValueError("negative context probability")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-10):
        raise ValueError("context probabilities must sum to one")

    def aggregate(policy: Mapping[Hashable, Subset]) -> tuple[float, float]:
        accepted = sum(
            probabilities[context]
            * problems[context].accepted_tokens(policy[context])
            for context in probabilities
        )
        round_time = sum(
            probabilities[context]
            * problems[context].round_time_ms(policy[context])
            for context in probabilities
        )
        if round_time <= 0:
            raise ValueError("expected round time must be positive")
        return accepted, round_time

    empty_policy = {context: () for context in probabilities}
    empty_accepted, empty_time = aggregate(empty_policy)
    rho = (
        empty_accepted / empty_time
        if initial_throughput_tokens_per_ms is None
        else initial_throughput_tokens_per_ms
    )
    policy: dict[Hashable, Subset] = dict(empty_policy)
    total_evaluations = 0
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        policy = {}
        for context, problem in problems.items():
            def transformed(
                subset: Subset,
                p: CaptureProblem = problem,
                trial_rho: float = rho,
            ) -> float:
                return p.accepted_tokens(subset) - trial_rho * p.round_time_ms(subset)

            result = inner_selector(context, problem, transformed)
            if not result.feasible:
                raise ValueError(f"inner selector returned an infeasible set for {context!r}")
            policy[context] = result.selected
            total_evaluations += result.evaluations

        accepted, round_time = aggregate(policy)
        residual = accepted - rho * round_time
        new_rho = accepted / round_time
        if abs(residual) <= tolerance or abs(new_rho - rho) <= tolerance:
            return AdaptiveDinkelbachResult(
                dict(policy),
                accepted,
                round_time,
                new_rho,
                residual,
                iteration,
                total_evaluations,
            )
        rho = new_rho

    accepted, round_time = aggregate(policy)
    return AdaptiveDinkelbachResult(
        dict(policy),
        accepted,
        round_time,
        accepted / round_time,
        residual,
        max_iterations,
        total_evaluations,
    )


def _best_partition_gain(
    value: ValueFunction,
    base: Subset,
    residual: Subset,
    max_bundle_size: int,
) -> float:
    base_value = value(base)

    @lru_cache(maxsize=None)
    def recurse(remaining: Subset) -> float:
        if not remaining:
            return 0.0
        first = remaining[0]
        tail = remaining[1:]
        best = -math.inf
        for extra_size in range(min(max_bundle_size - 1, len(tail)) + 1):
            for extra in itertools.combinations(tail, extra_size):
                bundle = canonical_subset((first, *extra))
                rest = canonical_subset(x for x in remaining if x not in bundle)
                gain = value(canonical_subset((*base, *bundle))) - base_value
                best = max(best, gain + recurse(rest))
        return best

    return recurse(residual)


def bundle_submodularity_ratio(
    layers: Sequence[int],
    value: ValueFunction,
    *,
    max_items: int,
    max_bundle_size: int,
) -> float:
    """Exact small-instance q-bundle submodularity ratio.

    For each disjoint base S and residual T with |S union T| <= max_items, the
    numerator is the best partition of T into bundles of size at most q, with
    every bundle marginal evaluated at the same base S.  The returned ratio is
    clipped to [0, 1].
    """
    ordered = canonical_subset(layers)
    ratio = 1.0
    for base in powerset(ordered, max_size=max_items):
        remaining = tuple(x for x in ordered if x not in base)
        max_residual = max_items - len(base)
        for residual in powerset(remaining, max_size=max_residual):
            if not residual:
                continue
            denominator = value(canonical_subset((*base, *residual))) - value(base)
            if denominator <= _EPS:
                continue
            numerator = _best_partition_gain(value, base, residual, max_bundle_size)
            ratio = min(ratio, max(0.0, numerator / denominator))
    return min(1.0, ratio)


def minimum_submodularity_gap(
    layers: Sequence[int], value: ValueFunction, *, max_base_size: int | None = None
) -> tuple[float, tuple[Subset, Subset, int]]:
    """Enumerate the smallest diminishing-returns gap on a small ground set."""
    ordered = canonical_subset(layers)
    best_gap = math.inf
    witness: tuple[Subset, Subset, int] = ((), (), ordered[0])
    base_limit = len(ordered) if max_base_size is None else max_base_size
    for b in powerset(ordered, max_size=base_limit):
        outside = tuple(x for x in ordered if x not in b)
        for element in outside:
            b_without = tuple(x for x in b if x != element)
            for a in powerset(b_without):
                if element in a:
                    continue
                gain_a = value(canonical_subset((*a, element))) - value(a)
                gain_b = value(canonical_subset((*b, element))) - value(b)
                gap = gain_a - gain_b
                if gap < best_gap:
                    best_gap = gap
                    witness = (a, b, element)
    return best_gap, witness


def fano_block_bits(
    conditional_entropy_bits: float,
    support_size: int,
    target_error: float,
) -> float:
    """Conditional Fano lower bound on capture entropy in bits."""
    if conditional_entropy_bits < 0 or support_size < 2:
        raise ValueError("invalid entropy or support")
    if conditional_entropy_bits > math.log2(support_size) + 1e-10:
        raise ValueError("conditional entropy exceeds the support-size bound")
    if not 0 <= target_error <= 1:
        raise ValueError("target_error must lie in [0,1]")
    # The Bayes error on a support of size M is never larger than 1-1/M.
    # Clamping is required when the requested error tolerance is looser than
    # random guessing; otherwise the Fano right side need not be monotone in
    # the supplied upper bound.
    effective_error = min(target_error, 1.0 - 1.0 / support_size)
    binary_entropy = 0.0
    if 0 < effective_error < 1:
        binary_entropy = -(
            effective_error * math.log2(effective_error)
            + (1 - effective_error) * math.log2(1 - effective_error)
        )
    bound = (
        conditional_entropy_bits
        - binary_entropy
        - effective_error * math.log2(support_size - 1)
    )
    return max(0.0, bound)


def prefix_acceptance_fano_bits(
    *,
    block_length: int,
    expected_prefix: float,
    alphabet_size: int,
) -> float:
    """Task-specific Fano bound from E[exact-prefix length].

    The target is assumed conditionally uniform over alphabet_size^r prefixes.
    The general non-uniform statement replaces r log2(q) by H(Y_1:r | D).
    """
    if block_length < 1 or alphabet_size < 2:
        raise ValueError("invalid block length or alphabet")
    if not 0 <= expected_prefix <= block_length:
        raise ValueError("expected prefix is outside [0,K]")
    best = 0.0
    for r in range(1, block_length + 1):
        if expected_prefix <= r - 1:
            continue
        error_upper = (block_length - expected_prefix) / (block_length - r + 1)
        support = alphabet_size ** r
        useful_error = min(max(error_upper, 0.0), 1.0 - 1.0 / support)
        best = max(
            best,
            fano_block_bits(r * math.log2(alphabet_size), support, useful_error),
        )
    return best


def adaptive_context_values(
    probabilities: Mapping[Hashable, float],
    actions: Sequence[Subset],
    accepted: Mapping[tuple[Hashable, Subset], float],
    times: Mapping[tuple[Hashable, Subset], float],
) -> dict[str, Any]:
    """Exact fixed and context-adaptive renewal values for a finite example."""
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-10):
        raise ValueError("context probabilities must sum to one")
    canonical_actions = tuple(canonical_subset(a) for a in actions)

    def aggregate(policy: Mapping[Hashable, Subset]) -> tuple[float, float, float]:
        reward = sum(
            probabilities[x] * accepted[(x, policy[x])] for x in probabilities
        )
        time = sum(probabilities[x] * times[(x, policy[x])] for x in probabilities)
        return reward, time, reward / time

    best_fixed: tuple[float, Subset, float, float] | None = None
    for action in canonical_actions:
        policy = {x: action for x in probabilities}
        reward, time, throughput = aggregate(policy)
        candidate = (throughput, action, reward, time)
        if best_fixed is None or candidate > best_fixed:
            best_fixed = candidate
    assert best_fixed is not None

    # Dinkelbach pointwise maximization.  The finite loop terminates when the
    # policy and ratio stabilize.
    rho = best_fixed[0]
    adaptive_policy: dict[Hashable, Subset] = {}
    for _ in range(64):
        adaptive_policy = {}
        for x in probabilities:
            adaptive_policy[x] = max(
                canonical_actions,
                key=lambda action: (
                    accepted[(x, action)] - rho * times[(x, action)],
                    tuple(-z for z in action),
                ),
            )
        reward, time, new_rho = aggregate(adaptive_policy)
        if abs(new_rho - rho) <= 1e-12:
            rho = new_rho
            break
        rho = new_rho
    return {
        "fixed_action": list(best_fixed[1]),
        "fixed_accepted": best_fixed[2],
        "fixed_time": best_fixed[3],
        "fixed_throughput": best_fixed[0],
        "adaptive_policy": {str(k): list(v) for k, v in adaptive_policy.items()},
        "adaptive_accepted": reward,
        "adaptive_time": time,
        "adaptive_throughput": rho,
        "adaptivity_ratio": rho / best_fixed[0] if best_fixed[0] > 0 else math.inf,
    }


def kill_criterion_net_upper_bound(
    *,
    accepted_gain_upper: float,
    added_time_lower_ms: float,
    baseline_throughput_lower_tokens_per_ms: float,
) -> float:
    """Upper confidence bound on incremental Dinkelbach net reward."""
    if added_time_lower_ms < 0 or baseline_throughput_lower_tokens_per_ms < 0:
        raise ValueError("lower bounds must be non-negative")
    return accepted_gain_upper - baseline_throughput_lower_tokens_per_ms * added_time_lower_ms


def make_naive_bayes_network(
    errors: Sequence[float] = (0.08, 0.16, 0.24, 0.32),
) -> DiscreteLayeredBayesNet:
    layers = tuple(range(1, len(errors) + 1))
    outcomes: list[JointOutcome] = []
    for y in (0, 1):
        for noise in itertools.product((0, 1), repeat=len(errors)):
            probability = 0.5
            observations = []
            for bit, error in zip(noise, errors):
                if not 0 <= error <= 1:
                    raise ValueError("error probabilities must lie in [0,1]")
                probability *= error if bit else 1 - error
                observations.append(y ^ bit)
            outcomes.append(JointOutcome(probability, 0, (y,), tuple(observations)))
    return DiscreteLayeredBayesNet(layers, outcomes)


def make_xor_synergy_network(decoy_error: float = 0.25) -> DiscreteLayeredBayesNet:
    """Two zero-singleton XOR sensors plus two weak independent decoys."""
    layers = (1, 2, 3, 4)
    outcomes: list[JointOutcome] = []
    for x1, x2, n3, n4 in itertools.product((0, 1), repeat=4):
        y = x1 ^ x2
        probability = 0.25
        probability *= decoy_error if n3 else 1 - decoy_error
        probability *= decoy_error if n4 else 1 - decoy_error
        observations = (x1, x2, y ^ n3, y ^ n4)
        outcomes.append(JointOutcome(probability, 0, (y,), observations))
    return DiscreteLayeredBayesNet(layers, outcomes)


def make_duplicate_network(
    error: float = 0.15,
    independent_error: float = 0.22,
) -> DiscreteLayeredBayesNet:
    """Z2 is a deterministic copy of noisy Z1; Z3 is an independent sensor."""
    layers = (1, 2, 3)
    outcomes: list[JointOutcome] = []
    for y, n1, n3 in itertools.product((0, 1), repeat=3):
        probability = 0.5
        probability *= error if n1 else 1 - error
        probability *= independent_error if n3 else 1 - independent_error
        z1 = y ^ n1
        outcomes.append(JointOutcome(probability, 0, (y,), (z1, z1, y ^ n3)))
    return DiscreteLayeredBayesNet(layers, outcomes)


def _depth_loading(
    layers: Sequence[int],
    *,
    center: float,
    width: float,
    target_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    loadings = np.zeros((len(layers), target_dim), dtype=float)
    directions = rng.normal(size=(len(layers), target_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for i, layer in enumerate(layers):
        amplitude = 0.12 + 2.6 * math.exp(-0.5 * ((layer - center) / width) ** 2)
        phase = (layer - 1) / max(1, len(layers) - 1)
        structured = np.array([
            math.cos(math.pi * phase),
            math.sin(math.pi * phase),
            math.cos(2 * math.pi * phase),
            math.sin(2 * math.pi * phase),
        ])[:target_dim]
        structured /= max(np.linalg.norm(structured), _EPS)
        loadings[i] = amplitude * (0.88 * structured + 0.12 * directions[i])
    return loadings


def make_l45_capture_problem(
    kind: str,
    *,
    seed: int = 4070,
) -> tuple[CaptureProblem, ConditionalGaussianInformation]:
    """Construct deterministic L=45 synthetic capture problems.

    Candidate captures are layers 1 through 44.  ``kind`` is one of
    ``near_duplicate``, ``change_point``, or ``expensive_late``.
    """
    if kind not in {"near_duplicate", "change_point", "expensive_late"}:
        raise ValueError("unknown synthetic kind")
    layers = tuple(range(1, 45))
    seed_offset = {"near_duplicate": 1, "change_point": 2, "expensive_late": 3}[kind]
    rng = np.random.default_rng(seed + seed_offset)
    target_dim = 4
    target_cov = np.diag([1.0, 0.8, 0.55, 0.35])
    noise = np.eye(len(layers))

    if kind == "change_point":
        early = _depth_loading(layers, center=10, width=4.5, target_dim=target_dim, rng=rng)
        late = _depth_loading(layers, center=36, width=4.5, target_dim=target_dim, rng=rng)
        regimes = (
            GaussianRegime("early", 0.52, early, noise, target_cov),
            GaussianRegime("late", 0.48, late, noise, target_cov),
        )
    else:
        loading = _depth_loading(
            layers,
            center=24 if kind == "near_duplicate" else 40,
            width=8 if kind == "near_duplicate" else 3.2,
            target_dim=target_dim,
            rng=rng,
        )
        if kind == "near_duplicate":
            # Three nearly identical observations around the existing middle capture.
            source = layers.index(24)
            for layer in (23, 25):
                loading[layers.index(layer)] = loading[source] + 0.006 * rng.normal(size=target_dim)
            duplicate_indices = [layers.index(layer) for layer in (23, 24, 25)]
            for i, j in itertools.combinations(duplicate_indices, 2):
                noise[i, j] = noise[j, i] = 0.985
        if kind == "expensive_late":
            # Make layer 42 the strongest singleton sensor, then price its
            # synchronization latency high enough that throughput can reject it.
            loading[layers.index(42)] *= 2.8
        regimes = (GaussianRegime("all", 1.0, loading, noise, target_cov),)

    information = ConditionalGaussianInformation(layers, regimes)
    costs: dict[int, CaptureCost] = {}
    for layer in layers:
        encoded_bytes = 8192.0 + 512.0 * ((layer * 7) % 5)
        latency = 0.055 + 0.0018 * layer + 0.008 * (layer % 4 == 0)
        costs[layer] = CaptureCost(encoded_bytes, latency)
    if kind == "expensive_late":
        costs[42] = CaptureCost(costs[42].bytes, 24.0)
        costs[41] = CaptureCost(costs[41].bytes, 2.4)

    bandwidth = 210_000.0  # bytes/ms, synthetic D2D/H2D effective path
    constraints = SelectionConstraints(max_items=5)

    @lru_cache(maxsize=None)
    def info_fn(selected: Subset) -> float:
        return information.mutual_information(canonical_subset(selected))

    def regime_acceptance(regime: GaussianRegime, selected: Subset) -> float:
        info = information.regime_information(regime.name, canonical_subset(selected))
        return 0.35 + 7.65 * (1.0 - math.exp(-0.47 * info))

    @lru_cache(maxsize=None)
    def accepted_fn(selected: Subset) -> float:
        subset = canonical_subset(selected)
        return sum(
            regime.probability * regime_acceptance(regime, subset)
            for regime in regimes
        )

    @lru_cache(maxsize=None)
    def time_fn(selected: Subset) -> float:
        subset = canonical_subset(selected)
        capture_ms = sum(
            costs[layer].latency_ms + costs[layer].bytes / bandwidth
            for layer in subset
        )
        # Rejected prefixes cost verification/fallback work.  This term makes the
        # ratio objective differ from pure capture cost and from information.
        rejection_ms = 0.0
        for regime in regimes:
            accepted = regime_acceptance(regime, subset)
            penalty = 5.5 if kind != "expensive_late" else 7.0
            rejection_ms += regime.probability * penalty * (8.0 - accepted)
        return 34.0 + capture_ms + rejection_ms

    problem = CaptureProblem(
        name=kind,
        layers=layers,
        costs=costs,
        constraints=constraints,
        information=info_fn,
        accepted_tokens=accepted_fn,
        round_time_ms=time_fn,
        bandwidth_bytes_per_ms=bandwidth,
    )
    return problem, information


def make_small_problem_from_network(
    name: str,
    network: DiscreteLayeredBayesNet,
    *,
    max_items: int,
    expensive_layer: int | None = None,
) -> CaptureProblem:
    costs = {
        layer: CaptureCost(4096.0 + 256.0 * layer, 0.08 + 0.01 * layer)
        for layer in network.layers
    }
    if expensive_layer is not None:
        costs[expensive_layer] = CaptureCost(costs[expensive_layer].bytes, 9.0)
    bandwidth = 180_000.0

    @lru_cache(maxsize=None)
    def info(selected: Subset) -> float:
        return network.mutual_information(canonical_subset(selected))

    @lru_cache(maxsize=None)
    def accepted(selected: Subset) -> float:
        # One-token Bayes correctness is the engine-aligned reward in these small
        # classification examples.
        return 1.0 - network.bayes_block_error(canonical_subset(selected))

    @lru_cache(maxsize=None)
    def time_ms(selected: Subset) -> float:
        subset = canonical_subset(selected)
        capture = sum(
            costs[layer].latency_ms + costs[layer].bytes / bandwidth
            for layer in subset
        )
        return 1.0 + capture + 4.0 * (1.0 - accepted(subset))

    return CaptureProblem(
        name,
        network.layers,
        costs,
        SelectionConstraints(max_items=max_items),
        info,
        accepted,
        time_ms,
        bandwidth,
        current_layers=canonical_subset(
            layer for layer in _CURRENT_CAPTURE_LAYERS if layer in network.layers
        ),
    )


def _demo_payload() -> dict[str, Any]:
    xor = make_xor_synergy_network()
    problem = make_small_problem_from_network("xor", xor, max_items=2)
    exact = exact_select(problem, problem.information)
    greedy = greedy_select(problem, problem.information)
    pair = bundle_density_greedy(problem, problem.information)
    return {
        "xor_information": {
            "singletons": {str(layer): xor.mutual_information((layer,)) for layer in xor.layers},
            "xor_pair": xor.mutual_information((1, 2)),
            "greedy": asdict(greedy),
            "pair_bundle": asdict(pair),
            "optimum": asdict(exact),
            "gamma_q1": bundle_submodularity_ratio(
                xor.layers, xor.mutual_information, max_items=2, max_bundle_size=1
            ),
            "gamma_q2": bundle_submodularity_ratio(
                xor.layers, xor.mutual_information, max_items=2, max_bundle_size=2
            ),
        },
        "fano": {
            "block_bits": fano_block_bits(8.0, 256, 0.05),
            "prefix_bits": prefix_acceptance_fano_bits(
                block_length=8, expected_prefix=6.5, alphabet_size=2
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("choose --demo")
    payload = _demo_payload()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
