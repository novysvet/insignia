#!/usr/bin/env python3
"""CPU simulator for anytime A/B decisions under hostile WSL-like noise.

The environment has a latent Markov machine state, persistent cache state,
log-normal noise, a Pareto shock component with infinite variance by default,
a deterministic clock change point, and optional multi-run contender bursts.
It can run a carryover-guarded protocol or an intentionally confounded raw
AB/BA protocol.

Outputs compare:

* a conventional three-run median;
* a naive arithmetic mean;
* a fixed-sample paired t test on log ratios;
* the same fixed-sample t test invalidly peeked after every pair;
* an IID paired bootstrap interval for the median log ratio;
* the anytime sign e-process from ``anytime_ab.py``.

No model, CUDA installation, or GPU is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import special, stats

from anytime_ab import MixtureBettingEProcess, comparison_score


@dataclass(frozen=True)
class NoiseConfig:
    state_multipliers: tuple[float, ...] = (0.84, 1.0, 1.42)
    log_sigmas: tuple[float, ...] = (0.055, 0.12, 0.24)
    pareto_probabilities: tuple[float, ...] = (0.008, 0.025, 0.075)
    transition: tuple[tuple[float, ...], ...] = (
        (0.87, 0.12, 0.01),
        (0.10, 0.80, 0.10),
        (0.03, 0.24, 0.73),
    )
    pareto_shape: float = 1.35
    clock_change_after_run: int = 58
    clock_after_multiplier: float = 1.28
    burst_start_probability: float = 0.025
    burst_min_runs: int = 2
    burst_max_runs: int = 8
    burst_min_multiplier: float = 1.25
    burst_max_multiplier: float = 1.90

    def validate(self) -> None:
        n = len(self.state_multipliers)
        if n == 0 or len(self.log_sigmas) != n or len(self.pareto_probabilities) != n:
            raise ValueError("state vectors have inconsistent lengths")
        if len(self.transition) != n:
            raise ValueError("transition matrix has wrong height")
        for row in self.transition:
            if len(row) != n or any(p < 0 for p in row):
                raise ValueError("invalid transition row")
            if not math.isclose(sum(row), 1.0, abs_tol=1e-10):
                raise ValueError("transition row does not sum to one")
        if self.pareto_shape <= 1:
            raise ValueError("Pareto shape must exceed one for finite campaign means")
        if self.clock_change_after_run < 0:
            raise ValueError("clock change run must be non-negative")
        if not 0 <= self.burst_start_probability <= 1:
            raise ValueError("invalid burst probability")
        if not 1 <= self.burst_min_runs <= self.burst_max_runs:
            raise ValueError("invalid burst duration")


@dataclass(frozen=True)
class SimulationConfig:
    true_ratio: float = 1.04
    base_latency_ms: float = 500.0
    max_pairs: int = 80
    cache_mode: str = "cold"
    guarded: bool = True
    neutralization_strength: float = 1.0
    cache_decay: float = 0.58
    warm_strength_a: float = 0.78
    warm_strength_b: float = 0.78
    same_arm_cache_gain: float = 0.16
    carryover_a_to_b: float = 0.30
    carryover_b_to_a: float = -0.20
    prep_fraction: float = 0.12
    warmup_runs: int = 1
    b_sigma_multiplier: float = 1.0
    b_pareto_multiplier: float = 1.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    def validate(self) -> None:
        self.noise.validate()
        if self.true_ratio <= 0 or self.base_latency_ms <= 0:
            raise ValueError("ratio and base latency must be positive")
        if self.max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        if self.cache_mode not in {"cold", "warm"}:
            raise ValueError("cache_mode must be cold or warm")
        if not 0 <= self.neutralization_strength <= 1:
            raise ValueError("invalid neutralization strength")
        if not 0 <= self.cache_decay <= 1:
            raise ValueError("invalid cache decay")
        if min(self.warm_strength_a, self.warm_strength_b) < 0:
            raise ValueError("warm strengths must be non-negative")
        if self.prep_fraction < 0 or self.warmup_runs < 0:
            raise ValueError("invalid prep/warmup configuration")
        if min(self.b_sigma_multiplier, self.b_pareto_multiplier) <= 0:
            raise ValueError("B noise multipliers must be positive")


@dataclass(frozen=True)
class SimRun:
    arm: str
    slot: int
    measured: bool
    latency_ms: float
    machine_state: int
    machine_multiplier: float
    clock_multiplier: float
    burst_multiplier: float
    cache_before: float
    cache_factor: float
    lognormal_multiplier: float
    pareto_multiplier: float


@dataclass(frozen=True)
class SimPair:
    pair_index: int
    order: str
    a_ms: float
    b_ms: float
    campaign_ms: float
    true_ratio: float
    guarded: bool
    cache_mode: str
    state_start: int
    state_end: int
    clock_start: float
    burst_seen: bool
    runs: tuple[SimRun, ...]

    @property
    def ratio(self) -> float:
        return self.b_ms / self.a_ms

    def flat_dict(self) -> dict[str, Any]:
        a_run = next(r for r in self.runs if r.measured and r.arm == "A")
        b_run = next(r for r in self.runs if r.measured and r.arm == "B")
        return {
            "pair_index": self.pair_index,
            "order": self.order,
            "a_ms": self.a_ms,
            "b_ms": self.b_ms,
            "b_over_a": self.ratio,
            "true_ratio": self.true_ratio,
            "campaign_ms": self.campaign_ms,
            "guarded": self.guarded,
            "cache_mode": self.cache_mode,
            "state_start": self.state_start,
            "state_end": self.state_end,
            "clock_start": self.clock_start,
            "burst_seen": self.burst_seen,
            "a_state": a_run.machine_state,
            "b_state": b_run.machine_state,
            "a_cache_before": a_run.cache_before,
            "b_cache_before": b_run.cache_before,
            "a_cache_factor": a_run.cache_factor,
            "b_cache_factor": b_run.cache_factor,
            "a_lognormal": a_run.lognormal_multiplier,
            "b_lognormal": b_run.lognormal_multiplier,
            "a_pareto": a_run.pareto_multiplier,
            "b_pareto": b_run.pareto_multiplier,
            "a_burst": a_run.burst_multiplier,
            "b_burst": b_run.burst_multiplier,
        }


class HostileWSLSimulator:
    def __init__(self, seed: int, config: SimulationConfig):
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.machine_state = int(self.rng.integers(0, len(config.noise.state_multipliers)))
        self.cache_hotness = 0.0
        self.last_arm: str | None = None
        self.global_run = 0
        self.burst_remaining = 0
        self.burst_multiplier = 1.0

    @property
    def clock_multiplier(self) -> float:
        return (self.config.noise.clock_after_multiplier
                if self.global_run >= self.config.noise.clock_change_after_run
                else 1.0)

    def _prepare(self) -> float:
        cfg = self.config
        old_cache = self.cache_hotness
        self.cache_hotness = (1.0 - cfg.neutralization_strength) * old_cache
        if cfg.neutralization_strength >= 1.0 - 1e-15:
            self.last_arm = None
        # Preparation itself is noisy wall time, but is never part of the arm metric.
        return cfg.base_latency_ms * cfg.prep_fraction * math.exp(float(self.rng.normal(0, 0.08)))

    def _current_burst(self) -> float:
        ncfg = self.config.noise
        if self.burst_remaining <= 0 and self.rng.random() < ncfg.burst_start_probability:
            self.burst_remaining = int(self.rng.integers(
                ncfg.burst_min_runs, ncfg.burst_max_runs + 1))
            self.burst_multiplier = float(self.rng.uniform(
                ncfg.burst_min_multiplier, ncfg.burst_max_multiplier))
        return self.burst_multiplier if self.burst_remaining > 0 else 1.0

    def _cache_factor(self, arm: str) -> float:
        if self.last_arm is None:
            return 1.0
        cfg = self.config
        if self.last_arm == arm:
            gain = cfg.same_arm_cache_gain
        elif self.last_arm == "A" and arm == "B":
            gain = cfg.carryover_a_to_b
        elif self.last_arm == "B" and arm == "A":
            gain = cfg.carryover_b_to_a
        else:
            raise AssertionError("unreachable arm transition")
        return max(0.45, 1.0 - gain * self.cache_hotness)

    def _run(self, arm: str, *, slot: int, measured: bool) -> SimRun:
        cfg = self.config
        ncfg = cfg.noise
        state = self.machine_state
        state_mult = ncfg.state_multipliers[state]
        clock = self.clock_multiplier
        burst = self._current_burst()
        cache_before = self.cache_hotness
        cache_factor = self._cache_factor(arm)
        sigma = ncfg.log_sigmas[state] * (cfg.b_sigma_multiplier if arm == "B" else 1.0)
        lognormal = math.exp(float(self.rng.normal(0.0, sigma)))
        pareto_probability = ncfg.pareto_probabilities[state]
        if arm == "B":
            pareto_probability = min(1.0, pareto_probability * cfg.b_pareto_multiplier)
        pareto = (1.0 + float(self.rng.pareto(ncfg.pareto_shape))
                  if self.rng.random() < pareto_probability else 1.0)
        treatment = cfg.true_ratio if arm == "B" else 1.0
        latency = (cfg.base_latency_ms * treatment * state_mult * clock * burst
                   * cache_factor * lognormal * pareto)

        warm = cfg.warm_strength_b if arm == "B" else cfg.warm_strength_a
        self.cache_hotness = min(1.0, cfg.cache_decay * self.cache_hotness + warm)
        self.last_arm = arm
        self.machine_state = int(self.rng.choice(
            len(ncfg.transition), p=ncfg.transition[state]))
        self.global_run += 1
        if self.burst_remaining > 0:
            self.burst_remaining -= 1
            if self.burst_remaining == 0:
                self.burst_multiplier = 1.0
        return SimRun(
            arm=arm,
            slot=slot,
            measured=measured,
            latency_ms=latency,
            machine_state=state,
            machine_multiplier=state_mult,
            clock_multiplier=clock,
            burst_multiplier=burst,
            cache_before=cache_before,
            cache_factor=cache_factor,
            lognormal_multiplier=lognormal,
            pareto_multiplier=pareto,
        )

    def _guarded_subtrial(self, arm: str, slot: int) -> tuple[list[SimRun], float]:
        total = self._prepare()
        runs: list[SimRun] = []
        if self.config.cache_mode == "warm":
            for _ in range(self.config.warmup_runs):
                warmup = self._run(arm, slot=slot, measured=False)
                runs.append(warmup)
                total += warmup.latency_ms
        measured = self._run(arm, slot=slot, measured=True)
        runs.append(measured)
        total += measured.latency_ms
        return runs, total

    def pair(self, pair_index: int) -> SimPair:
        cfg = self.config
        state_start = self.machine_state
        clock_start = self.clock_multiplier
        order = "AB" if self.rng.integers(0, 2) == 0 else "BA"
        arms = list(order)
        runs: list[SimRun] = []
        campaign_ms = 0.0
        if cfg.guarded:
            for slot, arm in enumerate(arms, 1):
                subruns, cost = self._guarded_subtrial(arm, slot)
                runs.extend(subruns)
                campaign_ms += cost
        else:
            campaign_ms += self._prepare()
            if cfg.cache_mode == "warm":
                # A raw design warms only the first sequence member.  This is
                # intentionally a policy effect, not a direct-effect design.
                for _ in range(cfg.warmup_runs):
                    warmup = self._run(arms[0], slot=0, measured=False)
                    runs.append(warmup)
                    campaign_ms += warmup.latency_ms
            for slot, arm in enumerate(arms, 1):
                measured = self._run(arm, slot=slot, measured=True)
                runs.append(measured)
                campaign_ms += measured.latency_ms
        a_ms = next(r.latency_ms for r in runs if r.measured and r.arm == "A")
        b_ms = next(r.latency_ms for r in runs if r.measured and r.arm == "B")
        return SimPair(
            pair_index=pair_index,
            order=order,
            a_ms=a_ms,
            b_ms=b_ms,
            campaign_ms=campaign_ms,
            true_ratio=cfg.true_ratio,
            guarded=cfg.guarded,
            cache_mode=cfg.cache_mode,
            state_start=state_start,
            state_end=self.machine_state,
            clock_start=clock_start,
            burst_seen=any(r.burst_multiplier > 1 for r in runs),
            runs=tuple(runs),
        )

    def campaign(self) -> list[SimPair]:
        return [self.pair(i) for i in range(1, self.config.max_pairs + 1)]


@dataclass(frozen=True)
class MethodConfig:
    promote_ratio: float = 0.98
    reject_ratio: float = 1.02
    alpha: float = 0.05
    beta: float = 0.05
    naive_pairs: int = 10
    fixed_pairs: int = 24
    bootstrap_reps: int = 399
    anytime_min_pairs: int = 6
    max_pairs: int = 80


@dataclass(frozen=True)
class DecisionResult:
    method: str
    status: str
    pairs_used: int
    campaign_seconds: float
    estimate: float | None
    promote_e: float | None = None
    reject_e: float | None = None
    detail: str = ""


def _prefix_cost(records: Sequence[SimPair], n: int) -> float:
    return sum(r.campaign_ms for r in records[:n]) / 1000.0


def decide_three_run_median(records: Sequence[SimPair], cfg: MethodConfig) -> DecisionResult:
    n = min(3, len(records))
    a = np.asarray([r.a_ms for r in records[:n]])
    b = np.asarray([r.b_ms for r in records[:n]])
    estimate = float(np.median(b) / np.median(a))
    if estimate <= cfg.promote_ratio:
        status = "PROMOTE"
    elif estimate >= cfg.reject_ratio:
        status = "REJECT"
    else:
        status = "INCONCLUSIVE"
    return DecisionResult("three_run_median", status, n, _prefix_cost(records, n), estimate)


def decide_naive_mean(records: Sequence[SimPair], cfg: MethodConfig) -> DecisionResult:
    n = min(cfg.naive_pairs, len(records))
    estimate = (sum(r.b_ms for r in records[:n]) /
                sum(r.a_ms for r in records[:n]))
    if estimate <= cfg.promote_ratio:
        status = "PROMOTE"
    elif estimate >= cfg.reject_ratio:
        status = "REJECT"
    else:
        status = "INCONCLUSIVE"
    return DecisionResult("naive_mean", status, n, _prefix_cost(records, n), estimate)


def _paired_t_pvalues(log_ratios: np.ndarray, cfg: MethodConfig) -> tuple[float, float]:
    n = len(log_ratios)
    mean = float(log_ratios.mean())
    sd = float(log_ratios.std(ddof=1))
    if sd <= 1e-15:
        p_promote = 0.0 if mean < math.log(cfg.promote_ratio) else 1.0
        p_reject = 0.0 if mean > math.log(cfg.reject_ratio) else 1.0
        return p_promote, p_reject
    se = sd / math.sqrt(n)
    t_promote = (mean - math.log(cfg.promote_ratio)) / se
    t_reject = (mean - math.log(cfg.reject_ratio)) / se
    return float(stats.t.cdf(t_promote, n - 1)), float(stats.t.sf(t_reject, n - 1))


def decide_paired_t(records: Sequence[SimPair], cfg: MethodConfig) -> DecisionResult:
    n = min(cfg.fixed_pairs, len(records))
    log_ratios = np.log(np.asarray([r.ratio for r in records[:n]]))
    p_promote, p_reject = _paired_t_pvalues(log_ratios, cfg)
    if p_promote <= cfg.alpha:
        status = "PROMOTE"
    elif p_reject <= cfg.beta:
        status = "REJECT"
    else:
        status = "INCONCLUSIVE"
    return DecisionResult(
        "fixed_paired_t", status, n, _prefix_cost(records, n),
        float(math.exp(log_ratios.mean())),
        detail=f"p_promote={p_promote:.6g};p_reject={p_reject:.6g}",
    )



def decide_peeking_paired_t(records: Sequence[SimPair], cfg: MethodConfig) -> DecisionResult:
    """The common invalid practice: recompute a fixed-n t test after every pair."""

    n_max = min(cfg.max_pairs, len(records))
    start = max(3, cfg.anytime_min_pairs)
    values = np.log(np.asarray([r.ratio for r in records[:n_max]], dtype=float))
    counts = np.arange(1, n_max + 1, dtype=float)
    sums = np.cumsum(values)
    sumsq = np.cumsum(values * values)
    means = sums / counts
    variances = np.zeros(n_max, dtype=float)
    variances[1:] = np.maximum(
        0.0,
        (sumsq[1:] - sums[1:] * sums[1:] / counts[1:]) / (counts[1:] - 1.0),
    )
    se = np.sqrt(variances / counts)
    indices = np.arange(start - 1, n_max)
    good_se = se[indices] > 1e-15
    p_promote = np.ones(len(indices), dtype=float)
    p_reject = np.ones(len(indices), dtype=float)
    if np.any(good_se):
        df = counts[indices][good_se] - 1.0
        t_promote = ((means[indices][good_se] - math.log(cfg.promote_ratio)) /
                     se[indices][good_se])
        t_reject = ((means[indices][good_se] - math.log(cfg.reject_ratio)) /
                    se[indices][good_se])
        p_promote[good_se] = special.stdtr(df, t_promote)
        p_reject[good_se] = special.stdtr(df, -t_reject)
    zero_se = ~good_se
    if np.any(zero_se):
        p_promote[zero_se] = np.where(
            means[indices][zero_se] < math.log(cfg.promote_ratio), 0.0, 1.0)
        p_reject[zero_se] = np.where(
            means[indices][zero_se] > math.log(cfg.reject_ratio), 0.0, 1.0)
    crossed = (p_promote <= cfg.alpha) | (p_reject <= cfg.beta)
    if np.any(crossed):
        offset = int(np.argmax(crossed))
        n = int(indices[offset] + 1)
        status = ("PROMOTE" if p_promote[offset] / cfg.alpha <= p_reject[offset] / cfg.beta
                  else "REJECT")
        return DecisionResult(
            "peeking_paired_t", status, n, _prefix_cost(records, n),
            float(math.exp(means[n - 1])),
            detail=(f"p_promote={p_promote[offset]:.6g};"
                    f"p_reject={p_reject[offset]:.6g}"),
        )
    return DecisionResult(
        "peeking_paired_t", "INCONCLUSIVE", n_max, _prefix_cost(records, n_max),
        float(math.exp(means[-1])),
    )


def decide_bootstrap(
    records: Sequence[SimPair], cfg: MethodConfig, *, seed: int
) -> DecisionResult:
    n = min(cfg.fixed_pairs, len(records))
    values = np.log(np.asarray([r.ratio for r in records[:n]]))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(cfg.bootstrap_reps, n))
    boot = np.median(values[indices], axis=1)
    lower = float(np.quantile(boot, cfg.beta))
    upper = float(np.quantile(boot, 1.0 - cfg.alpha))
    if upper < math.log(cfg.promote_ratio):
        status = "PROMOTE"
    elif lower > math.log(cfg.reject_ratio):
        status = "REJECT"
    else:
        status = "INCONCLUSIVE"
    return DecisionResult(
        "iid_bootstrap_median", status, n, _prefix_cost(records, n),
        float(math.exp(np.median(values))),
        detail=f"ci=({math.exp(lower):.6g},{math.exp(upper):.6g})",
    )


def decide_anytime(records: Sequence[SimPair], cfg: MethodConfig) -> DecisionResult:
    promote = MixtureBettingEProcess()
    reject = MixtureBettingEProcess()
    n_max = min(cfg.max_pairs, len(records))
    for i, record in enumerate(records[:n_max], 1):
        promote.update(comparison_score(
            record.ratio, cfg.promote_ratio, lower_is_better=True))
        reject.update(comparison_score(
            record.ratio, cfg.reject_ratio, lower_is_better=False))
        if i >= cfg.anytime_min_pairs:
            if reject.crossed(cfg.beta):
                return DecisionResult(
                    "anytime_sign_e", "REJECT", i, _prefix_cost(records, i),
                    float(np.median([r.ratio for r in records[:i]])),
                    promote.max_e_value, reject.max_e_value,
                )
            if promote.crossed(cfg.alpha):
                return DecisionResult(
                    "anytime_sign_e", "PROMOTE", i, _prefix_cost(records, i),
                    float(np.median([r.ratio for r in records[:i]])),
                    promote.max_e_value, reject.max_e_value,
                )
    return DecisionResult(
        "anytime_sign_e", "INCONCLUSIVE", n_max, _prefix_cost(records, n_max),
        float(np.median([r.ratio for r in records[:n_max]])),
        promote.max_e_value, reject.max_e_value,
    )


def all_decisions(
    records: Sequence[SimPair], cfg: MethodConfig, *, bootstrap_seed: int
) -> tuple[DecisionResult, ...]:
    return (
        decide_three_run_median(records, cfg),
        decide_naive_mean(records, cfg),
        decide_paired_t(records, cfg),
        decide_peeking_paired_t(records, cfg),
        decide_bootstrap(records, cfg, seed=bootstrap_seed),
        decide_anytime(records, cfg),
    )


@dataclass(frozen=True)
class Scenario:
    name: str
    true_ratio: float
    guarded: bool
    cache_mode: str = "cold"
    assumption_status: str = "declared-null-model"


DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("boundary_guarded", 0.98, True),
    Scenario("slower_guarded", 1.04, True),
    Scenario("faster_guarded", 0.92, True),
    Scenario("slower_unguarded_carryover", 1.04, False,
             assumption_status="carryover-confounded"),
)


def _regret_seconds(
    result: DecisionResult,
    *, true_ratio: float,
    base_latency_ms: float,
    deployment_runs: int,
) -> float:
    measurement = result.campaign_seconds
    deployment = deployment_runs * base_latency_ms / 1000.0
    if result.status == "PROMOTE":
        decision_loss = max(0.0, true_ratio - 1.0) * deployment
    else:
        decision_loss = max(0.0, 1.0 - true_ratio) * deployment
    return measurement + decision_loss


def evaluate(
    *,
    trials: int,
    seed: int,
    method_config: MethodConfig,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    deployment_runs: int = 50_000,
) -> list[dict[str, Any]]:
    if trials < 1:
        raise ValueError("trials must be positive")
    master = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        accum: dict[str, dict[str, float]] = {}
        for _ in range(trials):
            campaign_seed = int(master.integers(0, 2**63 - 1))
            bootstrap_seed = int(master.integers(0, 2**63 - 1))
            sim_cfg = SimulationConfig(
                true_ratio=scenario.true_ratio,
                max_pairs=method_config.max_pairs,
                cache_mode=scenario.cache_mode,
                guarded=scenario.guarded,
            )
            records = HostileWSLSimulator(campaign_seed, sim_cfg).campaign()
            for result in all_decisions(records, method_config,
                                        bootstrap_seed=bootstrap_seed):
                stats_row = accum.setdefault(result.method, {
                    "promote": 0.0,
                    "reject": 0.0,
                    "inconclusive": 0.0,
                    "false_promotion": 0.0,
                    "false_rejection": 0.0,
                    "pairs": 0.0,
                    "campaign_seconds": 0.0,
                    "regret_seconds": 0.0,
                })
                stats_row[result.status.lower()] += 1
                if (
                    scenario.true_ratio >= method_config.promote_ratio
                    and result.status == "PROMOTE"
                ):
                    stats_row["false_promotion"] += 1
                if scenario.true_ratio < method_config.promote_ratio and result.status == "REJECT":
                    stats_row["false_rejection"] += 1
                stats_row["pairs"] += result.pairs_used
                stats_row["campaign_seconds"] += result.campaign_seconds
                stats_row["regret_seconds"] += _regret_seconds(
                    result,
                    true_ratio=scenario.true_ratio,
                    base_latency_ms=sim_cfg.base_latency_ms,
                    deployment_runs=deployment_runs,
                )
        for method, values in accum.items():
            promote_rate = values["promote"] / trials
            reject_rate = values["reject"] / trials
            false_promote = values["false_promotion"] / trials
            false_reject = values["false_rejection"] / trials
            rows.append({
                "scenario": scenario.name,
                "assumption_status": scenario.assumption_status,
                "true_ratio": scenario.true_ratio,
                "guarded": scenario.guarded,
                "method": method,
                "trials": trials,
                "promotion_rate": promote_rate,
                "rejection_rate": reject_rate,
                "inconclusive_rate": values["inconclusive"] / trials,
                "false_promotion_rate": false_promote,
                "false_promotion_se": math.sqrt(false_promote * (1 - false_promote) / trials),
                "false_rejection_rate": false_reject,
                "false_rejection_se": math.sqrt(false_reject * (1 - false_reject) / trials),
                "expected_pairs": values["pairs"] / trials,
                "expected_campaign_seconds": values["campaign_seconds"] / trials,
                "expected_regret_seconds": values["regret_seconds"] / trials,
            })
    return rows


def find_replay_seed(
    *,
    start: int,
    limit: int,
    true_ratio: float,
    method_config: MethodConfig,
) -> tuple[int, list[SimPair], DecisionResult]:
    for seed in range(start, start + limit):
        sim_cfg = SimulationConfig(
            true_ratio=true_ratio,
            max_pairs=max(3, method_config.max_pairs),
            guarded=True,
            cache_mode="cold",
        )
        records = HostileWSLSimulator(seed, sim_cfg).campaign()
        result = decide_three_run_median(records, method_config)
        if result.status == "PROMOTE":
            return seed, records[:3], result
    raise RuntimeError(f"no replay seed found in [{start}, {start + limit})")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_replay(path: Path, records: Sequence[SimPair]) -> None:
    write_csv(path, [r.flat_dict() for r in records])


def write_decision_log(path: Path, records: Sequence[SimPair], *, cell: str) -> None:
    rows = []
    for record in records:
        rows.append({
            "candidate_id": "B",
            "pair_id": f"sim-{record.pair_index:04d}",
            "epoch": 0,
            "stage": "full",
            "cell": cell,
            "case_id": "synthetic",
            "case_selection_probability": 1.0,
            "order": record.order,
            "a_value": record.a_ms,
            "b_value": record.b_ms,
            "parity_ok": "true",
            "prep_ok": str(record.guarded).lower(),
            "validity_code": "ok",
            "cost_seconds": record.campaign_ms / 1000.0,
            "randomization_u64": "simulated",
            "seed_commitment": "synthetic",
            "protocol_hash": "synthetic",
            "baseline_commit": "A",
            "candidate_commit": "B",
        })
    write_csv(path, rows)


def _cmd_simulate(args: argparse.Namespace) -> int:
    cfg = MethodConfig(
        alpha=args.alpha,
        beta=args.beta,
        max_pairs=args.max_pairs,
        bootstrap_reps=args.bootstrap_reps,
    )
    rows = evaluate(
        trials=args.trials,
        seed=args.seed,
        method_config=cfg,
        deployment_runs=args.deployment_runs,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", rows)
    (out / "summary.json").write_text(
        json.dumps({
            "seed": args.seed,
            "trials": args.trials,
            "deployment_runs": args.deployment_runs,
            "method_config": asdict(cfg),
            "scenarios": [asdict(s) for s in DEFAULT_SCENARIOS],
            "rows": rows,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def _cmd_find_seed(args: argparse.Namespace) -> int:
    cfg = MethodConfig(max_pairs=max(3, args.max_pairs))
    seed, records, result = find_replay_seed(
        start=args.start,
        limit=args.limit,
        true_ratio=args.true_ratio,
        method_config=cfg,
    )
    payload = {
        "seed": seed,
        "true_ratio": args.true_ratio,
        "decision": asdict(result),
        "pairs": [r.flat_dict() for r in records],
    }
    if args.output:
        write_replay(Path(args.output), records)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    cfg = MethodConfig(max_pairs=args.max_pairs)
    sim_cfg = SimulationConfig(
        true_ratio=args.true_ratio,
        max_pairs=args.max_pairs,
        guarded=not args.unguarded,
        cache_mode=args.cache_mode,
    )
    records = HostileWSLSimulator(args.seed, sim_cfg).campaign()
    decisions = all_decisions(records, cfg, bootstrap_seed=args.seed ^ 0x5A5A5A5A)
    payload = {
        "seed": args.seed,
        "simulation": asdict(sim_cfg),
        "decisions": [asdict(d) for d in decisions],
        "first_pairs": [r.flat_dict() for r in records[: min(10, len(records))]],
    }
    if args.output:
        write_replay(Path(args.output), records)
    if args.decision_log:
        write_decision_log(Path(args.decision_log), records,
                           cell="decode_short_warm")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="run the Monte Carlo comparison")
    simulate.add_argument("--trials", type=int, default=1000)
    simulate.add_argument("--seed", type=int, default=20260831)
    simulate.add_argument("--alpha", type=float, default=0.05)
    simulate.add_argument("--beta", type=float, default=0.05)
    simulate.add_argument("--max-pairs", type=int, default=80)
    simulate.add_argument("--bootstrap-reps", type=int, default=399)
    simulate.add_argument("--deployment-runs", type=int, default=50_000)
    simulate.add_argument("--output-dir", default="scratch/anytime-ab")
    simulate.set_defaults(func=_cmd_simulate)

    find_seed = sub.add_parser("find-replay-seed",
                               help="find a slower-B seed promoted by three medians")
    find_seed.add_argument("--start", type=int, default=0)
    find_seed.add_argument("--limit", type=int, default=100_000)
    find_seed.add_argument("--true-ratio", type=float, default=1.05)
    find_seed.add_argument("--max-pairs", type=int, default=80)
    find_seed.add_argument("--output")
    find_seed.set_defaults(func=_cmd_find_seed)

    replay = sub.add_parser("replay", help="replay one deterministic campaign")
    replay.add_argument("--seed", type=int, required=True)
    replay.add_argument("--true-ratio", type=float, default=1.05)
    replay.add_argument("--max-pairs", type=int, default=80)
    replay.add_argument("--cache-mode", choices=("cold", "warm"), default="cold")
    replay.add_argument("--unguarded", action="store_true")
    replay.add_argument("--output")
    replay.add_argument("--decision-log")
    replay.set_defaults(func=_cmd_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
