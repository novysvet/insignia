#!/usr/bin/env python3
"""Hardware-free synthetic evaluation for DFlash renewal control.

The simulator includes zero-prefix fallback, cache-dependent record cost,
approximation-dependent state transitions, post-verify observations, and one
exact retry from the pre-round snapshot.  It is a sanity check, not a claim
that checked-in point estimates are stationary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import beta as beta_distribution

DRAFT_MS = 17.5
CONTROLLER_MS_R4 = 3.1849
FALLBACK_MS = 643.2
SNAPSHOT_MS = 5.0
RESTORE_MS = 5.0
BATCH_FIXED_MS = 5.0
SEQ_ROW_FIXED_MS = 115.0
RECORD_MIB = 13.5
UNION = np.array([0, 336, 583.8, 833, 1067, 1268.7, 1446.2, 1506.2, 1626.4], float)
SUBSET = np.array([0, 168, 292, 405, 525, 640, 748, 850, 945], float)
EASY = np.array([.9630, .9259, .8889, .8889, .8235, .7647, .7647, .7000])
HARD = np.array([.7581, .6532, .5323, .3952, .3421, .2632, .1316, .0800])


@dataclass(frozen=True)
class Scenario:
    name: str
    base_record_ms: float
    record_sigma: float
    exact_easy_hard: float
    exact_hard_hard: float
    approx_easy_hard: float
    approx_hard_hard: float
    bad_easy: float
    bad_hard: float
    good_beta: tuple[float, float]
    bad_beta: tuple[float, float]
    shift_easy: tuple[float, ...]
    shift_hard: tuple[float, ...]
    drift_record_ms: float | None = None
    adversarial: bool = False


SCENARIOS = (
    Scenario("calibrated", .613, .08, .04, .75, .08, .88, .0005, .008,
             (2, 10), (8, 2),
             (.010, .012, .012, .010, 0, 0, 0, 0),
             (-.010, -.015, -.020, -.025, 0, 0, 0, 0)),
    Scenario("miscalibrated", .82, .18, .07, .84, .18, .95, .003, .030,
             (2, 8), (5, 3),
             (.025, .025, .020, .020, 0, 0, 0, 0),
             (.020, .015, .010, 0, 0, 0, 0, 0), drift_record_ms=1.20),
    Scenario("adversarial", .613, .10, .08, .90, .30, .98, .015, .080,
             (2, 10), (2.2, 9.5),
             (.030, .035, .040, .045, 0, 0, 0, 0),
             (.080, .090, .100, .110, 0, 0, 0, 0), adversarial=True),
)


@dataclass(frozen=True)
class Decision:
    name: str
    width: int
    sequential: bool
    approximate: bool
    retry_threshold: float = math.inf
    predicted_risk: float = 0.0
    predicted_ms_per_accepted: float = math.inf


@dataclass(frozen=True)
class Observation:
    episode_tokens: int
    episode_target: int
    cache: float


@dataclass(frozen=True)
class RoundResult:
    decision: Decision
    accepted: int
    committed: int
    time_ms: float
    record_ms: float
    retried: bool
    harmful_commit: bool
    signal: float | None
    exact_records: float
    next_regime: str


@dataclass(frozen=True)
class Summary:
    scenario: str
    policy: str
    episodes: int
    rounds: int
    accepted_tokens: int
    committed_tokens: int
    elapsed_ms: float
    accepted_tokens_per_second: float
    committed_tokens_per_second: float
    ms_per_committed_token: float
    zero_round_fraction: float
    retry_fraction: float
    approximate_fraction: float
    trajectory_violation_probability: float
    p95_round_ms: float
    mean_prefix: float
    mean_effective_record_ms: float
    max_round_cost_ratio_to_matched_exact: float


class Policy:
    name = "base"
    def reset_episode(self) -> None: pass
    def choose(self, observation: Observation) -> Decision: raise NotImplementedError
    def observe(self, result: RoundResult) -> None: pass


class AlwaysExactR4(Policy):
    name = "always_exact_r4"
    def choose(self, observation: Observation) -> Decision:
        return Decision("exact_batch_r4", 4, False, False)


class StaticApprox(Policy):
    name = "static_nominal_approx_r4"
    def choose(self, observation: Observation) -> Decision:
        return Decision("approx_subset_batch_r4", 4, False, True, .45)


class CostBoundedSubset(Policy):
    name = "cost_bounded_subset_wrapper"
    def choose(self, observation: Observation) -> Decision:
        return Decision("cost_bounded_subset_batch_r4", 4, False, True, .20)


class HazardEstimator:
    def __init__(self, prior: np.ndarray, strength: float = 20) -> None:
        self.success = np.zeros(8); self.failure = np.zeros(8)
        previous = 1.0
        for i, survival in enumerate(prior):
            hazard = float(np.clip(survival / previous, 1e-4, 1-1e-4))
            self.success[i] = strength * hazard
            self.failure[i] = strength * (1-hazard)
            previous = survival

    def update(self, prefix: int, width: int) -> None:
        self.success[:min(prefix, width)] += 1
        if prefix < width: self.failure[prefix] += 1

    def survival(self, conservative: bool = False) -> np.ndarray:
        a, b = self.success, self.failure
        mean = a / (a+b)
        if conservative:
            variance = a*b / ((a+b)**2 * (a+b+1))
            mean = np.clip(mean - 1.64*np.sqrt(variance), .02, .999)
        return np.cumprod(mean)


class AdaptiveExact(Policy):
    name = "adaptive_exact"
    def __init__(self) -> None:
        self.hazards = HazardEstimator(.65*EASY + .35*HARD)
        self.record_ms = .75
        self.round_index = 0

    @staticmethod
    def expected_cost(survival: np.ndarray, width: int, record_ms: float,
                      cache: float, sequential: bool) -> tuple[float, float]:
        s = survival[:width]
        reward, p1 = float(s.sum()), float(s[0])
        common = DRAFT_MS + (1-p1)*FALLBACK_MS
        factor = 1 - .04*cache
        if sequential:
            marginal = np.diff(UNION[:width+1])
            target = float(np.sum(s*(record_ms*factor*marginal + SEQ_ROW_FIXED_MS)))
        else:
            target = p1*(SNAPSHOT_MS+BATCH_FIXED_MS+record_ms*factor*UNION[width])
        return reward, common+target

    def choose(self, observation: Observation) -> Decision:
        self.round_index += 1
        survival = self.hazards.survival(conservative=True)
        choices = []
        for width in (2, 4, 7):
            ba, bt = self.expected_cost(survival, width, self.record_ms, observation.cache, False)
            sa, st = self.expected_cost(survival, width, self.record_ms, observation.cache, True)
            sequential = st < bt
            t, a = (st, sa) if sequential else (bt, ba)
            choices.append(Decision(
                f"exact_{'seq' if sequential else 'batch'}_r{width}",
                width, sequential, False, predicted_ms_per_accepted=t/max(a, 1e-9)))
        if self.round_index % 32 == 0:
            return choices[(self.round_index//32) % len(choices)]
        return min(choices, key=lambda x: x.predicted_ms_per_accepted)

    def observe(self, result: RoundResult) -> None:
        if not result.decision.approximate or result.retried:
            self.hazards.update(result.accepted, result.decision.width)
        self.record_ms = float(np.clip(.92*self.record_ms + .08*result.record_ms, .35, 3.5))


class RiskGatedMPC(AdaptiveExact):
    name = "risk_gated_mpc"
    def __init__(self, alpha: float = .05) -> None:
        super().__init__(); self.alpha = alpha; self.remaining = alpha
        self.ood = False; self.surprise = 0.0; self.predicted: np.ndarray | None = None

    def reset_episode(self) -> None:
        self.remaining = self.alpha; self.ood = False; self.surprise = 0.0; self.predicted = None

    @staticmethod
    def distribution(survival: np.ndarray, width: int) -> np.ndarray:
        s = survival[:width]; p = np.zeros(width+1)
        p[0] = 1-s[0]
        for i in range(1, width): p[i] = s[i-1]-s[i]
        p[width] = s[-1]
        p = np.clip(p, 0, 1); return p/p.sum()

    @staticmethod
    def economic_score_threshold(q_bad: float, posterior_threshold: float) -> float:
        if posterior_threshold <= 0: return 0.0
        if posterior_threshold >= 1: return 1.0
        ab, bb, ag, bg = 8., 2., 2., 10.
        log_beta_bad = math.lgamma(ab)+math.lgamma(bb)-math.lgamma(ab+bb)
        log_beta_good = math.lgamma(ag)+math.lgamma(bg)-math.lgamma(ag+bg)
        target = math.log(posterior_threshold*(1-q_bad)/((1-posterior_threshold)*q_bad))
        lo, hi = 1e-9, 1-1e-9
        for _ in range(28):
            x = .5*(lo+hi)
            log_lr = (log_beta_good-log_beta_bad + (ab-ag)*math.log(x)
                      + (bb-bg)*math.log1p(-x))
            if log_lr >= target: hi = x
            else: lo = x
        return hi

    def choose(self, observation: Observation) -> Decision:
        exact = super().choose(observation)
        survival = self.hazards.survival(conservative=True)
        exact_a, exact_t = self.expected_cost(survival, 4, self.record_ms,
                                              observation.cache, False)
        remaining_tokens = max(1, observation.episode_target-observation.episode_tokens)
        rounds_left = max(1., remaining_tokens/max(exact_a, 1.))
        alpha_round = self.remaining/rounds_left
        q_bad = .00035 + .020*max(0., .90-float(survival[0]))
        if q_bad <= alpha_round: risk_score = 1.0
        else:
            risk_score = float(beta_distribution.ppf(
                np.clip(alpha_round/q_bad, 1e-9, 1-1e-9), 8, 2))
        factor = 1-.04*observation.cache
        marginal_records = max(0., UNION[4]-.90*SUBSET[4])
        marginal_ms = RESTORE_MS+BATCH_FIXED_MS+self.record_ms*factor*marginal_records
        throughput = max(1e-6, exact_a/exact_t)
        posterior_threshold = float(np.clip(throughput*marginal_ms/max(exact_a,1e-9), 0, 1))
        retry_score = min(risk_score, self.economic_score_threshold(q_bad, posterior_threshold))
        bad_cdf = float(beta_distribution.cdf(retry_score, 8, 2))
        good_cdf = float(beta_distribution.cdf(retry_score, 2, 10))
        residual = q_bad*bad_cdf
        retry_probability = q_bad*(1-bad_cdf) + (1-q_bad)*(1-good_cdf)
        approx_survival = np.minimum.accumulate(np.clip(
            survival + np.array([.01, .01, .008, .005, 0, 0, 0, 0]), 0, .999))
        s = approx_survival[:4]; approx_a = float(s.sum()); p1 = float(s[0])
        first_time = (DRAFT_MS+CONTROLLER_MS_R4+(1-p1)*FALLBACK_MS
                      + p1*(SNAPSHOT_MS+BATCH_FIXED_MS+self.record_ms*factor*SUBSET[4]))
        expected_a = (1-retry_probability)*approx_a + retry_probability*exact_a
        expected_t = first_time + retry_probability*marginal_ms
        future_shadow = 140*(.04+2*residual) + .08*self.record_ms*(UNION[4]-SUBSET[4])
        ratio = (expected_t+future_shadow)/max(expected_a, 1e-9)
        # This profile is calibrated only for the fast-resident timing regime.
        timing_ood = not (.42 <= self.record_ms <= .78)
        allow = (not self.ood and not timing_ood and residual <= alpha_round+1e-12
                 and ratio < .99*exact.predicted_ms_per_accepted)
        if allow:
            self.remaining = max(0., self.remaining-residual)
            self.predicted = self.distribution(approx_survival, 4)
            return Decision("risk_gated_subset_batch_r4", 4, False, True,
                            retry_score, residual, ratio)
        self.predicted = self.distribution(survival, exact.width)
        return exact

    def observe(self, result: RoundResult) -> None:
        super().observe(result)
        if self.predicted is not None and result.accepted < len(self.predicted):
            p = max(float(self.predicted[result.accepted]), 1e-12)
            self.surprise = .85*self.surprise + max(0., -math.log(p)-3.5)
            if self.surprise > 2: self.ood = True
        if not (.40 <= result.record_ms <= .95): self.ood = True


POLICIES = {
    cls.name: cls for cls in
    (AlwaysExactR4, StaticApprox, CostBoundedSubset, AdaptiveExact, RiskGatedMPC)
}

def prefix_distribution(survival: np.ndarray, width: int) -> np.ndarray:
    s = np.minimum.accumulate(np.clip(survival[:width], 0, 1))
    p = np.zeros(width+1); p[0] = 1-s[0]
    for i in range(1, width): p[i] = s[i-1]-s[i]
    p[width] = s[-1]; p = np.clip(p, 0, 1)
    return p/p.sum()


def sample_prefix(rng: np.random.Generator, survival: np.ndarray, width: int) -> int:
    return int(rng.choice(np.arange(width+1), p=prefix_distribution(survival, width)))


def effective_record_ms(scenario: Scenario, rng: np.random.Generator,
                        round_index: int, planned_rounds: int) -> float:
    base = scenario.base_record_ms
    if scenario.drift_record_ms is not None and round_index > planned_rounds//2:
        base = scenario.drift_record_ms
    if scenario.adversarial and round_index % 17 in (13, 14, 15, 16):
        base *= 2.4
    return float(base*rng.lognormal(0, scenario.record_sigma))


def survival_for(scenario: Scenario, regime: str, approximate: bool,
                 round_index: int) -> np.ndarray:
    exact = EASY if regime == "easy" else HARD
    if not approximate: return exact
    shift = np.array(scenario.shift_easy if regime == "easy" else scenario.shift_hard)
    result = np.minimum.accumulate(np.clip(exact+shift, .02, .999))
    if scenario.adversarial and round_index % 9 in (6, 7, 8):
        result = np.maximum(result, np.array([.98, .94, .90, .86, 0, 0, 0, 0]))
        result = np.minimum.accumulate(result)
    return result


def run_round(scenario: Scenario, decision: Decision, regime: str, cache: float,
              rng: np.random.Generator, round_index: int,
              planned_rounds: int) -> RoundResult:
    record_ms = effective_record_ms(scenario, rng, round_index, planned_rounds)
    factor = 1-.04*cache
    survival = survival_for(scenario, regime, decision.approximate, round_index)
    first_prefix = sample_prefix(rng, survival, decision.width)
    bad = False; signal = None
    if decision.approximate:
        q = scenario.bad_easy if regime == "easy" else scenario.bad_hard
        bad = bool(first_prefix > 0 and rng.random() < q)
        if first_prefix > 0:
            signal = float(rng.beta(*(scenario.bad_beta if bad else scenario.good_beta)))

    if first_prefix == 0:
        time_ms = DRAFT_MS + (CONTROLLER_MS_R4 if decision.approximate else 0) + FALLBACK_MS
        accepted, committed, retried, harmful = 0, 1, False, False
        exact_records = UNION[1]
    else:
        records = (SUBSET[decision.width] if decision.approximate else
                   UNION[first_prefix] if decision.sequential else UNION[decision.width])
        target = record_ms*factor*records + (
            SEQ_ROW_FIXED_MS*first_prefix if decision.sequential else BATCH_FIXED_MS)
        time_ms = (DRAFT_MS + (CONTROLLER_MS_R4*decision.width/4 if decision.approximate else 0)
                   + (0 if decision.sequential else SNAPSHOT_MS) + target)
        retried = bool(decision.approximate and signal is not None
                       and signal >= decision.retry_threshold)
        if retried:
            exact_survival = survival_for(scenario, regime, False, round_index)
            accepted = max(1, sample_prefix(rng, exact_survival, decision.width))
            marginal = max(0., UNION[decision.width]-.90*SUBSET[decision.width])
            time_ms += RESTORE_MS+BATCH_FIXED_MS+record_ms*factor*marginal
            committed, harmful, exact_records = accepted, False, UNION[decision.width]
        else:
            accepted, committed, harmful = first_prefix, first_prefix, bad
            exact_records = (.90*SUBSET[decision.width]
                             if decision.approximate else records)

    if decision.approximate and not retried:
        p_hard = scenario.approx_easy_hard if regime == "easy" else scenario.approx_hard_hard
        if harmful: p_hard = min(.995, p_hard+.10)
    else:
        p_hard = scenario.exact_easy_hard if regime == "easy" else scenario.exact_hard_hard
    if scenario.adversarial and round_index % 12 in (8, 9, 10, 11):
        p_hard = max(p_hard, .95)
    next_regime = "hard" if rng.random() < p_hard else "easy"
    return RoundResult(decision, accepted, committed, float(time_ms), record_ms,
                       retried, harmful, signal, float(exact_records), next_regime)


def matched_exact_cost(result: RoundResult, cache: float) -> float:
    if result.accepted == 0: return DRAFT_MS+FALLBACK_MS
    return (DRAFT_MS+SNAPSHOT_MS+BATCH_FIXED_MS + result.record_ms*(1-.04*cache)
            * UNION[result.decision.width])


def simulate(scenario: Scenario, policy: Policy, *, episodes: int,
             episode_tokens: int, seed: int) -> tuple[Summary, list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    planned_rounds = max(1, episodes*episode_tokens//3)
    accepted = committed = rounds = violations = zero = retries = approximate = 0
    elapsed = 0.; prefixes: list[int] = []; times: list[float] = []
    record_times: list[float] = []; cost_ratios: list[float] = []; trace = []
    regime, cache = "easy", .20
    for episode in range(episodes):
        policy.reset_episode(); episode_done = 0; episode_bad = False
        regime, cache = "easy", .20
        while episode_done < episode_tokens:
            decision = policy.choose(Observation(episode_done, episode_tokens, cache))
            before = cache
            result = run_round(scenario, decision, regime, cache, rng, rounds, planned_rounds)
            policy.observe(result)
            rounds += 1; accepted += result.accepted; committed += result.committed
            episode_done += result.committed; elapsed += result.time_ms
            prefixes.append(result.accepted); times.append(result.time_ms)
            record_times.append(result.record_ms)
            zero += result.accepted == 0; retries += result.retried
            approximate += decision.approximate; episode_bad |= result.harmful_commit
            cost_ratios.append(result.time_ms/max(matched_exact_cost(result, before), 1e-9))
            load = min(1., result.exact_records/max(UNION[decision.width], 1))
            cache = float(np.clip(.78*cache+.25*load, 0, 1)); regime = result.next_regime
            if len(trace) < 40:
                trace.append({"scenario": scenario.name, "policy": policy.name,
                              "episode": episode, "round": rounds,
                              "action": decision.name, "width": decision.width,
                              "accepted": result.accepted, "committed": result.committed,
                              "time_ms": result.time_ms, "record_ms": result.record_ms,
                              "cache_before": before, "cache_after": cache,
                              "retried": result.retried,
                              "harmful_commit": result.harmful_commit,
                              "signal": result.signal, "next_regime": regime})
        violations += episode_bad
    summary = Summary(
        scenario.name, policy.name, episodes, rounds, accepted, committed, elapsed,
        1000*accepted/elapsed, 1000*committed/elapsed, elapsed/committed,
        zero/rounds, retries/rounds, approximate/rounds, violations/episodes,
        float(np.quantile(times, .95)), float(np.mean(prefixes)),
        float(np.mean(record_times)), float(max(cost_ratios)))
    return summary, trace


def write_csv(path: Path, summaries: list[Summary]) -> None:
    rows = [asdict(x) for x in summaries]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("scratch/dflash-renewal-control"))
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episode-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--policies", nargs="*", choices=tuple(POLICIES),
                        default=list(POLICIES))
    parser.add_argument("--scenarios", nargs="*",
                        choices=tuple(s.name for s in SCENARIOS),
                        default=[s.name for s in SCENARIOS])
    args = parser.parse_args(argv)
    scenario_map = {s.name: s for s in SCENARIOS}
    summaries: list[Summary] = []; traces = []; run_index = 0
    for scenario_name in args.scenarios:
        for policy_name in args.policies:
            summary, trace = simulate(
                scenario_map[scenario_name], POLICIES[policy_name](),
                episodes=args.episodes, episode_tokens=args.episode_tokens,
                seed=args.seed+1009*run_index)
            summaries.append(summary); traces.extend(trace); run_index += 1
            print(f"{summary.scenario:14s} {summary.policy:29s} "
                  f"accepted={summary.accepted_tokens_per_second:6.3f} tok/s "
                  f"committed={summary.ms_per_committed_token:7.2f} ms/tok "
                  f"trajectory_bad={summary.trajectory_violation_probability:7.4f} "
                  f"approx={summary.approximate_fraction:6.3f} "
                  f"retry={summary.retry_fraction:6.3f}")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"summary.csv", summaries)
    (out/"summary.json").write_text(json.dumps([asdict(x) for x in summaries], indent=2)+"\n")
    (out/"trace-sample.jsonl").write_text("".join(json.dumps(x, sort_keys=True)+"\n" for x in traces))
    metadata = {"seed": args.seed, "episodes": args.episodes,
                "episode_tokens": args.episode_tokens, "record_mib": RECORD_MIB,
                "controller_ms_r4": CONTROLLER_MS_R4,
                "historical_point_estimates_are_stationary": False,
                "scenarios": [asdict(scenario_map[x]) for x in args.scenarios]}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

