#!/usr/bin/env python3
"""Finite semi-Markov solver and control helpers for DFlash renewal control.

The exact solver uses time-normalized occupation measures.  It is exact for a
finite communicating SMDP, and for a POMDP after the caller discretizes its
belief state.  Accepted-prefix reward may be zero; no division by an individual
prefix length appears anywhere.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

_EPS = 1e-12


@dataclass(frozen=True)
class Outcome:
    probability: float
    next_state: str
    accepted: float
    committed: float
    time_ms: float
    damage: float = 0.0
    collapse: float = 0.0
    trajectory_end: float = 0.0
    trajectory_violation: float = 0.0


@dataclass(frozen=True)
class Action:
    name: str
    outcomes: tuple[Outcome, ...]

    def expectation(self, field: str) -> float:
        return sum(o.probability * float(getattr(o, field)) for o in self.outcomes)

    def validate(self, states: set[str]) -> None:
        if not self.outcomes:
            raise ValueError(f"action {self.name!r} has no outcomes")
        total = 0.0
        for o in self.outcomes:
            if o.probability < -_EPS:
                raise ValueError(f"negative probability in {self.name!r}")
            if o.next_state not in states:
                raise ValueError(f"unknown next state {o.next_state!r}")
            if o.time_ms <= 0:
                raise ValueError("sojourn times must be positive")
            if min(o.accepted, o.committed, o.damage, o.collapse,
                   o.trajectory_end, o.trajectory_violation) < 0:
                raise ValueError("rewards and quality counters must be non-negative")
            if o.trajectory_violation > o.trajectory_end + _EPS:
                raise ValueError("trajectory_violation exceeds trajectory_end")
            total += o.probability
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"probabilities for {self.name!r} sum to {total}")


@dataclass(frozen=True)
class SemiMarkovModel:
    states: tuple[str, ...]
    actions: Mapping[str, tuple[Action, ...]]
    initial_state: str | None = None

    def validate(self) -> None:
        if not self.states or len(set(self.states)) != len(self.states):
            raise ValueError("states must be non-empty and unique")
        state_set = set(self.states)
        if self.initial_state is not None and self.initial_state not in state_set:
            raise ValueError("initial_state is unknown")
        for state in self.states:
            choices = self.actions.get(state, ())
            if not choices:
                raise ValueError(f"state {state!r} has no actions")
            names = [a.name for a in choices]
            if len(names) != len(set(names)):
                raise ValueError(f"state {state!r} has duplicate action names")
            for action in choices:
                action.validate(state_set)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "states": list(self.states),
            "initial_state": self.initial_state,
            "actions": {
                state: [
                    {"name": a.name, "outcomes": [asdict(o) for o in a.outcomes]}
                    for a in self.actions[state]
                ]
                for state in self.states
            },
        }

    @staticmethod
    def from_json_dict(payload: Mapping[str, Any]) -> "SemiMarkovModel":
        states = tuple(str(s) for s in payload["states"])
        actions: dict[str, tuple[Action, ...]] = {}
        for state in states:
            actions[state] = tuple(
                Action(
                    str(raw["name"]),
                    tuple(Outcome(**outcome) for outcome in raw["outcomes"]),
                )
                for raw in payload["actions"][state]
            )
        model = SemiMarkovModel(states, actions, payload.get("initial_state"))
        model.validate()
        return model


@dataclass(frozen=True)
class PolicyEntry:
    state: str
    action: str
    probability: float
    occupation_per_ms: float


@dataclass(frozen=True)
class SolveResult:
    accepted_tokens_per_second: float
    committed_tokens_per_second: float
    damage_per_accepted_token: float
    collapse_per_accepted_token: float
    trajectory_violation_probability: float
    trajectories_per_second: float
    expected_round_ms: float
    expected_accepted_per_round: float
    expected_committed_per_round: float
    policy: tuple[PolicyEntry, ...]
    raw_occupation: tuple[float, ...]
    variable_labels: tuple[tuple[str, str], ...]
    solver_message: str

    def to_json_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["variable_labels"] = [list(x) for x in self.variable_labels]
        return result


def _flatten(model: SemiMarkovModel):
    model.validate()
    state_index = {s: i for i, s in enumerate(model.states)}
    variables = [(s, a) for s in model.states for a in model.actions[s]]
    n, m = len(variables), len(model.states)
    transition = np.zeros((n, m))
    arrays = {name: np.zeros(n) for name in (
        "accepted", "committed", "time_ms", "damage", "collapse",
        "trajectory_end", "trajectory_violation"
    )}
    for i, (_, action) in enumerate(variables):
        for o in action.outcomes:
            transition[i, state_index[o.next_state]] += o.probability
            for name, values in arrays.items():
                values[i] += o.probability * float(getattr(o, name))
    return variables, transition, arrays


def solve_occupation_lp(
    model: SemiMarkovModel,
    *,
    max_damage_per_accepted_token: float | None = None,
    max_collapse_per_accepted_token: float | None = None,
    max_trajectory_violation_probability: float | None = None,
    objective: str = "accepted",
) -> SolveResult:
    """Solve a finite communicating SMDP by a time-normalized LP.

    x(s,a) is the number of action starts per millisecond.  Constraints are
    flow conservation, sum x E[T]=1, and optional linearized quality ratios.
    A non-communicating input can select any closed recurrent class; restrict it
    to the initial state's reachable communicating class when that matters.
    """

    if objective not in {"accepted", "committed"}:
        raise ValueError("objective must be accepted or committed")
    if max_damage_per_accepted_token is not None and max_damage_per_accepted_token < 0:
        raise ValueError("negative damage budget")
    if max_collapse_per_accepted_token is not None and max_collapse_per_accepted_token < 0:
        raise ValueError("negative collapse budget")
    if (max_trajectory_violation_probability is not None and
            not 0 <= max_trajectory_violation_probability <= 1):
        raise ValueError("trajectory budget must lie in [0,1]")

    variables, transition, a = _flatten(model)
    n_states, n_vars = len(model.states), len(variables)
    state_index = {s: i for i, s in enumerate(model.states)}
    flow = np.zeros((n_states, n_vars))
    for i, (state, _) in enumerate(variables):
        flow[state_index[state], i] += 1
        flow[:, i] -= transition[i]
    A_eq = np.vstack([flow[:-1], a["time_ms"][None, :]])
    b_eq = np.r_[np.zeros(max(0, n_states - 1)), 1.0]

    inequalities, rhs = [], []
    if max_damage_per_accepted_token is not None:
        inequalities.append(a["damage"] - max_damage_per_accepted_token * a["accepted"])
        rhs.append(0.0)
    if max_collapse_per_accepted_token is not None:
        inequalities.append(a["collapse"] - max_collapse_per_accepted_token * a["accepted"])
        rhs.append(0.0)
    if max_trajectory_violation_probability is not None:
        inequalities.append(
            a["trajectory_violation"]
            - max_trajectory_violation_probability * a["trajectory_end"]
        )
        rhs.append(0.0)

    result = linprog(
        -a[objective],
        A_ub=np.vstack(inequalities) if inequalities else None,
        b_ub=np.asarray(rhs) if inequalities else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * n_vars,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"occupation LP failed: {result.message}")
    x = np.asarray(result.x)
    rates = {name: float(x @ values) for name, values in a.items()}
    starts = float(x.sum())
    if starts <= _EPS:
        raise RuntimeError("zero decision-epoch mass")

    policy: list[PolicyEntry] = []
    for state in model.states:
        positions = [i for i, (s, _) in enumerate(variables) if s == state]
        mass = float(x[positions].sum())
        if mass <= 1e-10:
            continue
        for i in positions:
            probability = float(x[i] / mass)
            if probability > 1e-9:
                policy.append(PolicyEntry(state, variables[i][1].name, probability, float(x[i])))

    accepted_rate = rates["accepted"]
    trajectory_rate = rates["trajectory_end"]
    return SolveResult(
        accepted_tokens_per_second=1000 * accepted_rate,
        committed_tokens_per_second=1000 * rates["committed"],
        damage_per_accepted_token=(rates["damage"] / accepted_rate if accepted_rate > _EPS else math.inf),
        collapse_per_accepted_token=(rates["collapse"] / accepted_rate if accepted_rate > _EPS else math.inf),
        trajectory_violation_probability=(
            rates["trajectory_violation"] / trajectory_rate
            if trajectory_rate > _EPS else math.nan
        ),
        trajectories_per_second=1000 * trajectory_rate,
        expected_round_ms=1 / starts,
        expected_accepted_per_round=accepted_rate / starts,
        expected_committed_per_round=rates["committed"] / starts,
        policy=tuple(policy),
        raw_occupation=tuple(float(v) for v in x),
        variable_labels=tuple((s, action.name) for s, action in variables),
        solver_message=str(result.message),
    )


def dinkelbach_value(*, expected_tokens: float, expected_time_ms: float,
                      throughput_tokens_per_ms: float, expected_damage: float = 0,
                      quality_multiplier: float = 0,
                      damage_budget_per_token: float = 0) -> float:
    return ((1 + quality_multiplier * damage_budget_per_token) * expected_tokens
            - throughput_tokens_per_ms * expected_time_ms
            - quality_multiplier * expected_damage)


def retry_threshold(*, throughput_tokens_per_ms: float, marginal_retry_ms: float,
                    bad_state_gain_tokens: float,
                    bad_state_damage_reduction: float = 0,
                    quality_multiplier: float = 0,
                    good_state_gain_tokens: float = 0,
                    good_state_damage_reduction: float = 0) -> float:
    good = good_state_gain_tokens + quality_multiplier * good_state_damage_reduction
    bad = bad_state_gain_tokens + quality_multiplier * bad_state_damage_reduction
    if bad <= good:
        raise ValueError("retry advantage is not increasing in bad-state probability")
    return (throughput_tokens_per_ms * marginal_retry_ms - good) / (bad - good)


def stop_continue_threshold(*, throughput_tokens_per_ms: float,
                            marginal_verify_ms: float,
                            success_gain_tokens: float,
                            success_continuation_value: float = 0,
                            failure_continuation_value: float = 0,
                            stop_value: float = 0,
                            expected_damage: float = 0,
                            quality_multiplier: float = 0) -> float:
    slope = success_gain_tokens + success_continuation_value - failure_continuation_value
    if slope <= 0:
        raise ValueError("continue advantage is not increasing in success probability")
    intercept = (failure_continuation_value - stop_value
                 - throughput_tokens_per_ms * marginal_verify_ms
                 - quality_multiplier * expected_damage)
    return -intercept / slope


def sequential_rows_from_prefix(prefix: int, width: int, *, known_first: bool = True) -> int:
    if width < 1 or not 0 <= prefix <= width:
        raise ValueError("invalid prefix or width")
    return prefix if known_first else min(width, prefix + 1)


def safe_retry_competitive_bound(*, exact_round_ms: float,
                                 controller_snapshot_restore_ms: float,
                                 extra_missing_records: int = 0,
                                 worst_record_ms: float = 0) -> float:
    if exact_round_ms <= 0 or controller_snapshot_restore_ms < 0:
        raise ValueError("invalid cost")
    if extra_missing_records < 0 or worst_record_ms < 0:
        raise ValueError("invalid extra-record cost")
    return 2 + (controller_snapshot_restore_ms
                + extra_missing_records * worst_record_ms) / exact_round_ms


def nonthreshold_counterexample() -> list[dict[str, float | str | bool]]:
    rows = []
    for label, q_bad, retry_ms in (
        ("low", 0.20, 0.10), ("middle", 0.50, 0.80), ("high", 0.80, 0.10)
    ):
        advantage = q_bad - retry_ms
        rows.append({"signal": label, "posterior_bad": q_bad,
                     "marginal_retry_ms": retry_ms,
                     "retry_advantage": advantage, "retry": advantage >= 0})
    return rows


@dataclass(frozen=True)
class DemoScenario:
    name: str = "calibrated"
    record_ms: float = 0.613
    controller_ms: float = 3.1849
    draft_ms: float = 17.5
    fallback_ms: float = 643.2
    snapshot_ms: float = 5.0
    restore_ms: float = 5.0


_CAMPAIGN_K4 = (1/27, 1/27, 1/27, 0, 24/27)
_GSM_K4 = (0.242, 0.105, 0.121, 0.137, 0.395)
_UNION = (0, 336, 583.8, 833, 1067)
_APPROX_UNION = (0, 190, 330, 455, 560)


def _next_states(regime: str, cache: str, approximate: bool):
    hard = (0.05 if regime == "easy" else 0.82) + (0.05 if approximate and regime == "easy" else 0.08 if approximate else 0)
    hard = min(0.98, hard)
    warm = (0.90 if cache == "warm" else 0.72) - (0.08 if approximate else 0)
    return tuple(
        (f"{r}/{c}", pr * pc)
        for r, pr in (("hard", hard), ("easy", 1-hard))
        for c, pc in (("warm", warm), ("cold", 1-warm))
    )


def _action_outcomes(state: str, name: str, scenario: DemoScenario) -> tuple[Outcome, ...]:
    regime, cache = state.split("/")
    approximate, sequential = name.startswith("approx"), "seq" in name
    width = 2 if name.endswith("r2") else 4
    distribution = _CAMPAIGN_K4 if regime == "easy" else _GSM_K4
    if width == 2:
        distribution = (distribution[0], distribution[1], sum(distribution[2:]))
    if approximate:
        distribution = ((0.04, 0.04, 0.06, 0.10, 0.76)
                        if regime == "easy" else
                        (0.28, 0.12, 0.14, 0.16, 0.30))
        bad = 0.004 if regime == "easy" else 0.045
        retry_p = 0.003 if regime == "easy" else 0.030
    else:
        bad, retry_p = 0, 0
    warm_factor = 0.76 if cache == "warm" else 1.0
    outcomes: list[Outcome] = []
    for prefix, p in enumerate(distribution):
        if p <= 0:
            continue
        if prefix == 0:
            duration, committed = scenario.draft_ms + scenario.fallback_ms, 1.0
        else:
            records = (_UNION[prefix] if sequential else
                       _UNION[2] if width == 2 else
                       _APPROX_UNION[4] if approximate else _UNION[4])
            duration = scenario.draft_ms + scenario.controller_ms + scenario.record_ms * warm_factor * records
            if not sequential:
                duration += scenario.snapshot_ms + (scenario.restore_ms if prefix < width else 0)
            if approximate:
                marginal = max(0, _UNION[width] - 0.78 * records)
                duration += retry_p * (scenario.restore_ms + scenario.record_ms * warm_factor * marginal)
            committed = float(prefix)
        damage = bad * (1 + 0.5 * prefix)
        collapse = bad * (0.10 if regime == "easy" else 0.35)
        for next_state, pn in _next_states(regime, cache, approximate):
            outcomes.append(Outcome(p * pn, next_state, float(prefix), committed,
                                    duration, damage, collapse))
    return tuple(outcomes)


def build_demo_model(scenario: DemoScenario | None = None) -> SemiMarkovModel:
    scenario = scenario or DemoScenario()
    states = tuple(f"{r}/{c}" for r in ("easy", "hard") for c in ("cold", "warm"))
    actions = {
        state: tuple(Action(name, _action_outcomes(state, name, scenario)) for name in (
            "exact_batch_r4", "exact_seq_r4", "exact_seq_r2", "approx_batch_r4"
        ))
        for state in states
    }
    model = SemiMarkovModel(states, actions, "easy/cold")
    model.validate()
    return model


def _print_result(result: SolveResult) -> None:
    print(f"accepted throughput : {result.accepted_tokens_per_second:.4f} tok/s")
    print(f"committed throughput: {result.committed_tokens_per_second:.4f} tok/s")
    print(f"damage/accepted     : {result.damage_per_accepted_token:.6g}")
    print(f"collapse/accepted   : {result.collapse_per_accepted_token:.6g}")
    print("trajectory violation: " + (
        f"{result.trajectory_violation_probability:.6g}"
        if math.isfinite(result.trajectory_violation_probability) else "not modeled"
    ))
    print(f"mean round          : {result.expected_round_ms:.3f} ms")
    print(f"accepted/round      : {result.expected_accepted_per_round:.4f}")
    print("policy:")
    for p in result.policy:
        print(f"  {p.state:12s} {p.action:18s} p={p.probability:.6f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--write-demo-model", type=Path)
    parser.add_argument("--max-damage-per-token", type=float, default=0.01)
    parser.add_argument("--max-collapse-per-token", type=float)
    parser.add_argument("--max-trajectory-violation", type=float)
    parser.add_argument("--objective", choices=("accepted", "committed"), default="accepted")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--counterexample", action="store_true")
    args = parser.parse_args(argv)
    if args.counterexample:
        print(json.dumps(nonthreshold_counterexample(), indent=2)); return 0
    demo = build_demo_model()
    if args.write_demo_model:
        args.write_demo_model.parent.mkdir(parents=True, exist_ok=True)
        args.write_demo_model.write_text(json.dumps(demo.to_json_dict(), indent=2) + "\n")
        return 0
    if args.model:
        model = SemiMarkovModel.from_json_dict(json.loads(args.model.read_text()))
    elif args.demo:
        model = demo
    else:
        parser.error("choose --demo, --model, --write-demo-model, or --counterexample")
    result = solve_occupation_lp(
        model,
        max_damage_per_accepted_token=args.max_damage_per_token,
        max_collapse_per_accepted_token=args.max_collapse_per_token,
        max_trajectory_violation_probability=args.max_trajectory_violation,
        objective=args.objective,
    )
    _print_result(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

