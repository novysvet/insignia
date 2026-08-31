#!/usr/bin/env python3
"""Anytime-valid paired performance decisions for hostile benchmark noise.

The default test is deliberately small: turn every guarded A/B pair into a
bounded score in {-1, 0, +1}, then bet a finite mixture of fixed fractions.
It needs no IID timing model, no finite latency moments, and no fixed sample
size.  Its guarantee is conditional: under the null, the next score must have
non-positive conditional mean given everything observed so far.

This module also contains:

* a long-format CSV decision engine for real benchmark logs;
* alpha spending across candidates and change-point epochs;
* a finite Bayesian escalation dynamic program and a computable myopic/EVPI
  bound.  The Bayesian policy chooses measurements; it never replaces the
  frequentist promotion gate.

See ``audits/s11-anytime-wsl-noise.md`` for assumptions and proofs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import secrets
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_LAMBDAS: tuple[float, ...] = tuple(i / 50 for i in range(1, 50))
_EPS = 1e-15


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    m = max(values)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def polynomial_alpha(total_alpha: float, zero_based_index: int) -> float:
    """Summable alpha allocation: 6*alpha/(pi^2*(k+1)^2)."""

    if not 0 < total_alpha < 1:
        raise ValueError("total_alpha must lie in (0, 1)")
    if zero_based_index < 0:
        raise ValueError("index must be non-negative")
    return total_alpha * 6.0 / (math.pi**2 * (zero_based_index + 1) ** 2)


def epoch_alpha(candidate_alpha: float, zero_based_epoch: int) -> float:
    """Spend a candidate's alpha across adaptively started epochs."""

    return polynomial_alpha(candidate_alpha, zero_based_epoch)


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_commitment(secret: bytes) -> str:
    """Commit to a randomization secret without revealing future assignments."""

    if len(secret) < 16:
        raise ValueError("randomization secret must contain at least 16 bytes")
    return hashlib.sha256(b"insignia-anytime-ab-seed-v1\x00" + secret).hexdigest()


@dataclass(frozen=True)
class RandomizationAssignment:
    order: str
    randomization_u64: str
    seed_commitment: str
    protocol_hash: str
    message_sha256: str


def derive_randomization(
    secret: bytes,
    *,
    protocol_hash: str,
    candidate_id: str,
    epoch: int,
    pair_id: str,
    cell: str,
    case_id: str,
) -> RandomizationAssignment:
    """Derive one replayable fair order after the case and cell are locked.

    A trusted scheduler keeps ``secret`` unavailable to the experimenter until
    the campaign closes.  The HMAC input binds the draw to the exact protocol,
    candidate, epoch, pair, metric/state cell, and benchmark case.  Given a
    uniformly random secret, the least-significant digest bit is a fair coin.
    """

    if len(secret) < 16:
        raise ValueError("randomization secret must contain at least 16 bytes")
    if len(protocol_hash) != 64 or any(c not in "0123456789abcdef" for c in protocol_hash):
        raise ValueError("protocol_hash must be a lowercase SHA-256 digest")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    fields = {
        "candidate_id": candidate_id,
        "case_id": case_id,
        "cell": cell,
        "epoch": epoch,
        "pair_id": pair_id,
        "protocol_hash": protocol_hash,
        "version": "insignia-anytime-ab-randomization-v1",
    }
    if any(not str(fields[name]) for name in ("candidate_id", "case_id", "cell", "pair_id")):
        raise ValueError("candidate_id, case_id, cell, and pair_id must be non-empty")
    message = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(secret, message, hashlib.sha256).digest()
    draw = int.from_bytes(digest[:8], "big")
    return RandomizationAssignment(
        order="AB" if draw & 1 == 0 else "BA",
        randomization_u64=f"{draw:016x}",
        seed_commitment=seed_commitment(secret),
        protocol_hash=protocol_hash,
        message_sha256=hashlib.sha256(message).hexdigest(),
    )


def comparison_score(value: float, threshold: float, *, lower_is_better: bool) -> float:
    """Return +1 for evidence in the requested direction, -1 against, 0 on tie.

    Exact ties are neutral.  No tolerance is used because a tolerance silently
    changes the estimand; callers that need timer-quantization handling should
    predeclare and log an explicit randomized tie breaker.
    """

    if not (math.isfinite(value) and math.isfinite(threshold)):
        raise ValueError("comparison values must be finite")
    if value == threshold:
        return 0.0
    if lower_is_better:
        return 1.0 if value < threshold else -1.0
    return 1.0 if value > threshold else -1.0


def bounded_log_ratio_score(
    oriented_ratio: float,
    threshold_ratio: float,
    *,
    minimum_value: float,
    maximum_value: float,
) -> float:
    """Score a capped average log-ratio claim in [-1, 1].

    Each arm value must have been deterministically mapped into the predeclared
    interval ``[minimum_value, maximum_value]``; timeouts are recorded at the
    cap rather than deleted.  Let D=log(oriented_ratio), delta=log(threshold),
    and K=log(maximum/minimum).  The returned score is

        (delta - D) / (K + |delta|).

    It lies in [-1, 1].  Under the conditional null E[D | past] >= delta, its
    conditional mean is non-positive, so it can be fed to
    ``MixtureBettingEProcess``.  The estimand is a capped average log effect,
    not an uncapped latency mean or a median.
    """

    if min(oriented_ratio, threshold_ratio, minimum_value, maximum_value) <= 0:
        raise ValueError("ratios and bounds must be positive")
    if not minimum_value < maximum_value:
        raise ValueError("minimum_value must be below maximum_value")
    if not all(
        math.isfinite(x)
        for x in (oriented_ratio, threshold_ratio, minimum_value, maximum_value)
    ):
        raise ValueError("ratios and bounds must be finite")
    k = math.log(maximum_value / minimum_value)
    delta = math.log(threshold_ratio)
    if abs(delta) > k + 1e-12:
        raise ValueError("threshold ratio lies outside the bounded outcome range")
    score = (delta - math.log(oriented_ratio)) / (k + abs(delta))
    if score < -1 - 1e-12 or score > 1 + 1e-12:
        raise ValueError("oriented ratio is inconsistent with the declared bounds")
    return min(1.0, max(-1.0, score))


@dataclass
class MixtureBettingEProcess:
    """Finite mixture of fixed-fraction test supermartingales.

    For scores X_t in [-1, 1] satisfying E[X_t | F_{t-1}] <= 0, every fixed
    lambda in [0, 1) gives capital product_t (1 + lambda X_t).  A convex
    mixture remains a nonnegative supermartingale.  Crossing 1/alpha therefore
    has probability at most alpha at any stopping time.
    """

    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    weights: tuple[float, ...] | None = None
    log_capitals: list[float] = field(init=False)
    count: int = field(default=0, init=False)
    positive: int = field(default=0, init=False)
    negative: int = field(default=0, init=False)
    ties: int = field(default=0, init=False)
    max_log_e: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not self.lambdas:
            raise ValueError("at least one betting fraction is required")
        if any(not 0 <= lam < 1 for lam in self.lambdas):
            raise ValueError("betting fractions must lie in [0, 1)")
        if self.weights is None:
            self.weights = tuple([1.0 / len(self.lambdas)] * len(self.lambdas))
        if len(self.weights) != len(self.lambdas):
            raise ValueError("weights and lambdas must have equal length")
        if any(w < 0 for w in self.weights):
            raise ValueError("mixture weights must be non-negative")
        total = sum(self.weights)
        if total <= 0:
            raise ValueError("mixture weights must have positive sum")
        self.weights = tuple(w / total for w in self.weights)
        self.log_capitals = [0.0] * len(self.lambdas)

    def update(self, score: float) -> float:
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("score must lie in [-1, 1]")
        self.count += 1
        if score > 0:
            self.positive += 1
        elif score < 0:
            self.negative += 1
        else:
            self.ties += 1
        for j, lam in enumerate(self.lambdas):
            factor = 1.0 + lam * score
            if factor < 0:
                raise ArithmeticError("negative betting factor")
            if factor == 0:
                self.log_capitals[j] = -math.inf
            elif not math.isinf(self.log_capitals[j]):
                self.log_capitals[j] += math.log(factor)
        log_e = self.log_e_value
        self.max_log_e = max(self.max_log_e, log_e)
        return math.exp(log_e) if log_e < 709 else math.inf

    @property
    def log_e_value(self) -> float:
        terms = []
        assert self.weights is not None
        for weight, capital in zip(self.weights, self.log_capitals):
            if weight > 0:
                terms.append(math.log(weight) + capital)
        return _logsumexp(terms)

    @property
    def e_value(self) -> float:
        log_e = self.log_e_value
        return math.exp(log_e) if log_e < 709 else math.inf

    @property
    def max_e_value(self) -> float:
        return math.exp(self.max_log_e) if self.max_log_e < 709 else math.inf

    def crossed(self, alpha: float) -> bool:
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        return self.max_log_e >= math.log(1.0 / alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "positive": self.positive,
            "negative": self.negative,
            "ties": self.ties,
            "e_value": self.e_value,
            "max_e_value": self.max_e_value,
        }


def normalize_anchor(value: float, lower_bound: float, upper_bound: float) -> float:
    """Map a predeclared bounded anchor metric to [0, 1].

    Values outside the physical/logging range are winsorized rather than
    deleted.  The bounds must be chosen before observing the campaign.
    """

    if not all(math.isfinite(x) for x in (value, lower_bound, upper_bound)):
        raise ValueError("anchor value and bounds must be finite")
    if not lower_bound < upper_bound:
        raise ValueError("anchor lower bound must be below its upper bound")
    return min(1.0, max(0.0, (value - lower_bound) / (upper_bound - lower_bound)))


@dataclass
class TwoSidedAnchorAlarm:
    """Anytime-valid detector for departure from a bounded mean envelope.

    The normalized anchor W_t must lie in [0, 1].  Under a stable epoch the
    declared null is

        lower_mean <= E[W_t | F_{t-1}] <= upper_mean.

    The upward and downward bets each receive half of ``alpha``.  A crossing
    is therefore a level-alpha anytime alarm by Ville's inequality and a union
    bound.  Detection power is not guaranteed against an arbitrary change.
    """

    lower_mean: float
    upper_mean: float
    alpha: float
    upward: MixtureBettingEProcess = field(default_factory=MixtureBettingEProcess)
    downward: MixtureBettingEProcess = field(default_factory=MixtureBettingEProcess)

    def __post_init__(self) -> None:
        if not 0 <= self.lower_mean <= self.upper_mean <= 1:
            raise ValueError("mean envelope must lie in [0, 1]")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")

    @staticmethod
    def _scale(center: float) -> float:
        return max(center, 1.0 - center)

    def update(self, normalized_value: float) -> bool:
        if not math.isfinite(normalized_value) or not 0 <= normalized_value <= 1:
            raise ValueError("normalized anchor value must lie in [0, 1]")
        upward_score = (
            (normalized_value - self.upper_mean) / self._scale(self.upper_mean)
        )
        downward_score = (
            (self.lower_mean - normalized_value) / self._scale(self.lower_mean)
        )
        self.upward.update(upward_score)
        self.downward.update(downward_score)
        return self.crossed

    @property
    def crossed(self) -> bool:
        return self.upward.crossed(self.alpha / 2.0) or self.downward.crossed(
            self.alpha / 2.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_mean": self.lower_mean,
            "upper_mean": self.upper_mean,
            "alpha": self.alpha,
            "crossed": self.crossed,
            "upward": self.upward.to_dict(),
            "downward": self.downward.to_dict(),
            "one_sided_threshold": 2.0 / self.alpha,
        }


@dataclass(frozen=True)
class CellRule:
    """A single metric/state requirement in oriented-ratio form.

    For ``direction='lower'``, oriented_ratio = B/A.  For ``direction='higher'``,
    oriented_ratio = A/B, so smaller is always better after orientation.
    ``promote_ratio`` may encode improvement (0.98) or non-inferiority (1.02).
    """

    name: str
    stage: str
    metric: str
    direction: str
    promote_ratio: float
    reject_ratio: float
    required_for_promotion: bool = True
    can_reject: bool = True
    min_pairs: int = 12
    max_pairs: int = 80
    reject_weight: float = 1.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("cell name is empty")
        if self.direction not in {"lower", "higher"}:
            raise ValueError(f"unknown direction {self.direction!r}")
        if min(self.promote_ratio, self.reject_ratio) <= 0:
            raise ValueError("ratios must be positive")
        if self.promote_ratio >= self.reject_ratio:
            raise ValueError("promote_ratio must be below reject_ratio after orientation")
        if not 1 <= self.min_pairs <= self.max_pairs:
            raise ValueError("invalid min/max pair limits")
        if self.reject_weight < 0:
            raise ValueError("reject_weight must be non-negative")

    def orient(self, a_value: float, b_value: float) -> float:
        if not (math.isfinite(a_value) and math.isfinite(b_value)):
            raise ValueError("metric values must be finite")
        if a_value <= 0 or b_value <= 0:
            raise ValueError("ratio metrics must be strictly positive")
        return b_value / a_value if self.direction == "lower" else a_value / b_value


@dataclass(frozen=True)
class ProtocolConfig:
    version: str
    candidate_id: str
    candidate_alpha: float
    reject_beta: float
    max_campaign_seconds: float
    cells: tuple[CellRule, ...]
    plausible_payback_seconds: float | None = None
    program_alpha: float | None = None
    candidate_index: int | None = None
    allowed_nonmeasurement_codes: tuple[str, ...] = (
        "prestart_infrastructure_fault",
        "timer_unavailable",
        "operator_abort_before_measurement",
    )
    randomization_rule: str = (
        "Lock stage, cell, case, prep recipe, and epoch; then draw an HMAC-SHA256 "
        "fair AB/BA coin from a seed whose SHA-256 commitment was recorded before "
        "the campaign. Reveal the seed only after closure."
    )
    cold_cache_rule: str = (
        "Before each measured arm independently: restart the engine process, clear "
        "engine-owned caches, run the fixed page-cache scrub recipe, verify the "
        "declared counters, then measure exactly once. Record timeouts at the cap and "
        "pad to the fixed slot boundary. Never use arm 1 to prepare arm 2."
    )
    warm_cache_rule: str = (
        "Before each measured arm independently: apply the same neutralizer, execute "
        "the predeclared untimed self-warm workload for that arm, then measure exactly "
        "once. Record timeouts at the cap, pad to the fixed slot boundary, and charge "
        "prep and warmup wall time to campaign cost."
    )
    required_log_fields: tuple[str, ...] = (
        "protocol_hash",
        "candidate_id",
        "baseline_commit",
        "candidate_commit",
        "pair_id",
        "epoch",
        "stage",
        "cell",
        "case_id",
        "case_selection_probability",
        "case_locked_at",
        "randomization_u64",
        "order",
        "seed_commitment",
        "prep_recipe_hash",
        "prep_ok",
        "warmup_count",
        "warmup_wall_seconds",
        "a_value",
        "b_value",
        "a_prefill_ms",
        "b_prefill_ms",
        "a_decode_ms",
        "b_decode_ms",
        "a_committed_tokens",
        "b_committed_tokens",
        "expert_bytes",
        "acceptance",
        "hardware_counters",
        "parity_ok",
        "validity_code",
        "cost_seconds",
        "host_fingerprint",
        "wsl_kernel",
        "gpu_driver",
        "cuda_version",
        "gpu_clocks",
        "gpu_temperature",
        "host_load",
        "page_faults",
        "disk_bytes",
        "change_alarm_state",
    )
    force_full_campaign_when: tuple[str, ...] = (
        "candidate survives the cheap screens and promotion remains economically plausible",
        "candidate changes floating-point order, routing, cache residency, "
        "asynchronous scheduling, or I/O",
        "screening effects are heterogeneous across prompt length or cache state",
        "multiple optimizations interact or a selected factorial interaction is nonzero",
        "screening evidence lies near a promotion or rejection boundary",
        "a clock, driver, host-load, cache-prep, or anchor change alarm starts a new epoch",
        "the maximum plausible deployment savings exceeds the remaining full-campaign cost",
    )

    def validate(self) -> None:
        if not self.version or not self.candidate_id:
            raise ValueError("version and candidate_id are required")
        if not 0 < self.candidate_alpha < 1:
            raise ValueError("candidate_alpha must lie in (0, 1)")
        if not 0 < self.reject_beta < 1:
            raise ValueError("reject_beta must lie in (0, 1)")
        if self.max_campaign_seconds <= 0:
            raise ValueError("max campaign time must be positive")
        if (
            self.plausible_payback_seconds is not None
            and self.plausible_payback_seconds <= 0
        ):
            raise ValueError("plausible payback must be positive when supplied")
        if (self.program_alpha is None) != (self.candidate_index is None):
            raise ValueError("program_alpha and candidate_index must be supplied together")
        if self.program_alpha is not None:
            if not 0 < self.program_alpha < 1:
                raise ValueError("program_alpha must lie in (0, 1)")
            assert self.candidate_index is not None
            if self.candidate_index < 0:
                raise ValueError("candidate_index must be non-negative")
            allocated = polynomial_alpha(self.program_alpha, self.candidate_index)
            if not math.isclose(self.candidate_alpha, allocated, rel_tol=1e-12):
                raise ValueError(
                    "candidate_alpha does not match the declared program allocation"
                )
        if not self.cells:
            raise ValueError("at least one cell is required")
        if not self.randomization_rule or not self.cold_cache_rule or not self.warm_cache_rule:
            raise ValueError("operational randomization/cache rules must be non-empty")
        if not self.required_log_fields or len(set(self.required_log_fields)) != len(
            self.required_log_fields
        ):
            raise ValueError("required log fields must be non-empty and unique")
        names = [c.name for c in self.cells]
        if len(names) != len(set(names)):
            raise ValueError("cell names must be unique")
        for cell in self.cells:
            cell.validate()
        if not any(c.required_for_promotion for c in self.cells):
            raise ValueError("at least one promotion cell is required")

    @property
    def by_name(self) -> dict[str, CellRule]:
        return {c.name: c for c in self.cells}

    def beta_by_cell(self) -> dict[str, float]:
        active = [c for c in self.cells if c.can_reject and c.reject_weight > 0]
        total = sum(c.reject_weight for c in active)
        if total <= 0:
            return {}
        return {c.name: self.reject_beta * c.reject_weight / total for c in active}

    @property
    def measurement_cap_seconds(self) -> float:
        """Hard stop: engineering cap intersected with plausible repayment."""

        if self.plausible_payback_seconds is None:
            return self.max_campaign_seconds
        return min(self.max_campaign_seconds, self.plausible_payback_seconds)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cells"] = [asdict(c) for c in self.cells]
        return payload

    @staticmethod
    def from_json_dict(payload: Mapping[str, Any]) -> "ProtocolConfig":
        config = ProtocolConfig(
            version=str(payload["version"]),
            candidate_id=str(payload["candidate_id"]),
            candidate_alpha=float(payload["candidate_alpha"]),
            reject_beta=float(payload["reject_beta"]),
            max_campaign_seconds=float(payload["max_campaign_seconds"]),
            cells=tuple(CellRule(**raw) for raw in payload["cells"]),
            plausible_payback_seconds=(
                None
                if payload.get("plausible_payback_seconds") is None
                else float(payload["plausible_payback_seconds"])
            ),
            program_alpha=(
                None
                if payload.get("program_alpha") is None
                else float(payload["program_alpha"])
            ),
            candidate_index=(
                None
                if payload.get("candidate_index") is None
                else int(payload["candidate_index"])
            ),
            allowed_nonmeasurement_codes=tuple(
                payload.get("allowed_nonmeasurement_codes", (
                    "prestart_infrastructure_fault",
                    "timer_unavailable",
                    "operator_abort_before_measurement",
                ))
            ),
            randomization_rule=str(
                payload.get(
                    "randomization_rule",
                    "Lock the case before drawing a fair AB/BA assignment.",
                )
            ),
            cold_cache_rule=str(
                payload.get(
                    "cold_cache_rule",
                    "Apply the same predeclared neutralizer before each arm.",
                )
            ),
            warm_cache_rule=str(
                payload.get(
                    "warm_cache_rule",
                    "Self-warm each arm independently with a fixed workload.",
                )
            ),
            required_log_fields=tuple(
                payload.get("required_log_fields", ("pair_id", "cell", "case_id", "order"))
            ),
            force_full_campaign_when=tuple(payload.get("force_full_campaign_when", ())),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class PairObservation:
    pair_id: str
    epoch: int
    cell: str
    case_id: str
    order: str
    a_value: float | None
    b_value: float | None
    parity_ok: bool
    prep_ok: bool
    validity_code: str
    cost_seconds: float
    randomization_u64: str = ""
    protocol_hash: str = ""
    candidate_id: str = ""
    stage: str = ""
    seed_commitment: str = ""
    case_selection_probability: float | None = None
    baseline_commit: str = ""
    candidate_commit: str = ""

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"cannot parse boolean {value!r}")

    @staticmethod
    def _parse_optional_float(value: Any) -> float | None:
        if value is None or str(value).strip() == "":
            return None
        return float(value)

    @staticmethod
    def from_mapping(raw: Mapping[str, Any]) -> "PairObservation":
        return PairObservation(
            pair_id=str(raw["pair_id"]),
            epoch=int(raw.get("epoch", 0)),
            cell=str(raw["cell"]),
            case_id=str(raw.get("case_id", "")),
            order=str(raw["order"]).upper(),
            a_value=PairObservation._parse_optional_float(raw.get("a_value")),
            b_value=PairObservation._parse_optional_float(raw.get("b_value")),
            parity_ok=PairObservation._parse_bool(raw.get("parity_ok", True)),
            prep_ok=PairObservation._parse_bool(raw.get("prep_ok", True)),
            validity_code=str(raw.get("validity_code", "ok")),
            cost_seconds=float(raw.get("cost_seconds", 0.0)),
            randomization_u64=str(raw.get("randomization_u64", "")),
            protocol_hash=str(raw.get("protocol_hash", "")),
            candidate_id=str(raw.get("candidate_id", "")),
            stage=str(raw.get("stage", "")),
            seed_commitment=str(raw.get("seed_commitment", "")),
            case_selection_probability=PairObservation._parse_optional_float(
                raw.get("case_selection_probability")
            ),
            baseline_commit=str(raw.get("baseline_commit", "")),
            candidate_commit=str(raw.get("candidate_commit", "")),
        )


@dataclass
class CellState:
    rule: CellRule
    promote: MixtureBettingEProcess = field(default_factory=MixtureBettingEProcess)
    reject: MixtureBettingEProcess = field(default_factory=MixtureBettingEProcess)
    ratios: list[float] = field(default_factory=list)

    def update(self, ratio: float) -> None:
        self.ratios.append(ratio)
        self.promote.update(comparison_score(
            ratio, self.rule.promote_ratio, lower_is_better=True))
        self.reject.update(comparison_score(
            ratio, self.rule.reject_ratio, lower_is_better=False))

    @property
    def count(self) -> int:
        return len(self.ratios)


class DecisionEngine:
    """Replayable promotion decision over a long-format pair log.

    Evidence is kept separate by epoch.  An alarm raised after pair t may start
    epoch e+1 at pair t+1; promotion from epoch e uses the summable allocation
    ``epoch_alpha(candidate_alpha, e)``.  Old observations are never deleted or
    silently reassigned to the new regime.
    """

    def __init__(self, config: ProtocolConfig):
        config.validate()
        self.config = config
        self.states_by_epoch: dict[int, dict[str, CellState]] = {}
        self.seen: set[tuple[int, str, str]] = set()
        self.total_cost_seconds = 0.0
        self.parity_failed = False
        self.invalid_reasons: list[str] = []
        self.excluded_records = 0
        self.records_processed = 0
        self.post_terminal_records = 0
        self.first_terminal_status: str | None = None
        self.first_terminal_after_records: int | None = None
        self.first_terminal_epoch: int | None = None
        self.bound_identity: dict[str, str] = {}
        self.last_epoch_seen = -1

    def _epoch_states(self, epoch: int) -> dict[str, CellState]:
        if epoch not in self.states_by_epoch:
            self.states_by_epoch[epoch] = {
                c.name: CellState(c) for c in self.config.cells
            }
        return self.states_by_epoch[epoch]

    def _beta_by_cell_epoch(self, epoch: int) -> dict[str, float]:
        base = self.config.beta_by_cell()
        epoch_total = epoch_alpha(self.config.reject_beta, epoch)
        # beta_by_cell was normalized to reject_beta; rescale its weights to
        # the epoch's summable allocation.
        return {
            name: epoch_total * value / self.config.reject_beta
            for name, value in base.items()
        }

    def invalidate(self, reason: str) -> None:
        self.invalid_reasons.append(reason)

    def _bind_identity(self, field_name: str, value: str) -> None:
        if not value:
            return
        prior = self.bound_identity.get(field_name)
        if prior is None:
            self.bound_identity[field_name] = value
        elif prior != value:
            self.invalid_reasons.append(
                f"log identity changed for {field_name}: {prior!r} -> {value!r}"
            )

    def _automatic_terminal_status(self) -> tuple[str | None, int | None]:
        if self.parity_failed:
            return "REJECT", None
        for epoch in sorted(self.states_by_epoch):
            states = self.states_by_epoch[epoch]
            if self._epoch_rejects(epoch, states):
                return "REJECT", epoch
        for epoch in sorted(self.states_by_epoch):
            states = self.states_by_epoch[epoch]
            if self._epoch_promotes(epoch, states):
                return "PROMOTE", epoch
        exhausted = self.total_cost_seconds >= self.config.measurement_cap_seconds
        for states in self.states_by_epoch.values():
            required = [s for s in states.values() if s.rule.required_for_promotion]
            exhausted = exhausted or any(
                state.count >= state.rule.max_pairs for state in required
            )
        return ("INCONCLUSIVE", None) if exhausted else (None, None)

    def _freeze_first_terminal(self) -> None:
        if self.first_terminal_status is not None or self.invalid_reasons:
            return
        status, epoch = self._automatic_terminal_status()
        if status is not None:
            self.first_terminal_status = status
            self.first_terminal_after_records = self.records_processed
            self.first_terminal_epoch = epoch

    def update(self, observation: PairObservation) -> None:
        if self.first_terminal_status is not None:
            self.post_terminal_records += 1
            return
        self.records_processed += 1
        self._bind_identity("protocol_hash", observation.protocol_hash)
        self._bind_identity("candidate_id", observation.candidate_id)
        self._bind_identity("seed_commitment", observation.seed_commitment)
        self._bind_identity("baseline_commit", observation.baseline_commit)
        self._bind_identity("candidate_commit", observation.candidate_commit)
        if observation.candidate_id and observation.candidate_id != self.config.candidate_id:
            self.invalid_reasons.append(
                f"pair {observation.pair_id} candidate_id {observation.candidate_id!r} "
                f"does not match protocol {self.config.candidate_id!r}"
            )
            return
        if observation.epoch < 0:
            self.invalid_reasons.append(f"negative epoch for pair {observation.pair_id}")
            return
        if observation.epoch < self.last_epoch_seen:
            self.invalid_reasons.append(
                f"epoch moved backward from {self.last_epoch_seen} to {observation.epoch}"
            )
            return
        self.last_epoch_seen = observation.epoch
        if not observation.pair_id:
            self.invalid_reasons.append("empty pair_id")
            return
        if not observation.case_id:
            self.invalid_reasons.append(
                f"pair {observation.pair_id} has an empty case_id"
            )
            return
        key = (observation.epoch, observation.cell, observation.pair_id)
        if key in self.seen:
            self.invalid_reasons.append(f"duplicate pair row {key}")
            return
        self.seen.add(key)
        if observation.cell not in self.config.by_name:
            self.invalid_reasons.append(f"unknown cell {observation.cell!r}")
            return
        expected_stage = self.config.by_name[observation.cell].stage
        if observation.stage and observation.stage != expected_stage:
            self.invalid_reasons.append(
                f"pair {observation.pair_id} stage {observation.stage!r} "
                f"does not match cell stage {expected_stage!r}"
            )
            return
        if (
            observation.case_selection_probability is not None
            and not 0 < observation.case_selection_probability <= 1
        ):
            self.invalid_reasons.append(
                f"pair {observation.pair_id} has invalid case selection probability"
            )
            return
        if observation.order not in {"AB", "BA"}:
            self.invalid_reasons.append(
                f"pair {observation.pair_id} has invalid order {observation.order!r}")
            return
        if observation.cost_seconds < 0 or not math.isfinite(observation.cost_seconds):
            self.invalid_reasons.append(f"pair {observation.pair_id} has invalid cost")
            return
        self.total_cost_seconds += observation.cost_seconds
        if not observation.parity_ok:
            self.parity_failed = True
            self._freeze_first_terminal()
            return
        if not observation.prep_ok:
            self.invalid_reasons.append(
                f"pair {observation.pair_id} failed the declared cache preparation")
            return
        if observation.validity_code != "ok":
            if observation.validity_code not in self.config.allowed_nonmeasurement_codes:
                self.invalid_reasons.append(
                    f"pair {observation.pair_id} has undeclared exclusion code "
                    f"{observation.validity_code!r}")
                return
            if observation.a_value is not None or observation.b_value is not None:
                self.invalid_reasons.append(
                    f"pair {observation.pair_id} was excluded after a metric was observed")
                return
            self.excluded_records += 1
            self._freeze_first_terminal()
            return
        if observation.a_value is None or observation.b_value is None:
            self.invalid_reasons.append(
                f"pair {observation.pair_id} is marked ok but lacks both arm values")
            return
        states = self._epoch_states(observation.epoch)
        try:
            ratio = states[observation.cell].rule.orient(
                observation.a_value, observation.b_value)
        except ValueError as exc:
            self.invalid_reasons.append(f"pair {observation.pair_id}: {exc}")
            return
        states[observation.cell].update(ratio)
        self._freeze_first_terminal()

    def _epoch_rejects(self, epoch: int, states: Mapping[str, CellState]) -> bool:
        beta = self._beta_by_cell_epoch(epoch)
        return any(
            name in beta and state.reject.crossed(beta[name])
            for name, state in states.items()
        )

    def _epoch_promotes(self, epoch: int, states: Mapping[str, CellState]) -> bool:
        required = [s for s in states.values() if s.rule.required_for_promotion]
        alpha = epoch_alpha(self.config.candidate_alpha, epoch)
        return (
            all(s.count >= s.rule.min_pairs for s in required)
            and all(s.promote.crossed(alpha) for s in required)
        )

    def status(self, *, campaign_closed: bool = False) -> str:
        if self.invalid_reasons:
            return "INVALID"
        if self.parity_failed:
            return "REJECT"
        if self.first_terminal_status is not None:
            return self.first_terminal_status
        if campaign_closed:
            return "INCONCLUSIVE"
        return "CONTINUE"

    @staticmethod
    def _median(values: Sequence[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        n = len(ordered)
        if n % 2:
            return ordered[n // 2]
        return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])

    def snapshot(self, *, campaign_closed: bool = False) -> dict[str, Any]:
        epochs: dict[str, Any] = {}
        for epoch in sorted(self.states_by_epoch):
            states = self.states_by_epoch[epoch]
            beta = self._beta_by_cell_epoch(epoch)
            alpha = epoch_alpha(self.config.candidate_alpha, epoch)
            epochs[str(epoch)] = {
                "promotion_alpha": alpha,
                "epoch_promotes": self._epoch_promotes(epoch, states),
                "epoch_rejects": self._epoch_rejects(epoch, states),
                "cells": {
                    name: {
                        "stage": state.rule.stage,
                        "metric": state.rule.metric,
                        "direction": state.rule.direction,
                        "pairs": state.count,
                        "promote_ratio": state.rule.promote_ratio,
                        "reject_ratio": state.rule.reject_ratio,
                        "required_for_promotion": state.rule.required_for_promotion,
                        "promote": state.promote.to_dict(),
                        "promote_threshold": 1.0 / alpha,
                        "reject": state.reject.to_dict(),
                        "reject_threshold": (1.0 / beta[name]) if name in beta else None,
                        "median_oriented_ratio": self._median(state.ratios),
                    }
                    for name, state in states.items()
                },
            }
        return {
            "candidate_id": self.config.candidate_id,
            "status": self.status(campaign_closed=campaign_closed),
            "candidate_alpha": self.config.candidate_alpha,
            "reject_beta": self.config.reject_beta,
            "measurement_cap_seconds": self.config.measurement_cap_seconds,
            "total_cost_seconds": self.total_cost_seconds,
            "excluded_records": self.excluded_records,
            "records_processed": self.records_processed,
            "post_terminal_records_ignored": self.post_terminal_records,
            "first_terminal_status": self.first_terminal_status,
            "first_terminal_after_records": self.first_terminal_after_records,
            "first_terminal_epoch": self.first_terminal_epoch,
            "last_epoch_seen": self.last_epoch_seen,
            "parity_failed": self.parity_failed,
            "invalid_reasons": list(self.invalid_reasons),
            "bound_identity": dict(self.bound_identity),
            "epochs": epochs,
        }


def default_protocol(
    candidate_id: str = "candidate",
    *,
    candidate_alpha: float = 0.01,
    program_alpha: float | None = None,
    candidate_index: int | None = None,
) -> ProtocolConfig:
    """Example full-model protocol for a decode-oriented optimization.

    The two decode cells claim at least a 2% median improvement.  The prefill
    cells are 2% non-inferiority guardrails.  Cheap fixture/short-run screens
    are not permitted to promote the candidate.
    """

    config = ProtocolConfig(
        version="insignia-anytime-ab-v1",
        candidate_id=candidate_id,
        candidate_alpha=candidate_alpha,
        reject_beta=0.05,
        max_campaign_seconds=4 * 3600,
        plausible_payback_seconds=6 * 3600,
        program_alpha=program_alpha,
        candidate_index=candidate_index,
        cells=(
            CellRule(
                name="kernel_fixture",
                stage="fixture",
                metric="fixture_ns_per_item",
                direction="lower",
                promote_ratio=0.98,
                reject_ratio=1.02,
                required_for_promotion=False,
                min_pairs=6,
                max_pairs=20,
                reject_weight=0.5,
            ),
            CellRule(
                name="short_decode_warm_screen",
                stage="short_model",
                metric="decode_ms_per_committed_token",
                direction="lower",
                promote_ratio=0.98,
                reject_ratio=1.02,
                required_for_promotion=False,
                min_pairs=6,
                max_pairs=24,
                reject_weight=0.5,
            ),
            CellRule(
                name="decode_short_warm",
                stage="full",
                metric="decode_ms_per_committed_token",
                direction="lower",
                promote_ratio=0.98,
                reject_ratio=1.02,
                min_pairs=12,
                max_pairs=80,
                reject_weight=1.0,
            ),
            CellRule(
                name="decode_long_warm",
                stage="full",
                metric="decode_ms_per_committed_token",
                direction="lower",
                promote_ratio=0.98,
                reject_ratio=1.02,
                min_pairs=12,
                max_pairs=80,
                reject_weight=1.0,
            ),
            CellRule(
                name="prefill_short_cold",
                stage="full",
                metric="prefill_ms",
                direction="lower",
                promote_ratio=1.02,
                reject_ratio=1.05,
                min_pairs=12,
                max_pairs=80,
                reject_weight=1.0,
            ),
            CellRule(
                name="prefill_long_cold",
                stage="full",
                metric="prefill_ms",
                direction="lower",
                promote_ratio=1.02,
                reject_ratio=1.05,
                min_pairs=12,
                max_pairs=80,
                reject_weight=1.0,
            ),
        ),
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Finite escalation dynamic program
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscalationStage:
    name: str
    cost: float
    outcomes: tuple[str, ...]
    likelihood: tuple[tuple[float, ...], ...]
    prerequisites: tuple[str, ...] = ()
    required_for_promotion: bool = False

    def validate(self, n_theta: int) -> None:
        if self.cost < 0:
            raise ValueError("stage cost must be non-negative")
        if not self.outcomes or len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("stage outcomes must be non-empty and unique")
        if len(self.likelihood) != n_theta:
            raise ValueError("one likelihood row is required per theta")
        for row in self.likelihood:
            if len(row) != len(self.outcomes):
                raise ValueError("likelihood row has wrong width")
            if any(p < -_EPS for p in row) or not math.isclose(sum(row), 1.0, abs_tol=1e-10):
                raise ValueError("likelihood rows must be probability vectors")


@dataclass(frozen=True)
class EscalationModel:
    theta: tuple[str, ...]
    prior: tuple[float, ...]
    promote_utility: tuple[float, ...]
    stages: tuple[EscalationStage, ...]

    def validate(self) -> None:
        n = len(self.theta)
        if n == 0 or len(set(self.theta)) != n:
            raise ValueError("theta states must be non-empty and unique")
        if len(self.prior) != n or len(self.promote_utility) != n:
            raise ValueError("prior and utility must match theta")
        if any(p < 0 for p in self.prior) or not math.isclose(sum(self.prior), 1.0, abs_tol=1e-10):
            raise ValueError("prior must be a probability vector")
        names = {s.name for s in self.stages}
        if len(names) != len(self.stages):
            raise ValueError("stage names must be unique")
        prerequisites: dict[str, tuple[str, ...]] = {}
        for stage in self.stages:
            stage.validate(n)
            if any(req not in names for req in stage.prerequisites):
                raise ValueError(f"stage {stage.name!r} has an unknown prerequisite")
            prerequisites[stage.name] = stage.prerequisites

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError("stage prerequisites contain a cycle")
            if name in visited:
                return
            visiting.add(name)
            for prerequisite in prerequisites[name]:
                visit(prerequisite)
            visiting.remove(name)
            visited.add(name)

        for name in names:
            visit(name)


@dataclass(frozen=True)
class EscalationSolution:
    value: float
    action: str
    lower_bound: float
    rollout_action: str
    perfect_information_upper_bound: float
    certified_gap: float


def _posterior(
    belief: tuple[float, ...], stage: EscalationStage, outcome_index: int
) -> tuple[tuple[float, ...], float]:
    weights = tuple(
        belief[i] * stage.likelihood[i][outcome_index]
        for i in range(len(belief))
    )
    probability = sum(weights)
    if probability <= _EPS:
        return belief, 0.0
    return tuple(w / probability for w in weights), probability


def solve_escalation_dp(model: EscalationModel) -> EscalationSolution:
    """Solve the finite measurement-selection problem exactly by backward DP.

    A stage can be used at most once.  ``keep A`` has utility zero.  ``promote
    B`` has theta-dependent utility and is available only after every stage
    marked ``required_for_promotion`` has been observed.  This finite model is
    intentionally small enough to audit; real evidence state can be discretized
    and added to theta/history in the same recursion.
    """

    model.validate()
    stage_index = {stage.name: i for i, stage in enumerate(model.stages)}
    required_mask = 0
    for i, stage in enumerate(model.stages):
        if stage.required_for_promotion:
            required_mask |= 1 << i

    def terminal_value(belief: tuple[float, ...], used_mask: int) -> tuple[float, str]:
        keep = 0.0
        if (used_mask & required_mask) != required_mask:
            return keep, "KEEP_A"
        promote = sum(b * u for b, u in zip(belief, model.promote_utility))
        return (promote, "PROMOTE_B") if promote > keep else (keep, "KEEP_A")

    @lru_cache(maxsize=None)
    def recurse(belief_key: tuple[float, ...], used_mask: int) -> tuple[float, str]:
        belief = belief_key
        best_value, best_action = terminal_value(belief, used_mask)
        for i, stage in enumerate(model.stages):
            bit = 1 << i
            if used_mask & bit:
                continue
            if any(not (used_mask & (1 << stage_index[req])) for req in stage.prerequisites):
                continue
            value = -stage.cost
            for y in range(len(stage.outcomes)):
                post, prob = _posterior(belief, stage, y)
                if prob > 0:
                    rounded = tuple(round(x, 14) for x in post)
                    child, _ = recurse(rounded, used_mask | bit)
                    value += prob * child
            if value > best_value + 1e-12:
                best_value, best_action = value, f"MEASURE:{stage.name}"
        return best_value, best_action

    prior = tuple(round(x, 14) for x in model.prior)
    exact_value, exact_action = recurse(prior, 0)

    # Practical one-step rollout lower bound.  Its base policy either keeps A
    # now or pays the entire remaining prerequisite closure needed for the
    # promotion gate, ignores those future observations, and then promotes when
    # the current posterior mean utility covers that cost.  This is deliberately
    # crude but feasible, so one-step lookahead over it remains a valid lower
    # bound.  Recomputing it after each observation gives a cheap rollout policy.
    completion_mask = required_mask
    changed = True
    while changed:
        changed = False
        for i, stage in enumerate(model.stages):
            if completion_mask & (1 << i):
                for req in stage.prerequisites:
                    bit = 1 << stage_index[req]
                    if not completion_mask & bit:
                        completion_mask |= bit
                        changed = True

    def forced_completion_value(belief: tuple[float, ...], used_mask: int) -> float:
        remaining_cost = sum(
            stage.cost for i, stage in enumerate(model.stages)
            if (completion_mask & (1 << i)) and not (used_mask & (1 << i))
        )
        mean_utility = sum(b * u for b, u in zip(belief, model.promote_utility))
        return max(0.0, mean_utility - remaining_cost)

    lower = forced_completion_value(prior, 0)
    rollout_action = "FORCED_COMPLETION" if lower > 0 else "KEEP_A"
    for i, stage in enumerate(model.stages):
        if stage.prerequisites:
            continue
        value = -stage.cost
        for y in range(len(stage.outcomes)):
            post, prob = _posterior(prior, stage, y)
            if prob > 0:
                value += prob * forced_completion_value(post, 1 << i)
        if value > lower + 1e-12:
            lower = value
            rollout_action = f"MEASURE:{stage.name}"

    # Perfect information ignores measurement costs and gate costs, so it is an
    # upper bound on every admissible policy.
    upper = sum(
        b * max(0.0, u) for b, u in zip(prior, model.promote_utility)
    )
    gap = max(0.0, upper - lower)
    return EscalationSolution(
        value=exact_value,
        action=exact_action,
        lower_bound=lower,
        rollout_action=rollout_action,
        perfect_information_upper_bound=upper,
        certified_gap=gap,
    )


def default_escalation_model() -> EscalationModel:
    """A small, editable ladder: fixture -> short -> long -> full."""

    theta = ("harmful", "neutral", "small_win", "large_win")
    # Utility units are seconds saved over the plausible deployment horizon.
    utility = (-18_000.0, 0.0, 9_000.0, 45_000.0)
    stages = (
        EscalationStage(
            "fixture", 45.0, ("bad", "mixed", "good"),
            (
                (0.70, 0.25, 0.05),
                (0.25, 0.55, 0.20),
                (0.10, 0.45, 0.45),
                (0.03, 0.17, 0.80),
            ),
        ),
        EscalationStage(
            "short_model", 420.0, ("bad", "mixed", "good"),
            (
                (0.80, 0.17, 0.03),
                (0.20, 0.60, 0.20),
                (0.05, 0.35, 0.60),
                (0.01, 0.09, 0.90),
            ),
            prerequisites=("fixture",),
        ),
        EscalationStage(
            "long_prompt", 1_800.0, ("bad", "mixed", "good"),
            (
                (0.88, 0.10, 0.02),
                (0.15, 0.70, 0.15),
                (0.03, 0.27, 0.70),
                (0.005, 0.045, 0.95),
            ),
            prerequisites=("short_model",),
        ),
        EscalationStage(
            "full_campaign", 7_200.0, ("fail", "pass"),
            (
                (0.99, 0.01),
                (0.90, 0.10),
                (0.25, 0.75),
                (0.03, 0.97),
            ),
            prerequisites=("long_prompt",),
            required_for_promotion=True,
        ),
    )
    model = EscalationModel(theta, (0.35, 0.35, 0.20, 0.10), utility, stages)
    model.validate()
    return model


def read_observations(path: Path) -> list[PairObservation]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [PairObservation.from_mapping(row) for row in csv.DictReader(handle)]


def _read_secret(path: Path) -> bytes:
    secret = path.read_bytes()
    if len(secret) < 16:
        raise ValueError("randomization secret must contain at least 16 bytes")
    return secret


def _cmd_protocol(args: argparse.Namespace) -> int:
    candidate_alpha = args.candidate_alpha
    candidate_index = None
    program_alpha = None
    if args.program_alpha is not None:
        if args.candidate_index is None:
            raise ValueError("--program-alpha requires --candidate-index")
        program_alpha = args.program_alpha
        candidate_index = args.candidate_index
        candidate_alpha = polynomial_alpha(program_alpha, candidate_index)
    elif args.candidate_index is not None:
        raise ValueError("--candidate-index requires --program-alpha")
    config = default_protocol(
        args.candidate_id,
        candidate_alpha=candidate_alpha,
        program_alpha=program_alpha,
        candidate_index=candidate_index,
    )
    payload = json.dumps(config.to_json_dict(), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def _cmd_make_seed(args: argparse.Namespace) -> int:
    path = Path(args.output)
    if path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force to replace it")
    if args.bytes < 16:
        raise ValueError("--bytes must be at least 16")
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(args.bytes)
    path.write_bytes(secret)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Some mounted Windows/WSL filesystems do not implement POSIX modes.
        pass
    print(json.dumps({
        "secret_file": str(path),
        "secret_bytes": len(secret),
        "seed_commitment": seed_commitment(secret),
        "instruction": (
            "Record the commitment before the first case is selected; reveal "
            "the secret only after campaign closure."
        ),
    }, indent=2, sort_keys=True))
    return 0


def _cmd_randomize(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = ProtocolConfig.from_json_dict(payload)
    assignment = derive_randomization(
        _read_secret(Path(args.secret_file)),
        protocol_hash=file_sha256(protocol_path),
        candidate_id=config.candidate_id,
        epoch=args.epoch,
        pair_id=args.pair_id,
        cell=args.cell,
        case_id=args.case_id,
    )
    print(json.dumps(asdict(assignment), indent=2, sort_keys=True))
    return 0


def _cmd_log_template(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = ProtocolConfig.from_json_dict(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow(config.required_log_fields)
    print(json.dumps({
        "output": str(output),
        "columns": list(config.required_log_fields),
    }, indent=2, sort_keys=True))
    return 0


def _cmd_decide(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = ProtocolConfig.from_json_dict(payload)
    engine = DecisionEngine(config)
    protocol_hash = file_sha256(protocol_path)
    secret = _read_secret(Path(args.secret_file)) if args.secret_file else None
    for observation in read_observations(Path(args.log)):
        if not observation.protocol_hash:
            engine.invalidate(f"pair {observation.pair_id} lacks protocol_hash")
        elif observation.protocol_hash != protocol_hash:
            engine.invalidate(
                f"pair {observation.pair_id} protocol hash does not match {protocol_hash}"
            )
        if not observation.candidate_id:
            engine.invalidate(f"pair {observation.pair_id} lacks candidate_id")
        if not observation.stage:
            engine.invalidate(f"pair {observation.pair_id} lacks stage")
        if not observation.baseline_commit or not observation.candidate_commit:
            engine.invalidate(
                f"pair {observation.pair_id} lacks baseline/candidate commit identity"
            )
        if secret is not None:
            if not observation.randomization_u64:
                engine.invalidate(
                    f"pair {observation.pair_id} lacks randomization_u64"
                )
            expected_commitment = seed_commitment(secret)
            if observation.seed_commitment != expected_commitment:
                engine.invalidate(
                    f"pair {observation.pair_id} seed commitment does not verify"
                )
            try:
                expected = derive_randomization(
                    secret,
                    protocol_hash=protocol_hash,
                    candidate_id=config.candidate_id,
                    epoch=observation.epoch,
                    pair_id=observation.pair_id,
                    cell=observation.cell,
                    case_id=observation.case_id,
                )
                if observation.order != expected.order:
                    engine.invalidate(
                        f"pair {observation.pair_id} order {observation.order} "
                        f"does not match committed draw {expected.order}"
                    )
                if (
                    observation.randomization_u64
                    and observation.randomization_u64.lower()
                    != expected.randomization_u64
                ):
                    engine.invalidate(
                        f"pair {observation.pair_id} randomization_u64 does not verify"
                    )
            except ValueError as exc:
                engine.invalidate(f"pair {observation.pair_id}: {exc}")
        engine.update(observation)
    if secret is None and engine.status(campaign_closed=args.closed) == "PROMOTE":
        engine.invalidate(
            "promotion cannot be released until --secret-file verifies the committed assignments"
        )
    print(json.dumps(engine.snapshot(campaign_closed=args.closed), indent=2, sort_keys=True))
    return 0 if engine.status(campaign_closed=args.closed) != "INVALID" else 2


def _cmd_dp(args: argparse.Namespace) -> int:
    solution = solve_escalation_dp(default_escalation_model())
    print(json.dumps(asdict(solution), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    protocol = sub.add_parser("protocol", help="write an editable protocol JSON")
    protocol.add_argument("--candidate-id", default="candidate")
    protocol.add_argument("--candidate-alpha", type=float, default=0.01)
    protocol.add_argument("--program-alpha", type=float)
    protocol.add_argument("--candidate-index", type=int)
    protocol.add_argument("--output")
    protocol.set_defaults(func=_cmd_protocol)

    make_seed = sub.add_parser(
        "make-seed", help="create a private randomization seed and print its commitment"
    )
    make_seed.add_argument("--output", required=True)
    make_seed.add_argument("--bytes", type=int, default=32)
    make_seed.add_argument("--force", action="store_true")
    make_seed.set_defaults(func=_cmd_make_seed)

    randomize = sub.add_parser(
        "randomize", help="draw one HMAC-bound AB/BA order after locking the case"
    )
    randomize.add_argument("--secret-file", required=True)
    randomize.add_argument("--protocol", required=True)
    randomize.add_argument("--epoch", type=int, required=True)
    randomize.add_argument("--pair-id", required=True)
    randomize.add_argument("--cell", required=True)
    randomize.add_argument("--case-id", required=True)
    randomize.set_defaults(func=_cmd_randomize)

    log_template = sub.add_parser(
        "log-template", help="write a header-only CSV with the protocol's required fields"
    )
    log_template.add_argument("--protocol", required=True)
    log_template.add_argument("--output", required=True)
    log_template.set_defaults(func=_cmd_log_template)

    decide = sub.add_parser("decide", help="replay a long-format pair CSV")
    decide.add_argument("--protocol", required=True)
    decide.add_argument("--log", required=True)
    decide.add_argument(
        "--secret-file",
        help="after campaign closure, verify every logged HMAC assignment",
    )
    decide.add_argument("--closed", action="store_true",
                        help="return INCONCLUSIVE instead of CONTINUE when evidence is weak")
    decide.set_defaults(func=_cmd_decide)

    dp = sub.add_parser("dp-demo", help="solve the finite escalation example")
    dp.set_defaults(func=_cmd_dp)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
