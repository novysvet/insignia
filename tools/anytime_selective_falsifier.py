#!/usr/bin/env python3
"""Anytime-valid randomized auditing for a feedback-coupled fast-path policy.

The online certificate is design based.  It conditions on the adaptively
visited contexts and on every fast potential outcome, then uses only the audit
coin as randomness.  Contexts may drift, fast actions may change later
contexts, labels may arrive late, and the controller may abstain.  The
non-negotiable condition is overlap: every proposed fast action that can be
committed without an exact shadow has a logged audit probability bounded away
from zero.

Two targets are kept separate.

``intended selective risk``
    The prefix-average severity of contexts that the policy intends to commit
    fast, weighting round t by 1-q_t because an audit overrides the fast commit.
    This is a marginal, design-based policy-value target.  It is not a
    pointwise conditional guarantee for every context.

``committed catastrophic budget``
    The realized severity of unaudited fast commits.  A one-sided confidence
    ledger plus a prospective gate can enforce

        committed_loss <= startup_budget + epsilon * fast_commits

    simultaneously over time with probability at least 1-delta.  When the
    ledger, fingerprint, support check, or delay contract is invalid, the only
    permitted default is exact work.

The confidence update is an audit-capture e-process specialized to bounded,
nonnegative severity.  It is sharper for sparse catastrophic loss than a
variance-only Hoeffding or Bernstein interval and remains valid under arbitrary
policy feedback and covariate drift.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

_EPS = 1e-12


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for certificate-critical state."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_certificate_fingerprint(
    *,
    controller_digest: str,
    feature_schema_digest: str,
    verifier_digest: str,
    loss_schema_digest: str,
    threshold: float,
    information_price: float,
    audit_protocol_version: str,
    block_semantics_version: str,
    cache_transition_version: str,
    calibration_epoch: str,
    supported_keys: Sequence[str],
    max_severity: float,
    q_min: float,
    q_max: float,
    cost_model_digest: str,
    reset_policy_digest: str,
) -> str:
    """Fingerprint every semantic object whose change invalidates evidence."""

    return canonical_fingerprint({
        "controller_digest": controller_digest,
        "feature_schema_digest": feature_schema_digest,
        "verifier_digest": verifier_digest,
        "loss_schema_digest": loss_schema_digest,
        "threshold": float(threshold),
        "information_price": float(information_price),
        "audit_protocol_version": audit_protocol_version,
        "block_semantics_version": block_semantics_version,
        "cache_transition_version": cache_transition_version,
        "calibration_epoch": calibration_epoch,
        "supported_keys": sorted(str(k) for k in supported_keys),
        "max_severity": float(max_severity),
        "q_min": float(q_min),
        "q_max": float(q_max),
        "cost_model_digest": cost_model_digest,
        "reset_policy_digest": reset_policy_digest,
    })


def build_runtime_state_fingerprint(
    *,
    certificate_fingerprint: str,
    request_id: str,
    round_id: int,
    prefix_digest: str,
    target_logits_digest: str,
    draft_logits_digest: str,
    route_state_digest: str,
    cache_state_digest: str,
    hidden_summary_digest: str,
    history_digest: str,
    support_key: str,
    score: float,
    threshold: float,
    audit_probability: float,
    severity_bound: float,
) -> str:
    """Bind one audit coin and eventual label to its pre-coin causal state.

    Only digests are required so the log need not retain large logits or hidden
    tensors.  The caller must compute every digest before revealing the audit
    coin.  The certificate fingerprint binds model and verifier semantics; this
    round fingerprint binds the concrete prefix, routing/cache state, decision,
    propensity, and severity range.
    """

    if round_id < 0:
        raise ValueError("round_id must be non-negative")
    if not (0 <= audit_probability <= 1):
        raise ValueError("audit_probability must lie in [0,1]")
    if severity_bound <= 0 or not math.isfinite(severity_bound):
        raise ValueError("severity_bound must be finite and positive")
    return canonical_fingerprint({
        "certificate_fingerprint": certificate_fingerprint,
        "request_id": request_id,
        "round_id": int(round_id),
        "prefix_digest": prefix_digest,
        "target_logits_digest": target_logits_digest,
        "draft_logits_digest": draft_logits_digest,
        "route_state_digest": route_state_digest,
        "cache_state_digest": cache_state_digest,
        "hidden_summary_digest": hidden_summary_digest,
        "history_digest": history_digest,
        "support_key": support_key,
        "score": float(score),
        "threshold": float(threshold),
        "audit_probability": float(audit_probability),
        "severity_bound": float(severity_bound),
    })


def runtime_support_key(
    *,
    score: float,
    cache_fraction: float,
    route_entropy: float,
    divergence_proxy: float,
    score_bins: Sequence[float] = (0.02, 0.05, 0.10, 0.20),
    cache_bins: Sequence[float] = (0.35, 0.65, 0.85),
    entropy_bins: Sequence[float] = (0.35, 0.65, 0.85),
    divergence_bins: Sequence[float] = (0.10, 0.30, 0.60),
) -> str:
    """Quantize runtime state into a frozen policy support guard."""

    values = (score, cache_fraction, route_entropy, divergence_proxy)
    if not all(math.isfinite(v) for v in values):
        return "invalid"
    bins = (score_bins, cache_bins, entropy_bins, divergence_bins)
    indices = tuple(int(np.searchsorted(b, v, side="right")) for b, v in zip(bins, values))
    return "s%dc%de%dd%d" % indices


def counter_based_audit_uniform(
    secret_key: bytes,
    *,
    request_id: str,
    round_id: int,
    state_fingerprint: str,
    domain: str = "insignia-random-audit-v1",
) -> float:
    """Generate a reproducible audit uniform from a secret, domain-separated key.

    The controller must fix and log q_t before this function is called.  The
    secret key is unavailable to the controller during the request; a key ID
    may be logged and the key may be disclosed later for replay.  The theorem
    idealizes this pseudorandom draw as an independent Uniform[0,1) coin.
    """

    if not secret_key:
        raise ValueError("secret_key must be non-empty")
    if round_id < 0:
        raise ValueError("round_id must be non-negative")
    message = "\0".join((domain, request_id, str(round_id), state_fingerprint)).encode("utf-8")
    digest = hmac.new(secret_key, message, hashlib.sha256).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    # Midpoint mapping avoids exact zero and one and is stable across runtimes.
    return (integer + 0.5) / 2**64


@dataclass(frozen=True)
class AuditRecord:
    round_id: int
    audit_probability: float
    severity_bound: float
    audited: bool
    audit_uniform: float
    state_fingerprint: str
    runtime_state_fingerprint: str
    support_key: str
    label: float | None = None
    label_due_round: int | None = None

    def validate(self, *, q_min: float, expected_fingerprint: str | None) -> None:
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        q = self.audit_probability
        if not (q_min - _EPS <= q <= 1.0):
            raise ValueError(f"audit propensity {q} is outside [{q_min}, 1]")
        if not (0 <= self.audit_uniform < 1):
            raise ValueError("audit_uniform must lie in [0,1)")
        if self.audited != (self.audit_uniform < q):
            raise ValueError("audit outcome does not match the logged uniform and propensity")
        if self.severity_bound <= 0 or not math.isfinite(self.severity_bound):
            raise ValueError("severity_bound must be finite and positive")
        if expected_fingerprint is not None and self.state_fingerprint != expected_fingerprint:
            raise ValueError("state fingerprint does not match the active certificate")
        if len(self.runtime_state_fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in self.runtime_state_fingerprint
        ):
            raise ValueError("runtime_state_fingerprint must be a lowercase SHA-256 digest")
        if not self.support_key:
            raise ValueError("support_key must be non-empty")
        if self.label_due_round is not None and self.label_due_round < self.round_id:
            raise ValueError("label_due_round cannot precede round_id")
        if self.audited:
            if self.label is None and self.label_due_round is None:
                raise ValueError("an unresolved audit must log a due round")
            if self.label is not None and not (-_EPS <= self.label <= self.severity_bound + _EPS):
                raise ValueError("audit label is outside the logged severity bound")
        else:
            if self.label is not None:
                raise ValueError("an unaudited record cannot carry a counterfactual label")
            if self.label_due_round is not None:
                raise ValueError("an unaudited record cannot have a label due round")


class AuditCaptureBoundary:
    r"""A mixture e-process for randomly captured nonnegative loss.

    At a decision, a fixed value v in [0,b] is audited with predictable
    probability q.  For any rho with rho*b < q, define

        lambda(rho,q,b) = -log(1-rho*b/q) / b.

    Convexity of log(1-q+q exp(-lambda v)) gives

        E[exp(rho v - lambda Z v) | v,q] <= 1.

    Therefore

        M_n(rho) = exp(rho * sum(v_t) - sum(lambda_t Z_t v_t))

    is a nonnegative supermartingale even when v_t, q_t, and future contexts are
    adaptive.  A fixed mixture over rho is another supermartingale.  Inverting
    Ville's inequality yields a one-sided, time-uniform bound on total v_t.
    """

    _DEFAULT_RATIOS = (0.08, 0.16, 0.28, 0.42, 0.58, 0.72, 0.84, 0.94)
    _DEFAULT_WEIGHTS = (0.02, 0.03, 0.05, 0.08, 0.12, 0.17, 0.23, 0.30)

    def __init__(
        self,
        *,
        delta: float,
        max_rho: float,
        rho_ratios: Sequence[float] = _DEFAULT_RATIOS,
        mixture_weights: Sequence[float] = _DEFAULT_WEIGHTS,
    ) -> None:
        if not (0 < delta < 1):
            raise ValueError("delta must lie in (0,1)")
        if max_rho <= 0 or not math.isfinite(max_rho):
            raise ValueError("max_rho must be finite and positive")
        ratios = np.asarray(rho_ratios, dtype=float)
        weights = np.asarray(mixture_weights, dtype=float)
        if ratios.ndim != 1 or weights.shape != ratios.shape or len(ratios) == 0:
            raise ValueError("rho ratios and mixture weights must have equal non-empty shape")
        if np.any(ratios <= 0) or np.any(ratios >= 1) or np.any(np.diff(ratios) <= 0):
            raise ValueError("rho ratios must be strictly increasing inside (0,1)")
        if np.any(weights <= 0) or not math.isfinite(float(weights.sum())):
            raise ValueError("mixture weights must be finite and positive")
        self.delta = float(delta)
        self.max_rho = float(max_rho)
        self.rhos = self.max_rho * ratios
        self.weights = weights / weights.sum()
        self.log_weights = np.log(self.weights)
        self.audit_scores = np.zeros_like(self.rhos)
        self.resolved_audits = 0
        self.observed_audit_value = 0.0
        self._root_cache: float | None = None

    def update(
        self,
        *,
        audit_probability: float,
        target_bound: float,
        observed_value: float,
    ) -> None:
        """Add one resolved audited value to every predeclared e-process."""

        q = float(audit_probability)
        b = float(target_bound)
        v = float(observed_value)
        if b <= _EPS:
            if abs(v) > _EPS:
                raise ValueError("zero target bound requires zero observed value")
            return
        if not (0 < q <= 1) or not (0 <= v <= b + _EPS):
            raise ValueError("invalid audit update")
        ratios = self.rhos * b / q
        if np.any(ratios >= 1 - 1e-13):
            raise ValueError("rho grid violates rho * target_bound < audit_probability")
        lambdas = -np.log1p(-ratios) / b
        self.audit_scores += lambdas * min(v, b)
        self._root_cache = None
        self.resolved_audits += 1
        self.observed_audit_value += min(v, b)

    def _log_mixture_at(self, total_value: float) -> float:
        terms = self.log_weights + self.rhos * total_value - self.audit_scores
        maximum = float(np.max(terms))
        return maximum + math.log(float(np.exp(terms - maximum).sum()))

    def upper(self, deterministic_bound: float) -> float:
        """Return the mixture-CS upper endpoint, intersected with a hard bound."""

        cap = float(deterministic_bound)
        if cap < 0 or not math.isfinite(cap):
            raise ValueError("deterministic_bound must be finite and non-negative")
        if cap <= _EPS:
            return 0.0
        if self._root_cache is None:
            target = math.log(1.0 / self.delta)
            lo = 0.0
            hi = max(1.0, (target + float(np.max(self.audit_scores))) / float(np.min(self.rhos)))
            while self._log_mixture_at(hi) <= target:
                hi *= 2.0
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                if self._log_mixture_at(mid) <= target:
                    lo = mid
                else:
                    hi = mid
            self._root_cache = hi
        return min(cap, self._root_cache)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "kind": "audit-capture-mixture-e-process",
            "delta": self.delta,
            "max_rho": self.max_rho,
            "rhos": [float(x) for x in self.rhos],
            "mixture_weights": [float(x) for x in self.weights],
            "resolved_audits": self.resolved_audits,
            "observed_audit_value": self.observed_audit_value,
            "update": "lambda_t(rho)=-log(1-rho*b_t/q_t)/b_t",
        }


class BernsteinGridBoundary:
    """Reference one-sided martingale Bernstein boundary.

    This class is retained for comparisons and tests.  The production ledger
    below uses the nonnegative audit-capture e-process because rare severity is
    the relevant regime.
    """

    def __init__(
        self,
        *,
        delta: float,
        max_increment: float,
        grid_size: int = 96,
        min_lambda_ratio: float = 1e-8,
        max_lambda_ratio: float = 2.90,
    ) -> None:
        if not (0 < delta < 1):
            raise ValueError("delta must lie in (0,1)")
        if max_increment <= 0 or not math.isfinite(max_increment):
            raise ValueError("max_increment must be finite and positive")
        if grid_size < 4:
            raise ValueError("grid_size must be at least four")
        if not (0 < min_lambda_ratio < max_lambda_ratio < 3):
            raise ValueError("lambda ratios must lie strictly inside (0,3)")
        self.delta = float(delta)
        self.max_increment = float(max_increment)
        self.grid_size = int(grid_size)
        self.lambdas = np.geomspace(
            min_lambda_ratio / self.max_increment,
            max_lambda_ratio / self.max_increment,
            self.grid_size,
        )
        self.log_penalty = math.log(self.grid_size / self.delta)

    def __call__(self, variance_process: float) -> float:
        if variance_process <= _EPS:
            return 0.0
        if not math.isfinite(variance_process) or variance_process < 0:
            raise ValueError("variance_process must be finite and non-negative")
        b = self.max_increment
        lam = self.lambdas
        psi = lam * lam / (2.0 * (1.0 - lam * b / 3.0))
        return float(np.min((self.log_penalty + psi * variance_process) / lam))


@dataclass(frozen=True)
class AuditLedgerSnapshot:
    records: int
    processed_prefix: int
    pending_audits: int
    fast_commits: int
    audits: int
    intended_fast_weight: float
    audited_loss: float
    pending_fast_bound: float
    proposal_loss_upper: float
    proposal_risk_upper: float
    intended_loss_upper: float
    intended_selective_risk_upper: float
    committed_loss_upper: float
    deterministic_committed_bound: float
    risk_budget: float
    pathwise_budget_certificate_valid: bool
    intended_risk_certificate_valid: bool
    certificate_valid: bool
    invalid_reason: str | None


class RandomAuditLedger:
    """Decision-ordered anytime ledger for selective and committed fast loss.

    Let S_t denote a terminal fast proposal, Q_t its predictable audit
    propensity, Z_t the audit indicator, and Y_t in [0,B_t] its block-level fast
    potential severity.  An audit executes the fast path in a rollback shadow,
    computes the exact result, commits exact, and eventually reveals Y_t.

    The ledger builds two simultaneous capture confidence sequences:

    * proposal total L_n = sum S_t Y_t;
    * intended-commit total I_n = sum S_t (1-Q_t) Y_t.

    The actual committed fast loss is

        C_n = L_n - sum S_t Z_t Y_t.

    Delayed labels are processed only through the longest resolved decision
    prefix.  Every later unresolved target is charged at its full bound.  This
    is valid even when delays depend on severity.
    """

    def __init__(
        self,
        *,
        delta: float,
        epsilon: float,
        startup_budget: float,
        max_severity: float = 1.0,
        q_min: float = 0.02,
        state_fingerprint: str | None = None,
    ) -> None:
        if not (0 < delta < 1):
            raise ValueError("delta must lie in (0,1)")
        if epsilon < 0 or startup_budget < 0:
            raise ValueError("risk budgets must be non-negative")
        if max_severity <= 0 or not math.isfinite(max_severity):
            raise ValueError("max_severity must be finite and positive")
        if not (0 < q_min <= 1):
            raise ValueError("q_min must lie in (0,1]")
        self.delta = float(delta)
        self.epsilon = float(epsilon)
        self.startup_budget = float(startup_budget)
        self.max_severity = float(max_severity)
        self.q_min = float(q_min)
        self.state_fingerprint = state_fingerprint

        # Split the error budget between the policy-value and actual-commit
        # ledgers.  The proposal ledger also supplies the committed-loss bound.
        proposal_delta = self.delta / 2.0
        intended_delta = self.delta / 2.0
        self._proposal_capture = AuditCaptureBoundary(
            delta=proposal_delta,
            max_rho=0.999 * self.q_min / self.max_severity,
        )
        max_intended_bound = max((1.0 - self.q_min) * self.max_severity, _EPS)
        self._intended_capture = AuditCaptureBoundary(
            delta=intended_delta,
            max_rho=0.999 * self.q_min / max_intended_bound,
        )

        self.records: list[AuditRecord] = []
        self._round_to_index: dict[int, int] = {}
        self._processed = 0
        self._fast_commits = 0
        self._audits = 0
        self._audited_loss = 0.0
        self._processed_proposal_bound = 0.0
        self._processed_intended_bound = 0.0
        self._total_proposal_bound = 0.0
        self._total_intended_bound = 0.0
        self._intended_fast_weight = 0.0
        self._deterministic_committed_bound = 0.0
        self._invalid_reason: str | None = None
        self._snapshot_cache: AuditLedgerSnapshot | None = None

    @property
    def invalid_reason(self) -> str | None:
        return self._invalid_reason

    def invalidate(self, reason: str) -> None:
        if self._invalid_reason is None:
            self._invalid_reason = str(reason)
        self._snapshot_cache = None

    def append(self, record: AuditRecord) -> None:
        if self._invalid_reason is not None:
            raise RuntimeError(f"ledger is invalid: {self._invalid_reason}")
        if record.round_id in self._round_to_index:
            self.invalidate("duplicate round id")
            raise ValueError("duplicate round id")
        if self.records and record.round_id <= self.records[-1].round_id:
            self.invalidate("round ids are not strictly increasing")
            raise ValueError("round ids must be strictly increasing")
        try:
            record.validate(q_min=self.q_min, expected_fingerprint=self.state_fingerprint)
        except Exception as exc:
            self.invalidate(str(exc))
            raise
        if record.severity_bound > self.max_severity + _EPS:
            self.invalidate("record severity bound exceeds ledger maximum")
            raise ValueError("record severity bound exceeds ledger maximum")

        self._snapshot_cache = None
        self._round_to_index[record.round_id] = len(self.records)
        self.records.append(record)
        self._total_proposal_bound += record.severity_bound
        intended_weight = 1.0 - record.audit_probability
        self._intended_fast_weight += intended_weight
        self._total_intended_bound += intended_weight * record.severity_bound
        if record.audited:
            self._audits += 1
        else:
            self._fast_commits += 1
            self._deterministic_committed_bound += record.severity_bound
        self._drain_resolved_prefix()

    def resolve(self, round_id: int, label: float) -> None:
        if self._invalid_reason is not None:
            raise RuntimeError(f"ledger is invalid: {self._invalid_reason}")
        try:
            index = self._round_to_index[round_id]
        except KeyError as exc:
            self.invalidate("label references unknown round")
            raise KeyError(f"unknown round id {round_id}") from exc
        record = self.records[index]
        if not record.audited:
            self.invalidate("label supplied for unaudited round")
            raise ValueError("cannot resolve an unaudited round")
        if record.label is not None:
            self.invalidate("duplicate label resolution")
            raise ValueError("audit label was already resolved")
        if not (-_EPS <= label <= record.severity_bound + _EPS):
            self.invalidate("audit label outside severity bound")
            raise ValueError("audit label is outside the logged bound")
        clipped = float(np.clip(label, 0, record.severity_bound))
        self._snapshot_cache = None
        self.records[index] = replace(record, label=clipped)
        self._drain_resolved_prefix()

    def _drain_resolved_prefix(self) -> None:
        while self._processed < len(self.records):
            record = self.records[self._processed]
            if record.audited and record.label is None:
                break
            q = record.audit_probability
            b = record.severity_bound
            intended_weight = 1.0 - q
            self._processed_proposal_bound += b
            self._processed_intended_bound += intended_weight * b
            if record.audited:
                assert record.label is not None
                y = record.label
                self._audited_loss += y
                self._proposal_capture.update(
                    audit_probability=q,
                    target_bound=b,
                    observed_value=y,
                )
                self._intended_capture.update(
                    audit_probability=q,
                    target_bound=intended_weight * b,
                    observed_value=intended_weight * y,
                )
            self._processed += 1

    def _pending_fast_bound(self) -> float:
        return float(sum(
            r.severity_bound for r in self.records[self._processed:] if not r.audited
        ))

    def _pending_proposal_bound(self) -> float:
        return self._total_proposal_bound - self._processed_proposal_bound

    def _pending_intended_bound(self) -> float:
        return self._total_intended_bound - self._processed_intended_bound

    def _proposal_prefix_upper(self, deterministic_bound: float | None = None) -> float:
        bound = self._processed_proposal_bound if deterministic_bound is None else deterministic_bound
        return self._proposal_capture.upper(bound)

    def proposal_loss_upper(self) -> float:
        stochastic = self._proposal_prefix_upper() + self._pending_proposal_bound()
        return min(self._total_proposal_bound, stochastic)

    def proposal_risk_upper(self) -> float:
        if not self.records:
            return math.inf
        return self.proposal_loss_upper() / len(self.records)

    def intended_loss_upper(self) -> float:
        prefix = self._intended_capture.upper(self._processed_intended_bound)
        stochastic = prefix + self._pending_intended_bound()
        return min(self._total_intended_bound, stochastic)

    def intended_selective_risk_upper(self) -> float:
        if self._intended_fast_weight <= _EPS:
            return math.inf
        return self.intended_loss_upper() / self._intended_fast_weight

    def upper_committed_loss(self) -> float:
        prefix = max(0.0, self._proposal_prefix_upper() - self._audited_loss)
        stochastic = prefix + self._pending_fast_bound()
        return min(self._deterministic_committed_bound, stochastic)

    def risk_budget(self, fast_commits: int | None = None) -> float:
        n = self._fast_commits if fast_commits is None else fast_commits
        return self.startup_budget + self.epsilon * n

    def prospective_upper_if_fast(self, *, severity_bound: float) -> float:
        """Upper ledger value after the current proposal is not audited."""

        b = float(severity_bound)
        if not (0 < b <= self.max_severity + _EPS):
            raise ValueError("prospective severity bound is invalid")
        deterministic = self._deterministic_committed_bound + b
        if self._processed == len(self.records):
            proposal_prefix = self._proposal_prefix_upper(
                self._processed_proposal_bound + b
            )
            stochastic = max(0.0, proposal_prefix - self._audited_loss)
        else:
            stochastic = (
                max(0.0, self._proposal_prefix_upper() - self._audited_loss)
                + self._pending_fast_bound()
                + b
            )
        return min(deterministic, stochastic)

    def can_commit_fast(self, *, severity_bound: float) -> bool:
        """Prospectively gate the unaudited branch of the next audit coin."""

        if self._invalid_reason is not None:
            return False
        upper = self.prospective_upper_if_fast(severity_bound=severity_bound)
        return upper <= self.risk_budget(self._fast_commits + 1) + 1e-12

    def snapshot(self) -> AuditLedgerSnapshot:
        if self._snapshot_cache is not None:
            return self._snapshot_cache
        pending_audits = sum(
            1 for r in self.records[self._processed:] if r.audited and r.label is None
        )
        proposal_loss = self.proposal_loss_upper()
        proposal_risk = proposal_loss / len(self.records) if self.records else math.inf
        intended_loss = self.intended_loss_upper()
        intended_upper = (
            intended_loss / self._intended_fast_weight
            if self._intended_fast_weight > _EPS else math.inf
        )
        committed_upper = self.upper_committed_loss()
        budget = self.risk_budget()
        pathwise_valid = committed_upper <= budget + 1e-12
        intended_valid = intended_upper <= self.epsilon + 1e-12
        self._snapshot_cache = AuditLedgerSnapshot(
            records=len(self.records),
            processed_prefix=self._processed,
            pending_audits=pending_audits,
            fast_commits=self._fast_commits,
            audits=self._audits,
            intended_fast_weight=self._intended_fast_weight,
            audited_loss=self._audited_loss,
            pending_fast_bound=self._pending_fast_bound(),
            proposal_loss_upper=proposal_loss,
            proposal_risk_upper=proposal_risk,
            intended_loss_upper=intended_loss,
            intended_selective_risk_upper=intended_upper,
            committed_loss_upper=committed_upper,
            deterministic_committed_bound=self._deterministic_committed_bound,
            risk_budget=budget,
            pathwise_budget_certificate_valid=pathwise_valid,
            intended_risk_certificate_valid=intended_valid,
            certificate_valid=(
                self._invalid_reason is None and pathwise_valid and intended_valid
            ),
            invalid_reason=self._invalid_reason,
        )
        return self._snapshot_cache

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "protocol": {
                "targets": {
                    "marginal": "sum (1-q_t) Y_t(fast) / sum (1-q_t)",
                    "pathwise": "committed severity <= startup_budget + epsilon * fast_commits",
                    "conditional": "not claimed without a structural model",
                },
                "epsilon": self.epsilon,
                "startup_budget": self.startup_budget,
                "delta": self.delta,
                "delta_split": {
                    "proposal_and_committed": self.delta / 2.0,
                    "intended_selective_risk": self.delta / 2.0,
                },
                "q_min": self.q_min,
                "max_severity": self.max_severity,
                "state_fingerprint": self.state_fingerprint,
                "confidence_update": {
                    "proposal": self._proposal_capture.to_json_dict(),
                    "intended": self._intended_capture.to_json_dict(),
                },
                "delay_rule": (
                    "process only the longest resolved decision prefix; charge the later "
                    "suffix at logged worst-case bounds"
                ),
                "default_action": "exact",
            },
            "snapshot": asdict(self.snapshot()),
        }


def pathwise_reset_reserve(
    *,
    global_startup_budget: float,
    epsilon: float,
    closed_fast_commits: Sequence[int],
    closed_committed_uppers: Sequence[float],
) -> float | None:
    """Return the unspent global reserve available to a new reset epoch.

    If closed epoch e has actual committed loss C_e <= U_e and K_e fast
    commits, assigning the new epoch

        beta_new = beta0 + epsilon*sum(K_e) - sum(U_e)

    preserves the single global contract after adding the new epoch's local
    ledger.  A materially negative value means the closed evidence cannot
    certify any remaining global reserve, so the caller must stay exact rather
    than clipping the value to zero and silently restarting.  A reset must
    never mint another copy of ``global_startup_budget``.
    """

    if global_startup_budget < 0 or epsilon < 0:
        raise ValueError("pathwise budgets must be non-negative")
    if len(closed_fast_commits) != len(closed_committed_uppers):
        raise ValueError("closed epoch counts and upper endpoints must align")
    commits = 0
    upper = 0.0
    for count, endpoint in zip(closed_fast_commits, closed_committed_uppers):
        if count < 0 or int(count) != count:
            raise ValueError("fast-commit counts must be non-negative integers")
        if endpoint < 0 or not math.isfinite(endpoint):
            raise ValueError("committed-loss upper endpoints must be finite and non-negative")
        commits += int(count)
        upper += float(endpoint)
    reserve = float(global_startup_budget) + epsilon*commits - upper
    if reserve < -1e-10:
        return None
    return max(0.0, reserve)


def certificate_reset_reason(
    *,
    expected_fingerprint: str,
    observed_fingerprint: str,
    support_key: str,
    supported_keys: Sequence[str],
    audit_probability: float,
    q_min: float,
    oldest_pending_delay: int,
    max_label_delay: int,
    drift_alarm: bool,
    model_modified: bool,
    online_risk_upper: float,
    epsilon: float,
) -> str | None:
    """Return the first fail-closed reset reason, or None when still valid."""

    if observed_fingerprint != expected_fingerprint:
        return "state_fingerprint_mismatch"
    if support_key == "invalid" or support_key not in set(supported_keys):
        return "unsupported_runtime_state"
    if audit_probability < q_min - _EPS or audit_probability > 1 + _EPS:
        return "invalid_logged_propensity"
    if oldest_pending_delay > max_label_delay:
        return "audit_label_delay_exceeded"
    if drift_alarm:
        return "online_change_alarm"
    if model_modified:
        return "model_changed_without_fresh_evidence"
    if online_risk_upper > epsilon + 1e-12:
        return "anytime_risk_upper_crossed"
    return None


def draw_random_audit(rng: np.random.Generator, probability: float) -> tuple[bool, float]:
    """Draw and return the exact audit decision plus the logged uniform."""

    if not (0 <= probability <= 1):
        raise ValueError("audit probability must lie in [0,1]")
    uniform = float(rng.random())
    return uniform < probability, uniform


def optimal_audit_probability(
    *,
    audit_opportunity_cost: float,
    information_price: float,
    severity_bound: float = 1.0,
    q_min: float = 0.02,
    q_max: float = 0.50,
) -> float:
    """Square-root audit allocation including exact and cache opportunity cost.

    Relative to an unaudited fast commit, let kappa be the lost time reward from
    auditing, including immediate exact-shadow cost and a shadow price for the
    audit cache transition.  The Horvitz-Thompson variance proxy is

        B^2 (1-q)/q.

    Maximizing reward minus ``information_price * variance`` gives

        q* = B * sqrt(information_price / kappa),

    clipped to the operational overlap interval.  If auditing is no worse than
    committing fast, q_max is optimal.
    """

    if severity_bound <= 0 or information_price < 0:
        raise ValueError("severity_bound must be positive and information_price non-negative")
    if not (0 < q_min <= q_max <= 1):
        raise ValueError("invalid audit probability interval")
    kappa = float(audit_opportunity_cost)
    if kappa <= _EPS:
        return float(q_max)
    raw = severity_bound * math.sqrt(information_price / kappa)
    return float(np.clip(raw, q_min, q_max))


def weighted_hoeffding_upper(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    delta: float,
    bound: float = 1.0,
) -> float:
    """One-sided fixed-sample bound for a predeclared weighted mean.

    This is a held-out-epoch tool, not an anytime or feedback-robust guarantee.
    It assumes independent/exchangeable held-out rows and weights fixed without
    their labels.  The online audit ledger does not make this assumption.
    """

    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.shape != w.shape or y.ndim != 1:
        raise ValueError("values and weights must be one-dimensional with equal shape")
    if not (0 < delta < 1) or bound <= 0:
        raise ValueError("invalid confidence parameters")
    if np.any(y < -_EPS) or np.any(y > bound + _EPS) or np.any(w < 0):
        raise ValueError("values or weights outside their bounds")
    total = float(w.sum())
    if total <= _EPS:
        return math.inf
    mean = float(w @ y / total)
    width = bound * math.sqrt(math.log(1.0 / delta) * float(w @ w) / (2.0 * total * total))
    return min(bound, mean + width)


def weighted_betting_upper(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    delta: float,
    bound: float = 1.0,
    lambda_ratios: Sequence[float] = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90),
) -> float:
    """Low-loss fixed-epoch upper bound from a mixture betting e-value.

    This routine assumes the held-out ``(weight, value)`` pairs are IID or
    exchangeable and that their weights were fixed without looking at their
    labels.  For a candidate weighted population mean m, under the null

        E[w (Y-m)] >= 0,

    each factor ``1-lambda*w*(Y-m)`` has expectation at most one.  A fixed
    mixture over lambda is an e-value.  Inverting it gives a one-sided upper
    bound that is much sharper than range-only Hoeffding when loss is sparse.
    It is deliberately confined to fresh held-out epochs; online feedback uses
    :class:`RandomAuditLedger` instead.
    """

    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.shape != w.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("values and weights must have equal non-empty one-dimensional shape")
    if not (0 < delta < 1) or bound <= 0:
        raise ValueError("invalid confidence parameters")
    if np.any(y < -_EPS) or np.any(y > bound + _EPS) or np.any(w < 0):
        raise ValueError("values or weights outside their bounds")
    positive = w > _EPS
    if not np.any(positive):
        return math.inf
    y = y[positive] / bound
    w = w[positive]
    max_weight = float(np.max(w))
    ratios = np.asarray(lambda_ratios, dtype=float)
    if np.any(ratios <= 0) or np.any(ratios >= 1):
        raise ValueError("lambda ratios must lie in (0,1)")
    lambdas = ratios / max_weight
    log_weights = np.full(len(lambdas), -math.log(len(lambdas)))
    target = math.log(1.0 / delta)

    def log_evalue(mean: float) -> float:
        increments = 1.0 - lambdas[:, None] * w[None, :] * (y[None, :] - mean)
        if np.any(increments <= 0):
            return math.inf
        components = log_weights + np.log(increments).sum(axis=1)
        maximum = float(np.max(components))
        return maximum + math.log(float(np.exp(components-maximum).sum()))

    if log_evalue(1.0) < target:
        return bound
    lo, hi = 0.0, 1.0
    for _ in range(56):
        mid = 0.5*(lo+hi)
        if log_evalue(mid) < target:
            lo = mid
        else:
            hi = mid
    return bound*hi


@dataclass(frozen=True)
class CalibrationRow:
    score: float
    severity: float
    fast_gain_ms: float
    audit_gain_ms: float
    severity_bound: float = 1.0
    support_key: str = "default"


@dataclass(frozen=True)
class ThresholdAuditPlan:
    threshold: float
    information_price: float
    expected_saved_ms_per_round: float
    screening_risk: float
    screening_selected_mass: float
    screening_method: str
    mean_audit_probability: float
    supported_keys: tuple[str, ...]
    candidate_count: int

    @property
    def heldout_risk_upper(self) -> float:
        """Compatibility alias for optional held-out plan constructors."""

        return self.screening_risk

    @property
    def heldout_selected_mass(self) -> float:
        """Compatibility alias for optional held-out plan constructors."""

        return self.screening_selected_mass


def select_threshold_and_audit_plan(
    design_rows: Sequence[CalibrationRow],
    certificate_rows: Sequence[CalibrationRow],
    *,
    thresholds: Sequence[float],
    information_prices: Sequence[float],
    epsilon: float,
    delta: float,
    q_min: float,
    q_max: float,
    min_selected_mass: float = 40.0,
    min_support_count: int = 4,
) -> ThresholdAuditPlan | None:
    """Optimize a finite threshold/audit grid with simultaneous held-out tests.

    The design split estimates reward.  The independent certificate split is
    used once with a Bonferroni allocation across the predeclared finite grid,
    so choosing the highest-reward certified pair does not reuse nominal
    evidence.  A newly tuned model must start a new plan on future data or
    receive a fresh error allocation.
    """

    if not design_rows or not certificate_rows:
        return None
    candidates = [(float(t), float(lam)) for t in thresholds for lam in information_prices]
    if not candidates:
        return None
    per_candidate_delta = delta / len(candidates)
    certified: list[ThresholdAuditPlan] = []

    def q_for(row: CalibrationRow, info_price: float) -> float:
        opportunity = row.fast_gain_ms - row.audit_gain_ms
        return optimal_audit_probability(
            audit_opportunity_cost=opportunity,
            information_price=info_price,
            severity_bound=row.severity_bound,
            q_min=q_min,
            q_max=q_max,
        )

    for threshold, info_price in candidates:
        design_selected = [r for r in design_rows if r.score <= threshold]
        if not design_selected:
            continue
        expected_reward = 0.0
        design_q: list[float] = []
        for row in design_selected:
            q = q_for(row, info_price)
            design_q.append(q)
            expected_reward += (1 - q) * row.fast_gain_ms + q * row.audit_gain_ms
        expected_reward /= len(design_rows)

        cert_selected = [r for r in certificate_rows if r.score <= threshold]
        if not cert_selected:
            continue
        cert_q = np.asarray([q_for(r, info_price) for r in cert_selected])
        commit_weights = 1.0 - cert_q
        mass = float(commit_weights.sum())
        if mass < min_selected_mass:
            continue
        upper = weighted_hoeffding_upper(
            [r.severity for r in cert_selected],
            commit_weights,
            delta=per_candidate_delta,
            bound=max(r.severity_bound for r in cert_selected),
        )
        if upper > epsilon + 1e-12:
            continue
        counts: dict[str, int] = {}
        for row in cert_selected:
            counts[row.support_key] = counts.get(row.support_key, 0) + 1
        support = tuple(sorted(k for k, count in counts.items() if count >= min_support_count))
        if not support:
            continue
        certified.append(ThresholdAuditPlan(
            threshold=threshold,
            information_price=info_price,
            expected_saved_ms_per_round=expected_reward,
            screening_risk=upper,
            screening_selected_mass=mass,
            screening_method="heldout_bonferroni_hoeffding",
            mean_audit_probability=float(np.mean(design_q)),
            supported_keys=support,
            candidate_count=len(candidates),
        ))
    if not certified:
        return None
    return max(certified, key=lambda p: (p.expected_saved_ms_per_round, -p.heldout_risk_upper))


def design_then_certify_threshold_and_audit_plan(
    design_rows: Sequence[CalibrationRow],
    certificate_rows: Sequence[CalibrationRow],
    *,
    thresholds: Sequence[float],
    information_prices: Sequence[float],
    epsilon: float,
    delta: float,
    q_min: float,
    q_max: float,
    design_risk_margin: float = 0.0,
    min_selected_mass: float = 40.0,
    min_support_count: int = 4,
) -> ThresholdAuditPlan | None:
    """Choose one pair on design data, then test it once on fresh evidence.

    Unlike :func:`select_threshold_and_audit_plan`, this procedure never looks
    at certificate labels while choosing among candidates.  It therefore pays
    no grid multiplicity penalty, but it cannot try another pair on the same
    certificate split after a failure.
    """

    if not design_rows or not certificate_rows:
        return None
    candidates = [(float(t), float(lam)) for t in thresholds for lam in information_prices]
    if not candidates:
        return None

    def q_for(row: CalibrationRow, info_price: float) -> float:
        return optimal_audit_probability(
            audit_opportunity_cost=row.fast_gain_ms - row.audit_gain_ms,
            information_price=info_price,
            severity_bound=row.severity_bound,
            q_min=q_min,
            q_max=q_max,
        )

    ranked: list[tuple[float, float, float, float]] = []
    for threshold, info_price in candidates:
        selected = [r for r in design_rows if r.score <= threshold]
        if not selected:
            continue
        q_values = np.asarray([q_for(r, info_price) for r in selected])
        weights = 1.0 - q_values
        mass = float(weights.sum())
        if mass < min_selected_mass:
            continue
        risk = float(np.dot(weights, [r.severity for r in selected]) / mass)
        if risk > epsilon - design_risk_margin + 1e-12:
            continue
        reward = sum(
            (1 - q) * row.fast_gain_ms + q * row.audit_gain_ms
            for row, q in zip(selected, q_values)
        ) / len(design_rows)
        ranked.append((reward, -risk, threshold, info_price))
    if not ranked:
        return None
    _, _, threshold, info_price = max(ranked)

    cert_selected = [r for r in certificate_rows if r.score <= threshold]
    if not cert_selected:
        return None
    cert_q = np.asarray([q_for(r, info_price) for r in cert_selected])
    cert_weights = 1.0 - cert_q
    mass = float(cert_weights.sum())
    if mass < min_selected_mass:
        return None
    upper = weighted_betting_upper(
        [r.severity for r in cert_selected],
        cert_weights,
        delta=delta,
        bound=max(r.severity_bound for r in cert_selected),
    )
    if upper > epsilon + 1e-12:
        return None
    counts: dict[str, int] = {}
    for row in cert_selected:
        counts[row.support_key] = counts.get(row.support_key, 0) + 1
    support = tuple(sorted(k for k, count in counts.items() if count >= min_support_count))
    if not support:
        return None
    design_selected = [r for r in design_rows if r.score <= threshold]
    design_q = [q_for(r, info_price) for r in design_selected]
    reward = sum(
        (1 - q) * row.fast_gain_ms + q * row.audit_gain_ms
        for row, q in zip(design_selected, design_q)
    ) / len(design_rows)
    return ThresholdAuditPlan(
        threshold=threshold,
        information_price=info_price,
        expected_saved_ms_per_round=reward,
        screening_risk=upper,
        screening_selected_mass=mass,
        screening_method="one_shot_heldout_betting",
        mean_audit_probability=float(np.mean(design_q)),
        supported_keys=support,
        candidate_count=len(candidates),
    )


def select_design_only_threshold_and_audit_plan(
    design_rows: Sequence[CalibrationRow],
    support_rows: Sequence[CalibrationRow],
    *,
    thresholds: Sequence[float],
    information_prices: Sequence[float],
    epsilon: float,
    q_min: float,
    q_max: float,
    design_risk_margin: float = 0.0,
    min_selected_mass: float = 40.0,
    min_support_count: int = 4,
) -> ThresholdAuditPlan | None:
    """Tune on reusable historical labels, then use only unlabeled support.

    This constructor makes no coverage claim for its historical screening
    metric.  The model, threshold, and audit price may be selected after
    arbitrary reuse of ``design_rows``.  ``support_rows`` contribute only
    pre-coin features, cost forecasts, and support keys; their severity labels
    are never read.  Valid deployment coverage must start on fresh future audit
    coins after the returned plan is frozen.
    """

    if not design_rows or not support_rows:
        return None
    candidates = [(float(t), float(lam)) for t in thresholds for lam in information_prices]
    if not candidates:
        return None

    def q_for(row: CalibrationRow, info_price: float) -> float:
        return optimal_audit_probability(
            audit_opportunity_cost=row.fast_gain_ms - row.audit_gain_ms,
            information_price=info_price,
            severity_bound=row.severity_bound,
            q_min=q_min,
            q_max=q_max,
        )

    ranked: list[tuple[float, float, float, float, float, float]] = []
    for threshold, info_price in candidates:
        selected = [r for r in design_rows if r.score <= threshold]
        if not selected:
            continue
        q_values = np.asarray([q_for(r, info_price) for r in selected])
        weights = 1.0 - q_values
        mass = float(weights.sum())
        if mass < min_selected_mass:
            continue
        empirical_risk = float(np.dot(weights, [r.severity for r in selected]) / mass)
        if empirical_risk > epsilon - design_risk_margin + 1e-12:
            continue
        reward = sum(
            (1 - q) * row.fast_gain_ms + q * row.audit_gain_ms
            for row, q in zip(selected, q_values)
        ) / len(design_rows)
        ranked.append((reward, -empirical_risk, threshold, info_price, mass, float(np.mean(q_values))))
    if not ranked:
        return None
    reward, neg_risk, threshold, info_price, mass, mean_q = max(ranked)

    counts: dict[str, int] = {}
    for row in support_rows:
        if row.score <= threshold:
            counts[row.support_key] = counts.get(row.support_key, 0) + 1
    support = tuple(sorted(k for k, count in counts.items() if count >= min_support_count))
    if not support:
        return None
    return ThresholdAuditPlan(
        threshold=threshold,
        information_price=info_price,
        expected_saved_ms_per_round=reward,
        screening_risk=-neg_risk,
        screening_selected_mass=mass,
        screening_method="adaptive_design_only_no_coverage_claim",
        mean_audit_probability=mean_q,
        supported_keys=support,
        candidate_count=len(candidates),
    )


def select_historical_screen_threshold_and_audit_plan(
    design_rows: Sequence[CalibrationRow],
    later_screen_rows: Sequence[CalibrationRow],
    *,
    thresholds: Sequence[float],
    information_prices: Sequence[float],
    epsilon: float,
    q_min: float,
    q_max: float,
    empirical_margin: float = 0.0,
    min_selected_mass: float = 40.0,
    min_support_count: int = 4,
    local_screen_mass: float = 120.0,
) -> ThresholdAuditPlan | None:
    """Optimize with reused adaptive history without claiming calibration coverage.

    Both row sets may be feedback dependent, repeatedly inspected, and reused.
    The later rows provide a temporal stress screen and support cells. Their
    empirical risks are tuning diagnostics only; no IID, exchangeability, or
    fixed-time confidence statement is attached to them. Formal coverage starts
    from fresh future audit e-values after the returned policy is frozen.

    Support is risk screened cell by cell. A threshold may score a row as fast
    but the runtime still abstains unless its support key passed the later
    empirical cell screen. A recent-window screen prevents a low historical
    average from hiding a concentrated bad segment.
    """

    if not design_rows or not later_screen_rows:
        return None
    candidates = [(float(t), float(lam)) for t in thresholds for lam in information_prices]
    if not candidates:
        return None

    def q_for(row: CalibrationRow, info_price: float) -> float:
        return optimal_audit_probability(
            audit_opportunity_cost=row.fast_gain_ms - row.audit_gain_ms,
            information_price=info_price,
            severity_bound=row.severity_bound,
            q_min=q_min,
            q_max=q_max,
        )

    def weighted_risk(rows: Sequence[CalibrationRow], qs: np.ndarray) -> tuple[float, float]:
        weights = 1.0 - qs
        mass = float(weights.sum())
        if mass <= _EPS:
            return math.inf, mass
        risk = float(np.dot(weights, [r.severity for r in rows]) / mass)
        return risk, mass

    def max_recent_risk(
        ordered_rows: Sequence[CalibrationRow],
        info_price: float,
        support: set[str],
        threshold: float,
    ) -> float:
        if local_screen_mass <= 0:
            return 0.0
        window: list[tuple[float, float]] = []
        mass = 0.0
        loss = 0.0
        worst = 0.0
        for row in ordered_rows:
            if row.score > threshold or row.support_key not in support:
                continue
            weight = 1.0 - q_for(row, info_price)
            weighted_loss = weight * row.severity
            window.append((weight, weighted_loss))
            mass += weight
            loss += weighted_loss
            while mass > local_screen_mass and window:
                old_weight, old_loss = window.pop(0)
                mass -= old_weight
                loss -= old_loss
            if mass >= 0.5*local_screen_mass:
                worst = max(worst, loss / max(mass, _EPS))
        return worst

    accepted: list[ThresholdAuditPlan] = []
    limit = epsilon - empirical_margin
    for threshold, info_price in candidates:
        raw_screen = [r for r in later_screen_rows if r.score <= threshold]
        if not raw_screen:
            continue

        # A support key is enabled only when its own later empirical risk is
        # below the screen. This is a tuning guard, not a confidence statement.
        grouped: dict[str, list[CalibrationRow]] = {}
        for row in raw_screen:
            grouped.setdefault(row.support_key, []).append(row)
        support: set[str] = set()
        worst_cell = 0.0
        for key, group in grouped.items():
            if len(group) < min_support_count:
                continue
            group_q = np.asarray([q_for(r, info_price) for r in group])
            group_risk, group_mass = weighted_risk(group, group_q)
            if group_mass >= min_selected_mass/4.0 and group_risk <= limit + 1e-12:
                support.add(key)
                worst_cell = max(worst_cell, group_risk)
        if not support:
            continue

        design_selected = [
            r for r in design_rows
            if r.score <= threshold and r.support_key in support
        ]
        screen_selected = [r for r in raw_screen if r.support_key in support]
        if not design_selected or not screen_selected:
            continue
        design_q = np.asarray([q_for(r, info_price) for r in design_selected])
        screen_q = np.asarray([q_for(r, info_price) for r in screen_selected])
        design_risk, design_mass = weighted_risk(design_selected, design_q)
        screen_risk, screen_mass = weighted_risk(screen_selected, screen_q)
        if min(design_mass, screen_mass) < min_selected_mass:
            continue
        local_risk = max_recent_risk(
            later_screen_rows, info_price, support, threshold
        )
        observed_worst = max(design_risk, screen_risk, worst_cell, local_risk)
        if observed_worst > limit + 1e-12:
            continue
        reward = sum(
            (1 - q) * row.fast_gain_ms + q * row.audit_gain_ms
            for row, q in zip(design_selected, design_q)
        ) / len(design_rows)
        accepted.append(ThresholdAuditPlan(
            threshold=threshold,
            information_price=info_price,
            expected_saved_ms_per_round=reward,
            screening_risk=observed_worst,
            screening_selected_mass=screen_mass,
            screening_method="adaptive_historical_cell_and_window_screen_no_coverage_claim",
            mean_audit_probability=float(np.mean(design_q)),
            supported_keys=tuple(sorted(support)),
            candidate_count=len(candidates),
        ))
    if not accepted:
        return None
    return max(accepted, key=lambda p: (p.expected_saved_ms_per_round, -p.screening_risk))


def aggregate_block_severity(
    row_losses: Sequence[float],
    *,
    row_weights: Sequence[float] | None = None,
    collapse: bool = False,
    collapse_weight: float = 1.0,
    bound: float = 1.0,
) -> float:
    """Aggregate one causally coupled DFlash block into one audit outcome.

    Rows after an early divergence are not treated as independent examples.  A
    block receives one propensity and one severity label.
    """

    losses = np.asarray(row_losses, dtype=float)
    if np.any(losses < 0):
        raise ValueError("row losses must be non-negative")
    if row_weights is None:
        weights = np.ones_like(losses)
    else:
        weights = np.asarray(row_weights, dtype=float)
        if weights.shape != losses.shape or np.any(weights < 0):
            raise ValueError("invalid row weights")
    total = float(weights @ losses)
    if collapse:
        total += collapse_weight
    return float(np.clip(total, 0, bound))


@dataclass(frozen=True)
class FeatureOutcome:
    probability: float
    next_state: str


@dataclass(frozen=True)
class FeatureAcquisition:
    name: str
    cost_ms: float
    outcomes: tuple[FeatureOutcome, ...]


@dataclass(frozen=True)
class FeatureState:
    name: str
    fast_gain_ms: float
    risk_upper: float
    certificate_valid: bool
    acquisitions: tuple[FeatureAcquisition, ...] = ()


@dataclass(frozen=True)
class FeaturePolicyEntry:
    state: str
    action: str
    value_ms: float


def solve_sequential_feature_acquisition(
    states: Mapping[str, FeatureState],
    *,
    root: str,
    epsilon: float,
) -> tuple[float, tuple[FeaturePolicyEntry, ...]]:
    r"""Solve a finite acyclic exact/fast/feature decision tree by Bellman recursion.

    Exact fallback has value zero.  A terminal fast action is available only
    when its state-specific risk certificate is valid and at most epsilon.
    Feature or partial-exact action j has value

        -cost_j + sum_o p(o) V(next_o).

    Cycles are rejected because a finite feature-acquisition protocol must have
    a hard work bound.
    """

    if root not in states:
        raise KeyError(root)
    memo: dict[str, float] = {}
    actions: dict[str, str] = {}
    visiting: set[str] = set()

    def solve(name: str) -> float:
        if name in memo:
            return memo[name]
        if name in visiting:
            raise ValueError("feature acquisition graph contains a cycle")
        visiting.add(name)
        state = states[name]
        best_value = 0.0
        best_action = "exact"
        if state.certificate_valid and state.risk_upper <= epsilon + 1e-12:
            if state.fast_gain_ms > best_value:
                best_value = state.fast_gain_ms
                best_action = "fast"
        for acquisition in state.acquisitions:
            if acquisition.cost_ms < 0:
                raise ValueError("feature costs must be non-negative")
            probability = sum(o.probability for o in acquisition.outcomes)
            if abs(probability - 1.0) > 1e-9 or any(o.probability < 0 for o in acquisition.outcomes):
                raise ValueError("feature outcomes must form a probability distribution")
            value = -acquisition.cost_ms
            for outcome in acquisition.outcomes:
                if outcome.next_state not in states:
                    raise KeyError(outcome.next_state)
                value += outcome.probability * solve(outcome.next_state)
            if value > best_value + 1e-12:
                best_value = value
                best_action = f"acquire:{acquisition.name}"
        visiting.remove(name)
        memo[name] = best_value
        actions[name] = best_action
        return best_value

    root_value = solve(root)
    entries = tuple(
        FeaturePolicyEntry(state=name, action=actions[name], value_ms=memo[name])
        for name in sorted(memo)
    )
    return root_value, entries


@dataclass(frozen=True)
class NoOverlapLogRow:
    round_id: int
    state: str
    action: str
    observed_exact_loss: float
    next_state: str


class NoOverlapEnvironment:
    """Two worlds with identical exact-policy logs and opposite fast risk."""

    def __init__(self, *, hidden_fast_bad: bool) -> None:
        self.hidden_fast_bad = bool(hidden_fast_bad)

    def log_exact(self, rounds: int = 32) -> tuple[NoOverlapLogRow, ...]:
        state = "anchor"
        rows: list[NoOverlapLogRow] = []
        for t in range(rounds):
            rows.append(NoOverlapLogRow(t, state, "exact", 0.0, "anchor"))
            state = "anchor"
        return tuple(rows)

    def deploy_fast(self, rounds: int = 8) -> dict[str, Any]:
        state = "anchor"
        loss = 0.0
        visited: list[str] = []
        for _ in range(rounds):
            visited.append(state)
            if self.hidden_fast_bad:
                loss += 1.0
                state = "collapsed"
            else:
                state = "anchor"
        return {
            "fast_rounds": rounds,
            "cumulative_loss": loss,
            "selective_risk": loss / max(rounds, 1),
            "final_state": state,
            "visited_states": visited,
        }


def indistinguishable_environment_demo(rounds: int = 32) -> dict[str, Any]:
    good = NoOverlapEnvironment(hidden_fast_bad=False)
    bad = NoOverlapEnvironment(hidden_fast_bad=True)
    good_log = [asdict(r) for r in good.log_exact(rounds)]
    bad_log = [asdict(r) for r in bad.log_exact(rounds)]
    good_bytes = json.dumps(good_log, sort_keys=True, separators=(",", ":")).encode()
    bad_bytes = json.dumps(bad_log, sort_keys=True, separators=(",", ":")).encode()
    return {
        "logging_policy": "always_exact_no_shadow",
        "logs_identical": good_bytes == bad_bytes,
        "good_log_sha256": hashlib.sha256(good_bytes).hexdigest(),
        "bad_log_sha256": hashlib.sha256(bad_bytes).hexdigest(),
        "good_deployment": good.deploy_fast(),
        "bad_deployment": bad.deploy_fast(),
    }


def _demo_ledger(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    fingerprint = canonical_fingerprint({
        "controller": "demo-v1",
        "verifier": "exact-v1",
        "loss": "block-severity-v1",
    })
    ledger = RandomAuditLedger(
        delta=0.05,
        epsilon=0.08,
        startup_budget=8.0,
        q_min=0.15,
        state_fingerprint=fingerprint,
    )
    labels_due: list[tuple[int, int, float]] = []
    true_committed = 0.0
    true_intended_loss = 0.0
    intended_weight = 0.0
    any_budget_violation = False
    any_coverage_violation = False
    for t in range(160):
        for _, round_id, label in [x for x in labels_due if x[0] == t]:
            ledger.resolve(round_id, label)
        labels_due = [x for x in labels_due if x[0] != t]
        q = 0.22
        if not ledger.can_commit_fast(severity_bound=1.0):
            q = 1.0
        audited, uniform = draw_random_audit(rng, q)
        draw = rng.random()
        severity = 1.0 if draw < 0.012 else (0.08 if draw < 0.12 else 0.0)
        record = AuditRecord(
            round_id=t,
            audit_probability=q,
            severity_bound=1.0,
            audited=audited,
            audit_uniform=uniform,
            state_fingerprint=fingerprint,
            runtime_state_fingerprint=canonical_fingerprint({
                "certificate": fingerprint, "round": t, "q": q, "bound": 1.0
            }),
            support_key="demo",
            label=None,
            label_due_round=t + 1 if audited else None,
        )
        ledger.append(record)
        true_intended_loss += (1 - q) * severity
        intended_weight += 1 - q
        if audited:
            labels_due.append((t + 1, t, severity))
        else:
            true_committed += severity
        snapshot = ledger.snapshot()
        any_budget_violation |= true_committed > snapshot.risk_budget + 1e-12
        any_coverage_violation |= true_committed > snapshot.committed_loss_upper + 1e-12
    for _, round_id, label in sorted(labels_due):
        ledger.resolve(round_id, label)
    snapshot = ledger.snapshot()
    true_intended_risk = true_intended_loss / max(intended_weight, _EPS)
    return {
        "true_committed_loss": true_committed,
        "true_intended_selective_risk": true_intended_risk,
        "committed_covered": true_committed <= snapshot.committed_loss_upper + 1e-12,
        "intended_covered": true_intended_risk <= snapshot.intended_selective_risk_upper + 1e-12,
        "any_budget_violation": any_budget_violation,
        "any_coverage_violation": any_coverage_violation,
        "ledger": ledger.to_json_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    payload = {
        "impossibility": indistinguishable_environment_demo(),
        "ledger_demo": _demo_ledger(args.seed),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
