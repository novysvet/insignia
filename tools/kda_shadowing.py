#!/usr/bin/env python3
"""Deterministic CPU shadowing laboratory for recurrent KDA state.

The state recurrence is

    S_t = A_t S_{t-1} B_t + beta_t u_t v_t^T.

The approximate path may perturb every coefficient and apply a deterministic
finite-precision map to the new state.  This module provides:

* exact error decompositions;
* minimax operator-norm, transition-product, diagonal, and KDA Lyapunov bounds;
* an online scalar certificate that does not compare against the exact state;
* deterministic contractive, marginal, switching-nonnormal, and tight
  adversarial sequences for d <= 32;
* exact SymPy checks for d <= 4;
* exact finite-horizon reset solvers and the causal lazy optimum;
* a max-scaled E4M3FN state quantizer with a discontinuous power-of-two scale.

Run from the repository root:

    python tools/kda_shadowing.py --out scratch/kda-shadowing
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:  # The repository already owns the bit-exact E4M3FN encoder.
    from quantize_dflash2 import encode_e4m3fn
except ModuleNotFoundError:  # Useful when imported as tools.kda_shadowing.
    from tools.quantize_dflash2 import encode_e4m3fn

Array = np.ndarray
Quantizer = Callable[[Array], tuple[Array, dict[str, float]]]
_EPS = 1.0e-12


@dataclass(frozen=True)
class StepPair:
    """One exact/approximate recurrence step."""

    A: Array
    B: Array
    beta: float
    u: Array
    v: Array
    hat_A: Array
    hat_B: Array
    hat_beta: float
    hat_u: Array
    hat_v: Array
    quantizer: Quantizer | None = None
    label: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: tuple[StepPair, ...]
    S0: Array
    hat_S0: Array
    description: str


@dataclass(frozen=True)
class ErrorTerms:
    propagated: Array
    coefficient: Array
    update: Array
    rounding: Array

    @property
    def local_defect(self) -> Array:
        return self.coefficient + self.update + self.rounding

    @property
    def reconstructed(self) -> Array:
        return self.propagated + self.local_defect


@dataclass(frozen=True)
class ResetSolution:
    boundaries: tuple[int, ...]
    resets: int
    net_saving: float
    segment_peaks: tuple[float, ...]
    final_exact: bool


# ---------------------------------------------------------------------------
# Basic recurrence and exact decompositions
# ---------------------------------------------------------------------------


def _as_square(name: str, value: Array, d: int | None = None) -> Array:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    if d is not None and value.shape != (d, d):
        raise ValueError(f"{name} has shape {value.shape}, expected {(d, d)}")
    return value


def recurrence_step(S: Array, A: Array, B: Array, beta: float, u: Array, v: Array) -> Array:
    S = _as_square("S", S)
    d = S.shape[0]
    A = _as_square("A", A, d)
    B = _as_square("B", B, d)
    u = np.asarray(u, dtype=np.float64).reshape(d)
    v = np.asarray(v, dtype=np.float64).reshape(d)
    return A @ S @ B + float(beta) * np.outer(u, v)


def rank_one_update_error_exact(step: StepPair) -> Array:
    return (step.hat_beta * np.outer(step.hat_u, step.hat_v)
            - step.beta * np.outer(step.u, step.v))


def rank_one_update_error_bound(step: StepPair) -> float:
    """Triangle bound from an exact three-term telescoping decomposition."""

    db = abs(step.hat_beta - step.beta)
    du = np.linalg.norm(step.hat_u - step.u)
    dv = np.linalg.norm(step.hat_v - step.v)
    return float(
        db * np.linalg.norm(step.u) * np.linalg.norm(step.v)
        + abs(step.hat_beta) * du * np.linalg.norm(step.v)
        + abs(step.hat_beta) * np.linalg.norm(step.hat_u) * dv
    )


def exact_error_terms(
    S_prev: Array,
    hat_S_prev: Array,
    S_next: Array,
    hat_S_next: Array,
    step: StepPair,
    rounding: Array,
) -> ErrorTerms:
    """Exact-propagator decomposition.

    With E_A = hat_A-A and E_B = hat_B-B,

      Delta_t = A Delta_{t-1} B
              + E_A hat_S_{t-1} hat_B + A hat_S_{t-1} E_B
              + (hat_R-R) + q_t.

    This form is useful when a measured/fused norm of hat_S is available.
    """

    delta_prev = hat_S_prev - S_prev
    E_A = step.hat_A - step.A
    E_B = step.hat_B - step.B
    propagated = step.A @ delta_prev @ step.B
    coefficient = E_A @ hat_S_prev @ step.hat_B + step.A @ hat_S_prev @ E_B
    update = rank_one_update_error_exact(step)
    terms = ErrorTerms(propagated, coefficient, update, rounding)
    delta_next = hat_S_next - S_next
    if not np.allclose(terms.reconstructed, delta_next, rtol=2e-10, atol=2e-10):
        mismatch = np.linalg.norm(terms.reconstructed - delta_next)
        raise AssertionError(f"exact error decomposition failed by {mismatch:.3e}")
    return terms


def approximate_propagator_error_terms(
    S_prev: Array,
    hat_S_prev: Array,
    S_next: Array,
    hat_S_next: Array,
    step: StepPair,
    rounding: Array,
) -> ErrorTerms:
    """Exact decomposition propagated by the approximate gates.

      Delta_t = hat_A Delta_{t-1} hat_B
              + E_A S_{t-1} B + A S_{t-1} E_B + E_A S_{t-1} E_B
              + (hat_R-R) + q_t.

    This form supports a no-state-scan online certificate when an upper bound
    on ||S_{t-1}||_F is tracked recursively.
    """

    delta_prev = hat_S_prev - S_prev
    E_A = step.hat_A - step.A
    E_B = step.hat_B - step.B
    propagated = step.hat_A @ delta_prev @ step.hat_B
    coefficient = (E_A @ S_prev @ step.B
                   + step.A @ S_prev @ E_B
                   + E_A @ S_prev @ E_B)
    update = rank_one_update_error_exact(step)
    terms = ErrorTerms(propagated, coefficient, update, rounding)
    delta_next = hat_S_next - S_next
    if not np.allclose(terms.reconstructed, delta_next, rtol=2e-10, atol=2e-10):
        mismatch = np.linalg.norm(terms.reconstructed - delta_next)
        raise AssertionError(f"approximate-propagator decomposition failed by {mismatch:.3e}")
    return terms


# ---------------------------------------------------------------------------
# General finite-horizon bounds
# ---------------------------------------------------------------------------


def scalar_operator_bound_step(
    radius: float,
    A_norm: float,
    B_norm: float,
    coefficient_norm: float,
    update_norm: float,
    rounding_norm: float,
) -> float:
    return (A_norm * B_norm * radius
            + coefficient_norm + update_norm + rounding_norm)


def finite_horizon_product_bound(
    initial_radius: float,
    gains: Sequence[float],
    local_errors: Sequence[float],
) -> list[float]:
    """The minimax-sharp scalar convolution under only per-step norm caps."""

    if len(gains) != len(local_errors):
        raise ValueError("gains and local_errors must have equal length")
    value = float(initial_radius)
    trace: list[float] = []
    for gain, error in zip(gains, local_errors):
        if gain < 0 or error < 0:
            raise ValueError("bounds must be nonnegative")
        value = float(gain) * value + float(error)
        trace.append(value)
    return trace


def transition_product_bound(
    delta0: Array,
    As: Sequence[Array],
    Bs: Sequence[Array],
    defects: Sequence[Array],
) -> list[float]:
    """Sharper oracle bound using norms of actual transition products.

    This still applies triangle inequalities to each injected defect, but it
    keeps cancellations/rotations inside products A_t...A_j and B_j...B_t.
    It is not the cheap production certificate.
    """

    if not (len(As) == len(Bs) == len(defects)):
        raise ValueError("transition lists must have equal length")
    d = delta0.shape[0]
    out: list[float] = []
    for end in range(len(As)):
        left = np.eye(d)
        for k in range(end, -1, -1):
            left = left @ As[k]
        right = np.eye(d)
        for k in range(0, end + 1):
            right = right @ Bs[k]
        value = (np.linalg.norm(left, 2) * np.linalg.norm(right, 2)
                 * np.linalg.norm(delta0, "fro"))
        for injection in range(end + 1):
            left_tail = np.eye(d)
            for k in range(end, injection, -1):
                left_tail = left_tail @ As[k]
            right_tail = np.eye(d)
            for k in range(injection + 1, end + 1):
                right_tail = right_tail @ Bs[k]
            value += (np.linalg.norm(left_tail, 2)
                      * np.linalg.norm(right_tail, 2)
                      * np.linalg.norm(defects[injection], "fro"))
        out.append(float(value))
    return out


def diagonal_majorant_step(radius: Array, A: Array, B: Array, defect: Array) -> Array:
    """Entrywise-sharp majorant when A and B are diagonal."""

    if not (np.allclose(A, np.diag(np.diag(A)))
            and np.allclose(B, np.diag(np.diag(B)))):
        raise ValueError("diagonal majorant requires diagonal gates")
    return (np.abs(np.diag(A))[:, None] * np.abs(np.diag(B))[None, :]
            * radius + np.abs(defect))


# ---------------------------------------------------------------------------
# KDA specialization and Lyapunov identity
# ---------------------------------------------------------------------------


def normalized(vector: Array) -> Array:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm <= _EPS:
        out = np.zeros_like(vector)
        out[0] = 1.0
        return out
    return vector / norm


def kda_left_gate(k: Array, beta: float, decay: Array) -> Array:
    """A = (I-beta k k^T) diag(decay), the GLM KDA state map."""

    k = np.asarray(k, dtype=np.float64)
    decay = np.asarray(decay, dtype=np.float64)
    d = k.size
    if decay.shape != (d,):
        raise ValueError("decay and key dimensions disagree")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    if np.linalg.norm(k) > 1.0 + 1e-10:
        raise ValueError("KDA key must have norm at most one")
    if np.any(decay < 0.0) or np.any(decay > 1.0 + 1e-12):
        raise ValueError("KDA decays must lie in [0,1]")
    return (np.eye(d) - beta * np.outer(k, k)) @ np.diag(decay)


def kda_energy_terms(X: Array, k: Array, beta: float, decay: Array) -> dict[str, float]:
    """Return both sides of the exact homogeneous Lyapunov identity."""

    X = np.asarray(X, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    decay = np.asarray(decay, dtype=np.float64)
    D_X = decay[:, None] * X
    k2 = float(k @ k)
    projection_coeff = 2.0 * beta - beta * beta * k2
    decay_dissipation = float(np.sum((1.0 - decay * decay)[:, None] * X * X))
    projection_dissipation = float(projection_coeff * np.linalg.norm(k @ D_X) ** 2)
    A_X = kda_left_gate(k, beta, decay) @ X
    before = float(np.linalg.norm(X, "fro") ** 2)
    after = float(np.linalg.norm(A_X, "fro") ** 2)
    rhs = before - decay_dissipation - projection_dissipation
    return {
        "before": before,
        "after": after,
        "decay_dissipation": decay_dissipation,
        "projection_dissipation": projection_dissipation,
        "identity_rhs": rhs,
        "identity_error": abs(after - rhs),
    }


def kda_driven_energy_terms(
    S: Array,
    k: Array,
    beta: float,
    decay: Array,
    v: Array,
) -> dict[str, float]:
    """Exact energy identity for the full driven KDA update.

    With Y=D S, m=k^T Y, and z=v-m,

      ||Y + beta k z^T||_F^2
        = ||Y||_F^2 - beta ||m||_2^2 + beta ||v||_2^2
          - beta (1-beta ||k||_2^2) ||z||_2^2.

    The final term is nonnegative under the KDA contract.  Dropping both
    negative terms gives the cheap state-norm majorant

      M_t^2 <= rho_t^2 M_{t-1}^2 + beta_t ||v_t||_2^2.
    """

    S = np.asarray(S, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    decay = np.asarray(decay, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    Y = decay[:, None] * S
    memory = k @ Y
    innovation = v - memory
    next_state = Y + beta * np.outer(k, innovation)
    k2 = float(k @ k)
    before_decay = float(np.linalg.norm(Y, "fro") ** 2)
    after = float(np.linalg.norm(next_state, "fro") ** 2)
    rhs = (before_decay
           - beta * float(np.linalg.norm(memory) ** 2)
           + beta * float(np.linalg.norm(v) ** 2)
           - beta * (1.0 - beta * k2) * float(np.linalg.norm(innovation) ** 2))
    return {
        "decayed_energy": before_decay,
        "after": after,
        "memory_norm": float(np.linalg.norm(memory)),
        "innovation_norm": float(np.linalg.norm(innovation)),
        "identity_rhs": rhs,
        "identity_error": abs(after - rhs),
    }


def kda_state_energy_majorant_step(
    state_bound: float,
    decay_cap: float,
    beta_upper: float,
    v_norm_bound: float,
) -> float:
    """Update a Frobenius state bound using the driven KDA energy identity."""

    if min(state_bound, decay_cap, beta_upper, v_norm_bound) < 0:
        raise ValueError("KDA energy-majorant inputs must be nonnegative")
    if decay_cap > 1.0 + 1e-12 or beta_upper > 1.0 + 1e-12:
        raise ValueError("KDA energy majorant requires decay,beta <= 1")
    return math.sqrt(
        (decay_cap * state_bound) ** 2 + beta_upper * v_norm_bound ** 2
    )


def kda_propagation_cap(decay_upper: Array) -> float:
    decay_upper = np.asarray(decay_upper, dtype=np.float64)
    return float(np.max(np.abs(decay_upper)))


def kda_map_perturbation_bound(
    k: Array,
    beta: float,
    decay: Array,
    hat_k: Array,
    hat_beta: float,
    hat_decay: Array,
) -> float:
    """Structured ||hat_A-A||_2 bound without forming a dxd SVD.

    hat_A-A = hat_P(hat_D-D) + (hat_P-P)D, and both P factors are
    nonexpansive for beta in [0,1] and ||k|| <= 1.
    """

    k = np.asarray(k, dtype=np.float64)
    hat_k = np.asarray(hat_k, dtype=np.float64)
    decay = np.asarray(decay, dtype=np.float64)
    hat_decay = np.asarray(hat_decay, dtype=np.float64)
    eps_D = float(np.max(np.abs(hat_decay - decay)))
    eps_k = float(np.linalg.norm(hat_k - k))
    proj = (abs(hat_beta - beta) * np.linalg.norm(k) ** 2
            + abs(hat_beta) * eps_k * np.linalg.norm(k)
            + abs(hat_beta) * np.linalg.norm(hat_k) * eps_k)
    return eps_D + proj * float(np.max(np.abs(decay)))


def decay_interval_upper(hat_log_decay: Array, log_error_inf: float) -> Array:
    """Safe exact-decay upper endpoint under |hat_g-g| <= log_error_inf."""

    hat_log_decay = np.asarray(hat_log_decay, dtype=np.float64)
    if log_error_inf < 0:
        raise ValueError("negative error radius")
    # The model gate contract caps exact log-decay at zero.
    return np.exp(np.minimum(0.0, hat_log_decay + log_error_inf))


# ---------------------------------------------------------------------------
# Online certificate
# ---------------------------------------------------------------------------


def exact_state_norm_majorant_step(
    state_bound: float,
    A_norm_bound: float,
    B_norm_bound: float,
    beta_abs_bound: float,
    u_norm_bound: float,
    v_norm_bound: float,
) -> float:
    return (A_norm_bound * B_norm_bound * state_bound
            + beta_abs_bound * u_norm_bound * v_norm_bound)


def online_error_certificate_step(
    error_bound: float,
    exact_state_bound: float,
    *,
    A_norm_bound: float,
    B_norm_bound: float,
    hat_A_norm_bound: float,
    hat_B_norm_bound: float,
    A_error_bound: float,
    B_error_bound: float,
    update_error_bound: float,
    rounding_error_bound: float,
) -> float:
    """No-exact-state-scan certificate based on the approximate propagator.

    The coefficient term is bounded by

      (eps_A ||B|| + ||A|| eps_B + eps_A eps_B) ||S||_F.
    """

    coefficient = (A_error_bound * B_norm_bound
                   + A_norm_bound * B_error_bound
                   + A_error_bound * B_error_bound) * exact_state_bound
    return (hat_A_norm_bound * hat_B_norm_bound * error_bound
            + coefficient + update_error_bound + rounding_error_bound)


def future_radius_envelope(
    current_radius: float,
    future_gain_bounds: Sequence[float],
    future_local_error_bounds: Sequence[float],
) -> float:
    return finite_horizon_product_bound(
        current_radius, future_gain_bounds, future_local_error_bounds
    )[-1] if future_gain_bounds else float(current_radius)


# ---------------------------------------------------------------------------
# E4M3FN block quantization and discontinuous scale
# ---------------------------------------------------------------------------


def decode_e4m3fn(codes: Array) -> Array:
    codes = np.asarray(codes, dtype=np.uint8)
    sign = np.where((codes & np.uint8(0x80)) != 0, -1.0, 1.0)
    mag = (codes & np.uint8(0x7F)).astype(np.int32)
    if np.any(mag == 0x7F):
        raise ValueError("E4M3FN NaN code cannot be decoded as finite")
    exponent = mag >> 3
    mantissa = mag & 7
    values = np.empty(codes.shape, dtype=np.float64)
    sub = exponent == 0
    values[sub] = mantissa[sub] * (2.0 ** -9)
    values[~sub] = (1.0 + mantissa[~sub] / 8.0) * np.exp2(exponent[~sub] - 7)
    return sign * values


def _e4m3_positive_codebook() -> Array:
    return decode_e4m3fn(np.arange(0x7F, dtype=np.uint8))


_E4M3_POSITIVE = _e4m3_positive_codebook()
_E4M3_HALF_MAX_GAP = float(np.max(np.diff(_E4M3_POSITIVE)) / 2.0)
_E4M3_MAX = 448.0


def fp8_block_quantize(
    values: Array,
    *,
    power_of_two_scale: bool = True,
) -> tuple[Array, dict[str, float]]:
    """Max-scaled E4M3FN with an optional discontinuous power-of-two scale.

    Power-of-two scale rounds upward, so no finite input overflows.  The
    returned Frobenius error bound is deterministic and independent of scale
    continuity: every normalized value lies in an E4M3 nearest-neighbor cell.
    """

    source = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("non-finite quantizer input")
    amax = float(np.max(np.abs(source))) if source.size else 0.0
    if amax == 0.0:
        scale = 1.0
    else:
        ideal = amax / _E4M3_MAX
        if power_of_two_scale:
            scale = float(2.0 ** math.ceil(math.log2(ideal)))
        else:
            scale = ideal
    normalized_values = np.asarray(source / scale, dtype=np.float32)
    np.clip(normalized_values, -_E4M3_MAX, _E4M3_MAX, out=normalized_values)
    codes = encode_e4m3fn(normalized_values)
    restored = decode_e4m3fn(codes) * scale
    error = restored - source
    if power_of_two_scale and amax > 0:
        upper = _E4M3_MAX * scale
        lower = upper / 2.0
        scale_margin = min(max(0.0, upper - amax), max(0.0, amax - lower))
    else:
        scale_margin = math.inf
    info = {
        "scale": scale,
        "amax": amax,
        "scale_margin": scale_margin,
        "rounding_norm": float(np.linalg.norm(error, "fro")),
        "rounding_bound": float(_E4M3_HALF_MAX_GAP * scale * math.sqrt(source.size)),
        "saturated": float(np.count_nonzero(np.abs(normalized_values) >= _E4M3_MAX)),
    }
    return restored, info


def uniform_quantizer(step_size: float) -> Quantizer:
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    def apply(values: Array) -> tuple[Array, dict[str, float]]:
        restored = np.rint(np.asarray(values) / step_size) * step_size
        error = restored - values
        return restored, {
            "scale": step_size,
            "amax": float(np.max(np.abs(values))),
            "scale_margin": math.inf,
            "rounding_norm": float(np.linalg.norm(error, "fro")),
            "rounding_bound": float(0.5 * step_size * math.sqrt(values.size)),
            "saturated": 0.0,
        }

    return apply


def fp8_scale_discontinuity_demo(epsilon: float = 1.0e-7) -> dict[str, float]:
    low = np.array([[448.0 - epsilon, 0.0019]], dtype=np.float64)
    high = np.array([[448.0 + epsilon, 0.0019]], dtype=np.float64)
    q_low, i_low = fp8_block_quantize(low, power_of_two_scale=True)
    q_high, i_high = fp8_block_quantize(high, power_of_two_scale=True)
    input_gap = float(np.linalg.norm(high - low))
    output_gap = float(np.linalg.norm(q_high - q_low))
    return {
        "epsilon": epsilon,
        "low_scale": i_low["scale"],
        "high_scale": i_high["scale"],
        "input_gap": input_gap,
        "output_gap": output_gap,
        "local_lipschitz_ratio": output_gap / input_gap,
        "low_small_value": float(q_low[0, 1]),
        "high_small_value": float(q_high[0, 1]),
        "low_rounding_bound": i_low["rounding_bound"],
        "high_rounding_bound": i_high["rounding_bound"],
    }


def fp8_rank_obstruction_demo() -> dict[str, Any]:
    """A rank-one matrix whose E4M3 block image has rank two.

    This makes the backward-error obstruction concrete: after state
    quantization, a single rank-one recurrence update cannot generally explain
    the stored state by perturbing only beta, u, and v.
    """

    left = np.array([1.0 / 10.0, 29.0 / 195.0], dtype=np.float64)
    right = np.array([1.0 / 10.0, 16.0 / 65.0], dtype=np.float64)
    source = np.outer(left, right)
    restored, info = fp8_block_quantize(source, power_of_two_scale=True)
    expected = np.array([
        [5.0 / 512.0, 13.0 / 512.0],
        [15.0 / 1024.0, 9.0 / 256.0],
    ])
    if not np.array_equal(restored, expected):
        raise AssertionError("E4M3 rank-obstruction codes changed")
    return {
        "source": source.tolist(),
        "quantized": restored.tolist(),
        "scale": info["scale"],
        "source_rank": int(np.linalg.matrix_rank(source, tol=1e-14)),
        "quantized_rank": int(np.linalg.matrix_rank(restored, tol=1e-14)),
        "quantized_determinant": float(np.linalg.det(restored)),
    }


# ---------------------------------------------------------------------------
# Deterministic sequence generators
# ---------------------------------------------------------------------------


def _zero_update(d: int) -> tuple[float, Array, Array]:
    return 0.0, np.zeros(d), np.zeros(d)


def make_contractive_kda(d: int = 8, steps: int = 24, seed: int = 7) -> Scenario:
    if not 1 <= d <= 32:
        raise ValueError("d must lie in [1,32]")
    rng = np.random.default_rng(seed)
    sequence: list[StepPair] = []
    for token in range(steps):
        k = normalized(rng.normal(size=d))
        v = rng.normal(scale=0.25, size=d)
        beta = float(0.15 + 0.7 * rng.random())
        decay = rng.uniform(0.72, 0.91, size=d)
        hat_k = normalized(k + rng.normal(scale=2e-3, size=d))
        hat_v = v + rng.normal(scale=1e-3, size=d)
        hat_beta = float(np.clip(beta + rng.normal(scale=8e-4), 0.0, 1.0))
        hat_decay = np.clip(decay + rng.normal(scale=4e-4, size=d), 0.0, 1.0)
        A = kda_left_gate(k, beta, decay)
        hat_A = kda_left_gate(hat_k, hat_beta, hat_decay)
        sequence.append(StepPair(
            A=A, B=np.eye(d), beta=beta, u=k, v=v,
            hat_A=hat_A, hat_B=np.eye(d), hat_beta=hat_beta,
            hat_u=hat_k, hat_v=hat_v,
            quantizer=uniform_quantizer(2.0e-5), label="contractive-kda",
            metadata={"decay": decay, "hat_decay": hat_decay, "token": token},
        ))
    zero = np.zeros((d, d), dtype=np.float64)
    return Scenario(
        "contractive", tuple(sequence), zero.copy(), zero.copy(),
        "Actual KDA structure with strict diagonal decay, coefficient noise, and state rounding.",
    )


def _fp8_power2_quantizer(values: Array) -> tuple[Array, dict[str, float]]:
    return fp8_block_quantize(values, power_of_two_scale=True)


def make_fp8_kda(d: int = 8, steps: int = 24, seed: int = 7) -> Scenario:
    """The contractive KDA sequence with discontinuous block-scaled E4M3 state."""

    base = make_contractive_kda(d=d, steps=steps, seed=seed)
    sequence = tuple(StepPair(
        A=step.A,
        B=step.B,
        beta=step.beta,
        u=step.u,
        v=step.v,
        hat_A=step.hat_A,
        hat_B=step.hat_B,
        hat_beta=step.hat_beta,
        hat_u=step.hat_u,
        hat_v=step.hat_v,
        quantizer=_fp8_power2_quantizer,
        label=step.label,
        metadata=step.metadata,
    ) for step in base.steps)
    return Scenario(
        "contractive-fp8",
        sequence,
        base.S0.copy(),
        base.hat_S0.copy(),
        "Actual KDA structure with coefficient noise and power-of-two max-scaled E4M3FN state.",
    )


def make_marginal(d: int = 8, steps: int = 24) -> Scenario:
    if not 1 <= d <= 32:
        raise ValueError("d must lie in [1,32]")
    e = np.zeros(d)
    e[0] = 1.0
    sequence: list[StepPair] = []
    for token in range(steps):
        eta = 2.0e-4 * (1.0 + (token % 3) / 4.0)
        sequence.append(StepPair(
            A=np.eye(d), B=np.eye(d), beta=0.0, u=e, v=np.zeros(d),
            hat_A=np.eye(d), hat_B=np.eye(d), hat_beta=1.0,
            hat_u=e, hat_v=eta * e,
            label="marginal", metadata={"token": token},
        ))
    zero = np.zeros((d, d), dtype=np.float64)
    return Scenario(
        "marginal", tuple(sequence), zero.copy(), zero.copy(),
        "Identity propagation: aligned rank-one local defects accumulate linearly.",
    )


def make_switching_nonnormal(d: int = 2, steps: int = 12, r: float = 0.5, shear: float = 2.0) -> Scenario:
    if not 2 <= d <= 32:
        raise ValueError("switching example needs 2 <= d <= 32")
    A1 = np.eye(d) * r
    A2 = np.eye(d) * r
    A1[0, 1] = shear
    A2[1, 0] = shear
    beta, u, v = _zero_update(d)
    sequence = tuple(StepPair(
        A=(A1 if token % 2 == 0 else A2), B=np.eye(d), beta=beta, u=u, v=v,
        hat_A=(A1 if token % 2 == 0 else A2), hat_B=np.eye(d),
        hat_beta=beta, hat_u=u, hat_v=v,
        label="switching-nonnormal", metadata={"token": token},
    ) for token in range(steps))
    S0 = np.zeros((d, d), dtype=np.float64)
    hat_S0 = S0.copy()
    hat_S0[1, 0] = 1.0e-6
    return Scenario(
        "switching-nonnormal", sequence, S0, hat_S0,
        "Each step has spectral radius r<1, but alternating shears amplify the carried error.",
    )


def make_adversarial_tight(d: int = 4, steps: int = 12, gain: float = 1.08, eta: float = 1e-4) -> Scenario:
    if not 1 <= d <= 32:
        raise ValueError("d must lie in [1,32]")
    e = np.zeros(d)
    e[0] = 1.0
    A = gain * np.eye(d)
    B = np.eye(d)
    sequence = tuple(StepPair(
        A=A, B=B, beta=0.0, u=e, v=np.zeros(d),
        hat_A=A, hat_B=B, hat_beta=1.0, hat_u=e, hat_v=eta * e,
        label="adversarial-tight", metadata={"token": token},
    ) for token in range(steps))
    S0 = np.zeros((d, d), dtype=np.float64)
    hat_S0 = S0.copy()
    hat_S0[0, 0] = 2e-4
    return Scenario(
        "adversarial", sequence, S0, hat_S0,
        "All singular directions and rank-one defects align, attaining the scalar product bound.",
    )


def make_diagonal_shared_directions(d: int = 4, steps: int = 20) -> Scenario:
    if not 2 <= d <= 32:
        raise ValueError("diagonal example needs d >= 2")
    sequence: list[StepPair] = []
    beta, u, v = _zero_update(d)
    for token in range(steps):
        diag = np.full(d, 0.6)
        if token % 2 == 0:
            diag[0], diag[1] = 1.2, 0.5
        else:
            diag[0], diag[1] = 0.5, 1.2
        A = np.diag(diag)
        sequence.append(StepPair(
            A=A, B=np.eye(d), beta=beta, u=u, v=v,
            hat_A=A, hat_B=np.eye(d), hat_beta=beta, hat_u=u, hat_v=v,
            label="diagonal-shared", metadata={"token": token},
        ))
    S0 = np.zeros((d, d), dtype=np.float64)
    hat_S0 = S0.copy()
    hat_S0[0, 0] = 1e-3
    hat_S0[1, 1] = 1e-3
    return Scenario(
        "diagonal-shared", tuple(sequence), S0, hat_S0,
        "Per-step max norm exceeds one, while each fixed coordinate contracts over two steps.",
    )


def all_default_scenarios(d: int = 8, steps: int = 24) -> tuple[Scenario, ...]:
    return (
        make_contractive_kda(d=d, steps=steps),
        make_fp8_kda(d=d, steps=steps),
        make_marginal(d=d, steps=steps),
        make_switching_nonnormal(d=max(2, min(d, 8)), steps=min(steps, 14)),
        make_adversarial_tight(d=max(1, min(d, 8)), steps=min(steps, 16)),
        make_diagonal_shared_directions(d=max(2, min(d, 8)), steps=min(steps, 24)),
    )


# ---------------------------------------------------------------------------
# Simulator and bound comparison
# ---------------------------------------------------------------------------


def simulate_scenario(scenario: Scenario) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    S = np.asarray(scenario.S0, dtype=np.float64).copy()
    hat_S = np.asarray(scenario.hat_S0, dtype=np.float64).copy()
    d = S.shape[0]
    initial_delta = hat_S - S
    operator_radius = float(np.linalg.norm(initial_delta, "fro"))
    online_radius = operator_radius
    online_realized_radius = operator_radius
    exact_state_bound = float(np.linalg.norm(S, "fro"))
    diagonal_radius = np.abs(initial_delta)
    diagonal_active = True
    structural_radius = operator_radius
    structural_active = all(step.label == "contractive-kda" for step in scenario.steps)

    records: list[dict[str, Any]] = []
    As: list[Array] = []
    Bs: list[Array] = []
    defects: list[Array] = []

    for token, step in enumerate(scenario.steps, start=1):
        S_next = recurrence_step(S, step.A, step.B, step.beta, step.u, step.v)
        prequant = recurrence_step(
            hat_S, step.hat_A, step.hat_B, step.hat_beta, step.hat_u, step.hat_v
        )
        if step.quantizer is None:
            hat_S_next = prequant
            q_info = {
                "scale": 0.0,
                "amax": float(np.max(np.abs(prequant))),
                "scale_margin": math.inf,
                "rounding_norm": 0.0,
                "rounding_bound": 0.0,
                "saturated": 0.0,
            }
        else:
            hat_S_next, q_info = step.quantizer(prequant)
        rounding = hat_S_next - prequant
        terms = exact_error_terms(S, hat_S, S_next, hat_S_next, step, rounding)
        approximate_terms = approximate_propagator_error_terms(
            S, hat_S, S_next, hat_S_next, step, rounding
        )

        A_norm = float(np.linalg.norm(step.A, 2))
        B_norm = float(np.linalg.norm(step.B, 2))
        hat_A_norm = float(np.linalg.norm(step.hat_A, 2))
        hat_B_norm = float(np.linalg.norm(step.hat_B, 2))
        A_error = float(np.linalg.norm(step.hat_A - step.A, 2))
        B_error = float(np.linalg.norm(step.hat_B - step.B, 2))
        coefficient_norm = float(np.linalg.norm(terms.coefficient, "fro"))
        update_norm = float(np.linalg.norm(terms.update, "fro"))
        rounding_norm = float(np.linalg.norm(terms.rounding, "fro"))
        operator_radius = scalar_operator_bound_step(
            operator_radius, A_norm, B_norm,
            coefficient_norm, update_norm, float(q_info["rounding_bound"]),
        )

        update_bound = rank_one_update_error_bound(step)
        online_radius = online_error_certificate_step(
            online_radius,
            exact_state_bound,
            A_norm_bound=A_norm,
            B_norm_bound=B_norm,
            hat_A_norm_bound=hat_A_norm,
            hat_B_norm_bound=hat_B_norm,
            A_error_bound=A_error,
            B_error_bound=B_error,
            update_error_bound=update_bound,
            rounding_error_bound=float(q_info["rounding_bound"]),
        )
        online_realized_radius = online_error_certificate_step(
            online_realized_radius,
            exact_state_bound,
            A_norm_bound=A_norm,
            B_norm_bound=B_norm,
            hat_A_norm_bound=hat_A_norm,
            hat_B_norm_bound=hat_B_norm,
            A_error_bound=A_error,
            B_error_bound=B_error,
            update_error_bound=update_bound,
            rounding_error_bound=rounding_norm,
        )
        driven_energy_error = math.nan
        if structural_active:
            decay = np.asarray(step.metadata["decay"], dtype=np.float64)
            driven_energy_error = kda_driven_energy_terms(
                S, step.u, step.beta, decay, step.v
            )["identity_error"]
            exact_state_bound = kda_state_energy_majorant_step(
                exact_state_bound,
                kda_propagation_cap(decay),
                abs(step.beta),
                float(np.linalg.norm(step.v)),
            )
        else:
            exact_state_bound = exact_state_norm_majorant_step(
                exact_state_bound, A_norm, B_norm, abs(step.beta),
                float(np.linalg.norm(step.u)), float(np.linalg.norm(step.v)),
            )

        diagonal_bound = math.nan
        if diagonal_active:
            if (np.allclose(step.A, np.diag(np.diag(step.A)))
                    and np.allclose(step.B, np.diag(np.diag(step.B)))):
                diagonal_radius = diagonal_majorant_step(
                    diagonal_radius, step.A, step.B, terms.local_defect
                )
                diagonal_bound = float(np.linalg.norm(diagonal_radius, "fro"))
            else:
                diagonal_active = False

        structural_bound = math.nan
        energy_error = math.nan
        if structural_active:
            decay = np.asarray(step.metadata["decay"], dtype=np.float64)
            cap = kda_propagation_cap(decay)
            structural_radius = (cap * structural_radius + coefficient_norm
                                 + update_norm + float(q_info["rounding_bound"]))
            structural_bound = structural_radius
            energy_error = kda_energy_terms(
                hat_S - S, step.u, step.beta, decay
            )["identity_error"]

        observed = float(np.linalg.norm(hat_S_next - S_next, "fro"))
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(step.A))))
        records.append({
            "scenario": scenario.name,
            "token": token,
            "observed": observed,
            "operator_bound": operator_radius,
            "online_bound": online_radius,
            "online_realized_bound": online_realized_radius,
            "diagonal_bound": diagonal_bound,
            "structural_bound": structural_bound,
            "transition_bound": math.nan,
            "A_norm": A_norm,
            "B_norm": B_norm,
            "spectral_radius_A": spectral_radius,
            "coefficient_error": coefficient_norm,
            "update_error": update_norm,
            "rounding_error": rounding_norm,
            "rounding_bound": float(q_info["rounding_bound"]),
            "state_bound": exact_state_bound,
            "energy_identity_error": energy_error,
            "driven_energy_identity_error": driven_energy_error,
            "decomposition_error": float(np.linalg.norm(
                terms.reconstructed - (hat_S_next - S_next), "fro"
            )),
            "approx_decomposition_error": float(np.linalg.norm(
                approximate_terms.reconstructed - (hat_S_next - S_next), "fro"
            )),
        })
        As.append(step.A)
        Bs.append(step.B)
        defects.append(terms.local_defect)
        S, hat_S = S_next, hat_S_next

    transition = transition_product_bound(initial_delta, As, Bs, defects)
    for row, value in zip(records, transition):
        row["transition_bound"] = value

    finite_bounds = [
        key for key in ("operator_bound", "online_bound", "online_realized_bound",
                        "diagonal_bound", "structural_bound", "transition_bound")
        if any(math.isfinite(float(row[key])) for row in records)
    ]
    violations: dict[str, float] = {}
    for key in finite_bounds:
        worst = 0.0
        for row in records:
            bound = float(row[key])
            if math.isfinite(bound):
                worst = max(worst, float(row["observed"]) - bound)
        violations[key] = worst

    final = records[-1] if records else {}
    summary = {
        "name": scenario.name,
        "description": scenario.description,
        "dimension": d,
        "steps": len(scenario.steps),
        "final_observed": final.get("observed", 0.0),
        "peak_observed": max((float(row["observed"]) for row in records), default=0.0),
        "final_operator_bound": final.get("operator_bound", 0.0),
        "final_online_bound": final.get("online_bound", 0.0),
        "final_online_realized_bound": final.get("online_realized_bound", 0.0),
        "final_transition_bound": final.get("transition_bound", 0.0),
        "final_diagonal_bound": final.get("diagonal_bound", math.nan),
        "final_structural_bound": final.get("structural_bound", math.nan),
        "max_individual_spectral_radius": max(
            (float(row["spectral_radius_A"]) for row in records), default=0.0
        ),
        "max_individual_operator_norm": max(
            (float(row["A_norm"]) for row in records), default=0.0
        ),
        "bound_violations": violations,
    }
    return records, summary


# ---------------------------------------------------------------------------
# Reset scheduling
# ---------------------------------------------------------------------------


def segment_radius_trace(
    gains: Sequence[float], local_errors: Sequence[float], start: int, end: int
) -> list[float]:
    radius = 0.0
    trace: list[float] = []
    for token in range(start, end):
        radius = float(gains[token]) * radius + float(local_errors[token])
        trace.append(radius)
    return trace


def _segment_peak(
    gains: Sequence[float], local_errors: Sequence[float], start: int, end: int
) -> float:
    trace = segment_radius_trace(gains, local_errors, start, end)
    return max(trace, default=0.0)


def solve_reset_dp(
    gains: Sequence[float],
    local_errors: Sequence[float],
    budget: float,
    reset_cost: float,
    token_saving: float,
    *,
    require_final_exact: bool = True,
    allowed_reset_boundaries: Iterable[int] | None = None,
) -> ResetSolution:
    """Exact finite-horizon partition DP with every token approximate.

    A segment starts from an exact state (radius zero).  Internal boundaries
    pay reset_cost.  If require_final_exact, restoring after the last segment
    also pays reset_cost.
    """

    n = len(gains)
    if len(local_errors) != n:
        raise ValueError("gains and local_errors must have equal length")
    if budget < 0 or reset_cost < 0 or token_saving < 0:
        raise ValueError("negative budget/cost/saving")
    if allowed_reset_boundaries is None:
        allowed = set(range(n + 1))
    else:
        allowed = {int(value) for value in allowed_reset_boundaries}
        if any(value < 0 or value > n for value in allowed):
            raise ValueError("reset boundary outside the horizon")
        allowed.update((0, n))
    feasible = [[False] * (n + 1) for _ in range(n + 1)]
    peak = [[0.0] * (n + 1) for _ in range(n + 1)]
    for start in range(n):
        radius = 0.0
        maximum = 0.0
        for end in range(start + 1, n + 1):
            token = end - 1
            radius = float(gains[token]) * radius + float(local_errors[token])
            maximum = max(maximum, radius)
            peak[start][end] = maximum
            feasible[start][end] = maximum <= budget + 1e-12

    best = [-math.inf] * (n + 1)
    previous = [-1] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        for start in range(0, end):
            if start not in allowed:
                continue
            if best[start] == -math.inf or not feasible[start][end]:
                continue
            candidate = (best[start] + (end - start) * token_saving
                         - (reset_cost if start > 0 else 0.0))
            if candidate > best[end] + 1e-12:
                best[end] = candidate
                previous[end] = start
    if best[n] == -math.inf:
        raise RuntimeError("no all-approximate reset schedule is feasible")

    boundaries = [n]
    cursor = n
    while cursor > 0:
        cursor = previous[cursor]
        if cursor < 0:
            raise AssertionError("broken reset backpointer")
        boundaries.append(cursor)
    boundaries.reverse()
    segment_peaks = tuple(
        peak[boundaries[i]][boundaries[i + 1]] for i in range(len(boundaries) - 1)
    )
    resets = max(0, len(boundaries) - 2) + int(require_final_exact and n > 0)
    net = best[n] - (reset_cost if require_final_exact and n > 0 else 0.0)
    return ResetSolution(tuple(boundaries), resets, float(net), segment_peaks, require_final_exact)


def solve_reset_bruteforce(
    gains: Sequence[float],
    local_errors: Sequence[float],
    budget: float,
    reset_cost: float,
    token_saving: float,
    *,
    require_final_exact: bool = True,
    allowed_reset_boundaries: Iterable[int] | None = None,
) -> ResetSolution:
    """Exact exhaustive reset solver, intended for n <= 20 checks."""

    n = len(gains)
    if n > 20:
        raise ValueError("bruteforce solver is restricted to n <= 20")
    if allowed_reset_boundaries is None:
        internal = list(range(1, n))
    else:
        allowed = {int(value) for value in allowed_reset_boundaries}
        if any(value < 0 or value > n for value in allowed):
            raise ValueError("reset boundary outside the horizon")
        internal = sorted(value for value in allowed if 0 < value < n)
    best: ResetSolution | None = None
    for mask in range(1 << len(internal)):
        boundaries = [0]
        for bit, boundary in enumerate(internal):
            if mask & (1 << bit):
                boundaries.append(boundary)
        boundaries.append(n)
        peaks = tuple(
            _segment_peak(gains, local_errors, boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        )
        if any(value > budget + 1e-12 for value in peaks):
            continue
        resets = max(0, len(boundaries) - 2) + int(require_final_exact and n > 0)
        net = n * token_saving - resets * reset_cost
        candidate = ResetSolution(tuple(boundaries), resets, float(net), peaks, require_final_exact)
        if best is None or candidate.net_saving > best.net_saving + 1e-12:
            best = candidate
    if best is None:
        raise RuntimeError("no all-approximate reset schedule is feasible")
    return best


def lazy_reset_schedule(
    gains: Sequence[float],
    local_errors: Sequence[float],
    budget: float,
    reset_cost: float,
    token_saving: float,
    *,
    require_final_exact: bool = True,
) -> ResetSolution:
    """Causal reset immediately before the first would-be budget violation."""

    if len(gains) != len(local_errors):
        raise ValueError("gains and local_errors must have equal length")
    boundaries = [0]
    peaks: list[float] = []
    radius = 0.0
    current_peak = 0.0
    for token, (gain, error) in enumerate(zip(gains, local_errors)):
        candidate = float(gain) * radius + float(error)
        if candidate > budget + 1e-12:
            if token == boundaries[-1]:
                raise RuntimeError(f"token {token} is infeasible even immediately after reset")
            peaks.append(current_peak)
            boundaries.append(token)
            radius = float(error)
            current_peak = radius
            if radius > budget + 1e-12:
                raise RuntimeError(f"token {token} is infeasible even immediately after reset")
        else:
            radius = candidate
            current_peak = max(current_peak, radius)
    boundaries.append(len(gains))
    if gains:
        peaks.append(current_peak)
    resets = max(0, len(boundaries) - 2) + int(require_final_exact and bool(gains))
    net = len(gains) * token_saving - resets * reset_cost
    return ResetSolution(tuple(boundaries), resets, float(net), tuple(peaks), require_final_exact)


def lazy_boundary_reset_schedule(
    gains: Sequence[float],
    local_errors: Sequence[float],
    budget: float,
    reset_cost: float,
    token_saving: float,
    *,
    allowed_reset_boundaries: Iterable[int],
    require_final_exact: bool = True,
) -> ResetSolution:
    """Causal latest-feasible reset policy for legal boundary intervals.

    At each legal boundary, the coefficient/error envelopes for the next legal
    interval are assumed known (or safely upper-bounded) before that interval
    is committed.  A reset is taken exactly when the incoming radius would make
    that interval unsafe.  Monotonicity makes this pathwise minimum-reset among
    all schedules with the same legal boundaries and information.
    """

    n = len(gains)
    if len(local_errors) != n:
        raise ValueError("gains and local_errors must have equal length")
    legal = sorted({int(value) for value in allowed_reset_boundaries} | {0, n})
    if any(value < 0 or value > n for value in legal):
        raise ValueError("reset boundary outside the horizon")
    if budget < 0 or reset_cost < 0 or token_saving < 0:
        raise ValueError("negative budget/cost/saving")

    reset_boundaries = [0]
    segment_peaks: list[float] = []
    radius = 0.0
    current_peak = 0.0
    segment_start = 0

    for start, end in zip(legal[:-1], legal[1:]):
        trial_radius = radius
        trial_peak = current_peak
        safe = True
        for token in range(start, end):
            trial_radius = float(gains[token]) * trial_radius + float(local_errors[token])
            trial_peak = max(trial_peak, trial_radius)
            if trial_radius > budget + 1e-12:
                safe = False
                break
        if safe:
            radius = trial_radius
            current_peak = trial_peak
            continue

        # No reset is legal inside [start,end), so every feasible schedule must
        # reset at start.  Resetting there gives the smallest possible radius.
        if start == segment_start:
            raise RuntimeError(
                f"legal interval [{start},{end}) is infeasible even from an exact state"
            )
        segment_peaks.append(current_peak)
        reset_boundaries.append(start)
        segment_start = start
        radius = 0.0
        current_peak = 0.0
        for token in range(start, end):
            radius = float(gains[token]) * radius + float(local_errors[token])
            current_peak = max(current_peak, radius)
            if radius > budget + 1e-12:
                raise RuntimeError(
                    f"legal interval [{start},{end}) is infeasible even from an exact state"
                )

    reset_boundaries.append(n)
    if n:
        segment_peaks.append(current_peak)
    resets = max(0, len(reset_boundaries) - 2) + int(require_final_exact and n > 0)
    net = n * token_saving - resets * reset_cost
    return ResetSolution(
        tuple(reset_boundaries), resets, float(net), tuple(segment_peaks), require_final_exact
    )


def exact_replay_radius_from_snapshot(
    incoming_radius: float,
    exact_gains: Sequence[float],
    accepted: int | None = None,
) -> float:
    """Radius after exact replay from a possibly inexact rollback snapshot.

    Exact replay removes new local approximation errors.  It does not erase
    error already present in the snapshot.  Reset-to-zero requires
    incoming_radius == 0, meaning the snapshot is a genuine exact anchor.
    """

    if incoming_radius < 0 or any(gain < 0 for gain in exact_gains):
        raise ValueError("radii and gains must be nonnegative")
    count = len(exact_gains) if accepted is None else int(accepted)
    if count < 0 or count > len(exact_gains):
        raise ValueError("accepted length outside the block")
    radius = float(incoming_radius)
    for gain in exact_gains[:count]:
        radius *= float(gain)
    return radius


def reset_demo_instance() -> dict[str, Any]:
    gains = [0.93, 1.0, 0.88, 1.0, 0.97, 0.82, 1.0, 0.9, 0.95, 1.0]
    local = [0.08, 0.11, 0.07, 0.13, 0.09, 0.06, 0.14, 0.08, 0.1, 0.07]
    budget = 0.32
    reset_cost = 0.45
    token_saving = 0.12
    dp = solve_reset_dp(gains, local, budget, reset_cost, token_saving)
    brute = solve_reset_bruteforce(gains, local, budget, reset_cost, token_saving)
    lazy = lazy_reset_schedule(gains, local, budget, reset_cost, token_saving)
    block_boundaries = [0, 2, 4, 6, 8, 10]
    block_dp = solve_reset_dp(
        gains, local, budget, reset_cost, token_saving,
        allowed_reset_boundaries=block_boundaries,
    )
    block_brute = solve_reset_bruteforce(
        gains, local, budget, reset_cost, token_saving,
        allowed_reset_boundaries=block_boundaries,
    )
    block_lazy = lazy_boundary_reset_schedule(
        gains, local, budget, reset_cost, token_saving,
        allowed_reset_boundaries=block_boundaries,
    )
    inherited = 0.2
    replay_gains = [0.95, 0.9, 0.85]
    return {
        "gains": gains,
        "local_errors": local,
        "budget": budget,
        "reset_cost": reset_cost,
        "token_saving": token_saving,
        "dynamic_program": asdict(dp),
        "bruteforce": asdict(brute),
        "causal_lazy": asdict(lazy),
        "legal_block_boundaries": block_boundaries,
        "block_dynamic_program": asdict(block_dp),
        "block_bruteforce": asdict(block_brute),
        "block_causal_lazy": asdict(block_lazy),
        "rollback_snapshot_demo": {
            "incoming_radius": inherited,
            "exact_replay_gains": replay_gains,
            "accepted": 2,
            "restored_radius": exact_replay_radius_from_snapshot(
                inherited, replay_gains, accepted=2
            ),
            "genuine_exact_anchor_radius": exact_replay_radius_from_snapshot(
                0.0, replay_gains, accepted=2
            ),
        },
    }


# ---------------------------------------------------------------------------
# Exact/symbolic checks and adversarial proxy failures
# ---------------------------------------------------------------------------


def symbolic_checks() -> dict[str, Any]:
    try:
        import sympy as sp
    except ModuleNotFoundError as exc:  # pragma: no cover - repo test hosts have SymPy.
        raise RuntimeError("symbolic checks require sympy") from exc

    R = sp.Rational
    d = 3
    S = sp.Matrix([[R(1, 3), R(-1, 4), R(2, 5)],
                   [R(1, 7), R(3, 8), R(-2, 9)],
                   [R(4, 11), R(1, 6), R(5, 13)]])
    hat_S = S + sp.Matrix([[R(1, 101), 0, 0], [0, R(-1, 97), 0], [0, 0, R(1, 89)]])
    A = sp.diag(R(3, 4), R(2, 3), R(4, 5))
    B = sp.diag(R(5, 6), R(7, 8), R(9, 10))
    hat_A = A + sp.Matrix([[0, R(1, 200), 0], [0, 0, 0], [0, 0, R(-1, 180)]])
    hat_B = B + sp.Matrix([[R(1, 170), 0, 0], [0, 0, 0], [0, R(-1, 190), 0]])
    beta, hat_beta = R(2, 5), R(3, 7)
    u = sp.Matrix([R(3, 5), R(4, 5), 0])
    v = sp.Matrix([R(1, 2), R(-1, 3), R(2, 7)])
    hat_u = u + sp.Matrix([R(1, 100), R(-1, 120), R(1, 140)])
    hat_v = v + sp.Matrix([R(-1, 110), R(1, 130), R(1, 150)])
    q = sp.Matrix([[R(1, 1000), 0, 0], [0, R(-1, 900), 0], [0, 0, R(1, 800)]])
    S_next = A * S * B + beta * u * v.T
    pre = hat_A * hat_S * hat_B + hat_beta * hat_u * hat_v.T
    hat_next = pre + q
    delta = hat_next - S_next
    E_A, E_B = hat_A - A, hat_B - B
    rhs_exact = (A * (hat_S - S) * B
                 + E_A * hat_S * hat_B + A * hat_S * E_B
                 + hat_beta * hat_u * hat_v.T - beta * u * v.T + q)
    rhs_hat = (hat_A * (hat_S - S) * hat_B
               + E_A * S * B + A * S * E_B + E_A * S * E_B
               + hat_beta * hat_u * hat_v.T - beta * u * v.T + q)

    k = sp.Matrix([R(3, 5), R(4, 5), 0])
    decay = sp.diag(R(3, 4), R(5, 6), R(7, 8))
    beta_k = R(2, 3)
    X = sp.Matrix([[1, 2, 3], [R(-1, 2), R(2, 3), R(3, 4)], [R(1, 5), R(-2, 7), R(4, 9)]])
    P = sp.eye(d) - beta_k * k * k.T
    AX = P * decay * X
    norm_sq = lambda M: sp.simplify(sum(entry * entry for entry in M))
    decay_vec = [decay[i, i] for i in range(d)]
    decay_diss = sp.simplify(sum(
        (1 - decay_vec[i] ** 2) * X[i, j] ** 2 for i in range(d) for j in range(d)
    ))
    projection_coeff = 2 * beta_k - beta_k ** 2 * (k.T * k)[0]
    projected_row = k.T * decay * X
    projection_diss = sp.simplify(projection_coeff * norm_sq(projected_row))
    energy_residual = sp.simplify(norm_sq(AX) - (norm_sq(X) - decay_diss - projection_diss))

    driven_v = sp.Matrix([R(2, 5), R(-3, 7), R(5, 11)])
    driven_Y = decay * X
    driven_memory = k.T * driven_Y
    driven_innovation = driven_v.T - driven_memory
    driven_next = driven_Y + beta_k * k * driven_innovation
    driven_rhs = (norm_sq(driven_Y)
                  - beta_k * norm_sq(driven_memory)
                  + beta_k * norm_sq(driven_v)
                  - beta_k * (1 - beta_k * (k.T * k)[0])
                  * norm_sq(driven_innovation))
    driven_energy_residual = sp.simplify(norm_sq(driven_next) - driven_rhs)

    r, shear = R(1, 2), R(2, 1)
    A1 = sp.Matrix([[r, shear], [0, r]])
    A2 = sp.Matrix([[r, 0], [shear, r]])
    pair = A2 * A1
    pair_max_eigen = max([sp.N(ev) for ev in pair.eigenvals().keys()])

    dimension_sweep: dict[str, dict[str, bool]] = {}
    for small_d in range(1, 5):
        small_S = sp.Matrix(small_d, small_d, lambda i, j: R(
            (i + 1) * (j + 2), 11 + i + 2 * j
        ))
        small_hat_S = small_S + sp.diag(*[R(1, 101 + i) for i in range(small_d)])
        small_A = sp.diag(*[R(i + 2, i + 3) for i in range(small_d)])
        small_B = sp.diag(*[R(i + 3, i + 4) for i in range(small_d)])
        small_hat_A = small_A + sp.diag(*[R(1, 211 + i) for i in range(small_d)])
        small_hat_B = small_B - sp.diag(*[R(1, 307 + i) for i in range(small_d)])
        small_u = sp.Matrix([R(i + 1, small_d + 3) for i in range(small_d)])
        small_v = sp.Matrix([R((-1) ** i * (i + 2), small_d + 5) for i in range(small_d)])
        small_hat_u = small_u + sp.Matrix([R(1, 401 + i) for i in range(small_d)])
        small_hat_v = small_v - sp.Matrix([R(1, 503 + i) for i in range(small_d)])
        small_beta, small_hat_beta = R(2, 5), R(3, 7)
        small_q = sp.diag(*[R((-1) ** i, 701 + i) for i in range(small_d)])
        small_next = small_A * small_S * small_B + small_beta * small_u * small_v.T
        small_pre = (small_hat_A * small_hat_S * small_hat_B
                     + small_hat_beta * small_hat_u * small_hat_v.T)
        small_delta = small_pre + small_q - small_next
        small_EA, small_EB = small_hat_A - small_A, small_hat_B - small_B
        small_rhs_exact = (
            small_A * (small_hat_S - small_S) * small_B
            + small_EA * small_hat_S * small_hat_B
            + small_A * small_hat_S * small_EB
            + small_hat_beta * small_hat_u * small_hat_v.T
            - small_beta * small_u * small_v.T + small_q
        )
        small_rhs_hat = (
            small_hat_A * (small_hat_S - small_S) * small_hat_B
            + small_EA * small_S * small_B
            + small_A * small_S * small_EB
            + small_EA * small_S * small_EB
            + small_hat_beta * small_hat_u * small_hat_v.T
            - small_beta * small_u * small_v.T + small_q
        )

        small_k = sp.zeros(small_d, 1)
        small_k[0] = 1
        small_decay = sp.diag(*[R(i + 2, i + 3) for i in range(small_d)])
        small_kbeta = R(2, 3)
        small_P = sp.eye(small_d) - small_kbeta * small_k * small_k.T
        small_DX = small_decay * small_S
        small_AX = small_P * small_DX
        small_c = 2 * small_kbeta - small_kbeta ** 2 * (small_k.T * small_k)[0]
        small_decay_diss = sum(
            (1 - small_decay[i, i] ** 2) * small_S[i, j] ** 2
            for i in range(small_d) for j in range(small_d)
        )
        small_proj = small_k.T * small_DX
        small_hom_rhs = norm_sq(small_S) - small_decay_diss - small_c * norm_sq(small_proj)
        small_drive_v = sp.Matrix([R(i + 2, small_d + 7) for i in range(small_d)])
        small_memory = small_k.T * small_DX
        small_innovation = small_drive_v.T - small_memory
        small_driven = small_DX + small_kbeta * small_k * small_innovation
        small_driven_rhs = (
            norm_sq(small_DX) - small_kbeta * norm_sq(small_memory)
            + small_kbeta * norm_sq(small_drive_v)
            - small_kbeta * (1 - small_kbeta * (small_k.T * small_k)[0])
            * norm_sq(small_innovation)
        )
        dimension_sweep[str(small_d)] = {
            "exact_decomposition": small_rhs_exact == small_delta,
            "approximate_decomposition": small_rhs_hat == small_delta,
            "homogeneous_energy": sp.simplify(norm_sq(small_AX) - small_hom_rhs) == 0,
            "driven_energy": sp.simplify(norm_sq(small_driven) - small_driven_rhs) == 0,
        }

    obstruction = sp.Matrix([
        [R(5, 512), R(13, 512)],
        [R(15, 1024), R(9, 256)],
    ])
    source_rank_one = sp.Matrix([R(1, 10), R(29, 195)]) * sp.Matrix([
        R(1, 10), R(16, 65)
    ]).T
    return {
        "dimension": d,
        "max_symbolic_dimension": 4,
        "dimension_sweep": dimension_sweep,
        "exact_propagator_decomposition_zero": rhs_exact == delta,
        "approximate_propagator_decomposition_zero": rhs_hat == delta,
        "kda_energy_identity_zero": energy_residual == 0,
        "kda_driven_energy_identity_zero": driven_energy_residual == 0,
        "switch_step_characteristic_polynomial": str(A1.charpoly().as_expr().factor()),
        "switch_pair_characteristic_polynomial": str(pair.charpoly().as_expr().factor()),
        "switch_pair_max_eigenvalue": float(pair_max_eigen),
        "rank_one_prequantized_state_rank": int(source_rank_one.rank()),
        "rank_two_quantized_state_rank": int(obstruction.rank()),
        "rank_two_quantized_state_determinant": str(sp.factor(obstruction.det())),
        "rank_one_update_can_match_quantized_state": obstruction.rank() <= 1,
    }


def false_confidence_examples() -> dict[str, Any]:
    # 1. Spectral radius ignores nonnormal switching.
    switching = make_switching_nonnormal(d=2, steps=10)
    trace, summary = simulate_scenario(switching)

    # 2. A current readout can miss an error that a later query exposes.
    delta = np.zeros((2, 2))
    delta[1, 1] = 1.0
    q_now = np.array([1.0, 0.0])
    q_later = np.array([0.0, 1.0])
    now = float(np.linalg.norm(q_now @ delta))
    later = float(np.linalg.norm(q_later @ delta))

    # 3. Mean diagonal decay misses the undamped coordinate.
    diagonal = np.diag([1.0, 0.01, 0.01, 0.01])
    aligned = np.zeros((4, 4))
    aligned[0, 0] = 1.0
    propagated = diagonal @ aligned

    return {
        "spectral_radius": {
            "max_individual": summary["max_individual_spectral_radius"],
            "peak_error": summary["peak_observed"],
            "initial_error": float(np.linalg.norm(switching.hat_S0 - switching.S0)),
            "amplification": summary["peak_observed"] /
                             float(np.linalg.norm(switching.hat_S0 - switching.S0)),
            "last_error": trace[-1]["observed"],
        },
        "readout_sketch": {
            "current_readout_error": now,
            "later_readout_error": later,
            "state_error_norm": float(np.linalg.norm(delta, "fro")),
        },
        "mean_decay": {
            "mean_decay": float(np.mean(np.diag(diagonal))),
            "operator_norm": float(np.linalg.norm(diagonal, 2)),
            "error_before": float(np.linalg.norm(aligned, "fro")),
            "error_after": float(np.linalg.norm(propagated, "fro")),
        },
    }


# ---------------------------------------------------------------------------
# CLI artifact generation
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_artifacts(output: Path, *, d: int, steps: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scenario in all_default_scenarios(d=d, steps=steps):
        rows, summary = simulate_scenario(scenario)
        all_rows.extend(rows)
        summaries.append(summary)
        for name, violation in summary["bound_violations"].items():
            if violation > 1e-8:
                raise AssertionError(f"{scenario.name}: {name} violated by {violation}")

    trace_path = output / "trace.csv"
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    reset = reset_demo_instance()
    symbolic = symbolic_checks()
    false_confidence = false_confidence_examples()
    fp8 = fp8_scale_discontinuity_demo()
    fp8_rank = fp8_rank_obstruction_demo()
    payload = {
        "config": {"dimension": d, "steps": steps},
        "scenarios": summaries,
        "symbolic_checks": symbolic,
        "false_confidence": false_confidence,
        "fp8_scale_discontinuity": fp8,
        "fp8_rank_obstruction": fp8_rank,
        "reset_demo": reset,
        "e4m3_half_max_gap": _E4M3_HALF_MAX_GAP,
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "reset-demo.json").write_text(
        json.dumps(_json_safe(reset), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "fp8-scale-demo.json").write_text(
        json.dumps(_json_safe(fp8), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "fp8-rank-demo.json").write_text(
        json.dumps(_json_safe(fp8_rank), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    print("scenario                 peak error       op bound  online worst  online fused  transition")
    for item in payload["scenarios"]:
        print(f"{item['name']:<24} {item['peak_observed']:>11.4e} "
              f"{item['final_operator_bound']:>14.4e} "
              f"{item['final_online_bound']:>13.4e} "
              f"{item['final_online_realized_bound']:>13.4e} "
              f"{item['final_transition_bound']:>11.4e}")
    switch = payload["false_confidence"]["spectral_radius"]
    print(f"switching: rho(step)<={switch['max_individual']:.3f}, "
          f"transient amplification={switch['amplification']:.1f}x")
    fp8 = payload["fp8_scale_discontinuity"]
    print(f"FP8 scale jump {fp8['low_scale']} -> {fp8['high_scale']}, "
          f"local ratio={fp8['local_lipschitz_ratio']:.3e}")
    reset = payload["reset_demo"]
    print("reset DP boundaries:", reset["dynamic_program"]["boundaries"],
          "net", reset["dynamic_program"]["net_saving"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("scratch/kda-shadowing"))
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args(argv)
    if not 1 <= args.d <= 32:
        parser.error("--d must lie in [1,32]")
    if args.steps < 1:
        parser.error("--steps must be positive")
    payload = write_artifacts(args.out, d=args.d, steps=args.steps)
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
