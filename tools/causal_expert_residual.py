#!/usr/bin/env python3
"""Causal predictor/residual coding helpers for sparse expert records.

The module is deliberately hardware-free.  It provides finite synthetic expert
families, a lossless random-access XOR residual container, empirical
conditional rate-distortion calculations, exact finite stopping/cache solvers,
and FP32 order checks.  Every correlation in the generators is an explicit
parameter; no model trace is implied.
"""

from __future__ import annotations

import itertools
import math
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_EPS = 1e-12
_MAGIC = b"CER1"
_VERSION = 1
_HEADER = struct.Struct("<4sHHIIIIQQ24s")  # exactly 64 bytes
_DESCRIPTOR = struct.Struct("<QIIIIII")  # exactly 32 bytes
_MODE_ZERO = 0
_MODE_RAW_XOR = 1
_MODE_SPARSE_XOR = 2


# ---------------------------------------------------------------------------
# Information-theoretic lower bound helpers


@dataclass(frozen=True)
class NoFreeLunchBound:
    experts: int
    query_points_per_expert: int
    output_bits: int
    resident_bits: int
    independent_cells: int
    total_representation_bits: int
    external_bits_for_exhaustive_queries: int
    mean_external_bits_per_query: float


def arbitrary_map_lower_bound(
    experts: int,
    query_points_per_expert: int,
    output_bits: int,
    resident_bits: int = 0,
) -> NoFreeLunchBound:
    """Return the counting/exhaustive-query bound for arbitrary finite maps.

    There are ``experts * query_points_per_expert`` independent output cells,
    each containing ``output_bits`` arbitrary bits.  A zero-error encoder needs
    that many total bits.  If ``resident_bits`` are present before a uniformly
    permuted exhaustive query sequence, the external transcript still carries
    at least total entropy minus resident information.
    """

    values = (experts, query_points_per_expert, output_bits)
    if any(int(v) <= 0 for v in values):
        raise ValueError("experts, query points, and output bits must be positive")
    if resident_bits < 0:
        raise ValueError("resident_bits must be non-negative")
    cells = int(experts) * int(query_points_per_expert)
    total = cells * int(output_bits)
    external = max(0, total - int(resident_bits))
    return NoFreeLunchBound(
        experts=int(experts),
        query_points_per_expert=int(query_points_per_expert),
        output_bits=int(output_bits),
        resident_bits=int(resident_bits),
        independent_cells=cells,
        total_representation_bits=total,
        external_bits_for_exhaustive_queries=external,
        mean_external_bits_per_query=external / cells,
    )


# ---------------------------------------------------------------------------
# Discrete entropy and information helpers


def _as_rows(values: np.ndarray | Sequence[Any]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    else:
        array = array.reshape(array.shape[0], -1)
    return np.ascontiguousarray(array)


def encode_categories(*values: np.ndarray | Sequence[Any]) -> np.ndarray:
    """Encode one or more equal-length discrete arrays as dense integer IDs."""

    if not values:
        raise ValueError("at least one value array is required")
    rows = [_as_rows(v) for v in values]
    n = rows[0].shape[0]
    if any(r.shape[0] != n for r in rows):
        raise ValueError("category arrays must have equal first dimension")
    joined = np.concatenate(rows, axis=1)
    _, inverse = np.unique(joined, axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def entropy_bits(values: np.ndarray | Sequence[Any]) -> float:
    ids = encode_categories(values)
    counts = np.bincount(ids)
    probabilities = counts[counts > 0].astype(np.float64) / ids.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def conditional_entropy_bits(
    values: np.ndarray | Sequence[Any], context: np.ndarray | Sequence[Any]
) -> float:
    return entropy_bits(encode_categories(values, context)) - entropy_bits(context)


def mutual_information_bits(
    left: np.ndarray | Sequence[Any], right: np.ndarray | Sequence[Any]
) -> float:
    value = entropy_bits(left) + entropy_bits(right) - entropy_bits(encode_categories(left, right))
    return max(0.0, float(value))


def conditional_mutual_information_bits(
    left: np.ndarray | Sequence[Any],
    right: np.ndarray | Sequence[Any],
    context: np.ndarray | Sequence[Any],
) -> float:
    value = conditional_entropy_bits(left, context) - conditional_entropy_bits(
        left, encode_categories(right, context)
    )
    return max(0.0, float(value))


# ---------------------------------------------------------------------------
# Synthetic expert families and causal contexts


@dataclass(frozen=True)
class SyntheticParameters:
    seed: int = 9
    experts: int = 8
    input_dim: int = 16
    output_dim: int = 12
    rank: int = 4
    chunks: int = 4
    tokens: int = 4096
    activation_states: int = 8
    route_repeat_probability: float = 0.75
    route_logit_accuracy: float = 0.9
    logit_activation_accuracy: float = 0.8
    sparse_chunk_probability: float = 0.25
    sparse_entry_probability: float = 0.20
    residual_scale: float = 0.08
    drift_scale: float = 0.01

    def validate(self) -> None:
        for name in (
            "experts",
            "input_dim",
            "output_dim",
            "rank",
            "chunks",
            "tokens",
            "activation_states",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rank > min(self.input_dim, self.output_dim):
            raise ValueError("rank exceeds matrix dimensions")
        for name in (
            "route_repeat_probability",
            "route_logit_accuracy",
            "logit_activation_accuracy",
            "sparse_chunk_probability",
            "sparse_entry_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.residual_scale < 0 or self.drift_scale < 0:
            raise ValueError("scales must be non-negative")


@dataclass(frozen=True)
class SyntheticExpertTrace:
    family: str
    weights: np.ndarray
    predictor_weights: np.ndarray
    routes: np.ndarray
    activations: np.ndarray
    activation_class: np.ndarray
    previous_logits: np.ndarray
    previous_top_route: np.ndarray
    previous_margin_bin: np.ndarray
    exact_outputs: np.ndarray
    predicted_outputs: np.ndarray
    residual_outputs: np.ndarray
    basis: np.ndarray | None
    coefficients: np.ndarray | None
    resident_predictor_bytes: int
    parameters: Mapping[str, Any]


def _float32_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(np.matmul(left.astype(np.float32), right.astype(np.float32)), dtype=np.float32)


def _row_slices(rows: int, chunks: int) -> tuple[slice, ...]:
    if chunks > rows:
        raise ValueError("chunks cannot exceed output rows")
    bounds = np.linspace(0, rows, chunks + 1, dtype=np.int64)
    return tuple(slice(int(bounds[j]), int(bounds[j + 1])) for j in range(chunks))


def _markov_categories(
    rng: np.random.Generator, count: int, categories: int, repeat_probability: float
) -> np.ndarray:
    result = np.empty(count, dtype=np.int64)
    result[0] = int(rng.integers(categories))
    for index in range(1, count):
        if rng.random() < repeat_probability:
            result[index] = result[index - 1]
        else:
            jump = int(rng.integers(1, categories)) if categories > 1 else 0
            result[index] = (result[index - 1] + jump) % categories
    return result


def _noisy_hint(
    rng: np.random.Generator, truth: np.ndarray, categories: int, accuracy: float
) -> np.ndarray:
    hint = np.asarray(truth, dtype=np.int64).copy()
    wrong = rng.random(hint.size) >= accuracy
    if categories > 1 and np.any(wrong):
        jump = rng.integers(1, categories, size=int(np.sum(wrong)))
        hint[wrong] = (hint[wrong] + jump) % categories
    return hint


def _make_context(
    rng: np.random.Generator,
    routes: np.ndarray,
    activation_class: np.ndarray,
    experts: int,
    activation_states: int,
    route_accuracy: float,
    activation_accuracy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top = _noisy_hint(rng, routes, experts, route_accuracy)
    margin_bin = _noisy_hint(rng, activation_class, activation_states, activation_accuracy)
    logits = np.full((routes.size, experts), -2.0, dtype=np.float32)
    base_margin = 1.0 + 0.35 * margin_bin.astype(np.float32)
    logits[np.arange(routes.size), top] = base_margin
    # The deterministic sub-top term makes the margin recoverable without
    # leaking any field other than the declared activation hint.
    runner_up = (top + 1) % experts
    logits[np.arange(routes.size), runner_up] = -0.25
    return logits, top, margin_bin


def _balanced_adversarial_context(
    tokens: int, experts: int, activation_states: int, route_accuracy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    period = experts * activation_states
    if tokens % period:
        raise ValueError(
            "route_only_adversary requires tokens divisible by experts * activation_states"
        )
    blocks = tokens // period
    correct_blocks = int(round(route_accuracy * blocks))
    wrong_blocks = blocks - correct_blocks
    wrong_block_indices = set()
    if wrong_blocks:
        wrong_block_indices = set(
            np.floor((np.arange(wrong_blocks) + 0.5) * blocks / wrong_blocks)
            .astype(np.int64)
            .tolist()
        )
    route = np.empty(tokens, dtype=np.int64)
    activation = np.empty(tokens, dtype=np.int64)
    top = np.empty(tokens, dtype=np.int64)
    margin = np.empty(tokens, dtype=np.int64)
    for t in range(tokens):
        block = t // period
        k = t % period
        route[t] = k % experts
        activation[t] = (k // experts) % activation_states
        # Every block contains the full route x activation cross-product.
        # Correctness varies only by whole block, so route prediction can be
        # strong without acquiring any information about residual activation.
        top[t] = (
            route[t] if block not in wrong_block_indices else (route[t] + 1) % experts
        )
        margin[t] = route[t] % activation_states
    logits = np.full((tokens, experts), -2.0, dtype=np.float32)
    logits[np.arange(tokens), top] = 3.0
    logits[np.arange(tokens), (top + 1) % experts] = -0.25
    return route, activation, logits, top, margin


def make_synthetic_trace(
    family: str, parameters: SyntheticParameters | None = None
) -> SyntheticExpertTrace:
    """Generate a finite causal expert trace with explicit correlation knobs."""

    p = parameters or SyntheticParameters()
    p.validate()
    rng = np.random.default_rng(p.seed)
    e, din, dout, rank = p.experts, p.input_dim, p.output_dim, p.rank
    basis: np.ndarray | None = None
    coefficients: np.ndarray | None = None

    if family == "independent_random":
        weights = rng.normal(0.0, 0.25, size=(e, dout, din)).astype(np.float32)
        predictor = np.zeros_like(weights)
        resident_bytes = 0
    elif family in {"exact_shared_basis", "shared_basis_sparse_residual"}:
        basis = rng.normal(0.0, 0.3, size=(rank, din)).astype(np.float32)
        coefficients = rng.normal(0.0, 0.3, size=(e, dout, rank)).astype(np.float32)
        predictor = _float32_matmul(coefficients, basis)
        weights = predictor.copy()
        if family == "shared_basis_sparse_residual":
            for expert in range(e):
                for rows in _row_slices(dout, p.chunks):
                    if rng.random() >= p.sparse_chunk_probability:
                        continue
                    block = weights[expert, rows, :]
                    mask = rng.random(block.shape) < p.sparse_entry_probability
                    delta = rng.normal(0.0, p.residual_scale, size=block.shape).astype(np.float32)
                    block[...] = np.asarray(block + mask * delta, dtype=np.float32)
        resident_bytes = int(basis.nbytes + coefficients.nbytes)
    elif family == "slowly_drifting":
        weights = np.empty((e, dout, din), dtype=np.float32)
        weights[0] = rng.normal(0.0, 0.25, size=(dout, din)).astype(np.float32)
        for expert in range(1, e):
            step = rng.normal(0.0, p.drift_scale, size=(dout, din)).astype(np.float32)
            weights[expert] = np.asarray(weights[expert - 1] + step, dtype=np.float32)
        # One resident cluster anchor gives true O(1) random access.  The
        # distance to the anchor grows with expert index under the declared
        # per-step drift_scale parameter.
        predictor = np.broadcast_to(weights[0], weights.shape).copy()
        resident_bytes = int(dout * din * np.dtype(np.float32).itemsize)
    elif family == "route_only_adversary":
        base = rng.normal(0.0, 0.2, size=(dout, din)).astype(np.float32)
        common = rng.normal(0.0, p.residual_scale, size=(dout, din)).astype(np.float32)
        predictor = np.broadcast_to(base, (e, dout, din)).copy()
        weights = np.broadcast_to(np.asarray(base + common, dtype=np.float32), (e, dout, din)).copy()
        resident_bytes = int(base.nbytes)
    else:
        raise ValueError(f"unknown family {family!r}")

    if family == "route_only_adversary":
        routes, activation_class, logits, top, margin = _balanced_adversarial_context(
            p.tokens, e, p.activation_states, p.route_logit_accuracy
        )
        # A finite activation codebook, crossed exactly with every route in each
        # complete period, makes residuals independent of route-side context.
        codebook_rng = np.random.default_rng(p.seed + 991)
        codebook = codebook_rng.normal(
            0.0, 1.0 / math.sqrt(din), size=(p.activation_states, din)
        ).astype(np.float32)
        activations = codebook[activation_class]
    else:
        routes = _markov_categories(rng, p.tokens, e, p.route_repeat_probability)
        activation_class = _markov_categories(
            rng, p.tokens, p.activation_states, min(0.98, p.route_repeat_probability + 0.1)
        )
        codebook = rng.normal(
            0.0, 1.0 / math.sqrt(din), size=(p.activation_states, din)
        ).astype(np.float32)
        noise = rng.normal(0.0, 0.03, size=(p.tokens, din)).astype(np.float32)
        activations = np.asarray(codebook[activation_class] + noise, dtype=np.float32)
        logits, top, margin = _make_context(
            rng,
            routes,
            activation_class,
            e,
            p.activation_states,
            p.route_logit_accuracy,
            p.logit_activation_accuracy,
        )

    selected_w = weights[routes]
    selected_p = predictor[routes]
    exact_outputs = np.einsum("toi,ti->to", selected_w, activations, optimize=False).astype(np.float32)
    predicted_outputs = np.einsum("toi,ti->to", selected_p, activations, optimize=False).astype(np.float32)
    residual_outputs = np.asarray(exact_outputs - predicted_outputs, dtype=np.float32)

    return SyntheticExpertTrace(
        family=family,
        weights=np.ascontiguousarray(weights),
        predictor_weights=np.ascontiguousarray(predictor),
        routes=routes,
        activations=np.ascontiguousarray(activations),
        activation_class=activation_class,
        previous_logits=logits,
        previous_top_route=top,
        previous_margin_bin=margin,
        exact_outputs=exact_outputs,
        predicted_outputs=predicted_outputs,
        residual_outputs=residual_outputs,
        basis=basis,
        coefficients=coefficients,
        resident_predictor_bytes=resident_bytes,
        parameters=asdict(p),
    )


def fit_shared_right_basis(weights: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``W_e ~= A_e B`` using a shared SVD right basis."""

    array = np.asarray(weights, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("weights must have shape [experts, output, input]")
    experts, output, input_dim = array.shape
    if not 0 < rank <= min(experts * output, input_dim):
        raise ValueError("invalid rank")
    stacked = array.reshape(experts * output, input_dim).astype(np.float64)
    _, _, vh = np.linalg.svd(stacked, full_matrices=False)
    basis = vh[:rank].astype(np.float32)
    coefficients = np.einsum("eoi,ri->eor", array, basis, optimize=False).astype(np.float32)
    predictor = _float32_matmul(coefficients, basis)
    return basis, coefficients, predictor


# ---------------------------------------------------------------------------
# Lossless random-access XOR residual container


@dataclass(frozen=True)
class ResidualDescriptor:
    offset: int
    stored_bytes: int
    extent_bytes: int
    raw_bytes: int
    crc32: int
    mode: int
    changed_bytes: int


@dataclass(frozen=True)
class ResidualContainerStats:
    experts: int
    chunks: int
    rows: int
    cols: int
    alignment: int
    directory_bytes: int
    payload_logical_bytes: int
    payload_extent_bytes: int
    file_bytes: int
    raw_full_bytes: int
    changed_bytes: int
    zero_chunks: int
    sparse_chunks: int
    raw_chunks: int
    mean_extent_bytes_per_expert: float


@dataclass(frozen=True)
class ResidualContainer:
    blob: bytes
    descriptors: tuple[ResidualDescriptor, ...]
    stats: ResidualContainerStats


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (int(value) + alignment - 1) & -alignment


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix, dtype="<f4").tobytes(order="C")


def encode_xor_residual_container(
    weights: np.ndarray,
    predictor_weights: np.ndarray,
    *,
    chunks: int,
    alignment: int = 4096,
) -> ResidualContainer:
    """Encode exact matrix bytes as independently readable XOR chunks.

    Predictor generation is outside the container.  A chunk stores either no
    bytes, raw XOR bytes, or sorted changed-byte positions plus XOR values.
    Each non-empty extent starts and ends on ``alignment`` boundaries.
    """

    actual = np.ascontiguousarray(weights, dtype="<f4")
    predicted = np.ascontiguousarray(predictor_weights, dtype="<f4")
    if actual.shape != predicted.shape or actual.ndim != 3:
        raise ValueError("weights and predictor_weights must share [E, O, I] shape")
    experts, rows, cols = actual.shape
    slices = _row_slices(rows, chunks)
    descriptor_count = experts * chunks
    directory_end = _HEADER.size + descriptor_count * _DESCRIPTOR.size
    payload_start = _align_up(directory_end, alignment)
    blob = bytearray(payload_start)
    descriptors: list[ResidualDescriptor] = []
    logical_payload = 0
    extent_payload = 0
    changed_total = 0
    mode_counts = {_MODE_ZERO: 0, _MODE_SPARSE_XOR: 0, _MODE_RAW_XOR: 0}

    for expert in range(experts):
        for rowslice in slices:
            actual_bytes = _matrix_bytes(actual[expert, rowslice, :])
            predicted_bytes = _matrix_bytes(predicted[expert, rowslice, :])
            raw = np.frombuffer(actual_bytes, dtype=np.uint8) ^ np.frombuffer(
                predicted_bytes, dtype=np.uint8
            )
            positions = np.flatnonzero(raw).astype("<u4")
            changed = int(positions.size)
            crc = zlib.crc32(actual_bytes) & 0xFFFFFFFF
            if changed == 0:
                descriptor = ResidualDescriptor(0, 0, 0, len(actual_bytes), crc, _MODE_ZERO, 0)
            else:
                sparse_payload = positions.tobytes() + raw[positions.astype(np.int64)].tobytes()
                raw_payload = raw.tobytes()
                if len(sparse_payload) < len(raw_payload):
                    payload = sparse_payload
                    mode = _MODE_SPARSE_XOR
                else:
                    payload = raw_payload
                    mode = _MODE_RAW_XOR
                offset = _align_up(len(blob), alignment)
                if offset > len(blob):
                    blob.extend(b"\0" * (offset - len(blob)))
                extent = _align_up(len(payload), alignment)
                blob.extend(payload)
                blob.extend(b"\0" * (extent - len(payload)))
                descriptor = ResidualDescriptor(
                    offset, len(payload), extent, len(actual_bytes), crc, mode, changed
                )
                logical_payload += len(payload)
                extent_payload += extent
            descriptors.append(descriptor)
            changed_total += changed
            mode_counts[descriptor.mode] += 1

    header = _HEADER.pack(
        _MAGIC,
        _VERSION,
        int(math.log2(alignment)),
        experts,
        chunks,
        rows,
        cols,
        _HEADER.size,
        payload_start,
        b"\0" * 24,
    )
    blob[: _HEADER.size] = header
    cursor = _HEADER.size
    for descriptor in descriptors:
        blob[cursor : cursor + _DESCRIPTOR.size] = _DESCRIPTOR.pack(
            descriptor.offset,
            descriptor.stored_bytes,
            descriptor.extent_bytes,
            descriptor.raw_bytes,
            descriptor.crc32,
            descriptor.mode,
            descriptor.changed_bytes,
        )
        cursor += _DESCRIPTOR.size

    raw_full = int(actual.nbytes)
    stats = ResidualContainerStats(
        experts=experts,
        chunks=chunks,
        rows=rows,
        cols=cols,
        alignment=alignment,
        directory_bytes=directory_end,
        payload_logical_bytes=logical_payload,
        payload_extent_bytes=extent_payload,
        file_bytes=len(blob),
        raw_full_bytes=raw_full,
        changed_bytes=changed_total,
        zero_chunks=mode_counts[_MODE_ZERO],
        sparse_chunks=mode_counts[_MODE_SPARSE_XOR],
        raw_chunks=mode_counts[_MODE_RAW_XOR],
        mean_extent_bytes_per_expert=extent_payload / experts,
    )
    return ResidualContainer(bytes(blob), tuple(descriptors), stats)


def parse_xor_residual_container(blob: bytes) -> ResidualContainer:
    if len(blob) < _HEADER.size:
        raise ValueError("container is truncated")
    magic, version, log2_alignment, experts, chunks, rows, cols, directory_offset, payload_start, _ = (
        _HEADER.unpack_from(blob, 0)
    )
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("unsupported residual container")
    if directory_offset != _HEADER.size:
        raise ValueError("invalid directory offset")
    alignment = 1 << log2_alignment
    descriptor_count = experts * chunks
    directory_end = directory_offset + descriptor_count * _DESCRIPTOR.size
    if directory_end > len(blob) or payload_start < directory_end:
        raise ValueError("invalid residual directory")
    slices = _row_slices(rows, chunks)
    raw_sizes = [int((s.stop - s.start) * cols * 4) for s in slices]
    descriptors: list[ResidualDescriptor] = []
    intervals: list[tuple[int, int]] = []
    logical = extent = changed = 0
    counts = {_MODE_ZERO: 0, _MODE_RAW_XOR: 0, _MODE_SPARSE_XOR: 0}
    cursor = directory_offset
    for index in range(descriptor_count):
        fields = _DESCRIPTOR.unpack_from(blob, cursor)
        descriptor = ResidualDescriptor(*map(int, fields))
        cursor += _DESCRIPTOR.size
        expected_raw = raw_sizes[index % chunks]
        if descriptor.raw_bytes != expected_raw:
            raise ValueError("descriptor raw byte count does not match shape")
        if descriptor.mode not in counts:
            raise ValueError("unknown residual mode")
        if descriptor.mode == _MODE_ZERO:
            if descriptor.stored_bytes or descriptor.extent_bytes or descriptor.changed_bytes:
                raise ValueError("non-empty ZERO descriptor")
        else:
            if descriptor.offset < payload_start or descriptor.offset % alignment:
                raise ValueError("payload offset is not aligned")
            if descriptor.extent_bytes % alignment or descriptor.stored_bytes > descriptor.extent_bytes:
                raise ValueError("invalid payload extent")
            end = descriptor.offset + descriptor.extent_bytes
            if end > len(blob):
                raise ValueError("payload extends past file")
            intervals.append((descriptor.offset, end))
            logical += descriptor.stored_bytes
            extent += descriptor.extent_bytes
        if descriptor.changed_bytes > descriptor.raw_bytes:
            raise ValueError("changed byte count exceeds chunk size")
        changed += descriptor.changed_bytes
        counts[descriptor.mode] += 1
        descriptors.append(descriptor)
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("payload extents overlap")
    stats = ResidualContainerStats(
        experts=experts,
        chunks=chunks,
        rows=rows,
        cols=cols,
        alignment=alignment,
        directory_bytes=directory_end,
        payload_logical_bytes=logical,
        payload_extent_bytes=extent,
        file_bytes=len(blob),
        raw_full_bytes=experts * rows * cols * 4,
        changed_bytes=changed,
        zero_chunks=counts[_MODE_ZERO],
        sparse_chunks=counts[_MODE_SPARSE_XOR],
        raw_chunks=counts[_MODE_RAW_XOR],
        mean_extent_bytes_per_expert=extent / experts,
    )
    return ResidualContainer(blob, tuple(descriptors), stats)


def decode_xor_residual_chunk(
    container: ResidualContainer | bytes,
    predictor_chunk: np.ndarray,
    expert: int,
    chunk: int,
) -> np.ndarray:
    parsed = parse_xor_residual_container(container) if isinstance(container, bytes) else container
    stats = parsed.stats
    if not 0 <= expert < stats.experts or not 0 <= chunk < stats.chunks:
        raise IndexError("expert or chunk index out of range")
    rowslice = _row_slices(stats.rows, stats.chunks)[chunk]
    predicted = np.ascontiguousarray(predictor_chunk, dtype="<f4")
    expected_shape = (rowslice.stop - rowslice.start, stats.cols)
    if predicted.shape != expected_shape:
        raise ValueError(f"predictor chunk must have shape {expected_shape}")
    descriptor = parsed.descriptors[expert * stats.chunks + chunk]
    base = np.frombuffer(_matrix_bytes(predicted), dtype=np.uint8).copy()
    if descriptor.mode == _MODE_RAW_XOR:
        payload = np.frombuffer(
            parsed.blob, dtype=np.uint8, count=descriptor.stored_bytes, offset=descriptor.offset
        )
        if payload.size != base.size:
            raise ValueError("raw XOR payload has wrong length")
        base ^= payload
    elif descriptor.mode == _MODE_SPARSE_XOR:
        n = descriptor.changed_bytes
        expected = n * 5
        if descriptor.stored_bytes != expected:
            raise ValueError("sparse XOR payload has wrong length")
        positions = np.frombuffer(parsed.blob, dtype="<u4", count=n, offset=descriptor.offset)
        values = np.frombuffer(
            parsed.blob, dtype=np.uint8, count=n, offset=descriptor.offset + 4 * n
        )
        if n and (int(positions[-1]) >= base.size or np.any(positions[1:] <= positions[:-1])):
            raise ValueError("invalid sparse XOR positions")
        base[positions.astype(np.int64)] ^= values
    elif descriptor.mode != _MODE_ZERO:
        raise ValueError("unsupported mode")
    decoded_bytes = base.tobytes()
    if zlib.crc32(decoded_bytes) & 0xFFFFFFFF != descriptor.crc32:
        raise ValueError("decoded chunk CRC mismatch")
    return np.frombuffer(decoded_bytes, dtype="<f4").reshape(expected_shape).copy()


def decode_xor_residual_expert(
    container: ResidualContainer | bytes, predictor_expert: np.ndarray, expert: int
) -> np.ndarray:
    parsed = parse_xor_residual_container(container) if isinstance(container, bytes) else container
    predictor = np.asarray(predictor_expert, dtype=np.float32)
    if predictor.shape != (parsed.stats.rows, parsed.stats.cols):
        raise ValueError("predictor expert shape does not match container")
    pieces = []
    for chunk, rowslice in enumerate(_row_slices(parsed.stats.rows, parsed.stats.chunks)):
        pieces.append(decode_xor_residual_chunk(parsed, predictor[rowslice], expert, chunk))
    return np.concatenate(pieces, axis=0)


def decode_xor_residual_all(
    container: ResidualContainer | bytes, predictor_weights: np.ndarray
) -> np.ndarray:
    parsed = parse_xor_residual_container(container) if isinstance(container, bytes) else container
    predictor = np.asarray(predictor_weights, dtype=np.float32)
    expected = (parsed.stats.experts, parsed.stats.rows, parsed.stats.cols)
    if predictor.shape != expected:
        raise ValueError(f"predictor weights must have shape {expected}")
    return np.stack(
        [decode_xor_residual_expert(parsed, predictor[e], e) for e in range(parsed.stats.experts)]
    )


@dataclass(frozen=True)
class RepresentationLedger:
    experts: int
    chunks: int
    full_expert_bytes: int
    raw_full_bytes: int
    resident_predictor_bytes: int
    resident_directory_bytes: int
    residual_file_bytes: int
    residual_logical_bytes: int
    residual_extent_bytes: int
    mean_residual_extent_bytes_per_expert: float
    exact_total_bytes: int
    exact_total_ratio: float
    mean_read_ratio: float
    changed_byte_fraction: float
    zero_chunk_fraction: float


def representation_ledger(
    trace: SyntheticExpertTrace, container: ResidualContainer
) -> RepresentationLedger:
    s = container.stats
    full_expert = s.rows * s.cols * 4
    exact_total = trace.resident_predictor_bytes + s.file_bytes
    return RepresentationLedger(
        experts=s.experts,
        chunks=s.chunks,
        full_expert_bytes=full_expert,
        raw_full_bytes=s.raw_full_bytes,
        resident_predictor_bytes=trace.resident_predictor_bytes,
        resident_directory_bytes=s.directory_bytes,
        residual_file_bytes=s.file_bytes,
        residual_logical_bytes=s.payload_logical_bytes,
        residual_extent_bytes=s.payload_extent_bytes,
        mean_residual_extent_bytes_per_expert=s.mean_extent_bytes_per_expert,
        exact_total_bytes=exact_total,
        exact_total_ratio=exact_total / s.raw_full_bytes,
        mean_read_ratio=s.mean_extent_bytes_per_expert / full_expert,
        changed_byte_fraction=s.changed_bytes / s.raw_full_bytes,
        zero_chunk_fraction=s.zero_chunks / (s.experts * s.chunks),
    )


# ---------------------------------------------------------------------------
# Quantization and conditional rate-distortion


@dataclass(frozen=True)
class QuantizedValues:
    symbols: np.ndarray
    reconstruction: np.ndarray
    centers: np.ndarray
    scale: float
    clipping: float


def quantize_symmetric(
    values: np.ndarray,
    *,
    levels: int = 9,
    clip_standard_deviations: float = 3.0,
    scale: float | None = None,
) -> QuantizedValues:
    if levels < 2:
        raise ValueError("levels must be at least two")
    array = np.asarray(values, dtype=np.float64)
    if scale is None:
        scale = float(np.sqrt(np.mean(array * array)))
    scale = max(float(scale), 1e-12)
    clipping = clip_standard_deviations * scale
    centers = np.linspace(-clipping, clipping, levels, dtype=np.float64)
    step = centers[1] - centers[0]
    symbols = np.rint((np.clip(array, -clipping, clipping) + clipping) / step).astype(np.int64)
    symbols = np.clip(symbols, 0, levels - 1)
    reconstruction = centers[symbols]
    return QuantizedValues(symbols, reconstruction, centers, scale, clipping)


@dataclass(frozen=True)
class RateDistortionPoint:
    beta: float
    distortion: float
    rate_bits: float


def _ba_for_distribution(
    probability: np.ndarray,
    distortion: np.ndarray,
    beta: float,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, float]:
    source = np.asarray(probability, dtype=np.float64)
    source = source / source.sum()
    if beta == 0.0:
        expected = source @ distortion
        best = int(np.argmin(expected))
        return float(expected[best]), 0.0
    reproduction = np.full(distortion.shape[1], 1.0 / distortion.shape[1], dtype=np.float64)
    for _ in range(max_iterations):
        log_weight = np.log(np.maximum(reproduction, 1e-300))[None, :] - beta * distortion
        log_weight -= np.max(log_weight, axis=1, keepdims=True)
        channel = np.exp(log_weight)
        channel /= np.sum(channel, axis=1, keepdims=True)
        new_reproduction = source @ channel
        if np.max(np.abs(new_reproduction - reproduction)) < tolerance:
            reproduction = new_reproduction
            break
        reproduction = new_reproduction
    log_weight = np.log(np.maximum(reproduction, 1e-300))[None, :] - beta * distortion
    log_weight -= np.max(log_weight, axis=1, keepdims=True)
    channel = np.exp(log_weight)
    channel /= np.sum(channel, axis=1, keepdims=True)
    achieved_distortion = float(np.sum(source[:, None] * channel * distortion))
    ratio = channel / np.maximum(reproduction[None, :], 1e-300)
    information = float(
        np.sum(source[:, None] * channel * np.log2(np.maximum(ratio, 1e-300)))
    )
    return achieved_distortion, max(0.0, information)


def conditional_rate_distortion_curve(
    symbols: np.ndarray,
    context: np.ndarray,
    centers: np.ndarray,
    *,
    betas: Sequence[float] | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 2_000,
) -> tuple[RateDistortionPoint, ...]:
    """Finite-alphabet conditional RD with side information at both ends.

    A shared Lagrange multiplier allocates distortion across context values.
    Context-conditioned Blahut--Arimoto updates are batched in one tensor.  The
    returned lossless endpoint is ``H(symbol | context)``.
    """

    z = np.asarray(symbols, dtype=np.int64).reshape(-1)
    c = encode_categories(context)
    if z.size != c.size:
        raise ValueError("symbols and context must have equal length")
    alphabet = int(np.asarray(centers).size)
    if np.any(z < 0) or np.any(z >= alphabet):
        raise ValueError("symbols lie outside reconstruction alphabet")
    values = np.asarray(centers, dtype=np.float64)
    distortion = (values[:, None] - values[None, :]) ** 2
    if betas is None:
        betas = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)

    context_count = int(c.max()) + 1
    counts = np.zeros((context_count, alphabet), dtype=np.float64)
    np.add.at(counts, (c, z), 1.0)
    totals = counts.sum(axis=1)
    keep = totals > 0
    counts = counts[keep]
    totals = totals[keep]
    source = counts / totals[:, None]
    context_probability = totals / z.size

    points: list[RateDistortionPoint] = []
    for beta_value in betas:
        beta = float(beta_value)
        if beta == 0.0:
            expected = source @ distortion
            context_distortion = np.min(expected, axis=1)
            total_d = float(np.dot(context_probability, context_distortion))
            points.append(RateDistortionPoint(beta, total_d, 0.0))
            continue
        reproduction = np.full(
            (source.shape[0], alphabet), 1.0 / alphabet, dtype=np.float64
        )
        for _ in range(max_iterations):
            log_weight = (
                np.log(np.maximum(reproduction, 1e-300))[:, None, :]
                - beta * distortion[None, :, :]
            )
            log_weight -= np.max(log_weight, axis=2, keepdims=True)
            channel = np.exp(log_weight)
            channel /= np.sum(channel, axis=2, keepdims=True)
            new_reproduction = np.einsum("cx,cxy->cy", source, channel, optimize=False)
            if np.max(np.abs(new_reproduction - reproduction)) < tolerance:
                reproduction = new_reproduction
                break
            reproduction = new_reproduction
        log_weight = (
            np.log(np.maximum(reproduction, 1e-300))[:, None, :]
            - beta * distortion[None, :, :]
        )
        log_weight -= np.max(log_weight, axis=2, keepdims=True)
        channel = np.exp(log_weight)
        channel /= np.sum(channel, axis=2, keepdims=True)
        context_distortion = np.sum(
            source[:, :, None] * channel * distortion[None, :, :], axis=(1, 2)
        )
        ratio = channel / np.maximum(reproduction[:, None, :], 1e-300)
        context_information = np.sum(
            source[:, :, None]
            * channel
            * np.log2(np.maximum(ratio, 1e-300)),
            axis=(1, 2),
        )
        points.append(
            RateDistortionPoint(
                beta,
                float(np.dot(context_probability, context_distortion)),
                max(0.0, float(np.dot(context_probability, context_information))),
            )
        )
    points.append(
        RateDistortionPoint(float("inf"), 0.0, conditional_entropy_bits(z, c))
    )
    # Remove numerically dominated BA iterates, then sort from the zero-rate
    # end toward the lossless endpoint.  A point is dominated when another
    # point attains no larger distortion and no larger rate, with one strict.
    frontier: list[RateDistortionPoint] = []
    for index, point in enumerate(points):
        dominated = False
        for other_index, other in enumerate(points):
            if index == other_index:
                continue
            no_worse = (
                other.distortion <= point.distortion + 1e-12
                and other.rate_bits <= point.rate_bits + 1e-12
            )
            strict = (
                other.distortion < point.distortion - 1e-12
                or other.rate_bits < point.rate_bits - 1e-12
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    frontier.sort(key=lambda point: (-point.distortion, point.rate_bits))
    filtered: list[RateDistortionPoint] = []
    for point in frontier:
        if filtered and (
            abs(point.distortion - filtered[-1].distortion) <= 1e-12
            and abs(point.rate_bits - filtered[-1].rate_bits) <= 1e-12
        ):
            continue
        filtered.append(point)
    return tuple(filtered)


# ---------------------------------------------------------------------------
# Exact finite prefix stopping policies


@dataclass(frozen=True)
class PrefixActionTable:
    context_probability: np.ndarray  # [C]
    cumulative_cost: np.ndarray  # [C, J+1]
    expected_distortion: np.ndarray  # [C, J+1]
    tail_risk: np.ndarray  # [C, J+1]

    def validate(self) -> None:
        p = np.asarray(self.context_probability, dtype=np.float64)
        cost = np.asarray(self.cumulative_cost, dtype=np.float64)
        distortion = np.asarray(self.expected_distortion, dtype=np.float64)
        risk = np.asarray(self.tail_risk, dtype=np.float64)
        if p.ndim != 1 or cost.ndim != 2 or cost.shape != distortion.shape or cost.shape != risk.shape:
            raise ValueError("invalid prefix table shapes")
        if cost.shape[0] != p.size or cost.shape[1] < 2:
            raise ValueError("prefix table must provide each context and at least one chunk")
        if np.any(p < 0) or not math.isclose(float(p.sum()), 1.0, abs_tol=1e-9):
            raise ValueError("context probabilities must sum to one")
        if np.any(cost < 0) or np.any(distortion < 0) or np.any((risk < 0) | (risk > 1)):
            raise ValueError("cost/distortion/risk values are invalid")
        if np.any(np.diff(cost, axis=1) < -1e-12):
            raise ValueError("cumulative cost must be nondecreasing")
        if np.any(np.diff(distortion, axis=1) > 1e-12):
            raise ValueError("distortion must be nonincreasing")


@dataclass(frozen=True)
class DeterministicPrefixPolicy:
    prefixes: tuple[int, ...]
    expected_cost: float
    constraint_value: float
    expected_distortion: float
    selective_risk_numerator: float
    approximate_probability: float


@dataclass(frozen=True)
class PrefixPolicySolution:
    mode: str
    bound: float
    expected_cost: float
    constraint_value: float
    components: tuple[tuple[float, DeterministicPrefixPolicy], ...]


def enumerate_prefix_policies(
    table: PrefixActionTable,
    *,
    mode: str,
    selective_alpha: float = 0.0,
) -> tuple[DeterministicPrefixPolicy, ...]:
    table.validate()
    p = np.asarray(table.context_probability, dtype=np.float64)
    cost = np.asarray(table.cumulative_cost, dtype=np.float64)
    distortion = np.asarray(table.expected_distortion, dtype=np.float64)
    risk = np.asarray(table.tail_risk, dtype=np.float64)
    contexts, choices = cost.shape
    full_prefix = choices - 1
    policies: list[DeterministicPrefixPolicy] = []
    for prefixes in itertools.product(range(choices), repeat=contexts):
        idx = np.asarray(prefixes, dtype=np.int64)
        rows = np.arange(contexts)
        expected_cost = float(np.sum(p * cost[rows, idx]))
        expected_distortion = float(np.sum(p * distortion[rows, idx]))
        approximate = idx < full_prefix
        approximate_probability = float(np.sum(p * approximate))
        risk_numerator = float(np.sum(p * risk[rows, idx] * approximate))
        if mode == "expected_distortion":
            constraint = expected_distortion
        elif mode == "selective_risk":
            constraint = risk_numerator - selective_alpha * approximate_probability
        else:
            raise ValueError("mode must be expected_distortion or selective_risk")
        policies.append(
            DeterministicPrefixPolicy(
                tuple(map(int, prefixes)),
                expected_cost,
                constraint,
                expected_distortion,
                risk_numerator,
                approximate_probability,
            )
        )
    return tuple(policies)


def solve_prefix_policy(
    table: PrefixActionTable,
    *,
    mode: str,
    bound: float,
    selective_alpha: float = 0.0,
    allow_randomization: bool = True,
) -> PrefixPolicySolution:
    """Solve a finite policy with one expected linear constraint exactly.

    For selective risk, ``bound`` should normally be zero and the constraint is
    ``E[(risk-alpha) 1{approximate}] <= 0``.  With one scalar constraint, an
    optimal randomized solution uses at most two deterministic policies.
    """

    policies = enumerate_prefix_policies(table, mode=mode, selective_alpha=selective_alpha)
    best_cost = math.inf
    best_components: tuple[tuple[float, DeterministicPrefixPolicy], ...] | None = None
    best_constraint = math.inf
    for policy in policies:
        if policy.constraint_value <= bound + 1e-12 and policy.expected_cost < best_cost:
            best_cost = policy.expected_cost
            best_constraint = policy.constraint_value
            best_components = ((1.0, policy),)
    if allow_randomization:
        for left_index, left in enumerate(policies):
            for right in policies[left_index + 1 :]:
                g0, g1 = left.constraint_value, right.constraint_value
                if abs(g0 - g1) <= 1e-15:
                    continue
                probability_left = (bound - g1) / (g0 - g1)
                if probability_left < -1e-12 or probability_left > 1.0 + 1e-12:
                    continue
                probability_left = min(1.0, max(0.0, probability_left))
                mixture_cost = probability_left * left.expected_cost + (1.0 - probability_left) * right.expected_cost
                mixture_constraint = probability_left * g0 + (1.0 - probability_left) * g1
                if mixture_constraint <= bound + 1e-10 and mixture_cost < best_cost - 1e-12:
                    best_cost = mixture_cost
                    best_constraint = mixture_constraint
                    best_components = (
                        (probability_left, left),
                        (1.0 - probability_left, right),
                    )
    if best_components is None:
        raise ValueError("prefix constraint is infeasible")
    return PrefixPolicySolution(mode, float(bound), best_cost, best_constraint, best_components)


def build_prefix_table_from_residuals(
    residual_chunks: np.ndarray,
    context: np.ndarray,
    chunk_costs: Sequence[float],
    *,
    tail_error_threshold: float,
) -> PrefixActionTable:
    """Estimate prefix distortion/risk for additive output chunks.

    ``residual_chunks`` has shape ``[samples, chunks, output]`` and the exact
    residual is their sum.  Prefix ``k`` includes chunks ``0..k-1``.
    """

    chunks_array = np.asarray(residual_chunks, dtype=np.float64)
    if chunks_array.ndim != 3:
        raise ValueError("residual_chunks must have shape [samples, chunks, output]")
    samples, chunks, _ = chunks_array.shape
    c = encode_categories(context)
    if c.size != samples:
        raise ValueError("context length does not match residual samples")
    costs = np.asarray(chunk_costs, dtype=np.float64)
    if costs.shape != (chunks,) or np.any(costs < 0):
        raise ValueError("chunk_costs has invalid shape or values")
    contexts = int(c.max()) + 1
    probabilities = np.bincount(c, minlength=contexts).astype(np.float64) / samples
    cumulative_cost = np.zeros((contexts, chunks + 1), dtype=np.float64)
    cumulative_cost[:, 1:] = np.cumsum(costs)[None, :]
    distortion = np.zeros_like(cumulative_cost)
    risk = np.zeros_like(cumulative_cost)
    for context_id in range(contexts):
        selected = chunks_array[c == context_id]
        if selected.size == 0:
            continue
        for prefix in range(chunks + 1):
            tail = np.sum(selected[:, prefix:, :], axis=1)
            squared = np.sum(tail * tail, axis=1)
            distortion[context_id, prefix] = float(np.mean(squared))
            risk[context_id, prefix] = float(np.mean(np.sqrt(squared) > tail_error_threshold))
    return PrefixActionTable(probabilities, cumulative_cost, distortion, risk)


# ---------------------------------------------------------------------------
# Exact small cache/representation optimizer


@dataclass(frozen=True)
class BasisPlacement:
    name: str
    ram_bytes: int
    vram_bytes: int
    fixed_cost: float = 0.0
    available: bool = True


@dataclass(frozen=True)
class ChunkPlacement:
    """One tier choice for an independently placeable residual chunk."""

    name: str
    ram_bytes: int
    vram_bytes: int
    expected_access_cost: float


@dataclass(frozen=True)
class ResidualChunkObject:
    """A residual chunk required by one representation option.

    ``access_probability`` is conditional on requesting the parent expert.  A
    chunk can be stored on disk, in RAM, or in VRAM by supplying corresponding
    placements.  Disk is represented by a zero-residency placement with a
    positive access cost.
    """

    name: str
    access_probability: float
    placements: tuple[ChunkPlacement, ...]


@dataclass(frozen=True)
class CacheOption:
    name: str
    ram_bytes: int
    vram_bytes: int
    expected_cost: float
    distortion: float = 0.0
    risk: float = 0.0
    requires_basis: bool = False
    residual_chunks: tuple[ResidualChunkObject, ...] = ()


@dataclass(frozen=True)
class CacheItem:
    name: str
    probability: float
    options: tuple[CacheOption, ...]


@dataclass(frozen=True)
class CacheProblem:
    items: tuple[CacheItem, ...]
    basis_placements: tuple[BasisPlacement, ...]
    ram_capacity: int
    vram_capacity: int
    max_expected_distortion: float = math.inf
    max_expected_risk: float = math.inf


@dataclass(frozen=True)
class CacheSolution:
    basis: str
    choices: tuple[tuple[str, str], ...]
    chunk_choices: tuple[tuple[str, str, str], ...]
    ram_bytes: int
    vram_bytes: int
    expected_cost: float
    expected_distortion: float
    expected_risk: float
    enumerated_configurations: int


def _validate_cache_problem(problem: CacheProblem) -> None:
    if problem.ram_capacity < 0 or problem.vram_capacity < 0:
        raise ValueError("cache capacities must be non-negative")
    if not problem.items or not problem.basis_placements:
        raise ValueError("cache problem needs items and basis placements")
    item_names = [item.name for item in problem.items]
    if len(set(item_names)) != len(item_names):
        raise ValueError("cache item names must be unique")
    total_probability = sum(item.probability for item in problem.items)
    if not math.isclose(total_probability, 1.0, abs_tol=1e-9):
        raise ValueError("item probabilities must sum to one")
    if any(item.probability < 0 or not item.options for item in problem.items):
        raise ValueError("cache items need non-negative probability and options")
    for basis in problem.basis_placements:
        if basis.ram_bytes < 0 or basis.vram_bytes < 0 or basis.fixed_cost < 0:
            raise ValueError("basis placement has invalid bytes or cost")
    for item in problem.items:
        for option in item.options:
            if (
                option.ram_bytes < 0
                or option.vram_bytes < 0
                or option.expected_cost < 0
                or option.distortion < 0
                or not 0.0 <= option.risk <= 1.0
            ):
                raise ValueError("cache option has invalid bytes, cost, distortion, or risk")
            names: set[str] = set()
            for chunk in option.residual_chunks:
                if chunk.name in names:
                    raise ValueError("residual chunk names must be unique within an option")
                names.add(chunk.name)
                if not 0.0 <= chunk.access_probability <= 1.0 or not chunk.placements:
                    raise ValueError("residual chunk has invalid access probability or placements")
                for placement in chunk.placements:
                    if (
                        placement.ram_bytes < 0
                        or placement.vram_bytes < 0
                        or placement.expected_access_cost < 0
                    ):
                        raise ValueError("chunk placement has invalid bytes or cost")


def solve_small_cache_problem(problem: CacheProblem) -> CacheSolution:
    """Exhaustively solve a finite representation, basis, and chunk placement.

    Every full expert, low-precision copy, basis placement, and independently
    addressable residual chunk is charged against the same RAM/VRAM budgets.
    The enumeration is exact for the supplied finite option sets.
    """

    _validate_cache_problem(problem)
    best: CacheSolution | None = None
    enumerated = 0
    for basis in problem.basis_placements:
        if not basis.available:
            continue
        basis_enabled = basis.name != "none"
        for options in itertools.product(*(item.options for item in problem.items)):
            chunk_objects: list[tuple[CacheItem, ResidualChunkObject]] = []
            for item, option in zip(problem.items, options):
                chunk_objects.extend((item, chunk) for chunk in option.residual_chunks)
            placement_products: Iterable[tuple[ChunkPlacement, ...]]
            if chunk_objects:
                placement_products = itertools.product(
                    *(chunk.placements for _, chunk in chunk_objects)
                )
            else:
                placement_products = ((),)
            for placements in placement_products:
                enumerated += 1
                if any(option.requires_basis and not basis_enabled for option in options):
                    continue
                ram = basis.ram_bytes + sum(option.ram_bytes for option in options)
                vram = basis.vram_bytes + sum(option.vram_bytes for option in options)
                ram += sum(placement.ram_bytes for placement in placements)
                vram += sum(placement.vram_bytes for placement in placements)
                if ram > problem.ram_capacity or vram > problem.vram_capacity:
                    continue

                per_item_cost = {
                    item.name: option.expected_cost
                    for item, option in zip(problem.items, options)
                }
                chunk_choices: list[tuple[str, str, str]] = []
                for (item, chunk), placement in zip(chunk_objects, placements):
                    per_item_cost[item.name] += (
                        chunk.access_probability * placement.expected_access_cost
                    )
                    chunk_choices.append((item.name, chunk.name, placement.name))
                expected_cost = basis.fixed_cost + sum(
                    item.probability * per_item_cost[item.name] for item in problem.items
                )
                expected_distortion = sum(
                    item.probability * option.distortion
                    for item, option in zip(problem.items, options)
                )
                expected_risk = sum(
                    item.probability * option.risk
                    for item, option in zip(problem.items, options)
                )
                if expected_distortion > problem.max_expected_distortion + 1e-12:
                    continue
                if expected_risk > problem.max_expected_risk + 1e-12:
                    continue
                if best is None or expected_cost < best.expected_cost - 1e-12:
                    best = CacheSolution(
                        basis=basis.name,
                        choices=tuple(
                            (item.name, option.name)
                            for item, option in zip(problem.items, options)
                        ),
                        chunk_choices=tuple(chunk_choices),
                        ram_bytes=ram,
                        vram_bytes=vram,
                        expected_cost=expected_cost,
                        expected_distortion=expected_distortion,
                        expected_risk=expected_risk,
                        enumerated_configurations=0,
                    )
    if best is None:
        raise ValueError("cache problem is infeasible")
    return CacheSolution(**{**asdict(best), "enumerated_configurations": enumerated})


def _demo_residual_chunks() -> tuple[ResidualChunkObject, ...]:
    placements = (
        ChunkPlacement("disk", 0, 0, 2.4),
        ChunkPlacement("ram", 2, 0, 0.6),
        ChunkPlacement("vram", 0, 2, 0.08),
    )
    return (
        ResidualChunkObject("head", 1.0, placements),
        ResidualChunkObject("tail", 0.35, placements),
    )


def demo_cache_problem(
    *, ram_capacity: int = 18, vram_capacity: int = 10, max_distortion: float = 0.03
) -> CacheProblem:
    probabilities = (0.50, 0.30, 0.20)
    items: list[CacheItem] = []
    for index, probability in enumerate(probabilities):
        items.append(
            CacheItem(
                f"expert-{index}",
                probability,
                (
                    CacheOption("disk-full", 0, 0, 10.0),
                    CacheOption("ram-full", 8, 0, 3.0),
                    CacheOption("vram-full", 0, 8, 0.2),
                    CacheOption(
                        "basis+residual",
                        0,
                        0,
                        0.35,
                        requires_basis=True,
                        residual_chunks=_demo_residual_chunks(),
                    ),
                    CacheOption(
                        "low-precision", 0, 4, 0.4, distortion=0.05, risk=0.02
                    ),
                ),
            )
        )
    return CacheProblem(
        tuple(items),
        (
            BasisPlacement("none", 0, 0),
            BasisPlacement("ram-basis", 6, 0, fixed_cost=0.1),
            BasisPlacement("vram-basis", 0, 4, fixed_cost=0.02),
        ),
        ram_capacity,
        vram_capacity,
        max_expected_distortion=max_distortion,
        max_expected_risk=0.02,
    )


# ---------------------------------------------------------------------------
# Parametric system-cost model


@dataclass(frozen=True)
class HardwareCostModel:
    ssd_gib_per_s: float = 4.0
    h2d_gib_per_s: float = 24.0
    gpu_tflop_per_s: float = 20.0
    cpu_gop_per_s: float = 50.0
    synchronization_us: float = 5.0
    layer_deadline_us: float = 1000.0


@dataclass(frozen=True)
class PredictedSystemCost:
    scheme: str
    requests: int
    cache_hit_rate: float
    read_bytes: float
    h2d_bytes: float
    gpu_operations: float
    cpu_operations: float
    synchronizations: float
    io_us: float
    h2d_us: float
    gpu_us: float
    cpu_us: float
    synchronization_time_us: float
    serial_total_us: float
    overlapped_total_us: float
    deadline_met: bool


def _transfer_us(byte_count: float, gib_per_s: float) -> float:
    if gib_per_s <= 0:
        raise ValueError("bandwidth must be positive")
    return float(byte_count) / (gib_per_s * (1 << 30)) * 1e6


def predict_system_costs(
    ledger: RepresentationLedger,
    *,
    input_dim: int,
    output_dim: int,
    rank: int,
    residual_nnz_per_expert: float,
    requests: int = 336,
    cache_hit_rate: float = 0.0,
    residual_chunk_fraction: float = 1.0,
    prediction_miss_probability: float = 0.0,
    hardware: HardwareCostModel | None = None,
) -> tuple[PredictedSystemCost, PredictedSystemCost]:
    h = hardware or HardwareCostModel()
    if not 0.0 <= cache_hit_rate <= 1.0:
        raise ValueError("cache_hit_rate must lie in [0, 1]")
    if not 0.0 <= residual_chunk_fraction <= 1.0:
        raise ValueError("residual_chunk_fraction must lie in [0, 1]")
    misses = requests * (1.0 - cache_hit_rate)

    def make(
        name: str,
        read_bytes: float,
        gpu_ops: float,
        cpu_ops: float,
        synchronizations: float,
    ) -> PredictedSystemCost:
        io_us = _transfer_us(read_bytes, h.ssd_gib_per_s)
        h2d_us = _transfer_us(read_bytes, h.h2d_gib_per_s)
        gpu_us = gpu_ops / (h.gpu_tflop_per_s * 1e12) * 1e6
        cpu_us = cpu_ops / (h.cpu_gop_per_s * 1e9) * 1e6
        sync_us = synchronizations * h.synchronization_us
        serial = io_us + h2d_us + gpu_us + cpu_us + sync_us
        overlapped = max(io_us + h2d_us, gpu_us + cpu_us) + sync_us
        return PredictedSystemCost(
            name,
            requests,
            cache_hit_rate,
            read_bytes,
            read_bytes,
            gpu_ops,
            cpu_ops,
            synchronizations,
            io_us,
            h2d_us,
            gpu_us,
            cpu_us,
            sync_us,
            serial,
            overlapped,
            overlapped <= h.layer_deadline_us,
        )

    baseline_read = misses * ledger.full_expert_bytes
    baseline_gpu = requests * 2.0 * input_dim * output_dim
    baseline = make("full-record", baseline_read, baseline_gpu, 0.0, misses)

    residual_read_per_miss = ledger.mean_residual_extent_bytes_per_expert * residual_chunk_fraction
    # A miss after optimistic truncation pays the remaining residual bytes and
    # one extra synchronization; this is an explicit parameter, not measured.
    residual_read = misses * residual_read_per_miss
    residual_read += misses * prediction_miss_probability * (
        ledger.mean_residual_extent_bytes_per_expert - residual_read_per_miss
    )
    predictor_gpu_per_request = 2.0 * rank * input_dim + 2.0 * output_dim * rank
    residual_gpu_per_request = 2.0 * residual_nnz_per_expert * residual_chunk_fraction
    scheme_gpu = requests * (predictor_gpu_per_request + residual_gpu_per_request)
    scheme_cpu = misses * ledger.changed_byte_fraction * ledger.full_expert_bytes
    synchronizations = misses * (
        residual_chunk_fraction * ledger.chunks
        + prediction_miss_probability * (1.0 - residual_chunk_fraction) * ledger.chunks
    )
    scheme = make("predictor+residual", residual_read, scheme_gpu, scheme_cpu, synchronizations)
    return baseline, scheme


# ---------------------------------------------------------------------------
# FP32 order checks for speculative continuation


def f32_bits(value: np.float32 | float) -> int:
    return int(np.asarray(value, dtype=np.float32).view(np.uint32).item())


def canonical_f32_dot(weights: np.ndarray, activation: np.ndarray) -> np.float32:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    x = np.asarray(activation, dtype=np.float32).reshape(-1)
    if w.size != x.size:
        raise ValueError("dot operands have different lengths")
    accumulator = np.float32(0.0)
    for weight, value in zip(w, x):
        product = np.float32(weight * value)
        accumulator = np.float32(accumulator + product)
    return accumulator


def prefix_then_continue_f32_dot(
    weights: np.ndarray, activation: np.ndarray, split: int
) -> np.float32:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    x = np.asarray(activation, dtype=np.float32).reshape(-1)
    if not 0 <= split <= w.size or w.size != x.size:
        raise ValueError("invalid split or operand lengths")
    accumulator = np.float32(0.0)
    for index in range(split):
        accumulator = np.float32(accumulator + np.float32(w[index] * x[index]))
    # Storing/reloading the FP32 accumulator models the continuation buffer.
    accumulator = np.asarray(accumulator, dtype=np.float32).copy().item()
    accumulator = np.float32(accumulator)
    for index in range(split, w.size):
        accumulator = np.float32(accumulator + np.float32(w[index] * x[index]))
    return accumulator


def additive_correction_f32_dot(
    predictor_weights: np.ndarray, residual_weights: np.ndarray, activation: np.ndarray
) -> np.float32:
    predicted = canonical_f32_dot(predictor_weights, activation)
    correction = canonical_f32_dot(residual_weights, activation)
    return np.float32(predicted + correction)


@dataclass(frozen=True)
class FP32Counterexample:
    predictor_weights: tuple[float, ...]
    residual_weights: tuple[float, ...]
    activation: tuple[float, ...]
    canonical_bits: int
    additive_bits: int
    canonical_value: float
    additive_value: float


def find_fp32_additive_counterexample(
    *, seed: int = 9, width: int = 8, attempts: int = 200_000
) -> FP32Counterexample:
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        predictor = rng.normal(0.0, 10.0, size=width).astype(np.float32)
        residual = rng.normal(0.0, 1e-3, size=width).astype(np.float32)
        activation = rng.normal(0.0, 10.0, size=width).astype(np.float32)
        exact_weights = np.asarray(predictor + residual, dtype=np.float32)
        canonical = canonical_f32_dot(exact_weights, activation)
        additive = additive_correction_f32_dot(predictor, residual, activation)
        if f32_bits(canonical) != f32_bits(additive):
            return FP32Counterexample(
                tuple(map(float, predictor)),
                tuple(map(float, residual)),
                tuple(map(float, activation)),
                f32_bits(canonical),
                f32_bits(additive),
                float(canonical),
                float(additive),
            )
    raise RuntimeError("no FP32 additive counterexample found")


def f32_fma(weight: np.float32 | float, value: np.float32 | float, accumulator: np.float32 | float) -> np.float32:
    """Round one multiply-add to FP32 after a fused host operation."""

    return np.float32(math.fma(float(np.float32(weight)), float(np.float32(value)), float(np.float32(accumulator))))


def canonical_f32_fma_dot(weights: np.ndarray, activation: np.ndarray) -> np.float32:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    x = np.asarray(activation, dtype=np.float32).reshape(-1)
    if w.size != x.size:
        raise ValueError("dot operands have different lengths")
    accumulator = np.float32(0.0)
    for weight, value in zip(w, x):
        accumulator = f32_fma(weight, value, accumulator)
    return accumulator


def prefix_then_continue_f32_fma_dot(
    weights: np.ndarray, activation: np.ndarray, split: int
) -> np.float32:
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    x = np.asarray(activation, dtype=np.float32).reshape(-1)
    if not 0 <= split <= w.size or w.size != x.size:
        raise ValueError("invalid split or operand lengths")
    accumulator = np.float32(0.0)
    for index in range(split):
        accumulator = f32_fma(w[index], x[index], accumulator)
    accumulator = np.asarray(accumulator, dtype=np.float32).copy().item()
    accumulator = np.float32(accumulator)
    for index in range(split, w.size):
        accumulator = f32_fma(w[index], x[index], accumulator)
    return accumulator


def additive_correction_f32_fma_dot(
    predictor_weights: np.ndarray, residual_weights: np.ndarray, activation: np.ndarray
) -> np.float32:
    predicted = canonical_f32_fma_dot(predictor_weights, activation)
    correction = canonical_f32_fma_dot(residual_weights, activation)
    return np.float32(predicted + correction)


def find_fp32_fma_additive_counterexample(
    *, seed: int = 19, width: int = 8, attempts: int = 200_000
) -> FP32Counterexample:
    rng = np.random.default_rng(seed)
    for _ in range(attempts):
        predictor = rng.normal(0.0, 10.0, size=width).astype(np.float32)
        residual = rng.normal(0.0, 1e-3, size=width).astype(np.float32)
        activation = rng.normal(0.0, 10.0, size=width).astype(np.float32)
        exact_weights = np.asarray(predictor + residual, dtype=np.float32)
        canonical = canonical_f32_fma_dot(exact_weights, activation)
        additive = additive_correction_f32_fma_dot(predictor, residual, activation)
        if f32_bits(canonical) != f32_bits(additive):
            return FP32Counterexample(
                tuple(map(float, predictor)),
                tuple(map(float, residual)),
                tuple(map(float, activation)),
                f32_bits(canonical),
                f32_bits(additive),
                float(canonical),
                float(additive),
            )
    raise RuntimeError("no FP32 FMA additive counterexample found")


__all__ = [name for name in globals() if not name.startswith("_")]
