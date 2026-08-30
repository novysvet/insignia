#!/usr/bin/env python3
"""Join exact-teacher falsifier events with target and DFlash logit dumps.

The binary event trace contains only causal runtime observations and the local
8x8 expert-contribution Gram label. This tool adds compact vocabulary features
and downstream full-logit labels without copying 154,880 floats into every
training sample.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import struct
from pathlib import Path

import numpy as np


TRACE_HEADER = struct.Struct("<8sHHIHHHHHHII28s")
TRACE_MAGIC = b"INSFAL1\0"
TRACE_VERSION = 2

EVENT_DTYPE = np.dtype([
    ("epoch", "<u4"),
    ("layer", "<u2"),
    ("row", "u1"),
    ("tokens", "u1"),
    ("verify_row", "u1"),
    ("exec_k", "u1"),
    ("flags", "<u2"),
    ("candidate_residency", "<u4", (4,)),
    ("expert", "<u2", (8,)),
    ("weight", "<f4", (8,)),
    ("candidate_expert", "<u2", (32,)),
    ("candidate_logit", "<f4", (32,)),
    ("candidate_choice", "<f4", (32,)),
    ("router_summary", "<f4", (8,)),
    ("hidden_countsketch", "<f4", (64,)),
    ("contribution_gram", "<f4", (36,)),
    ("tail", "<f4", (4,)),
    ("reserved", "u1", (52,)),
])
assert EVENT_DTYPE.itemsize == 896

ROW_SCALAR_NAMES = (
    "prior_entropy_norm", "prior_top1_p", "prior_margin",
    "draft_entropy_norm", "draft_top1_p", "draft_margin",
    "calibration_js", "calibration_centered_cos", "calibration_top1_disagree",
    "prior_draft_top8_overlap", "prior_draft_top32_overlap",
    "block_fraction", "row_fraction",
)

ROW_LABEL_NAMES = (
    "mse", "centered_mse", "cosine", "centered_cosine", "kl_exact_approx",
    "js", "top1_mismatch", "top10_overlap", "max_abs",
)

EVENT_DERIVED_NAMES = (
    "host_ready_fraction", "host_inflight_fraction", "device_fraction",
    "pinned_fraction", "previous_layer_overlap", "previous_row_overlap",
    "previous_round_overlap", "row_union_fraction", "row_unique_fraction",
)


def read_trace(path: Path) -> tuple[dict[str, int], np.memmap]:
    with path.open("rb") as handle:
        raw = handle.read(TRACE_HEADER.size)
    if len(raw) != TRACE_HEADER.size:
        raise SystemExit(f"{path}: truncated falsifier header")
    (magic, version, header_bytes, record_bytes, layers, experts, topk,
     candidate_k, hidden_sketch, _, hidden, flags, _) = TRACE_HEADER.unpack(raw)
    if magic != TRACE_MAGIC:
        raise SystemExit(f"{path}: bad falsifier magic {magic!r}")
    if version != TRACE_VERSION or header_bytes != TRACE_HEADER.size:
        raise SystemExit(f"{path}: unsupported falsifier schema v{version}/{header_bytes}")
    if (record_bytes != EVENT_DTYPE.itemsize or topk != 8 or candidate_k != 32 or
            hidden_sketch != 64):
        raise SystemExit(f"{path}: unsupported falsifier record geometry")
    payload = path.stat().st_size - header_bytes
    if payload < 0 or payload % record_bytes:
        raise SystemExit(f"{path}: partial falsifier event")
    events = np.memmap(path, dtype=EVENT_DTYPE, mode="r", offset=header_bytes,
                       shape=(payload // record_bytes,))
    if not len(events):
        raise SystemExit(f"{path}: no falsifier events")
    return {
        "layers": layers,
        "experts": experts,
        "topk": topk,
        "candidate_k": candidate_k,
        "hidden": hidden,
        "flags": flags,
    }, events


def read_logits(path: Path, vocab: int) -> np.memmap:
    raw = np.memmap(path, dtype="<f4", mode="r")
    if raw.size % vocab:
        raise SystemExit(f"{path}: partial {vocab}-wide logit record")
    return raw.reshape((-1, vocab))


def softmax_stats(raw: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    values = raw.astype(np.float64)
    maximum = float(np.max(values))
    shifted = np.exp(values - maximum)
    total = float(np.sum(shifted))
    probabilities = shifted / total
    top = np.partition(values, -2)[-2:]
    first, second = float(max(top)), float(min(top))
    entropy = math.log(total) + maximum - float(np.dot(probabilities, values))
    return probabilities, (entropy / math.log(len(values)), 1.0 / total, first - second)


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    a -= float(np.mean(a))
    b -= float(np.mean(b))
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator else 1.0


def raw_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator else 1.0


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    mixture = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    return 0.5 * (
        float(np.dot(left[left_mask], np.log(left[left_mask] / mixture[left_mask]))) +
        float(np.dot(right[right_mask], np.log(right[right_mask] / mixture[right_mask])))
    )


def top_values(raw: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argpartition(raw, -count)[-count:]
    indices = indices[np.argsort(raw[indices])[::-1]]
    values = raw[indices].astype(np.float32)
    values -= np.float32(np.mean(raw, dtype=np.float64))
    return indices.astype(np.int32), values


class CountSketch:
    def __init__(self, width: int, dimension: int):
        if dimension < 1 or dimension & (dimension - 1):
            raise SystemExit("--sketch-dim must be a power of two")
        values = np.arange(1, width + 1, dtype=np.uint32)
        values *= np.uint32(0x9E3779B1)
        values ^= values >> np.uint32(16)
        values *= np.uint32(0x85EBCA6B)
        values ^= values >> np.uint32(13)
        self.bucket = (values & np.uint32(dimension - 1)).astype(np.intp)
        self.sign = np.where(values & np.uint32(dimension), 1.0, -1.0)
        self.dimension = dimension
        self.scale = math.sqrt(dimension / width)

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        values = raw.astype(np.float64)
        deviation = values - float(np.mean(values))
        standard_deviation = float(np.std(deviation))
        if standard_deviation:
            deviation /= standard_deviation
        result = np.bincount(self.bucket, weights=self.sign * deviation,
                             minlength=self.dimension)
        return (result * self.scale).astype(np.float32)


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    return len(set(map(int, left)) & set(map(int, right))) / len(left)


def event_derived(events: np.ndarray, epoch_rank: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros((len(events), len(EVENT_DERIVED_NAMES)), dtype=np.float32)
    multiplicity = np.zeros((len(events), 8), dtype=np.float32)
    by_key = {(int(event["epoch"]), int(event["layer"]), int(event["verify_row"])): index
              for index, event in enumerate(events)}
    rank_epoch = {rank: epoch for epoch, rank in epoch_rank.items()}
    groups: dict[tuple[int, int], list[int]] = {}
    for index, event in enumerate(events):
        groups.setdefault((int(event["epoch"]), int(event["layer"])), []).append(index)
    group_counts: dict[tuple[int, int], dict[int, int]] = {}
    for key, indices in groups.items():
        counts: dict[int, int] = {}
        for index in indices:
            for expert in events[index]["expert"]:
                counts[int(expert)] = counts.get(int(expert), 0) + 1
        group_counts[key] = counts
    for index, event in enumerate(events):
        candidate_rank = {int(expert): rank
                          for rank, expert in enumerate(event["candidate_expert"])}
        missing = [int(expert) for expert in event["expert"] if int(expert) not in candidate_rank]
        if missing:
            raise SystemExit(f"baseline experts absent from top-32 candidates: {missing}")
        for group, mask in enumerate(event["candidate_residency"]):
            resident = sum((int(mask) >> candidate_rank[int(expert)]) & 1
                           for expert in event["expert"])
            result[index, group] = resident / 8.0
        epoch = int(event["epoch"])
        layer = int(event["layer"])
        row = int(event["verify_row"])
        experts = event["expert"]
        previous_layer = by_key.get((epoch, layer - 1, row))
        previous_row = by_key.get((epoch, layer, row - 1))
        previous_epoch = rank_epoch.get(epoch_rank[epoch] - 1)
        previous_round = (by_key.get((previous_epoch, layer, row))
                          if previous_epoch is not None else None)
        for column, other in ((4, previous_layer), (5, previous_row), (6, previous_round)):
            if other is not None:
                result[index, column] = overlap(experts, events[other]["expert"])

        peers = groups[(epoch, layer)]
        counts = group_counts[(epoch, layer)]
        union = set(counts)
        unique = sum(counts[int(expert)] == 1 for expert in experts)
        result[index, 7] = len(union) / (8.0 * len(peers))
        result[index, 8] = unique / 8.0
        multiplicity[index] = [counts[int(expert)] / len(peers) for expert in experts]
    return result, multiplicity


def validate_events(events: np.ndarray, layers: int) -> tuple[list[int], dict[int, int]]:
    keys = [(int(row["epoch"]), int(row["layer"]), int(row["verify_row"]))
            for row in events]
    if len(keys) != len(set(keys)):
        duplicates = [(key, count) for key, count in Counter(keys).items() if count > 1]
        raise SystemExit(f"duplicate falsifier (epoch,layer,row) event: {duplicates[:8]}")
    epochs = sorted({int(value) for value in events["epoch"]})
    rank = {epoch: index for index, epoch in enumerate(epochs)}
    sparse_layers = sorted({int(value) for value in events["layer"]})
    if sparse_layers and (min(sparse_layers) < 0 or max(sparse_layers) >= layers):
        raise SystemExit("trace layer exceeds header geometry")
    for epoch in epochs:
        epoch_events = events[events["epoch"] == epoch]
        rows = sorted({int(value) for value in epoch_events["verify_row"]})
        for row in rows:
            observed = sorted(int(value) for value in epoch_events["layer"]
                              [epoch_events["verify_row"] == row])
            if observed != sparse_layers:
                raise SystemExit(f"epoch {epoch} row {row}: incomplete sparse-layer sequence")
    return epochs, rank


def repair_legacy_zero_epochs(raw_events: np.ndarray) -> np.ndarray:
    """Recover forced-quality blocks written before force_logits advanced epochs."""
    keys = [(int(row["epoch"]), int(row["layer"]), int(row["verify_row"]))
            for row in raw_events]
    if len(keys) == len(set(keys)):
        return np.asarray(raw_events)
    if len({key[0] for key in keys}) != 1:
        return np.asarray(raw_events)
    repaired = np.array(raw_events, copy=True)
    block = 0
    previous_layer = int(repaired[0]["layer"])
    base_epoch = int(repaired[0]["epoch"])
    for event in repaired:
        layer = int(event["layer"])
        if layer < previous_layer:
            block += 1
        event["epoch"] = base_epoch + block
        previous_layer = layer
    return repaired


def build(args: argparse.Namespace) -> None:
    geometry, raw_events = read_trace(args.trace)
    repaired_events = repair_legacy_zero_epochs(raw_events)
    order = np.lexsort((repaired_events["verify_row"], repaired_events["layer"],
                        repaired_events["epoch"]))
    events = np.asarray(repaired_events[order])
    epochs, epoch_rank = validate_events(events, geometry["layers"])
    exact = read_logits(args.exact_logits, args.vocab)
    approximate = read_logits(args.approx_logits, args.vocab)
    draft = read_logits(args.draft_logits, args.vocab)
    if exact.shape != approximate.shape:
        raise SystemExit("exact and approximate target-logit shapes differ")
    if draft.shape[0] % args.draft_rows:
        raise SystemExit("DFlash logit dump has a partial block")
    draft = draft.reshape((-1, args.draft_rows, args.vocab))
    expected_blocks = math.ceil(max(0, len(exact) - 1) / args.verify_k)
    if len(epochs) != expected_blocks or len(draft) != expected_blocks:
        raise SystemExit(
            f"alignment mismatch: epochs={len(epochs)} draft={len(draft)} "
            f"target-derived blocks={expected_blocks}")

    sketcher = CountSketch(args.vocab, args.sketch_dim)
    row_meta: list[tuple[int, int, int, int]] = []
    row_scalars: list[list[float]] = []
    row_sketches: list[np.ndarray] = []
    prior_top_ids: list[np.ndarray] = []
    prior_top_values: list[np.ndarray] = []
    draft_top_ids: list[np.ndarray] = []
    draft_top_values: list[np.ndarray] = []
    row_labels: list[list[float]] = []
    row_top1: list[tuple[int, int]] = []

    for block, epoch in enumerate(epochs):
        start = block * args.verify_k
        prior = approximate[start]
        prior_probability, prior_stats = softmax_stats(prior)
        calibration = draft[block, 0]
        calibration_probability, _ = softmax_stats(calibration)
        calibration_js = js_divergence(prior_probability, calibration_probability)
        calibration_cos = centered_cosine(prior, calibration)
        calibration_disagree = float(np.argmax(prior) != np.argmax(calibration))
        prior_ids, prior_values = top_values(prior, args.top_logits)
        stop = min(start + args.verify_k + 1, len(exact))
        for output_record in range(start + 1, stop):
            row = output_record - start - 1
            draft_row = row + 1
            if draft_row >= args.draft_rows:
                raise SystemExit(f"block {block}: draft row {draft_row} unavailable")
            current_draft = draft[block, draft_row]
            draft_probability, draft_stats = softmax_stats(current_draft)
            current_ids, current_values = top_values(current_draft, args.top_logits)
            exact_row = exact[output_record]
            approximate_row = approximate[output_record]
            exact_probability, _ = softmax_stats(exact_row)
            approximate_probability, _ = softmax_stats(approximate_row)
            delta = approximate_row.astype(np.float64) - exact_row.astype(np.float64)
            centered_delta = delta - float(np.mean(delta))
            top10_exact = np.argpartition(exact_row, -10)[-10:]
            top10_approximate = np.argpartition(approximate_row, -10)[-10:]
            row_meta.append((epoch, block, row, output_record))
            row_scalars.append([
                *prior_stats, *draft_stats, calibration_js, calibration_cos,
                calibration_disagree, overlap(prior_ids[:8], current_ids[:8]),
                overlap(prior_ids, current_ids), block / max(1, len(epochs) - 1),
                row / max(1, args.verify_k - 1),
            ])
            row_sketches.append(np.stack((sketcher(prior), sketcher(current_draft),
                                          sketcher(current_draft - prior))))
            prior_top_ids.append(prior_ids)
            prior_top_values.append(prior_values)
            draft_top_ids.append(current_ids)
            draft_top_values.append(current_values)
            row_labels.append([
                float(np.dot(delta, delta) / args.vocab),
                float(np.dot(centered_delta, centered_delta) / args.vocab),
                raw_cosine(exact_row, approximate_row),
                centered_cosine(exact_row, approximate_row),
                float(np.dot(exact_probability,
                             np.log(np.maximum(exact_probability, 1e-300) /
                                    np.maximum(approximate_probability, 1e-300)))),
                js_divergence(exact_probability, approximate_probability),
                float(np.argmax(exact_row) != np.argmax(approximate_row)),
                len(set(map(int, top10_exact)) & set(map(int, top10_approximate))) / 10.0,
                float(np.max(np.abs(delta))),
            ])
            row_top1.append((int(np.argmax(exact_row)), int(np.argmax(approximate_row))))

    event_rows = {(int(epoch), int(row)): index
                  for index, (epoch, _, row, _) in enumerate(row_meta)}
    event_row_index = np.asarray([
        event_rows.get((int(event["epoch"]), int(event["verify_row"])), -1)
        for event in events
    ], dtype=np.int32)
    if np.any(event_row_index < 0):
        raise SystemExit("an event has no matching final-logit row")

    derived, multiplicity = event_derived(events, epoch_rank)
    gram_sum = np.zeros(len(events), dtype=np.float64)
    for index, upper in enumerate(events["contribution_gram"]):
        cursor = 0
        total = 0.0
        for left in range(8):
            for right in range(left, 8):
                value = float(upper[cursor])
                total += value if left == right else 2.0 * value
                cursor += 1
        gram_sum[index] = total
    exact_norm = events["tail"][:, 0].astype(np.float64)
    relative = np.abs(gram_sum - exact_norm) / np.maximum(np.abs(exact_norm), 1e-12)
    if float(np.quantile(relative, 0.99)) > args.gram_rel_tolerance:
        raise SystemExit(f"Gram/exact norm validation failed: p99 relative error "
                         f"{float(np.quantile(relative, 0.99)):.3g}")

    metadata = {
        "schema": "insignia-falsifier-dataset-v2",
        "trace": str(args.trace),
        "prompt_id": args.prompt_id,
        "family": args.family,
        "policy": args.policy,
        "geometry": geometry,
        "verify_k": args.verify_k,
        "draft_rows": args.draft_rows,
        "vocab": args.vocab,
        "top_logits": args.top_logits,
        "sketch_dim": args.sketch_dim,
        "epochs": epochs,
        "event_derived_names": EVENT_DERIVED_NAMES,
        "row_scalar_names": ROW_SCALAR_NAMES,
        "row_label_names": ROW_LABEL_NAMES,
        "causality": "event features precede expert execution; Gram and row labels are targets",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        event_meta=np.stack((events["epoch"], events["layer"], events["verify_row"],
                             events["tokens"], events["flags"], events["exec_k"]),
                            axis=1).astype(np.int64),
        event_row_index=event_row_index,
        expert_ids=events["expert"].astype(np.int16),
        candidate_ids=events["candidate_expert"].astype(np.int16),
        candidate_logits=events["candidate_logit"].astype(np.float32),
        candidate_choice=events["candidate_choice"].astype(np.float32),
        candidate_residency=events["candidate_residency"].astype(np.uint32),
        expert_multiplicity=multiplicity,
        router_features=np.concatenate((events["weight"], events["router_summary"]),
                                       axis=1).astype(np.float32),
        hidden_countsketch=events["hidden_countsketch"].astype(np.float32),
        event_derived=derived,
        contribution_gram=events["contribution_gram"].astype(np.float32),
        event_tail=events["tail"].astype(np.float32),
        row_meta=np.asarray(row_meta, dtype=np.int32),
        row_scalars=np.asarray(row_scalars, dtype=np.float32),
        row_logit_sketch=np.asarray(row_sketches, dtype=np.float32),
        prior_top_ids=np.asarray(prior_top_ids, dtype=np.int32),
        prior_top_values=np.asarray(prior_top_values, dtype=np.float32),
        draft_top_ids=np.asarray(draft_top_ids, dtype=np.int32),
        draft_top_values=np.asarray(draft_top_values, dtype=np.float32),
        row_labels=np.asarray(row_labels, dtype=np.float32),
        row_top1=np.asarray(row_top1, dtype=np.int32),
    )
    print(f"wrote {args.output}: {len(events)} events, {len(row_meta)} rows, "
          f"Gram p99 relative check {float(np.quantile(relative, 0.99)):.3g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("exact_logits", type=Path)
    parser.add_argument("approx_logits", type=Path)
    parser.add_argument("draft_logits", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt-id", default="unknown")
    parser.add_argument("--family", default="unknown")
    parser.add_argument("--policy", default="unknown")
    parser.add_argument("--vocab", type=int, default=154880)
    parser.add_argument("--verify-k", type=int, default=4)
    parser.add_argument("--draft-rows", type=int, default=7)
    parser.add_argument("--top-logits", type=int, default=32)
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--gram-rel-tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    if args.top_logits < 10 or args.top_logits > args.vocab:
        parser.error("--top-logits must be in [10,vocab]")
    return args


if __name__ == "__main__":
    build(parse_args())
