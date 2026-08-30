#!/usr/bin/env python3
"""Test whether previous target logits add causal signal beyond router mass.

With --approx-logits, also simulate a causal whole-block exact fallback: the
logits at record b*k are available before verify block b, whose outputs are
records b*k+1..b*k+k.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = mean(left), mean(right)
    covariance = math.fsum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_var = math.fsum((x - left_mean) ** 2 for x in left)
    right_var = math.fsum((y - right_mean) ** 2 for y in right)
    return covariance / math.sqrt(left_var * right_var) if left_var and right_var else 0.0


def read_metrics(path: Path) -> dict[tuple[int, int, int], dict[int, dict[str, float]]]:
    groups: dict[tuple[int, int, int], dict[int, dict[str, float]]] = defaultdict(dict)
    occurrences: Counter[tuple[int, int, int]] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["semantics"] != "zero":
                continue
            layer, row, topm = int(raw["layer"]), int(raw["row"]), int(raw["topm"])
            base = (layer, row, topm)
            batch = occurrences[base]
            occurrences[base] += 1
            groups[(batch, layer, row)][topm] = {
                "mse": float(raw["mse"]),
                "rel_l2": float(raw["rel_l2"]),
                "cosine": float(raw["cosine"]),
                "retained_mass": float(raw["retained_mass"]),
            }
    exact = {"mse": 0.0, "rel_l2": 0.0, "cosine": 1.0, "retained_mass": 1.0}
    for key, choices in groups.items():
        if set(choices) != set(range(1, 8)):
            raise SystemExit(f"incomplete frontier at {key}: {sorted(choices)}")
        choices[8] = exact
    return groups


def choose(choices: dict[int, dict[str, float]], threshold: float, min_k: int) -> int:
    for topm in range(min_k, 9):
        if choices[topm]["retained_mass"] >= threshold:
            return topm
    return 8


def vector_logit_features(raw: np.ndarray) -> dict[str, float]:
    logits = raw.astype(np.float64)
    top = np.partition(logits, -2)[-2:]
    maximum, second = float(max(top)), float(min(top))
    shifted = np.exp(logits - maximum)
    normalizer = float(np.sum(shifted))
    probabilities = shifted / normalizer
    entropy = math.log(normalizer) + maximum - float(np.dot(probabilities, logits))
    return {
        "entropy": entropy,
        "entropy_norm": entropy / math.log(len(logits)),
        "top1_p": 1.0 / normalizer,
        "margin": maximum - second,
    }


def logit_features(path: Path, vocab: int, records: list[int]) -> dict[int, dict[str, float]]:
    raw = np.memmap(path, dtype="<f4", mode="r")
    if raw.size % vocab:
        raise SystemExit(f"{path}: partial logit record")
    matrix = raw.reshape((-1, vocab))
    result = {}
    for record in records:
        if record >= len(matrix):
            raise SystemExit(f"{path}: needs record {record}, has {len(matrix)}")
        result[record] = vector_logit_features(matrix[record])
    return result


def draft_logit_guard(path: Path, approx: np.ndarray, blocks: list[dict],
                      vocab: int) -> None:
    raw = np.memmap(path, dtype="<f4", mode="r")
    rows_per_block = 7
    if raw.size % (rows_per_block * vocab):
        raise SystemExit(f"{path}: partial DFlash logit block")
    draft = raw.reshape((-1, rows_per_block, vocab))
    if len(draft) != len(blocks):
        raise SystemExit(f"{path}: has {len(draft)} blocks, expected {len(blocks)}")

    samples = []
    for index, block in enumerate(blocks):
        target = approx[block["start"]].astype(np.float64)
        first = draft[index, 0].astype(np.float64)
        target_shift = np.exp(target - float(np.max(target)))
        draft_shift = np.exp(first - float(np.max(first)))
        target_prob = target_shift / float(np.sum(target_shift))
        draft_prob = draft_shift / float(np.sum(draft_shift))
        mixture = 0.5 * (target_prob + draft_prob)
        positive_target = target_prob > 0
        positive_draft = draft_prob > 0
        js = 0.5 * (
            float(np.dot(target_prob[positive_target],
                         np.log(target_prob[positive_target] / mixture[positive_target]))) +
            float(np.dot(draft_prob[positive_draft],
                         np.log(draft_prob[positive_draft] / mixture[positive_draft]))))
        target -= float(np.mean(target))
        first -= float(np.mean(first))
        calibration_cos = float(np.dot(target, first) / math.sqrt(
            float(np.dot(target, target)) * float(np.dot(first, first))))
        calibration_disagree = float(np.argmax(target) != np.argmax(first))

        # Target output after candidate r predicts candidate r+1, so DFlash
        # row r+1 is the causal same-position uncertainty signal.
        for offset, error in enumerate(block["rows"], start=1):
            if offset >= rows_per_block:
                continue
            samples.append({
                **error,
                **{f"draft_{key}": value for key, value in
                   vector_logit_features(draft[index, offset]).items()},
                "calibration_js": js,
                "calibration_cos": calibration_cos,
                "calibration_disagree": calibration_disagree,
                "row": float(offset),
            })

    risks = (("draft_entropy_norm", 1.0), ("draft_top1_p", -1.0),
             ("draft_margin", -1.0), ("calibration_js", 1.0),
             ("calibration_cos", -1.0), ("calibration_disagree", 1.0),
             ("row", 1.0))
    print("\nCausal current-block DFlash risk (same-position draft logits):")
    print("| risk feature | MSE | cosine loss | top1 mismatch |")
    print("|---|---:|---:|---:|")
    for name, sign in risks:
        values = [sign * sample[name] for sample in samples]
        print(f"| {name} | {pearson(values, [s['mse'] for s in samples]):+.4f} "
              f"| {pearson(values, [s['cos_loss'] for s in samples]):+.4f} "
              f"| {pearson(values, [float(s['mismatch']) for s in samples]):+.4f} |")

    print("\nCausal per-row fallback frontier (highest predicted risk exactified):")
    print("| risk feature | exact rows | remaining MSE | cosine | top1 mismatches |")
    print("|---|---:|---:|---:|---:|")
    total = len(samples)
    for name, sign in risks:
        order = sorted(range(total), key=lambda i: sign * samples[i][name], reverse=True)
        for guarded_count in sorted({0, math.ceil(total / 4), math.ceil(total / 2)}):
            guarded = set(order[:guarded_count])
            kept = [sample for index, sample in enumerate(samples) if index not in guarded]
            mse = math.fsum(sample["mse"] for sample in kept) / total
            cos_loss = math.fsum(sample["cos_loss"] for sample in kept) / total
            mismatches = sum(sample["mismatch"] for sample in kept)
            print(f"| {name} | {guarded_count}/{total} | {mse:.4e} "
                  f"| {1.0 - cos_loss:.6f} | {mismatches} |")


def final_logit_guard(exact_path: Path, approx_path: Path, draft_path: Path | None,
                      vocab: int, verify_k: int) -> None:
    exact_raw = np.memmap(exact_path, dtype="<f4", mode="r")
    approx_raw = np.memmap(approx_path, dtype="<f4", mode="r")
    if exact_raw.size != approx_raw.size or exact_raw.size % vocab:
        raise SystemExit("exact/approx logit dumps have incompatible shapes")
    exact = exact_raw.reshape((-1, vocab))
    approx = approx_raw.reshape((-1, vocab))
    if len(exact) < 2:
        raise SystemExit("logit guard needs at least two records")

    starts = list(range(0, len(exact) - 1, verify_k))
    features = logit_features(approx_path, vocab, starts)
    blocks = []
    for start in starts:
        stop = min(start + verify_k + 1, len(exact))
        rows = []
        for record in range(start + 1, stop):
            left = exact[record].astype(np.float64)
            right = approx[record].astype(np.float64)
            delta = right - left
            left_norm = float(np.dot(left, left))
            right_norm = float(np.dot(right, right))
            cosine = (float(np.dot(left, right)) / math.sqrt(left_norm * right_norm)
                      if left_norm and right_norm else 1.0)
            rows.append({
                "record": record,
                "mse": float(np.dot(delta, delta) / vocab),
                "cos_loss": 1.0 - cosine,
                "mismatch": int(np.argmax(left) != np.argmax(right)),
            })
        blocks.append({
            "start": start,
            "rows": rows,
            "mse": mean([row["mse"] for row in rows]),
            "cos_loss": mean([row["cos_loss"] for row in rows]),
            "mismatch": sum(row["mismatch"] for row in rows),
            **features[start],
        })

    print("\nCausal final-logit block risk (--approx-logits):")
    print("| block | prior record | entropy/logV | top1 p | margin | output rows | MSE | cosine | top1 mismatches |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index, block in enumerate(blocks):
        print(f"| {index} | {block['start']} | {block['entropy_norm']:.5f} "
              f"| {block['top1_p']:.5f} | {block['margin']:.4f} "
              f"| {len(block['rows'])} | {block['mse']:.4e} "
              f"| {1.0 - block['cos_loss']:.6f} | {block['mismatch']} |")

    print("\nPrior-logit risk versus the following block (Pearson r):")
    print("| risk feature | MSE | cosine loss | top1 mismatches |")
    print("|---|---:|---:|---:|")
    risks = (("entropy_norm", 1.0), ("top1_p", -1.0), ("margin", -1.0))
    for name, sign in risks:
        risk = [sign * block[name] for block in blocks]
        print(f"| {name} | {pearson(risk, [b['mse'] for b in blocks]):+.4f} "
              f"| {pearson(risk, [b['cos_loss'] for b in blocks]):+.4f} "
              f"| {pearson(risk, [float(b['mismatch']) for b in blocks]):+.4f} |")

    print("\nCausal whole-block fallback frontier (highest predicted risk exactified):")
    print("| risk feature | exact blocks | exact output rows | remaining MSE | cosine | top1 mismatches |")
    print("|---|---:|---:|---:|---:|---:|")
    total_rows = sum(len(block["rows"]) for block in blocks)
    for name, sign in risks:
        order = sorted(range(len(blocks)), key=lambda i: sign * blocks[i][name], reverse=True)
        for guarded_count in sorted({0, math.ceil(len(blocks) / 4), math.ceil(len(blocks) / 2)}):
            guarded = set(order[:guarded_count])
            rows = [row for index, block in enumerate(blocks) if index not in guarded
                    for row in block["rows"]]
            exact_rows = total_rows - len(rows)
            mse = math.fsum(row["mse"] for row in rows) / total_rows
            cos_loss = math.fsum(row["cos_loss"] for row in rows) / total_rows
            mismatches = sum(row["mismatch"] for row in rows)
            print(f"| {name} | {guarded_count}/{len(blocks)} | {exact_rows}/{total_rows} "
                  f"| {mse:.4e} | {1.0 - cos_loss:.6f} | {mismatches} |")
    if draft_path:
        draft_logit_guard(draft_path, approx, blocks, vocab)


def residuals(x: list[float], y: list[float]) -> list[float]:
    x_mean, y_mean = mean(x), mean(y)
    variance = math.fsum((value - x_mean) ** 2 for value in x)
    slope = (math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / variance
             if variance else 0.0)
    intercept = y_mean - slope * x_mean
    return [target - (intercept + slope * feature) for feature, target in zip(x, y)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("logits", type=Path)
    parser.add_argument("--vocab", type=int, default=154880)
    parser.add_argument("--verify-k", type=int, default=4)
    parser.add_argument("--mass", type=float, default=0.8)
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--approx-logits", type=Path,
                        help="approximate final-logit dump for causal block fallback analysis")
    parser.add_argument("--draft-logits", type=Path,
                        help="matching DFlash draft-logit dump from the approximate arm")
    args = parser.parse_args()

    groups = read_metrics(args.metrics)
    batches = sorted({batch for batch, _, _ in groups})
    features = logit_features(args.logits, args.vocab,
                              [batch * args.verify_k for batch in batches])
    aggregate: dict[int, dict[str, float]] = {}
    all_omitted, all_rel, all_cos_loss, all_batch = [], [], [], []
    for batch in batches:
        selected = []
        selected_k = []
        for (candidate_batch, _, _), choices in groups.items():
            if candidate_batch != batch:
                continue
            topm = choose(choices, args.mass, args.min_k)
            selected_k.append(topm)
            selected.append(choices[topm])
        omitted = [1.0 - row["retained_mass"] for row in selected]
        rel = [row["rel_l2"] for row in selected]
        cos_loss = [1.0 - row["cosine"] for row in selected]
        all_omitted.extend(omitted)
        all_rel.extend(rel)
        all_cos_loss.extend(cos_loss)
        all_batch.extend([batch] * len(selected))
        aggregate[batch] = {
            "mean_k": mean([float(value) for value in selected_k]),
            "rel_mean": mean(rel),
            "rel_p99": quantile(rel, 0.99),
            "cos_mean": 1.0 - mean(cos_loss),
            "cos_p01": 1.0 - quantile(cos_loss, 0.99),
        }

    rel_residual = residuals(all_omitted, all_rel)
    cos_residual = residuals(all_omitted, all_cos_loss)
    for batch in batches:
        indices = [index for index, value in enumerate(all_batch) if value == batch]
        aggregate[batch]["rel_resid"] = mean([rel_residual[index] for index in indices])
        aggregate[batch]["cos_resid"] = mean([cos_residual[index] for index in indices])

    print(f"mass={args.mass:.3f} min_k={args.min_k}; one observation per causal verify block")
    print("| batch | prior record | entropy/logV | top1 p | margin | mean k | rel-L2 mean | rel-L2 p99 | cosine mean | cosine p01 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for batch in batches:
        feature = features[batch * args.verify_k]
        row = aggregate[batch]
        print(f"| {batch} | {batch * args.verify_k} | {feature['entropy_norm']:.5f} "
              f"| {feature['top1_p']:.5f} | {feature['margin']:.4f} | {row['mean_k']:.3f} "
              f"| {row['rel_mean']:.4f} | {row['rel_p99']:.4f} "
              f"| {row['cos_mean']:.6f} | {row['cos_p01']:.6f} |")

    print("\nBlock-level Pearson r (n is small; residual removes the linear omitted-mass effect):")
    print("| prior feature | mean k | rel-L2 mean | rel-L2 residual | cosine-loss residual |")
    print("|---|---:|---:|---:|---:|")
    targets = {name: [aggregate[batch][name] for batch in batches]
               for name in ("mean_k", "rel_mean", "rel_resid", "cos_resid")}
    for name, sign in (("entropy_norm", 1.0), ("top1_p", -1.0), ("margin", -1.0)):
        uncertainty = [sign * features[batch * args.verify_k][name] for batch in batches]
        print(f"| {name} (uncertainty-signed) "
              f"| {pearson(uncertainty, targets['mean_k']):+.4f} "
              f"| {pearson(uncertainty, targets['rel_mean']):+.4f} "
              f"| {pearson(uncertainty, targets['rel_resid']):+.4f} "
              f"| {pearson(uncertainty, targets['cos_resid']):+.4f} |")

    if args.approx_logits:
        final_logit_guard(args.logits, args.approx_logits, args.draft_logits,
                          args.vocab, args.verify_k)


if __name__ == "__main__":
    main()
