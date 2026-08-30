#!/usr/bin/env python3
"""Test whether previous target logits add causal signal beyond router mass."""

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


def logit_features(path: Path, vocab: int, records: list[int]) -> dict[int, dict[str, float]]:
    raw = np.memmap(path, dtype="<f4", mode="r")
    if raw.size % vocab:
        raise SystemExit(f"{path}: partial logit record")
    matrix = raw.reshape((-1, vocab))
    result = {}
    for record in records:
        if record >= len(matrix):
            raise SystemExit(f"{path}: needs record {record}, has {len(matrix)}")
        logits = matrix[record].astype(np.float64)
        top = np.partition(logits, -2)[-2:]
        maximum, second = float(max(top)), float(min(top))
        shifted = np.exp(logits - maximum)
        normalizer = float(np.sum(shifted))
        probabilities = shifted / normalizer
        entropy = math.log(normalizer) + maximum - float(np.dot(probabilities, logits))
        result[record] = {
            "entropy": entropy,
            "entropy_norm": entropy / math.log(vocab),
            "top1_p": 1.0 / normalizer,
            "margin": maximum - second,
        }
    return result


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


if __name__ == "__main__":
    main()
