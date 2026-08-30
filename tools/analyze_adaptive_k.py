#!/usr/bin/env python3
"""Replay causal top-p expert selection over exact MoE contribution metrics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


METRICS = ("mse", "rel_l2", "cosine", "max_abs", "norm_ratio", "retained_mass")


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
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw["semantics"] != "zero":
                continue
            key = (int(raw["epoch"]), int(raw["layer"]), int(raw["row"]))
            topm = int(raw["topm"])
            groups[key][topm] = {name: float(raw[name]) for name in METRICS}

    exact = {"mse": 0.0, "rel_l2": 0.0, "cosine": 1.0,
             "max_abs": 0.0, "norm_ratio": 1.0, "retained_mass": 1.0}
    for key, choices in groups.items():
        if set(choices) != set(range(1, 8)):
            raise SystemExit(f"{path}: incomplete top-m frontier at {key}: {sorted(choices)}")
        choices[8] = exact
    if not groups:
        raise SystemExit(f"{path}: no zero-semantics rows")
    return groups


def choose(choices: dict[int, dict[str, float]], threshold: float, min_k: int, max_k: int) -> int:
    for topm in range(min_k, max_k + 1):
        if choices[topm]["retained_mass"] >= threshold:
            return topm
    return max_k


def summarize(groups: dict[tuple[int, int, int], dict[int, dict[str, float]]],
              threshold: float, min_k: int, max_k: int) -> tuple[list[int], list[dict[str, float]]]:
    selected_k, selected_rows = [], []
    for choices in groups.values():
        topm = choose(choices, threshold, min_k, max_k)
        selected_k.append(topm)
        selected_rows.append(choices[topm])
    return selected_k, selected_rows


def format_histogram(values: list[int]) -> str:
    counts = Counter(values)
    return " ".join(f"k{k}:{100.0 * counts[k] / len(values):.1f}%" for k in sorted(counts))


def report(label: str, groups: dict[tuple[int, int, int], dict[int, dict[str, float]]],
           thresholds: list[float], min_k: int, max_k: int, profile_threshold: float) -> None:
    print(f"## {label}\n")
    print(f"Rows: {len(groups)}; allowed k: {min_k}..{max_k}; retained weights are not renormalized.\n")
    print("| mass p | mean k | slots vs k8 | MSE mean | MSE p99 | rel-L2 median | rel-L2 p99 | cosine mean | cosine p01 | cosine min | k distribution |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for threshold in thresholds:
        selected_k, rows = summarize(groups, threshold, min_k, max_k)
        series = lambda name: [row[name] for row in rows]
        print(
            f"| {threshold:.3f} | {mean([float(k) for k in selected_k]):.3f} "
            f"| {mean([float(k) for k in selected_k]) / 8.0:.1%} "
            f"| {mean(series('mse')):.3e} | {quantile(series('mse'), 0.99):.3e} "
            f"| {quantile(series('rel_l2'), 0.5):.4f} | {quantile(series('rel_l2'), 0.99):.4f} "
            f"| {mean(series('cosine')):.6f} | {quantile(series('cosine'), 0.01):.6f} "
            f"| {min(series('cosine')):.6f} | {format_histogram(selected_k)} |"
        )

    print("\nRouter-mass signal quality at fixed k (Pearson r):\n")
    print("| k | omitted mass vs rel-L2 | omitted mass vs cosine loss |")
    print("|---:|---:|---:|")
    for topm in range(1, 8):
        rows = [choices[topm] for choices in groups.values()]
        omitted = [1.0 - row["retained_mass"] for row in rows]
        rel_l2 = [row["rel_l2"] for row in rows]
        cosine_loss = [1.0 - row["cosine"] for row in rows]
        print(f"| {topm} | {pearson(omitted, rel_l2):+.4f} | {pearson(omitted, cosine_loss):+.4f} |")

    by_layer: dict[int, list[int]] = defaultdict(list)
    for (_, layer, _), choices in groups.items():
        by_layer[layer].append(choose(choices, profile_threshold, min_k, max_k))
    print(f"\nLayer profile at mass p={profile_threshold:.3f}:\n")
    print("| layer | mean k |")
    print("|---:|---:|")
    for layer, values in sorted(by_layer.items()):
        print(f"| {layer} | {mean([float(value) for value in values]):.3f} |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--thresholds", default=".65,.70,.75,.80,.825,.85,.875,.90,.925,.95")
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--profile-threshold", type=float, default=0.85)
    args = parser.parse_args()
    if not 1 <= args.min_k <= args.max_k <= 8:
        raise SystemExit("require 1 <= min-k <= max-k <= 8")
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if any(not 0.0 < value <= 1.0 for value in thresholds + [args.profile_threshold]):
        raise SystemExit("thresholds must be in (0, 1]")

    for path in args.metrics:
        report(path.parent.name, read_metrics(path), thresholds,
               args.min_k, args.max_k, args.profile_threshold)


if __name__ == "__main__":
    main()
