#!/usr/bin/env python3
"""Summarize INSIGNIA_GLM53_DF_MOE_METRICS routed-output diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics")
    args = parser.parse_args()

    groups: dict[tuple[int, str], list[dict[str, float | int]]] = defaultdict(list)
    layers: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    with open(args.metrics, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float | int] = {
                "layer": int(raw["layer"]),
                "mse": float(raw["mse"]),
                "rel_l2": float(raw["rel_l2"]),
                "cosine": float(raw["cosine"]),
                "max_abs": float(raw["max_abs"]),
                "norm_ratio": float(raw["norm_ratio"]),
                "retained_mass": float(raw["retained_mass"]),
                "exact_cancel": float(raw["exact_cancel"]),
                "approx_cancel": float(raw["approx_cancel"]),
                "replay_max_abs": float(raw["replay_max_abs"]),
            }
            key = (int(raw["topm"]), raw["semantics"])
            groups[key].append(row)
            layers[(key[0], key[1], int(row["layer"]))].append(float(row["cosine"]))

    if not groups:
        raise SystemExit("no metrics rows")

    print("| top-m | semantics | rows | mass mean | MSE mean | MSE p99 | rel-L2 median | rel-L2 p99 | cosine mean | cosine p01 | cosine min | max-abs p99 | norm ratio median | worst layer (cos p01) |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (topm, semantics), rows in sorted(groups.items()):
        series = lambda name: [float(row[name]) for row in rows]
        layer_scores = [
            (quantile(cosines, 0.01), layer)
            for (candidate_m, candidate_semantics, layer), cosines in layers.items()
            if candidate_m == topm and candidate_semantics == semantics
        ]
        worst_cos, worst_layer = min(layer_scores)
        print(
            f"| {topm} | {semantics} | {len(rows)} | {mean(series('retained_mass')):.4f} "
            f"| {mean(series('mse')):.3e} | {quantile(series('mse'), 0.99):.3e} "
            f"| {quantile(series('rel_l2'), 0.5):.4f} | {quantile(series('rel_l2'), 0.99):.4f} "
            f"| {mean(series('cosine')):.6f} | {quantile(series('cosine'), 0.01):.6f} "
            f"| {min(series('cosine')):.6f} | {quantile(series('max_abs'), 0.99):.4f} "
            f"| {quantile(series('norm_ratio'), 0.5):.4f} | L{worst_layer} ({worst_cos:.6f}) |"
        )

    replay_max = max(float(row["replay_max_abs"]) for rows in groups.values() for row in rows)
    cancellation = [float(row["exact_cancel"]) for rows in groups.values() for row in rows]
    print()
    print(f"Exact top-8 host replay max abs error: {replay_max:.9g}")
    print(f"Exact cancellation ratio: median {quantile(cancellation, 0.5):.4f}, p99 {quantile(cancellation, 0.99):.4f}, max {max(cancellation):.4f}")


if __name__ == "__main__":
    main()
