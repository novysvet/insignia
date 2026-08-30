#!/usr/bin/env python3
"""Prompt-held-out NumPy baselines for the DFlash falsifier corpus.

This is deliberately dependency-free and small enough to run on any CPU.  It
tests whether causal previous-logit features alone, or those features plus the
on-policy router/cache/hidden trajectory, can rank unsafe approximate rows.
The split unit is a complete prompt; adjacent rows are never randomized across
train and test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


LAMBDAS = np.asarray((0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0))
LOGISTIC_L2 = 0.1


def route_sketch(layers: np.ndarray, ids: np.ndarray, width: int = 64) -> np.ndarray:
    result = np.zeros(width, dtype=np.float64)
    for layer, experts in zip(layers, ids, strict=True):
        for expert in experts:
            key = (int(layer) + 1) * 0x9E3779B1 ^ (int(expert) + 1) * 0x85EBCA6B
            key ^= key >> 16
            bucket = key & (width - 1)
            result[bucket] += 1.0 if key & width else -1.0
    return result / math.sqrt(max(1, len(layers) * ids.shape[1]))


def summarize_rows(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    row_scalars = data["row_scalars"].astype(np.float64)
    row_sketch = data["row_logit_sketch"].astype(np.float64).reshape((len(row_scalars), -1))
    logit_features = np.concatenate((row_scalars, row_sketch), axis=1)

    event_row = data["event_row_index"]
    event_meta = data["event_meta"]
    router = data["router_features"].astype(np.float64)
    derived = data["event_derived"].astype(np.float64)
    hidden = data["hidden_countsketch"].astype(np.float64)
    input_norm = data["event_tail"][:, 3].astype(np.float64)
    candidate_choice = data["candidate_choice"].astype(np.float64)
    candidate_ids = data["candidate_ids"].astype(np.int64)
    executed_ids = data["expert_ids"].astype(np.int64)

    runtime_rows = []
    for row in range(len(row_scalars)):
        indices = np.flatnonzero(event_row == row)
        if not len(indices):
            raise SystemExit(f"row {row} has no falsifier events")
        indices = indices[np.argsort(event_meta[indices, 1])]
        layers = event_meta[indices, 1]
        event_router = router[indices]
        event_derived = derived[indices]
        event_hidden = hidden[indices]
        event_norm = input_norm[indices, None]
        choices = candidate_choice[indices]
        gaps = np.stack((choices[:, 0] - choices[:, 7],
                         choices[:, 7] - choices[:, 8],
                         choices[:, 0] - choices[:, 31],
                         np.std(choices, axis=1)), axis=1)
        baseline_ids = candidate_ids[indices, :8]
        actual_ids = executed_ids[indices]
        substitutions = np.asarray([
            8 - len(set(map(int, baseline)) & set(map(int, actual)))
            for baseline, actual in zip(baseline_ids, actual_ids, strict=True)
        ], dtype=np.float64)[:, None] / 8.0
        exec_fraction = event_meta[indices, 5:6].astype(np.float64) / 8.0

        aggregate = []
        for values in (event_router, event_derived, event_hidden, gaps, event_norm,
                       substitutions, exec_fraction):
            aggregate.extend((np.mean(values, axis=0), np.std(values, axis=0),
                              np.min(values, axis=0), np.max(values, axis=0)))

        # Preserve coarse layer position without flattening all 42 events.
        layer_signal = np.concatenate((substitutions,
                                       event_derived[:, (0, 2)],
                                       event_router[:, (13, 14, 15)],
                                       event_norm), axis=1)
        bins = [np.mean(chunk, axis=0) for chunk in np.array_split(layer_signal, 6)]
        aggregate.extend((np.concatenate(bins),
                          route_sketch(layers, baseline_ids),
                          route_sketch(layers, actual_ids)))
        runtime_rows.append(np.concatenate([np.ravel(value) for value in aggregate]))
    runtime = np.asarray(runtime_rows, dtype=np.float64)
    return logit_features, np.concatenate((logit_features, runtime), axis=1)


def load(paths: list[Path]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                                     dict[str, np.ndarray]]:
    logits: dict[str, np.ndarray] = {}
    full: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            prompt = str(metadata["prompt_id"])
            if prompt in labels:
                raise SystemExit(f"duplicate prompt id {prompt}")
            if bool(metadata.get("gram_present", True)):
                raise SystemExit(f"{path}: expected an on-policy feature-only dataset")
            logits[prompt], full[prompt] = summarize_rows(data)
            labels[prompt] = data["row_labels"].astype(np.float64)
    if len(labels) < 3:
        raise SystemExit("prompt-held-out evaluation requires at least three prompts")
    return logits, full, labels


def ridge_fit(x: np.ndarray, y: np.ndarray, regularization: float) -> tuple:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    centered = (x - mean) / scale
    y_mean = float(np.mean(y))
    target = y - y_mean
    # The corpus is wide and short; the dual solve is both faster and more
    # stable than forming a feature-by-feature normal matrix.
    gram = centered @ centered.T
    dual = np.linalg.solve(gram + regularization * np.eye(len(gram)), target)
    return mean, scale, centered.T @ dual, y_mean


def ridge_predict(model: tuple, x: np.ndarray) -> np.ndarray:
    mean, scale, weights, y_mean = model
    return (x - mean) / scale @ weights + y_mean


def logistic_fit(x: np.ndarray, y: np.ndarray, regularization: float = LOGISTIC_L2,
                 steps: int = 500) -> tuple:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    positives = int(np.sum(y > 0.5))
    negatives = len(y) - positives
    if not positives or not negatives:
        probability = (positives + 0.5) / (len(y) + 1.0)
        return mean, scale, np.zeros(x.shape[1]), math.log(probability / (1.0 - probability))
    sample_weight = np.where(y > 0.5, 0.5 / positives, 0.5 / negatives)
    design = np.concatenate((standardized, np.ones((len(x), 1))), axis=1)
    weighted_design = design * np.sqrt(sample_weight)[:, None]
    spectral = float(np.linalg.norm(weighted_design, ord=2))
    learning_rate = 1.0 / (0.25 * spectral * spectral + regularization + 1e-9)
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    accelerated = parameters.copy()
    momentum = 1.0
    for _ in range(steps):
        probability = 1.0 / (1.0 + np.exp(-np.clip(design @ accelerated, -40.0, 40.0)))
        gradient = design.T @ (sample_weight * (probability - y))
        gradient[:-1] += regularization * accelerated[:-1]
        updated = accelerated - learning_rate * gradient
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - parameters)
        parameters = updated
        momentum = next_momentum
    return mean, scale, parameters[:-1], float(parameters[-1])


def logistic_predict(model: tuple, x: np.ndarray) -> np.ndarray:
    mean, scale, weights, bias = model
    score = (x - mean) / scale @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))


def choose_lambda(features: dict[str, np.ndarray], targets: dict[str, np.ndarray],
                  train_prompts: list[str]) -> float:
    losses = np.zeros(len(LAMBDAS), dtype=np.float64)
    for validation in train_prompts:
        inner = [prompt for prompt in train_prompts if prompt != validation]
        train_x = np.concatenate([features[prompt] for prompt in inner])
        train_y = np.concatenate([targets[prompt] for prompt in inner])
        validation_x = features[validation]
        validation_y = targets[validation]
        variance = max(float(np.var(validation_y)), 1e-6)
        for index, regularization in enumerate(LAMBDAS):
            prediction = ridge_predict(ridge_fit(train_x, train_y, regularization),
                                       validation_x)
            losses[index] += float(np.mean((prediction - validation_y) ** 2)) / variance
    return float(LAMBDAS[int(np.argmin(losses))])


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        stop = cursor + 1
        while stop < len(values) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        result[order[cursor:stop]] = 0.5 * (cursor + stop - 1)
        cursor = stop
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = ranks(left), ranks(right)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def risk_targets(labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for prompt, row in labels.items():
        mse, cosine, kl, js, flip = row[:, 0], row[:, 2], row[:, 4], row[:, 5], row[:, 6]
        result[prompt] = (np.log1p(mse / 0.05) +
                          np.log1p(np.maximum(0.0, 1.0 - cosine) / 0.005) +
                          np.log1p(kl / 0.001) + np.log1p(js / 0.0005) + 4.0 * flip)
    return result


def evaluate(features: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> list[dict]:
    targets = risk_targets(labels)
    prompts = sorted(labels)
    rows = []
    for held_out in prompts:
        train = [prompt for prompt in prompts if prompt != held_out]
        regularization = choose_lambda(features, targets, train)
        train_x = np.concatenate([features[prompt] for prompt in train])
        train_y = np.concatenate([targets[prompt] for prompt in train])
        actual = targets[held_out]
        prediction = ridge_predict(ridge_fit(train_x, train_y, regularization),
                                   features[held_out])
        constant = np.full(len(actual), np.mean(train_y))
        denominator = math.sqrt(float(np.mean((constant - actual) ** 2)))
        nrmse = (math.sqrt(float(np.mean((prediction - actual) ** 2))) / denominator
                 if denominator else 0.0)
        count = max(1, math.ceil(0.25 * len(actual)))
        selected = np.argsort(prediction)[-count:]
        risk_capture = float(np.sum(actual[selected]) / np.sum(actual))
        flips = np.flatnonzero(labels[held_out][:, 6] > 0)
        flip_rank = None
        if len(flips):
            descending = np.argsort(prediction)[::-1]
            flip_rank = min(int(np.flatnonzero(descending == flip)[0]) + 1 for flip in flips)
        rows.append({
            "held_out": held_out,
            "lambda": regularization,
            "nrmse_vs_constant": nrmse,
            "spearman": spearman(prediction, actual),
            "top25_risk_capture": risk_capture,
            "flip_rank": flip_rank,
            "rows": len(actual),
        })
    return rows


def binary_auc(probability: np.ndarray, target: np.ndarray) -> float | None:
    positive = probability[target > 0.5]
    negative = probability[target <= 0.5]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0.0) + 0.5 * np.sum(comparisons == 0.0)) /
                 comparisons.size)


def average_precision(probability: np.ndarray, target: np.ndarray) -> float | None:
    positives = int(np.sum(target > 0.5))
    if not positives:
        return None
    order = np.argsort(probability)[::-1]
    hits = np.cumsum(target[order] > 0.5)
    positive_ranks = np.flatnonzero(target[order] > 0.5)
    return float(np.mean(hits[positive_ranks] / (positive_ranks + 1)))


def evaluate_flips(features: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> list[dict]:
    prompts = sorted(labels)
    rows = []
    for held_out in prompts:
        train = [prompt for prompt in prompts if prompt != held_out]
        train_x = np.concatenate([features[prompt] for prompt in train])
        train_y = np.concatenate([labels[prompt][:, 6] for prompt in train])
        target = labels[held_out][:, 6]
        probability = logistic_predict(logistic_fit(train_x, train_y), features[held_out])
        count = max(1, math.ceil(0.25 * len(target)))
        selected = np.argsort(probability)[-count:]
        positives = int(np.sum(target > 0.5))
        captured = int(np.sum(target[selected] > 0.5))
        descending = np.argsort(probability)[::-1]
        flip_ranks = [int(np.flatnonzero(descending == index)[0]) + 1
                      for index in np.flatnonzero(target > 0.5)]
        rows.append({
            "held_out": held_out,
            "auc": binary_auc(probability, target),
            "average_precision": average_precision(probability, target),
            "top25_flip_recall": (captured / positives if positives else None),
            "best_flip_rank": min(flip_ranks) if flip_ranks else None,
            "worst_flip_rank": max(flip_ranks) if flip_ranks else None,
            "rows": len(target),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    logit_features, full_features, labels = load(args.datasets)
    result = {
        "schema": "insignia-falsifier-baseline-v1",
        "prompts": sorted(labels),
        "logit_features": evaluate(logit_features, labels),
        "full_runtime_features": evaluate(full_features, labels),
        "logit_flip_classifier": evaluate_flips(logit_features, labels),
        "full_runtime_flip_classifier": evaluate_flips(full_features, labels),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print("| features | held out | lambda | NRMSE/constant | Spearman | top-25% risk captured | flip rank |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for feature_name in ("logit_features", "full_runtime_features"):
        for row in result[feature_name]:
            flip_rank = "-" if row["flip_rank"] is None else str(row["flip_rank"])
            print(f"| {feature_name} | {row['held_out']} | {row['lambda']:.2g} | "
                  f"{row['nrmse_vs_constant']:.3f} | {row['spearman']:.3f} | "
                  f"{row['top25_risk_capture']:.1%} | {flip_rank}/{row['rows']} |")
    print()
    print("| flip classifier | held out | AUROC | AP | top-25% flip recall | flip rank(s) |")
    print("|---|---|---:|---:|---:|---:|")
    for feature_name in ("logit_flip_classifier", "full_runtime_flip_classifier"):
        for row in result[feature_name]:
            auc = "-" if row["auc"] is None else f"{row['auc']:.3f}"
            ap = "-" if row["average_precision"] is None else f"{row['average_precision']:.3f}"
            recall = ("-" if row["top25_flip_recall"] is None
                      else f"{row['top25_flip_recall']:.1%}")
            ranks_text = ("-" if row["best_flip_rank"] is None else
                          f"{row['best_flip_rank']}-{row['worst_flip_rank']}/{row['rows']}")
            print(f"| {feature_name} | {row['held_out']} | {auc} | {ap} | "
                  f"{recall} | {ranks_text} |")


if __name__ == "__main__":
    main()
