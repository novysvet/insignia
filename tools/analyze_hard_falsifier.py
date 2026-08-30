#!/usr/bin/env python3
"""Rank causal falsifier signals on a hard-prompt teacher replay.

The input NPZ is produced by build_falsifier_dataset.py.  Raw approximate
target and DFlash logits add temporal-distribution features that the original
three-prompt baseline did not contain.  Every ranked feature is available
before the corresponding approximate expert execution.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def probabilities(raw: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = raw.astype(np.float64)
    maximum = float(np.max(values))
    shifted = np.exp(values - maximum)
    total = float(np.sum(shifted))
    probability = shifted / total
    top = np.argpartition(values, -32)[-32:]
    top = top[np.argsort(values[top])[::-1]]
    entropy = math.log(total) + maximum - float(np.dot(probability, values))
    return probability, {
        "entropy_norm": entropy / math.log(len(values)),
        "top1_p": 1.0 / total,
        "margin": float(values[top[0]] - values[top[1]]),
        "top1": float(top[0]),
        "top8": top[:8],
        "top32": top,
    }


def js(left: np.ndarray, right: np.ndarray) -> float:
    mixture = 0.5 * (left + right)
    left_mask = left > 0
    right_mask = right > 0
    return 0.5 * (
        float(np.dot(left[left_mask], np.log(left[left_mask] / mixture[left_mask]))) +
        float(np.dot(right[right_mask], np.log(right[right_mask] / mixture[right_mask])))
    )


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    a -= float(np.mean(a))
    b -= float(np.mean(b))
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator else 1.0


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    return len(set(map(int, left)) & set(map(int, right))) / len(left)


def point_biserial(values: np.ndarray, labels: np.ndarray) -> float:
    if not np.any(labels) or np.all(labels):
        return 0.0
    centered = values - float(np.mean(values))
    binary = labels.astype(np.float64) - float(np.mean(labels))
    denominator = math.sqrt(float(np.dot(centered, centered)) *
                            float(np.dot(binary, binary)))
    return float(np.dot(centered, binary) / denominator) if denominator else 0.0


def auc(values: np.ndarray, labels: np.ndarray) -> float:
    positive = values[labels]
    negative = values[~labels]
    if not len(positive) or not len(negative):
        return 0.5
    wins = 0.0
    for value in positive:
        wins += float(np.count_nonzero(value > negative))
        wins += 0.5 * float(np.count_nonzero(value == negative))
    return wins / (len(positive) * len(negative))


def add_pair_features(target: dict[str, np.ndarray], prefix: str,
                      left_raw: np.ndarray, left_prob: np.ndarray,
                      left_stats: dict[str, float], right_raw: np.ndarray,
                      right_prob: np.ndarray, right_stats: dict[str, float],
                      index: int) -> None:
    target[f"{prefix}_js"][index] = js(left_prob, right_prob)
    target[f"{prefix}_cos_loss"][index] = 1.0 - centered_cosine(left_raw, right_raw)
    target[f"{prefix}_top1_disagree"][index] = left_stats["top1"] != right_stats["top1"]
    target[f"{prefix}_top8_churn"][index] = 1.0 - overlap(
        left_stats["top8"], right_stats["top8"])
    target[f"{prefix}_top32_churn"][index] = 1.0 - overlap(
        left_stats["top32"], right_stats["top32"])


def raw_features(row_meta: np.ndarray, approximate_path: Path, draft_path: Path,
                 vocab: int, draft_rows: int, verify_k: int) -> dict[str, np.ndarray]:
    approximate = np.memmap(approximate_path, dtype="<f4", mode="r")
    draft = np.memmap(draft_path, dtype="<f4", mode="r")
    if approximate.size % vocab or draft.size % (draft_rows * vocab):
        raise SystemExit("partial raw logit dump")
    approximate = approximate.reshape((-1, vocab))
    draft = draft.reshape((-1, draft_rows, vocab))
    count = len(row_meta)
    pair_prefixes = ("prior_previous", "prior_current_draft",
                     "draft_adjacent", "draft_from_row0")
    result = {
        f"{prefix}_{suffix}": np.zeros(count, dtype=np.float64)
        for prefix in pair_prefixes
        for suffix in ("js", "cos_loss", "top1_disagree", "top8_churn", "top32_churn")
    }
    for name in ("prior_entropy_delta", "prior_top1_p_drop", "prior_margin_drop",
                 "draft_entropy_delta", "draft_top1_p_drop", "draft_margin_drop",
                 "calibration_js_delta"):
        result[name] = np.zeros(count, dtype=np.float64)

    by_block: dict[int, list[int]] = defaultdict(list)
    for index, meta in enumerate(row_meta):
        by_block[int(meta[1])].append(index)
    previous_calibration = 0.0
    for block in sorted(by_block):
        indices = sorted(by_block[block], key=lambda index: int(row_meta[index, 2]))
        start = block * verify_k
        prior_raw = approximate[start]
        prior_probability, prior_stats = probabilities(prior_raw)
        previous_start = max(0, start - verify_k)
        previous_raw = approximate[previous_start]
        previous_probability, previous_stats = probabilities(previous_raw)
        draft_probabilities = []
        draft_stats = []
        for draft_row in range(draft_rows):
            probability, stats = probabilities(draft[block, draft_row])
            draft_probabilities.append(probability)
            draft_stats.append(stats)
        calibration = js(prior_probability, draft_probabilities[0])
        for index in indices:
            row = int(row_meta[index, 2])
            current = min(row + 1, draft_rows - 1)
            adjacent = max(0, current - 1)
            add_pair_features(result, "prior_previous", prior_raw, prior_probability,
                              prior_stats, previous_raw, previous_probability,
                              previous_stats, index)
            add_pair_features(result, "prior_current_draft", prior_raw, prior_probability,
                              prior_stats, draft[block, current],
                              draft_probabilities[current], draft_stats[current], index)
            add_pair_features(result, "draft_adjacent", draft[block, adjacent],
                              draft_probabilities[adjacent], draft_stats[adjacent],
                              draft[block, current], draft_probabilities[current],
                              draft_stats[current], index)
            add_pair_features(result, "draft_from_row0", draft[block, 0],
                              draft_probabilities[0], draft_stats[0],
                              draft[block, current], draft_probabilities[current],
                              draft_stats[current], index)
            result["prior_entropy_delta"][index] = (
                prior_stats["entropy_norm"] - previous_stats["entropy_norm"])
            result["prior_top1_p_drop"][index] = (
                previous_stats["top1_p"] - prior_stats["top1_p"])
            result["prior_margin_drop"][index] = (
                previous_stats["margin"] - prior_stats["margin"])
            result["draft_entropy_delta"][index] = (
                draft_stats[current]["entropy_norm"] - draft_stats[adjacent]["entropy_norm"])
            result["draft_top1_p_drop"][index] = (
                draft_stats[adjacent]["top1_p"] - draft_stats[current]["top1_p"])
            result["draft_margin_drop"][index] = (
                draft_stats[adjacent]["margin"] - draft_stats[current]["margin"])
            result["calibration_js_delta"][index] = calibration - previous_calibration
        previous_calibration = calibration
    return result


def layer_association(event_meta: np.ndarray, event_rows: np.ndarray,
                      executed: np.ndarray, candidates: np.ndarray,
                      choice: np.ndarray, labels: np.ndarray) -> list[tuple]:
    rows = []
    for layer in sorted(set(map(int, event_meta[:, 1]))):
        indices = np.flatnonzero(event_meta[:, 1] == layer)
        changed = []
        regret = []
        risky = []
        for index in indices:
            row = int(event_rows[index])
            if row < 0:
                continue
            exec_k = int(event_meta[index, 5])
            baseline = list(map(int, candidates[index, :exec_k]))
            selected = list(map(int, executed[index, :exec_k]))
            rank = {int(expert): candidate_rank
                    for candidate_rank, expert in enumerate(candidates[index])}
            baseline_score = math.fsum(float(choice[index, rank[expert]])
                                       for expert in baseline)
            selected_score = math.fsum(float(choice[index, rank[expert]])
                                       for expert in selected)
            scale = math.fsum(abs(float(choice[index, candidate_rank]))
                              for candidate_rank in range(8))
            changed.append(float(selected != baseline))
            regret.append(max(0.0, baseline_score - selected_score) / scale if scale else 0.0)
            risky.append(bool(labels[row]))
        changed_array = np.asarray(changed)
        regret_array = np.asarray(regret)
        risky_array = np.asarray(risky)
        changed_gap = (float(np.mean(changed_array[risky_array])) -
                       float(np.mean(changed_array[~risky_array])))
        regret_gap = (float(np.mean(regret_array[risky_array])) -
                      float(np.mean(regret_array[~risky_array])))
        rows.append((layer, changed_gap, regret_gap,
                     float(np.mean(changed_array[risky_array])),
                     float(np.mean(changed_array[~risky_array]))))
    return sorted(rows, key=lambda row: (abs(row[1]), abs(row[2])), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--approx-logits", type=Path, required=True)
    parser.add_argument("--draft-logits", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        row_meta = data["row_meta"]
        row_scalars = data["row_scalars"]
        labels = data["row_labels"][:, 6].astype(bool)
        event_meta = data["event_meta"]
        event_rows = data["event_row_index"]
        executed = data["expert_ids"]
        candidates = data["candidate_ids"]
        choice = data["candidate_choice"]
    names = metadata["row_scalar_names"]
    features = {name: row_scalars[:, index].astype(np.float64)
                for index, name in enumerate(names)}
    features.update(raw_features(
        row_meta, args.approx_logits, args.draft_logits,
        int(metadata["vocab"]), int(metadata["draft_rows"]),
        int(metadata["verify_k"])))

    ranking = []
    for name, values in features.items():
        correlation = point_biserial(values, labels)
        sign = 1.0 if correlation >= 0.0 else -1.0
        risk = sign * values
        order = np.argsort(risk)[::-1]
        quarter = max(1, math.ceil(len(order) / 4))
        captured = int(np.count_nonzero(labels[order[:quarter]]))
        ranking.append((abs(correlation), max(auc(risk, labels), 1.0 - auc(risk, labels)),
                        name, correlation, sign, captured,
                        float(np.min(values[labels])), float(np.max(values[labels]))))
    ranking.sort(reverse=True)
    total_flips = int(np.count_nonzero(labels))
    print(f"rows={len(labels)} top1_mismatches={total_flips}")
    print("| feature | point-biserial r | AUROC | flips in top 25% | flip range | risk direction |")
    print("|---|---:|---:|---:|---:|---|")
    for _, area, name, correlation, sign, captured, minimum, maximum in ranking[:args.top]:
        print(f"| {name} | {correlation:+.4f} | {area:.4f} | "
              f"{captured}/{total_flips} | [{minimum:.5g}, {maximum:.5g}] | "
              f"{'high' if sign > 0 else 'low'} |")

    row_exec_k = np.full(len(labels), -1, dtype=np.int32)
    for event_index, row_index in enumerate(event_rows):
        if row_index >= 0:
            value = int(event_meta[event_index, 5])
            if row_exec_k[row_index] >= 0 and row_exec_k[row_index] != value:
                raise SystemExit("exec_k differs by layer for one verifier row")
            row_exec_k[row_index] = value
    print("\nResidual top-1 mismatch locations:")
    print("| output record | block | row | exec k | draft p | p drop | entropy delta | margin |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for index in np.flatnonzero(labels):
        print(f"| {int(row_meta[index, 3])} | {int(row_meta[index, 1])} "
              f"| {int(row_meta[index, 2])} | {row_exec_k[index]} "
              f"| {features['draft_top1_p'][index]:.6f} "
              f"| {features['draft_top1_p_drop'][index]:+.6f} "
              f"| {features['draft_entropy_delta'][index]:+.6f} "
              f"| {features['draft_margin'][index]:.6f} |")

    print("\nCausal whole-block guard frontier (direction fitted only for this diagnostic):")
    print("| feature | exact blocks | exact rows | flips captured | approximate rows kept |")
    print("|---|---:|---:|---:|---:|")
    blocks = sorted(set(map(int, row_meta[:, 1])))
    by_block = {block: np.flatnonzero(row_meta[:, 1] == block) for block in blocks}
    for _, _, name, _, sign, _, _, _ in ranking[:8]:
        score = {block: float(np.max(sign * features[name][indices]))
                 for block, indices in by_block.items()}
        order = sorted(blocks, key=score.__getitem__, reverse=True)
        for fraction in (0.125, 0.25):
            exact_count = max(1, math.ceil(len(blocks) * fraction))
            exact = set(order[:exact_count])
            exact_rows = np.concatenate([by_block[block] for block in exact])
            captured = int(np.count_nonzero(labels[exact_rows]))
            print(f"| {name} | {exact_count}/{len(blocks)} | {len(exact_rows)}/{len(labels)} "
                  f"| {captured}/{total_flips} | {len(labels) - len(exact_rows)}/{len(labels)} |")

    print("\nLayer association of cache-tail substitutions with failing rows:")
    print("| layer | changed-rate gap | regret gap | changed on flips | changed on safe |")
    print("|---:|---:|---:|---:|---:|")
    for layer, changed_gap, regret_gap, risky_rate, safe_rate in layer_association(
            event_meta, event_rows, executed, candidates, choice, labels)[:12]:
        print(f"| {layer} | {changed_gap:+.3f} | {regret_gap:+.6f} "
              f"| {risky_rate:.3f} | {safe_rate:.3f} |")


if __name__ == "__main__":
    main()
