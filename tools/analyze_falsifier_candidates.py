#!/usr/bin/env python3
"""Measure the immediate cache-aware near-tie ceiling in a falsifier dataset.

This is an offline baseline, not a quality oracle. It selects resident top-32
candidates under corrected-router-score regret and reports both independent-row
and actual layer-union transfer counts. Alternative-expert MSE/cos/KL still
requires on-policy teacher interventions.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Choice:
    experts: tuple[int, ...]
    regret: float
    regret_ratio: float
    substitutions: int
    disk: int
    h2d: int


def resident(mask: np.ndarray, rank: int) -> tuple[bool, bool, bool]:
    host = bool((int(mask[0]) >> rank) & 1)
    inflight = bool((int(mask[1]) >> rank) & 1)
    device = bool((int(mask[2]) >> rank) & 1)
    return host, inflight, device


def transfer(mask: np.ndarray, rank: int) -> tuple[int, int]:
    host, inflight, device = resident(mask, rank)
    if device:
        return 0, 0
    return (0 if host or inflight else 1), 1


def select_event(baseline: np.ndarray, candidates: np.ndarray, scores: np.ndarray,
                 residency: np.ndarray, candidate_k: int, retain: int,
                 epsilon: float) -> Choice:
    candidate_ids = list(map(int, candidates[:candidate_k]))
    rank = {expert: index for index, expert in enumerate(candidate_ids)}
    baseline_ids = tuple(map(int, baseline))
    missing = [expert for expert in baseline_ids if expert not in rank]
    if missing:
        raise ValueError(f"baseline experts missing from top-{candidate_k}: {missing}")
    score = {expert: float(scores[index]) for expert, index in rank.items()}
    baseline_score = math.fsum(score[expert] for expert in baseline_ids)
    scale = math.fsum(abs(score[expert]) for expert in baseline_ids)
    fixed = baseline_ids[:retain]
    pool = [expert for expert in candidate_ids if expert not in fixed]
    need = len(baseline_ids) - retain
    choices: list[tuple[tuple, Choice]] = []
    for tail in itertools.combinations(pool, need):
        selected = fixed + tail
        selected_score = math.fsum(score[expert] for expert in selected)
        regret = max(0.0, baseline_score - selected_score)
        ratio = regret / scale if scale else 0.0
        if ratio > epsilon + 2e-7:
            continue
        disk = h2d = 0
        for expert in selected:
            d, h = transfer(residency, rank[expert])
            disk += d
            h2d += h
        substitutions = len(set(selected) - set(baseline_ids))
        ordered = tuple(sorted(selected, key=rank.__getitem__))
        key = (disk, h2d, regret, substitutions, ordered)
        choices.append((key, Choice(ordered, regret, ratio, substitutions, disk, h2d)))
    if not choices:
        raise ValueError("baseline unexpectedly infeasible")
    choices.sort(key=lambda item: item[0])
    return choices[0][1]


def event_choices(baseline: np.ndarray, candidates: np.ndarray, scores: np.ndarray,
                  residency: np.ndarray, candidate_k: int, retain: int,
                  epsilon: float, limit: int) -> list[Choice]:
    """Return locally strong actions while retaining the exact baseline."""
    candidate_ids = list(map(int, candidates[:candidate_k]))
    rank = {expert: index for index, expert in enumerate(candidate_ids)}
    baseline_ids = tuple(map(int, baseline))
    score = {expert: float(scores[index]) for expert, index in rank.items()}
    baseline_score = math.fsum(score[expert] for expert in baseline_ids)
    scale = math.fsum(abs(score[expert]) for expert in baseline_ids)
    fixed = baseline_ids[:retain]
    pool = [expert for expert in candidate_ids if expert not in fixed]
    ranked: list[tuple[tuple, Choice]] = []
    for tail in itertools.combinations(pool, len(baseline_ids) - retain):
        selected = fixed + tail
        selected_score = math.fsum(score[expert] for expert in selected)
        regret = max(0.0, baseline_score - selected_score)
        ratio = regret / scale if scale else 0.0
        if ratio > epsilon + 2e-7:
            continue
        disk = h2d = 0
        for expert in selected:
            d, h = transfer(residency, rank[expert])
            disk += d
            h2d += h
        substitutions = len(set(selected) - set(baseline_ids))
        ordered = tuple(sorted(selected, key=rank.__getitem__))
        choice = Choice(ordered, regret, ratio, substitutions, disk, h2d)
        ranked.append(((disk, h2d, regret, substitutions, ordered), choice))
    if not ranked:
        raise ValueError("baseline unexpectedly infeasible")
    ranked.sort(key=lambda item: item[0])
    selected = [choice for _, choice in ranked[:limit]]
    baseline_choice = next(choice for _, choice in ranked
                           if choice.experts == baseline_ids)
    if baseline_choice not in selected:
        selected[-1] = baseline_choice
    return selected


def joint_group(indices: list[int], expert_ids: np.ndarray, candidate_ids: np.ndarray,
                candidate_choice: np.ndarray, residency: np.ndarray,
                candidate_k: int, retain: int, epsilon: float,
                option_limit: int) -> dict[int, Choice]:
    options = [event_choices(expert_ids[index], candidate_ids[index],
                             candidate_choice[index], residency[index],
                             candidate_k, retain, epsilon, option_limit)
               for index in indices]
    best_key: tuple | None = None
    best_assignment: tuple[Choice, ...] | None = None
    for assignment in itertools.product(*options):
        mapping = dict(zip(indices, assignment, strict=True))
        union, disk, h2d = union_cost(indices, mapping, candidate_ids, residency)
        key = (disk, h2d, union,
               math.fsum(choice.regret for choice in assignment),
               sum(choice.substitutions for choice in assignment))
        if best_key is None or key < best_key:
            best_key = key
            best_assignment = assignment
    assert best_assignment is not None
    return dict(zip(indices, best_assignment, strict=True))


def union_cost(indices: list[int], selections: dict[int, Choice],
               candidate_ids: np.ndarray, residency: np.ndarray) -> tuple[int, int, int]:
    union: set[int] = set()
    rank_by_event: dict[int, dict[int, int]] = {}
    for index in indices:
        rank_by_event[index] = {int(expert): rank
                                for rank, expert in enumerate(candidate_ids[index])}
        union.update(selections[index].experts)
    disk = h2d = 0
    for expert in union:
        source = next(index for index in indices if expert in rank_by_event[index])
        d, h = transfer(residency[source], rank_by_event[source][expert])
        disk += d
        h2d += h
    return len(union), disk, h2d


def union_ids(indices: list[int], selected_ids: np.ndarray,
              candidate_ids: np.ndarray, residency: np.ndarray) -> tuple[int, int, int]:
    union = {int(expert) for index in indices for expert in selected_ids[index]}
    disk = h2d = 0
    for expert in union:
        source = next(index for index in indices
                      if expert in set(map(int, candidate_ids[index])))
        rank = int(np.flatnonzero(candidate_ids[source] == expert)[0])
        d, h = transfer(residency[source], rank)
        disk += d
        h2d += h
    return len(union), disk, h2d


def baseline_choices(expert_ids: np.ndarray, candidate_ids: np.ndarray,
                     residency: np.ndarray) -> dict[int, Choice]:
    result = {}
    for index, baseline in enumerate(expert_ids):
        rank = {int(expert): candidate_rank
                for candidate_rank, expert in enumerate(candidate_ids[index])}
        disk = h2d = 0
        for expert in map(int, baseline):
            d, h = transfer(residency[index], rank[expert])
            disk += d
            h2d += h
        result[index] = Choice(tuple(map(int, baseline)), 0.0, 0.0, 0, disk, h2d)
    return result


def analyze(path: Path, candidate_ks: list[int], retains: list[int],
            epsilons: list[float], joint_options: int = 0) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        event_meta = data["event_meta"]
        # Exact traces store the baseline in expert_ids. Feature-only traces
        # store the executed action, while their untouched baseline is the
        # first eight corrected-score candidates.
        executed_ids = data["expert_ids"]
        feature_only = not metadata.get("gram_present", True)
        expert_ids = data["candidate_ids"][:, :8] if feature_only else executed_ids
        candidate_ids = data["candidate_ids"]
        candidate_choice = data["candidate_choice"]
        residency = data["candidate_residency"]
    if np.any(np.diff(candidate_choice, axis=1) > 1e-6):
        raise SystemExit("candidate corrected scores are not monotone")
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, meta in enumerate(event_meta):
        groups[(int(meta[0]), int(meta[1]))].append(index)
    baseline = baseline_choices(expert_ids, candidate_ids, residency)
    baseline_union = [union_cost(indices, baseline, candidate_ids, residency)
                      for indices in groups.values()]
    baseline_records = {
        "union": sum(row[0] for row in baseline_union),
        "disk": sum(row[1] for row in baseline_union),
        "h2d": sum(row[2] for row in baseline_union),
    }
    observed_policy = None
    if feature_only:
        observed_union = [union_ids(indices, executed_ids, candidate_ids, residency)
                          for indices in groups.values()]
        observed_records = {
            "union": sum(row[0] for row in observed_union),
            "disk": sum(row[1] for row in observed_union),
            "h2d": sum(row[2] for row in observed_union),
        }
        observed_policy = {
            "records": observed_records,
            "union_saved_fraction":
                ((baseline_records["union"] - observed_records["union"]) /
                 baseline_records["union"]),
            "disk_saved_fraction":
                ((baseline_records["disk"] - observed_records["disk"]) /
                 baseline_records["disk"] if baseline_records["disk"] else 0.0),
            "h2d_saved_fraction":
                ((baseline_records["h2d"] - observed_records["h2d"]) /
                 baseline_records["h2d"] if baseline_records["h2d"] else 0.0),
        }
    points = []
    for candidate_k, retain, epsilon in itertools.product(candidate_ks, retains, epsilons):
        choices = {
            index: select_event(expert_ids[index], candidate_ids[index],
                                candidate_choice[index], residency[index],
                                candidate_k, retain, epsilon)
            for index in range(len(event_meta))
        }
        selected_union = [union_cost(indices, choices, candidate_ids, residency)
                          for indices in groups.values()]
        records = {
            "union": sum(row[0] for row in selected_union),
            "disk": sum(row[1] for row in selected_union),
            "h2d": sum(row[2] for row in selected_union),
        }
        substitutions = sum(choice.substitutions for choice in choices.values())
        changed = sum(choice.substitutions > 0 for choice in choices.values())
        regrets = [choice.regret_ratio for choice in choices.values()]
        point = {
            "candidate_k": candidate_k,
            "retain": retain,
            "epsilon": epsilon,
            "events": len(event_meta),
            "layer_groups": len(groups),
            "changed_events": changed,
            "substitutions": substitutions,
            "substitutions_per_layer": substitutions / len(groups),
            "mean_regret_ratio": math.fsum(regrets) / len(regrets),
            "max_regret_ratio": max(regrets),
            "union_records": records["union"],
            "disk_records": records["disk"],
            "h2d_records": records["h2d"],
            "union_saved_fraction": ((baseline_records["union"] - records["union"]) /
                                     baseline_records["union"]),
            "disk_saved_fraction": ((baseline_records["disk"] - records["disk"]) /
                                    baseline_records["disk"]
                                    if baseline_records["disk"] else 0.0),
            "h2d_saved_fraction": ((baseline_records["h2d"] - records["h2d"]) /
                                   baseline_records["h2d"]
                                   if baseline_records["h2d"] else 0.0),
        }
        if joint_options:
            joint: dict[int, Choice] = {}
            for indices in groups.values():
                joint.update(joint_group(indices, expert_ids, candidate_ids,
                                         candidate_choice, residency, candidate_k,
                                         retain, epsilon, joint_options))
            joint_union = [union_cost(indices, joint, candidate_ids, residency)
                           for indices in groups.values()]
            joint_records = {
                "union": sum(row[0] for row in joint_union),
                "disk": sum(row[1] for row in joint_union),
                "h2d": sum(row[2] for row in joint_union),
            }
            point.update({
                "joint_options": joint_options,
                "joint_union_records": joint_records["union"],
                "joint_disk_records": joint_records["disk"],
                "joint_h2d_records": joint_records["h2d"],
                "joint_union_saved_fraction":
                    ((baseline_records["union"] - joint_records["union"]) /
                     baseline_records["union"]),
                "joint_disk_saved_fraction":
                    ((baseline_records["disk"] - joint_records["disk"]) /
                     baseline_records["disk"] if baseline_records["disk"] else 0.0),
                "joint_h2d_saved_fraction":
                    ((baseline_records["h2d"] - joint_records["h2d"]) /
                     baseline_records["h2d"] if baseline_records["h2d"] else 0.0),
            })
        points.append(point)
    return {"dataset": str(path), "metadata": metadata,
            "baseline_records": baseline_records, "observed_policy": observed_policy,
            "points": points}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--candidate-k", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--retain", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--epsilon", type=float, nargs="+",
                        default=[0.001, 0.0025, 0.005, 0.01])
    parser.add_argument("--joint-options", type=int, default=0,
                        help="jointly optimize each layer group over this many actions/row")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.joint_options and args.joint_options < 2:
        parser.error("--joint-options must be 0 or at least 2")
    result = analyze(args.dataset, args.candidate_k, args.retain, args.epsilon,
                     args.joint_options)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    if result["observed_policy"] is not None:
        observed = result["observed_policy"]
        print("observed policy: "
              f"disk {observed['disk_saved_fraction']:.2%}, "
              f"H2D {observed['h2d_saved_fraction']:.2%}, "
              f"union {observed['union_saved_fraction']:.2%} saved")
    print("| K | retain | regret budget | substitutions/layer | disk saved | H2D saved | union saved | joint disk | joint H2D | joint union | max regret |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for point in result["points"]:
        print(f"| {point['candidate_k']} | {point['retain']} | {point['epsilon']:.4f} "
              f"| {point['substitutions_per_layer']:.3f} "
              f"| {point['disk_saved_fraction']:.2%} | {point['h2d_saved_fraction']:.2%} "
              f"| {point['union_saved_fraction']:.2%} "
              f"| {point.get('joint_disk_saved_fraction', 0.0):.2%} "
              f"| {point.get('joint_h2d_saved_fraction', 0.0):.2%} "
              f"| {point.get('joint_union_saved_fraction', 0.0):.2%} "
              f"| {point['max_regret_ratio']:.5f} |")


if __name__ == "__main__":
    main()
