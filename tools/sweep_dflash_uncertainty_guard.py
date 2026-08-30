#!/usr/bin/env python3
"""Sweep prefix-closed DFlash uncertainty guards across prompt-held-out cases."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyze_hard_falsifier import raw_features


@dataclass
class Case:
    name: str
    row_meta: np.ndarray
    labels: np.ndarray
    features: dict[str, np.ndarray]


@dataclass(frozen=True)
class Clause:
    feature: str
    direction: str
    threshold: float


def parse_case(specification: str) -> tuple[str, Path, Path, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("case must be NAME=DATASET,APPROX_LOGITS,DRAFT_LOGITS")
    name, paths = specification.split("=", 1)
    values = paths.split(",")
    if not name or len(values) != 3:
        raise argparse.ArgumentTypeError("case must be NAME=DATASET,APPROX_LOGITS,DRAFT_LOGITS")
    return name, *(Path(value) for value in values)


def load_case(specification: tuple[str, Path, Path, Path]) -> Case:
    name, dataset, approximate, draft = specification
    with np.load(dataset, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        row_meta = data["row_meta"]
        row_scalars = data["row_scalars"]
        labels = data["row_labels"][:, 6].astype(bool)
    features = {
        feature: row_scalars[:, index].astype(np.float64)
        for index, feature in enumerate(metadata["row_scalar_names"])
    }
    features.update(raw_features(
        row_meta, approximate, draft, int(metadata["vocab"]),
        int(metadata["draft_rows"]), int(metadata["verify_k"])))
    return Case(name, row_meta, labels, features)


def guarded_rows(case: Case, clauses: tuple[Clause, ...], whole_block: bool) -> np.ndarray:
    risky = np.zeros(len(case.labels), dtype=bool)
    for clause in clauses:
        values = case.features[clause.feature]
        risky |= values >= clause.threshold if clause.direction == "high" else values <= clause.threshold
    guarded = np.zeros(len(case.labels), dtype=bool)
    for block in sorted(set(map(int, case.row_meta[:, 1]))):
        indices = np.flatnonzero(case.row_meta[:, 1] == block)
        flagged = indices[risky[indices]]
        if not len(flagged):
            continue
        if whole_block:
            guarded[indices] = True
            continue
        highest_row = max(int(case.row_meta[index, 2]) for index in flagged)
        guarded[indices[case.row_meta[indices, 2] <= highest_row]] = True
    return guarded


def evaluate(cases: list[Case], clauses: tuple[Clause, ...], whole_block: bool) -> dict:
    per_case = {}
    guarded_total = rows_total = captured_total = flips_total = 0
    for case in cases:
        guarded = guarded_rows(case, clauses, whole_block)
        captured = int(np.count_nonzero(case.labels & guarded))
        flips = int(np.count_nonzero(case.labels))
        per_case[case.name] = {
            "guarded": int(np.count_nonzero(guarded)), "rows": len(guarded),
            "captured": captured, "flips": flips,
        }
        guarded_total += int(np.count_nonzero(guarded))
        rows_total += len(guarded)
        captured_total += captured
        flips_total += flips
    return {
        "clauses": clauses, "cases": per_case,
        "guarded": guarded_total, "rows": rows_total,
        "captured": captured_total, "flips": flips_total,
    }


def thresholds(cases: list[Case], feature: str, direction: str) -> list[float]:
    values = np.concatenate([case.features[feature] for case in cases])
    probabilities = (0.35, 0.45, 0.55, 0.65, 0.72, 0.78, 0.84, 0.88, 0.91, 0.94)
    if direction == "low":
        probabilities = tuple(1.0 - value for value in probabilities)
    return sorted(set(float(np.quantile(values, probability)) for probability in probabilities))


def format_rule(clauses: tuple[Clause, ...]) -> str:
    operator = {"high": ">=", "low": "<="}
    return " OR ".join(
        f"{clause.feature} {operator[clause.direction]} {clause.threshold:.6g}"
        for clause in clauses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=parse_case, action="append", required=True)
    parser.add_argument("--hard-case", default="hard")
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--whole-block", action="store_true",
                        help="retry/guard every row when any post-verify row is risky")
    args = parser.parse_args()
    cases = [load_case(specification) for specification in args.case]
    if args.hard_case not in {case.name for case in cases}:
        parser.error("--hard-case must name one supplied case")

    orientations = {
        "draft_entropy_norm": "high",
        "draft_top1_p": "low",
        "draft_margin": "low",
        "draft_entropy_delta": "high",
        "draft_top1_p_drop": "high",
        "draft_margin_drop": "high",
        "draft_adjacent_top8_churn": "high",
        "target_entropy_norm": "high",
        "target_top1_p": "low",
        "target_margin": "low",
        "target_entropy_delta": "high",
        "target_top1_p_drop": "high",
        "target_margin_drop": "high",
        "target_current_draft_top1_disagree": "high",
    }
    grids = {feature: thresholds(cases, feature, direction)
             for feature, direction in orientations.items()}
    families = [
        ("draft_entropy_norm",), ("draft_top1_p",), ("draft_margin",),
        ("draft_entropy_delta",), ("draft_top1_p_drop",),
        ("draft_adjacent_top8_churn",),
        ("draft_entropy_delta", "draft_top1_p"),
        ("draft_entropy_delta", "draft_margin"),
        ("draft_entropy_delta", "draft_entropy_norm"),
        ("draft_top1_p_drop", "draft_top1_p"),
        ("draft_entropy_delta", "draft_top1_p", "draft_margin"),
        ("target_entropy_norm",), ("target_top1_p",), ("target_margin",),
        ("target_entropy_delta",), ("target_top1_p_drop",),
        ("target_current_draft_top1_disagree",),
        ("target_top1_p", "target_margin"),
        ("target_entropy_norm", "target_margin"),
        ("target_top1_p_drop", "target_top1_p"),
        ("target_entropy_delta", "target_top1_p"),
        ("target_top1_p", "draft_top1_p"),
        ("target_top1_p", "draft_entropy_delta"),
    ]
    results = []
    seen = set()
    for family in families:
        for values in itertools.product(*(grids[feature] for feature in family)):
            clauses = tuple(Clause(feature, orientations[feature], threshold)
                            for feature, threshold in zip(family, values, strict=True))
            signature = tuple((clause.feature, clause.direction, clause.threshold)
                              for clause in clauses)
            if signature in seen:
                continue
            seen.add(signature)
            result = evaluate(cases, clauses, args.whole_block)
            hard = result["cases"][args.hard_case]
            result["sort"] = (
                hard["flips"] - hard["captured"],
                result["flips"] - result["captured"],
                result["guarded"] / result["rows"],
                len(clauses),
            )
            results.append(result)
    results.sort(key=lambda result: result["sort"])

    print(f"cases={len(cases)} rows={sum(len(case.labels) for case in cases)} "
          f"flips={sum(np.count_nonzero(case.labels) for case in cases)}")
    print("| rule | total guarded | total flips | " + " | ".join(case.name for case in cases) + " |")
    print("|---|---:|---:|" + "---:|" * len(cases))
    for result in results[:args.top]:
        cells = []
        for case in cases:
            value = result["cases"][case.name]
            cells.append(f"{value['captured']}/{value['flips']} @ {value['guarded']}/{value['rows']}")
        print(f"| {format_rule(result['clauses'])} | "
              f"{result['guarded']}/{result['rows']} | "
              f"{result['captured']}/{result['flips']} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
