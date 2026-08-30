#!/usr/bin/env python3
"""Causal online feature state shared by native Falsifier implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def expert_overlap(left: np.ndarray, right: np.ndarray) -> float:
    return len(set(map(int, left)) & set(map(int, right))) / 8.0


@dataclass
class OnlineEventState:
    """Streaming equivalent of build_falsifier_dataset.event_derived.

    Call `begin_round` once per DFlash verify round, then `layer` in increasing
    target-layer order. All rows for one layer must arrive together because
    union/multiplicity are layer-group properties.
    """

    previous_round: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    current_round: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    round_id: int | None = None

    def begin_round(self, round_id: int) -> None:
        if self.round_id is not None:
            self.previous_round = self.current_round
        self.current_round = {}
        self.round_id = round_id

    def layer(self, layer: int, verify_rows: np.ndarray, experts: np.ndarray,
              candidate_ids: np.ndarray,
              residency_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.round_id is None:
            raise ValueError("begin_round must precede layer")
        rows = np.asarray(verify_rows, dtype=np.int64)
        selected = np.asarray(experts, dtype=np.int64)
        candidates = np.asarray(candidate_ids, dtype=np.int64)
        residency = np.asarray(residency_masks, dtype=np.uint32)
        count = len(rows)
        if (selected.shape != (count, 8) or candidates.shape != (count, 32)
                or residency.shape != (count, 4)):
            raise ValueError("invalid online Falsifier layer geometry")
        if len(set(map(int, rows))) != count:
            raise ValueError("duplicate verify row in one layer group")

        derived = np.zeros((count, 9), dtype=np.float32)
        multiplicity = np.zeros((count, 8), dtype=np.float32)
        occurrence: dict[int, int] = {}
        current_by_row = {int(row): selected[index]
                          for index, row in enumerate(rows)}
        for row_experts in selected:
            for expert in row_experts:
                occurrence[int(expert)] = occurrence.get(int(expert), 0) + 1

        for index, row_value in enumerate(rows):
            row = int(row_value)
            rank = {int(expert): candidate
                    for candidate, expert in enumerate(candidates[index])}
            missing = [int(expert) for expert in selected[index]
                       if int(expert) not in rank]
            if missing:
                raise ValueError(f"selected experts absent from candidates: {missing}")
            for group in range(4):
                mask = int(residency[index, group])
                resident = sum((mask >> rank[int(expert)]) & 1
                               for expert in selected[index])
                derived[index, group] = resident / 8.0

            previous_layer = self.current_round.get((layer - 1, row))
            previous_row = current_by_row.get(row - 1)
            previous_round = self.previous_round.get((layer, row))
            for column, other in ((4, previous_layer), (5, previous_row),
                                  (6, previous_round)):
                if other is not None:
                    derived[index, column] = expert_overlap(
                        selected[index], other)
            derived[index, 7] = len(occurrence) / (8.0 * count)
            derived[index, 8] = sum(
                occurrence[int(expert)] == 1 for expert in selected[index]) / 8.0
            multiplicity[index] = [occurrence[int(expert)] / count
                                   for expert in selected[index]]
            self.current_round[(layer, row)] = selected[index].copy()
        return derived, multiplicity
