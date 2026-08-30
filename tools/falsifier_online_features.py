#!/usr/bin/env python3
"""Causal online feature state shared by native Falsifier implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from build_falsifier_dataset import (
    CountSketch,
    centered_cosine,
    js_divergence,
    overlap,
    softmax_stats,
    top_values,
)


def expert_overlap(left: np.ndarray, right: np.ndarray) -> float:
    return len(set(map(int, left)) & set(map(int, right))) / 8.0


@dataclass
class OnlineLogitState:
    """Build the 16 scalars and 3x64 sketches before target verification."""

    vocab: int
    sketch_dimension: int = 64
    top_logits: int = 32
    round_index: int = 0
    previous_prior: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.sketch = CountSketch(self.vocab, self.sketch_dimension)

    def begin_round(self, prior_logits: np.ndarray, draft_logits: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
        prior = np.asarray(prior_logits, dtype=np.float32)
        draft = np.asarray(draft_logits, dtype=np.float32)
        if prior.shape != (self.vocab,) or draft.ndim != 2 \
                or draft.shape[1] != self.vocab or draft.shape[0] < 2:
            raise ValueError("invalid online logit geometry")
        previous = prior if self.previous_prior is None else self.previous_prior
        prior_probability, prior_stats = softmax_stats(prior)
        previous_probability, _ = softmax_stats(previous)
        temporal_js = js_divergence(previous_probability, prior_probability)
        temporal_cos = centered_cosine(previous, prior)
        temporal_disagree = float(np.argmax(previous) != np.argmax(prior))
        calibration_probability, _ = softmax_stats(draft[0])
        calibration_js = js_divergence(prior_probability, calibration_probability)
        calibration_cos = centered_cosine(prior, draft[0])
        calibration_disagree = float(np.argmax(prior) != np.argmax(draft[0]))
        prior_ids, _ = top_values(prior, self.top_logits)
        scalars: list[list[float]] = []
        sketches: list[np.ndarray] = []
        rows = draft.shape[0] - 1
        round_position = min(
            1.0, np.log1p(self.round_index) / np.log1p(256.0))
        for row in range(rows):
            current = draft[row + 1]
            _, draft_stats = softmax_stats(current)
            current_ids, _ = top_values(current, self.top_logits)
            scalars.append([
                *prior_stats, *draft_stats, calibration_js, calibration_cos,
                calibration_disagree, temporal_js, temporal_cos,
                temporal_disagree, overlap(prior_ids[:8], current_ids[:8]),
                overlap(prior_ids, current_ids), round_position,
                row / max(1, rows - 1),
            ])
            sketches.append(np.stack((
                self.sketch(prior), self.sketch(current),
                self.sketch(current - prior),
            )))
        self.previous_prior = prior.copy()
        self.round_index += 1
        return (np.asarray(scalars, dtype=np.float32),
                np.asarray(sketches, dtype=np.float32))


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
