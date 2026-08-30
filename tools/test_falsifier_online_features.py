#!/usr/bin/env python3
"""Replay dataset shards through the causal online feature state."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from falsifier_online_features import OnlineEventState


def check_shard(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        meta = np.asarray(archive["event_meta"], dtype=np.int64)
        experts = np.asarray(archive["expert_ids"], dtype=np.int64)
        candidates = np.asarray(archive["candidate_ids"], dtype=np.int64)
        masks = np.asarray(archive["candidate_residency"], dtype=np.uint32)
        reference_derived = np.asarray(archive["event_derived"], dtype=np.float32)
        reference_multiplicity = np.asarray(
            archive["expert_multiplicity"], dtype=np.float32)
    state = OnlineEventState()
    actual_derived = np.zeros_like(reference_derived)
    actual_multiplicity = np.zeros_like(reference_multiplicity)
    for epoch in sorted(set(map(int, meta[:, 0]))):
        state.begin_round(epoch)
        epoch_indices = np.flatnonzero(meta[:, 0] == epoch)
        for layer in sorted(set(map(int, meta[epoch_indices, 1]))):
            indices = epoch_indices[meta[epoch_indices, 1] == layer]
            order = np.argsort(meta[indices, 2])
            indices = indices[order]
            derived, multiplicity = state.layer(
                layer, meta[indices, 2], experts[indices], candidates[indices],
                masks[indices])
            actual_derived[indices] = derived
            actual_multiplicity[indices] = multiplicity
    assert np.array_equal(actual_derived, reference_derived), path
    assert np.array_equal(actual_multiplicity, reference_multiplicity), path


def test_corpus() -> None:
    paths = sorted(Path(value) for value in glob.glob(
        "scratch/falsifier-data-20260830/*-onpolicy.npz"))
    if not paths:
        raise AssertionError("local falsifier corpus is unavailable")
    for path in paths:
        check_shard(path)


if __name__ == "__main__":
    test_corpus()
    print("online Falsifier feature tests passed")
