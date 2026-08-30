#!/usr/bin/env python3
"""Dataset/provenance tests for train_falsifier_moe.py."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from falsifier_moe import FalsifierMoEConfig
from train_falsifier_moe import (
    PromptShard,
    TraceChunkDataset,
    decode_residency,
    immediate_targets,
    split_by_prompt,
    trajectory_targets,
)


def test_targets() -> None:
    masks = np.asarray([[0b101, 0b010, 0, 0]], dtype=np.uint32)
    decoded = decode_residency(masks)
    assert decoded.shape == (1, 32, 4)
    assert decoded[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert decoded[0, 1].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert decoded[0, 2].tolist() == [1.0, 0.0, 0.0, 0.0]

    labels = np.zeros((5, 9), dtype=np.float32)
    labels[:, 2] = 1.0
    labels[1, 6] = 1.0
    labels[3, 0] = 3.0
    hazard, peak = trajectory_targets(labels, (1, 2, 4))
    assert hazard[0].tolist() == [0.0, 1.0, 1.0]
    assert hazard[1].tolist() == [1.0, 1.0, 1.0]
    assert hazard[2].tolist() == [0.0, 0.0, 0.0]
    assert peak[0, 0] == 0.0 and peak[2, 2] > 1.0
    immediate = immediate_targets(labels)
    assert immediate.shape == (5, 5)
    assert immediate[1, 4] == 1.0


def write_shard(path: Path, prompt_id: str, gram: bool, legacy: bool = False) -> None:
    events = 4
    rows = 2
    metadata = {
        "schema": "insignia-falsifier-dataset-v2",
        "prompt_id": prompt_id,
        "family": "synthetic",
        "policy": "exact" if gram else "on-policy",
        "geometry": {"candidate_k": 32},
    }
    if not legacy:
        metadata["gram_present"] = gram
    row_labels = np.zeros((rows, 9), dtype=np.float32)
    row_labels[:, 2] = 1.0
    row_labels[1, 6] = 1.0
    arrays = {
        "metadata": np.asarray(json.dumps(metadata)),
        "event_meta": np.zeros((events, 6), dtype=np.int64),
        "event_row_index": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "candidate_ids": np.zeros((events, 32), dtype=np.int16),
        "candidate_logits": np.zeros((events, 32), dtype=np.float32),
        "candidate_choice": np.zeros((events, 32), dtype=np.float32),
        "candidate_residency": np.zeros((events, 4), dtype=np.uint32),
        "expert_multiplicity": np.zeros((events, 8), dtype=np.float32),
        "router_features": np.zeros((events, 16), dtype=np.float32),
        "hidden_countsketch": np.zeros((events, 64), dtype=np.float32),
        "event_derived": np.zeros((events, 9), dtype=np.float32),
        "contribution_gram": np.zeros((events, 36), dtype=np.float32),
        "row_meta": np.zeros((rows, 4), dtype=np.int32),
        "row_scalars": np.zeros((rows, 13), dtype=np.float32),
        "row_logit_sketch": np.zeros((rows, 3, 64), dtype=np.float32),
        "row_labels": row_labels,
    }
    if not legacy:
        arrays["event_label_mask"] = np.full(events, gram, dtype=np.bool_)
    np.savez_compressed(path, **arrays)


def test_provenance_and_padding() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        exact_path = root / "exact.npz"
        policy_path = root / "policy.npz"
        write_shard(exact_path, "exact-prompt", gram=True, legacy=True)
        write_shard(policy_path, "policy-prompt", gram=False)
        exact = PromptShard(exact_path, (8, 16, 32))
        policy = PromptShard(policy_path, (8, 16, 32))
        assert exact.gram_present and bool(np.all(exact.targets["gram_mask"]))
        assert not bool(np.any(exact.targets["row_mask"]))
        assert not policy.gram_present and bool(np.all(policy.targets["row_mask"]))
        assert not bool(np.any(policy.targets["gram_mask"]))
        assert not bool(np.any(policy.targets["free_hazard_mask"]))

        train, validation, ids = split_by_prompt([exact, policy], 0.5, 53)
        assert len(train) == len(validation) == 1 and len(ids) == 1
        dataset = TraceChunkDataset([policy], sequence_length=16)
        item = dataset[0]
        assert item["inputs"]["candidate_residency"].shape == (16, 32, 4)
        assert int(item["inputs"]["valid"].sum()) == 4
        assert not bool(torch.any(item["inputs"]["valid"][4:]))


if __name__ == "__main__":
    test_targets()
    test_provenance_and_padding()
    print("falsifier MoE training tests: PASS")

