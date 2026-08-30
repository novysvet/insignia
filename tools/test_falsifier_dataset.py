#!/usr/bin/env python3
"""Synthetic schema/alignment test for build_falsifier_dataset.py."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from analyze_falsifier_candidates import analyze
from build_falsifier_dataset import EVENT_DTYPE, TRACE_HEADER, build, repair_legacy_zero_epochs


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trace = root / "teacher.bin"
        feature_trace = root / "on-policy.bin"
        exact_path = root / "exact.f32"
        approximate_path = root / "approximate.f32"
        draft_path = root / "draft.f32"
        output = root / "dataset.npz"
        feature_output = root / "feature-dataset.npz"

        events = np.zeros(8, dtype=EVENT_DTYPE)
        cursor = 0
        for epoch in range(2):
            for layer in (1, 2):
                for row in range(2):
                    event = events[cursor]
                    event["epoch"] = epoch + 7
                    event["layer"] = layer
                    event["row"] = row
                    event["tokens"] = 2
                    event["verify_row"] = row
                    event["exec_k"] = 8
                    event["expert"] = np.arange(8) + row
                    event["weight"] = np.linspace(0.6, 0.025, 8)
                    event["candidate_expert"] = np.arange(32)
                    event["candidate_logit"] = np.linspace(4, 1, 32)
                    event["candidate_choice"] = np.linspace(1, 0, 32)
                    event["candidate_residency"] = np.asarray(
                        [0x0000FFFF, 0, 0x00FF00FF, 0], dtype=np.uint32)
                    event["router_summary"] = np.arange(8)
                    diagonal = np.linspace(0.01, 0.08, 8)
                    upper = np.zeros(36, dtype=np.float32)
                    index = 0
                    for left in range(8):
                        for right in range(left, 8):
                            if left == right:
                                upper[index] = diagonal[left]
                            index += 1
                    event["contribution_gram"] = upper
                    event["tail"][0] = np.sum(diagonal)
                    cursor += 1
        header = TRACE_HEADER.pack(b"INSFAL1\0", 2, 64, EVENT_DTYPE.itemsize,
                                   4, 64, 8, 32, 64, 0, 32, 3, bytes(28))
        with trace.open("wb") as handle:
            handle.write(header)
            events.tofile(handle)
        legacy = events.copy()
        legacy["epoch"] = 0
        repaired = repair_legacy_zero_epochs(legacy)
        assert list(np.unique(repaired["epoch"])) == [0, 1]

        generator = np.random.default_rng(123)
        exact = generator.normal(size=(5, 32)).astype("<f4")
        approximate = exact.copy()
        exact[3] = 0
        approximate[3] = 0
        exact[3, 1] = 10
        approximate[3, 2] = 10
        draft = generator.normal(size=(2, 3, 32)).astype("<f4")
        exact.tofile(exact_path)
        approximate.tofile(approximate_path)
        draft.tofile(draft_path)

        build(argparse.Namespace(
            trace=trace,
            exact_logits=exact_path,
            approx_logits=approximate_path,
            draft_logits=draft_path,
            output=output,
            prompt_id="synthetic",
            family="test",
            policy="one-flip",
            vocab=32,
            verify_k=2,
            draft_rows=3,
            top_logits=10,
            sketch_dim=8,
            gram_rel_tolerance=1e-6,
        ))
        with np.load(output) as dataset:
            metadata = json.loads(str(dataset["metadata"]))
            assert metadata["gram_present"] is True
            assert dataset["event_meta"].shape == (8, 6)
            assert dataset["router_features"].shape == (8, 16)
            assert dataset["candidate_ids"].shape == (8, 32)
            assert dataset["candidate_residency"].shape == (8, 4)
            assert dataset["contribution_gram"].shape == (8, 36)
            assert np.all(dataset["event_label_mask"])
            assert dataset["row_meta"].shape == (4, 4)
            assert dataset["row_logit_sketch"].shape == (4, 3, 8)
            assert int(np.sum(dataset["row_labels"][:, 6])) == 1
            assert tuple(dataset["row_top1"][2]) == (1, 2)

        feature_events = events.copy()
        feature_events["flags"] |= np.uint16(1 << 2)
        feature_events["contribution_gram"] = 0
        feature_events["tail"][:, :3] = 0
        feature_header = TRACE_HEADER.pack(
            b"INSFAL1\0", 2, 64, EVENT_DTYPE.itemsize,
            4, 64, 8, 32, 64, 0, 32, 2, bytes(28))
        with feature_trace.open("wb") as handle:
            handle.write(feature_header)
            feature_events.tofile(handle)
        build(argparse.Namespace(
            trace=feature_trace,
            exact_logits=exact_path,
            approx_logits=approximate_path,
            draft_logits=draft_path,
            output=feature_output,
            prompt_id="synthetic",
            family="test",
            policy="on-policy",
            vocab=32,
            verify_k=2,
            draft_rows=3,
            top_logits=10,
            sketch_dim=8,
            gram_rel_tolerance=1e-6,
        ))
        with np.load(feature_output) as dataset:
            metadata = json.loads(str(dataset["metadata"]))
            assert metadata["gram_present"] is False
            assert not np.any(dataset["event_label_mask"])
            assert np.all(np.isnan(dataset["contribution_gram"]))
            assert np.all(np.isnan(dataset["event_tail"][:, :3]))
            assert np.all(np.isfinite(dataset["event_tail"][:, 3]))
        observed = analyze(feature_output, [16], [7], [0.01])
        assert observed["observed_policy"] is not None
        assert observed["observed_policy"]["records"]["union"] > 0
        ceiling = analyze(output, [16], [7], [0.01])
        assert len(ceiling["points"]) == 1
        assert ceiling["points"][0]["layer_groups"] == 4
        joint = analyze(output, [16], [7], [0.01], joint_options=4)
        point = joint["points"][0]
        assert point["joint_options"] == 4
        assert point["joint_union_saved_fraction"] >= point["union_saved_fraction"]
        print("falsifier dataset synthetic test: PASS")


if __name__ == "__main__":
    main()
