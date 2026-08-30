#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import random
import struct
from collections import Counter
import sys
import tempfile
import unittest

import numpy as np

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import analyze_fp8_residency as analysis
import fp8_residency_codec as codec
import sample_fp8_cache as sample


FORMAT_MODE = {
    "byte_palette4": codec.Mode.BYTE_PALETTE4,
    "byte_palette5": codec.Mode.BYTE_PALETTE5,
    "byte_palette6": codec.Mode.BYTE_PALETTE6,
    "mag_palette4": codec.Mode.MAG_PALETTE4,
    "mag_palette5": codec.Mode.MAG_PALETTE5,
    "mag_palette6": codec.Mode.MAG_PALETTE6,
    "exp_palette2": codec.Mode.EXP_PALETTE2,
    "exp_palette3": codec.Mode.EXP_PALETTE3,
    "mag_xor4": codec.Mode.MAG_XOR4,
    "zero_sparse": codec.Mode.ZERO_SPARSE,
    "bitplane_const": codec.Mode.BITPLANE_CONST,
}


class CeilingFormulaTest(unittest.TestCase):
    def test_vectorized_sizes_match_reference_encoder(self) -> None:
        rng = random.Random(0xA11CE)
        for symbols in (64, 128, 256, 512, 1024):
            raw_tiles: list[bytes] = []
            for case in range(40):
                selector = case % 8
                if selector == 0:
                    raw = bytes(rng.randrange(256) for _ in range(symbols))
                elif selector == 1:
                    alphabet = [rng.randrange(256) for _ in range(rng.randrange(1, 20))]
                    raw = bytes(rng.choice(alphabet) for _ in range(symbols))
                elif selector == 2:
                    raw = bytes(rng.choice((0, 128, 127, 255)) for _ in range(symbols))
                elif selector == 3:
                    raw = bytes([rng.randrange(256)] * symbols)
                elif selector == 4:
                    raw = bytes((rng.randrange(8) | (rng.randrange(4) << 3) | (rng.randrange(2) << 7)) for _ in range(symbols))
                elif selector == 5:
                    base = rng.randrange(128)
                    raw = bytes((base ^ rng.randrange(15)) | (rng.randrange(2) << 7) for _ in range(symbols))
                elif selector == 6:
                    raw = bytes(range(256)) * (symbols // 256) if symbols >= 256 else bytes(range(symbols))
                else:
                    raw = bytes((index * 17 + index // 64) & 255 for index in range(symbols))
                raw_tiles.append(raw)
            tiles = np.frombuffer(b"".join(raw_tiles), dtype=np.uint8).reshape(len(raw_tiles), symbols)
            measured = analysis.measure_tile_batch(tiles)
            for row, raw in enumerate(raw_tiles):
                raw_padded = codec.align_up(len(raw))
                for name, mode in FORMAT_MODE.items():
                    encoded = codec.encode_mode(mode, raw)
                    expected = raw_padded if encoded.padded_bytes >= raw_padded else encoded.padded_bytes
                    self.assertEqual(
                        int(measured.selected_padded[name][row]),
                        expected,
                        (symbols, row, name, encoded.parameter, encoded.stored_bytes),
                    )
                    self.assertEqual(
                        bool(measured.raw_selected[name][row]),
                        encoded.padded_bytes >= raw_padded,
                    )
                mixed = codec.encode_tile(raw)
                self.assertEqual(int(measured.selected_padded["mixed"][row]), mixed.padded_bytes)
                self.assertEqual(bool(measured.raw_selected["mixed"][row]), mixed.mode == codec.Mode.RAW)
            expected_modes = Counter(
                "raw" if (chosen := codec.encode_tile(raw)).mode == codec.Mode.RAW
                else analysis.MODE_TO_FORMAT[chosen.mode]
                for raw in raw_tiles
            )
            self.assertEqual(measured.mixed_mode_counts, expected_modes)

    def test_entropy_known_distribution(self) -> None:
        histogram = np.zeros(256, dtype=np.uint64)
        histogram[0] = 4
        histogram[1] = 4
        self.assertAlmostEqual(analysis.entropy_bits(histogram), 1.0)

    def test_pair_metrics_identity(self) -> None:
        joint = np.zeros((256, 256), dtype=np.uint64)
        for value in range(256):
            joint[value, value] = 1
        metrics = analysis.pair_metrics(joint)
        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertAlmostEqual(metrics["equal_frequency"], 1.0)
        self.assertAlmostEqual(metrics["mutual_information_bits"], 8.0)
        self.assertAlmostEqual(metrics["mean_absolute_delta"], 0.0)


def _align4096(value: int) -> int:
    return (value + 4095) & -4096


def _make_analysis_cache(root: pathlib.Path) -> sample.SourceCache:
    train_names: list[str] = []
    heldout_names: list[str] = []
    for layer in range(500):
        name = f"model.language_model.layers.{layer}.self_attn.q_proj.weight"
        split = analysis.sha_split("dense", name, 25)
        target = heldout_names if split == "heldout" else train_names
        if len(target) < (4 if split == "heldout" else 8):
            target.append(name)
        if len(train_names) == 8 and len(heldout_names) == 4:
            break
    names = sorted(train_names + heldout_names)
    rows, cols = 16, 128
    cursor = 0
    records: list[tuple[str, int, int, int, int]] = []
    spans: list[tuple[int, bytes]] = []
    for ordinal, name in enumerate(names):
        weight_offset = _align4096(cursor)
        weight_bytes = rows * cols
        scale_offset = _align4096(weight_offset + weight_bytes)
        scale_bytes = rows * (cols // 64) * 2
        cursor = _align4096(scale_offset + scale_bytes)
        if ordinal % 3 == 0:
            weights = bytes([0, 0x80, 0x38, 0xB8] * (weight_bytes // 4))
        elif ordinal % 3 == 1:
            alphabet = (0x20, 0x28, 0x30, 0x38, 0xA0, 0xA8, 0xB0, 0xB8)
            weights = bytes(alphabet[(index + ordinal) & 7] for index in range(weight_bytes))
        else:
            weights = bytes(((index % 11) + ((index + ordinal) & 1) * 0x80) for index in range(weight_bytes))
        scales = bytes((index * 17 + ordinal) & 0xFF for index in range(scale_bytes))
        spans.extend(((weight_offset, weights), (scale_offset, scales)))
        records.append((name, weight_offset, weight_bytes, scale_offset, scale_bytes))

    prefix = root / "dense"
    data = bytearray(cursor)
    for offset, payload in spans:
        data[offset : offset + len(payload)] = payload
    pathlib.Path(str(prefix) + ".bin").write_bytes(data)
    with pathlib.Path(str(prefix) + ".index").open("wb") as output:
        output.write(sample.INDEX_HEADER.pack(sample.MAGIC, 1, 64, len(records), cursor))
        for name, weight_offset, weight_bytes, scale_offset, scale_bytes in records:
            encoded = name.encode()
            output.write(
                sample.INDEX_RECORD.pack(
                    len(encoded), rows, cols, weight_offset, weight_bytes,
                    scale_offset, scale_bytes,
                )
            )
            output.write(encoded)
    return sample.parse_cache(prefix, "dense")


class EndToEndAnalysisTest(unittest.TestCase):
    def test_directory_analysis_outputs_and_gates(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            cache = _make_analysis_cache(root)
            plan = [
                sample.SampleSlice(
                    record=record, row_begin=0, rows=record.rows, selection="complete-small",
                    row_band="all", stratum_bytes=record.weight_bytes,
                )
                for record in cache.records
            ]
            sample_root = root / "sample"
            sample.write_sample_directory(sample_root, [cache], {"dense": plan})
            source_inventory = sample.source_inventory(cache)
            collection = {
                "schema": "fp8-residency-sample-v1",
                "source_inventory": [source_inventory],
                "archive": {"full_sha256": "0" * 64},
            }
            identity_ratios = {(record.source_kind, record.family): 1.0 for record in cache.records}
            disk_bytes, logical_bytes, encoded_weights = analysis._layout_bytes(
                [
                    analysis.TensorInventory(
                        source_kind=record.source_kind, tensor_name=record.name,
                        family=record.family, stage=record.stage, layer=record.layer,
                        ordinal=record.ordinal, rows=record.rows, cols=record.cols,
                        weight_bytes=record.weight_bytes, scale_bytes=record.scale_bytes,
                    )
                    for record in cache.records
                ],
                identity_ratios,
            )
            self.assertEqual(disk_bytes, source_inventory["data_bytes"])
            self.assertEqual(
                logical_bytes,
                source_inventory["weight_bytes"] + source_inventory["scale_bytes"],
            )
            self.assertEqual(encoded_weights, source_inventory["weight_bytes"])
            collection_path = root / "collection.json"
            collection_path.write_text(json.dumps(collection), encoding="utf-8")
            output = root / "analysis"
            with contextlib.redirect_stdout(io.StringIO()):
                status = analysis.main(
                    [
                        str(sample_root), "--collection-json", str(collection_path),
                        "--output-dir", str(output), "--minimum-dense-weight-mib", "0",
                        "--holdout-percent", "25", "--bootstrap-replicates", "16",
                    ]
                )
            self.assertEqual(status, 0)
            expected = {
                "tensor_inventory.csv", "family_entropy.csv", "tensor_tile_stats.csv",
                "tensor_codec_results.csv", "family_codec_results.csv",
                "heldout_results.json", "summary.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            summary = json.loads((output / "summary.json").read_text())
            result = summary["heldout"]["weighted_whole_cache_heldout"]["dense"]
            self.assertEqual(summary["analyzed_tensor_count"], len(cache.records))
            self.assertLess(result["heldout_weight_ratio"], 1.0)
            self.assertGreater(result["estimated_logical_vram_bytes_saved"], 0)
            self.assertIn(
                result["decision"],
                {
                    "eligible_for_gpu_microbenchmark",
                    "eligible_for_allocator_only_probe",
                    "stop_or_revisit_fixed_block_formats",
                },
            )
            self.assertTrue(result["allocator_reclaim_is_modeled_not_measured"])
            self.assertEqual(result["bootstrap_replicates"], 16)

    def test_selection_tie_prefers_mma_tile_then_decode_cost(self) -> None:
        self.assertLess(
            analysis._config_tie_key((1024, "byte_palette4")),
            analysis._config_tie_key((512, "byte_palette4")),
        )
        self.assertLess(
            analysis._config_tie_key((1024, "byte_palette4")),
            analysis._config_tie_key((1024, "mixed")),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
