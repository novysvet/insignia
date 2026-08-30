#!/usr/bin/env python3
"""Property and malformed-input tests for sample_fp8_cache.py."""

from __future__ import annotations

import csv
import hashlib
import os
import pathlib
import shutil
import struct
import tempfile
import unittest

import sample_fp8_cache as sample


def align(value: int) -> int:
    return (value + sample.ALIGNMENT - 1) & -sample.ALIGNMENT


def deterministic_bytes(size: int, salt: int) -> bytes:
    # Covers every possible E4M3 byte repeatedly, including +/-0 and both NaN
    # encodings, without assigning numeric meaning to the payload.
    return bytes(((index * 73 + salt) & 0xFF) for index in range(size))


def make_cache(root: pathlib.Path, name: str, tensors: list[tuple[str, int, int]]) -> pathlib.Path:
    prefix = root / name
    cursor = 0
    records = []
    spans: list[tuple[int, bytes]] = []
    for ordinal, (tensor_name, rows, cols) in enumerate(tensors):
        weight_bytes = rows * cols
        scale_bytes = rows * (cols // sample.GROUP_SIZE) * 2
        weight_offset = align(cursor)
        scale_offset = align(weight_offset + weight_bytes)
        cursor = align(scale_offset + scale_bytes)
        weights = deterministic_bytes(weight_bytes, 11 + ordinal)
        # Raw FP16 bytes intentionally include zeros, infinities and NaN payloads.
        scales = deterministic_bytes(scale_bytes, 197 + ordinal)
        records.append((tensor_name, rows, cols, weight_offset, weight_bytes,
                        scale_offset, scale_bytes))
        spans.extend(((weight_offset, weights), (scale_offset, scales)))

    data = bytearray(cursor)
    for offset, payload in spans:
        data[offset : offset + len(payload)] = payload
    pathlib.Path(str(prefix) + ".bin").write_bytes(data)
    with pathlib.Path(str(prefix) + ".index").open("wb") as output:
        output.write(sample.INDEX_HEADER.pack(
            sample.MAGIC, sample.VERSION, sample.GROUP_SIZE, len(records), len(data)
        ))
        for record in records:
            tensor_name, rows, cols, weight_offset, weight_bytes, scale_offset, scale_bytes = record
            encoded = tensor_name.encode("utf-8")
            output.write(sample.INDEX_RECORD.pack(
                len(encoded), rows, cols, weight_offset, weight_bytes,
                scale_offset, scale_bytes,
            ))
            output.write(encoded)
    return prefix


class SourceIndexTests(unittest.TestCase):
    def test_parse_valid_cache_and_preserve_all_byte_values(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            prefix = make_cache(root, "dense", [
                ("model.language_model.layers.0.self_attn.q_proj.weight", 8, 256),
                ("model.language_model.layers.22.self_attn.q_proj.weight", 16, 256),
                ("model.language_model.layers.44.self_attn.q_proj.weight", 32, 256),
                ("lm_head.weight", 4, 256),
            ])
            cache = sample.parse_cache(prefix, "dense")
            self.assertEqual(len(cache.records), 4)
            self.assertEqual({record.stage for record in cache.records},
                             {"early", "middle", "late", "global"})
            raw = pathlib.Path(str(prefix) + ".bin").read_bytes()
            values = set()
            for record in cache.records:
                values.update(raw[record.weight_offset : record.weight_offset + record.weight_bytes])
            self.assertEqual(values, set(range(256)))

    def test_rejects_bad_magic_trailing_index_and_truncated_data(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            prefix = make_cache(root, "dense", [("lm_head.weight", 4, 64)])
            index = pathlib.Path(str(prefix) + ".index")
            original = index.read_bytes()

            index.write_bytes(b"IGLMQ8A1" + original[8:])
            with self.assertRaises(sample.SampleError):
                sample.parse_cache(prefix, "dense")

            index.write_bytes(original + b"x")
            with self.assertRaises(sample.SampleError):
                sample.parse_cache(prefix, "dense")

            index.write_bytes(original)
            data = pathlib.Path(str(prefix) + ".bin")
            data.write_bytes(data.read_bytes()[:-1])
            with self.assertRaises(sample.SampleError):
                sample.parse_cache(prefix, "dense")

    def test_rejects_wrong_geometry_and_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            prefix = make_cache(root, "dense", [("lm_head.weight", 4, 64)])
            index = pathlib.Path(str(prefix) + ".index")
            raw = bytearray(index.read_bytes())
            fields = list(sample.INDEX_RECORD.unpack_from(raw, sample.INDEX_HEADER.size))
            fields[4] += 1  # weight_bytes
            sample.INDEX_RECORD.pack_into(raw, sample.INDEX_HEADER.size, *fields)
            index.write_bytes(raw)
            with self.assertRaises(sample.SampleError):
                sample.parse_cache(prefix, "dense")

            prefix = make_cache(root, "overlap", [
                ("model.language_model.layers.0.self_attn.q_proj.weight", 4, 64),
                ("model.language_model.layers.1.self_attn.k_proj.weight", 4, 64),
            ])
            index = pathlib.Path(str(prefix) + ".index")
            raw = bytearray(index.read_bytes())
            cursor = sample.INDEX_HEADER.size
            first = list(sample.INDEX_RECORD.unpack_from(raw, cursor))
            cursor += sample.INDEX_RECORD.size + first[0]
            second = list(sample.INDEX_RECORD.unpack_from(raw, cursor))
            second[3] = first[3]  # second weight_offset overlaps first weights
            sample.INDEX_RECORD.pack_into(raw, cursor, *second)
            index.write_bytes(raw)
            with self.assertRaises(sample.SampleError):
                sample.parse_cache(prefix, "dense", strict_alignment=False)


class CollectionTests(unittest.TestCase):
    def _cache(self, root: pathlib.Path) -> sample.SourceCache:
        prefix = make_cache(root, "dense", [
            ("model.language_model.layers.0.self_attn.q_proj.weight", 64, 256),
            ("model.language_model.layers.22.self_attn.q_proj.weight", 96, 256),
            ("model.language_model.layers.44.self_attn.q_proj.weight", 128, 256),
            ("model.language_model.layers.0.mlp.gate_proj.weight", 48, 512),
            ("model.language_model.layers.22.mlp.gate_proj.weight", 80, 512),
            ("model.language_model.layers.44.mlp.gate_proj.weight", 112, 512),
            ("lm_head.weight", 8, 256),
        ])
        return sample.parse_cache(prefix, "dense")

    def test_deterministic_plan_covers_every_family_stage(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            cache = self._cache(pathlib.Path(text))
            kwargs = dict(
                target_weight_bytes=96 << 10,
                minimum_weight_bytes=64 << 10,
                complete_below_bytes=4 << 10,
                coverage_band_bytes=1 << 10,
            )
            first = sample.plan_source(cache, **kwargs)
            second = sample.plan_source(cache, **kwargs)
            self.assertEqual(first, second)
            expected = {(record.family, record.stage) for record in cache.records}
            actual = {(entry.record.family, entry.record.stage) for entry in first}
            self.assertEqual(actual, expected)
            by_name: dict[str, list[tuple[int, int]]] = {}
            for entry in first:
                by_name.setdefault(entry.record.name, []).append(
                    (entry.row_begin, entry.row_begin + entry.rows)
                )
            for ranges in by_name.values():
                ranges.sort()
                self.assertTrue(all(right[0] >= left[1]
                                    for left, right in zip(ranges, ranges[1:])))

    def test_write_validate_and_source_compare(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            cache = self._cache(root)
            plan = sample.plan_source(
                cache,
                target_weight_bytes=96 << 10,
                minimum_weight_bytes=64 << 10,
                complete_below_bytes=4 << 10,
                coverage_band_bytes=1 << 10,
            )
            output = root / "sample"
            sample.write_sample_directory(output, [cache], {"dense": plan})
            report = sample.validate_sample_directory(
                output, caches=[cache], minimum_dense_weight_bytes=64 << 10
            )
            self.assertGreaterEqual(report["totals"]["dense_weight_bytes"], 64 << 10)
            with (output / "manifest.tsv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source, delimiter="\t"))
            self.assertTrue(rows)
            for row in rows:
                weight = output / row["weight_file"]
                scale = output / row["scale_file"]
                self.assertEqual(hashlib.sha256(weight.read_bytes()).hexdigest(), row["weight_sha256"])
                self.assertEqual(hashlib.sha256(scale.read_bytes()).hexdigest(), row["scale_sha256"])

    def test_tamper_and_extra_file_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            cache = self._cache(root)
            plan = sample.plan_source(
                cache,
                target_weight_bytes=64 << 10,
                minimum_weight_bytes=32 << 10,
                complete_below_bytes=2 << 10,
                coverage_band_bytes=512,
            )
            output = root / "sample"
            written = sample.write_sample_directory(output, [cache], {"dense": plan})
            target = output / written[0].weight_file
            payload = bytearray(target.read_bytes())
            payload[0] ^= 0xFF
            target.write_bytes(payload)
            with self.assertRaises(sample.SampleError):
                sample.validate_sample_directory(output, minimum_dense_weight_bytes=0)

            shutil.rmtree(output)
            sample.write_sample_directory(output, [cache], {"dense": plan})
            (output / "AGENTS.md").write_text("forbidden tracked source\n", encoding="utf-8")
            with self.assertRaises(sample.SampleError):
                sample.validate_sample_directory(output, minimum_dense_weight_bytes=0)

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required for archive splitting")
    def test_split_archive_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            cache = self._cache(root)
            plan = sample.plan_source(
                cache,
                target_weight_bytes=64 << 10,
                minimum_weight_bytes=32 << 10,
                complete_below_bytes=2 << 10,
                coverage_band_bytes=512,
            )
            directory = root / "sample"
            sample.write_sample_directory(directory, [cache], {"dense": plan})
            archive_path = root / "fp8-residency-sample-v1.tar.zst"
            result = sample.build_archive(
                directory, archive_path, zstd_level=1, part_bytes=1024
            )
            self.assertGreater(len(result.paths), 1)
            self.assertFalse(archive_path.exists())
            self.assertTrue(result.paths[0].name.endswith(".part-000"))
            with sample.materialize_sample(result.paths[0]) as extracted:
                report = sample.validate_sample_directory(
                    extracted, minimum_dense_weight_bytes=32 << 10
                )
            self.assertGreater(report["manifest_rows"], 0)

    @unittest.skipUnless(shutil.which("zstd"), "zstd is required for archive round-trip")
    def test_archive_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = pathlib.Path(text)
            cache = self._cache(root)
            plan = sample.plan_source(
                cache,
                target_weight_bytes=64 << 10,
                minimum_weight_bytes=32 << 10,
                complete_below_bytes=2 << 10,
                coverage_band_bytes=512,
            )
            directory = root / "sample"
            sample.write_sample_directory(directory, [cache], {"dense": plan})
            archive_path = root / "fp8-residency-sample-v1.tar.zst"
            result = sample.build_archive(
                directory, archive_path, zstd_level=1, part_bytes=1 << 30
            )
            self.assertEqual(len(result.paths), 1)
            with sample.materialize_sample(archive_path) as extracted:
                report = sample.validate_sample_directory(
                    extracted, minimum_dense_weight_bytes=32 << 10
                )
            self.assertGreater(report["manifest_rows"], 0)


if __name__ == "__main__":
    unittest.main()
