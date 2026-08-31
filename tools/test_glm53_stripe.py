#!/usr/bin/env python3
"""Fast, CPU-only regression tests for GLM dual-SSD stripe tooling."""

from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import tempfile
import unittest

from stripe_mount import StripeMountError, parse_mountinfo, validate_stripe_mount
from glm53_route_analysis import stripe_miss_weights, write_stripe_miss_weights
from stripe_repack import LOCK_NAME, MAGIC, load_index, load_weights, plan_alt_records, repack
from stripe_verify import verify


def _noop_mount_validator(*_args, **_kwargs):
    return object()


def _huge_host_free(_path):
    return 1 << 60


def _write_fixture(root: pathlib.Path, *, layers: int = 2, experts: int = 10):
    source = root / "source"
    source.mkdir()
    shard_path = source / "model-00001.bin"
    entries = []
    payload = bytearray()

    def add(name: str, dtype: int, size: int, salt: int):
        offset = len(payload)
        payload.extend(bytes(((salt + index * 17) & 255) for index in range(size)))
        entries.append([name, dtype, (size,), 0, 0, offset, size])

    add("model.language_model.embed_tokens.weight", 2, 23, 3)
    for layer in range(layers):
        for expert in range(experts):
            stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            for projection_index, projection in enumerate(("down_proj", "gate_proj", "up_proj")):
                salt = layer * 91 + expert * 13 + projection_index * 7
                add(stem + projection + ".weight", 4, 31 + expert % 3, salt)
                add(stem + projection + ".weight_scale", 7, 17 + layer, salt + 1)
                add(stem + projection + ".weight_scale_2", 1, 4, salt + 2)
    shard_path.write_bytes(payload)
    index = root / "source.index"
    total = sum(entry[6] for entry in entries)
    with index.open("wb") as output:
        output.write(struct.pack(
            "<8s11IQ", MAGIC, 1, 0, 1, len(entries), 4, layers, 32,
            experts, 2, 8, 4, total,
        ))
        encoded = shard_path.name.encode()
        output.write(struct.pack("<HQ", len(encoded), len(payload)))
        output.write(encoded)
        for name, dtype, shape, shard, flags, offset, length in entries:
            encoded = name.encode()
            output.write(struct.pack(
                "<HBBHHQQ", len(encoded), dtype, len(shape), shard, flags, offset, length,
            ))
            output.write(encoded)
            output.write(struct.pack("<" + "I" * len(shape), *shape))
    return source, index


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MountGuardTests(unittest.TestCase):
    def test_mountinfo_parser_unescapes_paths(self):
        entries = parse_mountinfo(
            "36 25 8:1 / / rw - ext4 /dev/root rw\n"
            "40 36 8:2 / /stripe\\040disk rw - ext4 /dev/stripe rw\n"
        )
        self.assertEqual(entries[1].mount_point, pathlib.Path("/stripe disk"))
        self.assertEqual(entries[1].source, "/dev/stripe")

    def test_same_root_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            stripe = root / "stripe"
            source.mkdir()
            stripe.mkdir()
            with self.assertRaisesRegex(StripeMountError, "not an exact mount point|same st_dev"):
                validate_stripe_mount(source, stripe, expected_label=None)

    def test_distinct_exact_mount_is_accepted(self):
        shared_memory = pathlib.Path("/dev/shm")
        if not shared_memory.is_dir():
            self.skipTest("/dev/shm is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            result = validate_stripe_mount(
                temporary, shared_memory, expected_label=None, expected_fs="tmpfs",
            )
        self.assertEqual(result.destination, shared_memory)
        self.assertEqual(result.fs_type, "tmpfs")


class PlannerTests(unittest.TestCase):
    def test_route_trace_weights_count_nvme_misses_and_reset_per_prompt(self):
        a = 3 * 1024 + 1
        b = 3 * 1024 + 2
        c = 4 * 1024 + 7
        streams = [
            [(0, a), (0, b), (1, a), (1, c), (2, b)],
            [(0, a)],
        ]
        misses, requests, hits = stripe_miss_weights(streams, 2)
        self.assertEqual((requests, hits), (6, 1))
        self.assertEqual(misses, {a: 2, b: 2, c: 1})
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "weights.tsv"
            write_stripe_miss_weights(streams, 2, path)
            self.assertEqual(load_weights(path), {
                (3, 1): 2.0,
                (3, 2): 2.0,
                (4, 7): 1.0,
            })

    def test_uniform_plan_is_deterministic_70_30(self):
        first = plan_alt_records([3, 4], 10, main_gbps=7.0, alt_gbps=3.0, seed=11)
        second = plan_alt_records([4, 3], 10, main_gbps=7.0, alt_gbps=3.0, seed=11)
        self.assertEqual(first.selected, second.selected)
        for layer in (3, 4):
            self.assertEqual(sum(key[0] == layer for key in first.selected), 3)
        self.assertAlmostEqual(first.alt_weight / first.total_weight, 0.3)

    def test_skewed_miss_weights_balance_normalized_service(self):
        weights = {(3, expert): weight for expert, weight in enumerate(
            (40.0, 20.0, 10.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0, 1.0)
        )}
        plan = plan_alt_records(
            [3], 10, main_gbps=7.0, alt_gbps=3.0,
            weights=weights, weight_floor=0.0, seed=29,
        )
        self.assertEqual(plan.selected, plan_alt_records(
            [3], 10, main_gbps=7.0, alt_gbps=3.0,
            weights=weights, weight_floor=0.0, seed=29,
        ).selected)
        main_time = (plan.total_weight - plan.alt_weight) / 7.0
        alt_time = plan.alt_weight / 3.0
        self.assertLessEqual(abs(main_time - alt_time), 1.0)

    def test_subset_dp_escapes_greedy_39_trap(self):
        weights = {(3, expert): weight for expert, weight in enumerate(
            (39.0, 20.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0)
        )}
        plan = plan_alt_records(
            [3], 10, main_gbps=7.0, alt_gbps=3.0,
            weights=weights, weight_floor=0.0, seed=29,
        )
        self.assertAlmostEqual(plan.alt_weight, 30.0)

    def test_production_geometry_is_deterministic(self):
        first = plan_alt_records(list(range(3, 45)), 288)
        second = plan_alt_records(list(reversed(range(3, 45))), 288)
        self.assertEqual(first.selected, second.selected)
        self.assertEqual(len(first.selected), 42 * 87)


class LauncherTests(unittest.TestCase):
    def test_stripe_launchers_use_original_primary_plus_overlay(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for name in ("bench-df-ab.sh", "hang-diag.sh", "hang-gdb.sh", "watch-decode.sh"):
            payload = (root / "build" / name).read_text()
            with self.subTest(name=name):
                self.assertIn(
                    "INSIGNIA_GLM53_STRIPE_INDEX=/var/lib/insignia/glm53-flash-text-striped.index",
                    payload,
                )
                self.assertIn("/var/lib/insignia/glm53-flash-text.index", payload)


class RepackTests(unittest.TestCase):
    def test_unmounted_destination_fails_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, index = _write_fixture(root)
            stripe = root / "stripe"
            stripe.mkdir()
            output = root / "striped.index"
            with self.assertRaises(StripeMountError):
                repack(source, index, stripe, output, expected_label=None)
            self.assertEqual(list(stripe.iterdir()), [])
            self.assertFalse(output.exists())

    def test_host_backing_space_fails_before_writes(self):
        with (tempfile.TemporaryDirectory() as temporary,
              tempfile.TemporaryDirectory(dir="/dev/shm") as stripe_temporary):
            root = pathlib.Path(temporary)
            source, index = _write_fixture(root)
            stripe = pathlib.Path(stripe_temporary)
            with self.assertRaisesRegex(StripeMountError, "dynamic VHDX needs"):
                repack(
                    source, index, stripe, root / "striped.index",
                    main_gbps=7.0, alt_gbps=3.0,
                    mount_validator=_noop_mount_validator,
                    host_free_provider=lambda _path: 0,
                )
            self.assertEqual(list(stripe.iterdir()), [])

    def test_force_cannot_alias_authoritative_source_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, index = _write_fixture(root)
            before = _digest(index)
            with self.assertRaisesRegex(ValueError, "output index aliases source index"):
                repack(
                    source, index, root / "stripe", index,
                    mount_validator=_noop_mount_validator,
                    host_free_provider=_huge_host_free,
                    force=True,
                )
            self.assertEqual(_digest(index), before)

    def test_repack_and_verifier_are_deterministic_and_byte_exact(self):
        with (tempfile.TemporaryDirectory() as temporary,
              tempfile.TemporaryDirectory(dir="/dev/shm") as stripe_temporary):
            root = pathlib.Path(temporary)
            source, index = _write_fixture(root)
            stripe = pathlib.Path(stripe_temporary)
            output = root / "striped.index"
            manifest = root / "stripe.manifest.json"
            result = repack(
                source, index, stripe, output, manifest_path=manifest,
                main_gbps=7.0, alt_gbps=3.0, shard_bytes=8192,
                mount_validator=_noop_mount_validator,
                host_free_provider=_huge_host_free,
            )
            index_digest = _digest(output)
            manifest_digest = _digest(manifest)
            stripe_digests = {path.name: _digest(path) for path in stripe.iterdir()}
            checked = verify(
                index, output, source, stripe, manifest_path=manifest,
                sample_records=0, hash_shards=True,
                mount_validator=_noop_mount_validator,
            )
            self.assertEqual(checked.selected_records, result.records)
            self.assertEqual(checked.checked_records, result.records)
            self.assertEqual(checked.checked_tensors, result.records * 9)
            self.assertFalse(list(root.rglob("*.partial")))

            repeated = repack(
                source, index, stripe, output, manifest_path=manifest,
                main_gbps=7.0, alt_gbps=3.0, shard_bytes=8192,
                mount_validator=_noop_mount_validator,
                host_free_provider=_huge_host_free, force=True,
            )
            self.assertEqual(result.generation, repeated.generation)
            self.assertEqual(index_digest, _digest(output))
            self.assertEqual(manifest_digest, _digest(manifest))
            self.assertEqual(stripe_digests,
                             {path.name: _digest(path) for path in stripe.iterdir()})

            source_head, source_shards, _ = load_index(index)
            striped_head, striped_shards, _ = load_index(output)
            self.assertEqual(striped_head[2] - source_head[2], result.shards)
            self.assertTrue(all(name.startswith(f"stripe-{result.generation}-")
                                for name, _ in striped_shards[len(source_shards):]))

    def test_injected_failure_publishes_nothing(self):
        with (tempfile.TemporaryDirectory() as temporary,
              tempfile.TemporaryDirectory(dir="/dev/shm") as stripe_temporary):
            root = pathlib.Path(temporary)
            source, index = _write_fixture(root)
            stripe = pathlib.Path(stripe_temporary)
            output = root / "striped.index"
            manifest = root / "stripe.manifest.json"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                repack(
                    source, index, stripe, output, manifest_path=manifest,
                    main_gbps=7.0, alt_gbps=3.0,
                    mount_validator=_noop_mount_validator,
                    host_free_provider=_huge_host_free,
                    fault_after_records=1,
                )
            self.assertFalse(output.exists())
            self.assertFalse(manifest.exists())
            self.assertEqual([path.name for path in stripe.iterdir()], [LOCK_NAME])
            self.assertFalse(list(root.rglob("*.partial")))

    def test_mount_identity_loss_is_caught_without_path_writes(self):
        with (tempfile.TemporaryDirectory() as temporary,
              tempfile.TemporaryDirectory(dir="/dev/shm") as stripe_parent_text):
            root = pathlib.Path(temporary)
            source, index = _write_fixture(root)
            stripe_parent = pathlib.Path(stripe_parent_text)
            stripe = stripe_parent / "stripe"
            detached = stripe_parent / "detached"
            stripe.mkdir()
            calls = 0

            def detach_on_third_check(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    stripe.rename(detached)
                    stripe.mkdir()
                return object()

            with self.assertRaisesRegex(StripeMountError, "identity changed"):
                repack(
                    source, index, stripe, root / "striped.index",
                    main_gbps=7.0, alt_gbps=3.0,
                    mount_validator=detach_on_third_check,
                    host_free_provider=_huge_host_free,
                )
            self.assertEqual(list(stripe.iterdir()), [])
            self.assertEqual([path.name for path in detached.iterdir()], [LOCK_NAME])

    def test_manifest_is_rolled_back_if_index_publication_fails(self):
        with (tempfile.TemporaryDirectory() as temporary,
              tempfile.TemporaryDirectory(dir="/dev/shm") as stripe_temporary):
            root = pathlib.Path(temporary)
            source, index = _write_fixture(root)
            stripe = pathlib.Path(stripe_temporary)
            output = root / "striped.index"
            manifest = root / "stripe.manifest.json"
            repack(
                source, index, stripe, output, manifest_path=manifest,
                main_gbps=7.0, alt_gbps=3.0,
                mount_validator=_noop_mount_validator,
                host_free_provider=_huge_host_free,
            )
            old_index = _digest(output)
            old_manifest = _digest(manifest)
            old_files = sorted(path.name for path in stripe.iterdir())
            with self.assertRaisesRegex(RuntimeError, "index publication"):
                repack(
                    source, index, stripe, output, manifest_path=manifest,
                    main_gbps=7.0, alt_gbps=3.0, seed=30,
                    mount_validator=_noop_mount_validator,
                    host_free_provider=_huge_host_free,
                    force=True, fault_before_index_publish=True,
                )
            self.assertEqual(_digest(output), old_index)
            self.assertEqual(_digest(manifest), old_manifest)
            self.assertEqual(sorted(path.name for path in stripe.iterdir()), old_files)
            self.assertFalse((root / "stripe.manifest.json.rollback").exists())

    def test_manifest_tamper_is_detected(self):
        with (tempfile.TemporaryDirectory() as temporary,
              tempfile.TemporaryDirectory(dir="/dev/shm") as stripe_temporary):
            root = pathlib.Path(temporary)
            source, index = _write_fixture(root)
            stripe = pathlib.Path(stripe_temporary)
            output = root / "striped.index"
            manifest = root / "stripe.manifest.json"
            repack(
                source, index, stripe, output, manifest_path=manifest,
                main_gbps=7.0, alt_gbps=3.0,
                mount_validator=_noop_mount_validator,
                host_free_provider=_huge_host_free,
            )
            payload = json.loads(manifest.read_text())
            payload["source_index_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "source index digest"):
                verify(
                    index, output, source, stripe, manifest_path=manifest,
                    mount_validator=_noop_mount_validator,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
