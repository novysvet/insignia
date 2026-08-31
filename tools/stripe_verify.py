#!/usr/bin/env python3
"""Fully verify GLM stripe placement, provenance, structure, and byte parity."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import random
from collections.abc import Callable

from stripe_mount import StripeMountError, validate_stripe_mount
from stripe_repack import (
    MEMBER_SUFFIX,
    PLANNER_ALGORITHM,
    PROJ_ORDER,
    load_index,
    normalized_weights_sha256,
    plan_alt_records,
)


@dataclasses.dataclass(frozen=True)
class VerifyResult:
    selected_records: int
    checked_records: int
    checked_tensors: int
    checked_bytes: int


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sha256_at(directory_fd: int, name: str) -> str:
    digest = hashlib.sha256()
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory_fd)
    try:
        while block := os.read(fd, 8 << 20):
            digest.update(block)
    finally:
        os.close(fd)
    return digest.hexdigest()


def _expert_key(name: str) -> tuple[int, int] | None:
    try:
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        expert = int(name.split(".mlp.experts.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None
    return layer, expert


def _member_names(layer: int, expert: int) -> list[str]:
    stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    return [stem + projection + suffix
            for projection in PROJ_ORDER for suffix in MEMBER_SUFFIX]


def verify(
    original_index: pathlib.Path,
    striped_index: pathlib.Path,
    source_dir: pathlib.Path,
    stripe_dir: pathlib.Path,
    *,
    manifest_path: pathlib.Path | None = None,
    sample_records: int | None = 0,
    seed: int = 7,
    hash_shards: bool = True,
    expected_label: str | None = "stripe",
    expected_uuid: str | None = None,
    mount_validator: Callable[..., object] = validate_stripe_mount,
) -> VerifyResult:
    original_index = pathlib.Path(original_index)
    striped_index = pathlib.Path(striped_index)
    source_dir = pathlib.Path(source_dir)
    stripe_dir = pathlib.Path(stripe_dir)
    manifest_path = (striped_index.with_suffix(striped_index.suffix + ".stripe-manifest.json")
                     if manifest_path is None else pathlib.Path(manifest_path))
    mount_validator(
        source_dir, stripe_dir, expected_label=expected_label,
        expected_uuid=expected_uuid, expected_fs="ext4", min_free_bytes=0,
    )
    stripe_directory_fd = os.open(
        stripe_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    anchor = os.fstat(stripe_directory_fd)

    def validate_anchor() -> None:
        mount_validator(
            source_dir, stripe_dir, expected_label=expected_label,
            expected_uuid=expected_uuid, expected_fs="ext4", min_free_bytes=0,
        )
        path_stat = os.stat(stripe_dir, follow_symlinks=False)
        fd_stat = os.fstat(stripe_directory_fd)
        if ((fd_stat.st_dev, fd_stat.st_ino) != (anchor.st_dev, anchor.st_ino) or
                (path_stat.st_dev, path_stat.st_ino) != (anchor.st_dev, anchor.st_ino)):
            raise StripeMountError(f"stripe mount identity changed during verification: {stripe_dir}")

    validate_anchor()

    original_head, original_shards, original_entries = load_index(original_index)
    striped_head, striped_shards, striped_entries = load_index(striped_index)
    if original_head[:2] != striped_head[:2] or original_head[3:] != striped_head[3:]:
        raise ValueError("source and stripe index geometry differ")
    if striped_head[2] <= original_head[2]:
        raise ValueError("stripe index has no appended shards")
    original_count = len(original_shards)
    if striped_shards[:original_count] != original_shards:
        raise ValueError("stripe index changed the authoritative source shard table")
    appended_names = [name for name, _ in striped_shards[original_count:]]
    if len(set(appended_names)) != len(appended_names):
        raise ValueError("stripe index contains duplicate appended shard names")
    for name in appended_names:
        pure = pathlib.PurePosixPath(name)
        if pure.is_absolute() or len(pure.parts) != 1 or pure.name != name:
            raise ValueError(f"unsafe stripe shard name: {name!r}")
    original_by_name = {entry[0]: entry for entry in original_entries}
    striped_by_name = {entry[0]: entry for entry in striped_entries}
    if original_by_name.keys() != striped_by_name.keys():
        raise ValueError("source and stripe tensor sets differ")

    selected: set[tuple[int, int]] = set()
    changed_names: set[str] = set()
    for name, source in original_by_name.items():
        target = striped_by_name[name]
        if (source[1], source[2], source[6]) != (target[1], target[2], target[6]):
            raise ValueError(f"tensor metadata changed: {name}")
        if source[4] != target[4]:
            raise ValueError(f"tensor flags changed: {name}")
        source_location = (source[3], source[4], source[5], source[6])
        target_location = (target[3], target[4], target[5], target[6])
        if source_location == target_location:
            continue
        key = _expert_key(name)
        if key is None:
            raise ValueError(f"non-expert tensor was remapped: {name}")
        if target[3] < original_count:
            raise ValueError(f"remapped tensor does not use an appended shard: {name}")
        selected.add(key)
        changed_names.add(name)

    if not selected:
        raise ValueError("stripe index remaps no expert records")
    expected_changed: set[str] = set()
    for layer, expert in selected:
        names = _member_names(layer, expert)
        expected_changed.update(names)
        shards = set()
        prior_end = None
        for name in names:
            if name not in changed_names:
                raise ValueError(f"partially remapped expert record: {layer}/{expert}: {name}")
            entry = striped_by_name[name]
            shards.add(entry[3])
            if entry[3] >= len(striped_shards) or entry[5] + entry[6] > striped_shards[entry[3]][1]:
                raise ValueError(f"out-of-range stripe tensor: {name}")
            if prior_end is not None:
                gap = entry[5] - prior_end
                if gap < 0 or gap > 16:
                    raise ValueError(f"non-compact stripe record: {layer}/{expert}")
            prior_end = entry[5] + entry[6]
        if len(shards) != 1:
            raise ValueError(f"expert record spans stripe shards: {layer}/{expert}")
        if striped_by_name[names[0]][5] % 4096:
            raise ValueError(f"stripe record is not 4096-byte aligned: {layer}/{expert}")
    if changed_names != expected_changed:
        unexpected = sorted(changed_names - expected_changed)
        raise ValueError(f"unexpected remapped tensors: {unexpected[:3]}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "IG53STRIPE1":
        raise ValueError("bad stripe manifest format")
    if manifest.get("source_index_sha256") != _sha256(original_index):
        raise ValueError("source index digest does not match manifest")
    if manifest.get("stripe_index_sha256") != _sha256(striped_index):
        raise ValueError("stripe index digest does not match manifest")
    manifest_records = manifest.get("records", [])
    manifest_keys = {(int(record["layer"]), int(record["expert"]))
                     for record in manifest_records}
    if len(manifest_keys) != len(manifest_records):
        raise ValueError("manifest contains duplicate expert records")
    if manifest_keys != selected:
        raise ValueError("manifest record set does not match stripe index")
    selection_sha = hashlib.sha256(
        "".join(f"{layer}\t{expert}\n" for layer, expert in sorted(selected)).encode()
    ).hexdigest()
    if manifest.get("selection_sha256") != selection_sha:
        raise ValueError("manifest selection digest does not match stripe index")

    planner = manifest.get("planner", {})
    if planner.get("algorithm") != PLANNER_ALGORITHM:
        raise ValueError("unsupported or missing stripe planner algorithm")
    weight_payload = manifest.get("weights", {})
    weight_entries = weight_payload.get("entries", [])
    weights: dict[tuple[int, int], float] = {}
    for row in weight_entries:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("malformed normalized weight entry")
        key = int(row[0]), int(row[1])
        if key in weights:
            raise ValueError(f"duplicate normalized weight entry: {key}")
        weights[key] = float(row[2])
    if weight_payload.get("normalized_sha256") != normalized_weights_sha256(weights):
        raise ValueError("normalized route-weight digest mismatch")
    layers = sorted({key[0] for name in original_by_name if (key := _expert_key(name)) is not None})
    experts = int(original_head[7])
    valid_keys = {(layer, expert) for layer in layers for expert in range(experts)}
    unknown = sorted(set(weights) - valid_keys)
    if unknown:
        raise ValueError(f"manifest weights contain unknown keys: {unknown[:3]}")
    regenerated = plan_alt_records(
        layers,
        experts,
        main_gbps=float(planner["main_gbps"]),
        alt_gbps=float(planner["alt_gbps"]),
        weights=weights or None,
        weight_floor=float(planner["weight_floor"]),
        seed=int(planner["seed"]),
    )
    if regenerated.selected != selected:
        raise ValueError("stripe placement does not match regenerated route-weighted plan")
    worst_imbalance = max(
        abs((total - alt) / regenerated.main_gbps - alt / regenerated.alt_gbps) /
        max((total - alt) / regenerated.main_gbps, alt / regenerated.alt_gbps, 1e-30)
        for total, alt, _ in regenerated.per_layer.values()
    )
    if worst_imbalance > float(planner["max_service_imbalance"]) + 1e-12:
        raise ValueError("regenerated placement exceeds the service-imbalance gate")
    for record in manifest_records:
        layer, expert = int(record["layer"]), int(record["expert"])
        names = _member_names(layer, expert)
        entries = [striped_by_name[name] for name in names]
        stripe_shard = entries[0][3] - original_count
        record_begin = entries[0][5]
        record_end = max(entry[5] + entry[6] for entry in entries)
        declared_end = int(record["offset"]) + int(record["bytes"])
        if (stripe_shard != int(record["shard"]) or
                record_begin != int(record["offset"]) or
                record_end > declared_end or declared_end - record_end >= 4096):
            raise ValueError(f"manifest location mismatch for expert {layer}/{expert}")
    appended = striped_shards[original_count:]
    manifest_shards = manifest.get("shards", [])
    if len(appended) != len(manifest_shards):
        raise ValueError("manifest shard count does not match stripe index")
    for index, ((name, size), declared) in enumerate(zip(appended, manifest_shards)):
        if name != declared.get("name") or size != int(declared.get("size", -1)):
            raise ValueError(f"manifest metadata mismatch for stripe shard {index}")
        shard_stat = os.stat(name, dir_fd=stripe_directory_fd, follow_symlinks=False)
        if shard_stat.st_size != size:
            raise ValueError(f"stripe shard size mismatch: {stripe_dir / name}")
        if hash_shards and _sha256_at(stripe_directory_fd, name) != declared.get("sha256"):
            raise ValueError(f"stripe shard digest mismatch: {stripe_dir / name}")

    ordered = sorted(selected)
    if sample_records is None or sample_records <= 0 or sample_records >= len(ordered):
        checked = ordered
    else:
        checked = sorted(random.Random(seed).sample(ordered, sample_records))

    source_fds: dict[int, int] = {}
    stripe_fds: dict[int, int] = {}
    checked_bytes = checked_tensors = 0
    try:
        for layer, expert in checked:
            for name in _member_names(layer, expert):
                source = original_by_name[name]
                target = striped_by_name[name]
                source_fd = source_fds.get(source[3])
                if source_fd is None:
                    source_fd = os.open(source_dir / original_shards[source[3]][0], os.O_RDONLY)
                    source_fds[source[3]] = source_fd
                target_fd = stripe_fds.get(target[3])
                if target_fd is None:
                    target_fd = os.open(
                        striped_shards[target[3]][0], os.O_RDONLY | os.O_CLOEXEC,
                        dir_fd=stripe_directory_fd,
                    )
                    stripe_fds[target[3]] = target_fd
                reference = os.pread(source_fd, source[6], source[5])
                candidate = os.pread(target_fd, target[6], target[5])
                if len(reference) != source[6] or len(candidate) != target[6]:
                    raise OSError(f"short parity read: {name}")
                if reference != candidate:
                    raise ValueError(f"stripe byte mismatch: {name}")
                checked_bytes += len(reference)
                checked_tensors += 1
    finally:
        for fd in source_fds.values():
            os.close(fd)
        for fd in stripe_fds.values():
            os.close(fd)
    validate_anchor()
    os.close(stripe_directory_fd)
    return VerifyResult(len(selected), len(checked), checked_tensors, checked_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_index", type=pathlib.Path)
    parser.add_argument("striped_index", type=pathlib.Path)
    parser.add_argument("source_dir", type=pathlib.Path)
    parser.add_argument("stripe_dir", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--sample-records", type=int, default=0,
                        help="verify only N deterministic records; 0 verifies all (default: 0)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-shard-hash", action="store_true",
                        help="quick diagnostic only: skip full stripe-shard SHA-256")
    parser.add_argument("--expected-label", default="stripe")
    parser.add_argument("--expected-uuid")
    args = parser.parse_args()
    try:
        result = verify(
            args.original_index, args.striped_index, args.source_dir, args.stripe_dir,
            manifest_path=args.manifest, sample_records=args.sample_records,
            seed=args.seed, hash_shards=not args.skip_shard_hash,
            expected_label=args.expected_label or None, expected_uuid=args.expected_uuid,
        )
    except (KeyError, OSError, TypeError, ValueError,
            StripeMountError, json.JSONDecodeError) as error:
        parser.exit(1, f"stripe verify FAILED: {error}\n")
    qualifier = "OK" if args.sample_records <= 0 and not args.skip_shard_hash else "QUICK SAMPLE ONLY"
    print(
        f"stripe verify {qualifier}: {result.checked_records}/{result.selected_records} records, "
        f"{result.checked_tensors} tensors, {result.checked_bytes / 2**20:.2f} MiB byte-exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
