#!/usr/bin/env python3
"""Build a fail-closed, route-weighted GLM expert stripe on the E: VHDX.

The original compact store and index remain authoritative on C:.  Selected
expert records are copied, byte-exactly, into versioned shards on the verified
`/stripe` mount.  A companion index remaps only those records.  Publication is
atomic at the file level and the index is published last.

Weights are optional whitespace/CSV/TSV rows of ``layer expert miss_weight``.
Without them the deterministic planner balances record count.  With them it
balances expected miss traffic so ``C_bytes / C_GBps ~= E_bytes / E_GBps``.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import pathlib
import shutil
import stat
import struct
import sys
import time
from collections.abc import Callable, Mapping

from stripe_mount import StripeMountError, validate_stripe_mount


MAGIC = b"IGLMIDX1"
PROJ_ORDER = ("down_proj", "gate_proj", "up_proj")
MEMBER_SUFFIX = (".weight", ".weight_scale", ".weight_scale_2")
SHARD_BYTES = (4 << 30) - (128 << 20)
READ_CHUNK = 32 << 20
ALIGNMENT = 4096
PLAN_QUANTA = 1 << 18
GUEST_FREE_MARGIN = 1 << 30
HOST_FREE_MARGIN = 8 << 30
LOCK_NAME = ".insignia-stripe-repack.lock"
PLANNER_ALGORITHM = "quantized-subset-dp-v1"


@dataclasses.dataclass(frozen=True)
class PlanSummary:
    selected: frozenset[tuple[int, int]]
    main_gbps: float
    alt_gbps: float
    target_alt_fraction: float
    total_weight: float
    alt_weight: float
    per_layer: dict[int, tuple[float, float, int]]


@dataclasses.dataclass(frozen=True)
class RepackResult:
    index: pathlib.Path
    manifest: pathlib.Path
    generation: str
    records: int
    bytes_written: int
    shards: int


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & -alignment


def load_index(path: pathlib.Path):
    data = path.read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"bad GLM index magic: {path}")
    header_format = "<11IQ"
    header_size = struct.calcsize(header_format)
    if len(data) < 8 + header_size:
        raise ValueError(f"truncated GLM index header: {path}")
    head = struct.unpack_from(header_format, data, 8)
    off = 8 + header_size
    shards = []
    for _ in range(head[2]):
        if off + struct.calcsize("<HQ") > len(data):
            raise ValueError("truncated shard table")
        name_length, size = struct.unpack_from("<HQ", data, off)
        off += struct.calcsize("<HQ")
        name = data[off:off + name_length].decode()
        off += name_length
        shards.append([name, size])
    entries = []
    entry_header = struct.calcsize("<HBBHHQQ")
    for _ in range(head[3]):
        if off + entry_header > len(data):
            raise ValueError("truncated tensor table")
        name_length, dtype, ndim, shard, flags, absolute, length = struct.unpack_from(
            "<HBBHHQQ", data, off
        )
        off += entry_header
        name = data[off:off + name_length].decode()
        off += name_length
        shape_bytes = 4 * ndim
        if off + shape_bytes > len(data):
            raise ValueError("truncated tensor shape")
        shape = struct.unpack_from("<" + "I" * ndim, data, off)
        off += shape_bytes
        entries.append([name, dtype, shape, shard, flags, absolute, length])
    if off != len(data):
        raise ValueError(f"trailing bytes in GLM index: {len(data) - off}")
    return head, shards, entries


def _mix64(value: int) -> int:
    value &= (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _rank(layer: int, expert: int, seed: int) -> int:
    return _mix64(((layer & 0xFFFFFFFF) << 32) ^ expert ^ seed)


def _closest_subset(layer: int, weighted: list[tuple[int, float]], target: float,
                    seed: int) -> set[int]:
    """Deterministic FPTAS-style subset sum followed by exact local descent.

    A greedy selector gets trapped by ordinary route histograms (for example
    target=29.7 and weights 39,20,10 chooses 39 instead of 20+10).  At only 288
    experts, a 2**18-bin reachability bitset is cheap and bounds accumulated
    quantization error to roughly 0.06% of layer traffic.  History bitsets make
    reconstruction deterministic without retaining millions of Python sets.
    """
    positive = sorted(
        ((expert, weight) for expert, weight in weighted if weight > 0),
        key=lambda item: _rank(layer, item[0], seed),
    )
    if not positive or target <= 0:
        return set()
    total = sum(weight for _, weight in positive)
    quantized = [
        (expert, weight, max(1, round(weight * PLAN_QUANTA / total)))
        for expert, weight in positive
    ]
    reachable = 1
    history = [reachable]
    for _, _, quantum in quantized:
        reachable |= reachable << quantum
        history.append(reachable)
    quantized_total = sum(quantum for _, _, quantum in quantized)
    quantized_target = round(target * quantized_total / total)
    chosen_sum = None
    for distance in range(quantized_total + 1):
        # Prefer an undershoot on exact ties.  The exact-float descent below
        # removes the residual quantization error.
        for candidate in (quantized_target - distance, quantized_target + distance):
            if 0 < candidate < quantized_total and (reachable >> candidate) & 1:
                chosen_sum = candidate
                break
        if chosen_sum is not None:
            break
    if chosen_sum is None:
        raise ValueError(f"cannot construct a non-trivial stripe subset for layer {layer}")
    selected: set[int] = set()
    remaining = chosen_sum
    for index in range(len(quantized) - 1, -1, -1):
        expert, _, quantum = quantized[index]
        if (history[index] >> remaining) & 1:
            continue
        selected.add(expert)
        remaining -= quantum
    if remaining:
        raise AssertionError("stripe subset reconstruction failed")

    weights = dict(weighted)
    current = sum(weights[expert] for expert in selected)
    # Polish the quantized answer against the true floating-point weights.
    for _ in range(8):
        base_error = abs(current - target)
        best = None
        best_error = base_error
        for expert, weight in weighted:
            candidate = current - weight if expert in selected else current + weight
            error = abs(candidate - target)
            key = (error, 0, _rank(layer, expert, seed))
            if error + 1e-12 < best_error or (best is not None and error == best_error and key < best[0]):
                best_error = error
                best = (key, "toggle", expert, None, candidate)
        inside = sorted(selected, key=lambda expert: _rank(layer, expert, seed))
        outside = sorted((expert for expert, _ in weighted if expert not in selected),
                         key=lambda expert: _rank(layer, expert, seed))
        for old in inside:
            for new in outside:
                candidate = current - weights[old] + weights[new]
                error = abs(candidate - target)
                key = (error, 1, _rank(layer, old, seed), _rank(layer, new, seed))
                if error + 1e-12 < best_error or (best is not None and error == best_error and key < best[0]):
                    best_error = error
                    best = (key, "swap", old, new, candidate)
        if best is None or best_error + 1e-12 >= base_error:
            break
        _, action, first, second, current = best
        if action == "toggle":
            if first in selected:
                selected.remove(first)
            else:
                selected.add(first)
        else:
            selected.remove(first)
            selected.add(second)
    return selected


def _service_imbalance(total_weight: float, alt_weight: float,
                       main_gbps: float, alt_gbps: float) -> float:
    main_time = (total_weight - alt_weight) / main_gbps
    alt_time = alt_weight / alt_gbps
    return abs(main_time - alt_time) / max(main_time, alt_time, 1e-30)


def normalized_weights_sha256(weights: Mapping[tuple[int, int], float] | None) -> str:
    digest = hashlib.sha256()
    for (layer, expert), weight in sorted((weights or {}).items()):
        digest.update(f"{layer}\t{expert}\t{float(weight).hex()}\n".encode())
    return digest.hexdigest()


def plan_alt_records(
    sparse_layers: list[int] | tuple[int, ...],
    experts: int,
    *,
    main_gbps: float = 5.94,
    alt_gbps: float = 2.58,
    weights: Mapping[tuple[int, int], float] | None = None,
    weight_floor: float = 1.0,
    seed: int = 0x49534753,
) -> PlanSummary:
    if experts < 2:
        raise ValueError("striping requires at least two experts")
    if not math.isfinite(main_gbps) or not math.isfinite(alt_gbps) or main_gbps <= 0 or alt_gbps <= 0:
        raise ValueError("drive bandwidths must be finite and positive")
    if weight_floor < 0 or not math.isfinite(weight_floor):
        raise ValueError("weight_floor must be finite and non-negative")
    target_fraction = alt_gbps / (main_gbps + alt_gbps)
    selected: set[tuple[int, int]] = set()
    per_layer: dict[int, tuple[float, float, int]] = {}
    total_weight = alt_weight = 0.0
    for layer in sorted(set(sparse_layers)):
        weighted = []
        for expert in range(experts):
            observed = 0.0 if weights is None else float(weights.get((layer, expert), 0.0))
            if observed < 0 or not math.isfinite(observed):
                raise ValueError(f"invalid weight for layer {layer} expert {expert}: {observed}")
            weight = observed + (1.0 if weights is None else weight_floor)
            weighted.append((expert, weight))
        layer_total = sum(weight for _, weight in weighted)
        if layer_total <= 0:
            weighted = [(expert, 1.0) for expert in range(experts)]
            layer_total = float(experts)
        layer_selected = _closest_subset(layer, weighted, layer_total * target_fraction, seed)
        layer_alt = sum(weight for expert, weight in weighted if expert in layer_selected)
        selected.update((layer, expert) for expert in layer_selected)
        total_weight += layer_total
        alt_weight += layer_alt
        per_layer[layer] = (layer_total, layer_alt, len(layer_selected))
    return PlanSummary(frozenset(selected), main_gbps, alt_gbps, target_fraction,
                       total_weight, alt_weight, per_layer)


def load_weights(path: pathlib.Path) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.replace(",", " ").split()
        if fields[0].lower() in {"layer", "layer_id"}:
            continue
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected layer expert weight")
        layer, expert = int(fields[0]), int(fields[1])
        weight = float(fields[2])
        if weight < 0 or not math.isfinite(weight):
            raise ValueError(f"{path}:{line_number}: weight must be finite and non-negative")
        result[(layer, expert)] = result.get((layer, expert), 0.0) + weight
    if not result:
        raise ValueError(f"no weights found in {path}")
    return result


def _record_names(layer: int, expert: int) -> list[str]:
    stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    return [stem + projection + suffix
            for projection in PROJ_ORDER for suffix in MEMBER_SUFFIX]


def _estimated_bytes(plan: PlanSummary, by_name: Mapping[str, list]) -> int:
    total = 0
    for layer, expert in plan.selected:
        cursor = 0
        for index, name in enumerate(_record_names(layer, expert)):
            if index in (3, 6):
                cursor = align(cursor, 16)
            cursor += int(by_name[name][6])
        total += align(cursor)
    return total


def _sha256_file(path: pathlib.Path) -> str:
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


def host_free_bytes(path: pathlib.Path) -> int:
    return shutil.disk_usage(path).free


def _resolved_output(path: pathlib.Path) -> pathlib.Path:
    return path.resolve(strict=False)


def _ensure_distinct_paths(named_paths: Mapping[str, pathlib.Path]) -> None:
    seen: dict[pathlib.Path, str] = {}
    for name, path in named_paths.items():
        resolved = _resolved_output(path)
        prior = seen.get(resolved)
        if prior is not None:
            raise ValueError(f"{name} aliases {prior}: {resolved}")
        for prior_path, prior_name in seen.items():
            if path.exists() and prior_path.exists() and os.path.samefile(path, prior_path):
                raise ValueError(f"{name} aliases {prior_name}: {resolved}")
        seen[resolved] = name


def repack(
    src_dir: pathlib.Path,
    src_index: pathlib.Path,
    dst_dir: pathlib.Path,
    out_index: pathlib.Path,
    *,
    manifest_path: pathlib.Path | None = None,
    main_gbps: float = 5.94,
    alt_gbps: float = 2.58,
    weights: Mapping[tuple[int, int], float] | None = None,
    weight_floor: float = 1.0,
    seed: int = 0x49534753,
    expected_label: str | None = "stripe",
    expected_uuid: str | None = None,
    host_root: pathlib.Path = pathlib.Path("/mnt/e"),
    host_margin_bytes: int = HOST_FREE_MARGIN,
    host_free_provider: Callable[[pathlib.Path], int] = host_free_bytes,
    max_service_imbalance: float = 0.02,
    force: bool = False,
    shard_bytes: int = SHARD_BYTES,
    mount_validator: Callable[..., object] = validate_stripe_mount,
    fault_after_records: int | None = None,
    fault_before_index_publish: bool = False,
) -> RepackResult:
    src_dir = pathlib.Path(src_dir)
    src_index = pathlib.Path(src_index)
    dst_dir = pathlib.Path(dst_dir)
    out_index = pathlib.Path(out_index)
    manifest_path = (out_index.with_suffix(out_index.suffix + ".stripe-manifest.json")
                     if manifest_path is None else pathlib.Path(manifest_path))
    host_root = pathlib.Path(host_root)
    if shard_bytes < ALIGNMENT:
        raise ValueError("shard_bytes is too small")
    if host_margin_bytes < 0:
        raise ValueError("host_margin_bytes must be non-negative")
    if not math.isfinite(max_service_imbalance) or not 0 <= max_service_imbalance < 1:
        raise ValueError("max_service_imbalance must be finite and in [0, 1)")
    index_partial = out_index.with_name(out_index.name + ".partial")
    manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
    manifest_backup = manifest_path.with_name(manifest_path.name + ".rollback")
    _ensure_distinct_paths({
        "source index": src_index,
        "output index": out_index,
        "manifest": manifest_path,
        "output index partial": index_partial,
        "manifest partial": manifest_partial,
        "manifest rollback": manifest_backup,
    })
    if manifest_backup.exists():
        raise FileExistsError(
            f"stale rollback manifest requires inspection before repacking: {manifest_backup}"
        )
    if not force:
        for output in (out_index, manifest_path):
            if output.exists():
                raise FileExistsError(f"output exists: {output} (pass --force to replace)")

    head, shards, entries = load_index(src_index)
    (version, flags0, nshards, nentries, hidden, layers, vocab,
     experts, topk, moe_inter, hc, total) = head
    by_name = {entry[0]: entry for entry in entries}
    if len(by_name) != len(entries):
        raise ValueError("duplicate tensor name in source index")
    sparse_layers = [
        layer for layer in range(layers)
        if f"model.language_model.layers.{layer}.mlp.experts.0.down_proj.weight" in by_name
    ]
    if not sparse_layers:
        raise ValueError("no sparse expert layers found")
    valid_keys = {(layer, expert) for layer in sparse_layers for expert in range(experts)}
    unknown_weights = sorted(set((weights or {}).keys()) - valid_keys)
    if unknown_weights:
        raise ValueError(f"weights contain unknown layer/expert keys: {unknown_weights[:3]}")
    source_files = {
        (src_dir / name).resolve(strict=False): f"source shard {name}" for name, _ in shards
    }
    for output_name, output_path in (("output index", out_index), ("manifest", manifest_path),
                                     ("output index partial", index_partial),
                                     ("manifest partial", manifest_partial)):
        collision = source_files.get(output_path.resolve(strict=False))
        if collision:
            raise ValueError(f"{output_name} aliases {collision}: {output_path}")
    plan = plan_alt_records(
        sparse_layers, experts, main_gbps=main_gbps, alt_gbps=alt_gbps,
        weights=weights, weight_floor=weight_floor, seed=seed,
    )
    worst_layer_imbalance = max(
        _service_imbalance(layer_total, layer_alt, main_gbps, alt_gbps)
        for layer_total, layer_alt, _ in plan.per_layer.values()
    )
    aggregate_imbalance = _service_imbalance(
        plan.total_weight, plan.alt_weight, main_gbps, alt_gbps,
    )
    if worst_layer_imbalance > max_service_imbalance:
        raise ValueError(
            f"planner service imbalance {100 * worst_layer_imbalance:.3f}% exceeds "
            f"the {100 * max_service_imbalance:.3f}% gate"
        )
    estimated = _estimated_bytes(plan, by_name)

    # This check is intentionally before mkdir/open/unlink: an unmounted
    # `/stripe` must not cause even one byte to land in the C: root VHDX.
    mount_validator(
        src_dir, dst_dir, expected_label=expected_label,
        expected_uuid=expected_uuid, expected_fs="ext4",
        min_free_bytes=estimated + GUEST_FREE_MARGIN,
    )
    initial_host_free = int(host_free_provider(host_root))
    if initial_host_free < estimated + host_margin_bytes:
        raise StripeMountError(
            f"host volume {host_root} has {initial_host_free / 2**30:.2f} GiB free; "
            f"the dynamic VHDX needs {estimated / 2**30:.2f} GiB plus "
            f"a {host_margin_bytes / 2**30:.2f} GiB reserve"
        )

    source_index_sha = _sha256_file(src_index)
    generation_payload = json.dumps({
        "format": "IG53STRIPE1-layout1",
        "planner_algorithm": PLANNER_ALGORITHM,
        "source_index_sha256": source_index_sha,
        "main_gbps": main_gbps,
        "alt_gbps": alt_gbps,
        "weight_floor": weight_floor,
        "seed": seed,
        "shard_bytes": shard_bytes,
        "weights_sha256": normalized_weights_sha256(weights),
        "selected": sorted([list(key) for key in plan.selected]),
    }, sort_keys=True, separators=(",", ":")).encode()
    generation = hashlib.sha256(generation_payload).hexdigest()[:16]
    selection_sha = hashlib.sha256(
        "".join(f"{layer}\t{expert}\n" for layer, expert in sorted(plan.selected)).encode()
    ).hexdigest()

    fds: dict[int, int] = {}
    stripe_partial_names: list[str] = []
    metadata_partial_paths: list[pathlib.Path] = []
    final_names: list[str] = []
    preexisting_finals: set[str] = set()
    published_index = False
    published_manifest = False
    backed_up_manifest = False
    stripe_tensors: dict[str, tuple[int, int]] = {}
    record_manifest: list[dict[str, int]] = []
    shard_manifest: list[dict[str, object]] = []
    done_bytes = 0
    started = time.perf_counter()
    current = None
    current_name = ""
    current_partial = ""
    current_final = ""
    current_offset = 0
    current_digest = None
    destination_fd = -1
    lock_fd = -1
    anchor_identity: tuple[int, int] | None = None

    def validate_anchor(min_free_bytes: int = GUEST_FREE_MARGIN) -> None:
        mount_validator(
            src_dir, dst_dir, expected_label=expected_label,
            expected_uuid=expected_uuid, expected_fs="ext4",
            min_free_bytes=min_free_bytes,
        )
        fd_stat = os.fstat(destination_fd)
        path_stat = os.stat(dst_dir, follow_symlinks=False)
        current_identity = (fd_stat.st_dev, fd_stat.st_ino)
        if current_identity != anchor_identity or (
            path_stat.st_dev, path_stat.st_ino
        ) != anchor_identity:
            raise StripeMountError(
                f"stripe mount identity changed while repacking: {dst_dir}"
            )
        if fd_stat.st_dev == os.stat(src_dir).st_dev:
            raise StripeMountError("stripe destination and source have the same st_dev")

    def write_payload(payload: bytes) -> None:
        nonlocal current_offset, done_bytes
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = current.write(view[written:])
            if not count:
                raise OSError("short stripe write")
            current_digest.update(view[written:written + count])
            current_offset += count
            done_bytes += count
            written += count

    def write_zeroes(count: int) -> None:
        zeroes = b"\0" * min(ALIGNMENT, count)
        while count:
            take = min(count, len(zeroes))
            write_payload(zeroes[:take])
            count -= take

    def close_current() -> None:
        nonlocal current, current_partial, current_final
        if current is None:
            return
        current.flush()
        os.fsync(current.fileno())
        current.close()
        shard_manifest.append({
            "name": current_name,
            "size": current_offset,
            "sha256": current_digest.hexdigest(),
        })
        current = None

    def open_next() -> None:
        nonlocal current, current_name, current_partial, current_final
        nonlocal current_offset, current_digest
        close_current()
        validate_anchor()
        ordinal = len(shard_manifest) + 1
        current_name = f"stripe-{generation}-{ordinal:05d}.bin"
        current_final = current_name
        current_partial = current_name + ".partial"
        try:
            os.unlink(current_partial, dir_fd=destination_fd)
        except FileNotFoundError:
            pass
        try:
            final_stat = os.stat(current_final, dir_fd=destination_fd, follow_symlinks=False)
        except FileNotFoundError:
            final_stat = None
        if final_stat is not None and not stat.S_ISREG(final_stat.st_mode):
            raise ValueError(f"stripe shard is not a regular file: {dst_dir / current_final}")
        if final_stat is not None and not force:
            raise FileExistsError(f"stripe shard exists: {dst_dir / current_final}")
        if final_stat is not None:
            preexisting_finals.add(current_final)
        raw_fd = os.open(
            current_partial,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=destination_fd,
        )
        current = os.fdopen(raw_fd, "wb", buffering=0)
        stripe_partial_names.append(current_partial)
        final_names.append(current_final)
        current_offset = 0
        current_digest = hashlib.sha256()

    try:
        destination_fd = os.open(
            dst_dir,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        destination_stat = os.fstat(destination_fd)
        anchor_identity = (destination_stat.st_dev, destination_stat.st_ino)
        validate_anchor(min_free_bytes=estimated + GUEST_FREE_MARGIN)
        lock_fd = os.open(
            LOCK_NAME, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
            0o600, dir_fd=destination_fd,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another stripe repack owns {dst_dir / LOCK_NAME}") from error

        for shard_id, (name, expected_size) in enumerate(shards):
            path = src_dir / name
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise ValueError(f"size mismatch {path}: {actual_size} != {expected_size}")
            fds[shard_id] = os.open(path, os.O_RDONLY | os.O_CLOEXEC)

        open_next()
        for record_number, (layer, expert) in enumerate(sorted(plan.selected), 1):
            names = _record_names(layer, expert)
            parts = []
            for name in names:
                if name not in by_name:
                    raise ValueError(f"missing expert tensor: {name}")
                entry = by_name[name]
                parts.append((entry[3], entry[5], entry[6]))
            simulated = current_offset
            for index, (_, _, length) in enumerate(parts):
                if index in (3, 6):
                    simulated = align(simulated, 16)
                simulated += length
            simulated = align(simulated)
            if current_offset and simulated > shard_bytes:
                open_next()

            record_start = current_offset
            member_offsets = []
            for index, (source_shard, source_offset, length) in enumerate(parts):
                if source_shard >= len(shards) or source_offset + length > shards[source_shard][1]:
                    raise ValueError(f"out-of-range source tensor: {names[index]}")
                if index in (3, 6):
                    write_zeroes((-current_offset) % 16)
                member_offsets.append(current_offset)
                remaining = length
                position = source_offset
                while remaining:
                    step = min(remaining, READ_CHUNK)
                    block = os.pread(fds[source_shard], step, position)
                    if len(block) != step:
                        raise OSError(f"short source read for {names[index]}: {len(block)} != {step}")
                    write_payload(block)
                    remaining -= step
                    position += step
            write_zeroes((-current_offset) % ALIGNMENT)
            stripe_shard = len(shard_manifest)
            for name, member_offset in zip(names, member_offsets):
                stripe_tensors[name] = (stripe_shard, member_offset)
            record_manifest.append({
                "layer": layer,
                "expert": expert,
                "shard": stripe_shard,
                "offset": record_start,
                "bytes": current_offset - record_start,
            })
            if fault_after_records is not None and record_number >= fault_after_records:
                raise RuntimeError("injected stripe-repack failure")
            if record_number == 1 or record_number % 200 == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"{record_number}/{len(plan.selected)} records, "
                    f"{done_bytes / 2**30:.1f} GiB, {done_bytes / elapsed / 2**20:.0f} MiB/s",
                    flush=True,
                )
        close_current()

        out_index.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_partial_paths.extend((index_partial, manifest_partial))
        for path in (index_partial, manifest_partial):
            if path.exists():
                path.unlink()

        with index_partial.open("xb", buffering=0) as output:
            output.write(struct.pack(
                "<8s11IQ", MAGIC, version, flags0, nshards + len(shard_manifest),
                nentries, hidden, layers, vocab, experts, topk, moe_inter, hc, total,
            ))
            for name, size in shards:
                encoded = name.encode()
                output.write(struct.pack("<HQ", len(encoded), size))
                output.write(encoded)
            for shard in shard_manifest:
                encoded = str(shard["name"]).encode()
                output.write(struct.pack("<HQ", len(encoded), int(shard["size"])))
                output.write(encoded)
            for name, dtype, shape, shard, flags, absolute, length in entries:
                if name in stripe_tensors:
                    stripe_shard, absolute = stripe_tensors[name]
                    shard = nshards + stripe_shard
                encoded = name.encode()
                output.write(struct.pack(
                    "<HBBHHQQ", len(encoded), dtype, len(shape), shard, flags, absolute, length,
                ))
                output.write(encoded)
                output.write(struct.pack("<" + "I" * len(shape), *shape))
            output.flush()
            os.fsync(output.fileno())
        index_sha = _sha256_file(index_partial)

        manifest = {
            "format": "IG53STRIPE1",
            "generation": generation,
            "source_index": str(src_index),
            "source_index_sha256": source_index_sha,
            "stripe_index": str(out_index),
            "stripe_index_sha256": index_sha,
            "planner": {
                "algorithm": PLANNER_ALGORITHM,
                "main_gbps": main_gbps,
                "alt_gbps": alt_gbps,
                "target_alt_fraction": plan.target_alt_fraction,
                "actual_alt_weight_fraction": plan.alt_weight / max(plan.total_weight, 1e-30),
                "weight_floor": weight_floor,
                "seed": seed,
                "aggregate_service_imbalance": aggregate_imbalance,
                "worst_layer_service_imbalance": worst_layer_imbalance,
                "max_service_imbalance": max_service_imbalance,
            },
            "weights": {
                "normalized_sha256": normalized_weights_sha256(weights),
                "entries": [[layer, expert, weight] for (layer, expert), weight
                            in sorted((weights or {}).items())],
            },
            "host_volume": {
                "path": str(host_root),
                "initial_free_bytes": initial_host_free,
                "reserve_bytes": host_margin_bytes,
            },
            "source_shards": nshards,
            "selection_sha256": selection_sha,
            "records": record_manifest,
            "shards": shard_manifest,
        }
        encoded_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        with manifest_partial.open("xb", buffering=0) as output:
            output.write(encoded_manifest)
            output.flush()
            os.fsync(output.fileno())

        validate_anchor()
        final_host_free = int(host_free_provider(host_root))
        if final_host_free < host_margin_bytes:
            raise StripeMountError(
                f"host volume {host_root} fell to {final_host_free / 2**30:.2f} GiB free; "
                f"refusing publication below the {host_margin_bytes / 2**30:.2f} GiB reserve"
            )

        # Versioned shard names make these renames harmless to the old index.
        for shard, partial, final in zip(shard_manifest, stripe_partial_names, final_names):
            if final in preexisting_finals:
                final_stat = os.stat(final, dir_fd=destination_fd, follow_symlinks=False)
                if (not stat.S_ISREG(final_stat.st_mode) or
                        final_stat.st_size != int(shard["size"]) or
                        _sha256_at(destination_fd, final) != shard["sha256"]):
                    raise ValueError(
                        f"existing generation shard is inconsistent: {dst_dir / final}; "
                        "refusing a non-atomic in-place repair"
                    )
                os.unlink(partial, dir_fd=destination_fd)
            else:
                os.replace(
                    partial, final,
                    src_dir_fd=destination_fd, dst_dir_fd=destination_fd,
                )
        # Make E:'s versioned directory entries durable before publishing any
        # C:-side metadata that can reference them.
        os.fsync(destination_fd)
        if manifest_path.exists():
            os.replace(manifest_path, manifest_backup)
            backed_up_manifest = True
        os.replace(manifest_partial, manifest_path)
        published_manifest = True
        directory_fd = os.open(manifest_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if fault_before_index_publish:
            raise RuntimeError("injected failure before stripe-index publication")
        os.replace(index_partial, out_index)  # authoritative publication, last
        published_index = True
        directory_fd = os.open(out_index.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if backed_up_manifest:
            try:
                manifest_backup.unlink()
            except OSError as error:
                print(f"warning: could not remove rollback manifest {manifest_backup}: {error}",
                      file=sys.stderr)
    except Exception:
        if current is not None:
            current.close()
        for name in stripe_partial_names:
            try:
                os.unlink(name, dir_fd=destination_fd)
            except FileNotFoundError:
                pass
        for path in metadata_partial_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if not published_index:
            if published_manifest:
                try:
                    manifest_path.unlink()
                except FileNotFoundError:
                    pass
            if backed_up_manifest and manifest_backup.exists():
                os.replace(manifest_backup, manifest_path)
            for name in final_names:
                if name in preexisting_finals:
                    continue
                try:
                    os.unlink(name, dir_fd=destination_fd)
                except FileNotFoundError:
                    pass
        raise
    finally:
        for fd in fds.values():
            os.close(fd)
        if lock_fd >= 0:
            os.close(lock_fd)
        if destination_fd >= 0:
            os.close(destination_fd)

    elapsed = max(time.perf_counter() - started, 1e-9)
    print(
        f"wrote {len(shard_manifest)} stripe shards, {len(plan.selected)} records, "
        f"{done_bytes / 2**30:.2f} GiB in {elapsed:.1f}s; "
        f"planned E traffic {100 * plan.alt_weight / max(plan.total_weight, 1e-30):.2f}%"
    )
    return RepackResult(out_index, manifest_path, generation, len(plan.selected),
                        done_bytes, len(shard_manifest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_store_dir", type=pathlib.Path)
    parser.add_argument("src_index", type=pathlib.Path)
    parser.add_argument("dst_dir", type=pathlib.Path)
    parser.add_argument("new_index", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--weights", type=pathlib.Path,
                        help="layer/expert/miss-weight TSV, CSV, or whitespace file")
    parser.add_argument("--main-gbps", type=float, default=5.94)
    parser.add_argument("--alt-gbps", type=float, default=2.58)
    parser.add_argument("--weight-floor", type=float, default=1.0)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x49534753)
    parser.add_argument("--expected-label", default="stripe")
    parser.add_argument("--expected-uuid")
    parser.add_argument("--host-root", type=pathlib.Path, default=pathlib.Path("/mnt/e"),
                        help="Windows backing volume as mounted in WSL (default: /mnt/e)")
    parser.add_argument("--host-margin-gib", type=float, default=HOST_FREE_MARGIN / 2**30,
                        help="free space kept on the dynamic VHDX backing volume")
    parser.add_argument("--max-service-imbalance", type=float, default=0.02,
                        help="maximum per-layer normalized C/E service-time mismatch")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = repack(
            args.src_store_dir, args.src_index, args.dst_dir, args.new_index,
            manifest_path=args.manifest,
            main_gbps=args.main_gbps, alt_gbps=args.alt_gbps,
            weights=load_weights(args.weights) if args.weights else None,
            weight_floor=args.weight_floor, seed=args.seed,
            expected_label=args.expected_label or None,
            expected_uuid=args.expected_uuid,
            host_root=args.host_root,
            host_margin_bytes=max(0, int(args.host_margin_gib * 2**30)),
            max_service_imbalance=args.max_service_imbalance,
            force=args.force,
        )
    except (OSError, ValueError, StripeMountError) as error:
        parser.exit(1, f"stripe repack FAILED: {error}\n")
    print(f"index: {result.index}\nmanifest: {result.manifest}\ngeneration: {result.generation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
