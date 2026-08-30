#!/usr/bin/env python3
"""Collect and validate exact row-aligned samples from IGLMF8A1 caches.

The collector treats E4M3 weights and FP16 scales as opaque bytes.  It parses
only the cache index needed to identify independently addressable matrix rows,
then copies complete row ranges without numeric conversion.

The resulting archive contains only:

    manifest.tsv
    dense/*.weights.e4m3.bin
    dense/*.scales.f16le.bin
    dflash/*.weights.e4m3.bin       (when requested)
    dflash/*.scales.f16le.bin       (when requested)
    SHA256SUMS

No checkpoint tensors, cache index, repository source, or alignment padding are
included.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
from collections import defaultdict
from typing import Iterator, Mapping, NoReturn, Sequence

MAGIC = b"IGLMF8A1"
VERSION = 1
GROUP_SIZE = 64
ALIGNMENT = 4096
INDEX_HEADER = struct.Struct("<8sIIIQ")
INDEX_RECORD = struct.Struct("<HIIQQQQ")
COPY_CHUNK = 8 << 20
DEFAULT_PART_BYTES = 512 << 20

MANIFEST_COLUMNS = (
    "source_kind",
    "tensor_name",
    "rows",
    "cols",
    "group_size",
    "source_index_sha256",
    "sample_row_begin",
    "sample_rows",
    "weight_file",
    "weight_bytes",
    "weight_sha256",
    "scale_file",
    "scale_bytes",
    "scale_sha256",
    "sampling_reason",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_DFLASH_LAYER_RE = re.compile(r"^L(\d+)\.")
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class SampleError(RuntimeError):
    """Raised for malformed caches, samples, or unsafe archive contents."""


@dataclasses.dataclass(frozen=True)
class TensorRecord:
    source_kind: str
    ordinal: int
    name: str
    rows: int
    cols: int
    weight_offset: int
    weight_bytes: int
    scale_offset: int
    scale_bytes: int
    family: str
    layer: int | None
    stage: str

    @property
    def weight_row_bytes(self) -> int:
        return self.cols

    @property
    def scale_row_bytes(self) -> int:
        return (self.cols // GROUP_SIZE) * 2


@dataclasses.dataclass(frozen=True)
class SourceCache:
    source_kind: str
    prefix: pathlib.Path
    index_path: pathlib.Path
    data_path: pathlib.Path
    index_sha256: str
    data_bytes: int
    records: tuple[TensorRecord, ...]


@dataclasses.dataclass(frozen=True)
class SampleSlice:
    record: TensorRecord
    row_begin: int
    rows: int
    selection: str
    row_band: str
    stratum_bytes: int

    @property
    def weight_bytes(self) -> int:
        return self.rows * self.record.weight_row_bytes

    @property
    def scale_bytes(self) -> int:
        return self.rows * self.record.scale_row_bytes


@dataclasses.dataclass(frozen=True)
class WrittenSlice:
    sample: SampleSlice
    weight_file: str
    weight_bytes: int
    weight_sha256: str
    scale_file: str
    scale_bytes: int
    scale_sha256: str


@dataclasses.dataclass(frozen=True)
class ArchiveResult:
    full_archive_name: str
    full_archive_sha256: str
    full_archive_bytes: int
    paths: tuple[pathlib.Path, ...]
    receipt: pathlib.Path


def fail(message: str) -> NoReturn:
    raise SampleError(message)


def mib(value: float) -> int:
    if value < 0:
        fail("MiB values must be non-negative")
    return int(round(value * (1 << 20)))


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_plain_name(name: str, what: str) -> None:
    if not name or any(character in name for character in "\t\r\n\0"):
        fail(f"{what} contains an empty name or a TSV/control character: {name!r}")


def normalize_prefix(value: os.PathLike[str] | str) -> pathlib.Path:
    text = os.fspath(value)
    if text.endswith(".index"):
        text = text[: -len(".index")]
    elif text.endswith(".bin"):
        text = text[: -len(".bin")]
    return pathlib.Path(text).expanduser().resolve()


def extract_layer(source_kind: str, name: str) -> int | None:
    if source_kind == "dflash":
        match = _DFLASH_LAYER_RE.match(name)
    else:
        match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def matrix_family(source_kind: str, name: str) -> str:
    if source_kind == "dflash":
        match = _DFLASH_LAYER_RE.match(name)
        suffix = name[match.end() :] if match else name
        if suffix in {"fc.a", "fc.b"}:
            return "fc"
        return suffix

    match = _LAYER_RE.search(name)
    if match:
        suffix = name[match.end() :]
    elif name.startswith("model.language_model."):
        suffix = name[len("model.language_model.") :]
    else:
        suffix = name
    if suffix.endswith(".weight"):
        suffix = suffix[: -len(".weight")]
    parts = ["*" if part.isdecimal() else part for part in suffix.split(".")]
    return ".".join(parts)


def assign_stages(source_kind: str, names: Sequence[str]) -> dict[str, tuple[int | None, str, str]]:
    layers = sorted({layer for name in names if (layer := extract_layer(source_kind, name)) is not None})
    rank = {layer: index for index, layer in enumerate(layers)}
    labels = ("early", "middle", "late")
    result: dict[str, tuple[int | None, str, str]] = {}
    for name in names:
        layer = extract_layer(source_kind, name)
        if layer is None:
            stage = "global"
        elif len(layers) == 1:
            stage = "middle"
        else:
            stage = labels[min(2, (3 * rank[layer]) // len(layers))]
        result[name] = (layer, stage, matrix_family(source_kind, name))
    return result


def parse_cache(
    prefix_value: os.PathLike[str] | str,
    source_kind: str,
    *,
    strict_alignment: bool = True,
) -> SourceCache:
    if source_kind not in {"dense", "dflash"}:
        fail(f"unsupported source kind: {source_kind}")
    prefix = normalize_prefix(prefix_value)
    index_path = pathlib.Path(str(prefix) + ".index")
    data_path = pathlib.Path(str(prefix) + ".bin")
    try:
        index_bytes = index_path.read_bytes()
    except OSError as error:
        fail(f"cannot read {source_kind} index {index_path}: {error}")
    if len(index_bytes) < INDEX_HEADER.size:
        fail(f"truncated {source_kind} index header: {index_path}")

    magic, version, group_size, count, data_bytes = INDEX_HEADER.unpack_from(index_bytes, 0)
    if magic != MAGIC:
        fail(f"{index_path} has magic {magic!r}; exact FP8 sampling requires {MAGIC!r}")
    if version != VERSION:
        fail(f"{index_path} has unsupported version {version}")
    if group_size != GROUP_SIZE:
        fail(f"{index_path} has group size {group_size}, expected {GROUP_SIZE}")
    if count == 0:
        fail(f"{index_path} contains no tensors")

    cursor = INDEX_HEADER.size
    unpacked: list[tuple[int, str, int, int, int, int, int, int]] = []
    names: set[str] = set()
    spans: list[tuple[int, int, str]] = []
    for ordinal in range(count):
        if len(index_bytes) - cursor < INDEX_RECORD.size:
            fail(f"truncated {source_kind} record header at ordinal {ordinal}")
        (name_bytes, rows, cols, weight_offset, weight_bytes,
         scale_offset, scale_bytes) = INDEX_RECORD.unpack_from(index_bytes, cursor)
        cursor += INDEX_RECORD.size
        if name_bytes == 0 or len(index_bytes) - cursor < name_bytes:
            fail(f"truncated or empty tensor name at ordinal {ordinal}")
        raw_name = index_bytes[cursor : cursor + name_bytes]
        cursor += name_bytes
        try:
            name = raw_name.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            fail(f"tensor name at ordinal {ordinal} is not UTF-8: {error}")
        ensure_plain_name(name, "tensor name")
        if name in names:
            fail(f"duplicate tensor name in {index_path}: {name}")
        names.add(name)
        if rows <= 0 or cols <= 0 or cols % GROUP_SIZE:
            fail(f"invalid geometry for {name}: {rows}x{cols}, group {GROUP_SIZE}")
        expected_weights = rows * cols
        expected_scales = rows * (cols // GROUP_SIZE) * 2
        if weight_bytes != expected_weights:
            fail(f"wrong weight byte count for {name}: {weight_bytes} != {expected_weights}")
        if scale_bytes != expected_scales:
            fail(f"wrong scale byte count for {name}: {scale_bytes} != {expected_scales}")
        if weight_offset > data_bytes or weight_bytes > data_bytes - weight_offset:
            fail(f"weight span for {name} exceeds declared data size")
        if scale_offset > data_bytes or scale_bytes > data_bytes - scale_offset:
            fail(f"scale span for {name} exceeds declared data size")
        if strict_alignment and (weight_offset % ALIGNMENT or scale_offset % ALIGNMENT):
            fail(f"unaligned {source_kind} tensor in quantizer-format cache: {name}")
        spans.append((weight_offset, weight_offset + weight_bytes, f"{name} weights"))
        spans.append((scale_offset, scale_offset + scale_bytes, f"{name} scales"))
        unpacked.append((ordinal, name, rows, cols, weight_offset, weight_bytes,
                         scale_offset, scale_bytes))

    if cursor != len(index_bytes):
        fail(f"trailing bytes in {source_kind} index: {len(index_bytes) - cursor}")
    for left, right in zip(sorted(spans), sorted(spans)[1:]):
        if right[0] < left[1]:
            fail(f"overlapping cache spans: {left[2]} and {right[2]}")
    try:
        actual_data_bytes = data_path.stat().st_size
    except OSError as error:
        fail(f"cannot stat {source_kind} data file {data_path}: {error}")
    if actual_data_bytes != data_bytes:
        fail(f"{data_path} size {actual_data_bytes} != index-declared {data_bytes}")

    attributes = assign_stages(source_kind, [entry[1] for entry in unpacked])
    records = tuple(
        TensorRecord(
            source_kind=source_kind,
            ordinal=ordinal,
            name=name,
            rows=rows,
            cols=cols,
            weight_offset=weight_offset,
            weight_bytes=weight_bytes,
            scale_offset=scale_offset,
            scale_bytes=scale_bytes,
            layer=attributes[name][0],
            stage=attributes[name][1],
            family=attributes[name][2],
        )
        for (ordinal, name, rows, cols, weight_offset, weight_bytes,
             scale_offset, scale_bytes) in unpacked
    )
    return SourceCache(
        source_kind=source_kind,
        prefix=prefix,
        index_path=index_path,
        data_path=data_path,
        index_sha256=sha256_bytes(index_bytes),
        data_bytes=data_bytes,
        records=records,
    )


def ceil_rows(byte_count: int, row_bytes: int) -> int:
    return 0 if byte_count <= 0 else (byte_count + row_bytes - 1) // row_bytes


def _weighted_fill_rows(
    quotas: dict[TensorRecord, int],
    weights: Mapping[TensorRecord, int],
    remaining_bytes: int,
) -> int:
    """Grow row quotas by a weighted, capped water fill.

    Returns the unfilled byte target.  Quotas may overshoot the requested byte
    count by less than one row of the final selected matrix.
    """
    active = {record for record in quotas if quotas[record] < record.rows}
    while remaining_bytes > 0 and active:
        total_weight = sum(max(1, weights[record]) for record in active)
        proposals: dict[TensorRecord, int] = {}
        for record in sorted(active, key=lambda item: (item.ordinal, item.name)):
            capacity = record.rows - quotas[record]
            ideal_bytes = remaining_bytes * max(1, weights[record]) / total_weight
            rows = min(capacity, int(ideal_bytes // record.weight_row_bytes))
            if rows:
                proposals[record] = rows
        if not proposals:
            record = max(
                active,
                key=lambda item: (
                    max(1, weights[item]) / item.weight_row_bytes,
                    item.weight_bytes,
                    -item.ordinal,
                ),
            )
            proposals[record] = 1
        spent = 0
        for record, rows in proposals.items():
            quotas[record] += rows
            spent += rows * record.weight_row_bytes
            if quotas[record] >= record.rows:
                active.discard(record)
        remaining_bytes -= spent
    return remaining_bytes


def split_row_bands(record: TensorRecord, sampled_rows: int) -> list[tuple[int, int, str]]:
    if sampled_rows <= 0 or sampled_rows > record.rows:
        fail(f"internal invalid row quota for {record.name}: {sampled_rows}/{record.rows}")
    if sampled_rows == record.rows or sampled_rows * 2 >= record.rows:
        return [(0, record.rows, "all")]
    bands = min(3, sampled_rows)
    base, remainder = divmod(sampled_rows, bands)
    counts = [base + (1 if index < remainder else 0) for index in range(bands)]
    if bands == 1:
        return [((record.rows - counts[0]) // 2, counts[0], "middle")]
    if bands == 2:
        return [(0, counts[0], "start"),
                (record.rows - counts[1], counts[1], "end")]
    middle_begin = (record.rows - counts[1]) // 2
    if middle_begin < counts[0] or middle_begin + counts[1] > record.rows - counts[2]:
        return [(0, record.rows, "all")]
    return [
        (0, counts[0], "start"),
        (middle_begin, counts[1], "middle"),
        (record.rows - counts[2], counts[2], "end"),
    ]


def plan_source(
    cache: SourceCache,
    *,
    target_weight_bytes: int,
    minimum_weight_bytes: int,
    complete_below_bytes: int,
    coverage_band_bytes: int,
) -> list[SampleSlice]:
    if target_weight_bytes < minimum_weight_bytes:
        fail(f"{cache.source_kind} target is below its minimum")
    if not cache.records:
        return []

    total_source_weights = sum(record.weight_bytes for record in cache.records)
    desired = min(total_source_weights, max(target_weight_bytes, minimum_weight_bytes))
    complete = {
        record for record in cache.records
        if record.weight_bytes <= complete_below_bytes
    }

    strata: dict[tuple[str, str], list[TensorRecord]] = defaultdict(list)
    for record in cache.records:
        strata[(record.family, record.stage)].append(record)
    stratum_bytes = {
        key: sum(record.weight_bytes for record in records)
        for key, records in strata.items()
    }
    representatives: set[TensorRecord] = set()
    for records in strata.values():
        layer_values = sorted(record.layer for record in records if record.layer is not None)
        median_layer = layer_values[len(layer_values) // 2] if layer_values else None
        representatives.add(max(
            records,
            key=lambda record: (
                record.weight_bytes,
                record.rows,
                0 if median_layer is None or record.layer is None
                else -abs(record.layer - median_layer),
                -record.ordinal,
            ),
        ))

    quotas: dict[TensorRecord, int] = {}
    selections: dict[TensorRecord, str] = {}
    for record in sorted(complete, key=lambda item: item.ordinal):
        quotas[record] = record.rows
        selections[record] = "complete-small"

    for record in sorted(representatives, key=lambda item: item.ordinal):
        if record in quotas:
            continue
        minimum_rows = min(
            record.rows,
            max(3 if record.rows >= 3 else record.rows,
                ceil_rows(coverage_band_bytes * 3, record.weight_row_bytes)),
        )
        # Avoid converting a nearly complete quota into three overlapping files.
        if minimum_rows * 2 >= record.rows:
            minimum_rows = record.rows
        quotas[record] = minimum_rows
        selections[record] = "stratified-large"

    selected_bytes = sum(record.weight_row_bytes * rows for record, rows in quotas.items())
    weights = {
        record: stratum_bytes[(record.family, record.stage)]
        for record in quotas
    }
    remaining = desired - selected_bytes
    if remaining > 0:
        remaining = _weighted_fill_rows(quotas, weights, remaining)

    if remaining > 0:
        # A model with many tiny strata can exhaust the first representatives.
        # Add further large tensors in deterministic on-disk order and fill them.
        extras = sorted(
            (record for record in cache.records if record not in quotas),
            key=lambda record: (-record.weight_bytes, record.ordinal),
        )
        for record in extras:
            quotas[record] = 0
            selections[record] = "target-fill"
            weights[record] = stratum_bytes[(record.family, record.stage)]
            remaining = _weighted_fill_rows(quotas, weights, remaining)
            if remaining <= 0:
                break

    slices: list[SampleSlice] = []
    for record in sorted(quotas, key=lambda item: item.ordinal):
        rows = quotas[record]
        if rows <= 0:
            continue
        for row_begin, row_count, row_band in split_row_bands(record, rows):
            slices.append(SampleSlice(
                record=record,
                row_begin=row_begin,
                rows=row_count,
                selection=selections[record],
                row_band=row_band,
                stratum_bytes=stratum_bytes[(record.family, record.stage)],
            ))

    slices.sort(key=lambda item: (item.record.ordinal, item.row_begin))
    sampled_weight_bytes = sum(item.weight_bytes for item in slices)
    if sampled_weight_bytes < min(minimum_weight_bytes, total_source_weights):
        fail(
            f"{cache.source_kind} plan produced only {sampled_weight_bytes} weight bytes; "
            f"minimum is {minimum_weight_bytes}"
        )
    _validate_slice_plan(cache, slices)
    return slices


def _validate_slice_plan(cache: SourceCache, slices: Sequence[SampleSlice]) -> None:
    by_tensor: dict[str, list[tuple[int, int]]] = defaultdict(list)
    covered_strata: set[tuple[str, str]] = set()
    for sample in slices:
        record = sample.record
        if record.source_kind != cache.source_kind:
            fail("internal source-kind mismatch in sample plan")
        if sample.row_begin < 0 or sample.rows <= 0 or sample.row_begin + sample.rows > record.rows:
            fail(f"sample rows exceed {record.name}")
        by_tensor[record.name].append((sample.row_begin, sample.row_begin + sample.rows))
        covered_strata.add((record.family, record.stage))
    for name, ranges in by_tensor.items():
        ranges.sort()
        for left, right in zip(ranges, ranges[1:]):
            if right[0] < left[1]:
                fail(f"overlapping planned samples for {name}")
    expected_strata = {(record.family, record.stage) for record in cache.records}
    missing = sorted(expected_strata - covered_strata)
    if missing:
        fail(f"sampling plan misses matrix strata: {missing}")


def safe_slug(name: str, limit: int = 40) -> str:
    slug = _SAFE_COMPONENT_RE.sub("-", name).strip("-.") or "tensor"
    return slug[:limit]


def tensor_id(sample: SampleSlice) -> str:
    name_hash = hashlib.sha256(sample.record.name.encode("utf-8")).hexdigest()[:10]
    return (
        f"{sample.record.ordinal:04d}-{safe_slug(sample.record.name)}-{name_hash}"
        f"-r{sample.row_begin:08d}-n{sample.rows:08d}"
    )


def copy_span(fd: int, offset: int, byte_count: int, destination: pathlib.Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    remaining = byte_count
    cursor = offset
    with destination.open("xb") as output:
        while remaining:
            request = min(COPY_CHUNK, remaining)
            chunk = os.pread(fd, request, cursor)
            if len(chunk) != request:
                fail(
                    f"short source read at offset {cursor}: {len(chunk)} of {request} bytes"
                )
            output.write(chunk)
            digest.update(chunk)
            cursor += request
            remaining -= request
        output.flush()
        os.fsync(output.fileno())
    return byte_count, digest.hexdigest()


def quote_reason(value: str) -> str:
    return urllib.parse.quote(value, safe="._-*")


def sampling_reason(sample: SampleSlice) -> str:
    record = sample.record
    fields = (
        ("family", record.family),
        ("stage", record.stage),
        ("selection", sample.selection),
        ("row_band", sample.row_band),
        ("layer", "global" if record.layer is None else str(record.layer)),
        ("stratum_weight_bytes", str(sample.stratum_bytes)),
    )
    return ";".join(f"{key}={quote_reason(value)}" for key, value in fields)


def write_sample_directory(
    output_root: pathlib.Path,
    caches: Sequence[SourceCache],
    plans: Mapping[str, Sequence[SampleSlice]],
) -> list[WrittenSlice]:
    if output_root.exists():
        fail(f"sample directory already exists: {output_root}")
    output_root.mkdir(parents=True)
    cache_by_kind = {cache.source_kind: cache for cache in caches}
    written: list[WrittenSlice] = []
    descriptors: dict[str, int] = {}
    try:
        for kind, cache in cache_by_kind.items():
            descriptors[kind] = os.open(cache.data_path, os.O_RDONLY | os.O_CLOEXEC)
        for kind in ("dense", "dflash"):
            cache = cache_by_kind.get(kind)
            if cache is None:
                continue
            for sample in plans.get(kind, ()):
                identifier = tensor_id(sample)
                weight_rel = f"{kind}/{identifier}.weights.e4m3.bin"
                scale_rel = f"{kind}/{identifier}.scales.f16le.bin"
                weight_path = output_root / weight_rel
                scale_path = output_root / scale_rel
                record = sample.record
                weight_offset = record.weight_offset + sample.row_begin * record.weight_row_bytes
                scale_offset = record.scale_offset + sample.row_begin * record.scale_row_bytes
                weight_bytes, weight_hash = copy_span(
                    descriptors[kind], weight_offset, sample.weight_bytes, weight_path
                )
                scale_bytes, scale_hash = copy_span(
                    descriptors[kind], scale_offset, sample.scale_bytes, scale_path
                )
                written.append(WrittenSlice(
                    sample=sample,
                    weight_file=weight_rel,
                    weight_bytes=weight_bytes,
                    weight_sha256=weight_hash,
                    scale_file=scale_rel,
                    scale_bytes=scale_bytes,
                    scale_sha256=scale_hash,
                ))
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)

    written.sort(key=lambda item: (
        0 if item.sample.record.source_kind == "dense" else 1,
        item.sample.record.ordinal,
        item.sample.row_begin,
    ))
    manifest_path = output_root / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for item in written:
            sample = item.sample
            record = sample.record
            cache = cache_by_kind[record.source_kind]
            writer.writerow({
                "source_kind": record.source_kind,
                "tensor_name": record.name,
                "rows": record.rows,
                "cols": record.cols,
                "group_size": GROUP_SIZE,
                "source_index_sha256": cache.index_sha256,
                "sample_row_begin": sample.row_begin,
                "sample_rows": sample.rows,
                "weight_file": item.weight_file,
                "weight_bytes": item.weight_bytes,
                "weight_sha256": item.weight_sha256,
                "scale_file": item.scale_file,
                "scale_bytes": item.scale_bytes,
                "scale_sha256": item.scale_sha256,
                "sampling_reason": sampling_reason(sample),
            })
        output.flush()
        os.fsync(output.fileno())

    checksummed = [path for path in output_root.rglob("*") if path.is_file()]
    checksummed.sort(key=lambda path: path.relative_to(output_root).as_posix())
    sums_path = output_root / "SHA256SUMS"
    with sums_path.open("w", encoding="ascii", newline="\n") as output:
        for path in checksummed:
            relative = path.relative_to(output_root).as_posix()
            output.write(f"{sha256_file(path)}  {relative}\n")
        output.flush()
        os.fsync(output.fileno())
    return written


def safe_member_name(name: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "" in path.parts:
        fail(f"unsafe archive/sample path: {name!r}")
    return path


def parse_sha256sums(path: pathlib.Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as error:
        fail(f"cannot read {path}: {error}")
    for line_number, line in enumerate(lines, 1):
        if len(line) < 67 or line[64:66] != "  ":
            fail(f"malformed SHA256SUMS line {line_number}")
        digest, name = line[:64], line[66:]
        if not _HEX64.fullmatch(digest):
            fail(f"bad SHA-256 on SHA256SUMS line {line_number}")
        safe_member_name(name)
        if name == "SHA256SUMS" or name in result:
            fail(f"invalid or duplicate SHA256SUMS path: {name}")
        result[name] = digest
    if not result:
        fail("SHA256SUMS is empty")
    return result


def parse_reason(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in text.split(";"):
        if "=" not in item:
            fail(f"malformed sampling_reason component: {item!r}")
        key, value = item.split("=", 1)
        if not key or key in result:
            fail(f"duplicate/empty sampling_reason key: {key!r}")
        result[key] = urllib.parse.unquote(value)
    required = {"family", "stage", "selection", "row_band", "layer", "stratum_weight_bytes"}
    if set(result) != required:
        fail(f"sampling_reason keys {sorted(result)} != {sorted(required)}")
    try:
        if int(result["stratum_weight_bytes"]) <= 0:
            raise ValueError
    except ValueError:
        fail("sampling_reason has an invalid stratum_weight_bytes")
    return result


def parse_nonnegative(row: Mapping[str, str], key: str, *, positive: bool = False) -> int:
    text = row.get(key, "")
    try:
        value = int(text, 10)
    except ValueError:
        fail(f"manifest field {key} is not an integer: {text!r}")
    if value < 0 or (positive and value == 0):
        fail(f"manifest field {key} is out of range: {value}")
    return value


def compare_file_to_span(path: pathlib.Path, fd: int, offset: int, byte_count: int) -> None:
    remaining = byte_count
    cursor = offset
    with path.open("rb") as sample:
        while remaining:
            request = min(COPY_CHUNK, remaining)
            actual = sample.read(request)
            expected = os.pread(fd, request, cursor)
            if len(actual) != request or len(expected) != request:
                fail(f"short compare read for {path}")
            if actual != expected:
                fail(f"sample bytes differ from source cache: {path}")
            cursor += request
            remaining -= request
        if sample.read(1):
            fail(f"sample file has trailing bytes: {path}")


def validate_sample_directory(
    root: pathlib.Path,
    *,
    caches: Sequence[SourceCache] = (),
    minimum_dense_weight_bytes: int = 256 << 20,
) -> dict[str, object]:
    root = root.resolve()
    manifest_path = root / "manifest.tsv"
    sums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not sums_path.is_file():
        fail(f"sample root lacks manifest.tsv or SHA256SUMS: {root}")

    sums = parse_sha256sums(sums_path)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual_files != set(sums):
        missing = sorted(set(sums) - actual_files)
        extra = sorted(actual_files - set(sums))
        fail(f"SHA256SUMS file set mismatch; missing={missing}, extra={extra}")
    for relative, expected in sorted(sums.items()):
        actual = sha256_file(root / relative)
        if actual != expected:
            fail(f"SHA-256 mismatch for {relative}: {actual} != {expected}")

    cache_by_kind = {cache.source_kind: cache for cache in caches}
    record_by_kind_name = {
        kind: {record.name: record for record in cache.records}
        for kind, cache in cache_by_kind.items()
    }
    descriptors: dict[str, int] = {}
    for kind, cache in cache_by_kind.items():
        descriptors[kind] = os.open(cache.data_path, os.O_RDONLY | os.O_CLOEXEC)

    rows_seen: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    paths_seen: set[str] = set()
    totals = defaultdict(int)
    family_totals: dict[tuple[str, str, str], int] = defaultdict(int)
    stage_totals: dict[tuple[str, str], int] = defaultdict(int)
    source_hashes: dict[str, set[str]] = defaultdict(set)
    manifest_rows = 0
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
                fail(
                    f"manifest columns {tuple(reader.fieldnames or ())!r} "
                    f"!= {MANIFEST_COLUMNS!r}"
                )
            for manifest_rows, row in enumerate(reader, 1):
                kind = row["source_kind"]
                if kind not in {"dense", "dflash"}:
                    fail(f"manifest row {manifest_rows}: invalid source_kind {kind!r}")
                name = row["tensor_name"]
                ensure_plain_name(name, "manifest tensor name")
                rows = parse_nonnegative(row, "rows", positive=True)
                cols = parse_nonnegative(row, "cols", positive=True)
                group = parse_nonnegative(row, "group_size", positive=True)
                row_begin = parse_nonnegative(row, "sample_row_begin")
                sample_rows = parse_nonnegative(row, "sample_rows", positive=True)
                weight_bytes = parse_nonnegative(row, "weight_bytes", positive=True)
                scale_bytes = parse_nonnegative(row, "scale_bytes", positive=True)
                if group != GROUP_SIZE or cols % group:
                    fail(f"manifest row {manifest_rows}: invalid group geometry")
                if row_begin + sample_rows > rows:
                    fail(f"manifest row {manifest_rows}: sampled rows exceed tensor")
                expected_weight_bytes = sample_rows * cols
                expected_scale_bytes = sample_rows * (cols // group) * 2
                if weight_bytes != expected_weight_bytes or scale_bytes != expected_scale_bytes:
                    fail(f"manifest row {manifest_rows}: byte count disagrees with row geometry")

                source_hash = row["source_index_sha256"]
                if not _HEX64.fullmatch(source_hash):
                    fail(f"manifest row {manifest_rows}: bad source index SHA-256")
                source_hashes[kind].add(source_hash)
                weight_hash = row["weight_sha256"]
                scale_hash = row["scale_sha256"]
                if not _HEX64.fullmatch(weight_hash) or not _HEX64.fullmatch(scale_hash):
                    fail(f"manifest row {manifest_rows}: bad file SHA-256")

                weight_file = row["weight_file"]
                scale_file = row["scale_file"]
                for relative, suffix in (
                    (weight_file, ".weights.e4m3.bin"),
                    (scale_file, ".scales.f16le.bin"),
                ):
                    path = safe_member_name(relative)
                    if path.parts[0] != kind or not relative.endswith(suffix):
                        fail(f"manifest row {manifest_rows}: invalid sample path {relative}")
                    if relative in paths_seen:
                        fail(f"manifest row {manifest_rows}: duplicate path {relative}")
                    paths_seen.add(relative)
                weight_path = root / weight_file
                scale_path = root / scale_file
                if weight_path.stat().st_size != weight_bytes or scale_path.stat().st_size != scale_bytes:
                    fail(f"manifest row {manifest_rows}: sample file size mismatch")
                if sums.get(weight_file) != weight_hash or sums.get(scale_file) != scale_hash:
                    fail(f"manifest row {manifest_rows}: manifest/SHA256SUMS disagreement")

                reason = parse_reason(row["sampling_reason"])
                rows_seen[(kind, name)].append((row_begin, row_begin + sample_rows))
                totals[f"{kind}_weight_bytes"] += weight_bytes
                totals[f"{kind}_scale_bytes"] += scale_bytes
                totals[f"{kind}_slices"] += 1
                family_totals[(kind, reason["family"], reason["stage"])] += weight_bytes
                stage_totals[(kind, reason["stage"])] += weight_bytes

                cache = cache_by_kind.get(kind)
                if cache is not None:
                    if source_hash != cache.index_sha256:
                        fail(f"manifest source index hash does not match {cache.index_path}")
                    record = record_by_kind_name[kind].get(name)
                    if record is None:
                        fail(f"manifest tensor not present in {cache.index_path}: {name}")
                    if rows != record.rows or cols != record.cols:
                        fail(f"manifest geometry differs from source for {name}")
                    if reason["family"] != record.family or reason["stage"] != record.stage:
                        fail(f"manifest family/stage differs from source for {name}")
                    compare_file_to_span(
                        weight_path,
                        descriptors[kind],
                        record.weight_offset + row_begin * record.weight_row_bytes,
                        weight_bytes,
                    )
                    compare_file_to_span(
                        scale_path,
                        descriptors[kind],
                        record.scale_offset + row_begin * record.scale_row_bytes,
                        scale_bytes,
                    )
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)

    if manifest_rows == 0:
        fail("manifest has no sample rows")
    if set(sums) != paths_seen | {"manifest.tsv"}:
        fail("SHA256SUMS contains undeclared sample files")
    if len(sums) != len(paths_seen) + 1:
        fail("sample has unexpected metadata files")
    for kind, hashes in source_hashes.items():
        if len(hashes) != 1:
            fail(f"manifest uses multiple source index hashes for {kind}")
    for key, ranges in rows_seen.items():
        ranges.sort()
        for left, right in zip(ranges, ranges[1:]):
            if right[0] < left[1]:
                fail(f"overlapping sample rows for {key[0]} tensor {key[1]}")

    dense_weight_bytes = totals.get("dense_weight_bytes", 0)
    if dense_weight_bytes < minimum_dense_weight_bytes:
        fail(
            f"dense sample contains {dense_weight_bytes / 2**20:.3f} MiB of weight bytes; "
            f"minimum is {minimum_dense_weight_bytes / 2**20:.3f} MiB"
        )

    # With a source cache present, prove that every derived family/stage stratum
    # has at least one selected slice.
    for kind, cache in cache_by_kind.items():
        expected = {(record.family, record.stage) for record in cache.records}
        actual = {
            (family, stage)
            for (entry_kind, family, stage), byte_count in family_totals.items()
            if entry_kind == kind and byte_count > 0
        }
        missing = sorted(expected - actual)
        if missing:
            fail(f"sample misses {kind} family/stage strata: {missing}")

    report: dict[str, object] = {
        "schema": "fp8-residency-sample-v1",
        "sample_root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "sha256sums_sha256": sha256_file(sums_path),
        "manifest_rows": manifest_rows,
        "source_index_sha256": {
            kind: next(iter(hashes)) for kind, hashes in sorted(source_hashes.items())
        },
        "totals": dict(sorted(totals.items())),
        "family_stage_weight_bytes": [
            {
                "source_kind": kind,
                "family": family,
                "stage": stage,
                "weight_bytes": byte_count,
            }
            for (kind, family, stage), byte_count in sorted(family_totals.items())
        ],
        "stage_weight_bytes": [
            {"source_kind": kind, "stage": stage, "weight_bytes": byte_count}
            for (kind, stage), byte_count in sorted(stage_totals.items())
        ],
        "source_verified": sorted(cache_by_kind),
    }
    return report


def add_tar_file(archive: tarfile.TarFile, root: pathlib.Path, relative: str) -> None:
    path = root / relative
    info = tarfile.TarInfo(relative)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    with path.open("rb") as source:
        archive.addfile(info, source)


def build_archive(
    sample_root: pathlib.Path,
    output_path: pathlib.Path,
    *,
    zstd_level: int,
    part_bytes: int,
) -> ArchiveResult:
    if output_path.suffixes[-2:] != [".tar", ".zst"]:
        fail("archive output must end in .tar.zst")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        fail(f"archive output already exists: {output_path}")
    zstd = shutil.which("zstd")
    if not zstd:
        fail("zstd executable is required to create fp8-residency-sample-v1.tar.zst")

    allowed = sorted(
        path.relative_to(sample_root).as_posix()
        for path in sample_root.rglob("*")
        if path.is_file()
    )
    if "manifest.tsv" not in allowed or "SHA256SUMS" not in allowed:
        fail("sample directory is incomplete before archiving")
    with tempfile.TemporaryDirectory(prefix="fp8-sample-tar-", dir=output_path.parent) as temp:
        tar_path = pathlib.Path(temp) / output_path.name[: -len(".zst")]
        with tarfile.open(tar_path, "w", format=tarfile.USTAR_FORMAT) as archive:
            for relative in allowed:
                add_tar_file(archive, sample_root, relative)
        subprocess.run(
            [zstd, "-q", "-f", f"-{zstd_level}", "-T1", str(tar_path), "-o", str(output_path)],
            check=True,
        )
    subprocess.run([zstd, "-q", "-t", str(output_path)], check=True)
    full_hash = sha256_file(output_path)
    full_bytes = output_path.stat().st_size
    paths: list[pathlib.Path] = []
    receipt = pathlib.Path(str(output_path) + ".SHA256SUMS")
    receipt_lines = [f"{full_hash}  {output_path.name}\n"]
    if full_bytes > part_bytes:
        with output_path.open("rb") as source:
            part = 0
            while source.tell() < full_bytes:
                part_path = pathlib.Path(f"{output_path}.part-{part:03d}")
                remaining = min(part_bytes, full_bytes - source.tell())
                digest = hashlib.sha256()
                with part_path.open("xb") as destination:
                    while remaining:
                        chunk = source.read(min(COPY_CHUNK, remaining))
                        if not chunk:
                            fail("short read while splitting the archive")
                        destination.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                part_hash = digest.hexdigest()
                receipt_lines.append(f"{part_hash}  {part_path.name}\n")
                paths.append(part_path)
                part += 1
        output_path.unlink()
    else:
        paths.append(output_path)
    receipt.write_text("".join(receipt_lines), encoding="ascii", newline="\n")
    return ArchiveResult(
        full_archive_name=output_path.name,
        full_archive_sha256=full_hash,
        full_archive_bytes=full_bytes,
        paths=tuple(paths),
        receipt=receipt,
    )


def _copy_parts(first_part: pathlib.Path, destination: pathlib.Path) -> None:
    match = re.match(r"^(.*\.tar\.zst)\.part-(\d{3})$", first_part.name)
    if not match:
        fail(f"not a split archive part: {first_part}")
    if int(match.group(2)) != 0:
        fail("validation of split archives must start from .part-000")
    base_name = match.group(1)
    part_index = 0
    with destination.open("xb") as output:
        while True:
            part = first_part.with_name(f"{base_name}.part-{part_index:03d}")
            if not part.exists():
                break
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, COPY_CHUNK)
            part_index += 1
        output.flush()
        os.fsync(output.fileno())
    if part_index == 0:
        fail("split archive has no readable parts")


@contextlib.contextmanager
def materialize_sample(sample: pathlib.Path) -> Iterator[pathlib.Path]:
    sample = sample.expanduser().resolve()
    if sample.is_dir():
        yield sample
        return
    if not sample.is_file():
        fail(f"sample path does not exist: {sample}")
    zstd = shutil.which("zstd")
    with tempfile.TemporaryDirectory(prefix="fp8-sample-validate-") as temp_text:
        temp = pathlib.Path(temp_text)
        archive_path = sample
        if re.search(r"\.tar\.zst\.part-\d{3}$", sample.name):
            archive_path = temp / sample.name.split(".part-")[0]
            _copy_parts(sample, archive_path)
        if archive_path.name.endswith(".tar.zst"):
            if not zstd:
                fail("zstd executable is required to validate a .tar.zst archive")
            tar_path = temp / archive_path.name[: -len(".zst")]
            subprocess.run([zstd, "-q", "-d", "-f", str(archive_path), "-o", str(tar_path)], check=True)
        elif archive_path.name.endswith(".tar"):
            tar_path = archive_path
        else:
            fail("sample must be a directory, .tar, .tar.zst, or .tar.zst.part-000")

        root = temp / "sample"
        root.mkdir()
        with tarfile.open(tar_path, "r:") as archive:
            members = archive.getmembers()
            if not members:
                fail("sample archive is empty")
            for member in members:
                safe_member_name(member.name)
                if not member.isfile():
                    fail(f"archive contains a non-regular member: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    fail(f"cannot read archive member: {member.name}")
                destination = root / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    shutil.copyfileobj(source, output, COPY_CHUNK)
                if destination.stat().st_size != member.size:
                    fail(f"short archive member extraction: {member.name}")
        yield root


def source_inventory(cache: SourceCache) -> dict[str, object]:
    family_bytes: dict[tuple[str, str], int] = defaultdict(int)
    for record in cache.records:
        family_bytes[(record.family, record.stage)] += record.weight_bytes
    return {
        "source_kind": cache.source_kind,
        "prefix": str(cache.prefix),
        "index_path": str(cache.index_path),
        "data_path": str(cache.data_path),
        "index_sha256": cache.index_sha256,
        "data_bytes": cache.data_bytes,
        "tensor_count": len(cache.records),
        "weight_bytes": sum(record.weight_bytes for record in cache.records),
        "scale_bytes": sum(record.scale_bytes for record in cache.records),
        "family_stage_weight_bytes": [
            {"family": family, "stage": stage, "weight_bytes": byte_count}
            for (family, stage), byte_count in sorted(family_bytes.items())
        ],
        "tensors": [
            {
                "ordinal": record.ordinal,
                "tensor_name": record.name,
                "rows": record.rows,
                "cols": record.cols,
                "group_size": GROUP_SIZE,
                "weight_offset": record.weight_offset,
                "weight_bytes": record.weight_bytes,
                "scale_offset": record.scale_offset,
                "scale_bytes": record.scale_bytes,
                "family": record.family,
                "layer": record.layer,
                "stage": record.stage,
            }
            for record in cache.records
        ],
    }


def write_json(path: pathlib.Path | None, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def collect_command(args: argparse.Namespace) -> int:
    caches = [parse_cache(args.dense_prefix, "dense", strict_alignment=not args.allow_unaligned)]
    if args.dflash_prefix:
        caches.append(parse_cache(args.dflash_prefix, "dflash", strict_alignment=not args.allow_unaligned))
    cache_by_kind = {cache.source_kind: cache for cache in caches}
    plans: dict[str, list[SampleSlice]] = {
        "dense": plan_source(
            cache_by_kind["dense"],
            target_weight_bytes=mib(args.target_dense_weight_mib),
            minimum_weight_bytes=mib(args.minimum_dense_weight_mib),
            complete_below_bytes=mib(args.complete_below_mib),
            coverage_band_bytes=mib(args.coverage_band_mib),
        )
    }
    if "dflash" in cache_by_kind:
        plans["dflash"] = plan_source(
            cache_by_kind["dflash"],
            target_weight_bytes=mib(args.target_dflash_weight_mib),
            minimum_weight_bytes=0,
            complete_below_bytes=mib(args.complete_below_mib),
            coverage_band_bytes=mib(args.coverage_band_mib),
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = output.parent / (output.name + ".dir")
    if directory.exists():
        fail(f"temporary/final sample directory already exists: {directory}")
    try:
        write_sample_directory(directory, caches, plans)
        report = validate_sample_directory(
            directory,
            caches=caches,
            minimum_dense_weight_bytes=mib(args.minimum_dense_weight_mib),
        )
        archive = build_archive(
            directory,
            output,
            zstd_level=args.zstd_level,
            part_bytes=mib(args.part_mib),
        )
        report["archive"] = {
            "full_name": archive.full_archive_name,
            "full_sha256": archive.full_archive_sha256,
            "full_bytes": archive.full_archive_bytes,
            "files": [str(path) for path in archive.paths],
            "receipt": str(archive.receipt),
        }
        report["source_inventory"] = [source_inventory(cache) for cache in caches]
        if args.report_json:
            write_json(args.report_json.expanduser().resolve(), report)
        else:
            summary = report["totals"]
            assert isinstance(summary, dict)
            print(json.dumps({
                "archive": report["archive"],
                "dense_weight_mib": summary.get("dense_weight_bytes", 0) / 2**20,
                "dense_scale_mib": summary.get("dense_scale_bytes", 0) / 2**20,
                "dflash_weight_mib": summary.get("dflash_weight_bytes", 0) / 2**20,
                "manifest_rows": report["manifest_rows"],
            }, indent=2, sort_keys=True))
    finally:
        if directory.exists() and not args.keep_directory:
            shutil.rmtree(directory)
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    caches = [parse_cache(args.dense_prefix, "dense", strict_alignment=not args.allow_unaligned)]
    if args.dflash_prefix:
        caches.append(parse_cache(args.dflash_prefix, "dflash", strict_alignment=not args.allow_unaligned))
    report = {
        "schema": "iglmf8a1-source-inventory-v1",
        "sources": [source_inventory(cache) for cache in caches],
    }
    write_json(args.output.expanduser().resolve() if args.output else None, report)
    return 0


def validate_command(args: argparse.Namespace) -> int:
    caches: list[SourceCache] = []
    if args.dense_prefix:
        caches.append(parse_cache(args.dense_prefix, "dense", strict_alignment=not args.allow_unaligned))
    if args.dflash_prefix:
        caches.append(parse_cache(args.dflash_prefix, "dflash", strict_alignment=not args.allow_unaligned))
    sample_path = args.sample.expanduser().resolve()
    archive_hash = sha256_file(sample_path) if sample_path.is_file() else None
    with materialize_sample(sample_path) as root:
        report = validate_sample_directory(
            root,
            caches=caches,
            minimum_dense_weight_bytes=mib(args.minimum_dense_weight_mib),
        )
    if archive_hash:
        report["archive_or_part_sha256"] = archive_hash
        report["archive_or_part_path"] = str(sample_path)
    write_json(args.report_json.expanduser().resolve() if args.report_json else None, report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and validate exact row-aligned IGLMF8A1 cache samples."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="validate source cache geometry and emit JSON inventory")
    inspect.add_argument("--dense-prefix", type=pathlib.Path, required=True)
    inspect.add_argument("--dflash-prefix", type=pathlib.Path)
    inspect.add_argument("--output", type=pathlib.Path)
    inspect.add_argument("--allow-unaligned", action="store_true",
                         help="accept reader-valid spans not emitted by the tracked quantizers")
    inspect.set_defaults(function=inspect_command)

    collect = subparsers.add_parser("collect", help="build fp8-residency-sample-v1.tar.zst")
    collect.add_argument("--dense-prefix", type=pathlib.Path, required=True)
    collect.add_argument("--dflash-prefix", type=pathlib.Path)
    collect.add_argument("--output", type=pathlib.Path,
                         default=pathlib.Path("fp8-residency-sample-v1.tar.zst"))
    collect.add_argument("--target-dense-weight-mib", type=float, default=512.0)
    collect.add_argument("--minimum-dense-weight-mib", type=float, default=256.0)
    collect.add_argument("--target-dflash-weight-mib", type=float, default=64.0)
    collect.add_argument("--complete-below-mib", type=float, default=2.0)
    collect.add_argument("--coverage-band-mib", type=float, default=0.5,
                         help="minimum start/middle/end bytes per family-stage representative")
    collect.add_argument("--part-mib", type=float, default=512.0)
    collect.add_argument("--zstd-level", type=int, choices=range(1, 20), default=3)
    collect.add_argument("--keep-directory", action="store_true")
    collect.add_argument("--report-json", type=pathlib.Path)
    collect.add_argument("--allow-unaligned", action="store_true")
    collect.set_defaults(function=collect_command)

    validate = subparsers.add_parser("validate", help="verify a directory/archive and optionally source bytes")
    validate.add_argument("sample", type=pathlib.Path)
    validate.add_argument("--dense-prefix", type=pathlib.Path)
    validate.add_argument("--dflash-prefix", type=pathlib.Path)
    validate.add_argument("--minimum-dense-weight-mib", type=float, default=256.0)
    validate.add_argument("--report-json", type=pathlib.Path)
    validate.add_argument("--allow-unaligned", action="store_true")
    validate.set_defaults(function=validate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except SampleError as error:
        parser.error(str(error))
    except subprocess.CalledProcessError as error:
        parser.error(f"external command failed with status {error.returncode}: {error.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
