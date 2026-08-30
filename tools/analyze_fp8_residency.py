#!/usr/bin/env python3
"""Measure real E4M3 entropy and exact fixed-tile codec ceilings.

The analyzer consumes the validated ``fp8-residency-sample-v1`` directory or
archive.  It never interprets scale values for compression; FP16 scale bytes
remain an exact raw stream owned by the earlier scale-codec task.

Weight tiles follow the current Ada consumer geometry.  A nominal T-byte tile
contains T/64 rows from one group-64 column segment, so T=1024 is exactly one
16x64 shared-memory slab consumed by ``mma.sync.m16n8k32`` twice.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import fp8_residency_codec as codec
import sample_fp8_cache as sample_io

FORMAT_TO_MODE = {
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
MODE_TO_FORMAT = {mode: name for name, mode in FORMAT_TO_MODE.items()}
MIXED_FORMAT_PRIORITY = tuple(MODE_TO_FORMAT[mode] for mode in codec.MODE_PRIORITY)
FORMAT_NAMES = (*FORMAT_TO_MODE, "mixed")
TILE_SIZES = codec.ALLOWED_TILE_BYTES
# Equal byte ratios choose the tile that maps to one complete 16x64 consumer
# slab, then the lowest-decode-cost fixed mode.  Mixed mode wins only on bytes.
TILE_SELECTION_PRIORITY = (1024, 512, 256, 128)
TILE_SELECTION_RANK = {value: rank for rank, value in enumerate(TILE_SELECTION_PRIORITY)}
FORMAT_SELECTION_PRIORITY = (*MIXED_FORMAT_PRIORITY, "mixed")
FORMAT_SELECTION_RANK = {value: rank for rank, value in enumerate(FORMAT_SELECTION_PRIORITY)}
PAIR_BINS = 256 * 256

H0_KILL_BITS_PER_BYTE = 7.6
PADDED_RATIO_KILL = 0.94
PADDED_RATIO_GATE = 0.90
HIGH_VOLUME_RATIO_GATE = 0.85
HIGH_VOLUME_FAMILY_MIN_BYTES = 64 << 20
HIGH_VOLUME_FAMILY_FRACTION = 0.05
ALLOCATOR_RECLAIM_GATE_BYTES = 384 << 20
ALLOCATOR_RECLAIM_GATE_SLOTS = 28
DEFAULT_EXPANDED_EXPERT_SLOT_BYTES = int(13.5 * (1 << 20))
DISK_ALIGNMENT = 4096


class AnalysisError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ManifestSlice:
    source_kind: str
    tensor_name: str
    rows: int
    cols: int
    row_begin: int
    sample_rows: int
    weight_file: str
    scale_file: str
    weight_bytes: int
    scale_bytes: int
    family: str
    stage: str
    layer: int | None


@dataclasses.dataclass
class PairAccumulator:
    horizontal: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((256, 256), dtype=np.uint64)
    )
    vertical: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros((256, 256), dtype=np.uint64)
    )


@dataclasses.dataclass
class TileAccumulator:
    tile_bytes: int
    tile_count: int = 0
    raw_bytes: int = 0
    padded_bytes: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    raw_tiles: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    mixed_modes: Counter[str] = dataclasses.field(default_factory=Counter)
    distinct_chunks: list[np.ndarray] = dataclasses.field(default_factory=list)
    positive_zero_chunks: list[np.ndarray] = dataclasses.field(default_factory=list)
    negative_zero_chunks: list[np.ndarray] = dataclasses.field(default_factory=list)

    def add(self, measured: "BatchMeasurement") -> None:
        count, raw_bytes = measured.tiles, measured.raw_bytes
        self.tile_count += count
        self.raw_bytes += raw_bytes
        for name, values in measured.selected_padded.items():
            self.padded_bytes[name] += int(values.sum(dtype=np.int64))
            self.raw_tiles[name] += int(measured.raw_selected[name].sum(dtype=np.int64))
        self.mixed_modes.update(measured.mixed_mode_counts)
        self.distinct_chunks.append(measured.distinct.astype(np.uint16, copy=False))
        self.positive_zero_chunks.append(measured.positive_zero.astype(np.uint16, copy=False))
        self.negative_zero_chunks.append(measured.negative_zero.astype(np.uint16, copy=False))


@dataclasses.dataclass(frozen=True)
class BatchMeasurement:
    tiles: int
    raw_bytes: int
    selected_padded: dict[str, np.ndarray]
    raw_selected: dict[str, np.ndarray]
    mixed_mode_counts: Counter[str]
    distinct: np.ndarray
    positive_zero: np.ndarray
    negative_zero: np.ndarray


@dataclasses.dataclass(frozen=True)
class TensorInventory:
    source_kind: str
    tensor_name: str
    family: str
    stage: str
    layer: int | None
    ordinal: int
    rows: int
    cols: int
    weight_bytes: int
    scale_bytes: int


@dataclasses.dataclass
class TensorAnalysis:
    inventory: TensorInventory
    split: str
    sample_rows: int
    sample_weight_bytes: int
    sample_scale_bytes: int
    histogram: np.ndarray
    pairs: PairAccumulator
    tiles: dict[int, TileAccumulator]


def sha_split(source_kind: str, tensor_name: str, holdout_percent: int) -> str:
    digest = hashlib.sha256(f"{source_kind}\0{tensor_name}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 100
    return "heldout" if bucket < holdout_percent else "train"


def parse_manifest(root: pathlib.Path) -> list[ManifestSlice]:
    result: list[ManifestSlice] = []
    with (root / "manifest.tsv").open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != sample_io.MANIFEST_COLUMNS:
            raise AnalysisError("manifest schema differs from fp8-residency-sample-v1")
        for row in reader:
            reason = sample_io.parse_reason(row["sampling_reason"])
            layer_text = reason["layer"]
            result.append(
                ManifestSlice(
                    source_kind=row["source_kind"],
                    tensor_name=row["tensor_name"],
                    rows=int(row["rows"]),
                    cols=int(row["cols"]),
                    row_begin=int(row["sample_row_begin"]),
                    sample_rows=int(row["sample_rows"]),
                    weight_file=row["weight_file"],
                    scale_file=row["scale_file"],
                    weight_bytes=int(row["weight_bytes"]),
                    scale_bytes=int(row["scale_bytes"]),
                    family=reason["family"],
                    stage=reason["stage"],
                    layer=None if layer_text == "global" else int(layer_text),
                )
            )
    return result


def load_inventory(path: pathlib.Path) -> tuple[dict[tuple[str, str], TensorInventory], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "fp8-residency-sample-v1":
        raise AnalysisError("collection JSON has the wrong schema")
    inventory: dict[tuple[str, str], TensorInventory] = {}
    for source in document.get("source_inventory", []):
        kind = source["source_kind"]
        for tensor in source["tensors"]:
            item = TensorInventory(
                source_kind=kind,
                tensor_name=tensor["tensor_name"],
                family=tensor["family"],
                stage=tensor["stage"],
                layer=tensor.get("layer"),
                ordinal=int(tensor["ordinal"]),
                rows=int(tensor["rows"]),
                cols=int(tensor["cols"]),
                weight_bytes=int(tensor["weight_bytes"]),
                scale_bytes=int(tensor["scale_bytes"]),
            )
            key = (kind, item.tensor_name)
            if key in inventory:
                raise AnalysisError(f"duplicate inventory tensor {key}")
            inventory[key] = item
    if not inventory:
        raise AnalysisError("collection JSON contains no source inventory")
    return inventory, document


def entropy_bits(histogram: np.ndarray) -> float:
    total = int(histogram.sum(dtype=np.uint64))
    if not total:
        return float("nan")
    probabilities = histogram[histogram != 0].astype(np.float64) / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def pair_metrics(joint: np.ndarray) -> dict[str, float | int]:
    total = int(joint.sum(dtype=np.uint64))
    if not total:
        return {
            "pairs": 0,
            "pearson": float("nan"),
            "mutual_information_bits": float("nan"),
            "equal_frequency": float("nan"),
            "mean_absolute_delta": float("nan"),
        }
    matrix = joint.astype(np.float64)
    px = matrix.sum(axis=1)
    py = matrix.sum(axis=0)
    values = np.arange(256, dtype=np.float64)
    mean_x = float(values @ px / total)
    mean_y = float(values @ py / total)
    centered_x = values - mean_x
    centered_y = values - mean_y
    covariance = float(centered_x @ matrix @ centered_y / total)
    variance_x = float((centered_x * centered_x) @ px / total)
    variance_y = float((centered_y * centered_y) @ py / total)
    pearson = covariance / math.sqrt(variance_x * variance_y) if variance_x and variance_y else 0.0

    nz = matrix > 0
    row, col = np.nonzero(nz)
    counts = matrix[row, col]
    pxy = counts / total
    mi = float(
        np.sum(
            pxy
            * np.log2(
                counts * total / (px[row] * py[col])
            )
        )
    )
    delta = np.abs(values[:, None] - values[None, :])
    return {
        "pairs": total,
        "pearson": pearson,
        "mutual_information_bits": mi,
        "equal_frequency": float(np.trace(matrix) / total),
        "mean_absolute_delta": float((matrix * delta).sum() / total),
    }


def add_joint(destination: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
    if left.size != right.size:
        raise AssertionError("pair arrays differ in size")
    if not left.size:
        return
    codes = left.astype(np.uint16, copy=False).ravel().astype(np.uint32)
    codes = codes * 256 + right.astype(np.uint16, copy=False).ravel().astype(np.uint32)
    destination += np.bincount(codes, minlength=PAIR_BINS).reshape(256, 256).astype(np.uint64)


def analyze_bytes_and_pairs(
    root: pathlib.Path, slices: Sequence[ManifestSlice]
) -> tuple[np.ndarray, PairAccumulator]:
    histogram = np.zeros(256, dtype=np.uint64)
    pairs = PairAccumulator()
    for item in slices:
        matrix = np.memmap(
            root / item.weight_file,
            mode="r",
            dtype=np.uint8,
            shape=(item.sample_rows, item.cols),
        )
        histogram += np.bincount(matrix.reshape(-1), minlength=256).astype(np.uint64)
        # Keep pair temporaries bounded for the 16,384-column matrices.
        rows_per_chunk = max(1, (4 << 20) // item.cols)
        for begin in range(0, item.sample_rows, rows_per_chunk):
            end = min(item.sample_rows, begin + rows_per_chunk)
            chunk = np.asarray(matrix[begin:end])
            if item.cols > 1:
                # Exclude the boundary between adjacent group-64 scale domains.
                reshaped = chunk.reshape(end - begin, item.cols // 64, 64)
                add_joint(pairs.horizontal, reshaped[:, :, :-1], reshaped[:, :, 1:])
        if item.sample_rows > 1:
            for begin in range(0, item.sample_rows - 1, rows_per_chunk):
                end = min(item.sample_rows - 1, begin + rows_per_chunk)
                add_joint(
                    pairs.vertical,
                    np.asarray(matrix[begin:end]),
                    np.asarray(matrix[begin + 1 : end + 1]),
                )
        del matrix
    return histogram, pairs


def _histogram_rows(values: np.ndarray, bins: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must be [rows, symbols]")
    rows = values.shape[0]
    offsets = np.arange(rows, dtype=np.int64)[:, None] * bins
    codes = values.astype(np.int64, copy=False) + offsets
    return np.bincount(codes.ravel(), minlength=rows * bins).reshape(rows, bins)


def _best_palette_padded(
    histogram: np.ndarray,
    symbols: int,
    maximum_palette: int,
    fixed_bytes: int,
    *,
    exception_denominator: int = 1,
) -> np.ndarray:
    rows = histogram.shape[0]
    maximum_palette = min(maximum_palette, histogram.shape[1])
    if maximum_palette:
        boundary = histogram.shape[1] - maximum_palette
        top = np.partition(histogram, boundary, axis=1)[:, boundary:]
        top.sort(axis=1)
        top = top[:, ::-1]
        cumulative = np.cumsum(top, axis=1, dtype=np.int64)
    else:
        cumulative = np.empty((rows, 0), dtype=np.int64)
    best_padded = np.full(rows, np.iinfo(np.int64).max, dtype=np.int64)
    best_stored = np.full(rows, np.iinfo(np.int64).max, dtype=np.int64)
    for palette_count in range(maximum_palette + 1):
        covered = (
            np.zeros(rows, dtype=np.int64)
            if palette_count == 0
            else cumulative[:, palette_count - 1]
        )
        exceptions = symbols - covered
        exception_bytes = (exceptions + exception_denominator - 1) // exception_denominator
        stored = fixed_bytes + palette_count + exception_bytes
        padded = (stored + 15) & -16
        better = (padded < best_padded) | ((padded == best_padded) & (stored < best_stored))
        best_padded[better] = padded[better]
        best_stored[better] = stored[better]
    return best_padded


def _xor_exception_counts(tiles: np.ndarray) -> np.ndarray:
    rows, symbols = tiles.shape
    magnitudes = (tiles & 0x7F).reshape(rows * (symbols // 32), 32)
    bases = np.empty(magnitudes.shape[0], dtype=np.uint8)
    lanes_per_chunk = 32768
    for begin in range(0, magnitudes.shape[0], lanes_per_chunk):
        end = min(magnitudes.shape[0], begin + lanes_per_chunk)
        histogram = _histogram_rows(magnitudes[begin:end], 128)
        bases[begin:end] = np.argmax(histogram, axis=1).astype(np.uint8)
    residual = magnitudes ^ bases[:, None]
    return (residual >= 15).sum(axis=1, dtype=np.int64).reshape(rows, symbols // 32).sum(
        axis=1, dtype=np.int64
    )


def measure_tile_batch(tiles: np.ndarray) -> BatchMeasurement:
    tiles = np.ascontiguousarray(tiles, dtype=np.uint8)
    if tiles.ndim != 2 or tiles.shape[1] % 64:
        raise ValueError("tiles must be [N, positive multiple of 64]")
    count, symbols = tiles.shape
    full_hist = _histogram_rows(tiles, 256)
    mag_hist = full_hist[:, :128] + full_hist[:, 128:]
    exp_hist = np.empty((count, 16), dtype=np.int64)
    for exponent in range(16):
        begin = exponent * 8
        exp_hist[:, exponent] = (
            full_hist[:, begin : begin + 8].sum(axis=1, dtype=np.int64)
            + full_hist[:, 128 + begin : 128 + begin + 8].sum(axis=1, dtype=np.int64)
        )

    raw_padded = np.full(count, codec.align_up(symbols), dtype=np.int64)
    sign_bytes = (symbols + 7) // 8
    mantissa_bytes = (symbols * 3 + 7) // 8
    candidates: dict[str, np.ndarray] = {}
    for bits in (4, 5, 6):
        index_bytes = (symbols * bits + 7) // 8
        candidates[f"byte_palette{bits}"] = _best_palette_padded(
            full_hist, symbols, (1 << bits) - 1, index_bytes
        )
        candidates[f"mag_palette{bits}"] = _best_palette_padded(
            mag_hist, symbols, (1 << bits) - 1, sign_bytes + index_bytes
        )
    for bits in (2, 3):
        index_bytes = (symbols * bits + 7) // 8
        candidates[f"exp_palette{bits}"] = _best_palette_padded(
            exp_hist,
            symbols,
            (1 << bits) - 1,
            sign_bytes + mantissa_bytes + index_bytes,
            exception_denominator=2,
        )

    xor_exceptions = _xor_exception_counts(tiles)
    xor_stored = sign_bytes + symbols // 32 + (symbols + 1) // 2 + xor_exceptions
    candidates["mag_xor4"] = (xor_stored + 15) & -16

    zero_count = mag_hist[:, 0]
    zero_stored = (symbols + 7) // 8 + (zero_count + 7) // 8 + (symbols - zero_count)
    candidates["zero_sparse"] = (zero_stored + 15) & -16

    bit_values = np.arange(256, dtype=np.uint16)[:, None]
    bit_masks = ((bit_values >> np.arange(8, dtype=np.uint16)) & 1).astype(np.int64)
    ones = full_hist @ bit_masks
    variable_planes = ((ones != 0) & (ones != symbols)).sum(axis=1, dtype=np.int64)
    bitplane_stored = 2 + variable_planes * ((symbols + 7) // 8)
    candidates["bitplane_const"] = (bitplane_stored + 15) & -16

    selected: dict[str, np.ndarray] = {}
    raw_selected: dict[str, np.ndarray] = {}
    for name, padded in candidates.items():
        use_raw = padded >= raw_padded
        selected[name] = np.where(use_raw, raw_padded, padded)
        raw_selected[name] = use_raw

    # Match the reference encoder's decode-cost tie order exactly.  RAW is
    # row zero and wins every equal-padded-byte tie.
    stack_names = ["raw", *MIXED_FORMAT_PRIORITY]
    stack = np.vstack(
        [raw_padded, *(candidates[name] for name in MIXED_FORMAT_PRIORITY)]
    )
    mixed_choice = np.argmin(stack, axis=0)  # RAW is row zero and wins ties.
    selected["mixed"] = stack[mixed_choice, np.arange(count)]
    raw_selected["mixed"] = mixed_choice == 0
    mixed_modes = Counter()
    for index, occurrences in zip(*np.unique(mixed_choice, return_counts=True)):
        mixed_modes[stack_names[int(index)]] += int(occurrences)

    return BatchMeasurement(
        tiles=count,
        raw_bytes=count * symbols,
        selected_padded=selected,
        raw_selected=raw_selected,
        mixed_mode_counts=mixed_modes,
        distinct=np.count_nonzero(full_hist, axis=1),
        positive_zero=full_hist[:, 0],
        negative_zero=full_hist[:, 128],
    )


def iter_tile_batches(
    root: pathlib.Path,
    item: ManifestSlice,
    tile_bytes: int,
    *,
    target_values: int = 2 << 20,
) -> Iterator[np.ndarray]:
    tile_rows = tile_bytes // 64
    groups = item.cols // 64
    matrix = np.memmap(
        root / item.weight_file,
        mode="r",
        dtype=np.uint8,
        shape=(item.sample_rows, item.cols),
    )
    sample_begin = item.row_begin
    sample_end = item.row_begin + item.sample_rows
    first_full = ((sample_begin + tile_rows - 1) // tile_rows) * tile_rows
    last_full = (sample_end // tile_rows) * tile_rows
    max_tiles = max(groups, target_values // tile_bytes)
    blocks_per_batch = max(1, max_tiles // groups)
    for global_row in range(first_full, last_full, blocks_per_batch * tile_rows):
        block_count = min(
            blocks_per_batch,
            (last_full - global_row) // tile_rows,
        )
        local = global_row - sample_begin
        source = np.asarray(matrix[local : local + block_count * tile_rows])
        tiles = (
            source.reshape(block_count, tile_rows, groups, 64)
            .transpose(0, 2, 1, 3)
            .reshape(block_count * groups, tile_bytes)
        )
        yield np.ascontiguousarray(tiles)

    # Include the source matrix's final partial row block only when this slice
    # contains it completely.  Interior sample boundaries never synthesize the
    # missing rows of a tensor-core tile.
    tail_begin = (item.rows // tile_rows) * tile_rows
    if item.rows % tile_rows and sample_begin <= tail_begin and sample_end == item.rows:
        local = tail_begin - sample_begin
        valid_rows = item.rows - tail_begin
        source = np.asarray(matrix[local : local + valid_rows])
        tiles = (
            source.reshape(1, valid_rows, groups, 64)
            .transpose(0, 2, 1, 3)
            .reshape(groups, valid_rows * 64)
        )
        yield np.ascontiguousarray(tiles)
    del matrix


def quantiles(chunks: Sequence[np.ndarray], probabilities: Sequence[float]) -> list[float]:
    if not chunks:
        return [float("nan") for _ in probabilities]
    values = np.concatenate(chunks)
    return [float(value) for value in np.quantile(values, probabilities)]


def csv_write(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tensor_frequency_row(analysis: TensorAnalysis) -> dict[str, Any]:
    hist = analysis.histogram
    total = int(hist.sum(dtype=np.uint64))
    positive_zero = int(hist[0])
    negative_zero = int(hist[128])
    nan_count = int(hist[0x7F] + hist[0xFF])
    row: dict[str, Any] = {
        "source_kind": analysis.inventory.source_kind,
        "tensor_name": analysis.inventory.tensor_name,
        "family": analysis.inventory.family,
        "stage": analysis.inventory.stage,
        "layer": "" if analysis.inventory.layer is None else analysis.inventory.layer,
        "split": analysis.split,
        "rows": analysis.inventory.rows,
        "cols": analysis.inventory.cols,
        "full_weight_bytes": analysis.inventory.weight_bytes,
        "full_scale_bytes": analysis.inventory.scale_bytes,
        "sample_rows": analysis.sample_rows,
        "sample_weight_bytes": analysis.sample_weight_bytes,
        "sample_scale_bytes": analysis.sample_scale_bytes,
        "h0_bits_per_byte": entropy_bits(hist),
        "distinct_byte_values": int(np.count_nonzero(hist)),
        "positive_zero_frequency": positive_zero / total,
        "negative_zero_frequency": negative_zero / total,
        "magnitude_zero_frequency": (positive_zero + negative_zero) / total,
        "nan_byte_frequency": nan_count / total,
    }
    values = np.arange(256, dtype=np.uint16)
    for bit in range(8):
        ones = int(hist[((values >> bit) & 1).astype(bool)].sum(dtype=np.uint64))
        row[f"bit{bit}_one_frequency"] = ones / total
    for prefix, joint in (
        ("horizontal", analysis.pairs.horizontal),
        ("vertical", analysis.pairs.vertical),
    ):
        for key, value in pair_metrics(joint).items():
            row[f"{prefix}_{key}"] = value
    return row


def tile_rows_for_tensor(analysis: TensorAnalysis) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    statistic_rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    inventory = analysis.inventory
    for tile_bytes, accumulator in sorted(analysis.tiles.items()):
        if not accumulator.tile_count or not accumulator.raw_bytes:
            continue
        q_distinct = quantiles(accumulator.distinct_chunks, (0.5, 0.9, 0.99))
        q_positive = quantiles(accumulator.positive_zero_chunks, (0.5, 0.9, 0.99))
        q_negative = quantiles(accumulator.negative_zero_chunks, (0.5, 0.9, 0.99))
        distinct_total = sum(int(chunk.sum(dtype=np.int64)) for chunk in accumulator.distinct_chunks)
        pos_total = sum(int(chunk.sum(dtype=np.int64)) for chunk in accumulator.positive_zero_chunks)
        neg_total = sum(int(chunk.sum(dtype=np.int64)) for chunk in accumulator.negative_zero_chunks)
        statistic_rows.append(
            {
                "source_kind": inventory.source_kind,
                "tensor_name": inventory.tensor_name,
                "family": inventory.family,
                "stage": inventory.stage,
                "split": analysis.split,
                "tile_bytes": tile_bytes,
                "sample_tile_count": accumulator.tile_count,
                "analyzed_raw_bytes": accumulator.raw_bytes,
                "sample_weight_bytes": analysis.sample_weight_bytes,
                "tile_coverage": accumulator.raw_bytes / analysis.sample_weight_bytes,
                "distinct_mean": distinct_total / accumulator.tile_count,
                "distinct_p50": q_distinct[0],
                "distinct_p90": q_distinct[1],
                "distinct_p99": q_distinct[2],
                "positive_zero_per_tile_mean": pos_total / accumulator.tile_count,
                "positive_zero_per_tile_p50": q_positive[0],
                "positive_zero_per_tile_p90": q_positive[1],
                "positive_zero_per_tile_p99": q_positive[2],
                "negative_zero_per_tile_mean": neg_total / accumulator.tile_count,
                "negative_zero_per_tile_p50": q_negative[0],
                "negative_zero_per_tile_p90": q_negative[1],
                "negative_zero_per_tile_p99": q_negative[2],
            }
        )
        full_tile_count = codec.expected_tile_count(inventory.rows, inventory.cols, tile_bytes)
        for format_name in FORMAT_NAMES:
            padded = accumulator.padded_bytes[format_name]
            directory = codec.align_up(accumulator.tile_count * 8)
            sample_encoded = codec.HEADER_BYTES + directory + padded
            sample_ratio = sample_encoded / accumulator.raw_bytes
            payload_per_raw = padded / accumulator.raw_bytes
            estimated_payload = int(round(payload_per_raw * inventory.weight_bytes))
            estimated_container = (
                codec.HEADER_BYTES
                + codec.align_up(full_tile_count * 8)
                + estimated_payload
            )
            effective = min(inventory.weight_bytes, estimated_container)
            codec_rows.append(
                {
                    "source_kind": inventory.source_kind,
                    "tensor_name": inventory.tensor_name,
                    "family": inventory.family,
                    "stage": inventory.stage,
                    "split": analysis.split,
                    "rows": inventory.rows,
                    "cols": inventory.cols,
                    "full_weight_bytes": inventory.weight_bytes,
                    "full_scale_bytes": inventory.scale_bytes,
                    "tile_bytes": tile_bytes,
                    "format": format_name,
                    "sample_tile_count": accumulator.tile_count,
                    "sample_raw_bytes": accumulator.raw_bytes,
                    "sample_padded_payload_bytes": padded,
                    "sample_directory_bytes": directory,
                    "sample_container_bytes": sample_encoded,
                    "sample_padded_ratio": sample_ratio,
                    "raw_tile_fraction": accumulator.raw_tiles[format_name] / accumulator.tile_count,
                    "estimated_full_tile_count": full_tile_count,
                    "estimated_full_container_bytes": estimated_container,
                    "matrix_raw_escape": int(estimated_container >= inventory.weight_bytes),
                    "estimated_effective_weight_bytes": effective,
                    "estimated_full_weight_ratio": effective / inventory.weight_bytes,
                    "estimated_full_total_ratio_with_raw_scales": (
                        effective + inventory.scale_bytes
                    )
                    / (inventory.weight_bytes + inventory.scale_bytes),
                    "mixed_mode_counts": (
                        json.dumps(dict(sorted(accumulator.mixed_modes.items())), separators=(",", ":"))
                        if format_name == "mixed"
                        else ""
                    ),
                }
            )
    return statistic_rows, codec_rows


def aggregate_family_rows(
    codec_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, int, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in codec_rows:
        key = (
            str(row["source_kind"]),
            str(row["family"]),
            str(row["stage"]),
            str(row["split"]),
            int(row["tile_bytes"]),
            str(row["format"]),
        )
        aggregate = groups[key]
        weight = int(row["full_weight_bytes"])
        scale = int(row["full_scale_bytes"])
        aggregate["tensor_count"] += 1
        aggregate["full_weight_bytes"] += weight
        aggregate["full_scale_bytes"] += scale
        aggregate["effective_weight_bytes"] += float(row["estimated_effective_weight_bytes"])
        aggregate["effective_total_bytes"] += float(row["estimated_effective_weight_bytes"]) + scale
        aggregate["raw_tile_weighted"] += float(row["raw_tile_fraction"]) * int(row["sample_raw_bytes"])
        aggregate["sample_raw_bytes"] += int(row["sample_raw_bytes"])
    rows: list[dict[str, Any]] = []
    for key, aggregate in sorted(groups.items()):
        kind, family, stage, split, tile_bytes, format_name = key
        weight = aggregate["full_weight_bytes"]
        scale = aggregate["full_scale_bytes"]
        rows.append(
            {
                "source_kind": kind,
                "family": family,
                "stage": stage,
                "split": split,
                "tile_bytes": tile_bytes,
                "format": format_name,
                "tensor_count": int(aggregate["tensor_count"]),
                "full_weight_bytes": int(weight),
                "full_scale_bytes": int(scale),
                "estimated_effective_weight_bytes": int(round(aggregate["effective_weight_bytes"])),
                "estimated_weight_ratio": aggregate["effective_weight_bytes"] / weight,
                "estimated_total_ratio_with_raw_scales": aggregate["effective_total_bytes"] / (weight + scale),
                "raw_tile_fraction": aggregate["raw_tile_weighted"] / aggregate["sample_raw_bytes"],
            }
        )
    return rows


def _config_tie_key(config: tuple[int, str]) -> tuple[int, int]:
    tile_bytes, format_name = config
    return (
        TILE_SELECTION_RANK.get(tile_bytes, 1 << 20),
        FORMAT_SELECTION_RANK.get(format_name, 1 << 20),
    )


def _layout_bytes(
    inventory: Iterable[TensorInventory],
    family_ratios: Mapping[tuple[str, str], float],
) -> tuple[int, int, int]:
    """Return new disk bytes, logical VRAM bytes, and compressed weight bytes.

    This models the quantizers' current 4096-byte weight/scale span alignment.
    It is an allocator estimate, not evidence that the live arena consumed the
    reclaimed bytes.
    """
    cursor = 0
    logical = 0
    encoded_weights = 0
    for item in sorted(inventory, key=lambda value: value.ordinal):
        ratio = family_ratios[(item.source_kind, item.family)]
        weight_bytes = min(item.weight_bytes, int(round(item.weight_bytes * ratio)))
        cursor = (cursor + DISK_ALIGNMENT - 1) & -DISK_ALIGNMENT
        cursor += weight_bytes
        cursor = (cursor + DISK_ALIGNMENT - 1) & -DISK_ALIGNMENT
        cursor += item.scale_bytes
        cursor = (cursor + DISK_ALIGNMENT - 1) & -DISK_ALIGNMENT
        logical += weight_bytes + item.scale_bytes
        encoded_weights += weight_bytes
    return cursor, logical, encoded_weights


def select_configs_and_heldout(
    codec_rows: Sequence[Mapping[str, Any]],
    inventory: Mapping[tuple[str, str], TensorInventory],
    *,
    entropy_by_tensor: Mapping[tuple[str, str], float],
    source_data_bytes: Mapping[str, int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    allocator_slot_bytes: int = DEFAULT_EXPANDED_EXPERT_SLOT_BYTES,
) -> dict[str, Any]:
    train_groups: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in codec_rows:
        tensor_key = (str(row["source_kind"]), str(row["tensor_name"]))
        if row["split"] == "train":
            train_groups[(tensor_key[0], str(row["family"]), int(row["tile_bytes"]), str(row["format"]))].append(row)

    kinds = sorted({item.source_kind for item in inventory.values()})
    global_choices: dict[str, tuple[int, str]] = {}
    for kind in kinds:
        candidates: dict[tuple[int, str], tuple[float, int]] = defaultdict(lambda: (0.0, 0))
        for row in codec_rows:
            if row["source_kind"] != kind or row["split"] != "train":
                continue
            config = (int(row["tile_bytes"]), str(row["format"]))
            encoded, weight = candidates[config]
            candidates[config] = (
                encoded + float(row["estimated_effective_weight_bytes"]),
                weight + int(row["full_weight_bytes"]),
            )
        if not candidates:
            raise AnalysisError(f"no training tensors for source kind {kind}")
        global_choices[kind] = min(
            candidates,
            key=lambda config: (
                candidates[config][0] / candidates[config][1],
                *_config_tie_key(config),
            ),
        )

    sampled_families = sorted({(str(row["source_kind"]), str(row["family"])) for row in codec_rows})
    family_choices: dict[tuple[str, str], tuple[int, str]] = {}
    selection_rows: list[dict[str, Any]] = []
    for kind, family in sampled_families:
        candidates: dict[tuple[int, str], tuple[float, int, int]] = {}
        for tile_bytes in TILE_SIZES:
            for format_name in FORMAT_NAMES:
                rows = train_groups.get((kind, family, tile_bytes, format_name), [])
                if not rows:
                    continue
                encoded = sum(float(row["estimated_effective_weight_bytes"]) for row in rows)
                weight = sum(int(row["full_weight_bytes"]) for row in rows)
                candidates[(tile_bytes, format_name)] = (encoded, weight, len(rows))
        fallback = False
        if candidates:
            choice = min(
                candidates,
                key=lambda config: (
                    candidates[config][0] / candidates[config][1],
                    *_config_tie_key(config),
                ),
            )
            train_ratio = candidates[choice][0] / candidates[choice][1]
            train_tensors = candidates[choice][2]
        else:
            choice = global_choices[kind]
            train_ratio = float("nan")
            train_tensors = 0
            fallback = True
        family_choices[(kind, family)] = choice
        selection_rows.append(
            {
                "source_kind": kind,
                "family": family,
                "tile_bytes": choice[0],
                "format": choice[1],
                "train_weight_ratio": train_ratio,
                "train_tensor_count": train_tensors,
                "used_global_train_fallback": fallback,
            }
        )

    heldout_items: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in codec_rows:
        if row["split"] != "heldout":
            continue
        kind = str(row["source_kind"])
        family = str(row["family"])
        choice = family_choices[(kind, family)]
        if int(row["tile_bytes"]) != choice[0] or str(row["format"]) != choice[1]:
            continue
        heldout_items[(kind, family)].append(
            {
                "tensor_name": row["tensor_name"],
                "weight_bytes": int(row["full_weight_bytes"]),
                "scale_bytes": int(row["full_scale_bytes"]),
                "weight_ratio": float(row["estimated_full_weight_ratio"]),
                "total_ratio": float(row["estimated_full_total_ratio_with_raw_scales"]),
                "h0_bits_per_byte": float(
                    entropy_by_tensor[(kind, str(row["tensor_name"]))]
                ),
            }
        )

    inventory_family: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in inventory.values():
        aggregate = inventory_family[(item.source_kind, item.family)]
        aggregate["weight_bytes"] += item.weight_bytes
        aggregate["scale_bytes"] += item.scale_bytes
        aggregate["tensor_count"] += 1

    global_heldout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (kind, _family), items in heldout_items.items():
        global_heldout[kind].extend(items)
    for kind in kinds:
        if not global_heldout[kind]:
            raise AnalysisError(f"no held-out tensors for source kind {kind}")

    family_results: list[dict[str, Any]] = []
    weighted_results: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        weighted_effective_weight = 0.0
        weighted_entropy = 0.0
        total_weight = 0
        total_scale = 0
        fallback_weight = 0
        for (entry_kind, family), volume in sorted(inventory_family.items()):
            if entry_kind != kind:
                continue
            items = heldout_items.get((kind, family), [])
            used_fallback = not items
            source_items = items if items else global_heldout[kind]
            denominator = sum(item["weight_bytes"] for item in source_items)
            ratio = sum(item["weight_bytes"] * item["weight_ratio"] for item in source_items) / denominator
            h0 = sum(
                item["weight_bytes"] * item["h0_bits_per_byte"] for item in source_items
            ) / denominator
            weight = volume["weight_bytes"]
            scale = volume["scale_bytes"]
            weighted_effective_weight += ratio * weight
            weighted_entropy += h0 * weight
            total_weight += weight
            total_scale += scale
            if used_fallback:
                fallback_weight += weight
            choice = family_choices.get((kind, family), global_choices[kind])
            family_results.append(
                {
                    "source_kind": kind,
                    "family": family,
                    "tile_bytes": choice[0],
                    "format": choice[1],
                    "heldout_tensor_count": len(items),
                    "heldout_weight_ratio": ratio,
                    "heldout_h0_bits_per_byte": h0,
                    "full_family_weight_bytes": weight,
                    "full_family_scale_bytes": scale,
                    "used_global_heldout_fallback": used_fallback,
                }
            )
        weight_ratio = weighted_effective_weight / total_weight
        total_ratio = (weighted_effective_weight + total_scale) / (total_weight + total_scale)
        weighted_results[kind] = {
            "weight_bytes": total_weight,
            "scale_bytes": total_scale,
            "heldout_weight_ratio": weight_ratio,
            "heldout_total_ratio_with_raw_scales": total_ratio,
            "weighted_heldout_h0_bits_per_byte": weighted_entropy / total_weight,
            "estimated_weight_bytes_saved": int(round(total_weight - weighted_effective_weight)),
            "estimated_total_bytes_saved": int(round(total_weight - weighted_effective_weight)),
            "family_fallback_weight_bytes": fallback_weight,
        }

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap: dict[str, list[float]] = {kind: [] for kind in kinds}
    bootstrap_h0: dict[str, list[float]] = {kind: [] for kind in kinds}
    family_keys_by_kind = {
        kind: [key for key in inventory_family if key[0] == kind] for kind in kinds
    }
    for _ in range(bootstrap_replicates):
        for kind in kinds:
            effective = 0.0
            entropy = 0.0
            total_weight = 0
            for key in family_keys_by_kind[kind]:
                items = heldout_items.get(key, [])
                source_items = items if items else global_heldout[kind]
                indexes = rng.integers(0, len(source_items), size=len(source_items))
                selected = [source_items[int(index)] for index in indexes]
                denominator = sum(item["weight_bytes"] for item in selected)
                ratio = sum(item["weight_bytes"] * item["weight_ratio"] for item in selected) / denominator
                weight = inventory_family[key]["weight_bytes"]
                h0 = sum(
                    item["weight_bytes"] * item["h0_bits_per_byte"] for item in selected
                ) / denominator
                effective += ratio * weight
                entropy += h0 * weight
                total_weight += weight
            bootstrap[kind].append(effective / total_weight)
            bootstrap_h0[kind].append(entropy / total_weight)
    for kind in kinds:
        values = np.asarray(bootstrap[kind], dtype=np.float64)
        h0_values = np.asarray(bootstrap_h0[kind], dtype=np.float64)
        result = weighted_results[kind]
        result["bootstrap_replicates"] = bootstrap_replicates
        result["weight_ratio_ci95"] = [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        weight = result["weight_bytes"]
        scale = result["scale_bytes"]
        total_values = (values * weight + scale) / (weight + scale)
        result["total_ratio_with_raw_scales_ci95"] = [
            float(np.quantile(total_values, 0.025)),
            float(np.quantile(total_values, 0.975)),
        ]
        result["h0_bits_per_byte_ci95"] = [
            float(np.quantile(h0_values, 0.025)),
            float(np.quantile(h0_values, 0.975)),
        ]

    family_ratio_map = {
        (str(row["source_kind"]), str(row["family"])): float(row["heldout_weight_ratio"])
        for row in family_results
    }
    for kind in kinds:
        source_inventory = [item for item in inventory.values() if item.source_kind == kind]
        new_disk, new_logical, encoded_weights = _layout_bytes(
            source_inventory, family_ratio_map
        )
        result = weighted_results[kind]
        current_logical = result["weight_bytes"] + result["scale_bytes"]
        current_disk = int(source_data_bytes.get(kind, current_logical))
        vram_saved = current_logical - new_logical
        disk_saved = current_disk - new_disk
        potential_slots = max(0, vram_saved // allocator_slot_bytes)
        high_volume_min = max(
            HIGH_VOLUME_FAMILY_MIN_BYTES,
            int(math.ceil(result["weight_bytes"] * HIGH_VOLUME_FAMILY_FRACTION)),
        )
        high_volume = [
            row for row in family_results
            if row["source_kind"] == kind
            and row["heldout_tensor_count"] > 0
            and row["full_family_weight_bytes"] >= high_volume_min
        ]
        winner = min(high_volume, key=lambda row: row["heldout_weight_ratio"]) if high_volume else None
        h0 = result["weighted_heldout_h0_bits_per_byte"]
        ratio = result["heldout_weight_ratio"]
        kill = h0 > H0_KILL_BITS_PER_BYTE and ratio > PADDED_RATIO_KILL
        ratio_gate = ratio <= PADDED_RATIO_GATE and bool(
            winner and winner["heldout_weight_ratio"] <= HIGH_VOLUME_RATIO_GATE
        )
        allocator_model_gate = (
            vram_saved >= ALLOCATOR_RECLAIM_GATE_BYTES
            and potential_slots >= ALLOCATOR_RECLAIM_GATE_SLOTS
        )
        if kill:
            decision = "stop_before_cuda"
        elif ratio_gate:
            decision = "eligible_for_gpu_microbenchmark"
        elif allocator_model_gate:
            decision = "eligible_for_allocator_only_probe"
        else:
            decision = "stop_or_revisit_fixed_block_formats"
        result.update(
            {
                "estimated_encoded_weight_bytes_layout_model": encoded_weights,
                "estimated_logical_vram_bytes": new_logical,
                "estimated_logical_vram_bytes_saved": vram_saved,
                "current_disk_data_bytes": current_disk,
                "estimated_versioned_cache_disk_bytes": new_disk,
                "estimated_disk_bytes_saved": disk_saved,
                "estimated_disk_ratio": new_disk / current_disk,
                "allocator_slot_bytes_assumed": allocator_slot_bytes,
                "potential_expanded_expert_slots": potential_slots,
                "allocator_reclaim_is_modeled_not_measured": True,
                "kill_entropy_threshold_bits_per_byte": H0_KILL_BITS_PER_BYTE,
                "kill_ratio_threshold": PADDED_RATIO_KILL,
                "kill_criterion_met": kill,
                "ratio_gate_threshold": PADDED_RATIO_GATE,
                "high_volume_family_min_bytes": high_volume_min,
                "high_volume_family_min_fraction": HIGH_VOLUME_FAMILY_FRACTION,
                "high_volume_family_ratio_threshold": HIGH_VOLUME_RATIO_GATE,
                "best_high_volume_family": winner,
                "ratio_gate_passed": ratio_gate,
                "allocator_model_gate_bytes": ALLOCATOR_RECLAIM_GATE_BYTES,
                "allocator_model_gate_slots": ALLOCATOR_RECLAIM_GATE_SLOTS,
                "allocator_model_gate_passed": allocator_model_gate,
                "decision": decision,
            }
        )

    return {
        "split_method": "sha256(source_kind\\0tensor_name) modulo 100",
        "global_train_choices": {
            kind: {"tile_bytes": choice[0], "format": choice[1]}
            for kind, choice in global_choices.items()
        },
        "family_train_choices": selection_rows,
        "family_heldout_results": family_results,
        "weighted_whole_cache_heldout": weighted_results,
        "bootstrap_seed": bootstrap_seed,
        "selection_tie_break": {
            "tile_bytes": list(TILE_SELECTION_PRIORITY),
            "formats": list(FORMAT_SELECTION_PRIORITY),
        },
    }


def analyze(
    root: pathlib.Path,
    inventory: Mapping[tuple[str, str], TensorInventory],
    *,
    holdout_percent: int,
    maximum_sample_weight_bytes: int | None,
) -> tuple[list[TensorAnalysis], dict[tuple[str, str, str], PairAccumulator]]:
    manifest = parse_manifest(root)
    grouped: dict[tuple[str, str], list[ManifestSlice]] = defaultdict(list)
    for item in manifest:
        grouped[(item.source_kind, item.tensor_name)].append(item)
    analyses: list[TensorAnalysis] = []
    family_pairs: dict[tuple[str, str, str], PairAccumulator] = defaultdict(PairAccumulator)
    consumed = 0
    started = time.monotonic()
    for ordinal, (key, slices) in enumerate(sorted(grouped.items()), 1):
        if maximum_sample_weight_bytes is not None and consumed >= maximum_sample_weight_bytes:
            break
        item_inventory = inventory.get(key)
        if item_inventory is None:
            raise AnalysisError(f"sample tensor missing from collection inventory: {key}")
        first = slices[0]
        for item in slices:
            if (
                item.rows != item_inventory.rows
                or item.cols != item_inventory.cols
                or item.family != item_inventory.family
                or item.stage != item_inventory.stage
            ):
                raise AnalysisError(f"sample/inventory disagreement for {key}")
        histogram, pairs = analyze_bytes_and_pairs(root, slices)
        tile_accumulators = {size: TileAccumulator(size) for size in TILE_SIZES}
        for tile_bytes in TILE_SIZES:
            accumulator = tile_accumulators[tile_bytes]
            for item in slices:
                for tiles in iter_tile_batches(root, item, tile_bytes):
                    accumulator.add(measure_tile_batch(tiles))
        sample_weight = sum(item.weight_bytes for item in slices)
        sample_scale = sum(item.scale_bytes for item in slices)
        sample_rows = sum(item.sample_rows for item in slices)
        consumed += sample_weight
        split = sha_split(key[0], key[1], holdout_percent)
        analysis = TensorAnalysis(
            inventory=item_inventory,
            split=split,
            sample_rows=sample_rows,
            sample_weight_bytes=sample_weight,
            sample_scale_bytes=sample_scale,
            histogram=histogram,
            pairs=pairs,
            tiles=tile_accumulators,
        )
        analyses.append(analysis)
        family_key = (item_inventory.source_kind, item_inventory.family, item_inventory.stage)
        family_pairs[family_key].horizontal += pairs.horizontal
        family_pairs[family_key].vertical += pairs.vertical
        elapsed = time.monotonic() - started
        rate = consumed / elapsed / 1e6 if elapsed else 0.0
        print(
            f"[{ordinal:4d}/{len(grouped)}] {consumed / 2**20:8.2f} MiB "
            f"{rate:6.1f} MB/s {key[0]} {key[1]}",
            flush=True,
        )
    return analyses, family_pairs


def build_family_frequency_rows(
    analyses: Sequence[TensorAnalysis],
    family_pairs: Mapping[tuple[str, str, str], PairAccumulator],
) -> list[dict[str, Any]]:
    histograms: dict[tuple[str, str, str, str], np.ndarray] = defaultdict(
        lambda: np.zeros(256, dtype=np.uint64)
    )
    bytes_by_key: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for analysis in analyses:
        key = (
            analysis.inventory.source_kind,
            analysis.inventory.family,
            analysis.inventory.stage,
            analysis.split,
        )
        histograms[key] += analysis.histogram
        bytes_by_key[key] += analysis.sample_weight_bytes
    rows: list[dict[str, Any]] = []
    values = np.arange(256, dtype=np.uint16)
    for key, histogram in sorted(histograms.items()):
        kind, family, stage, split = key
        total = int(histogram.sum(dtype=np.uint64))
        pair = family_pairs[(kind, family, stage)]
        row: dict[str, Any] = {
            "source_kind": kind,
            "family": family,
            "stage": stage,
            "split": split,
            "sample_weight_bytes": bytes_by_key[key],
            "h0_bits_per_byte": entropy_bits(histogram),
            "distinct_byte_values": int(np.count_nonzero(histogram)),
            "positive_zero_frequency": int(histogram[0]) / total,
            "negative_zero_frequency": int(histogram[128]) / total,
            "magnitude_zero_frequency": int(histogram[0] + histogram[128]) / total,
            "nan_byte_frequency": int(histogram[0x7F] + histogram[0xFF]) / total,
        }
        for bit in range(8):
            row[f"bit{bit}_one_frequency"] = int(
                histogram[((values >> bit) & 1).astype(bool)].sum(dtype=np.uint64)
            ) / total
        # Family pair accumulators currently combine train and held-out data;
        # label them once on the all-split row to avoid implying split purity.
        if split == sorted({entry[3] for entry in histograms if entry[:3] == key[:3]})[0]:
            for prefix, joint in (("horizontal", pair.horizontal), ("vertical", pair.vertical)):
                for metric, value in pair_metrics(joint).items():
                    row[f"all_split_{prefix}_{metric}"] = value
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure real E4M3 entropy and exact padded fixed-block ratios on "
            "tensor-core-aligned group-64 tiles"
        )
    )
    parser.add_argument(
        "sample", type=pathlib.Path,
        help="validated sample directory, archive, or first consecutively numbered part",
    )
    parser.add_argument(
        "--collection-json", type=pathlib.Path, required=True,
        help="collector JSON containing the complete source tensor inventory",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, required=True,
        help="directory for machine-readable CSV/JSON results",
    )
    parser.add_argument("--minimum-dense-weight-mib", type=float, default=256.0)
    parser.add_argument("--holdout-percent", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0x5A10F8)
    parser.add_argument(
        "--allocator-slot-mib", type=float, default=13.5,
        help="modeled expanded-expert slot bytes; never reported as live allocator proof",
    )
    parser.add_argument(
        "--max-sample-weight-mib", type=float,
        help="bounded smoke only; a truncated run is not a coding decision",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.holdout_percent <= 50:
        parser.error("--holdout-percent must be 1..50")
    if args.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    if args.allocator_slot_mib <= 0:
        parser.error("--allocator-slot-mib must be positive")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory, collection = load_inventory(args.collection_json.resolve())
    maximum = (
        None
        if args.max_sample_weight_mib is None
        else int(round(args.max_sample_weight_mib * (1 << 20)))
    )
    with sample_io.materialize_sample(args.sample) as root:
        validation = sample_io.validate_sample_directory(
            root,
            minimum_dense_weight_bytes=int(round(args.minimum_dense_weight_mib * (1 << 20))),
        )
        analyses, family_pairs = analyze(
            root,
            inventory,
            holdout_percent=args.holdout_percent,
            maximum_sample_weight_bytes=maximum,
        )

    frequency_rows = [tensor_frequency_row(item) for item in analyses]
    tile_rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    for item in analyses:
        tensor_tiles, tensor_codecs = tile_rows_for_tensor(item)
        tile_rows.extend(tensor_tiles)
        codec_rows.extend(tensor_codecs)
    family_frequency_rows = build_family_frequency_rows(analyses, family_pairs)
    family_codec_rows = aggregate_family_rows(codec_rows)
    entropy_by_tensor = {
        (item.inventory.source_kind, item.inventory.tensor_name): entropy_bits(item.histogram)
        for item in analyses
    }
    source_data_bytes = {
        str(source["source_kind"]): int(source["data_bytes"])
        for source in collection.get("source_inventory", [])
    }
    heldout = select_configs_and_heldout(
        codec_rows,
        inventory,
        entropy_by_tensor=entropy_by_tensor,
        source_data_bytes=source_data_bytes,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        allocator_slot_bytes=int(round(args.allocator_slot_mib * (1 << 20))),
    )

    csv_write(output / "tensor_inventory.csv", frequency_rows)
    csv_write(output / "family_entropy.csv", family_frequency_rows)
    csv_write(output / "tensor_tile_stats.csv", tile_rows)
    csv_write(output / "tensor_codec_results.csv", codec_rows)
    csv_write(output / "family_codec_results.csv", family_codec_rows)
    (output / "heldout_results.json").write_text(
        json.dumps(heldout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema": "fp8-residency-analysis-v1",
        "sample_validation": validation,
        "collection_archive": collection.get("archive"),
        "collection_source_index_sha256": collection.get("source_index_sha256"),
        "analyzed_tensor_count": len(analyses),
        "analyzed_sample_weight_bytes": sum(item.sample_weight_bytes for item in analyses),
        "analyzed_sample_scale_bytes": sum(item.sample_scale_bytes for item in analyses),
        "tile_sizes": list(TILE_SIZES),
        "formats": list(FORMAT_NAMES),
        "heldout": heldout,
        "files": {
            "tensor_inventory": "tensor_inventory.csv",
            "family_entropy": "family_entropy.csv",
            "tensor_tile_stats": "tensor_tile_stats.csv",
            "tensor_codec_results": "tensor_codec_results.csv",
            "family_codec_results": "family_codec_results.csv",
            "heldout_results": "heldout_results.json",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["heldout"]["weighted_whole_cache_heldout"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, sample_io.SampleError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
