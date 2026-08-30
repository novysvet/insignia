#!/usr/bin/env python3
"""Deterministic exact fixed-tile codec for E4M3 matrix weight bytes.

The reference format is deliberately GPU-oriented:

* one independently addressable descriptor for every tensor-core weight tile;
* 16-byte-aligned payload blocks;
* a raw-tile escape when no compressed mode removes a full 16-byte sector;
* no global dictionary and no prefix dependency;
* exact reconstruction of all 256 byte values, including signed zero and NaNs.

A nominal tile contains ``tile_bytes / 64`` rows from one group-64 column
segment.  A 1024-byte tile is therefore the current Ada consumer's complete
16x64 shared-memory/MMA weight tile.  Smaller nominal tiles divide the same
consumer tile along the row dimension.

This module is the CPU reference and malformed-input validator.  It favors a
small, explicit format over encoder speed; the separate analyzer computes size
ceilings in vectorized batches.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import math
import pathlib
import struct
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Final

MAGIC: Final[bytes] = b"IF8XTC01"
VERSION: Final[int] = 1
HEADER_BYTES: Final[int] = 64
GROUP_SIZE: Final[int] = 64
BLOCK_ALIGNMENT: Final[int] = 16
ALLOWED_TILE_BYTES: Final[tuple[int, ...]] = (128, 256, 512, 1024)

# magic, version, header_bytes, tile_bytes, rows, cols, group_size, flags,
# raw_bytes, tile_count, directory_offset, data_offset
HEADER = struct.Struct("<8sHHIIIIIQQQQ")
assert HEADER.size == HEADER_BYTES

# payload offset in 16-byte units from data_offset, unpadded payload bytes,
# mode, mode parameter (palette count for palette modes).
DESCRIPTOR = struct.Struct("<IHBB")
assert DESCRIPTOR.size == 8

FLAG_CANONICAL_ZERO_PADDING: Final[int] = 1


class CodecError(RuntimeError):
    """Malformed container or mode payload."""


class Mode(enum.IntEnum):
    RAW = 0
    BYTE_PALETTE4 = 1
    BYTE_PALETTE5 = 2
    BYTE_PALETTE6 = 3
    MAG_PALETTE4 = 4
    MAG_PALETTE5 = 5
    MAG_PALETTE6 = 6
    EXP_PALETTE2 = 7
    EXP_PALETTE3 = 8
    MAG_XOR4 = 9
    ZERO_SPARSE = 10
    BITPLANE_CONST = 11


# Tie-breaking is a decode-cost preference.  RAW separately wins every tie in
# padded bytes, because equal traffic with additional decode cannot be useful.
MODE_PRIORITY: Final[tuple[Mode, ...]] = (
    Mode.BYTE_PALETTE4,
    Mode.MAG_PALETTE4,
    Mode.EXP_PALETTE2,
    Mode.BYTE_PALETTE5,
    Mode.MAG_PALETTE5,
    Mode.EXP_PALETTE3,
    Mode.MAG_XOR4,
    Mode.ZERO_SPARSE,
    Mode.BITPLANE_CONST,
    Mode.BYTE_PALETTE6,
    Mode.MAG_PALETTE6,
)
MODE_PRIORITY_RANK: Final[dict[Mode, int]] = {
    mode: rank for rank, mode in enumerate(MODE_PRIORITY)
}

PALETTE_BITS: Final[dict[Mode, int]] = {
    Mode.BYTE_PALETTE4: 4,
    Mode.BYTE_PALETTE5: 5,
    Mode.BYTE_PALETTE6: 6,
    Mode.MAG_PALETTE4: 4,
    Mode.MAG_PALETTE5: 5,
    Mode.MAG_PALETTE6: 6,
}
EXP_BITS: Final[dict[Mode, int]] = {
    Mode.EXP_PALETTE2: 2,
    Mode.EXP_PALETTE3: 3,
}


def align_up(value: int, alignment: int = BLOCK_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def _packed_bytes(count: int, bits: int) -> int:
    if count < 0 or bits <= 0 or bits > 8:
        raise ValueError("invalid bit-pack geometry")
    return (count * bits + 7) // 8


def pack_bits(values: Iterable[int], bits: int) -> bytes:
    """Pack unsigned values LSB-first into an exact-width byte stream."""
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be 1..8")
    maximum = 1 << bits
    output = bytearray()
    accumulator = 0
    available = 0
    for value in values:
        value = int(value)
        if value < 0 or value >= maximum:
            raise ValueError(f"value {value} does not fit in {bits} bits")
        accumulator |= value << available
        available += bits
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_bits(data: bytes | bytearray | memoryview, count: int, bits: int) -> list[int]:
    """Unpack and reject non-zero unused high bits or a wrong byte count."""
    expected = _packed_bytes(count, bits)
    view = memoryview(data).cast("B")
    if len(view) != expected:
        raise CodecError(f"packed stream has {len(view)} bytes, expected {expected}")
    if count == 0:
        if view:
            raise CodecError("non-empty zero-count packed stream")
        return []
    used_bits = count * bits
    remainder = used_bits & 7
    if remainder and view[-1] >> remainder:
        raise CodecError("packed stream has non-zero high padding bits")

    mask = (1 << bits) - 1
    values: list[int] = []
    accumulator = 0
    available = 0
    cursor = 0
    while len(values) < count:
        while available < bits:
            if cursor >= len(view):
                raise CodecError("truncated packed stream")
            accumulator |= int(view[cursor]) << available
            available += 8
            cursor += 1
        values.append(accumulator & mask)
        accumulator >>= bits
        available -= bits
    return values


def _bit_plane(values: Sequence[int], bit: int) -> bytes:
    return pack_bits(((value >> bit) & 1 for value in values), 1)


def _ordered_symbols(values: Sequence[int]) -> tuple[list[int], list[int]]:
    counts = Counter(values)
    ordered = sorted(counts, key=lambda value: (-counts[value], value))
    return ordered, [counts[value] for value in ordered]


def _choose_palette_count(
    counts: Sequence[int],
    maximum: int,
    fixed_bytes: int,
    exception_numerator: int = 1,
    exception_denominator: int = 1,
) -> int:
    """Choose the deterministic count minimizing padded then stored bytes.

    ``fixed_bytes`` already includes all mode fields except the palette and
    exceptions.  Exceptions cost ``ceil(count * numerator / denominator)``.
    """
    total = sum(counts)
    cumulative = 0
    best_key: tuple[int, int, int] | None = None
    best_count = 0
    for palette_count in range(0, min(maximum, len(counts)) + 1):
        if palette_count:
            cumulative += counts[palette_count - 1]
        exceptions = total - cumulative
        exception_bytes = (
            exceptions * exception_numerator + exception_denominator - 1
        ) // exception_denominator
        stored = fixed_bytes + palette_count + exception_bytes
        key = (align_up(stored), stored, palette_count)
        if best_key is None or key < best_key:
            best_key = key
            best_count = palette_count
    return best_count


@dataclasses.dataclass(frozen=True)
class TileEncoding:
    mode: Mode
    parameter: int
    payload: bytes

    @property
    def stored_bytes(self) -> int:
        return len(self.payload)

    @property
    def padded_bytes(self) -> int:
        return align_up(len(self.payload))


@dataclasses.dataclass(frozen=True)
class Descriptor:
    offset_units: int
    stored_bytes: int
    mode: Mode
    parameter: int
    raw_bytes: int

    @property
    def offset_bytes(self) -> int:
        return self.offset_units * BLOCK_ALIGNMENT

    @property
    def padded_bytes(self) -> int:
        return align_up(self.stored_bytes)


@dataclasses.dataclass(frozen=True)
class CodecStats:
    rows: int
    cols: int
    tile_bytes: int
    tile_count: int
    raw_bytes: int
    header_bytes: int
    directory_bytes: int
    directory_padding_bytes: int
    payload_bytes: int
    payload_padding_bytes: int
    encoded_bytes: int
    mode_counts: dict[str, int]

    @property
    def ratio(self) -> float:
        return self.encoded_bytes / self.raw_bytes

def _encode_byte_palette(raw: bytes, bits: int) -> tuple[int, bytes]:
    values = list(raw)
    ordered, counts = _ordered_symbols(values)
    sentinel = (1 << bits) - 1
    index_bytes = _packed_bytes(len(values), bits)
    palette_count = _choose_palette_count(counts, sentinel, index_bytes)
    palette = ordered[:palette_count]
    lookup = {value: index for index, value in enumerate(palette)}
    indices: list[int] = []
    exceptions = bytearray()
    for value in values:
        index = lookup.get(value)
        if index is None:
            indices.append(sentinel)
            exceptions.append(value)
        else:
            indices.append(index)
    return palette_count, bytes(palette) + pack_bits(indices, bits) + bytes(exceptions)


def _encode_mag_palette(raw: bytes, bits: int) -> tuple[int, bytes]:
    values = list(raw)
    magnitudes = [value & 0x7F for value in values]
    signs = _bit_plane(values, 7)
    ordered, counts = _ordered_symbols(magnitudes)
    sentinel = (1 << bits) - 1
    index_bytes = _packed_bytes(len(values), bits)
    fixed = len(signs) + index_bytes
    palette_count = _choose_palette_count(counts, sentinel, fixed)
    palette = ordered[:palette_count]
    lookup = {value: index for index, value in enumerate(palette)}
    indices: list[int] = []
    exceptions = bytearray()
    for magnitude in magnitudes:
        index = lookup.get(magnitude)
        if index is None:
            indices.append(sentinel)
            exceptions.append(magnitude)
        else:
            indices.append(index)
    return (
        palette_count,
        signs + bytes(palette) + pack_bits(indices, bits) + bytes(exceptions),
    )


def _encode_exp_palette(raw: bytes, bits: int) -> tuple[int, bytes]:
    values = list(raw)
    signs = _bit_plane(values, 7)
    mantissas = pack_bits((value & 7 for value in values), 3)
    exponents = [(value >> 3) & 0x0F for value in values]
    ordered, counts = _ordered_symbols(exponents)
    sentinel = (1 << bits) - 1
    index_bytes = _packed_bytes(len(values), bits)
    fixed = len(signs) + len(mantissas) + index_bytes
    palette_count = _choose_palette_count(
        counts, sentinel, fixed, exception_numerator=1, exception_denominator=2
    )
    palette = ordered[:palette_count]
    lookup = {value: index for index, value in enumerate(palette)}
    indices: list[int] = []
    exceptions: list[int] = []
    for exponent in exponents:
        index = lookup.get(exponent)
        if index is None:
            indices.append(sentinel)
            exceptions.append(exponent)
        else:
            indices.append(index)
    return (
        palette_count,
        signs
        + mantissas
        + bytes(palette)
        + pack_bits(indices, bits)
        + pack_bits(exceptions, 4),
    )


def _modal_magnitude(lane: Sequence[int]) -> int:
    counts = Counter(value & 0x7F for value in lane)
    return min(counts, key=lambda value: (-counts[value], value))


def _encode_mag_xor4(raw: bytes) -> bytes:
    values = list(raw)
    signs = _bit_plane(values, 7)
    magnitudes = [value & 0x7F for value in values]
    bases = bytearray()
    residuals: list[int] = []
    exceptions = bytearray()
    for begin in range(0, len(values), 32):
        lane = magnitudes[begin : begin + 32]
        base = _modal_magnitude(lane)
        bases.append(base)
        for magnitude in lane:
            residual = magnitude ^ base
            if residual < 15:
                residuals.append(residual)
            else:
                residuals.append(15)
                exceptions.append(magnitude)
    return signs + bytes(bases) + pack_bits(residuals, 4) + bytes(exceptions)


def _encode_zero_sparse(raw: bytes) -> bytes:
    values = list(raw)
    nonzero = [(value & 0x7F) != 0 for value in values]
    bitmap = pack_bits(nonzero, 1)
    zero_signs = pack_bits(
        ((value >> 7) & 1 for value, present in zip(values, nonzero) if not present),
        1,
    )
    literals = bytes(value for value, present in zip(values, nonzero) if present)
    return bitmap + zero_signs + literals


def _encode_bitplane_const(raw: bytes) -> bytes:
    values = list(raw)
    variable_mask = 0
    constant_mask = 0
    planes = bytearray()
    for bit in range(8):
        ones = sum((value >> bit) & 1 for value in values)
        if ones == 0:
            continue
        if ones == len(values):
            constant_mask |= 1 << bit
            continue
        variable_mask |= 1 << bit
        planes.extend(_bit_plane(values, bit))
    return bytes((variable_mask, constant_mask)) + bytes(planes)


def encode_mode(mode: Mode, raw: bytes | bytearray | memoryview) -> TileEncoding:
    data = bytes(raw)
    if not data or len(data) % GROUP_SIZE:
        raise ValueError("a tile must contain a positive whole number of group-64 rows")
    if mode == Mode.RAW:
        return TileEncoding(mode, 0, data)
    if mode in PALETTE_BITS:
        bits = PALETTE_BITS[mode]
        if mode in {Mode.BYTE_PALETTE4, Mode.BYTE_PALETTE5, Mode.BYTE_PALETTE6}:
            parameter, payload = _encode_byte_palette(data, bits)
        else:
            parameter, payload = _encode_mag_palette(data, bits)
        return TileEncoding(mode, parameter, payload)
    if mode in EXP_BITS:
        parameter, payload = _encode_exp_palette(data, EXP_BITS[mode])
        return TileEncoding(mode, parameter, payload)
    if mode == Mode.MAG_XOR4:
        return TileEncoding(mode, 0, _encode_mag_xor4(data))
    if mode == Mode.ZERO_SPARSE:
        return TileEncoding(mode, 0, _encode_zero_sparse(data))
    if mode == Mode.BITPLANE_CONST:
        return TileEncoding(mode, 0, _encode_bitplane_const(data))
    raise ValueError(f"unsupported mode {mode}")


def encode_tile(
    raw: bytes | bytearray | memoryview,
    enabled_modes: Iterable[Mode] | None = None,
) -> TileEncoding:
    data = bytes(raw)
    raw_encoding = encode_mode(Mode.RAW, data)
    raw_padded = raw_encoding.padded_bytes
    modes = tuple(enabled_modes) if enabled_modes is not None else MODE_PRIORITY
    candidates: list[TileEncoding] = []
    for mode in modes:
        mode = Mode(mode)
        if mode == Mode.RAW:
            continue
        candidate = encode_mode(mode, data)
        if candidate.padded_bytes < raw_padded:
            candidates.append(candidate)
    if not candidates:
        return raw_encoding
    return min(
        candidates,
        key=lambda item: (
            item.padded_bytes,
            MODE_PRIORITY_RANK.get(item.mode, 1 << 20),
            item.stored_bytes,
            item.parameter,
            item.payload,
        ),
    )


def _check_palette(palette: Sequence[int], maximum_value: int) -> None:
    if any(value < 0 or value > maximum_value for value in palette):
        raise CodecError("palette value out of range")
    if len(set(palette)) != len(palette):
        raise CodecError("palette contains duplicates")


def _decode_byte_palette(mode: Mode, parameter: int, payload: memoryview, count: int) -> bytes:
    bits = PALETTE_BITS[mode]
    sentinel = (1 << bits) - 1
    if parameter > sentinel:
        raise CodecError("palette count exceeds mode capacity")
    index_bytes = _packed_bytes(count, bits)
    minimum = parameter + index_bytes
    if len(payload) < minimum:
        raise CodecError("truncated byte-palette payload")
    palette = list(payload[:parameter])
    _check_palette(palette, 255)
    indices = unpack_bits(payload[parameter : parameter + index_bytes], count, bits)
    exceptions = payload[minimum:]
    expected_exceptions = sum(index == sentinel for index in indices)
    if len(exceptions) != expected_exceptions:
        raise CodecError("byte-palette exception count mismatch")
    output = bytearray(count)
    cursor = 0
    palette_set = set(palette)
    for position, index in enumerate(indices):
        if index == sentinel:
            value = int(exceptions[cursor])
            cursor += 1
            if value in palette_set:
                raise CodecError("non-canonical palette literal")
            output[position] = value
        elif index < parameter:
            output[position] = palette[index]
        else:
            raise CodecError("byte-palette index references an absent entry")
    return bytes(output)


def _decode_mag_palette(mode: Mode, parameter: int, payload: memoryview, count: int) -> bytes:
    bits = PALETTE_BITS[mode]
    sentinel = (1 << bits) - 1
    if parameter > sentinel:
        raise CodecError("magnitude palette count exceeds mode capacity")
    sign_bytes = _packed_bytes(count, 1)
    index_bytes = _packed_bytes(count, bits)
    minimum = sign_bytes + parameter + index_bytes
    if len(payload) < minimum:
        raise CodecError("truncated magnitude-palette payload")
    signs = unpack_bits(payload[:sign_bytes], count, 1)
    palette_begin = sign_bytes
    palette = list(payload[palette_begin : palette_begin + parameter])
    _check_palette(palette, 127)
    index_begin = palette_begin + parameter
    indices = unpack_bits(payload[index_begin : index_begin + index_bytes], count, bits)
    exceptions = payload[minimum:]
    expected_exceptions = sum(index == sentinel for index in indices)
    if len(exceptions) != expected_exceptions:
        raise CodecError("magnitude-palette exception count mismatch")
    output = bytearray(count)
    cursor = 0
    palette_set = set(palette)
    for position, index in enumerate(indices):
        if index == sentinel:
            magnitude = int(exceptions[cursor])
            cursor += 1
            if magnitude > 127:
                raise CodecError("magnitude literal has a sign bit")
            if magnitude in palette_set:
                raise CodecError("non-canonical magnitude literal")
        elif index < parameter:
            magnitude = palette[index]
        else:
            raise CodecError("magnitude-palette index references an absent entry")
        output[position] = magnitude | (signs[position] << 7)
    return bytes(output)


def _decode_exp_palette(mode: Mode, parameter: int, payload: memoryview, count: int) -> bytes:
    bits = EXP_BITS[mode]
    sentinel = (1 << bits) - 1
    if parameter > sentinel:
        raise CodecError("exponent palette count exceeds mode capacity")
    sign_bytes = _packed_bytes(count, 1)
    mantissa_bytes = _packed_bytes(count, 3)
    index_bytes = _packed_bytes(count, bits)
    minimum = sign_bytes + mantissa_bytes + parameter + index_bytes
    if len(payload) < minimum:
        raise CodecError("truncated exponent-palette payload")
    cursor = 0
    signs = unpack_bits(payload[cursor : cursor + sign_bytes], count, 1)
    cursor += sign_bytes
    mantissas = unpack_bits(payload[cursor : cursor + mantissa_bytes], count, 3)
    cursor += mantissa_bytes
    palette = list(payload[cursor : cursor + parameter])
    _check_palette(palette, 15)
    cursor += parameter
    indices = unpack_bits(payload[cursor : cursor + index_bytes], count, bits)
    cursor += index_bytes
    exception_count = sum(index == sentinel for index in indices)
    exception_bytes = _packed_bytes(exception_count, 4)
    if len(payload) != cursor + exception_bytes:
        raise CodecError("exponent-palette exception byte count mismatch")
    exceptions = unpack_bits(payload[cursor:], exception_count, 4)
    output = bytearray(count)
    exception_cursor = 0
    palette_set = set(palette)
    for position, index in enumerate(indices):
        if index == sentinel:
            exponent = exceptions[exception_cursor]
            exception_cursor += 1
            if exponent in palette_set:
                raise CodecError("non-canonical exponent literal")
        elif index < parameter:
            exponent = palette[index]
        else:
            raise CodecError("exponent-palette index references an absent entry")
        output[position] = (signs[position] << 7) | (exponent << 3) | mantissas[position]
    return bytes(output)


def _decode_mag_xor4(payload: memoryview, count: int) -> bytes:
    sign_bytes = _packed_bytes(count, 1)
    lanes = (count + 31) // 32
    residual_bytes = _packed_bytes(count, 4)
    minimum = sign_bytes + lanes + residual_bytes
    if len(payload) < minimum:
        raise CodecError("truncated magnitude-XOR payload")
    signs = unpack_bits(payload[:sign_bytes], count, 1)
    bases = list(payload[sign_bytes : sign_bytes + lanes])
    if any(base > 127 for base in bases):
        raise CodecError("magnitude-XOR base has a sign bit")
    residual_begin = sign_bytes + lanes
    residuals = unpack_bits(payload[residual_begin : residual_begin + residual_bytes], count, 4)
    exceptions = payload[minimum:]
    if len(exceptions) != sum(residual == 15 for residual in residuals):
        raise CodecError("magnitude-XOR exception count mismatch")
    output = bytearray(count)
    exception_cursor = 0
    for position, residual in enumerate(residuals):
        base = bases[position // 32]
        if residual == 15:
            magnitude = int(exceptions[exception_cursor])
            exception_cursor += 1
            if magnitude > 127:
                raise CodecError("magnitude-XOR literal has a sign bit")
            if (magnitude ^ base) < 15:
                raise CodecError("non-canonical magnitude-XOR literal")
        else:
            magnitude = base ^ residual
            if magnitude > 127:
                raise CodecError("magnitude-XOR residual leaves E4M3 magnitude range")
        output[position] = magnitude | (signs[position] << 7)
    return bytes(output)


def _decode_zero_sparse(payload: memoryview, count: int) -> bytes:
    bitmap_bytes = _packed_bytes(count, 1)
    if len(payload) < bitmap_bytes:
        raise CodecError("truncated zero-sparse bitmap")
    nonzero = unpack_bits(payload[:bitmap_bytes], count, 1)
    zero_count = count - sum(nonzero)
    zero_sign_bytes = _packed_bytes(zero_count, 1)
    minimum = bitmap_bytes + zero_sign_bytes
    nonzero_count = count - zero_count
    if len(payload) != minimum + nonzero_count:
        raise CodecError("zero-sparse literal count mismatch")
    zero_signs = unpack_bits(payload[bitmap_bytes:minimum], zero_count, 1)
    literals = payload[minimum:]
    output = bytearray(count)
    zero_cursor = 0
    literal_cursor = 0
    for position, present in enumerate(nonzero):
        if present:
            value = int(literals[literal_cursor])
            literal_cursor += 1
            if (value & 0x7F) == 0:
                raise CodecError("zero-sparse nonzero literal encodes a zero")
            output[position] = value
        else:
            output[position] = zero_signs[zero_cursor] << 7
            zero_cursor += 1
    return bytes(output)


def _decode_bitplane_const(payload: memoryview, count: int) -> bytes:
    if len(payload) < 2:
        raise CodecError("truncated bitplane header")
    variable_mask = int(payload[0])
    constant_mask = int(payload[1])
    if variable_mask & constant_mask:
        raise CodecError("variable and constant-one bitplane masks overlap")
    plane_bytes = _packed_bytes(count, 1)
    expected = 2 + variable_mask.bit_count() * plane_bytes
    if len(payload) != expected:
        raise CodecError("bitplane payload length mismatch")
    output = bytearray(count)
    cursor = 2
    for bit in range(8):
        if variable_mask & (1 << bit):
            values = unpack_bits(payload[cursor : cursor + plane_bytes], count, 1)
            cursor += plane_bytes
            if not any(values) or all(values):
                raise CodecError("non-canonical constant variable bitplane")
            for position, value in enumerate(values):
                output[position] |= value << bit
        elif constant_mask & (1 << bit):
            for position in range(count):
                output[position] |= 1 << bit
    return bytes(output)


def decode_mode(
    mode: Mode | int,
    parameter: int,
    payload: bytes | bytearray | memoryview,
    raw_bytes: int,
) -> bytes:
    try:
        mode = Mode(mode)
    except ValueError as error:
        raise CodecError(f"unknown tile mode {mode}") from error
    if raw_bytes <= 0 or raw_bytes % GROUP_SIZE:
        raise CodecError("invalid tile raw byte count")
    if parameter < 0 or parameter > 255:
        raise CodecError("tile parameter is out of range")
    view = memoryview(payload).cast("B")
    if mode == Mode.RAW:
        if parameter or len(view) != raw_bytes:
            raise CodecError("raw tile descriptor/payload mismatch")
        return bytes(view)
    if mode in {Mode.BYTE_PALETTE4, Mode.BYTE_PALETTE5, Mode.BYTE_PALETTE6}:
        return _decode_byte_palette(mode, parameter, view, raw_bytes)
    if mode in {Mode.MAG_PALETTE4, Mode.MAG_PALETTE5, Mode.MAG_PALETTE6}:
        return _decode_mag_palette(mode, parameter, view, raw_bytes)
    if mode in EXP_BITS:
        return _decode_exp_palette(mode, parameter, view, raw_bytes)
    if parameter:
        raise CodecError(f"mode {mode.name} requires a zero parameter")
    if mode == Mode.MAG_XOR4:
        return _decode_mag_xor4(view, raw_bytes)
    if mode == Mode.ZERO_SPARSE:
        return _decode_zero_sparse(view, raw_bytes)
    if mode == Mode.BITPLANE_CONST:
        return _decode_bitplane_const(view, raw_bytes)
    raise CodecError(f"unsupported tile mode {mode}")


def expected_tile_count(rows: int, cols: int, tile_bytes: int) -> int:
    if rows <= 0 or cols <= 0 or cols % GROUP_SIZE:
        raise ValueError("invalid matrix geometry")
    if tile_bytes not in ALLOWED_TILE_BYTES:
        raise ValueError(f"tile_bytes must be one of {ALLOWED_TILE_BYTES}")
    tile_rows = tile_bytes // GROUP_SIZE
    return ((rows + tile_rows - 1) // tile_rows) * (cols // GROUP_SIZE)


def tile_raw_bytes(rows: int, cols: int, tile_bytes: int, tile_index: int) -> int:
    groups = cols // GROUP_SIZE
    tile_rows = tile_bytes // GROUP_SIZE
    count = expected_tile_count(rows, cols, tile_bytes)
    if tile_index < 0 or tile_index >= count:
        raise IndexError("tile index out of range")
    row_block = tile_index // groups
    row_begin = row_block * tile_rows
    valid_rows = min(tile_rows, rows - row_begin)
    return valid_rows * GROUP_SIZE


def tile_coordinates(rows: int, cols: int, tile_bytes: int, tile_index: int) -> tuple[int, int, int]:
    groups = cols // GROUP_SIZE
    tile_rows = tile_bytes // GROUP_SIZE
    raw_count = tile_raw_bytes(rows, cols, tile_bytes, tile_index)
    return (tile_index // groups * tile_rows, tile_index % groups, raw_count // GROUP_SIZE)


def _gather_tile(
    raw: memoryview,
    rows: int,
    cols: int,
    tile_bytes: int,
    tile_index: int,
) -> bytes:
    row_begin, group, valid_rows = tile_coordinates(rows, cols, tile_bytes, tile_index)
    output = bytearray(valid_rows * GROUP_SIZE)
    for local_row in range(valid_rows):
        source_begin = (row_begin + local_row) * cols + group * GROUP_SIZE
        destination_begin = local_row * GROUP_SIZE
        output[destination_begin : destination_begin + GROUP_SIZE] = raw[
            source_begin : source_begin + GROUP_SIZE
        ]
    return bytes(output)


def encode_matrix(
    raw: bytes | bytearray | memoryview,
    rows: int,
    cols: int,
    tile_bytes: int = 1024,
    enabled_modes: Iterable[Mode] | None = None,
) -> bytes:
    view = memoryview(raw).cast("B")
    if rows > 0xFFFFFFFF or cols > 0xFFFFFFFF:
        raise ValueError("matrix dimensions exceed the version-1 header fields")
    if len(view) != rows * cols:
        raise ValueError(f"matrix has {len(view)} bytes, expected {rows * cols}")
    tile_count = expected_tile_count(rows, cols, tile_bytes)
    directory_offset = HEADER_BYTES
    directory_bytes = tile_count * DESCRIPTOR.size
    data_offset = align_up(directory_offset + directory_bytes)

    payload_offset = 0
    descriptors = bytearray()
    data = bytearray()
    for tile_index in range(tile_count):
        tile = _gather_tile(view, rows, cols, tile_bytes, tile_index)
        encoding = encode_tile(tile, enabled_modes)
        if payload_offset // BLOCK_ALIGNMENT > 0xFFFFFFFF:
            raise ValueError("compressed matrix exceeds descriptor offset range")
        if encoding.stored_bytes > 0xFFFF:
            raise ValueError("tile payload exceeds descriptor size field")
        descriptors.extend(
            DESCRIPTOR.pack(
                payload_offset // BLOCK_ALIGNMENT,
                encoding.stored_bytes,
                int(encoding.mode),
                encoding.parameter,
            )
        )
        data.extend(encoding.payload)
        padding = encoding.padded_bytes - encoding.stored_bytes
        if padding:
            data.extend(b"\0" * padding)
        payload_offset += encoding.padded_bytes
        if payload_offset > (0x100000000 * BLOCK_ALIGNMENT):
            raise ValueError("compressed matrix exceeds descriptor address range")

    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        tile_bytes,
        rows,
        cols,
        GROUP_SIZE,
        FLAG_CANONICAL_ZERO_PADDING,
        rows * cols,
        tile_count,
        directory_offset,
        data_offset,
    )
    output = bytearray(header)
    output.extend(descriptors)
    output.extend(b"\0" * (data_offset - len(output)))
    output.extend(data)
    return bytes(output)


class MatrixDecoder:
    """Validated random-access view of one compressed matrix."""

    def __init__(self, encoded: bytes | bytearray | memoryview, *, validate_payloads: bool = True):
        self._encoded = bytes(encoded)
        view = memoryview(self._encoded).cast("B")
        if len(view) < HEADER_BYTES:
            raise CodecError("truncated codec header")
        (
            magic,
            version,
            header_bytes,
            tile_bytes,
            rows,
            cols,
            group_size,
            flags,
            raw_bytes,
            tile_count,
            directory_offset,
            data_offset,
        ) = HEADER.unpack_from(view, 0)
        if magic != MAGIC:
            raise CodecError("bad codec magic")
        if version != VERSION or header_bytes != HEADER_BYTES:
            raise CodecError("unsupported codec version/header size")
        if tile_bytes not in ALLOWED_TILE_BYTES:
            raise CodecError("unsupported tile size")
        if rows <= 0 or cols <= 0 or cols % GROUP_SIZE or group_size != GROUP_SIZE:
            raise CodecError("invalid matrix/group geometry")
        if flags != FLAG_CANONICAL_ZERO_PADDING:
            raise CodecError("unsupported codec flags")
        if raw_bytes != rows * cols:
            raise CodecError("header raw byte count disagrees with geometry")
        expected_count = expected_tile_count(rows, cols, tile_bytes)
        if tile_count != expected_count:
            raise CodecError("header tile count disagrees with geometry")
        if directory_offset != HEADER_BYTES:
            raise CodecError("unexpected directory offset")
        expected_data_offset = align_up(directory_offset + tile_count * DESCRIPTOR.size)
        if data_offset != expected_data_offset or data_offset > len(view):
            raise CodecError("invalid data offset")
        directory_end = directory_offset + tile_count * DESCRIPTOR.size
        if any(view[directory_end:data_offset]):
            raise CodecError("non-zero directory alignment padding")

        self.rows = rows
        self.cols = cols
        self.tile_bytes = tile_bytes
        self.tile_count = tile_count
        self.raw_bytes = raw_bytes
        self.data_offset = data_offset
        self.descriptors: list[Descriptor] = []
        expected_payload_offset = 0
        for tile_index in range(tile_count):
            offset = directory_offset + tile_index * DESCRIPTOR.size
            offset_units, stored_bytes, mode_value, parameter = DESCRIPTOR.unpack_from(view, offset)
            try:
                mode = Mode(mode_value)
            except ValueError as error:
                raise CodecError(f"unknown mode {mode_value} in tile {tile_index}") from error
            raw_count = tile_raw_bytes(rows, cols, tile_bytes, tile_index)
            if stored_bytes <= 0:
                raise CodecError(f"zero stored byte count in tile {tile_index}")
            payload_offset = offset_units * BLOCK_ALIGNMENT
            if payload_offset != expected_payload_offset:
                raise CodecError(f"non-contiguous or overlapping payload at tile {tile_index}")
            padded = align_up(stored_bytes)
            absolute_begin = data_offset + payload_offset
            absolute_end = absolute_begin + padded
            if absolute_end > len(view):
                raise CodecError(f"tile {tile_index} exceeds encoded file")
            payload = view[absolute_begin : absolute_begin + stored_bytes]
            padding = view[absolute_begin + stored_bytes : absolute_end]
            if any(padding):
                raise CodecError(f"tile {tile_index} has non-zero payload padding")
            descriptor = Descriptor(offset_units, stored_bytes, mode, parameter, raw_count)
            self.descriptors.append(descriptor)
            if validate_payloads:
                decode_mode(mode, parameter, payload, raw_count)
            expected_payload_offset += padded
        if data_offset + expected_payload_offset != len(view):
            raise CodecError("trailing bytes after the final tile")

    def payload(self, tile_index: int) -> memoryview:
        if tile_index < 0 or tile_index >= self.tile_count:
            raise IndexError("tile index out of range")
        descriptor = self.descriptors[tile_index]
        begin = self.data_offset + descriptor.offset_bytes
        return memoryview(self._encoded)[begin : begin + descriptor.stored_bytes]

    def decode_tile(self, tile_index: int) -> bytes:
        descriptor = self.descriptors[tile_index]
        return decode_mode(
            descriptor.mode,
            descriptor.parameter,
            self.payload(tile_index),
            descriptor.raw_bytes,
        )

    def decode_matrix(self) -> bytes:
        output = bytearray(self.raw_bytes)
        groups = self.cols // GROUP_SIZE
        for tile_index in range(self.tile_count):
            row_begin, group, valid_rows = tile_coordinates(
                self.rows, self.cols, self.tile_bytes, tile_index
            )
            tile = self.decode_tile(tile_index)
            for local_row in range(valid_rows):
                source_begin = local_row * GROUP_SIZE
                destination_begin = (row_begin + local_row) * self.cols + group * GROUP_SIZE
                output[destination_begin : destination_begin + GROUP_SIZE] = tile[
                    source_begin : source_begin + GROUP_SIZE
                ]
        return bytes(output)

    def stats(self) -> CodecStats:
        directory_bytes = self.tile_count * DESCRIPTOR.size
        directory_padding = self.data_offset - HEADER_BYTES - directory_bytes
        payload_bytes = sum(descriptor.stored_bytes for descriptor in self.descriptors)
        payload_padded = sum(descriptor.padded_bytes for descriptor in self.descriptors)
        mode_counts = Counter(descriptor.mode.name for descriptor in self.descriptors)
        return CodecStats(
            rows=self.rows,
            cols=self.cols,
            tile_bytes=self.tile_bytes,
            tile_count=self.tile_count,
            raw_bytes=self.raw_bytes,
            header_bytes=HEADER_BYTES,
            directory_bytes=directory_bytes,
            directory_padding_bytes=directory_padding,
            payload_bytes=payload_bytes,
            payload_padding_bytes=payload_padded - payload_bytes,
            encoded_bytes=len(self._encoded),
            mode_counts=dict(sorted(mode_counts.items())),
        )


def decode_matrix(encoded: bytes | bytearray | memoryview) -> bytes:
    return MatrixDecoder(encoded).decode_matrix()


def encode_matrix_with_stats(
    raw: bytes | bytearray | memoryview,
    rows: int,
    cols: int,
    tile_bytes: int = 1024,
    enabled_modes: Iterable[Mode] | None = None,
) -> tuple[bytes, dict[str, object]]:
    encoded = encode_matrix(raw, rows, cols, tile_bytes, enabled_modes)
    decoder = MatrixDecoder(encoded)
    stats = dataclasses.asdict(decoder.stats())
    stats["ratio"] = len(encoded) / (rows * cols)
    stats["encoded_sha256"] = hashlib.sha256(encoded).hexdigest()
    stats["decoded_sha256"] = hashlib.sha256(decoder.decode_matrix()).hexdigest()
    return encoded, stats


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _emit_json(value: object, path: pathlib.Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(text)
    else:
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")


def _mode_name_map() -> dict[str, Mode]:
    return {mode.name.lower(): mode for mode in Mode if mode != Mode.RAW}


def _parse_enabled_modes(values: Sequence[str] | None) -> tuple[Mode, ...] | None:
    if not values:
        return None
    names = _mode_name_map()
    result: list[Mode] = []
    for text in values:
        key = text.strip().lower()
        if key not in names:
            choices = ", ".join(sorted(names))
            raise ValueError(f"unknown mode {text!r}; choose from {choices}")
        mode = names[key]
        if mode not in result:
            result.append(mode)
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact CPU reference codec for independently addressable FP8 MMA tiles"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    encode = commands.add_parser("encode", help="encode a raw row-major E4M3 matrix")
    encode.add_argument("input", type=pathlib.Path)
    encode.add_argument("output", type=pathlib.Path)
    encode.add_argument("--rows", type=int, required=True)
    encode.add_argument("--cols", type=int, required=True)
    encode.add_argument("--tile-bytes", type=int, choices=ALLOWED_TILE_BYTES, default=1024)
    encode.add_argument(
        "--mode",
        action="append",
        dest="modes",
        help="enable one exact mode by enum name; repeat to restrict the mode set",
    )
    encode.add_argument("--report-json", type=pathlib.Path)
    encode.add_argument(
        "--require-saving",
        action="store_true",
        help="fail instead of writing when the container is not smaller than raw weights",
    )

    decode = commands.add_parser("decode", help="validate and decode a reference container")
    decode.add_argument("input", type=pathlib.Path)
    decode.add_argument("output", type=pathlib.Path)
    decode.add_argument("--report-json", type=pathlib.Path)

    inspect = commands.add_parser("inspect", help="validate and report container geometry")
    inspect.add_argument("input", type=pathlib.Path)
    inspect.add_argument("--report-json", type=pathlib.Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "encode":
        raw = args.input.expanduser().resolve().read_bytes()
        modes = _parse_enabled_modes(args.modes)
        encoded, stats = encode_matrix_with_stats(
            raw, args.rows, args.cols, args.tile_bytes, modes
        )
        if args.require_saving and len(encoded) >= len(raw):
            raise ValueError(
                f"container ratio {len(encoded) / len(raw):.6f} is not smaller than raw"
            )
        _atomic_write(args.output, encoded)
        stats.update(
            {
                "input": str(args.input.expanduser().resolve()),
                "output": str(args.output.expanduser().resolve()),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        _emit_json(stats, args.report_json)
        return 0

    encoded = args.input.expanduser().resolve().read_bytes()
    decoder = MatrixDecoder(encoded)
    stats = dataclasses.asdict(decoder.stats())
    stats.update(
        {
            "ratio": decoder.stats().ratio,
            "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
            "input": str(args.input.expanduser().resolve()),
        }
    )
    if args.command == "decode":
        raw = decoder.decode_matrix()
        _atomic_write(args.output, raw)
        stats.update(
            {
                "output": str(args.output.expanduser().resolve()),
                "decoded_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    _emit_json(stats, args.report_json)
    return 0


__all__ = [
    "ALLOWED_TILE_BYTES",
    "BLOCK_ALIGNMENT",
    "CodecError",
    "CodecStats",
    "Descriptor",
    "GROUP_SIZE",
    "HEADER_BYTES",
    "MAGIC",
    "MODE_PRIORITY",
    "MatrixDecoder",
    "Mode",
    "TileEncoding",
    "align_up",
    "decode_matrix",
    "decode_mode",
    "encode_matrix",
    "encode_matrix_with_stats",
    "encode_mode",
    "encode_tile",
    "expected_tile_count",
    "pack_bits",
    "tile_coordinates",
    "tile_raw_bytes",
    "unpack_bits",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CodecError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
