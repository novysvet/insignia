#!/usr/bin/env python3
"""Dependency-free GGUF v3 header reader for the GLM-5.3 Q3 store.

The production converter needs only metadata, tensor locations, and exact
expert slices.  Keeping this reader independent from llama.cpp/gguf-py makes
the resulting index reproducible on the stripped glm-box toolchain.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import struct
from typing import Any, BinaryIO


VALUE_TYPES = {
    0: ("B", 1),   # UINT8
    1: ("b", 1),   # INT8
    2: ("H", 2),   # UINT16
    3: ("h", 2),   # INT16
    4: ("I", 4),   # UINT32
    5: ("i", 4),   # INT32
    6: ("f", 4),   # FLOAT32
    7: ("?", 1),   # BOOL
    10: ("Q", 8),  # UINT64
    11: ("q", 8),  # INT64
    12: ("d", 8),  # FLOAT64
}

# Stable ggml type ids.  Only formats present in the pinned artifact need a
# byte geometry, but names for nearby types make diagnostics unambiguous.
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


def _read_exact(file: BinaryIO, size: int) -> bytes:
    value = file.read(size)
    if len(value) != size:
        raise ValueError("truncated GGUF header")
    return value


def _scalar(file: BinaryIO, fmt: str) -> Any:
    return struct.unpack("<" + fmt, _read_exact(file, struct.calcsize(fmt)))[0]


def _string(file: BinaryIO) -> str:
    size = _scalar(file, "Q")
    if size > 1 << 30:
        raise ValueError(f"implausible GGUF string length {size}")
    return _read_exact(file, size).decode("utf-8")


def _value(file: BinaryIO, type_id: int) -> Any:
    if type_id in VALUE_TYPES:
        fmt, _ = VALUE_TYPES[type_id]
        return _scalar(file, fmt)
    if type_id == 8:
        return _string(file)
    if type_id == 9:
        element_type = _scalar(file, "I")
        count = _scalar(file, "Q")
        if count > 1 << 30:
            raise ValueError(f"implausible GGUF array length {count}")
        return [_value(file, element_type) for _ in range(count)]
    raise ValueError(f"unsupported GGUF metadata type {type_id}")


@dataclasses.dataclass(frozen=True)
class Tensor:
    name: str
    dimensions: tuple[int, ...]
    type_id: int
    type_name: str
    relative_offset: int
    data_offset: int
    elements: int
    bytes: int

    @property
    def experts(self) -> int:
        return self.dimensions[-1] if len(self.dimensions) >= 3 else 1

    def expert_span(self, expert: int) -> tuple[int, int]:
        if len(self.dimensions) < 3:
            raise ValueError(f"tensor is not expert-stacked: {self.name}")
        if not 0 <= expert < self.experts:
            raise ValueError(f"expert {expert} outside [0,{self.experts})")
        if self.bytes % self.experts:
            raise ValueError(f"expert tensor has nonintegral slice: {self.name}")
        stride = self.bytes // self.experts
        return self.data_offset + expert * stride, stride


class GGUFFile:
    def __init__(self, path: pathlib.Path):
        self.path = path.resolve()
        self.metadata: dict[str, Any] = {}
        self.tensors: list[Tensor] = []
        self.data_offset = 0
        self._parse()

    def _parse(self) -> None:
        file_bytes = self.path.stat().st_size
        with self.path.open("rb") as file:
            if _read_exact(file, 4) != b"GGUF":
                raise ValueError(f"bad GGUF magic: {self.path}")
            version = _scalar(file, "I")
            if version not in (2, 3):
                raise ValueError(f"unsupported GGUF version {version}")
            tensor_count = _scalar(file, "Q")
            metadata_count = _scalar(file, "Q")
            if tensor_count > 10_000_000 or metadata_count > 1_000_000:
                raise ValueError("implausible GGUF header counts")
            for _ in range(metadata_count):
                key = _string(file)
                type_id = _scalar(file, "I")
                if key in self.metadata:
                    raise ValueError(f"duplicate GGUF metadata key {key}")
                self.metadata[key] = _value(file, type_id)

            descriptors: list[tuple[str, tuple[int, ...], int, int]] = []
            for _ in range(tensor_count):
                name = _string(file)
                rank = _scalar(file, "I")
                if not 1 <= rank <= 4:
                    raise ValueError(f"unsupported rank {rank} for {name}")
                dimensions = tuple(_scalar(file, "Q") for _ in range(rank))
                type_id = _scalar(file, "I")
                relative_offset = _scalar(file, "Q")
                descriptors.append((name, dimensions, type_id, relative_offset))

            alignment = int(self.metadata.get("general.alignment", 32))
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError(f"invalid GGUF alignment {alignment}")
            self.data_offset = (file.tell() + alignment - 1) & -alignment

        names: set[str] = set()
        for name, dimensions, type_id, relative_offset in descriptors:
            if name in names:
                raise ValueError(f"duplicate GGUF tensor {name}")
            names.add(name)
            try:
                type_name, block_weights, block_bytes = GGML_TYPES[type_id]
            except KeyError as error:
                raise ValueError(f"unknown ggml type {type_id} for {name}") from error
            elements = math.prod(dimensions)
            if not dimensions[0] or dimensions[0] % block_weights:
                raise ValueError(
                    f"quant block does not divide ne[0] for {name}: "
                    f"{dimensions[0]} % {block_weights}"
                )
            tensor_bytes = elements // block_weights * block_bytes
            absolute = self.data_offset + relative_offset
            if absolute > file_bytes or tensor_bytes > file_bytes - absolute:
                raise ValueError(f"tensor exceeds shard: {name}")
            self.tensors.append(Tensor(
                name, dimensions, type_id, type_name, relative_offset,
                absolute, elements, tensor_bytes,
            ))

    def tensor(self, name: str) -> Tensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", type=pathlib.Path, nargs="+")
    parser.add_argument("--type", dest="type_name")
    parser.add_argument("--name", help="substring filter")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--expert", type=int)
    parser.add_argument("--extract", type=pathlib.Path)
    args = parser.parse_args()
    if (args.expert is None) != (args.extract is None):
        parser.error("--expert and --extract must be used together")
    selected: list[tuple[GGUFFile, Tensor]] = []
    for shard_path in args.shards:
        shard = GGUFFile(shard_path)
        for tensor in shard.tensors:
            if args.type_name and tensor.type_name != args.type_name:
                continue
            if args.name and args.name not in tensor.name:
                continue
            selected.append((shard, tensor))
    if args.extract is not None:
        if len(selected) != 1:
            raise SystemExit(f"extraction requires exactly one matching tensor, got {len(selected)}")
        shard, tensor = selected[0]
        offset, size = tensor.expert_span(args.expert)
        with shard.path.open("rb") as source, args.extract.open("wb") as output:
            source.seek(offset)
            remaining = size
            while remaining:
                data = source.read(min(8 << 20, remaining))
                if not data:
                    raise ValueError("short expert extraction")
                output.write(data)
                remaining -= len(data)
        print(f"extracted {tensor.name} expert {args.expert}: {size} bytes -> {args.extract}")
        return
    records = [{
        "shard": shard.path.name,
        **dataclasses.asdict(tensor),
    } for shard, tensor in selected]
    if args.json:
        print(json.dumps(records, indent=2))
    else:
        for record in records:
            dimensions = "x".join(str(value) for value in record["dimensions"])
            print(
                f"{record['type_name']}\t{dimensions}\t{record['bytes']}\t"
                f"{record['data_offset']}\t{record['shard']}\t{record['name']}"
            )


if __name__ == "__main__":
    main()

