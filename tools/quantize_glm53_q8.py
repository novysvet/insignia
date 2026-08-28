#!/usr/bin/env python3
"""Build a persistent group-64 Q8 or FP8 cache for GLM-5.3 BF16 matrices."""

import argparse
import json
import math
import mmap
import os
import pathlib
import shutil
import struct
import time

import numpy as np

from quantize_dflash2 import encode_e4m3fn


GROUP = 64
ALIGNMENT = 4096


def align(value):
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def load_metadata(root):
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    for shard_name in sorted(set(weight_map.values())):
        path = root / shard_name
        with path.open("rb") as source:
            header_size = struct.unpack("<Q", source.read(8))[0]
            headers[shard_name] = (json.loads(source.read(header_size)), 8 + header_size)
    return weight_map, headers


def select_records(root, weight_map, headers):
    records = []
    cursor = 0
    for name, shard_name in sorted(weight_map.items()):
        if not (name == "lm_head.weight" or name.startswith("model.language_model.")):
            continue
        meta, data_start = headers[shard_name]
        tensor = meta[name]
        shape = tensor["shape"]
        if tensor["dtype"] != "BF16" or len(shape) != 2 or shape[1] % GROUP:
            continue
        if name.endswith("embed_tokens.weight") or name.endswith(".fn.weight"):
            continue
        rows, cols = shape
        begin, end = tensor["data_offsets"]
        if end - begin != rows * cols * 2:
            raise ValueError(f"wrong BF16 byte count for {name}")
        weight_offset = align(cursor)
        weight_bytes = rows * cols
        scale_offset = align(weight_offset + weight_bytes)
        scale_bytes = rows * (cols // GROUP) * 2
        cursor = align(scale_offset + scale_bytes)
        records.append({
            "name": name,
            "shard": shard_name,
            "source_offset": data_start + begin,
            "rows": rows,
            "cols": cols,
            "weight_offset": weight_offset,
            "weight_bytes": weight_bytes,
            "scale_offset": scale_offset,
            "scale_bytes": scale_bytes,
        })
    return records, cursor


def write_index(path, records, data_bytes, cache_format):
    magic = b"IGLMQ8A1" if cache_format == "q8" else b"IGLMF8A1"
    with path.open("wb") as output:
        output.write(struct.pack("<8sIIIQ", magic, 1, GROUP, len(records), data_bytes))
        for record in records:
            name = record["name"].encode()
            output.write(struct.pack(
                "<HIIQQQQ", len(name), record["rows"], record["cols"],
                record["weight_offset"], record["weight_bytes"],
                record["scale_offset"], record["scale_bytes"],
            ))
            output.write(name)


def quantize(root, data_path, records, data_bytes, cache_format):
    free = shutil.disk_usage(data_path.parent).free
    if free < data_bytes + (2 << 30):
        raise RuntimeError(
            f"need {data_bytes / 2**30:.2f} GiB plus 2 GiB headroom; "
            f"only {free / 2**30:.2f} GiB is free"
        )
    descriptors = {}
    mappings = {}
    data_fd = os.open(data_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(data_fd, data_bytes)
    processed = 0
    started = time.monotonic()
    try:
        for ordinal, record in enumerate(records, 1):
            shard_name = record["shard"]
            if shard_name not in mappings:
                descriptor = os.open(root / shard_name, os.O_RDONLY)
                descriptors[shard_name] = descriptor
                mappings[shard_name] = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
            rows, cols = record["rows"], record["cols"]
            groups = cols // GROUP
            source = np.frombuffer(
                mappings[shard_name], dtype="<u2", count=rows * cols,
                offset=record["source_offset"],
            ).reshape(rows, cols)
            rows_per_chunk = max(1, (8 << 20) // cols)
            for row in range(0, rows, rows_per_chunk):
                count = min(rows_per_chunk, rows - row)
                bits = source[row : row + count].astype(np.uint32)
                bits <<= 16
                values = bits.view(np.float32).reshape(count, groups, GROUP)
                maximum = np.max(np.abs(values), axis=2)
                divisor = 127.0 if cache_format == "q8" else 448.0
                scales = maximum * np.float32(1.0 / divisor)
                divisors = np.where(scales > 0, scales, np.float32(1.0))
                normalized = np.ascontiguousarray(values / divisors[:, :, None])
                if cache_format == "q8":
                    quantized = np.rint(normalized)
                    np.clip(quantized, -127, 127, out=quantized)
                    quantized = np.ascontiguousarray(quantized.astype(np.int8))
                else:
                    np.clip(normalized, -448.0, 448.0, out=normalized)
                    quantized = encode_e4m3fn(normalized)
                    if np.any((quantized & 0x7F) == 0x7F):
                        raise RuntimeError(f"FP8 encoder emitted NaN for {record['name']}")
                scale16 = np.ascontiguousarray(scales.astype("<f2"))
                os.pwrite(data_fd, quantized.tobytes(), record["weight_offset"] + row * cols)
                os.pwrite(data_fd, scale16.tobytes(),
                          record["scale_offset"] + row * groups * 2)
            del source
            processed += record["weight_bytes"] + record["scale_bytes"]
            if ordinal == len(records) or processed // (1 << 30) != (
                processed - record["weight_bytes"] - record["scale_bytes"]
            ) // (1 << 30):
                elapsed = time.monotonic() - started
                print(
                    f"[{ordinal:4d}/{len(records)}] {processed / 2**30:6.2f} GiB "
                    f"{processed / elapsed / 1e9:5.2f} GB/s {record['name']}",
                    flush=True,
                )
        os.fsync(data_fd)
    finally:
        os.close(data_fd)
        for mapping in mappings.values():
            mapping.close()
        for descriptor in descriptors.values():
            os.close(descriptor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output_prefix", type=pathlib.Path)
    parser.add_argument("--format", choices=("q8", "fp8"), default="q8")
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    weight_map, headers = load_metadata(root)
    records, data_bytes = select_records(root, weight_map, headers)
    if not records:
        raise RuntimeError("no eligible GLM-5.3 BF16 matrices found")
    print(
        f"quantizing {len(records)} matrices into {data_bytes / 2**30:.3f} GiB "
        f"{args.format.upper()}-g{GROUP} cache",
        flush=True,
    )
    temporary_data = pathlib.Path(str(prefix) + ".bin.tmp")
    temporary_index = pathlib.Path(str(prefix) + ".index.tmp")
    quantize(root, temporary_data, records, data_bytes, args.format)
    write_index(temporary_index, records, data_bytes, args.format)
    temporary_data.replace(pathlib.Path(str(prefix) + ".bin"))
    temporary_index.replace(pathlib.Path(str(prefix) + ".index"))
    print(
        f"wrote {prefix}.bin ({data_bytes / 2**30:.3f} GiB) and "
        f"{prefix}.index ({pathlib.Path(str(prefix) + '.index').stat().st_size / 2**10:.1f} KiB)"
    )


if __name__ == "__main__":
    main()
