#!/usr/bin/env python3
"""Build and validate a compact sharded GLM-5.3 safetensors index."""

import argparse
import collections
import json
import math
import pathlib
import struct


MAGIC = b"IGLMIDX1"
DTYPES = {
    "F32": (1, 4),
    "BF16": (2, 2),
    "F16": (3, 2),
    "U8": (4, 1),
    "U32": (5, 4),
    "I8": (6, 1),
    "F8_E4M3": (7, 1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    root = args.checkpoint.resolve()
    config = json.loads((root / "config.json").read_text())["text_config"]
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    shard_ids = {name: index for index, name in enumerate(shard_names)}
    headers = {}
    shard_sizes = {}
    for name in shard_names:
        path = root / name
        shard_sizes[name] = path.stat().st_size
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(header_size))
        headers[name] = (header, 8 + header_size)

    entries = []
    census = collections.Counter()
    total_bytes = 0
    for tensor_name, shard_name in sorted(weight_map.items()):
        if args.text_only and tensor_name.startswith("model.visual."):
            continue
        header, data_start = headers[shard_name]
        if tensor_name not in header:
            raise ValueError(f"index points to missing tensor {tensor_name} in {shard_name}")
        meta = header[tensor_name]
        dtype_name = meta["dtype"]
        if dtype_name not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name} for {tensor_name}")
        dtype, item_size = DTYPES[dtype_name]
        begin, end = meta["data_offsets"]
        length = end - begin
        expected = math.prod(meta["shape"]) * item_size
        if length != expected:
            raise ValueError(f"size mismatch for {tensor_name}: {length} != {expected}")
        absolute = data_start + begin
        if absolute < data_start or absolute + length > shard_sizes[shard_name]:
            raise ValueError(f"out-of-range tensor {tensor_name}")
        entries.append((tensor_name, dtype, meta["shape"], shard_ids[shard_name], absolute, length))
        census[dtype_name] += length
        total_bytes += length

    flags = 1 if args.text_only else 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(struct.pack(
            "<8s11IQ", MAGIC, 1, flags, len(shard_names), len(entries),
            config["hidden_size"], config["num_hidden_layers"], config["vocab_size"],
            config["n_routed_experts"], config["num_experts_per_tok"],
            config["moe_intermediate_size"], config["hc_mult"], total_bytes,
        ))
        for name in shard_names:
            encoded = name.encode()
            output.write(struct.pack("<HQ", len(encoded), shard_sizes[name]))
            output.write(encoded)
        for name, dtype, shape, shard, absolute, length in entries:
            encoded = name.encode()
            output.write(struct.pack("<HBBHHQQ", len(encoded), dtype, len(shape), shard, 0, absolute, length))
            output.write(encoded)
            output.write(struct.pack("<" + "I" * len(shape), *shape))
    print(
        f"indexed {len(entries):,} tensors in {len(shard_names)} shards; "
        f"payload={total_bytes / 2**30:.3f} GiB index={args.output.stat().st_size / 2**20:.2f} MiB"
    )
    print(" ".join(f"{dtype}={size / 2**30:.3f}GiB" for dtype, size in sorted(census.items())))


if __name__ == "__main__":
    main()
