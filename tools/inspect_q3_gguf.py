#!/usr/bin/env python3
"""Emit the tensor directory of one or more GGUF shards as JSON lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gguf import GGUFReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--contains")
    args = parser.parse_args()

    for shard_id, path in enumerate(args.shards):
        reader = GGUFReader(path, "r")
        for tensor in reader.tensors:
            if args.contains and args.contains not in tensor.name:
                continue
            print(json.dumps({
                "shard": shard_id,
                "file": path.name,
                "name": tensor.name,
                "type": tensor.tensor_type.name,
                "type_id": int(tensor.tensor_type),
                "shape": [int(value) for value in tensor.shape],
                "elements": int(tensor.n_elements),
                "bytes": int(tensor.n_bytes),
                "offset": int(tensor.data_offset),
            }, separators=(",", ":")))


if __name__ == "__main__":
    main()
