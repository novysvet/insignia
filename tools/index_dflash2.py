#!/usr/bin/env python3
"""Index the single-file DFlash2 drafter safetensors as an IGLMIDX1 index.

The drafter lives at /var/lib/insignia/glm53-dflash2.safetensors; the engine
opens the result with the stock ShardedIndex loader (geometry fields carry
drafter values, nothing validates them).
"""

import argparse
import json
import pathlib
import struct

MAGIC = b"IGLMIDX1"
DTYPES = {"F32": 1, "BF16": 2, "F16": 3, "U8": 4, "U32": 5, "I8": 6, "F8_E4M3": 7}
ITEM = {1: 4, 2: 2, 3: 2, 4: 1, 5: 4, 6: 1, 7: 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    with args.checkpoint.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        data_start = 8 + header_size
        shard_size = args.checkpoint.stat().st_size

    entries = []
    total = 0
    for name, meta in sorted(header.items()):
        if name == "__metadata__":
            continue
        dtype = DTYPES[meta["dtype"]]
        shape = meta["shape"]
        begin, end = meta["data_offsets"]
        length = end - begin
        assert length == shape[0] * shape[1] * ITEM.get(dtype, 0 if len(shape) != 2 else None) if len(shape) == 2 else True
        entries.append((name, dtype, shape, 0, data_start + begin, length))
        total += length

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        # Geometry block carries drafter-side values; the engine never reads
        # them for the drafter index.
        output.write(struct.pack(
            "<8s11IQ", MAGIC, 1, 0, 1, len(entries),
            4096, 5, 154880, 0, 0, 12288, 0, total,
        ))
        encoded = args.checkpoint.name.encode()
        output.write(struct.pack("<HQ", len(encoded), shard_size))
        output.write(encoded)
        for name, dtype, shape, shard, absolute, length in entries:
            encoded = name.encode()
            output.write(struct.pack("<HBBHHQQ", len(encoded), dtype, len(shape), shard, 0, absolute, length))
            output.write(encoded)
            output.write(struct.pack("<" + "I" * len(shape), *shape))
    print(f"indexed {len(entries)} drafter tensors; payload={total / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
