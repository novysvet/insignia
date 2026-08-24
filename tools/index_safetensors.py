#!/usr/bin/env python3
import argparse
import json
import pathlib
import struct

MAGIC = b"INSIDX01"
DTYPES = {"F32": 1, "BF16": 2, "F16": 3, "U8": 4, "U32": 5, "I8": 6}

def main():
    ap = argparse.ArgumentParser(description="Build a zero-copy Insignia index for safetensors")
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    args = ap.parse_args()
    with args.model.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    tensors = sorted((name, spec) for name, spec in header.items() if name != "__metadata__")
    data_start = 8 + header_size
    source = str(args.model.resolve()).encode("utf-8")
    with args.output.open("wb") as out:
        out.write(struct.pack("<8sIIQ", MAGIC, 1, len(tensors), data_start))
        out.write(struct.pack("<I", len(source)))
        out.write(source)
        for name, spec in tensors:
            raw_name = name.encode("utf-8")
            shape = spec["shape"]
            begin, end = spec["data_offsets"]
            dtype = DTYPES.get(spec["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported dtype {spec['dtype']} for {name}")
            out.write(struct.pack("<HBBQQ", len(raw_name), dtype, len(shape), begin, end - begin))
            out.write(raw_name)
            out.write(struct.pack("<" + "Q" * len(shape), *shape))
    print(f"indexed {len(tensors)} tensors, payload starts at {data_start}, source={args.model}")

if __name__ == "__main__":
    main()
