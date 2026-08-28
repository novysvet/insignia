#!/usr/bin/env python3
"""Quantize the DFlash2 drafter's big BF16 matrices into the g64 FP8 cache.

Emits the same IGLMF8A1 format as quantize_glm53_q8.py --format fp8 (torch
e4m3 encoder, identical scaling) so the engine reads it with the stock
Q8Index. fc.weight [4096, 20480] exceeds the kernel's 256-group column cap,
so it is split into column halves fc.a/fc.b. Norms, conv base kernels and
the selector codebooks stay BF16 in the safetensors.
"""

import argparse
import json
import mmap
import os
import pathlib
import struct
import time

import numpy as np
import torch

GROUP = 64
ALIGNMENT = 4096


def align(value):
    return (value + ALIGNMENT - 1) & -ALIGNMENT


def quantize_matrix(mapping, offset, rows, cols, source_cols=None, source_col=0):
    groups = cols // GROUP
    source_cols = cols if source_cols is None else source_cols
    source = np.frombuffer(mapping, dtype="<u2", count=rows * source_cols,
                           offset=offset).reshape(rows, source_cols)
    source = np.ascontiguousarray(source[:, source_col:source_col + cols])
    bits = source.astype(np.uint32)
    bits <<= 16
    values = bits.view(np.float32).reshape(rows, groups, GROUP)
    maximum = np.max(np.abs(values), axis=2)
    scales = maximum * np.float32(1.0 / 448.0)
    divisors = np.where(scales > 0, scales, np.float32(1.0))
    normalized = np.ascontiguousarray(values / divisors[:, :, None])
    np.clip(normalized, -448.0, 448.0, out=normalized)
    quantized = torch.from_numpy(normalized).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    if np.any((quantized & 0x7F) == 0x7F):
        raise RuntimeError("FP8 encoder emitted NaN")
    return quantized, np.ascontiguousarray(scales.astype("<f2"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output_prefix", type=pathlib.Path)
    args = parser.parse_args()

    with args.checkpoint.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
        data_start = 8 + header_size

    mapping = mmap.mmap(os.open(args.checkpoint, os.O_RDONLY), 0, access=mmap.ACCESS_READ)

    plan = []  # (name, rows, cols, source_offset, source_cols, source_col)
    for layer in range(5):
        stem = f"layers.{layer}."
        for out, tensor in (
            ("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
            ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
            ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"),
            ("down", "mlp.down_proj.weight"),
            ("akp", "attention_conv.kernel_projection.weight"),
            ("mkp", "mlp_conv.kernel_projection.weight"),
        ):
            meta = header[stem + tensor]
            rows, cols = meta["shape"]
            plan.append((f"L{layer}.{out}", rows, cols,
                         data_start + meta["data_offsets"][0], cols, 0))
    meta = header["fc.weight"]
    fc_rows, fc_cols = meta["shape"]
    fc_offset = data_start + meta["data_offsets"][0]
    plan.append(("fc.a", fc_rows, fc_cols // 2, fc_offset, fc_cols, 0))
    plan.append(("fc.b", fc_rows, fc_cols // 2, fc_offset, fc_cols, fc_cols // 2))
    meta = header["candidate_selector.hidden_projection.weight"]
    hp_rows, hp_cols = meta["shape"]
    plan.append(("hp", hp_rows, hp_cols, data_start + meta["data_offsets"][0], hp_cols, 0))

    cursor = 0
    records = []
    for name, rows, cols, offset, source_cols, source_col in plan:
        records.append({
            "name": name, "rows": rows, "cols": cols, "source_offset": offset,
            "source_cols": source_cols, "source_col": source_col,
            "weight_offset": 0, "weight_bytes": rows * cols,
            "scale_offset": 0, "scale_bytes": rows * (cols // GROUP) * 2,
        })
        records[-1]["weight_offset"] = align(cursor)
        records[-1]["scale_offset"] = align(records[-1]["weight_offset"] + records[-1]["weight_bytes"])
        cursor = align(records[-1]["scale_offset"] + records[-1]["scale_bytes"])

    bin_path = args.output_prefix.parent / (args.output_prefix.name + ".bin")
    data_fd = os.open(bin_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(data_fd, cursor)
    started = time.time()
    for record in records:
        quantized, scales = quantize_matrix(
            mapping, record["source_offset"], record["rows"], record["cols"],
            record["source_cols"], record["source_col"])
        os.pwrite(data_fd, quantized.tobytes(), record["weight_offset"])
        os.pwrite(data_fd, scales.tobytes(), record["scale_offset"])
        print(f"  {record['name']}: {record['rows']}x{record['cols']}")
    os.close(data_fd)

    index_path = args.output_prefix.parent / (args.output_prefix.name + ".index")
    with index_path.open("wb") as output:
        output.write(struct.pack("<8sIIIQ", b"IGLMF8A1", 1, GROUP, len(records), cursor))
        for record in records:
            name = record["name"].encode()
            output.write(struct.pack(
                "<HIIQQQQ", len(name), record["rows"], record["cols"],
                record["weight_offset"], record["weight_bytes"],
                record["scale_offset"], record["scale_bytes"],
            ))
            output.write(name)
    print(f"quantized {len(records)} matrices, {cursor / 2**20:.1f} MiB in {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
