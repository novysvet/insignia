#!/usr/bin/env python3
"""Build an exact, O_DIRECT-friendly GLM NVFP4 expert sidecar.

The packed E2M1 weight nibbles are incompressible.  The E4M3 block scales are
not: per projection, the 15 most common bytes are encoded as nibbles and code
15 escapes to a raw byte stream.  Records stay independently page-aligned so
the runtime can read one compressed record and expand its scales with AVX2.
"""

import argparse
import json
import os
import pathlib
import struct
import time

import numpy as np

from glm53_expert_fixture import Checkpoint


ALIGNMENT = 4096
FILE_HEADER_BYTES = 4096
INDEX_ENTRY = struct.Struct("<QII")
RECORD_HEADER_BYTES = 128
PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
BODY_BYTES = 4 << 20
SCALE_BYTES = 512 << 10


def align(value, alignment=ALIGNMENT):
    return (value + alignment - 1) & -alignment


def encode_scales(raw):
    values = np.frombuffer(raw, np.uint8)
    if values.size != SCALE_BYTES or values.size & 1:
        raise ValueError(f"unexpected NVFP4 scale size {values.size}")
    counts = np.bincount(values, minlength=256)
    top = sorted(range(256), key=lambda code: (-int(counts[code]), code))[:15]
    encode = np.full(256, 15, np.uint8)
    encode[np.asarray(top, np.uint8)] = np.arange(15, dtype=np.uint8)
    codes = encode[values]
    packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8).tobytes()
    escapes = values[codes == 15].tobytes()
    codebook = bytes(top) + b"\0"
    return packed, escapes, codebook, codes


def verify_scales(raw, packed, escapes, codebook):
    encoded = np.frombuffer(packed, np.uint8)
    codes = np.empty(SCALE_BYTES, np.uint8)
    codes[0::2] = encoded & 15
    codes[1::2] = encoded >> 4
    decoded = np.frombuffer(codebook, np.uint8)[codes].copy()
    escaped = codes == 15
    replacement = np.frombuffer(escapes, np.uint8)
    if int(escaped.sum()) != replacement.size:
        raise RuntimeError("scale escape count mismatch")
    decoded[escaped] = replacement
    if decoded.tobytes() != raw:
        raise RuntimeError("scale codec is not byte exact")


def read_record(checkpoint, layer, expert, verify):
    stem = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    header = bytearray(RECORD_HEADER_BYTES)
    pieces = []
    escape_counts = []
    globals_ = []
    source_bytes = 0
    for projection in PROJECTIONS:
        body, body_meta = checkpoint.raw(stem + projection + ".weight")
        scales, scale_meta = checkpoint.raw(stem + projection + ".weight_scale")
        global_raw, global_meta = checkpoint.raw(stem + projection + ".weight_scale_2")
        if (len(body) != BODY_BYTES or len(scales) != SCALE_BYTES or len(global_raw) != 4):
            raise ValueError(f"unexpected expert geometry at layer {layer} expert {expert}")
        if (body_meta["dtype"] != "U8" or scale_meta["dtype"] != "F8_E4M3" or
                global_meta["dtype"] != "F32"):
            raise ValueError(f"unexpected expert types at layer {layer} expert {expert}")
        packed, escapes, codebook, _ = encode_scales(scales)
        if verify:
            verify_scales(scales, packed, escapes, codebook)
        pieces.append((body, packed, escapes, codebook))
        escape_counts.append(len(escapes))
        globals_.append(struct.unpack("<f", global_raw)[0])
        source_bytes += len(body) + len(scales) + len(global_raw)
    struct.pack_into(
        "<4sHH3I3f", header, 0, b"XPR1", layer, expert,
        *escape_counts, *globals_,
    )
    for projection, (_, _, _, codebook) in enumerate(pieces):
        header[32 + 16 * projection:48 + 16 * projection] = codebook
    record = bytearray(header)
    for body, packed, escapes, _ in pieces:
        record += body
        record += packed
        record += escapes
    return record, source_bytes, sum(escape_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="development-only maximum record count (0 packs all)")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.output.exists() and not args.force:
        parser.error(f"output exists: {args.output} (pass --force to replace)")

    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    layers = len(config["mlp_layer_types"])
    experts = int(config["n_routed_experts"])
    sparse = [
        layer for layer, kind in enumerate(config["mlp_layer_types"])
        if kind == "sparse"
    ]
    record_slots = layers * experts
    index_offset = FILE_HEADER_BYTES
    data_offset = align(index_offset + record_slots * INDEX_ENTRY.size)
    entries = [(0, 0, 0)] * record_slots
    checkpoint = Checkpoint(args.checkpoint)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    begin = time.perf_counter()
    records = source_total = stored_total = escapes_total = 0
    with temporary.open("w+b", buffering=16 << 20) as output:
        output.seek(data_offset)
        stop = False
        for layer in sparse:
            for expert in range(experts):
                if args.limit and records >= args.limit:
                    stop = True
                    break
                record, source_bytes, escapes = read_record(
                    checkpoint, layer, expert, not args.no_verify
                )
                offset = output.tell()
                if offset & (ALIGNMENT - 1):
                    raise RuntimeError("record offset lost page alignment")
                output.write(record)
                padded_bytes = align(len(record))
                output.write(b"\0" * (padded_bytes - len(record)))
                entries[layer * experts + expert] = (offset, len(record), padded_bytes)
                records += 1
                source_total += source_bytes
                stored_total += len(record)
                escapes_total += escapes
                if records == 1 or records % 64 == 0:
                    elapsed = time.perf_counter() - begin
                    print(
                        f"packed {records}/{len(sparse) * experts} records: "
                        f"{stored_total / source_total:.4f}x, "
                        f"{source_total / elapsed / 2**30:.2f} GiB/s source",
                        flush=True,
                    )
            if stop:
                break
        file_bytes = output.tell()
        header = struct.pack(
            "<8sIIIIQQQQQ", b"IG53XPK1", 1, layers, experts, records,
            index_offset, data_offset, file_bytes, source_total, stored_total,
        )
        output.seek(0)
        output.write(header)
        output.write(b"\0" * (FILE_HEADER_BYTES - len(header)))
        for entry in entries:
            output.write(INDEX_ENTRY.pack(*entry))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(args.output)
    elapsed = time.perf_counter() - begin
    print(
        f"wrote {args.output}: {records} records, {file_bytes / 2**30:.3f} GiB, "
        f"logical ratio {stored_total / source_total:.4f}x, "
        f"scale escapes {100 * escapes_total / max(1, records * 3 * SCALE_BYTES):.3f}%, "
        f"{elapsed:.1f} s",
    )


if __name__ == "__main__":
    main()
