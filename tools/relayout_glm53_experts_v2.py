#!/usr/bin/env python3
"""Relayout an IG53XPK1 expert sidecar (v1) into the v2 direct-read format.

v1 record:  [128 B header][body0 4 MiB][ps0 256 KiB][esc0][body1][ps1][esc1]
            [body2][ps2][esc2]  (esc = raw escape bytes, ps = packed scales)

v2 record:  [4096 B header page (v1 header + zero pad)]
            [body0][body1][body2]                      (window order, contiguous)
            per projection, 4096-aligned "region":
              [ps_p 256 KiB][esc_p][codebook_p 16 B][pad to 4]
              [escape-prefix table 1025 x u32][zero pad to 4096]

Every v2 region is byte-identical to what the v1 runtime assembles on the
CPU into its staging-window blob (packed bytes, escapes, codebook, alignment
pad, prefix table) plus trailing zero pad, so the reader threads do TWO
preambles per record (header page + one payload span) and zero CPU work:
no memcpy, no prefix build, no codebook placement. The window span mirrors
the file exactly, so both the merged (v2 transport) and per-projection (v1
transport) H2D copies work unchanged.

Only the record-internal layout changes; index geometry, codec, escapes, and
codebooks are carried over byte-for-byte. File header version becomes 2.
"""

import argparse
import pathlib
import struct
import time

import numpy as np

ALIGNMENT = 4096
FILE_HEADER_BYTES = 4096
INDEX_ENTRY = struct.Struct("<QII")
RECORD_HEADER_BYTES = 128
PROJECTIONS = 3
BODY_BYTES = 4 << 20
PACKED_SCALE_BYTES = 256 << 10
PREFIX_ENTRIES = PACKED_SCALE_BYTES // 256 + 1
FILE_HEADER = struct.Struct("<8sIIIIQQQQQ")
RECORD_HEADER = struct.Struct("<4sHH3I3f")


def align(value, granularity=ALIGNMENT):
    return (value + granularity - 1) & -granularity


def escape_prefix(packed, escape_count):
    """Replicates build_escape_prefix() from src/glm53_generate.cu."""
    codes = np.frombuffer(packed, dtype=np.uint8)
    # bool + bool is logical OR in numpy: cast BEFORE adding or 0xFF bytes undercount
    counts = ((codes & 15) == 15).astype(np.uint8) + ((codes >> 4) == 15)
    per_block = counts.reshape(-1, 256).sum(axis=1, dtype=np.uint32)
    prefix = np.zeros(PREFIX_ENTRIES, dtype=np.uint32)
    prefix[1:] = np.cumsum(per_block, dtype=np.uint32)
    if int(prefix[-1]) != escape_count:
        raise SystemExit(
            f"escape prefix mismatch: {int(prefix[-1])} counted vs {escape_count}")
    return prefix.tobytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--verify", action="store_true",
                        help="byte-compare every decoded region against v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"output exists: {args.output} (pass --force to replace)")

    source = args.input.open("rb", buffering=64 << 20)
    header = FILE_HEADER.unpack(source.read(64))
    magic, version, layers, experts, records, index_offset, data_offset, \
        file_bytes, source_bytes, _stored = header
    if magic != b"IG53XPK1" or version != 1:
        raise SystemExit(f"not a v1 IG53XPK1 sidecar: {args.input}")
    if args.input.stat().st_size != file_bytes:
        raise SystemExit("input size does not match header file_bytes")
    slots = layers * experts
    source.seek(index_offset)
    entries_in = [INDEX_ENTRY.unpack(source.read(INDEX_ENTRY.size))
                  for _ in range(slots)]

    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    begin = time.perf_counter()
    stored_total = 0
    done = 0
    entries_out = [(0, 0, 0)] * slots
    with temporary.open("w+b", buffering=64 << 20) as output:
        output.seek(data_offset)
        for slot, (offset_in, stored_in, padded_in) in enumerate(entries_in):
            if not offset_in:
                continue
            source.seek(offset_in)
            record = source.read(stored_in)
            magic_r, layer, expert, e0, e1, e2, g0, g1, g2 = \
                RECORD_HEADER.unpack_from(record, 0)
            if magic_r != b"XPR1":
                raise SystemExit(f"bad record magic at slot {slot}")
            escapes = (e0, e1, e2)
            want = RECORD_HEADER_BYTES + PROJECTIONS * (BODY_BYTES + PACKED_SCALE_BYTES) \
                + sum(escapes)
            if stored_in != want:
                raise SystemExit(f"v1 record {layer}/{expert} length mismatch")
            cursor = RECORD_HEADER_BYTES
            bodies = []
            regions = []
            for projection in range(PROJECTIONS):
                bodies.append(record[cursor:cursor + BODY_BYTES])
                cursor += BODY_BYTES
                packed = record[cursor:cursor + PACKED_SCALE_BYTES]
                cursor += PACKED_SCALE_BYTES
                tail = record[cursor:cursor + escapes[projection]]
                cursor += escapes[projection]
                codebook = record[32 + 16 * projection:48 + 16 * projection]
                region_no_pad = packed + tail + codebook
                region_no_pad += b"\0" * (align(len(region_no_pad), 4) - len(region_no_pad))
                region_no_pad += escape_prefix(packed, escapes[projection])
                regions.append(region_no_pad)
            if cursor != len(record):
                raise SystemExit(f"v1 record {layer}/{expert} trailing bytes")
            stored = ALIGNMENT + PROJECTIONS * BODY_BYTES + \
                sum(align(len(region)) for region in regions)
            offset_out = output.tell()
            if offset_out & (ALIGNMENT - 1):
                raise SystemExit("v2 record offset lost page alignment")
            block = bytearray(stored)
            block[:RECORD_HEADER_BYTES] = record[:RECORD_HEADER_BYTES]
            at = ALIGNMENT
            for body in bodies:
                block[at:at + BODY_BYTES] = body
                at += BODY_BYTES
            for region in regions:
                block[at:at + len(region)] = region
                at += align(len(region))
            if args.verify:
                # Reassemble the v1 order from the v2 block and byte-compare.
                rebuilt = bytearray(record[:RECORD_HEADER_BYTES])
                body_at = ALIGNMENT
                region_starts = []
                reg_at = ALIGNMENT + PROJECTIONS * BODY_BYTES
                for projection in range(PROJECTIONS):
                    region_starts.append(reg_at)
                    reg_at += align(len(regions[projection]))
                for projection in range(PROJECTIONS):
                    rebuilt += block[body_at:body_at + BODY_BYTES]
                    body_at += BODY_BYTES
                    start = region_starts[projection]
                    rebuilt += block[start:start + PACKED_SCALE_BYTES + escapes[projection]]
                if bytes(rebuilt) != record:
                    raise SystemExit(f"verify failed at record {layer}/{expert}")
            output.write(block)
            entries_out[slot] = (offset_out, stored, stored)
            stored_total += stored
            done += 1
            if done == 1 or done % 256 == 0:
                elapsed = time.perf_counter() - begin
                print(f"relayout {done}/{records}: "
                      f"{stored_total / 2**30:.2f} GiB written, "
                      f"{stored_total / elapsed / 2**30:.2f} GiB/s", flush=True)
        file_bytes_out = output.tell()
        header_out = FILE_HEADER.pack(
            b"IG53XPK1", 2, layers, experts, records,
            index_offset, data_offset, file_bytes_out, source_bytes, stored_total,
        )
        output.seek(0)
        output.write(header_out)
        output.write(b"\0" * (FILE_HEADER_BYTES - len(header_out)))
        for entry in entries_out:
            output.write(INDEX_ENTRY.pack(*entry))
        output.flush()
    temporary.replace(args.output)
    elapsed = time.perf_counter() - begin
    print(f"wrote {args.output}: {done} records, "
          f"{file_bytes_out / 2**30:.3f} GiB, in {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
