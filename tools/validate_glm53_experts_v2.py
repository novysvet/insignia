#!/usr/bin/env python3
"""Structurally validate an IG53XPK1-v2 expert sidecar without rewriting it.

The default pass reads every record header and every prefix directory, then
recounts packed-scale escapes for a small set of frontier records.  Pass
``--all-prefix`` to recount all packed scale nibbles (about 8.9 GiB of reads
for the full GLM-5.3-Flash sidecar).
"""

import argparse
import pathlib
import struct


ALIGNMENT = 4096
FILE_HEADER_BYTES = 4096
FILE_HEADER = struct.Struct("<8sIIIIQQQQQ")
INDEX_ENTRY = struct.Struct("<QII")
RECORD_HEADER_BYTES = 128
RECORD_HEADER = struct.Struct("<4sHH3I3f")
PROJECTIONS = 3
BODY_BYTES = 4 << 20
PACKED_SCALE_BYTES = 256 << 10
PREFIX_ENTRIES = PACKED_SCALE_BYTES // 256 + 1
PREFIX_BYTES = PREFIX_ENTRIES * 4
NIBBLE_ESCAPE_COUNTS = bytes(
    int((value & 15) == 15) + int((value >> 4) == 15)
    for value in range(256)
)


def align(value, granularity=ALIGNMENT):
    return (value + granularity - 1) & -granularity


def read_exact(handle, offset, size, what):
    handle.seek(offset)
    chunks = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            got = size - remaining
            raise SystemExit(
                f"{what}: short read at {offset}: got {got}, wanted {size}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def v2_stored_bytes(escapes):
    return ALIGNMENT + PROJECTIONS * BODY_BYTES + sum(
        align(align(PACKED_SCALE_BYTES + escape_count + 16, 4) + PREFIX_BYTES)
        for escape_count in escapes
    )


def parse_sample_ordinals(text, records):
    if text is None:
        # Includes both sides of the one-time safe-to-fast conversion frontier
        # used by the production GLM-5.3 sidecar, plus endpoints and midpoint.
        candidates = (0, 1, records // 2, 12073, 12074, 12075,
                      records - 2, records - 1)
    else:
        candidates = tuple(int(piece) for piece in text.split(",") if piece.strip())
    selected = sorted(set(candidate for candidate in candidates
                          if 0 <= candidate < records))
    if not selected:
        raise SystemExit("no prefix-sample ordinals are in range")
    return selected


def verify_prefix_counts(packed, prefix, label):
    translated = packed.translate(NIBBLE_ESCAPE_COUNTS)
    running = 0
    for block in range(PREFIX_ENTRIES - 1):
        running += sum(translated[block * 256:(block + 1) * 256])
        if prefix[block + 1] != running:
            raise SystemExit(
                f"{label}: prefix mismatch at block {block}: "
                f"stored {prefix[block + 1]}, counted {running}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar", type=pathlib.Path)
    parser.add_argument(
        "--all-prefix", action="store_true",
        help="recount every packed scale nibble instead of frontier samples")
    parser.add_argument(
        "--sample-ordinals",
        help="comma-separated occupied-record ordinals to recount")
    args = parser.parse_args()

    path = args.sidecar
    recovery_paths = (
        path.with_name(path.name + ".v2-inplace.json"),
        path.with_name(path.name + ".v2-inplace.json.tmp"),
        path.with_name(path.name + ".v2-inplace-record"),
        path.with_name(path.name + ".v2-inplace-record.tmp"),
    )
    present = [str(candidate) for candidate in recovery_paths if candidate.exists()]
    if present:
        raise SystemExit("unfinished in-place artifacts remain: " + ", ".join(present))

    actual_bytes = path.stat().st_size
    # Prefix directories are sparse within each 12.8 MiB record.  Buffered
    # read-ahead turns these 4 KiB probes into model-sized physical I/O.
    with path.open("rb", buffering=0) as handle:
        raw_header = read_exact(handle, 0, FILE_HEADER.size, "file header")
        magic, version, layers, experts, records, index_offset, data_offset, \
            file_bytes, source_bytes, stored_bytes = FILE_HEADER.unpack(raw_header)
        if magic != b"IG53XPK1" or version != 2:
            raise SystemExit(f"not an IG53XPK1-v2 sidecar: magic={magic!r}, v={version}")
        if index_offset != FILE_HEADER_BYTES:
            raise SystemExit(
                f"unexpected index offset {index_offset}, wanted {FILE_HEADER_BYTES}")
        if file_bytes != actual_bytes:
            raise SystemExit(
                f"header/stat size mismatch: {file_bytes} vs {actual_bytes}")
        slots = layers * experts
        expected_data_offset = align(index_offset + slots * INDEX_ENTRY.size)
        if data_offset != expected_data_offset:
            raise SystemExit(
                f"bad data offset {data_offset}, wanted {expected_data_offset}")
        if stored_bytes != file_bytes - data_offset:
            raise SystemExit(
                f"bad stored-byte total {stored_bytes}, wanted {file_bytes - data_offset}")
        if not source_bytes:
            raise SystemExit("source-byte total is zero")

        raw_index = read_exact(
            handle, index_offset, slots * INDEX_ENTRY.size, "expert index")
        entries = [INDEX_ENTRY.unpack_from(raw_index, offset)
                   for offset in range(0, len(raw_index), INDEX_ENTRY.size)]
        occupied = []
        cursor = data_offset
        for slot, entry in enumerate(entries):
            offset, stored, padded = entry
            if not offset:
                if entry != (0, 0, 0):
                    raise SystemExit(f"slot {slot}: malformed empty index entry {entry}")
                continue
            if offset != cursor:
                raise SystemExit(
                    f"slot {slot}: non-contiguous offset {offset}, wanted {cursor}")
            if offset & (ALIGNMENT - 1) or stored & (ALIGNMENT - 1):
                raise SystemExit(f"slot {slot}: unaligned span {entry}")
            if stored != padded:
                raise SystemExit(f"slot {slot}: stored/padded mismatch {entry}")
            if stored < ALIGNMENT + PROJECTIONS * BODY_BYTES:
                raise SystemExit(f"slot {slot}: record is too short: {stored}")
            occupied.append((slot, offset, stored))
            cursor += stored
        if len(occupied) != records:
            raise SystemExit(
                f"index has {len(occupied)} records, header declares {records}")
        if cursor != file_bytes:
            raise SystemExit(f"final index span ends at {cursor}, wanted {file_bytes}")

        samples = set(parse_sample_ordinals(args.sample_ordinals, records))
        prefix_recounts = 0
        for ordinal, (slot, offset, stored) in enumerate(occupied):
            page = read_exact(handle, offset, ALIGNMENT, f"record {ordinal} header")
            magic_r, layer, expert, e0, e1, e2, _g0, _g1, _g2 = \
                RECORD_HEADER.unpack_from(page, 0)
            expected_key = divmod(slot, experts)
            if magic_r != b"XPR1" or (layer, expert) != expected_key:
                raise SystemExit(
                    f"record {ordinal}: key {(magic_r, layer, expert)!r}, "
                    f"wanted {(b'XPR1', *expected_key)!r}")
            if any(page[RECORD_HEADER_BYTES:]):
                raise SystemExit(f"record {ordinal}: nonzero bytes in header-page padding")
            escapes = (e0, e1, e2)
            expected_span = v2_stored_bytes(escapes)
            if stored != expected_span:
                raise SystemExit(
                    f"record {ordinal}: index span {stored}, geometry says {expected_span}")

            region_at = offset + ALIGNMENT + PROJECTIONS * BODY_BYTES
            for projection, escape_count in enumerate(escapes):
                codebook_at = region_at + PACKED_SCALE_BYTES + escape_count
                codebook = read_exact(
                    handle, codebook_at, 16,
                    f"record {ordinal} projection {projection} codebook")
                header_codebook = page[32 + projection * 16:48 + projection * 16]
                if codebook != header_codebook:
                    raise SystemExit(
                        f"record {ordinal} projection {projection}: codebook mismatch")
                prefix_at = region_at + align(PACKED_SCALE_BYTES + escape_count + 16, 4)
                prefix_raw = read_exact(
                    handle, prefix_at, PREFIX_BYTES,
                    f"record {ordinal} projection {projection} prefix")
                prefix = struct.unpack(f"<{PREFIX_ENTRIES}I", prefix_raw)
                if prefix[0] != 0 or prefix[-1] != escape_count:
                    raise SystemExit(
                        f"record {ordinal} projection {projection}: "
                        f"bad prefix endpoints {prefix[0]}, {prefix[-1]}, "
                        f"wanted 0, {escape_count}")
                if any(left > right for left, right in zip(prefix, prefix[1:])):
                    raise SystemExit(
                        f"record {ordinal} projection {projection}: prefix is not monotone")
                if args.all_prefix or ordinal in samples:
                    packed = read_exact(
                        handle, region_at, PACKED_SCALE_BYTES,
                        f"record {ordinal} projection {projection} packed scales")
                    verify_prefix_counts(
                        packed, prefix, f"record {ordinal} projection {projection}")
                    prefix_recounts += 1
                region_at += align(
                    align(PACKED_SCALE_BYTES + escape_count + 16, 4) + PREFIX_BYTES)
            if region_at != offset + stored:
                raise SystemExit(
                    f"record {ordinal}: region walk ended at {region_at}, "
                    f"wanted {offset + stored}")
            if args.all_prefix and (ordinal + 1) % 1024 == 0:
                print(f"validated {ordinal + 1}/{records} records", flush=True)

    mode = "all" if args.all_prefix else ",".join(map(str, sorted(samples)))
    print(
        f"IG53XPK1-v2 PASS: {records} records, {file_bytes} bytes, "
        f"all headers/directories, prefix recounts={prefix_recounts} ({mode})")


if __name__ == "__main__":
    main()
