#!/usr/bin/env python3
"""Relayout an IG53XPK1 expert sidecar into the v2 direct-read format.

Normal mode writes a second file atomically. ``--in-place`` computes every
v2 offset first and then walks records from the end toward the beginning, so
the slightly larger destinations can only reuse source ranges already read.
The process-crash-recoverable form journals and backs up one source record at
a time. ``--fast-in-place`` skips those writes for a disposable v1 sidecar.
"""

import argparse
import json
import os
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


def write_exact(handle, data, what):
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = handle.write(view[written:])
        if count is None or count <= 0:
            raise OSError(f"{what}: short write after {written}/{len(view)} bytes")
        written += count


def escape_prefix(packed, escape_count):
    """Replicate build_escape_prefix() from src/glm53_generate.cu."""
    codes = np.frombuffer(packed, dtype=np.uint8)
    # Cast before adding: numpy bool + bool otherwise behaves like logical OR.
    counts = ((codes & 15) == 15).astype(np.uint8) + ((codes >> 4) == 15)
    per_block = counts.reshape(-1, 256).sum(axis=1, dtype=np.uint32)
    prefix = np.zeros(PREFIX_ENTRIES, dtype=np.uint32)
    prefix[1:] = np.cumsum(per_block, dtype=np.uint32)
    if int(prefix[-1]) != escape_count:
        raise SystemExit(
            f"escape prefix mismatch: {int(prefix[-1])} counted vs {escape_count}")
    return prefix.tobytes()


def v2_stored_bytes(escapes):
    return ALIGNMENT + PROJECTIONS * BODY_BYTES + sum(
        align(align(PACKED_SCALE_BYTES + escape_count + 16, 4) +
                    PREFIX_ENTRIES * 4)
        for escape_count in escapes
    )


def inspect_v1_header(header, stored_in, slot, experts):
    if len(header) != RECORD_HEADER_BYTES:
        raise SystemExit(f"truncated v1 record header at slot {slot}")
    magic_r, layer, expert, e0, e1, e2, _g0, _g1, _g2 = \
        RECORD_HEADER.unpack_from(header, 0)
    if magic_r != b"XPR1":
        raise SystemExit(f"bad record magic at slot {slot}")
    expected_layer, expected_expert = divmod(slot, experts)
    if (layer, expert) != (expected_layer, expected_expert):
        raise SystemExit(
            f"record key {layer}/{expert} disagrees with slot {slot} "
            f"({expected_layer}/{expected_expert})")
    escapes = (e0, e1, e2)
    want = RECORD_HEADER_BYTES + PROJECTIONS * (BODY_BYTES + PACKED_SCALE_BYTES) \
        + sum(escapes)
    if stored_in != want:
        raise SystemExit(f"v1 record {layer}/{expert} length mismatch")
    return layer, expert, escapes


def relayout_record(record, stored_in, slot, experts, verify):
    layer, expert, escapes = inspect_v1_header(
        record[:RECORD_HEADER_BYTES], stored_in, slot, experts)
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
        region = packed + tail + codebook
        region += b"\0" * (align(len(region), 4) - len(region))
        region += escape_prefix(packed, escapes[projection])
        regions.append(region)
    if cursor != len(record):
        raise SystemExit(f"v1 record {layer}/{expert} trailing bytes")

    stored_out = v2_stored_bytes(escapes)
    block = bytearray(stored_out)
    block[:RECORD_HEADER_BYTES] = record[:RECORD_HEADER_BYTES]
    at = ALIGNMENT
    for body in bodies:
        block[at:at + BODY_BYTES] = body
        at += BODY_BYTES
    for region in regions:
        block[at:at + len(region)] = region
        at += align(len(region))
    if at != stored_out:
        raise SystemExit(f"v2 record {layer}/{expert} layout mismatch")

    if verify:
        rebuilt = bytearray(record[:RECORD_HEADER_BYTES])
        body_at = ALIGNMENT
        region_at = ALIGNMENT + PROJECTIONS * BODY_BYTES
        region_starts = []
        for region in regions:
            region_starts.append(region_at)
            region_at += align(len(region))
        for projection in range(PROJECTIONS):
            rebuilt += block[body_at:body_at + BODY_BYTES]
            body_at += BODY_BYTES
            start = region_starts[projection]
            rebuilt += block[
                start:start + PACKED_SCALE_BYTES + escapes[projection]]
        if bytes(rebuilt) != record:
            raise SystemExit(f"verify failed at record {layer}/{expert}")
    return block


def write_state(path, *, file_bytes, target_bytes, records, next_ordinal,
                fast_started=False):
    state = {
        "magic": "IG53XPK1-v2-inplace",
        "source_bytes": file_bytes,
        "target_bytes": target_bytes,
        "records": records,
        "next_ordinal": next_ordinal,
        "fast_started": fast_started,
    }
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(state, sort_keys=True).encode("utf-8")
    with temporary.open("wb", buffering=0) as handle:
        write_exact(handle, payload, "write in-place journal")
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path, nargs="?")
    parser.add_argument("--verify", action="store_true",
                        help="byte-compare every decoded region against v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--in-place", action="store_true",
                        help="reverse-relayout INPUT without a second full sidecar")
    parser.add_argument("--fast-in-place", action="store_true",
                        help="skip recovery writes; interruption destroys the v1 artifact")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate geometry/journal without writing data")
    args = parser.parse_args()

    if args.fast_in_place and not args.in_place:
        parser.error("--fast-in-place requires --in-place")
    if args.in_place:
        if args.output is not None and args.output.resolve() != args.input.resolve():
            parser.error("--in-place does not accept a distinct output path")
        if args.force:
            parser.error("--force is meaningless with --in-place")
        output_path = args.input
    else:
        if args.output is None:
            parser.error("OUTPUT is required unless --in-place is used")
        output_path = args.output
        if output_path.exists() and not args.force:
            parser.error(f"output exists: {output_path} (pass --force to replace)")

    source_mode = "r+b" if args.in_place else "rb"
    source_buffering = 0 if args.in_place else 64 << 20
    with args.input.open(source_mode, buffering=source_buffering) as source:
        raw_header = source.read(64)
        if len(raw_header) != 64:
            raise SystemExit("truncated sidecar header")
        magic, version, layers, experts, records, index_offset, data_offset, \
            file_bytes, source_bytes, _stored = FILE_HEADER.unpack(raw_header)
        if magic != b"IG53XPK1" or version != 1:
            raise SystemExit(f"not a v1 IG53XPK1 sidecar: {args.input}")
        slots = layers * experts
        source.seek(index_offset)
        entries_in = [INDEX_ENTRY.unpack(source.read(INDEX_ENTRY.size))
                      for _ in range(slots)]
        occupied = [slot for slot, (offset, _stored, _padded)
                    in enumerate(entries_in) if offset]
        if len(occupied) != records:
            raise SystemExit(
                f"index has {len(occupied)} records but header declares {records}")
        previous_end = data_offset
        for slot in occupied:
            offset, stored, padded = entries_in[slot]
            if (offset & (ALIGNMENT - 1) or padded & (ALIGNMENT - 1) or
                    stored > padded or offset < previous_end or
                    offset > file_bytes or padded > file_bytes - offset):
                raise SystemExit(
                    f"source index is not ordered/non-overlapping at slot {slot}")
            previous_end = offset + padded

        journal = args.input.with_name(args.input.name + ".v2-inplace.json")
        backup = args.input.with_name(args.input.name + ".v2-inplace-record")
        backup_tmp = backup.with_name(backup.name + ".tmp")
        resume_state = None
        resume_next = len(occupied) - 1
        if args.in_place and journal.exists():
            resume_state = json.loads(journal.read_text(encoding="utf-8"))
            if (resume_state.get("magic") != "IG53XPK1-v2-inplace" or
                    resume_state.get("source_bytes") != file_bytes or
                    resume_state.get("records") != records):
                raise SystemExit(f"stale or foreign in-place journal: {journal}")
            if resume_state.get("fast_started"):
                raise SystemExit(
                    "an unrecoverable fast in-place pass was interrupted; regenerate v1")
            resume_next = int(resume_state["next_ordinal"])
            if not -1 <= resume_next < len(occupied):
                raise SystemExit("in-place journal ordinal is out of range")

        # Completed tail records may have lost their old source ranges, so a
        # resume reads their XPR1 headers from the v2 destinations computed so far.
        entries_out = [(0, 0, 0)] * slots
        offset_out = data_offset
        for ordinal, slot in enumerate(occupied):
            offset_in, stored_in, _padded_in = entries_in[slot]
            header_offset = (offset_out if resume_state is not None and
                             ordinal > resume_next else offset_in)
            source.seek(header_offset)
            record_header = source.read(RECORD_HEADER_BYTES)
            _layer, _expert, escapes = inspect_v1_header(
                record_header, stored_in, slot, experts)
            stored_out = v2_stored_bytes(escapes)
            if offset_out & (ALIGNMENT - 1):
                raise SystemExit("v2 record offset lost page alignment")
            if args.in_place and offset_out < offset_in:
                raise SystemExit(
                    f"in-place safety proof failed at slot {slot}: "
                    f"output {offset_out} precedes input {offset_in}")
            entries_out[slot] = (offset_out, stored_out, stored_out)
            offset_out += stored_out

        file_bytes_out = offset_out
        stored_total = file_bytes_out - data_offset
        current_size = args.input.stat().st_size
        if resume_state is not None and \
                resume_state.get("target_bytes") != file_bytes_out:
            raise SystemExit(f"target geometry disagrees with journal: {journal}")
        header_out = FILE_HEADER.pack(
            b"IG53XPK1", 2, layers, experts, records,
            index_offset, data_offset, file_bytes_out, source_bytes, stored_total)
        begin = time.perf_counter()

        if args.preflight_only:
            print(
                f"preflight PASS: {records} records, source={file_bytes} bytes, "
                f"target={file_bytes_out} bytes, next_ordinal={resume_next}")
            return

        if args.in_place:
            if resume_state is None:
                if current_size != file_bytes:
                    raise SystemExit("input size does not match header file_bytes")
                next_ordinal = len(occupied) - 1
                source.truncate(file_bytes_out)
                source.flush()
                write_state(
                    journal, file_bytes=file_bytes, target_bytes=file_bytes_out,
                    records=records, next_ordinal=next_ordinal)
            else:
                next_ordinal = resume_next
                if current_size != file_bytes_out:
                    raise SystemExit("resumed in-place file does not have target size")
                print(f"resuming reverse relayout at ordinal {next_ordinal}", flush=True)

            if args.fast_in_place:
                write_state(
                    journal, file_bytes=file_bytes, target_bytes=file_bytes_out,
                    records=records, next_ordinal=next_ordinal, fast_started=True)

            done = len(occupied) - 1 - next_ordinal
            moved_bytes = 0
            while next_ordinal >= 0:
                slot = occupied[next_ordinal]
                offset_in, stored_in, _padded_in = entries_in[slot]
                target, expected_out, _padded_out = entries_out[slot]

                record = None
                if backup.exists():
                    saved = backup.read_bytes()
                    if len(saved) >= 16:
                        saved_slot, saved_bytes = struct.unpack_from("<QQ", saved, 0)
                        if (saved_slot == slot and saved_bytes == stored_in and
                                len(saved) == 16 + stored_in):
                            record = saved[16:]
                    if record is None:
                        backup.unlink()
                if record is None:
                    source.seek(offset_in)
                    record = source.read(stored_in)
                    if len(record) != stored_in:
                        raise SystemExit(f"truncated v1 record at slot {slot}")
                    if not args.fast_in_place:
                        with backup_tmp.open("wb", buffering=0) as handle:
                            write_exact(handle, struct.pack("<QQ", slot, stored_in),
                                        "write in-place backup header")
                            write_exact(handle, record, "write in-place record backup")
                            os.fsync(handle.fileno())
                        os.replace(backup_tmp, backup)

                block = relayout_record(record, stored_in, slot, experts, args.verify)
                if len(block) != expected_out:
                    raise SystemExit(f"precomputed v2 size changed at slot {slot}")
                source.seek(target)
                write_exact(source, block, f"write v2 record at slot {slot}")
                source.flush()
                next_ordinal -= 1
                done += 1
                moved_bytes += len(block)
                if not args.fast_in_place:
                    write_state(
                        journal, file_bytes=file_bytes, target_bytes=file_bytes_out,
                        records=records, next_ordinal=next_ordinal)
                backup.unlink(missing_ok=True)
                if done == 1 or done % 256 == 0:
                    elapsed = time.perf_counter() - begin
                    print(
                        f"reverse relayout {done}/{records}: "
                        f"{moved_bytes / 2**30:.2f} GiB this run, "
                        f"{moved_bytes / max(elapsed, 1e-9) / 2**30:.2f} GiB/s",
                        flush=True)

            source.seek(0)
            write_exact(source, header_out, "write v2 file header")
            write_exact(source, b"\0" * (FILE_HEADER_BYTES - len(header_out)),
                        "write v2 header-page padding")
            source.seek(index_offset)
            write_exact(source, b"".join(INDEX_ENTRY.pack(*entry)
                                         for entry in entries_out),
                        "write v2 expert index")
            source.flush()
            os.fsync(source.fileno())
            journal.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)
            backup_tmp.unlink(missing_ok=True)
        else:
            if current_size != file_bytes:
                raise SystemExit("input size does not match header file_bytes")
            temporary = output_path.with_name(output_path.name + ".tmp")
            if temporary.exists():
                temporary.unlink()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            done = 0
            with temporary.open("w+b", buffering=64 << 20) as output:
                output.seek(data_offset)
                for slot in occupied:
                    offset_in, stored_in, _padded_in = entries_in[slot]
                    source.seek(offset_in)
                    record = source.read(stored_in)
                    block = relayout_record(record, stored_in, slot, experts, args.verify)
                    expected_offset, expected_out, _ = entries_out[slot]
                    if output.tell() != expected_offset or len(block) != expected_out:
                        raise SystemExit(f"precomputed v2 layout changed at slot {slot}")
                    write_exact(output, block, f"write v2 record at slot {slot}")
                    done += 1
                    if done == 1 or done % 256 == 0:
                        elapsed = time.perf_counter() - begin
                        written = output.tell() - data_offset
                        print(
                            f"relayout {done}/{records}: {written / 2**30:.2f} GiB "
                            f"written, {written / max(elapsed, 1e-9) / 2**30:.2f} GiB/s",
                            flush=True)
                output.seek(0)
                write_exact(output, header_out, "write v2 file header")
                write_exact(output, b"\0" * (FILE_HEADER_BYTES - len(header_out)),
                            "write v2 header-page padding")
                output.seek(index_offset)
                write_exact(output, b"".join(INDEX_ENTRY.pack(*entry)
                                             for entry in entries_out),
                            "write v2 expert index")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(output_path)

    elapsed = time.perf_counter() - begin
    print(f"wrote {output_path}: {records} records, "
          f"{file_bytes_out / 2**30:.3f} GiB, in {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
