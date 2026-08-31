#!/usr/bin/env python3
"""CPU-only exactness test and contention-safe v2 packed-read microbench."""

import argparse
import concurrent.futures
import os
import pathlib
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time


ALIGNMENT = 4096
FILE_HEADER = struct.Struct("<8sIIIIQQQQQ")
INDEX_ENTRY = struct.Struct("<QII")
RECORD_HEADER_BYTES = 128
PROJECTIONS = 3
BODY_BYTES = 4 << 20
PACKED_SCALE_BYTES = 256 << 10
PREFIX_ENTRIES = PACKED_SCALE_BYTES // 256 + 1
BODY_TOTAL = PROJECTIONS * BODY_BYTES
PAYLOAD_CAPACITY = BODY_TOTAL + (1536 << 10) + 64
ROOT = pathlib.Path(__file__).resolve().parents[1]


def align(value, granularity=ALIGNMENT):
    return (value + granularity - 1) & -granularity


def escape_prefix(packed, expected):
    prefix = [0]
    count = 0
    for block in range(PREFIX_ENTRIES - 1):
        chunk = packed[block * 256:(block + 1) * 256]
        count += sum((byte & 15) == 15 for byte in chunk)
        count += sum((byte >> 4) == 15 for byte in chunk)
        prefix.append(count)
    if count != expected:
        raise AssertionError(f"escape prefix counted {count}, expected {expected}")
    return struct.pack(f"<{PREFIX_ENTRIES}I", *prefix)


def make_v1(path):
    escape_counts = (0, 1, 4097)
    globals_ = (0.5, -1.25, 3.0)
    header = bytearray(RECORD_HEADER_BYTES)
    struct.pack_into("<4sHH3I3f", header, 0, b"XPR1", 0, 0,
                     *escape_counts, *globals_)
    bodies, packed_planes, escapes, codebooks = [], [], [], []
    for projection, escape_count in enumerate(escape_counts):
        bodies.append(bytes([0x31 + projection]) * BODY_BYTES)
        packed = bytearray([0x21 + 0x11 * projection]) * PACKED_SCALE_BYTES
        for index in range(escape_count):
            packed[index] = (packed[index] & 0xF0) | 0x0F
        packed_planes.append(bytes(packed))
        escapes.append(bytes((17 * projection + index) & 0xFF
                             for index in range(escape_count)))
        codebook = bytes((29 * projection + index) & 0xFF for index in range(16))
        codebooks.append(codebook)
        header[32 + 16 * projection:48 + 16 * projection] = codebook

    record = bytearray(header)
    for body, packed, tail in zip(bodies, packed_planes, escapes):
        record += body + packed + tail
    index_offset = ALIGNMENT
    data_offset = align(index_offset + INDEX_ENTRY.size)
    padded = align(len(record))
    file_bytes = data_offset + padded
    file_prefix = bytearray(data_offset)
    FILE_HEADER.pack_into(file_prefix, 0, b"IG53XPK1", 1, 1, 1, 1,
                          index_offset, data_offset, file_bytes,
                          3 * (BODY_BYTES + (512 << 10) + 4), len(record))
    INDEX_ENTRY.pack_into(file_prefix, index_offset, data_offset, len(record), padded)
    with path.open("wb") as output:
        output.write(file_prefix)
        output.write(record)
        output.write(b"\0" * (padded - len(record)))
    return {
        "header": bytes(header), "bodies": bodies, "packed": packed_planes,
        "escapes": escapes, "codebooks": codebooks,
    }


def expected_region(packed, escapes, codebook):
    region = bytearray(packed + escapes + codebook)
    region += b"\0" * (align(len(region), 4) - len(region))
    region += escape_prefix(packed, len(escapes))
    blob_bytes = len(region)
    region += b"\0" * (align(len(region)) - len(region))
    return bytes(region), blob_bytes


def verify_layout():
    with tempfile.TemporaryDirectory(prefix="insignia-packed-transport-") as temporary:
        temporary = pathlib.Path(temporary)
        source = temporary / "packed-v1.bin"
        target = temporary / "packed-v2.bin"
        fixture = make_v1(source)
        subprocess.run(
            [sys.executable, str(ROOT / "tools/relayout_glm53_experts_v2.py"),
             str(source), str(target), "--verify"],
            check=True, stdout=subprocess.DEVNULL,
        )
        with target.open("rb") as input_file:
            file_header = FILE_HEADER.unpack(input_file.read(FILE_HEADER.size))
            assert file_header[0] == b"IG53XPK1" and file_header[1] == 2
            input_file.seek(file_header[5])
            offset, stored, padded = INDEX_ENTRY.unpack(input_file.read(INDEX_ENTRY.size))
            assert offset % ALIGNMENT == 0
            assert stored == padded and stored % ALIGNMENT == 0
            assert stored - ALIGNMENT <= PAYLOAD_CAPACITY
            input_file.seek(offset)
            record = input_file.read(stored)

        expected = bytearray(stored)
        expected[:RECORD_HEADER_BYTES] = fixture["header"]
        cursor = ALIGNMENT
        for body in fixture["bodies"]:
            expected[cursor:cursor + BODY_BYTES] = body
            cursor += BODY_BYTES
        copied = BODY_TOTAL
        for projection in range(PROJECTIONS):
            region, blob_bytes = expected_region(
                fixture["packed"][projection], fixture["escapes"][projection],
                fixture["codebooks"][projection],
            )
            expected[cursor:cursor + len(region)] = region
            cursor += len(region)
            copied += (PACKED_SCALE_BYTES + len(fixture["escapes"][projection]) +
                       16 + PREFIX_ENTRIES * 4)
        assert cursor == stored
        assert record == expected
        assert record[ALIGNMENT:ALIGNMENT + BODY_TOTAL] == b"".join(fixture["bodies"])
        assert copied == (BODY_TOTAL + PROJECTIONS *
                          (PACKED_SCALE_BYTES + 16 + PREFIX_ENTRIES * 4) +
                          sum(map(len, fixture["escapes"])))
        return bytes(record), copied


def timed_read(fd, record_bytes, fused, workers, iterations):
    barrier = threading.Barrier(workers + 1)

    def reader():
        header = bytearray(ALIGNMENT)
        payload = bytearray(record_bytes - ALIGNMENT)
        barrier.wait()
        for _ in range(iterations):
            if fused:
                if os.preadv(fd, (header, payload), 0) != record_bytes:
                    raise AssertionError("short fused preadv")
            else:
                if os.preadv(fd, (header,), 0) != ALIGNMENT:
                    raise AssertionError("short split header preadv")
                if os.preadv(fd, (payload,), ALIGNMENT) != len(payload):
                    raise AssertionError("short split payload preadv")
        return header[0] ^ payload[-1]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(reader) for _ in range(workers)]
        begin = time.perf_counter_ns()
        barrier.wait()
        checksum = sum(future.result() for future in futures)
        elapsed = time.perf_counter_ns() - begin
    if checksum < 0:
        raise AssertionError("unreachable checksum")
    return elapsed / (workers * iterations)


def benchmark(record, workers, iterations, pairs):
    if not hasattr(os, "preadv") or not hasattr(os, "memfd_create"):
        print("packed transport bench SKIP: preadv/memfd unavailable")
        return
    fd = os.memfd_create("insignia-packed-transport", os.MFD_CLOEXEC)
    try:
        os.write(fd, record)
        header, payload = bytearray(ALIGNMENT), bytearray(len(record) - ALIGNMENT)
        assert os.preadv(fd, (header, payload), 0) == len(record)
        assert bytes(header) + bytes(payload) == record
        split, fused = [], []
        order = (False, True, True, False)
        for _ in range(pairs):
            for arm in order:
                sample = timed_read(fd, len(record), arm, workers, iterations)
                (fused if arm else split).append(sample)
        split_median = statistics.median(split)
        fused_median = statistics.median(fused)
        records = workers * iterations
        print(
            f"packed transport bench PASS: memfd, {workers} readers, "
            f"ABBA median over {len(split)} samples; "
            f"split={split_median / 1e6:.3f} ms/record, "
            f"fused={fused_median / 1e6:.3f} ms/record, "
            f"ratio={split_median / fused_median:.3f}x; "
            f"syscalls/batch={2 * records}->{records}"
        )
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=2)
    args = parser.parse_args()
    if min(args.workers, args.iterations, args.pairs) < 1:
        parser.error("workers, iterations, and pairs must be positive")
    record, copied = verify_layout()
    print(
        f"packed transport layout PASS: {len(record)} v2 bytes exact; "
        f"legacy GPU staging copied {copied} bytes in 15 CPU memcpy calls, "
        "v2 CPU staging memcpy calls=0"
    )
    if args.bench:
        benchmark(record, args.workers, args.iterations, args.pairs)


if __name__ == "__main__":
    main()
