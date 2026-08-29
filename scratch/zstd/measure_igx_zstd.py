#!/usr/bin/env python3
"""Measure real compressibility + decode throughput of the packed expert sidecar.

Reads /var/lib/insignia/glm53-experts-nvfp4x.igx (read-only), samples records
across sparse layers, decodes the nibble-coded scales exactly like the engine,
and measures: nibble/byte entropies, zstd -1/-19 ratios on bodies / scales /
whole records with and without a trained dictionary, single- and multi-core
zstd decode throughput.  All temp files go to /dev/shm and are removed.

Run (no files created on the host):
    ssh glm-box "wsl -d Arch -- python3 -" < measure_igx_zstd.py
"""

import math
import os
import shutil
import statistics
import struct
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

IGX = "/var/lib/insignia/glm53-experts-nvfp4x.igx"
TMP = "/dev/shm/insig-zstd-probe"
SAMPLES = 12
DICT_TRAIN_BODIES = 18      # 4 MiB each, from the tail of the sample set
PARALLEL_DECODE_BODIES = 16

M20 = 4 << 20
SK = 512 << 10
PK = SK >> 1


def entropy(counts):
    total = sum(counts)
    return -sum((n / total) * math.log2(n / total) for n in counts if n)


def nibble_entropy(arr_u8):
    lo = np.bincount(arr_u8 & 15, minlength=16).astype(np.float64)
    hi = np.bincount(arr_u8 >> 4, minlength=16).astype(np.float64)
    h = lo + hi
    n = h.sum()
    return float(-(h[h > 0] / n * np.log2(h[h > 0] / n)).sum()), h


def read_file_header(f):
    return struct.unpack("<8sIIIIQQQQQ", f.read(64))


def read_index(f, index_offset, layers, experts):
    f.seek(index_offset)
    raw = f.read(16 * layers * experts)
    return [struct.unpack_from("<QII", raw, i) for i in range(0, len(raw), 16)]


def read_record(f, offset, stored):
    """Exact engine-side decode of one packed record (bodies + scales)."""
    f.seek(offset)
    blob = f.read(stored)
    assert len(blob) == stored
    layer, expert = struct.unpack_from("<HH", blob, 4)
    escapes = struct.unpack_from("<3I", blob, 8)
    codebooks = [blob[32 + 16 * p:48 + 16 * p] for p in range(3)]
    at = 128
    bodies, scales, code_streams = [], [], []
    for p in range(3):
        bodies.append(blob[at:at + M20]); at += M20
        packed = blob[at:at + PK]; at += PK
        esc = blob[at:at + escapes[p]]; at += escapes[p]
        arr = np.frombuffer(packed, np.uint8)
        codes = np.empty(SK, np.uint8)
        codes[0::2] = arr & 15
        codes[1::2] = arr >> 4
        table = np.frombuffer(bytes(codebooks[p]) + b"\0", np.uint8)
        out = table[codes].copy()
        escaped = codes == 15
        assert int(escaped.sum()) == escapes[p]
        out[escaped] = np.frombuffer(esc, np.uint8)
        scales.append(out.tobytes())
        code_streams.append(packed)
    assert at == stored, (at, stored)
    return layer, expert, escapes, bodies, scales, code_streams


def zstd_compress(raw, level, dictionary=None):
    cmd = ["zstd", f"-{level}", "-q", "-c"]
    if dictionary:
        cmd += ["-D", dictionary]
    return subprocess.run(cmd, input=raw, stdout=subprocess.PIPE, check=True).stdout


def zstd_decode_speed(raw, packed, repeat=3):
    samples = []
    for _ in range(repeat):
        begin = time.perf_counter()
        restored = subprocess.run(["zstd", "-d", "-q", "-c"], input=packed,
                                  stdout=subprocess.PIPE, check=True).stdout
        samples.append(time.perf_counter() - begin)
        assert restored == raw
    return len(raw) / statistics.median(samples) / 2**30


def sample_keys(entries, layers, experts):
    sparse = sorted({l for l in range(layers) if entries[l * experts + experts - 1][0]})
    keys = []
    for s in range(SAMPLES):
        at = 0 if SAMPLES == 1 else round(s * (len(sparse) - 1) / (SAMPLES - 1))
        keys.append((sparse[at], (17 + 73 * s) % experts))
    return sparse, keys


def run():
    with open(IGX, "rb") as f:
        magic, ver, layers, experts, records, index_off, data_off, \
            file_bytes, source_total, stored_total = read_file_header(f)
        entries = read_index(f, index_off, layers, experts)
        sparse, keys = sample_keys(entries, layers, experts)
        print(f"igx: {magic.decode()} v{ver} layers={layers} experts={experts} "
              f"records={records} file={file_bytes / 2**30:.3f} GiB")
        print(f"source_total={source_total} stored_total={stored_total} "
              f"logical={stored_total / source_total:.5f}x "
              f"file={file_bytes / source_total:.5f}x "
              f"implied_escapes="
              f"{(stored_total - records * (128 + 3 * (M20 + PK))) / (records * 3 * SK) * 100:.3f}%")
        print(f"sparse layers: {len(sparse)} ({sparse[0]}..{sparse[-1]})")
        recs = []
        for layer, expert in keys:
            offset, stored, padded = entries[layer * experts + expert]
            layer_r, expert_r, escapes, bodies, scales, code_streams = \
                read_record(f, offset, stored)
            assert (layer_r, expert_r) == (layer, expert)
            recs.append((layer, expert, escapes, bodies, scales, code_streams))
            print(f"L{layer:02d} E{expert:03d} stored={stored} escapes="
                  f"{[round(e / SK * 100, 3) for e in escapes]}%")

    # ---- entropy over the aggregate sample -------------------------------
    body_h = np.zeros(16)
    per_record_body_h = []
    scale_hist = np.zeros(256)
    code_hist = np.zeros(16)
    for layer, expert, escapes, bodies, scales, code_streams in recs:
        for b in bodies:
            h, counts = nibble_entropy(np.frombuffer(b, np.uint8))
            per_record_body_h.append(h)
            body_h += counts
        for s in scales:
            scale_hist += np.bincount(np.frombuffer(s, np.uint8), minlength=256)
        for c in code_streams:
            _, counts = nibble_entropy(np.frombuffer(c, np.uint8))
            code_hist += counts
    print(f"\nbody nibble entropy (aggregate {SAMPLES} recs): "
          f"{entropy(body_h.tolist()):.4f}/4 bits -> ideal ratio {entropy(body_h.tolist()) / 4:.5f}x")
    print(f"per-record body entropy: min={min(per_record_body_h):.4f} "
          f"median={statistics.median(per_record_body_h):.4f} "
          f"max={max(per_record_body_h):.4f} bits/nibble")
    top = sorted(range(256), key=lambda c: -scale_hist[c])[:16]
    sc_total = scale_hist.sum()
    print(f"scale byte entropy: {entropy(scale_hist.tolist()):.4f}/8 bits; alphabet "
          f"{int((scale_hist > 0).sum())} codes; top-16: "
          + " ".join(f"0x{c:02x}:{100 * scale_hist[c] / sc_total:.2f}%" for c in top))
    print(f"scale nibble-code entropy (stored stream): "
          f"{entropy(code_hist.tolist()):.4f}/4 bits -> ideal ratio "
          f"{entropy(code_hist.tolist()) / 4:.5f}x on the 256 KiB packed stream")

    # ---- codec measurements ------------------------------------------------
    version = subprocess.run(["zstd", "--version"], capture_output=True, text=True).stdout.strip()
    print(f"\nzstd: {version}")

    body_blob = b"".join(b for r in recs[:4] for b in r[3])      # 48 MiB
    scale_blob = b"".join(s for r in recs[:4] for s in r[4])     # 6 MiB
    rec0 = recs[0]
    record_blob = b"".join(rec0[3]) + b"".join(rec0[4])

    for name, raw in (("bodies-48MiB", body_blob), ("scales-6MiB", scale_blob),
                      ("record0-13.5MiB", record_blob)):
        for level in (1, 19):
            begin = time.perf_counter()
            packed = zstd_compress(raw, level)
            ctime = time.perf_counter() - begin
            speed = zstd_decode_speed(raw, packed)
            print(f"{name:>18s} zstd-{level:<2d} ratio={len(packed) / len(raw):.4f}x "
                  f"compress={len(raw) / ctime / 2**20:.1f} MiB/s decode={speed:.2f} GiB/s")

    # ---- in-process (stdlib compression.zstd, no pipes) --------------------
    try:
        from compression import zstd as zstd_mod
        print(f"\nstdlib compression.zstd: lib {zstd_mod.ZSTD_VERSION}")
    except Exception as exc:
        zstd_mod = None
        print(f"stdlib compression.zstd unavailable: {exc}")
    if zstd_mod is not None:
        for name, raw in (("bodies-48MiB", body_blob), ("scales-6MiB", scale_blob),
                          ("record0-13.5MiB", record_blob)):
            for level in (1, 19):
                begin = time.perf_counter()
                packed = zstd_mod.compress(raw, level=level)
                ctime = time.perf_counter() - begin
                times = []
                for _ in range(3):
                    begin = time.perf_counter()
                    restored = zstd_mod.decompress(packed)
                    times.append(time.perf_counter() - begin)
                    assert restored == raw
                print(f"{name:>18s} zstd-{level:<2d} inproc ratio={len(packed) / len(raw):.4f}x "
                      f"compress={len(raw) / ctime / 2**20:.1f} MiB/s "
                      f"decode={len(raw) / statistics.median(times) / 2**30:.2f} GiB/s")
        # threaded in-process decode of independent 4 MiB frames
        frames = []
        for r in recs[:4]:
            for b in r[3]:
                frames.append((b, zstd_mod.compress(b, level=19)))
        for workers in (1, 4, 8):
            reps = []
            for _ in range(2):
                begin = time.perf_counter()
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    out = list(pool.map(lambda f: zstd_mod.decompress(f[1]), frames))
                assert all(out[i] == frames[i][0] for i in range(len(frames)))
                reps.append(sum(len(b) for b, _ in frames) / 2**30 /
                            (time.perf_counter() - begin))
            print(f"inproc threaded decode zstd-19 bodies: {workers} thread(s): "
                  f"median {statistics.median(reps):.2f} GiB/s over {len(frames)} x 4 MiB frames")

    # ---- dictionary (trained on bodies the tests never compress) ----------
    dict_train = [b for r in recs[4:] for b in r[3]][:DICT_TRAIN_BODIES]
    paths = []
    for i, b in enumerate(dict_train):
        p = f"{TMP}/train{i}"
        with open(p, "wb") as out:
            out.write(b)
        paths.append(p)
    dict_path = f"{TMP}/expert.dict"
    subprocess.run(["zstd", "--train", *paths, "-o", dict_path, "-q"],
                   check=True, capture_output=True)
    dict_size = os.path.getsize(dict_path)
    test_bodies = b"".join(recs[0][3] + recs[1][3])   # 6 unseen 4 MiB bodies
    test_scales = b"".join(recs[0][4])                # 1.5 MiB
    for level in (1, 19):
        plain_b = zstd_compress(test_bodies, level)
        dict_b = zstd_compress(test_bodies, level, dict_path)
        plain_s = zstd_compress(test_scales, level)
        dict_s = zstd_compress(test_scales, level, dict_path)
        print(f"dict({dict_size // 1024} KiB) zstd-{level}: "
              f"bodies plain={len(plain_b) / len(test_bodies):.4f}x "
              f"dict={len(dict_b) / len(test_bodies):.4f}x | "
              f"scales plain={len(plain_s) / len(test_scales):.4f}x "
              f"dict={len(dict_s) / len(test_scales):.4f}x")

    # ---- parallel CLI decode throughput (file -> /dev/null, no pipes) ------
    # Independent per-body frames: the windowed runtime design would decode
    # separate frames on separate reader threads.
    frame_files = []
    all_bodies = [b for r in recs for b in r[3]][:PARALLEL_DECODE_BODIES]
    for i, b in enumerate(all_bodies):
        p = f"{TMP}/body{i}.zst"
        with open(p, "wb") as out:
            out.write(zstd_compress(b, 19))
        frame_files.append((p, len(b)))
    for workers in (1, 4, 8):
        reps = []
        for rep in range(2):
            begin = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda item: subprocess.run(
                    ["zstd", "-d", "-q", "-c", item[0]],
                    stdout=subprocess.DEVNULL, check=True), frame_files))
            reps.append(sum(n for _, n in frame_files) / 2**30 /
                        (time.perf_counter() - begin))
        print(f"parallel CLI decode zstd-19 bodies: {workers} thread(s): "
              f"median {statistics.median(reps):.2f} GiB/s aggregate "
              f"({len(frame_files)} x 4 MiB frames)")


def main():
    os.makedirs(TMP, exist_ok=True)
    try:
        run()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
