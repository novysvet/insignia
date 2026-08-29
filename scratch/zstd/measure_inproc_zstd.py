#!/usr/bin/env python3
"""In-process stdlib compression.zstd measurement on real .igx payloads."""
import statistics
import struct
import time
from concurrent.futures import ThreadPoolExecutor

from compression import zstd as z

IGX = "/var/lib/insignia/glm53-experts-nvfp4x.igx"
M20 = 4 << 20
SK = 512 << 10
PK = SK >> 1


def read_record(f, offset, stored):
    f.seek(offset)
    blob = f.read(stored)
    escapes = struct.unpack_from("<3I", blob, 8)
    codebooks = [blob[32 + 16 * p:48 + 16 * p] for p in range(3)]
    at = 128
    bodies, scales = [], []
    import numpy as np
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
        out[codes == 15] = np.frombuffer(esc, np.uint8)
        scales.append(out.tobytes())
    return bodies, scales


with open(IGX, "rb") as f:
    hdr = struct.unpack("<8sIIIIQQQQQ", f.read(64))
    index_off = hdr[5]
    f.seek(index_off)
    raw = f.read(16 * 45 * 288)
    entries = [struct.unpack_from("<QII", raw, i) for i in range(0, len(raw), 16)]
    sparse = [l for l in range(45) if entries[l * 288 + 287][0]]
    recs = []
    for s in range(4):
        layer = sparse[round(s * (len(sparse) - 1) / 3)]
        expert = (17 + 73 * s) % 288
        offset, stored, padded = entries[layer * 288 + expert]
        recs.append(read_record(f, offset, stored))

bodies = [b for r in recs for b in r[0]]      # 12 x 4 MiB
scales = [s for r in recs for s in r[1]]      # 12 x 512 KiB
body_blob = b"".join(bodies)
scale_blob = b"".join(scales)
print(f"payloads: {len(bodies)} bodies ({len(body_blob) / 2**20:.0f} MiB), "
      f"{len(scales)} scale streams ({len(scale_blob) / 2**20:.1f} MiB)")

for name, blob in (("bodies", body_blob), ("scales", scale_blob)):
    for level in (1, 3, 19):
        begin = time.perf_counter()
        packed = z.compress(blob, level=level)
        ctime = time.perf_counter() - begin
        times = []
        for _ in range(3):
            begin = time.perf_counter()
            restored = z.decompress(packed)
            times.append(time.perf_counter() - begin)
            assert restored == blob
        print(f"{name:>7s} zstd-{level:<2d} inproc ratio={len(packed) / len(blob):.4f}x "
              f"compress={len(blob) / ctime / 2**20:.1f} MiB/s "
              f"decode={len(blob) / statistics.median(times) / 2**30:.2f} GiB/s")

# per-frame decode (engine granularity: one 4 MiB body per frame)
frames = [(b, z.compress(b, level=19)) for b in bodies]
per = []
for raw_b, packed in frames:
    begin = time.perf_counter()
    assert z.decompress(packed) == raw_b
    per.append(M20 / (time.perf_counter() - begin) / 2**30)
print(f"per-4MiB-frame zstd-19 decode: min={min(per):.2f} median={statistics.median(per):.2f} "
      f"max={max(per):.2f} GiB/s")

scale_frames = [(s, z.compress(s, level=19)) for s in scales]
pers = []
for raw_s, packed in scale_frames:
    begin = time.perf_counter()
    assert z.decompress(packed) == raw_s
    pers.append(SK / (time.perf_counter() - begin) / 2**30)
print(f"per-512KiB-frame zstd-19 scale decode: median={statistics.median(pers):.2f} GiB/s")

for workers in (1, 2, 4, 8):
    reps = []
    for _ in range(3):
        begin = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(lambda fr: z.decompress(fr[1]), frames))
        assert all(out[i] == frames[i][0] for i in range(len(frames)))
        reps.append(len(frames) * M20 / 2**30 / (time.perf_counter() - begin))
    print(f"inproc threaded body decode: {workers} thread(s): "
          f"median {statistics.median(reps):.2f} GiB/s")

# mixed record stream: what one reader thread must sustain (12 MiB bodies +
# 1.5 MiB scales as separate frames, decoded back-to-back)
record_frames = [pb for r in recs for pb in
                 [z.compress(x, level=19) for x in (r[0][0], r[0][1], r[0][2],
                                                    r[1][0], r[1][1], r[1][2])]]
expect = [x for r in recs for x in (r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2])]
begin = time.perf_counter()
out = [z.decompress(fr) for fr in record_frames]
assert out == expect
elapsed = time.perf_counter() - begin
print(f"sequential record decode (3 bodies + 3 scales, zstd-19): "
      f"{13.5 * len(recs) / elapsed:.2f} MiB/s per thread "
      f"({elapsed / len(recs) * 1000:.1f} ms per 13.5 MiB record)")
