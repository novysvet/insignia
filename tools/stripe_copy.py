#!/usr/bin/env python3
"""Rate-limited copier for the E: expert-stripe store (DRAM-less host drive).

Copies the named shards one at a time, pacing writes to stay inside the SLC
cache (~300 MB/s) and fsyncing each file. Run inside the SAME wsl session
that mounted the vhdx -- the mount does not survive VM recycles.
"""
import os
import sys
import time

SRC = "/var/lib/insignia/glm53-flash-text"
DST = "/var/lib/insignia/e2store"
RATE = 300.0 * 1024 * 1024  # bytes/sec budget
CHUNK = 8 << 20

names = [line.strip() for line in sys.argv[1:] if line.strip()]
if not names:
    for n in range(2, 121, 2):
        names.append("model-%05d-of-00120.safetensors" % n)

os.makedirs(DST, exist_ok=True)
for name in names:
    src = os.path.join(SRC, name)
    dst = os.path.join(DST, name)
    src_size = os.path.getsize(src)
    if os.path.exists(dst) and os.path.getsize(dst) == src_size:
        print("skip %s (already complete)" % name, flush=True)
        continue
    begin = time.time()
    budget = 0.0
    last = time.time()
    with open(src, "rb", buffering=0) as fi, open(dst, "wb", buffering=0) as fo:
        while True:
            block = fi.read(CHUNK)
            if not block:
                break
            fo.write(block)
            budget += len(block)
            now = time.time()
            budget -= (now - last) * RATE
            last = now
            if budget > CHUNK:  # ahead of the budget: sleep it off
                time.sleep(budget / RATE)
                last = time.time()
                budget = 0.0
        os.fsync(fo.fileno())
    got = os.path.getsize(dst)
    status = "ok" if got == src_size else "SIZE MISMATCH"
    print("%s %d bytes in %.1fs (%.0f MB/s) %s" % (name, got, time.time() - begin,
          got / 1048576.0 / (time.time() - begin), status), flush=True)
    if got != src_size:
        sys.exit(1)
print("all shards copied and size-verified", flush=True)
