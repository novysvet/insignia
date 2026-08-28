#!/usr/bin/env python3
"""Concurrent two-path large-read bandwidth probe (engine-like geometry).

Usage: dual_probe.py <pathA> <pathB> [duration_s]
A/B are large files (or "NONE" to skip one side). Measures:
  - solo aggregate GB/s per path with 4 reader threads
  - concurrent aggregate when both paths are hammered at once
Threads do sequential 13.5 MiB preads (O_DIRECT where supported) at pseudo-random
aligned offsets, mirroring the ExpertStager reader pool.
"""
import os
import sys
import threading
import time

CHUNK = 13 * 1024 * 1024 + 512 * 1024


def open_direct(path):
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError:
        return os.open(path, os.O_RDONLY)


def bench(path, nthreads, duration, stop, ready, go):
    fd = open_direct(path)
    size = os.fstat(fd).st_size
    n = max(1, (size - CHUNK) // (1 << 20))
    counts = [0] * nthreads

    def worker(tid):
        pos = (tid * 2654435761) % n
        for _ in range(64):  # warmup
            os.pread(fd, CHUNK, (pos << 20))
            pos = (pos * 1103515245 + 12345) % n
        ready[tid] = True
        while not go.is_set():
            time.sleep(0.0005)
        while not stop.is_set():
            os.pread(fd, CHUNK, (pos << 20))
            counts[tid] += 1
            pos = (pos * 1103515245 + 12345) % n

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(nthreads)]
    for t in threads:
        t.start()
    return threads, counts


def run(path, nthreads, duration, label):
    stop, go = threading.Event(), threading.Event()
    ready = [False] * nthreads
    threads, counts = bench(path, nthreads, duration, stop, ready, go)
    while not all(ready):
        time.sleep(0.05)
    t0 = time.perf_counter()
    go.set()
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join()
    dt = time.perf_counter() - t0
    bw = sum(counts) * CHUNK / dt / 1e9
    print(f"{label}: {bw:.2f} GB/s ({sum(counts)} reads in {dt:.1f}s)", flush=True)
    return bw


def main():
    path_a, path_b = sys.argv[1], sys.argv[2]
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    threads = 4
    solo_a = solo_b = 0.0
    if path_a != "NONE":
        solo_a = run(path_a, threads, duration, f"solo A ({path_a})")
    if path_b != "NONE":
        solo_b = run(path_b, threads, duration, f"solo B ({path_b})")
    if path_a != "NONE" and path_b != "NONE":
        stop, go = threading.Event(), threading.Event()
        ready_a = [False] * threads
        ready_b = [False] * threads

        def bench2(path, ready):
            fd = open_direct(path)
            size = os.fstat(fd).st_size
            n = max(1, (size - CHUNK) // (1 << 20))
            counts = [0] * threads

            def worker(tid):
                pos = (tid * 2654435761) % n
                for _ in range(64):
                    os.pread(fd, CHUNK, (pos << 20))
                    pos = (pos * 1103515245 + 12345) % n
                ready[tid] = True
                while not go.is_set():
                    time.sleep(0.0005)
                while not stop.is_set():
                    os.pread(fd, CHUNK, (pos << 20))
                    counts[tid] += 1
                    pos = (pos * 1103515245 + 12345) % n

            return [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)], counts

        ta, ca = bench2(path_a, ready_a)
        tb, cb = bench2(path_b, ready_b)
        for t in ta + tb:
            t.start()
        while not (all(ready_a) and all(ready_b)):
            time.sleep(0.05)
        t0 = time.perf_counter()
        go.set()
        time.sleep(duration)
        stop.set()
        for t in ta + tb:
            t.join()
        dt = time.perf_counter() - t0
        bwa = sum(ca) * CHUNK / dt / 1e9
        bwb = sum(cb) * CHUNK / dt / 1e9
        print(f"concurrent: A {bwa:.2f} + B {bwb:.2f} = {bwa + bwb:.2f} GB/s aggregate", flush=True)
        print(f"solo baseline sum: {solo_a + solo_b:.2f} GB/s", flush=True)


if __name__ == "__main__":
    main()
