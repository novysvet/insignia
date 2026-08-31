#!/usr/bin/env python3
"""Routing-locality analysis for INSIGNIA_GLM53_ROUTE_TRACE traces.

Trace line: "token_index layer e0..e7 s0..e7" (decode steps only; e-order is
the engine's execution order). Adjacency metrics compare consecutive decode steps of
the same layer within one file; cache/frequency stats concatenate files.

Usage: python glm53_route_analysis.py run1.txt [run2.txt ...] [--warmup N]
"""
import argparse
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

SLOT_MIB = 13.5
SIZES = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


def load(path):
    layers, stream = {}, []  # stream: (token_index, key=layer*1024+expert) in exec order
    for line in open(path, encoding="utf-8"):
        f = line.split()
        if len(f) < 4:
            continue
        tok, layer = int(f[0]), int(f[1])
        k = (len(f) - 2) // 2
        experts = [int(x) for x in f[2:2 + k]]
        scores = [float(x) for x in f[2 + k:2 + 2 * k]]
        layers.setdefault(layer, []).append((tok, experts, scores))
        stream.extend((tok, layer * 1024 + e) for e in experts)
    for rows in layers.values():
        rows.sort(key=lambda r: r[0])
    return layers, stream


def popcount(x):
    return bin(x).count("1")


def overlap_stats(files, topk):
    print("\n(a)/(c) adjacent decode steps, same layer")
    print("      I/8 = prev-token-set recall (== precision for 8-vs-8 sets);")
    print("      p@k  = precision of prefetching the prev step's top-k (engine order)")
    print(f"{'layer':>5} {'pairs':>6} {'mean|I|':>8} {'I/8':>6} {'jacc':>6} {'p@1':>6} {'p@2':>6} {'p@4':>6}")
    layers = sorted({l for f in files for l in f[0]})
    tot, tp = [0.0] * 6, 0
    for layer in layers:
        acc, pairs = [0.0] * 6, 0
        for layers1, _ in files:
            rows = layers1.get(layer, [])
            m = [0] * len(rows)
            for i, (_, e, _) in enumerate(rows):
                for x in e:
                    m[i] |= 1 << x
            for i in range(1, len(rows)):
                if rows[i][0] != rows[i - 1][0] + 1:
                    continue
                inter = popcount(m[i] & m[i - 1])
                cur, prev = set(rows[i][1]), rows[i - 1][1]
                acc[0] += inter
                acc[1] += inter / topk
                acc[2] += inter / popcount(m[i] | m[i - 1])
                acc[3] += int(prev[0] in cur)
                acc[4] += len(cur & set(prev[:2])) / 2
                acc[5] += len(cur & set(prev[:4])) / 4
                pairs += 1
        if not pairs:
            continue
        tot = [t + x for t, x in zip(tot, acc)]
        tp += pairs
        v = [x / pairs for x in acc]
        print(f"{layer:>5} {pairs:>6} {v[0]:>8.2f} {v[1]:>6.3f} {v[2]:>6.3f} "
              f"{v[3]:>6.3f} {v[4]:>6.3f} {v[5]:>6.3f}")
    v = [x / tp for x in tot]
    print(f"{'ALL':>5} {tp:>6} {v[0]:>8.2f} {v[1]:>6.3f} {v[2]:>6.3f} "
          f"{v[3]:>6.3f} {v[4]:>6.3f} {v[5]:>6.3f}")


def cache_curves(streams, warm):
    accesses = [k for st in streams for t, k in st if t >= warm]
    freq, n = Counter(accesses), len(accesses)
    print(f"\n(b) global (layer,expert) LRU vs (e) optimal static-by-frequency "
          f"(after warmup: {n} accesses, {SLOT_MIB} MiB/slot)")
    print(f"{'slots':>6} {'MiB':>9} {'LRU%':>7} {'static%':>8}")
    ranked = [k for k, _ in freq.most_common()]
    for cap in SIZES:
        cache, hits = OrderedDict(), 0
        for key in accesses:
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                cache[key] = True
                if len(cache) > cap:
                    cache.popitem(last=False)
        sh = sum(freq[k] for k in set(ranked[:cap])) / n
        print(f"{cap:>6} {cap * SLOT_MIB:>9.0f} {100 * hits / n:>7.2f} {100 * sh:>8.2f}")


def stripe_miss_weights(streams, capacity):
    """Return per-record LRU misses, resetting cache at each prompt trace.

    Striping should balance bytes that actually reach NVMe, not raw router
    frequency.  Each input stream is one independently benchmarked prompt, so
    carrying a warm cache between streams would leak locality across requests.
    """
    if capacity <= 0:
        raise ValueError("stripe cache capacity must be positive")
    misses = Counter()
    requests = hits = 0
    for stream in streams:
        cache = OrderedDict()
        for _token, key in stream:
            requests += 1
            if key in cache:
                hits += 1
                cache.move_to_end(key)
                continue
            misses[key] += 1
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return misses, requests, hits


def write_stripe_miss_weights(streams, capacity, path):
    """Write ``layer expert miss_weight`` rows accepted by stripe_repack.py."""
    misses, requests, hits = stripe_miss_weights(streams, capacity)
    output = Path(path)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        file.write("# Insignia GLM-5.3 route-trace LRU miss weights\n")
        file.write(f"# cache_slots {capacity}\n")
        file.write(f"# prompt_streams {len(streams)} requests {requests} hits {hits} "
                   f"misses {requests - hits}\n")
        file.write("layer expert miss_weight\n")
        for key, count in sorted(misses.items()):
            file.write(f"{key // 1024} {key % 1024} {count}\n")
    return misses, requests, hits


def entropy_tables(files, warm, experts):
    print(f"\n(d) per-layer expert frequency after warmup (uniform max = {np.log2(experts):.2f} bits)")
    print(f"{'layer':>5} {'accesses':>9} {'H(bits)':>8} {'top32%':>7} {'unique':>6}")
    layers = sorted({l for f in files for l in f[0]})
    allc = Counter()
    for layer in layers:
        cnt = Counter()
        for layers1, _ in files:
            for tok, es, _s in layers1.get(layer, []):
                if tok >= warm:
                    cnt.update(es)
        n = sum(cnt.values())
        if n:
            p = np.fromiter(cnt.values(), dtype=np.float64) / n
            top32 = sum(c for _, c in cnt.most_common(32)) / n
            print(f"{layer:>5} {n:>9} {float(-(p * np.log2(p)).sum()):>8.3f} "
                  f"{100 * top32:>7.1f} {len(cnt):>6}")
            allc.update(cnt)
    n = sum(allc.values())
    p = np.fromiter(allc.values(), dtype=np.float64) / n
    print(f"{'ALL':>5} {n:>9} {float(-(p * np.log2(p)).sum()):>8.3f} "
          f"{100 * sum(c for _, c in allc.most_common(32)) / n:>7.1f} {len(allc):>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--warmup", type=int, default=0, help="decode steps excluded from (b)/(d)/(e); 0 = 10%%")
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--stripe-weights", type=Path,
                    help="write per-(layer,expert) LRU miss weights for stripe_repack.py")
    ap.add_argument("--stripe-cache-slots", type=int, default=0,
                    help="host expert slots used for --stripe-weights (required)")
    a = ap.parse_args()
    files = [load(p) for p in a.traces]
    topk = max((len(rows[0][1]) for f in files for rows in f[0].values()), default=0)
    max_tok = max((t for _, st in files for t, _ in st), default=0)
    warm = a.warmup if a.warmup > 0 else max(1, max_tok // 10)
    print(f"files={len(files)} sparse layers={len({l for f in files for l in f[0]})} "
          f"topk={topk} max decode steps={max_tok + 1}; warmup cuts token_index < {warm}")
    overlap_stats(files, topk)
    cache_curves([st for _, st in files], warm)
    entropy_tables(files, warm, a.experts)
    if a.stripe_weights:
        if a.stripe_cache_slots <= 0:
            ap.error("--stripe-weights requires positive --stripe-cache-slots")
        misses, requests, hits = write_stripe_miss_weights(
            [stream for _, stream in files], a.stripe_cache_slots,
            a.stripe_weights,
        )
        print(f"wrote {len(misses)} nonzero miss weights to {a.stripe_weights} "
              f"({requests - hits}/{requests} LRU misses; cache resets per trace)")


if __name__ == "__main__":
    main()
