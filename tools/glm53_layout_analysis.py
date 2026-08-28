#!/usr/bin/env python3
"""Analyze tensor->shard layout: per layer, how are expert bytes spread across shards?
Also emits a stripe assignment plan (per-shard C vs E or per-record split)."""
import pathlib
import struct
import sys
import collections

MAGIC = b"IGLMIDX1"

def load_index(path):
    data = path.read_bytes()
    assert data[:8] == MAGIC
    (version, flags, nshards, nentries, hidden, layers, vocab,
     experts, topk, moe_inter, hc, total) = struct.unpack_from("<11IQ", data, 8)
    off = 8 + struct.calcsize("<11IQ")
    shards = []
    for _ in range(nshards):
        nlen, size = struct.unpack_from("<HQ", data, off)
        off += 10
        name = data[off:off + nlen].decode()
        off += nlen
        shards.append((name, size))
    entries = []
    for _ in range(nentries):
        nlen, dtype, ndim, shard, flags2, absolute, length = struct.unpack_from("<HBBHHQQ", data, off)
        off += struct.calcsize("<HBBHHQQ")
        name = data[off:off + nlen].decode()
        off += nlen
        shape = struct.unpack_from("<" + "I" * ndim, data, off)
        off += 4 * ndim
        entries.append((name, dtype, shape, shard, absolute, length))
    return shards, entries

def main():
    idx = pathlib.Path(sys.argv[1])
    shards, entries = load_index(idx)
    print(f"{len(entries)} tensors, {len(shards)} shards")
    # Per-layer expert byte spread across shards
    layer_shard = collections.defaultdict(lambda: collections.Counter())
    expert_layer_shard = collections.defaultdict(lambda: collections.Counter())
    for name, dtype, shape, shard, absolute, length in entries:
        parts = name.split(".")
        if parts[0] == "model" and parts[1] == "language_model" and parts[2] == "layers":
            layer = int(parts[3])
            layer_shard[layer][shard] += length
            if "experts." in name:
                expert_layer_shard[layer][shard] += length
    for layer in sorted(layer_shard)[:6]:
        c = expert_layer_shard[layer]
        total = sum(c.values())
        top = ", ".join(f"s{s}:{b/2**20:.0f}M" for s, b in sorted(c.items()))
        print(f"layer {layer:2d}: expert bytes {total/2**20:.0f} MiB over {len(c)} shards -> {top}")
    layers = sorted(expert_layer_shard)
    spread = [len(expert_layer_shard[l]) for l in layers]
    print(f"layers {layers[0]}..{layers[-1]}, shards-per-layer min {min(spread)} max {max(spread)}")
    # How many layers does each shard touch?
    shard_layers = collections.defaultdict(set)
    for layer, c in expert_layer_shard.items():
        for s in c:
            shard_layers[s].add(layer)
    counts = sorted(len(v) for v in shard_layers.values())
    print(f"layers-per-shard: min {counts[0]} median {counts[len(counts)//2]} max {counts[-1]}")

if __name__ == "__main__":
    main()
