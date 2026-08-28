#!/usr/bin/env python3
"""Build the cross-layer co-activation table (CCT) for expert prefetching.

Reads a routing trace ("token layer e0..e7 s0..s7" per line, produced by
INSIGNIA_GLM53_ROUTE_TRACE), counts adjacent-sparse-layer co-activations, and
writes a binary table: for each (L, L+1) sparse pair, the top-8 experts of
L+1 most co-activated with each expert of L.

Format: "IGCCT1\0" u32 pairs; then per pair: u16 layer, u16 next, and
288 x 8 x u16 expert ids (ranked by co-activation count, ties by id).
"""
import struct
import sys
from collections import defaultdict

import numpy as np

trace = sys.argv[1]
out = sys.argv[2]
topk = 8

by_token = defaultdict(dict)  # token -> layer -> [8 experts]
with open(trace) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 11:
            continue
        token, layer = int(parts[0]), int(parts[1])
        experts = [int(x) for x in parts[2:10]]
        by_token[token][layer] = experts

layers = sorted({layer for routing in by_token.values() for layer in routing})
n_experts = 288
pairs = [(a, b) for a, b in zip(layers, layers[1:]) if b == a + 1]

tables = []
for a, b in pairs:
    co = np.zeros((n_experts, n_experts), dtype=np.uint32)
    for routing in by_token.values():
        if a in routing and b in routing:
            for e in routing[a]:
                co[e, routing[b]] += 1
    # Top-8 per expert by count, ties broken by smaller id.
    order = np.argsort(-co, axis=1, kind="stable")[:, :topk].astype(np.uint16)
    tables.append((a, b, order))

with open(out, "wb") as f:
    f.write(b"IGCCT1\0")
    f.write(struct.pack("<I", len(tables)))
    f.write(struct.pack("<I", n_experts))
    f.write(struct.pack("<I", topk))
    for a, b, order in tables:
        f.write(struct.pack("<HH", a, b))
        f.write(order.astype("<u2").tobytes())
print(f"wrote {len(tables)} pairs, {n_experts} experts, top{topk} -> {out}")
